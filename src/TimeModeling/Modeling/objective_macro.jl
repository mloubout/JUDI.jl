export @judi_objective

"""
    @judi_objective function objective(x, d_obs)
        d_syn = F(x) * q
        r = d_syn - d_obs
        phi = 0.5f0 * norm(r)^2
        g = J' * r
        return phi, g
    end

Rewrite an FWI or LSRTM objective expressed with JUDI's linear-algebra API to the
corresponding fused PDE objective. Operator products in the body are inspected to
infer data and model preconditioners. Misfits may either return `(value, derivative)`
or be scalar functions of the residual with a ChainRules `rrule`.
"""
macro judi_objective(def)
    try
        esc(_rewrite_judi_objective(def))
    catch err
        err isa ArgumentError || rethrow()
        @warn "@judi_objective could not fuse this objective; using the original function definition. Reason: $(err.msg)"
        esc(def)
    end
end

_objective_error(msg) = throw(ArgumentError("@judi_objective: " * msg))

"""Return whether `ex` is a call to `f`, optionally with exactly `n` arguments."""
function _is_call(ex, f, n=nothing)
    ex isa Expr && ex.head == :call || return false
    ex.args[1] == f || return false
    return isnothing(n) || length(ex.args) == n + 1
end

"""Remove type annotations or defaults and return a function argument's name."""
function _argument_name(ex)
    ex isa Symbol && return ex
    ex isa Expr && ex.head in (:(::), :(=), :kw) && return _argument_name(ex.args[1])
    _objective_error("only ordinary positional arguments are supported")
end

"""Flatten a possibly nested multiplication expression from left to right."""
function _mul_factors(ex)
    ex isa Expr && ex.head == :call && ex.args[1] == :* || return Any[ex]
    reduce(vcat, (_mul_factors(arg) for arg in ex.args[2:end]); init=Any[])
end

# -----------------------------------------------------------------------------
# Syntax-tree parsing helpers
# -----------------------------------------------------------------------------

"""Collect assignments, a tuple-returning loss, and the final return expression."""
function _parse_objective_body(body)
    statements = body isa Expr && body.head == :block ? body.args : Any[body]
    assignments = Dict{Symbol, Any}()
    tuple_loss = nothing
    returned = nothing

    for statement in statements
        statement isa LineNumberNode && continue

        if statement isa Expr && statement.head == :(=)
            lhs, rhs = statement.args

            if lhs isa Symbol
                # Ordinary assignment, e.g. `residual = predicted - observed`.
                assignments[lhs] = rhs
            elseif lhs isa Expr && lhs.head == :tuple && length(lhs.args) == 2 &&
                   rhs isa Expr && rhs.head == :call && length(rhs.args) == 3
                # Misfit protocol: `value, derivative = misfit(predicted, observed)`.
                tuple_loss = (lhs.args[1], lhs.args[2], rhs.args[1], rhs.args[2], rhs.args[3])
            else
                _objective_error("unsupported assignment in objective body")
            end
        elseif statement isa Expr && statement.head == :return
            returned = only(statement.args)
        else
            _objective_error("the body may contain only assignments followed by `return phi, g`")
        end
    end

    returned isa Expr && returned.head == :tuple && length(returned.args) == 2 ||
        _objective_error("expected `return phi, g`")

    return assignments, tuple_loss, returned
end

"""Recursively substitute intermediate assignments, optionally preserving one name."""
function _resolve_assignments(ex, assignments; keep=nothing)
    ex == keep && return ex

    if ex isa Symbol && haskey(assignments, ex)
        return _resolve_assignments(assignments[ex], assignments; keep=keep)
    elseif ex isa Expr
        args = (_resolve_assignments(arg, assignments; keep=keep) for arg in ex.args)
        return Expr(ex.head, args...)
    end

    return ex
end

"""Resolve a chain of assignment aliases without expanding the resulting expression."""
function _resolve_alias(ex, assignments)
    ex isa Symbol && haskey(assignments, ex) || return ex
    return _resolve_alias(assignments[ex], assignments)
end

"""Return the residual name from the canonical `0.5 * norm(residual)^2` form."""
function _l2_residual_name(phi)
    factors = _is_call(phi, :*, 2) ? phi.args[2:3] : Any[]
    half = findfirst(value -> value isa Number && value == 0.5, factors)
    isnothing(half) && _objective_error(
        "use squared L2, `phi, dr = misfit(dsyn, dobs)`, or `phi = loss(r)`"
    )

    square = factors[3-half]
    _is_call(square, :^, 2) && square.args[3] == 2 ||
        _objective_error("the residual norm must be squared")

    norm_call = square.args[2]
    _is_call(norm_call, :norm, 1) || _objective_error("expected `norm(r)^2`")
    return norm_call.args[2]
end

"""Extract the prediction, derivative name, and fused-kernel misfit keyword."""
function _parse_misfit(phi_name, phi, tuple_loss, assignments, d_obs)
    # A two-output misfit already supplies the data derivative expected by JUDI.
    if !isnothing(tuple_loss) && phi_name == tuple_loss[1]
        derivative_name, misfit = tuple_loss[2], tuple_loss[3]
        prediction = _resolve_assignments(tuple_loss[4], assignments)
        observed = _resolve_assignments(tuple_loss[5], assignments)

        last(_mul_factors(observed)) == d_obs || _objective_error(
            "the misfit must compare prediction with (possibly preconditioned) d_obs"
        )
        return prediction, derivative_name, misfit
    end

    # A unary scalar loss needs ChainRules to produce its data derivative.
    scalar_loss = phi isa Expr && phi.head == :call && length(phi.args) == 2 ? phi : nothing
    if isnothing(scalar_loss)
        residual_name = _l2_residual_name(phi)
        misfit = nothing  # `fwi_objective` and `lsrtm_objective` default to L2.
    else
        residual_name = scalar_loss.args[2]
        residual_name isa Symbol ||
            _objective_error("a scalar misfit must be called on a named residual")
        wrapper = GlobalRef(@__MODULE__, :_chainrules_misfit)
        misfit = Expr(:call, wrapper, scalar_loss.args[1])
    end

    haskey(assignments, residual_name) ||
        _objective_error("the residual must be assigned to a name")
    residual = _resolve_assignments(assignments[residual_name], assignments)

    _is_call(residual, :-, 2) ||
        _objective_error("the residual must be prediction minus observed data")
    observed = _resolve_assignments(residual.args[3], assignments)
    last(_mul_factors(observed)) == d_obs || _objective_error(
        "the residual must subtract (possibly preconditioned) d_obs"
    )

    prediction = _resolve_assignments(residual.args[2], assignments)
    return prediction, residual_name, misfit
end

"""Check that the written gradient is an adjoint chain ending in the derivative."""
function _validate_gradient(gradient, derivative_name, assignments)
    expanded = _resolve_assignments(gradient, assignments; keep=derivative_name)
    factors = _mul_factors(expanded)

    last(factors) == derivative_name ||
        _objective_error("the gradient must end in the residual or misfit derivative")

    is_adjoint(ex) = (ex isa Expr && ex.head == Symbol("'") && length(ex.args) == 1) ||
                     _is_call(ex, :adjoint, 1)
    any(is_adjoint, factors[1:end-1]) ||
        _objective_error("the gradient must contain an adjointed operator")

    return nothing
end

"""Flatten the prediction and ensure that it depends on the optimization variable."""
function _prediction_factors(prediction, x)
    factors = _mul_factors(prediction)
    length(factors) >= 2 ||
        _objective_error("prediction must be `F(x) * q` or `J * x`")

    nonlinear_factor = any(
        factor -> factor isa Expr && factor.head == :call && x in factor.args[2:end],
        factors
    )
    last(factors) == x || nonlinear_factor ||
        _objective_error("prediction must depend on x")

    return factors
end

# -----------------------------------------------------------------------------
# Macro rewrite
# -----------------------------------------------------------------------------

function _rewrite_judi_objective(def)
    def isa Expr && def.head == :function ||
        _objective_error("expected a function definition")

    signature, body = def.args
    signature isa Expr && signature.head == :call ||
        _objective_error("short-form definitions are not supported")
    length(signature.args) == 3 ||
        _objective_error("the objective must take exactly (x, d_obs)")

    x, d_obs = _argument_name.(signature.args[2:3])
    assignments, tuple_loss, returned = _parse_objective_body(body)

    # Resolve only the names returned by the function here. Deeper expansion is
    # delayed until we know which residual/derivative symbol must remain intact.
    phi_name = returned.args[1]
    phi, gradient = (_resolve_alias(ex, assignments) for ex in returned.args)

    prediction, derivative_name, misfit =
        _parse_misfit(phi_name, phi, tuple_loss, assignments, d_obs)
    _validate_gradient(gradient, derivative_name, assignments)
    factors = _prediction_factors(prediction, x)

    # The runtime helper uses concrete operator types to distinguish FWI from
    # LSRTM and to split the flattened product around the propagator/Jacobian.
    helper = GlobalRef(@__MODULE__, :_judi_optimized_objective)
    call = Expr(:call, helper, Expr(:tuple, factors...), x, d_obs)
    if !isnothing(misfit)
        keywords = Expr(:parameters, Expr(:kw, :misfit, misfit))
        insert!(call.args, 2, keywords)
    end

    return Expr(:function, signature, Expr(:block, Expr(:return, call)))
end

_operator_product(xs) = foldl(*, xs)

# These small indirections keep runtime classification separate from the PDE
# API call. Besides making the control flow explicit, they let unit tests use a
# lightweight fake propagator and verify every forwarded argument without
# constructing a Devito model.
_fused_lsrtm(J, x, d_obs; kw...) =
    lsrtm_objective(J.model, J.q, d_obs, x; options=J.options, kw...)

_fused_fwi(F, q, d_obs; kw...) =
    fwi_objective(F.model, q, d_obs; options=F.options, kw...)

# -----------------------------------------------------------------------------
# Runtime dispatch
# -----------------------------------------------------------------------------

"""Execute the fused objective after classifying the runtime operator factors."""
function _judi_optimized_objective(factors::Tuple, x, d_obs; kw...)
    jacobian_index = findfirst(op -> op isa judiAbstractJacobian, factors)

    if !isnothing(jacobian_index)
        # LSRTM has the product `data_precon * J * model_precon * x`.
        J = factors[jacobian_index]
        last(factors) === x || _objective_error("LSRTM prediction must end in x")

        data_factors = factors[1:jacobian_index-1]
        model_factors = factors[jacobian_index+1:end-1]
        data_precon = isempty(data_factors) ? nothing : _operator_product(data_factors)
        model_precon = isempty(model_factors) ? LinearAlgebra.I : _operator_product(model_factors)

        return _fused_lsrtm(J, x, d_obs; data_precon=data_precon,
                            model_precon=model_precon, kw...)
    end

    # FWI has the product `data_precon * F(x) * q`. Unlike LSRTM there is no
    # model-space preconditioner between the propagator and its source.
    propagator_index = findfirst(op -> op isa judiPropagator, factors)
    isnothing(propagator_index) &&
        _objective_error("prediction contains no JUDI propagator")
    propagator_index == length(factors) - 1 ||
        _objective_error("FWI prediction must be `P * F(x) * q`")

    F = factors[propagator_index]
    data_factors = factors[1:propagator_index-1]
    data_precon = isempty(data_factors) ? nothing : _operator_product(data_factors)

    return _fused_fwi(F, factors[end], d_obs; data_precon=data_precon, kw...)
end

"""Adapt a scalar residual loss to JUDI's `(value, data_derivative)` protocol."""
function _chainrules_misfit(loss)
    function chainrules_loss(x, y)
        residual = x - y
        rule = rrule(loss, residual)
        isnothing(rule) &&
            throw(ArgumentError("no ChainRules rrule is defined for $(loss)"))

        value, pullback = rule
        tangents = pullback(one(value))
        derivative = ChainRulesCore.unthunk(tangents[2])

        return value, derivative
    end
end
