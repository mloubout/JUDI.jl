export @judi_objective

# This file has two deliberately separate responsibilities:
#
# 1. At macro-expansion time, inspect (but never execute) the user's objective,
#    identify its prediction/misfit/gradient expressions, and replace the body
#    with a compact description of the operator chain.
# 2. At runtime, inspect the concrete types in that chain, recover the JUDI
#    propagator or Jacobian and its surrounding preconditioners, and call the
#    existing fused FWI/LSRTM implementation.
#
# Keeping type-dependent decisions out of the macro is important: symbols such
# as `J`, `Pdata`, and `Pmodel` do not have values while Julia expands a macro.

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
        # Escape the complete rewritten definition so operator and data names
        # continue to resolve in the caller's module rather than inside JUDI.
        esc(_rewrite_judi_objective(def))
    catch err
        # An ArgumentError means the body is valid Julia but outside the small,
        # safe subset understood by this optimization. Preserve its semantics.
        # Other exceptions indicate an implementation bug and must not be hidden.
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
    # Typed/defaulted arguments are represented as nested Expr nodes. Only the
    # leftmost symbol matters when matching uses of `x` and `d_obs` in the body.
    ex isa Symbol && return ex
    ex isa Expr && ex.head in (:(::), :(=), :kw) && return _argument_name(ex.args[1])
    _objective_error("only ordinary positional arguments are supported")
end

"""Flatten a possibly nested multiplication expression from left to right."""
function _mul_factors(ex)
    # Julia may parse `A * B * x` as one n-ary call or as nested binary calls,
    # depending on how intermediate expressions were written. Recursion makes
    # both representations produce the same ordered `[A, B, x]` vector.
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
        # Line-number metadata is useful for diagnostics but has no semantics
        # for the objective pattern and must not be treated as a statement.
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
            # Record the `(phi, gradient)` expression. Unsupported control flow
            # is rejected by the surrounding branches before fusion can occur.
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
    # `keep` is used for the residual derivative. Expanding that one symbol
    # would turn `J' * residual` into `J' * (predicted - observed)` and lose the
    # boundary between the adjoint chain and its data-space input.
    ex == keep && return ex

    if ex isa Symbol && haskey(assignments, ex)
        return _resolve_assignments(assignments[ex], assignments; keep=keep)
    elseif ex isa Expr
        # Rebuild instead of mutating the user's syntax tree. Macro expansion
        # must not alter an expression that may be reused by tooling or fallback.
        args = (_resolve_assignments(arg, assignments; keep=keep) for arg in ex.args)
        return Expr(ex.head, args...)
    end

    return ex
end

"""Resolve a chain of assignment aliases without expanding the resulting expression."""
function _resolve_alias(ex, assignments)
    # Unlike `_resolve_assignments`, this follows only `a = b; b = expression`.
    # It intentionally leaves the final expression's internal symbols intact.
    ex isa Symbol && haskey(assignments, ex) || return ex
    return _resolve_alias(assignments[ex], assignments)
end

"""Return the residual name from the canonical `0.5 * norm(residual)^2` form."""
function _l2_residual_name(phi)
    # Multiplication is commutative for the scalar half, so accept both
    # `0.5 * norm(r)^2` and `norm(r)^2 * 0.5`.
    factors = _is_call(phi, :*, 2) ? phi.args[2:3] : Any[]
    half = findfirst(value -> value isa Number && value == 0.5, factors)
    isnothing(half) && _objective_error(
        "use squared L2, `phi, dr = misfit(dsyn, dobs)`, or `phi = loss(r)`"
    )

    # With two factors, `3-half` selects whichever one was not the scalar half.
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

        # A leading data preconditioner is allowed, but the rightmost operand
        # must still be the observed-data argument from the function signature.
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
        # GlobalRef prevents caller modules from needing to import this private
        # adapter and avoids accidental capture by a same-named local binding.
        misfit = Expr(:call, wrapper, scalar_loss.args[1])
    end

    haskey(assignments, residual_name) ||
        _objective_error("the residual must be assigned to a name")
    residual = _resolve_assignments(assignments[residual_name], assignments)

    # The fused PDE APIs accept synthetic and observed data separately. We can
    # therefore optimize only a residual whose subtraction exposes both sides.
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

    # Julia represents `A'` with head `'`, while an explicit `adjoint(A)` is a
    # normal call. Accept both spellings used by JUDI's linear-algebra API.
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

    # LSRTM ends in `x`; FWI embeds it in the propagator update `F(x)` and ends
    # in the source `q`. These are the only two supported dependency shapes.
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
    # The first two arguments have fixed roles. Additional positional arguments
    # carry context (for example a stochastic shot index) and remain available
    # to expressions in the rewritten body.
    length(signature.args) >= 3 ||
        _objective_error("the objective must take at least (x, d_obs)")

    # Phase 1: establish the two distinguished function arguments and collect
    # the declarative assignments from the body.
    x, d_obs = _argument_name.(signature.args[2:3])
    assignments, tuple_loss, returned = _parse_objective_body(body)

    # Resolve only the names returned by the function here. Deeper expansion is
    # delayed until we know which residual/derivative symbol must remain intact.
    phi_name = returned.args[1]
    phi, gradient = (_resolve_alias(ex, assignments) for ex in returned.args)

    # Phase 2: normalize the supported loss spelling, then independently check
    # that the written gradient and prediction agree with that spelling.
    prediction, derivative_name, misfit =
        _parse_misfit(phi_name, phi, tuple_loss, assignments, d_obs)
    _validate_gradient(gradient, derivative_name, assignments)
    factors = _prediction_factors(prediction, x)

    # The runtime helper uses concrete operator types to distinguish FWI from
    # LSRTM and to split the flattened product around the propagator/Jacobian.
    # Phase 3: emit only values needed at runtime. Building a tuple is crucial:
    # evaluating the original product here would launch the redundant PDE solve
    # this macro exists to eliminate.
    helper = GlobalRef(@__MODULE__, :_judi_optimized_objective)
    call = Expr(:call, helper, Expr(:tuple, factors...), x, d_obs)
    if !isnothing(misfit)
        keywords = Expr(:parameters, Expr(:kw, :misfit, misfit))
        insert!(call.args, 2, keywords)
    end

    # Preserve the original signature exactly (including its name and argument
    # annotations); only replace the function body with the fused call.
    return Expr(:function, signature, Expr(:block, Expr(:return, call)))
end

_operator_product(xs) = foldl(*, xs)

# -----------------------------------------------------------------------------
# Runtime dispatch
# -----------------------------------------------------------------------------

"""Execute the fused objective after classifying the runtime operator factors."""
function _judi_optimized_objective(factors::Tuple, x, d_obs; kw...)
    # Test for a Jacobian first because every judiAbstractJacobian is also a
    # judiPropagator through the abstract-type hierarchy.
    jacobian_index = findfirst(op -> op isa judiAbstractJacobian, factors)

    if !isnothing(jacobian_index)
        # LSRTM has the product `data_precon * J * model_precon * x`.
        J = factors[jacobian_index]
        last(factors) === x || _objective_error("LSRTM prediction must end in x")

        data_factors = factors[1:jacobian_index-1]
        model_factors = factors[jacobian_index+1:end-1]

        # Do not explicitly pass absent preconditioners. The public API defaults
        # them to `nothing`/`I`, but the distributed keyword splitter only accepts
        # concrete preconditioners. Forwarding `data_precon=nothing` would reach
        # `kw_i(nothing, shot)` and fail before starting the PDE solve.
        objective_kw = (; kw...)
        if !isempty(data_factors)
            objective_kw = merge(objective_kw, (; data_precon=_operator_product(data_factors)))
        end
        if !isempty(model_factors)
            objective_kw = merge(objective_kw, (; model_precon=_operator_product(model_factors)))
        end

        return lsrtm_objective(
            J.model, J.q, d_obs, x; options=J.options, objective_kw...
        )
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

    objective_kw = (; kw...)
    if !isempty(data_factors)
        objective_kw = merge(objective_kw, (; data_precon=_operator_product(data_factors)))
    end

    return fwi_objective(
        F.model, factors[end], d_obs; options=F.options, objective_kw...
    )
end

"""Adapt a scalar residual loss to JUDI's `(value, data_derivative)` protocol."""
function _chainrules_misfit(loss)
    function chainrules_loss(x, y)
        # The user's scalar loss is written on a residual, whereas JUDI's misfit
        # protocol receives synthetic and observed arrays as separate arguments.
        residual = x - y
        rule = rrule(loss, residual)
        isnothing(rule) &&
            throw(ArgumentError("no ChainRules rrule is defined for $(loss)"))

        # A scalar objective has cotangent one. The pullback tuple contains the
        # function tangent first and the residual tangent second; only the latter
        # is consumed by the adjoint wave-equation solve.
        value, pullback = rule
        tangents = pullback(one(value))
        derivative = ChainRulesCore.unthunk(tangents[2])

        return value, derivative
    end
end
