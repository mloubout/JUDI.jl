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
_is_call(ex, f, n=nothing) = ex isa Expr && ex.head == :call && ex.args[1] == f &&
    (isnothing(n) || length(ex.args) == n + 1)

function _argument_name(ex)
    ex isa Symbol && return ex
    ex isa Expr && ex.head in (:(::), :(=), :kw) && return _argument_name(ex.args[1])
    _objective_error("only ordinary positional arguments are supported")
end

function _mul_factors(ex)
    ex isa Expr && ex.head == :call && ex.args[1] == :* || return Any[ex]
    reduce(vcat, (_mul_factors(arg) for arg in ex.args[2:end]); init=Any[])
end

function _rewrite_judi_objective(def)
    def isa Expr && def.head == :function || _objective_error("expected a function definition")
    signature, body = def.args
    signature isa Expr && signature.head == :call || _objective_error("short-form definitions are not supported")
    length(signature.args) == 3 || _objective_error("the objective must take exactly (x, d_obs)")
    x, d_obs = _argument_name.(signature.args[2:3])

    statements = body isa Expr && body.head == :block ? body.args : Any[body]
    assignments = Dict{Symbol, Any}()
    tuple_loss = nothing
    returned = nothing
    for statement in statements
        statement isa LineNumberNode && continue
        if statement isa Expr && statement.head == :(=)
            lhs, rhs = statement.args
            if lhs isa Symbol
                assignments[lhs] = rhs
            elseif lhs isa Expr && lhs.head == :tuple && length(lhs.args) == 2 &&
                   rhs isa Expr && rhs.head == :call && length(rhs.args) == 3
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

    resolve(ex) = ex isa Symbol && haskey(assignments, ex) ? resolve(assignments[ex]) : ex
    function resolve_deep(ex, keep=nothing)
        ex == keep && return ex
        if ex isa Symbol && haskey(assignments, ex)
            return resolve_deep(assignments[ex], keep)
        elseif ex isa Expr
            return Expr(ex.head, (resolve_deep(arg, keep) for arg in ex.args)...)
        end
        ex
    end
    phi_name = returned.args[1]
    phi, gradient = resolve.(returned.args)
    misfit = nothing

    if !isnothing(tuple_loss) && phi_name == tuple_loss[1]
        residual_name, misfit = tuple_loss[2], tuple_loss[3]
        prediction = resolve_deep(tuple_loss[4])
        last(_mul_factors(resolve_deep(tuple_loss[5]))) == d_obs ||
            _objective_error("the misfit must compare prediction with (possibly preconditioned) d_obs")
    else
        # First identify the residual. This also permits the scalar `loss(r)` form.
        scalar_loss = phi isa Expr && phi.head == :call && length(phi.args) == 2 ? phi : nothing
        if !isnothing(scalar_loss)
            residual_name = scalar_loss.args[2]
            residual_name isa Symbol || _objective_error("a scalar misfit must be called on a named residual")
            wrapper = GlobalRef(@__MODULE__, :_chainrules_misfit)
            misfit = Expr(:call, wrapper, scalar_loss.args[1])
        else
            factors = _is_call(phi, :*, 2) ? phi.args[2:3] : Any[]
            half = findfirst(v -> v isa Number && v == .5, factors)
            isnothing(half) && _objective_error("use squared L2, `phi, dr = misfit(dsyn, dobs)`, or `phi = loss(r)`")
            square = factors[3-half]
            _is_call(square, :^, 2) && square.args[3] == 2 || _objective_error("the residual norm must be squared")
            normcall = square.args[2]
            _is_call(normcall, :norm, 1) || _objective_error("expected `norm(r)^2`")
            residual_name = normcall.args[2]
        end
        haskey(assignments, residual_name) || _objective_error("the residual must be assigned to a name")
        residual = resolve_deep(assignments[residual_name])
        _is_call(residual, :-, 2) || _objective_error("the residual must be prediction minus observed data")
        last(_mul_factors(resolve(residual.args[3]))) == d_obs ||
            _objective_error("the residual must subtract (possibly preconditioned) d_obs")
        prediction = resolve_deep(residual.args[2])
    end

    gradient_factors = _mul_factors(resolve_deep(gradient, residual_name))
    last(gradient_factors) == residual_name ||
        _objective_error("the gradient must end in the residual or misfit derivative")
    any(a -> (a isa Expr && a.head == Symbol("'") && length(a.args) == 1) ||
             _is_call(a, :adjoint, 1), gradient_factors[1:end-1]) ||
        _objective_error("the gradient must contain an adjointed operator")

    prediction_factors = _mul_factors(prediction)
    length(prediction_factors) >= 2 || _objective_error("prediction must be `F(x) * q` or `J * x`")
    last(prediction_factors) == x ||
        any(f -> f isa Expr && f.head == :call && x in f.args[2:end], prediction_factors) ||
        _objective_error("prediction must depend on x")

    helper = GlobalRef(@__MODULE__, :_judi_optimized_objective)
    call = Expr(:call, helper, Expr(:tuple, prediction_factors...), x, d_obs)
    !isnothing(misfit) && insert!(call.args, 2, Expr(:parameters, Expr(:kw, :misfit, misfit)))
    Expr(:function, signature, Expr(:block, Expr(:return, call)))
end

_operator_product(xs) = foldl(*, xs)

function _judi_optimized_objective(factors::Tuple, x, d_obs; kw...)
    ji = findfirst(op -> op isa judiAbstractJacobian, factors)
    if !isnothing(ji)
        J = factors[ji]
        last(factors) === x || _objective_error("LSRTM prediction must end in x")
        left, right = factors[1:ji-1], factors[ji+1:end-1]
        Pd = isempty(left) ? nothing : _operator_product(left)
        Pm = isempty(right) ? LinearAlgebra.I : _operator_product(right)
        return lsrtm_objective(J.model, J.q, d_obs, x; options=J.options,
                               data_precon=Pd, model_precon=Pm, kw...)
    end
    fi = findfirst(op -> op isa judiPropagator, factors)
    isnothing(fi) && _objective_error("prediction contains no JUDI propagator")
    fi == length(factors)-1 || _objective_error("FWI prediction must be `P * F(x) * q`")
    F = factors[fi]
    left = factors[1:fi-1]
    Pd = isempty(left) ? nothing : _operator_product(left)
    fwi_objective(F.model, factors[end], d_obs; options=F.options, data_precon=Pd, kw...)
end

function _chainrules_misfit(loss)
    function chainrules_loss(x, y)
        residual = x - y
        rule = rrule(loss, residual)
        isnothing(rule) && throw(ArgumentError("no ChainRules rrule is defined for $(loss)"))
        value, pullback = rule
        tangents = pullback(one(value))
        value, ChainRulesCore.unthunk(tangents[2])
    end
end
