export @judi_objective

"""
    @judi_objective function objective(x, d_obs)
        d_syn = F(x) * q
        r = d_syn - d_obs
        phi = 0.5f0 * norm(r)^2
        g = J' * r
        return phi, g
    end

Compile the canonical linear-algebra expression for an FWI or LSRTM least-squares
objective into JUDI's fused objective implementation. This avoids the extra forward
solve which would otherwise be performed by the Jacobian adjoint.

The macro accepts two forms for the prediction: `F(x) * q` (FWI) and `J * x`
(LSRTM). Intermediate names are arbitrary, but the residual must be prediction
minus the function's second argument, the objective must be `0.5 * norm(r)^2`,
and the returned gradient must have the form `A' * r`. Unsupported expressions
produce an error when the function is defined rather than silently changing their
semantics.
"""
macro judi_objective(def)
    try
        return esc(_rewrite_judi_objective(def))
    catch err
        err isa ArgumentError || rethrow()
        return :(throw(ArgumentError($(err.msg))))
    end
end

_objective_error(message) = throw(ArgumentError("@judi_objective: " * message))
_is_call(ex, f, n=nothing) = ex isa Expr && ex.head == :call && ex.args[1] == f && (isnothing(n) || length(ex.args) == n + 1)

function _argument_name(ex)
    ex isa Symbol && return ex
    ex isa Expr && ex.head in (:(::), :(=), :kw) && return _argument_name(ex.args[1])
    _objective_error("only ordinary positional arguments are supported")
end

function _rewrite_judi_objective(def)
    def isa Expr && def.head == :function || _objective_error("expected a function definition")
    signature, body = def.args
    signature isa Expr && signature.head == :call || _objective_error("short-form definitions are not supported")
    length(signature.args) == 3 || _objective_error("the objective must take exactly (x, d_obs)")
    x, d_obs = _argument_name.(signature.args[2:3])

    statements = body isa Expr && body.head == :block ? body.args : Any[body]
    assignments = Dict{Symbol, Any}()
    returned = nothing
    for statement in statements
        statement isa LineNumberNode && continue
        if statement isa Expr && statement.head == :(=) && statement.args[1] isa Symbol
            assignments[statement.args[1]] = statement.args[2]
        elseif statement isa Expr && statement.head == :return
            returned = only(statement.args)
        else
            _objective_error("the body may contain only assignments followed by return (phi, g)")
        end
    end
    returned isa Expr && returned.head == :tuple && length(returned.args) == 2 ||
        _objective_error("expected `return phi, g`")

    resolve(ex) = ex isa Symbol && haskey(assignments, ex) ? resolve(assignments[ex]) : ex
    phi, gradient = resolve.(returned.args)

    # Recognize 0.5 * norm(r)^2 (with the scalar on either side).
    factors = _is_call(phi, :*, 2) ? phi.args[2:3] : Any[]
    length(factors) == 2 || _objective_error("the objective must be `0.5 * norm(r)^2`")
    half_index = findfirst(v -> v isa Number && v == 0.5, factors)
    isnothing(half_index) && _objective_error("the objective must be `0.5 * norm(r)^2`")
    square = factors[3-half_index]
    _is_call(square, :^, 2) && square.args[3] == 2 || _objective_error("the residual norm must be squared")
    norm_call = square.args[2]
    _is_call(norm_call, :norm, 1) || _objective_error("the objective must use `norm(r)^2`")
    residual_name = norm_call.args[2]
    residual_name isa Symbol || _objective_error("the residual must be assigned to a name")
    residual = resolve(residual_name)
    _is_call(residual, :-, 2) && residual.args[3] == d_obs ||
        _objective_error("the residual must be `prediction - d_obs`")
    prediction = resolve(residual.args[2])

    gradient = resolve(gradient)
    _is_call(gradient, :*, 2) && gradient.args[3] == residual_name ||
        _objective_error("the gradient must be an adjoint operator times the residual")
    adj = gradient.args[2]
    ((adj isa Expr && adj.head == Symbol("'") && length(adj.args) == 1) ||
     _is_call(adj, :adjoint, 1)) || _objective_error("the gradient operator must be adjointed with `'`")

    _is_call(prediction, :*, 2) || _objective_error("prediction must be `F(x) * q` or `J * x`")
    operator, operand = prediction.args[2:3]
    helper = GlobalRef(@__MODULE__, :_judi_optimized_objective)
    call = if operator isa Expr && operator.head == :call && length(operator.args) == 2 && operator.args[2] == x
        # Nonlinear FWI: pass the updated propagator and source to the fused kernel.
        Expr(:call, helper, operator, operand, d_obs)
    elseif operand == x
        # Linear LSRTM: the Jacobian stores its background model and source.
        Expr(:call, helper, operator, x, d_obs)
    else
        _objective_error("prediction must be `F(x) * q` or `J * x`")
    end
    Expr(:function, signature, Expr(:block, body.args[1] isa LineNumberNode ? body.args[1] : nothing, Expr(:return, call)))
end

_judi_optimized_objective(F::judiPropagator, q, d_obs) =
    fwi_objective(F.model, q, d_obs; options=F.options)

_judi_optimized_objective(J::judiAbstractJacobian, dm, d_obs) =
    lsrtm_objective(J.model, J.q, d_obs, dm; options=J.options)
