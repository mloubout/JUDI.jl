# Macro-expansion tests complement the production PDE comparisons in
# test_gradients.jl. They verify the generated call shape without inventing
# replacement JUDI operators or overriding either objective implementation.

"""Return every call expression in an expanded syntax tree."""
function objective_calls(ex)
    calls = Expr[]
    ex isa Expr || return calls
    ex.head == :call && push!(calls, ex)
    for arg in ex.args
        append!(calls, objective_calls(arg))
    end
    return calls
end

"""Find calls to JUDI's private runtime objective dispatcher."""
function optimized_objective_calls(ex)
    filter(objective_calls(ex)) do call
        callee = call.args[1]
        callee isa GlobalRef && callee.name == :_judi_optimized_objective
    end
end

@testset "@judi_objective lowering" begin
    fwi = macroexpand(@__MODULE__, :(
        @judi_objective function lowered_fwi(x, d_obs)
            d_syn = F(x) * q
            residual = d_syn - d_obs
            value = 0.5f0 * norm(residual)^2
            gradient = J' * residual
            return value, gradient
        end
    ))
    calls = optimized_objective_calls(fwi)
    @test length(calls) == 1
    @test occursin("(F(x), q)", string(only(calls)))
    @test !occursin("norm", string(fwi))

    lsrtm = macroexpand(@__MODULE__, :(
        @judi_objective function lowered_lsrtm(x, d_obs)
            d_syn = Pdata * J * Pmodel * x
            residual = d_syn - Pdata * d_obs
            value = norm(residual)^2 * 0.5
            gradient = Pmodel' * J' * Pdata' * residual
            return value, gradient
        end
    ))
    calls = optimized_objective_calls(lsrtm)
    @test length(calls) == 1
    lowered_call = string(only(calls))
    @test occursin("Pdata", lowered_call)
    @test occursin("J", lowered_call)
    @test occursin("Pmodel", lowered_call)

    # Alias expansion must preserve the order of every data/model factor.
    split = macroexpand(@__MODULE__, :(
        @judi_objective function lowered_split(x, d_obs)
            data_J = Pdata1 * Pdata2 * J
            operator = data_J * Pmodel1 * Pmodel2
            d_syn = operator * x
            observed = Pdata1 * Pdata2 * d_obs
            residual = d_syn - observed
            value = 0.5 * norm(residual)^2
            data_gradient = Pdata2' * Pdata1' * residual
            migrated = J' * data_gradient
            gradient = Pmodel2' * Pmodel1' * migrated
            return value, gradient
        end
    ))
    split_call = string(only(optimized_objective_calls(split)))
    for name in ("Pdata1", "Pdata2", "J", "Pmodel1", "Pmodel2")
        @test occursin(name, split_call)
    end

    # A two-output misfit is forwarded as a keyword instead of being evaluated
    # by the expanded linear-algebra body.
    robust = macroexpand(@__MODULE__, :(
        @judi_objective function lowered_robust(x, d_obs)
            d_syn = Pdata * J * x
            value, derivative = studentst(d_syn, Pdata * d_obs)
            gradient = J' * Pdata' * derivative
            return value, gradient
        end
    ))
    robust_call = string(only(optimized_objective_calls(robust)))
    @test occursin("misfit", robust_call)
    @test occursin("studentst", robust_call)

    # A unary loss lowers through the package's ChainRules adapter.
    # Deliberately do not define `custom_loss` or an rrule here: lowering must
    # only construct syntax and must never execute the loss/adapter. The real
    # ChainRules execution path is covered with a proper PDE setup in
    # test_gradients.jl.
    @test !isdefined(@__MODULE__, :custom_loss)
    chainrules = macroexpand(@__MODULE__, :(
        @judi_objective function lowered_chainrules(x, d_obs)
            d_syn = J * x
            residual = d_syn - d_obs
            value = custom_loss(residual)
            gradient = J' * residual
            return value, gradient
        end
    ))
    chainrules_call = string(only(optimized_objective_calls(chainrules)))
    @test occursin("_chainrules_misfit", chainrules_call)
    @test occursin("custom_loss", chainrules_call)
    @test !isdefined(@__MODULE__, :custom_loss)

    # Unsupported Julia remains untouched and emits the documented warning.
    unsupported = :(
        @judi_objective function lowered_fallback(x, d_obs)
            prediction = J * x
            residual = prediction - d_obs
            if norm(residual) > 1
                residual = residual / norm(residual)
            end
            return norm(residual), J' * residual
        end
    )
    fallback = @test_logs((:warn, r"could not fuse.*original function"),
                          macroexpand(@__MODULE__, unsupported))
    @test isempty(optimized_objective_calls(fallback))
    @test occursin("if", string(fallback))
end
