_objective_test_loss(r) = sum(abs2, r)
function rrule(::typeof(_objective_test_loss), r)
    y = _objective_test_loss(r)
    y, dy -> (nothing, 2 .* r .* dy)
end

@testset "Optimized objective macro" begin
    fwi = macroexpand(@__MODULE__, :(@judi_objective function fwi_dsl(x, d)
        predicted = F(x) * q
        residual = predicted - d
        value = 0.5f0 * norm(residual)^2
        gradient = J' * residual
        return value, gradient
    end))
    @test occursin("_judi_optimized_objective", string(fwi))
    @test !occursin("norm", string(fwi))

    lsrtm = macroexpand(@__MODULE__, :(@judi_objective function ls_dsl(x, d)
        residual = J * x - d
        return 0.5 * norm(residual)^2, J' * residual
    end))
    @test occursin("_judi_optimized_objective", string(lsrtm))

    robust = macroexpand(@__MODULE__, :(@judi_objective function robust(x, d)
        predicted = P * J * M * x
        value, dr = studentst(predicted, P * d)
        gradient = M' * J' * P' * dr
        return value, gradient
    end))
    @test occursin("studentst", string(robust))

    # Products may be assembled over any number of intermediate assignments. An
    # illumination preconditioner is simply inferred as a model-space factor.
    split = macroexpand(@__MODULE__, :(@judi_objective function split_objective(x, d)
        PJ = P * J
        A = PJ * illumination
        predicted = A * x
        observed = P * d
        residual = predicted - observed
        value = 0.5 * norm(residual)^2
        adjoint_data = P' * residual
        migrated = J' * adjoint_data
        gradient = illumination' * migrated
        return value, gradient
    end))
    split_text = string(split)
    @test occursin("illumination", split_text)
    @test occursin("P", split_text)
    @test occursin("J", split_text)

    chainrules = macroexpand(@__MODULE__, :(@judi_objective function general_loss(x, d)
        predicted = J * x
        residual = predicted - d
        value = _objective_test_loss(residual)
        gradient = J' * residual
        return value, gradient
    end))
    @test occursin("_chainrules_misfit", string(chainrules))
    value, derivative = _chainrules_misfit(_objective_test_loss)([2.0, 4.0], [1.0, 1.0])
    @test value == 10.0
    @test derivative == [2.0, 6.0]

    @test_throws ArgumentError eval(:(@judi_objective function invalid_dsl(x, d)
        residual = J * x - d
        return 0.5 * norm(residual), J' * residual
    end))
end
