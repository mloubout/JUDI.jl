@testset "Optimized objective macro" begin
    expanded = macroexpand(@__MODULE__, :(@judi_objective function fwi_dsl(x, d)
        predicted = F(x) * q
        residual = predicted - d
        value = 0.5f0 * norm(residual)^2
        gradient = J' * residual
        return value, gradient
    end))
    @test occursin("_judi_optimized_objective", string(expanded))
    @test !occursin("norm", string(expanded))

    expanded_ls = macroexpand(@__MODULE__, :(@judi_objective function ls_dsl(x, d)
        residual = J * x - d
        return 0.5 * norm(residual)^2, J' * residual
    end))
    @test occursin("_judi_optimized_objective", string(expanded_ls))

    @test_throws ArgumentError eval(:(@judi_objective function invalid_dsl(x, d)
        residual = J * x - d
        return 0.5 * norm(residual), J' * residual
    end))
end
