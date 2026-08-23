_objective_test_loss(r) = sum(abs2, r)
function rrule(::typeof(_objective_test_loss), r)
    y = _objective_test_loss(r)
    y, dy -> (nothing, 2 .* r .* dy)
end

# Lightweight runtime doubles let these tests execute a rewritten objective
# without initializing Devito. `_fused_lsrtm` is the narrow seam immediately
# before the real `lsrtm_objective` call.
struct _ObjectiveTestContext
    model
    options
end

struct _ObjectiveTestJacobian <: judiAbstractJacobian{Float32, :born, Nothing}
    m
    n
    F::_ObjectiveTestContext
    q
end

struct _ObjectiveTestPreconditioner
    name::Symbol
end

struct _ObjectiveTestPropagator <: judiPropagator{Float32, :forward}
    model
    options
end

# These helpers are intentionally internal and therefore are not brought into
# Main by `using JUDI` in runtests.jl. Import every helper exercised directly
# by this test file rather than relying on package-internal name resolution.
import JUDI: _chainrules_misfit, _fused_fwi, _fused_lsrtm
function _fused_lsrtm(J::_ObjectiveTestJacobian, x, d_obs;
                      data_precon, model_precon, misfit=mse)
    # Return a deterministic value as well as all arguments inferred by the
    # macro, so the test checks behavior rather than merely printed AST text.
    value, derivative = misfit(x, d_obs)
    return value, (; derivative, J, data_precon, model_precon)
end


function _fused_fwi(F::_ObjectiveTestPropagator, q, d_obs;
                    data_precon, misfit=mse)
    value, derivative = misfit(F.model, d_obs)
    return value, (; derivative, F, q, data_precon)
end

_objective_runtime_misfit(x, y) = (sum(abs2, x - y), 2 .* (x - y))

const _objective_test_context = _ObjectiveTestContext(:model, :options)
const _objective_test_J = _ObjectiveTestJacobian(
    nothing, nothing, _objective_test_context, :source
)
const _objective_test_data_precon = _ObjectiveTestPreconditioner(:data)
const _objective_test_model_precon = _ObjectiveTestPreconditioner(:model)
const _objective_test_source = :source

_objective_test_F(x) = _ObjectiveTestPropagator(x, :options)

@judi_objective function executable_objective(x, d_obs)
    predicted = _objective_test_data_precon * _objective_test_J *
                _objective_test_model_precon * x
    value, derivative = _objective_runtime_misfit(
        predicted, _objective_test_data_precon * d_obs
    )
    gradient = _objective_test_model_precon' * _objective_test_J' *
               _objective_test_data_precon' * derivative
    return value, gradient
end


@judi_objective function executable_fwi_objective(x, d_obs)
    predicted = _objective_test_data_precon * _objective_test_F(x) *
                _objective_test_source
    value, derivative = _objective_runtime_misfit(
        predicted, _objective_test_data_precon * d_obs
    )
    gradient = _objective_test_J' * _objective_test_data_precon' * derivative
    return value, gradient
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

    invalid = :(@judi_objective function invalid_dsl(x, d)
        residual = J * x - d
        return 0.5 * norm(residual), J' * residual
    end)
    fallback = @test_logs((:warn, r"could not fuse.*original function"),
                          macroexpand(@__MODULE__, invalid))
    fallback_text = string(fallback)
    @test occursin("invalid_dsl", fallback_text)
    @test occursin("norm", fallback_text)
    @test !occursin("_judi_optimized_objective", fallback_text)

    unsupported = :(@judi_objective function unsupported_body(x, d)
        predicted = J * x
        residual = predicted - d
        if norm(residual) > 1
            residual = residual / norm(residual)
        end
        return norm(residual), J' * residual
    end)
    unsupported_fallback = @test_logs((:warn, r"body may contain only assignments"),
                                      macroexpand(@__MODULE__, unsupported))
    @test occursin("if", string(unsupported_fallback))

    @test_logs (:warn, r"could not fuse.*original function") eval(:(
        @judi_objective function passthrough_objective(x, d)
            value = sum(x) + sum(d)
            gradient = x .+ d
            return value, gradient
        end
    ))
    @test passthrough_objective([1, 2], [3, 4]) == (10, [4, 6])
end

@testset "Optimized objective runtime behavior" begin
    x = [2.0, 4.0]
    d_obs = [1.0, 1.0]
    value, result = executable_objective(x, d_obs)

    @test value == 10.0
    @test result.derivative == [2.0, 6.0]
    @test result.J === _objective_test_J
    @test result.data_precon === _objective_test_data_precon
    @test result.model_precon === _objective_test_model_precon

    fwi_value, fwi_result = executable_fwi_objective(x, d_obs)
    @test fwi_value == 10.0
    @test fwi_result.derivative == [2.0, 6.0]
    @test fwi_result.F.model === x
    @test fwi_result.q === _objective_test_source
    @test fwi_result.data_precon === _objective_test_data_precon
end
