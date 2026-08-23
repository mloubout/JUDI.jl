# 2D LS-RTM gradient test with 1 source
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020
#
# Ziyi Yin, ziyi.yin@gatech.edu
# Updated July 2021

### Model
model, model0, dm = setup_model(tti, viscoacoustic, 4)
q, srcGeometry, recGeometry, f0 = setup_geom(model)
dt = srcGeometry.dt[1]

opt = Options(sum_padding=true, free_surface=fs, f0=f0)
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# Observed data
dobs = F*q
dobs0 = F0*q
dm1 = 2f0*circshift(dm, 10)

# Real operators used by the @judi_objective integration tests below. These
# tests intentionally call the production fwi_objective/lsrtm_objective paths;
# no fake backend or dispatch override is involved.
objective_Ml = judiDataMute(q.geometry, dobs.geometry; t0=.2)
objective_Ml2 = judiTimeDerivative(dobs.geometry, 1)
objective_Mr = judiTopmute(model0; taperwidth=10)

# Scalar-loss coverage uses a real ChainRules rule and the same production PDE
# objective as the other cases. The two-argument form is the direct reference
# passed to fwi_objective for comparison.
objective_scalar_loss(r) = sum(abs2, r)
objective_scalar_misfit(x, y) = (objective_scalar_loss(x - y), 2 .* (x - y))
function ChainRulesCore.rrule(::typeof(objective_scalar_loss), r)
	value = objective_scalar_loss(r)
	return value, dy -> (ChainRulesCore.NoTangent(), 2 .* r .* dy)
end

@judi_objective function macro_fwi_l2(x, d_obs)
	d_syn = F0(x) * q
	r = d_syn - d_obs
	phi = .5f0 * norm(r)^2
	g = J' * r
	return phi, g
end

@judi_objective function macro_fwi_chainrules(x, d_obs)
	d_syn = objective_Ml * F0(x) * q
	r = d_syn - objective_Ml * d_obs
	phi = objective_scalar_loss(r)
	g = J' * objective_Ml' * r
	return phi, g
end

@judi_objective function macro_fwi_studentst(x, d_obs)
	d_syn = objective_Ml * F0(x) * q
	phi, dr = studentst(d_syn, objective_Ml * d_obs)
	g = J' * objective_Ml' * dr
	return phi, g
end


@judi_objective function macro_lsrtm_split(x, d_obs)
	data_J = objective_Ml * objective_Ml2 * J
	full_operator = data_J * objective_Mr * objective_Mr
	d_syn = full_operator * x
	d_precon = objective_Ml * objective_Ml2 * d_obs
	r = d_syn - d_precon
	phi = .5f0 * norm(r)^2
	data_adjoint = objective_Ml2' * objective_Ml' * r
	migrated = J' * data_adjoint
	g = objective_Mr' * objective_Mr' * migrated
	return phi, g
end

@judi_objective function macro_lsrtm_l2(x, d_obs)
	d_syn = objective_Ml * objective_Ml2 * J * objective_Mr * objective_Mr * x
	r = d_syn - objective_Ml * objective_Ml2 * d_obs
	phi = .5f0 * norm(r)^2
	g = objective_Mr' * objective_Mr' * J' * objective_Ml2' * objective_Ml' * r
	return phi, g
end


@testset "@judi_objective production FWI/LSRTM dispatch" begin
	# These checks guard against the macro silently taking its documented
	# linear-algebra fallback. The executable methods below must lower to JUDI's
	# runtime dispatcher; that dispatcher directly calls fwi_objective or
	# lsrtm_objective, and the result comparisons verify the selected branch.
	for objective in (macro_fwi_l2, macro_fwi_chainrules, macro_fwi_studentst,
				  macro_lsrtm_l2, macro_lsrtm_split)
		lowered = only(code_lowered(objective, Tuple{Any, Any}))
		@test occursin("_judi_optimized_objective", string(lowered))
	end

	# Baseline nonlinear FWI with the default mean-square misfit.
	macro_value, macro_gradient = @test_logs(
		(:debug, r"Executing fused fwi_objective"),
		match_mode=:any, min_level=Base.CoreLogging.Debug,
		macro_fwi_l2(model0, dobs)
	)
	direct_value, direct_gradient = fwi_objective(model0, q, dobs; options=opt)
	@test macro_value == direct_value
	@test macro_gradient == direct_gradient

	# A unary residual loss must obtain its derivative from its real ChainRules
	# rule and produce the same PDE result as the explicit two-output misfit.
	macro_value, macro_gradient = macro_fwi_chainrules(model0, dobs)
	direct_value, direct_gradient = fwi_objective(
		model0, q, dobs; options=opt,
		misfit=objective_scalar_misfit, data_precon=objective_Ml
	)
	@test macro_value == direct_value
	@test macro_gradient == direct_gradient

	# A custom two-output misfit and data preconditioner must both be forwarded.
	macro_value, macro_gradient = macro_fwi_studentst(model0, dobs)
	direct_value, direct_gradient = fwi_objective(
		model0, q, dobs; options=opt, misfit=studentst, data_precon=objective_Ml
	)
	@test macro_value == direct_value
	@test macro_gradient == direct_gradient

	# LSRTM exercises multi-factor inference on both sides of J. This catches
	# ordering errors while comparing against the public API itself.
	macro_value, macro_gradient = @test_logs(
		(:debug, r"Executing fused lsrtm_objective"),
		match_mode=:any, min_level=Base.CoreLogging.Debug,
		macro_lsrtm_l2(dm, dobs)
	)
	direct_value, direct_gradient = lsrtm_objective(
		model0, q, dobs, dm; options=opt,
		data_precon=objective_Ml*objective_Ml2,
		model_precon=objective_Mr*objective_Mr
	)
	@test macro_value == direct_value
	@test macro_gradient == direct_gradient

	# Splitting the exact same operator chain across intermediate assignments
	# must resolve to the same production lsrtm_objective invocation.
	split_value, split_gradient = macro_lsrtm_split(dm, dobs)
	@test split_value == direct_value
	@test split_gradient == direct_gradient
end

ftol = (tti | fs | viscoacoustic) ? 1f-1 : 1f-2

# ###################################################################################################

@testset "FWI gradient test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	# FWI gradient and function value for m0
	Jm0, grad = fwi_objective(model0, q, dobs; options=opt)
	# Check get same misfit as l2 misifit on forward data
	Jm01 = .5f0 * norm(F(model0)*q - dobs)^2
	@test Jm0 ≈ Jm01

	grad_test(x-> .5f0*norm(F(;m=x)*q - dobs)^2, model0.m , dm, grad)

end

###################################################################################################
@testset "FWI preconditionners test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	Ml = judiDataMute(q.geometry, dobs.geometry; t0=.2)
	Ml2 = judiTimeDerivative(dobs.geometry, 1)


	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=Ml)
	ghand = J'*Ml*(F0*q - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=[Ml, Ml2])
	ghand = J'*Ml*Ml2*(F0*q - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=Ml*Ml2)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)
end


@testset "LSRTM preconditionners test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	Mr = judiTopmute(model0; taperwidth=10)
	Ml = judiDataMute(q.geometry, dobs.geometry)
	Ml2 = judiTimeDerivative(dobs.geometry, 1)
	Mr2 = judiIllumination(J)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=Ml, model_precon=Mr)
	ghand = J'*Ml*(J*Mr*dm - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=[Ml, Ml2], model_precon=[Mr, Mr2])
	ghand = J'*Ml*Ml2*(J*Mr2*Mr*dm - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=Ml*Ml2, model_precon=Mr*Mr2)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

end

###################################################################################################

@testset "LSRTM gradient test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic). freesurface $(fs), nlind $(nlind)" for nlind=[true, false]
	@timeit TIMEROUTPUT "LSRTM gradient test, nlind=$(nlind)" begin
		# LS-RTM gradient and function value for m0
		dD = nlind ? (dobs - dobs0) : dobs
		Jm0, grad = lsrtm_objective(model0, q, dD, dm; options=opt, nlind=nlind)

		# Gradient test
		grad_test(x-> lsrtm_objective(model0, q, dD, x;options=opt, nlind=nlind)[1], dm, dm1, grad)

		# test that with zero dm we get the same as fwi_objective for residual
		if nlind
			Jls, gradls = @single_threaded lsrtm_objective(model0, q, dobs, 0f0.*dm; options=opt, nlind=true)
			Jfwi, gradfwi = @single_threaded fwi_objective(model0, q, dobs; options=opt)
			@test isapprox(Jls, Jfwi; rtol=0f0, atol=0f0)
			@test isapprox(gradls, gradfwi; rtol=0f0, atol=0f0)
		end
	end
end

# Test if lsrtm_objective produces the same value/gradient as is done by the correct algebra
@testset "LSRTM gradient linear algebra test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic), freesurface $(fs)" begin
	# Draw a random case to avoid long CI.
	ic = rand(["isic", "fwi", "as"])
	printstyled("LSRTM validity with dft, IC=$(ic)\n", color=:blue)
    @timeit TIMEROUTPUT "LSRTM validity with dft, IC=$(ic)" begin
		ftol = fs ? 1f-3 : 5f-4
		q_dist = generate_distribution(q)
		freq = [select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=2)
				for j=1:dobs.nsrc]

		J.options.free_surface = fs
		J.options.IC = ic
		J.options.frequencies = freq

		d_res = dobs0 + J*dm1 - dobs
		Jm0_1 = 0.5f0 * norm(d_res)^2f0
		grad_1 = @single_threaded J'*d_res

		opt = J.options
		Jm0, grad = @single_threaded lsrtm_objective(model0, q, dobs, dm1; options=opt, nlind=true)
		Jm01, grad1 = @single_threaded lsrtm_objective(model0, q, dobs-dobs0, dm1; options=opt, nlind=false)
	
		@show Jm0, Jm0_1, norm(grad), norm(grad_1), norm(grad1)
		@test isapprox(grad, grad_1; rtol=ftol)
		@test isapprox(Jm0, Jm0_1; rtol=ftol)
		@test isapprox(grad, grad1; rtol=ftol)
		@test isapprox(Jm0, Jm01; rtol=ftol)
	end
end
