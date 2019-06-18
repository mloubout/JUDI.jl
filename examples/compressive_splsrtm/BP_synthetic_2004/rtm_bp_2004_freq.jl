using JUDI.TimeModeling, SeisIO, JLD, PyPlot

# Load velocity model (replace with correct paths)
model_path = "/path/to/model/"
data_path = "/path/to/data/"
vp = load(join([model_path, "bp_synthetic_2004_migration_velocity.jld"]))["vp"] / 1f3

# Set up model structure
d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)
model0 = Model(n, d, o, m0)

# Scan directory for segy files and create out-of-core data container
container = segy_scan(data_path, "bp_observed_data", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = "SourceDepth")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)  # 27 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)

###################################################################################################

# Enable optimal checkpointing
opt = Options(limit_m = true,
              buffer_size = 3000f0,
              isic = true,
			  dft_subsampling_factor=8)

# Setup operators
q_dist = generate_distribution(q)
F = judiModeling(info, model0, q.geometry, d_obs.geometry; options=opt)
J = judiJacobian(F, q)

# Set up random frequencies
nfreq = 10
J.options.frequencies = Array{Any}(d_obs.nsrc)
for k=1:d_obs.nsrc
    J.options.frequencies[k] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=nfreq)
end

# Add random noise
d_lin = get_data(d_obs)
for j=1:d_lin.nsrc
    noise = randn(Float32, size(d_lin.data[j]))
    d_lin.data[j] += (noise/norm(vec(noise), Inf)*0.02f0)
end

# Topmute
Ml = judiMarineTopmute2D(35, d_lin.geometry)    # data topmute
d_lin = Ml*d_lin

# RTM
rtm = J'*d_lin
save("bp_synethic_2004_rtm_frequency.jld", "rtm", rtm)
