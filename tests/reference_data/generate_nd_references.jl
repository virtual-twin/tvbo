#!/usr/bin/env julia
"""
Generate HDF5 reference data from original NetworkDynamics.jl examples.

Run this script once to create reference .h5 files that pytest uses for
numerical comparison against TVBO-generated results.

Usage:
    julia --project=@. generate_nd_references.jl

Requires: NetworkDynamics, Graphs, OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK,
          SimpleWeightedGraphs, StableRNGs, HDF5, DelimitedFiles
"""

using Pkg
# Ensure all required packages are installed
for pkg in ["NetworkDynamics", "Graphs", "OrdinaryDiffEqTsit5",
            "OrdinaryDiffEqSDIRK", "SimpleWeightedGraphs", "StableRNGs",
            "HDF5", "DelimitedFiles"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using Graphs
using NetworkDynamics
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqSDIRK
using SimpleWeightedGraphs
using StableRNGs
using HDF5
using DelimitedFiles

outdir = @__DIR__

# =============================================================================
# Example 1: Network Diffusion (Getting Started)
# =============================================================================
println("=== Generating: diffusion ===")

function diffusionedge_g!(e_dst, v_src, v_dst, p, t)
    e_dst .= v_src .- v_dst
    nothing
end

function diffusionvertex_f!(dv, v, esum, p, t)
    dv .= esum
    nothing
end

N_diff = 20
k_diff = 4
g_diff = barabasi_albert(N_diff, k_diff; seed=1)
nd_diffusion_vertex = VertexModel(; f=diffusionvertex_f!, g=StateMask(1:1), dim=1)
nd_diffusion_edge = EdgeModel(; g=AntiSymmetric(diffusionedge_g!), outsym=[:flow])
nd_diff = Network(g_diff, nd_diffusion_vertex, nd_diffusion_edge)

rng_diff = StableRNG(1)
x0_diff = randn(rng_diff, N_diff)

ode_prob_diff = ODEProblem(nd_diff, x0_diff, (0.0, 2.0))
sol_diff = solve(ode_prob_diff, Tsit5())

# Sample at regular intervals for comparison
t_diff = collect(range(0.0, 2.0, length=201))
u_diff = hcat([sol_diff(t) for t in t_diff]...)  # (N, T)

# Save adjacency matrix for graph comparison
adj_diff = Matrix(Float64.(adjacency_matrix(g_diff)))

h5open(joinpath(outdir, "diffusion_reference.h5"), "w") do f
    f["t"] = t_diff
    f["u"] = u_diff  # (N_nodes, T)
    f["x0"] = x0_diff
    f["adjacency"] = adj_diff
    attrs(f)["N"] = N_diff
    attrs(f)["k"] = k_diff
    attrs(f)["n_sv"] = 1
    attrs(f)["duration"] = 2.0
    attrs(f)["solver"] = "Tsit5"
    attrs(f)["description"] = "Network diffusion on Barabasi-Albert graph (StableRNG seed=1)"
end
println("  Saved: diffusion_reference.h5  ($(length(t_diff)) timesteps, $N_diff nodes)")


# =============================================================================
# Example 2: Heterogeneous Kuramoto Oscillators
# =============================================================================
println("=== Generating: kuramoto ===")

N_kur = 8
g_kur = watts_strogatz(N_kur, 2, 0)

function kuramoto_edge!(e, θ_s, θ_d, (K,), t)
    e[1] = K * sin(θ_s[1] - θ_d[1])
    nothing
end
edge_kur = EdgeModel(g=AntiSymmetric(kuramoto_edge!), outdim=1, psym=[:K=>3])

function kuramoto_vertex!(dθ, θ, esum, (ω0,), t)
    dθ[1] = ω0 + esum[1]
    nothing
end
vertex_kur = VertexModel(f=kuramoto_vertex!, g=StateMask(1:1), sym=[:θ], psym=[:ω0], name=:kuramoto)

nw_kur = Network(g_kur, vertex_kur, edge_kur)

# Set parameters exactly as in the tutorial
p_kur = NWParameter(nw_kur)
ω_kur = collect(1:N_kur) ./ N_kur
ω_kur .-= sum(ω_kur) / N_kur
p_kur.v[:, :ω0] = ω_kur

x0_kur = collect(1:N_kur) ./ N_kur
x0_kur .-= sum(x0_kur) ./ N_kur

prob_kur = ODEProblem(nw_kur, x0_kur, (0.0, 10.0), pflat(p_kur))
sol_kur = solve(prob_kur, Tsit5())

t_kur = collect(range(0.0, 10.0, length=201))
u_kur = hcat([sol_kur(t) for t in t_kur]...)

adj_kur = Matrix(Float64.(adjacency_matrix(g_kur)))

h5open(joinpath(outdir, "kuramoto_reference.h5"), "w") do f
    f["t"] = t_kur
    f["u"] = u_kur
    f["x0"] = x0_kur
    f["omega0"] = ω_kur
    f["adjacency"] = adj_kur
    attrs(f)["N"] = N_kur
    attrs(f)["K"] = 3.0
    attrs(f)["n_sv"] = 1
    attrs(f)["duration"] = 10.0
    attrs(f)["solver"] = "Tsit5"
    attrs(f)["description"] = "Kuramoto on Watts-Strogatz ring (N=8, k=2, p=0)"
end
println("  Saved: kuramoto_reference.h5  ($(length(t_kur)) timesteps, $N_kur nodes)")


# =============================================================================
# Example 3: FitzHugh-Nagumo on Directed Weighted Brain Network
# =============================================================================
println("=== Generating: fitzhugh_nagumo ===")

# Load connectivity matrix
# The Norm_G_DTI.txt file is in the tvbo docs directory
dti_file = joinpath(@__DIR__, "..", "..", "docs", "Interoperability",
                    "NetworkDynamics.jl", "Norm_G_DTI.txt")
if !isfile(dti_file)
    # Fallback: try the NetworkDynamics package path
    dti_file = joinpath(pkgdir(NetworkDynamics), "docs", "examples", "Norm_G_DTI.txt")
end
@assert isfile(dti_file) "Cannot find Norm_G_DTI.txt at $dti_file"
G_fhn = readdlm(dti_file, ',', Float64, '\n')

g_weighted_fhn = SimpleWeightedDiGraph(G_fhn)
edge_weights_fhn = getfield.(collect(edges(g_weighted_fhn)), :weight)
g_directed_fhn = SimpleDiGraph(g_weighted_fhn)

Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, esum, p, t)
    (a, ϵ) = p
    dv[1] = v[1] - v[1]^3 / 3 - v[2] + esum[1]
    dv[2] = (v[1] - a) * ϵ
    nothing
end
vertex_fhn = VertexModel(f=fhn_electrical_vertex!, g=1, sym=[:u, :v], psym=[:a=>0.5, :ϵ=>0.05])

Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, (w, σ), t)
    e[1] = w * (v_s[1] - v_d[1]) * σ
    nothing
end
electricaledge_fhn = EdgeModel(g=Directed(electrical_edge!), outdim=1, psym=[:weight, :σ=>0.5])

fhn_network = Network(g_directed_fhn, vertex_fhn, electricaledge_fhn)

p_fhn = NWParameter(fhn_network)
p_fhn.e[1:ne(g_directed_fhn), :weight] = edge_weights_fhn

x0_fhn = randn(StableRNG(42), dim(fhn_network)) * 5

prob_fhn = ODEProblem(fhn_network, x0_fhn, (0.0, 200.0), pflat(p_fhn))
sol_fhn = solve(prob_fhn, AutoTsit5(TRBDF2()))

t_fhn = collect(range(0.0, 200.0, length=501))
u_fhn = hcat([sol_fhn(t) for t in t_fhn]...)

N_fhn = nv(g_directed_fhn)
adj_fhn = Matrix(Float64.(adjacency_matrix(g_directed_fhn)))

h5open(joinpath(outdir, "fitzhugh_nagumo_reference.h5"), "w") do f
    f["t"] = t_fhn
    f["u"] = u_fhn  # (2*N_nodes, T) — interleaved u,v states
    f["x0"] = x0_fhn
    f["edge_weights"] = edge_weights_fhn
    f["adjacency"] = adj_fhn
    f["G"] = G_fhn
    attrs(f)["N"] = N_fhn
    attrs(f)["n_sv"] = 2
    attrs(f)["a"] = 0.5
    attrs(f)["epsilon"] = 0.05
    attrs(f)["sigma"] = 0.5
    attrs(f)["duration"] = 200.0
    attrs(f)["solver"] = "AutoTsit5(TRBDF2)"
    attrs(f)["description"] = "FitzHugh-Nagumo on AAL-90 directed weighted brain network"
end
println("  Saved: fitzhugh_nagumo_reference.h5  ($(length(t_fhn)) timesteps, $N_fhn nodes, 2 state vars)")


# =============================================================================
# Example 4: 2D Network Diffusion (Getting Started Part 2)
# =============================================================================
println("=== Generating: diffusion_2d ===")

# Reuse the same diffusion functions — they broadcast over dimensions
N_diff2 = 10
k_diff2 = 4
g_diff2 = barabasi_albert(N_diff2, k_diff2; seed=1)
nd_diffusion_vertex_2 = VertexModel(; f=diffusionvertex_f!, g=1:2, dim=2, sym=[:x, :ϕ])
nd_diffusion_edge_2 = EdgeModel(; g=AntiSymmetric(diffusionedge_g!), outsym=[:flow_x, :flow_ϕ])
nd_2 = Network(g_diff2, nd_diffusion_vertex_2, nd_diffusion_edge_2)

rng_diff2 = StableRNG(1)
x0_2 = vec(transpose([randn(rng_diff2, N_diff2) .^ 2 randn(rng_diff2, N_diff2)]))

ode_prob_2 = ODEProblem(nd_2, x0_2, (0.0, 3.0))
sol_2 = solve(ode_prob_2, Tsit5())

t_diff2 = collect(range(0.0, 3.0, length=301))
u_diff2 = hcat([sol_2(t) for t in t_diff2]...)  # (2*N, T) interleaved

adj_diff2 = Matrix(Float64.(adjacency_matrix(g_diff2)))

h5open(joinpath(outdir, "diffusion_2d_reference.h5"), "w") do f
    f["t"] = t_diff2
    f["u"] = u_diff2
    f["x0"] = x0_2
    f["adjacency"] = adj_diff2
    attrs(f)["N"] = N_diff2
    attrs(f)["k"] = k_diff2
    attrs(f)["n_sv"] = 2
    attrs(f)["duration"] = 3.0
    attrs(f)["solver"] = "Tsit5"
    attrs(f)["description"] = "2D diffusion on Barabasi-Albert graph (StableRNG seed=1)"
end
println("  Saved: diffusion_2d_reference.h5  ($(length(t_diff2)) timesteps, $N_diff2 nodes, 2 state vars)")


# =============================================================================
# Example 5: Heterogeneous Kuramoto (3 vertex types)
# =============================================================================
println("=== Generating: heterogeneous_kuramoto ===")

N_het = 8
g_het = watts_strogatz(N_het, 2, 0)

# Same Kuramoto edge coupling
function het_kuramoto_edge!(e, θ_s, θ_d, (K,), t)
    e[1] = K * sin(θ_s[1] - θ_d[1])
    nothing
end
edge_het = EdgeModel(g=AntiSymmetric(het_kuramoto_edge!), outdim=1, psym=[:K=>3])

# Vertex 1: Standard Kuramoto
function het_kuramoto_vertex!(dθ, θ, esum, (ω0,), t)
    dθ[1] = ω0 + esum[1]
    nothing
end
vertex_het = VertexModel(f=het_kuramoto_vertex!, g=StateMask(1:1), sym=[:θ], psym=[:ω0], name=:kuramoto)

# Vertex 2: Static (no dynamics, outputs fixed θ)
function static_g(out, u, p, t)
    out[1] = p[1]
    nothing
end
static_het = VertexModel(g=static_g, outsym=[:θ], psym=[:θfix], ff=NoFeedForward(), name=:static)

# Vertex 3: Kuramoto with inertia (2D)
function kuramoto_inertia!(dv, v, esum, (ω0,), t)
    dv[1] = v[2]
    dv[2] = ω0 - 1.0 * v[2] + esum[1]
    nothing
end
inertia_het = VertexModel(f=kuramoto_inertia!, g=1:1, sym=[:θ, :ω], psym=[:ω0], name=:inertia)

# Build heterogeneous vertex array
vertex_array = VertexModel[vertex_het for _ in 1:N_het]
vertex_array[1] = static_het    # node 1 → static
vertex_array[5] = inertia_het   # node 5 → inertia

nw_het = Network(g_het, vertex_array, edge_het; dealias=true)

# Parameters: ω_i = i/N - mean(1:N/N)
ω_het = collect(1:N_het) ./ N_het
ω_het .-= sum(ω_het) / N_het

# Set via NWState
state_het = NWState(nw_het)
# Node 1: static, set θfix parameter
state_het.p.v[1, :θfix] = ω_het[1]
# Nodes 2-8: set ω0 parameter
for i in 2:N_het
    state_het.p.v[i, :ω0] = ω_het[i]
end

# Initial conditions
x0_het = collect(1:N_het) ./ N_het
x0_het .-= sum(x0_het) ./ N_het
# Set θ for nodes 2-8 (node 1 has no state)
for i in 2:N_het
    state_het.v[i, :θ] = x0_het[i]
end
# Node 5 inertia: initial angular velocity
state_het.v[5, :ω] = 5.0

prob_het = ODEProblem(nw_het, uflat(state_het), (0.0, 10.0), pflat(state_het))
sol_het = solve(prob_het, Tsit5())

t_het = collect(range(0.0, 10.0, length=201))
u_het = hcat([sol_het(t) for t in t_het]...)

adj_het = Matrix(Float64.(adjacency_matrix(g_het)))

# Record which vertex type each node has
vertex_types = ["kuramoto" for _ in 1:N_het]
vertex_types[1] = "static"
vertex_types[5] = "inertia"

h5open(joinpath(outdir, "heterogeneous_kuramoto_reference.h5"), "w") do f
    f["t"] = t_het
    f["u"] = u_het
    f["x0"] = uflat(state_het)
    f["omega0"] = ω_het
    f["adjacency"] = adj_het
    f["vertex_types"] = vertex_types
    attrs(f)["N"] = N_het
    attrs(f)["K"] = 3.0
    attrs(f)["n_sv_total"] = size(u_het, 1)  # total state dimension (not uniform per node)
    attrs(f)["duration"] = 10.0
    attrs(f)["solver"] = "Tsit5"
    attrs(f)["description"] = "Heterogeneous Kuramoto: 6 standard + 1 static + 1 inertia on ring"
end
println("  Saved: heterogeneous_kuramoto_reference.h5  ($(length(t_het)) timesteps, $N_het nodes, 3 vertex types)")


println("\n=== All reference data generated successfully! ===")
