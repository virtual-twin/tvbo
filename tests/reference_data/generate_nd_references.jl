#!/usr/bin/env julia
"""
Generate HDF5 reference data from original NetworkDynamics.jl tutorials.

Run this script once to create reference .h5 files that pytest uses for
numerical comparison against TVBO-generated results.

Usage:
    julia --project=@. generate_nd_references.jl

Time grids use solve(...; saveat=dt) matching each YAML's step_size so
that TVBO-generated code produces time-series on the exact same grid.
"""

using Pkg
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
# Example 1: Network Diffusion (Getting Started — 1D part)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/getting_started_with_network_dynamics/
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

N_diff = 20; k_diff = 4
g_diff = barabasi_albert(N_diff, k_diff; seed=1)
nd_diffusion_vertex = VertexModel(; f=diffusionvertex_f!, g=StateMask(1:1), dim=1)
nd_diffusion_edge = EdgeModel(; g=AntiSymmetric(diffusionedge_g!), outsym=[:flow])
nd_diff = Network(g_diff, nd_diffusion_vertex, nd_diffusion_edge)

x0_diff = randn(StableRNG(1), N_diff)
sol_diff = solve(ODEProblem(nd_diff, x0_diff, (0.0, 2.0)), Tsit5(); saveat=0.01)

h5open(joinpath(outdir, "diffusion_reference.h5"), "w") do f
    f["t"] = Array(sol_diff.t)
    f["u"] = Array(sol_diff)
    f["x0"] = x0_diff
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_diff)))
    attrs(f)["N"] = N_diff; attrs(f)["n_sv"] = 1
    attrs(f)["dt"] = 0.01;  attrs(f)["duration"] = 2.0
end
println("  ✓ diffusion_reference.h5")


# =============================================================================
# Example 2: Kuramoto (Heterogeneous System — homogeneous part)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/heterogeneous_system/
# =============================================================================
println("=== Generating: kuramoto ===")

N_kur = 8
g_kur = watts_strogatz(N_kur, 2, 0)

function kuramoto_edge!(e, θ_s, θ_d, (K,), t)
    e[1] = K * sin(θ_s[1] - θ_d[1]); nothing
end
function kuramoto_vertex!(dθ, θ, esum, (ω0,), t)
    dθ[1] = ω0 + esum[1]; nothing
end

edge_kur = EdgeModel(g=AntiSymmetric(kuramoto_edge!), outdim=1, psym=[:K=>3])
vertex_kur = VertexModel(f=kuramoto_vertex!, g=StateMask(1:1), sym=[:θ], psym=[:ω0], name=:kuramoto)
nw_kur = Network(g_kur, vertex_kur, edge_kur)

p_kur = NWParameter(nw_kur)
ω_kur = collect(1:N_kur) ./ N_kur; ω_kur .-= sum(ω_kur) / N_kur
p_kur.v[:, :ω0] = ω_kur

x0_kur = collect(1:N_kur) ./ N_kur; x0_kur .-= sum(x0_kur) ./ N_kur
sol_kur = solve(ODEProblem(nw_kur, x0_kur, (0.0, 10.0), pflat(p_kur)), Tsit5(); saveat=0.05)

h5open(joinpath(outdir, "kuramoto_reference.h5"), "w") do f
    f["t"] = Array(sol_kur.t); f["u"] = Array(sol_kur)
    f["x0"] = x0_kur; f["omega0"] = ω_kur
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_kur)))
    attrs(f)["N"] = N_kur; attrs(f)["K"] = 3.0; attrs(f)["n_sv"] = 1
    attrs(f)["dt"] = 0.05; attrs(f)["duration"] = 10.0
end
println("  ✓ kuramoto_reference.h5")


# =============================================================================
# Example 3: FitzHugh-Nagumo (Directed & Weighted Graphs)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/directed_and_weighted_graphs/
# =============================================================================
println("=== Generating: fitzhugh_nagumo ===")

dti_file = joinpath(@__DIR__, "..", "..", "docs", "Interoperability",
                    "NetworkDynamics.jl", "Norm_G_DTI.txt")
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
    e[1] = w * (v_s[1] - v_d[1]) * σ; nothing
end
electricaledge_fhn = EdgeModel(g=Directed(electrical_edge!), outdim=1, psym=[:weight, :σ=>0.5])

fhn_network = Network(g_directed_fhn, vertex_fhn, electricaledge_fhn)
p_fhn = NWParameter(fhn_network)
p_fhn.e[1:ne(g_directed_fhn), :weight] = edge_weights_fhn

x0_fhn = randn(StableRNG(42), dim(fhn_network)) * 5
sol_fhn = solve(ODEProblem(fhn_network, x0_fhn, (0.0, 200.0), pflat(p_fhn)),
                AutoTsit5(TRBDF2()); saveat=0.1)

N_fhn = nv(g_directed_fhn)
h5open(joinpath(outdir, "fitzhugh_nagumo_reference.h5"), "w") do f
    f["t"] = Array(sol_fhn.t); f["u"] = Array(sol_fhn)
    f["x0"] = x0_fhn; f["edge_weights"] = edge_weights_fhn
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_directed_fhn)))
    f["G"] = G_fhn
    attrs(f)["N"] = N_fhn; attrs(f)["n_sv"] = 2
    attrs(f)["dt"] = 0.1; attrs(f)["duration"] = 200.0
end
println("  ✓ fitzhugh_nagumo_reference.h5")


# =============================================================================
# Example 4: 2D Network Diffusion (Getting Started — 2D extension)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/getting_started_with_network_dynamics/
# =============================================================================
println("=== Generating: diffusion_2d ===")

N_diff2 = 10; k_diff2 = 4
g_diff2 = barabasi_albert(N_diff2, k_diff2; seed=1)
nd_diffusion_vertex_2 = VertexModel(; f=diffusionvertex_f!, g=1:2, dim=2, sym=[:x, :ϕ])
nd_diffusion_edge_2 = EdgeModel(; g=AntiSymmetric(diffusionedge_g!), outsym=[:flow_x, :flow_ϕ])
nd_2 = Network(g_diff2, nd_diffusion_vertex_2, nd_diffusion_edge_2)

rng_diff2 = StableRNG(1)
x0_2 = collect(vec(transpose([randn(rng_diff2, N_diff2) .^ 2 randn(rng_diff2, N_diff2)])))
sol_2 = solve(ODEProblem(nd_2, x0_2, (0.0, 3.0)), Tsit5(); saveat=0.01)

h5open(joinpath(outdir, "diffusion_2d_reference.h5"), "w") do f
    f["t"] = Array(sol_2.t); f["u"] = Array(sol_2)
    f["x0"] = x0_2
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_diff2)))
    attrs(f)["N"] = N_diff2; attrs(f)["n_sv"] = 2
    attrs(f)["dt"] = 0.01; attrs(f)["duration"] = 3.0
end
println("  ✓ diffusion_2d_reference.h5")


# =============================================================================
# Example 5: Heterogeneous Kuramoto (3 vertex types — full tutorial)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/heterogeneous_system/
# =============================================================================
println("=== Generating: heterogeneous_kuramoto ===")

N_het = 8
g_het = watts_strogatz(N_het, 2, 0)

function het_kuramoto_edge!(e, θ_s, θ_d, (K,), t)
    e[1] = K * sin(θ_s[1] - θ_d[1]); nothing
end
edge_het = EdgeModel(g=AntiSymmetric(het_kuramoto_edge!), outdim=1, psym=[:K=>3])

function het_kuramoto_vertex!(dθ, θ, esum, (ω0,), t)
    dθ[1] = ω0 + esum[1]; nothing
end
vertex_het = VertexModel(f=het_kuramoto_vertex!, g=StateMask(1:1), sym=[:θ], psym=[:ω0], name=:kuramoto)

function static_g(out, u, p, t)
    out[1] = p[1]; nothing
end
static_het = VertexModel(g=static_g, outsym=[:θ], psym=[:θfix], ff=NoFeedForward(), name=:static)

function kuramoto_inertia!(dv, v, esum, (ω0,), t)
    dv[1] = v[2]
    dv[2] = ω0 - 1.0 * v[2] + esum[1]
    nothing
end
inertia_het = VertexModel(f=kuramoto_inertia!, g=1:1, sym=[:θ, :ω], psym=[:ω0], name=:inertia)

vertex_array = VertexModel[vertex_het for _ in 1:N_het]
vertex_array[1] = static_het
vertex_array[5] = inertia_het
nw_het = Network(g_het, vertex_array, edge_het; dealias=true)

ω_het = collect(1:N_het) ./ N_het; ω_het .-= sum(ω_het) / N_het

state_het = NWState(nw_het)
state_het.p.v[1, :θfix] = ω_het[1]
for i in 2:N_het; state_het.p.v[i, :ω0] = ω_het[i]; end

x0_het = collect(1:N_het) ./ N_het; x0_het .-= sum(x0_het) ./ N_het
for i in 2:N_het; state_het.v[i, :θ] = x0_het[i]; end
state_het.v[5, :ω] = 5.0

sol_het = solve(ODEProblem(nw_het, uflat(state_het), (0.0, 10.0), pflat(state_het)),
                Tsit5(); saveat=0.05)

h5open(joinpath(outdir, "heterogeneous_kuramoto_reference.h5"), "w") do f
    f["t"] = Array(sol_het.t); f["u"] = Array(sol_het)
    f["x0"] = uflat(state_het); f["omega0"] = ω_het
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_het)))
    f["vertex_types"] = ["kuramoto" for _ in 1:N_het]
    f["vertex_types"][1] = "static"; f["vertex_types"][5] = "inertia"
    attrs(f)["N"] = N_het; attrs(f)["n_sv_total"] = size(Array(sol_het), 1)
    attrs(f)["dt"] = 0.05; attrs(f)["duration"] = 10.0
end
println("  ✓ heterogeneous_kuramoto_reference.h5")


println("\n=== All 5 references generated ===")
