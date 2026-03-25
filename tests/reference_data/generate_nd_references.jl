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
            "HDF5", "DelimitedFiles", "DiffEqCallbacks"]
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



# =============================================================================
# Example 6: Cascading Failure (Component Callbacks)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/cascading_failure/
# =============================================================================
println("=== Generating: cascading_failure ===")

using DiffEqCallbacks

function swing_equation(dv, v, esum, p, t)
    P, I, γ = p
    dv[1] = v[2]
    dv[2] = P - γ * v[2] + esum[1]
    dv[2] = dv[2] / I
    nothing
end
vertex_cf = VertexModel(f=swing_equation, g=1, sym=[:δ, :ω], psym=[:P_ref, :I=>1, :γ=>0.1])

function simple_edge(e, v_s, v_d, (K,), t)
    e[1] = K * sin(v_s[1] - v_d[1])
end
edge_cf = EdgeModel(; g=AntiSymmetric(simple_edge), outsym=[:P], psym=[:K=>1.63, :limit=>1])

g_cf = SimpleGraph([0 1 1 0 1;
                     1 0 1 1 0;
                     1 1 0 1 0;
                     0 1 1 0 1;
                     1 0 0 1 0])
nw_cf = Network(g_cf, vertex_cf, edge_cf; dealias=true)

set_default!(nw_cf, VIndex(1, :P_ref), -1.0)
set_default!(nw_cf, VIndex(2, :P_ref),  1.5)
set_default!(nw_cf, VIndex(3, :P_ref), -1.0)
set_default!(nw_cf, VIndex(4, :P_ref), -1.0)
set_default!(nw_cf, VIndex(5, :P_ref),  1.5)

u0_cf = find_fixpoint(nw_cf)
set_defaults!(nw_cf, u0_cf)

# Component callbacks
cond_cf = ComponentCondition([:P], [:limit]) do u, p, t
    abs(u[:P]) - p[:limit]
end
affect_cf = ComponentAffect([], [:K]) do u, p, ctx
    p[:K] = 0
end
edge_cb_cf = ContinuousComponentCallback(cond_cf, affect_cf)
for i in 1:ne(g_cf)
    set_callback!(nw_cf[EIndex(i)], edge_cb_cf)
end
trip_first_cf = PresetTimeComponentCallback(1.0, affect_cf)
add_callback!(nw_cf[EIndex(5)], trip_first_cf)

s_cf = NWState(nw_cf)
prob_cf = ODEProblem(nw_cf, s_cf, (0.0, 6.0))
sol_cf = solve(prob_cf, Tsit5(); saveat=0.01)

N_cf = nv(g_cf)
h5open(joinpath(outdir, "cascading_failure_reference.h5"), "w") do f
    f["t"] = Array(sol_cf.t); f["u"] = Array(sol_cf)
    f["x0"] = uflat(s_cf)
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_cf)))
    attrs(f)["N"] = N_cf; attrs(f)["n_sv"] = 2
    attrs(f)["dt"] = 0.01; attrs(f)["duration"] = 6.0
end
println("  ✓ cascading_failure_reference.h5")


# =============================================================================
# Example 7: Stress on Truss (Heterogeneous 2D, observed functions)
# https://juliadynamics.github.io/NetworkDynamics.jl/stable/generated/stress_on_truss/
# =============================================================================
println("=== Generating: stress_on_truss ===")

using LinearAlgebra: norm

function fixed_g(pos, x, p, t)
    pos .= p
end
vertex_fix = VertexModel(g=fixed_g, psym=[:xfix, :yfix], outsym=[:x, :y], ff=NoFeedForward())

function free_f(dx, x, Fsum, (M, γ, g), t)
    v = view(x, 1:2)
    dx[1:2] .= (Fsum .- γ .* v) ./ M
    dx[2] -= g
    dx[3:4] .= v
    nothing
end
vertex_free = VertexModel(f=free_f, g=3:4, sym=[:vx=>0, :vy=>0, :x, :y],
                          psym=[:M=>10, :γ=>200, :g=>9.81], insym=[:Fx, :Fy])

function edge_g!(F, pos_src, pos_dst, (K, L), t)
    dx = pos_dst[1] - pos_src[1]
    dy = pos_dst[2] - pos_src[2]
    d = sqrt(dx^2 + dy^2)
    Fabs = K * (L - d)
    F[1] = Fabs * dx / d
    F[2] = Fabs * dy / d
    nothing
end
function observedf(obsout, u, pos_src, pos_dst, (K, L), t)
    dx = pos_dst[1] - pos_src[1]
    dy = pos_dst[2] - pos_src[2]
    d = sqrt(dx^2 + dy^2)
    obsout[1] = K * (L - d)
    nothing
end
beam = EdgeModel(g=AntiSymmetric(edge_g!), psym=[:K=>0.5e6, :L], outsym=[:Fx, :Fy],
                 obsf=observedf, obssym=[:Fabs])

N_tr = 5
dx_tr = 1.0
shift_tr = 0.2
g_tr = SimpleGraph(2*N_tr + 1)
for i in 1:N_tr
    add_edge!(g_tr, i, i+N_tr)
    if i < N_tr
        add_edge!(g_tr, i+1, i+N_tr); add_edge!(g_tr, i, i+1); add_edge!(g_tr, i+N_tr, i+N_tr+1)
    end
end
add_edge!(g_tr, 2*N_tr, 2*N_tr+1)

pos0_tr = zeros(Float64, 2*N_tr+1, 2)
for i in 1:N_tr
    pos0_tr[i, 1] = (i-1)*dx_tr;  pos0_tr[i, 2] = 0.0
    pos0_tr[i+N_tr, 1] = i*dx_tr + shift_tr;  pos0_tr[i+N_tr, 2] = 1.0
end
pos0_tr[2*N_tr+1, 1] = N_tr*dx_tr + 1.0;  pos0_tr[2*N_tr+1, 2] = -1.0

fixed_tr = [1, 4]

verts_tr = VertexModel[vertex_free for _ in 1:nv(g_tr)]
for i in fixed_tr
    verts_tr[i] = vertex_fix
end
nw_tr = Network(g_tr, verts_tr, beam)

s_tr = NWState(nw_tr)
for i in 1:nv(g_tr)
    if i in fixed_tr
        s_tr.p.v[i, :xfix] = pos0_tr[i, 1]
        s_tr.p.v[i, :yfix] = pos0_tr[i, 2]
    else
        s_tr.v[i, :x] = pos0_tr[i, 1]
        s_tr.v[i, :y] = pos0_tr[i, 2]
    end
end
for (i, e) in enumerate(edges(g_tr))
    s_tr.p.e[i, :L] = norm(pos0_tr[src(e), :] - pos0_tr[dst(e), :])
end

s_tr.p.v[11, :M] = 200
s_tr.p.v[11, :γ] = 100

prob_tr = ODEProblem(nw_tr, s_tr, (0.0, 12.0))
sol_tr = solve(prob_tr, Tsit5(); saveat=0.01)

h5open(joinpath(outdir, "stress_on_truss_reference.h5"), "w") do f
    f["t"] = Array(sol_tr.t); f["u"] = Array(sol_tr)
    f["x0"] = uflat(s_tr)
    f["adjacency"] = Matrix(Float64.(adjacency_matrix(g_tr)))
    attrs(f)["N"] = nv(g_tr); attrs(f)["n_sv_free"] = 4; attrs(f)["n_fixed"] = length(fixed_tr)
    attrs(f)["dt"] = 0.01; attrs(f)["duration"] = 12.0
end
println("  ✓ stress_on_truss_reference.h5")


println("\n=== All 7 references generated ===")
