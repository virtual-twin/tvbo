<%!
from tvbo.utils import initial_value as _initial_value
%>\
## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl full experiment template.

Generates a self-contained Julia script:
  1. Package imports
  2. VertexModel(s) (node dynamics) via sub-template — one per unique dynamics
  3. EdgeModel(s) (coupling) via sub-template — one per unique coupling
  4. Graph construction from experiment.network
  5. Network + ODEProblem / SDEProblem + solve
  6. Optional plot

Supports:
  - Homogeneous networks: single VertexModel + EdgeModel for all nodes
  - Heterogeneous networks: multiple VertexModels assigned per-node via vertex_array
  - Multi-dimensional coupling: outdim > 1 for broadcasting edge functions
  - Static vertices: dynamics with no state_variables

Context: Pre-computed dict from BaseAdapter.prepare_context()
</%doc>
## Template args: all pre-computed by BaseAdapter.prepare_context()
<%page args="experiment, model, network, integration, \
dynamics_dict, node_dynamics_map, all_couplings, coupling, \
coupling_vars, outdim, outsym_names, \
n_nodes, nodes, graph_gen, has_graph_generator, edges_list, emf_names, \
has_edge_matrix, has_explicit_edges, is_directed, \
sv_names, n_sv, is_heterogeneous, is_stochastic, \
dt, duration, solver_method, fixed_step, needs_stiff, needs_weighted, \
weight_matrix, weight_sym, \
dist_info, needs_random, dist_seed, \
all_events, has_events, coupling_observed, find_fixpoint, \
is_static, parse_node_parameters, get_noise_sigmas, graph_generator_call"/>
<%!
from tvbo.adapters.julia_model import (
    julia_ode_package, needs_nanmath, needs_special_functions,
)
%>

## ── Packages ────────────────────────────────────────────────────────────────
using Graphs
using NetworkDynamics
% if is_stochastic:
using StochasticDiffEq
% else:
using ${julia_ode_package(solver_method)}
% endif
% if needs_stiff:
using OrdinaryDiffEqSDIRK
% endif
% if needs_weighted:
using DelimitedFiles
using SimpleWeightedGraphs
% endif
<%
# Optional Julia packages, gated per dynamics (shared detection with the other backends):
#  - SpecialFunctions for erf/erfc/…; NaNMath for domain-restricted powers in Piecewise.
_needs_special = any(needs_special_functions(dyn) for dyn in dynamics_dict.values())
_needs_nanmath = any(needs_nanmath(dyn) for dyn in dynamics_dict.values())
%>
% if _needs_special:
using SpecialFunctions
% endif
% if _needs_nanmath:
import NaNMath
% endif

## ── Vertex models (node dynamics) ───────────────────────────────────────────
% for dyn_name, dyn in dynamics_dict.items():
% if is_static(dyn):
## Static vertex: no dynamics, outputs parameter values
<%
    static_params = list((dyn.parameters or {}).keys())
    # Output symbols: use coupling vars from default model
    # (static vertex must output the same symbols that edges expect)
    static_outsyms = coupling_vars if coupling_vars else ['out']
    n_static_out = len(static_outsyms)
%>
function ${dyn.name}_g!(out, x, p, t)
    out .= p
    nothing
end
vertex_${dyn.name} = VertexModel(;
    g = ${dyn.name}_g!,
    outsym = [${", ".join(f':{s}' for s in static_outsyms)}],
% if static_params:
    psym = [${", ".join(f':{p}' for p in static_params)}],
% endif
    ff = NoFeedForward(),
    name = :${dyn.name},
)

% else:
<%include file="/tvbo-nd-vertex.jl.mako" args="model=dyn, all_couplings=all_couplings, outdim=outdim" />

% endif
% endfor

## ── Edge models (coupling) ──────────────────────────────────────────────────
% for c_name, c in all_couplings.items():
<%include file="/tvbo-nd-edge.jl.mako" args="coupling=c, is_directed=is_directed, outdim=outdim, outsym_names=outsym_names" />

% endfor

## ── Graph ───────────────────────────────────────────────────────────────────
% if has_edge_matrix:
G = readdlm("${emf_names[0]}", ',', Float64, '\n')
g_weighted = SimpleWeightedDiGraph(G)
edge_weights = getfield.(collect(edges(g_weighted)), :weight)
g = SimpleDiGraph(g_weighted)
% elif has_graph_generator:
g = ${graph_generator_call(graph_gen, n_nodes, 'julia')}
% elif weight_matrix is not None:
using SimpleWeightedGraphs
<%
import numpy as np
W = weight_matrix
%>
W = [${'; '.join(' '.join(f'{W[i,j]:.6g}' for j in range(n_nodes)) for i in range(n_nodes))}]
g = SimpleDiGraph(SimpleWeightedDiGraph(W))
% elif has_explicit_edges:
% if is_directed:
g = SimpleDiGraph(${n_nodes})
% else:
g = SimpleGraph(${n_nodes})
% endif
    % for e in edges_list:
add_edge!(g, ${e.source + 1}, ${e.target + 1})
    % endfor
% else:
% if n_nodes == 1:
## Single-node: self-loop with zero edge to avoid coupling feedback
g = SimpleDiGraph(1)
add_edge!(g, 1, 1)
## Zero-weight edge: outputs 0 so esum has no effect on single-node dynamics
function _zero_edge_g!(e_dst, v_src, v_dst, p, t)
% for _i in range(outdim):
    e_dst[${_i + 1}] = 0.0
% endfor
    nothing
end
<%
_zero_outsym = outsym_names[:outdim] if outsym_names else ['coupling']
%>\
edge_zero = EdgeModel(;
    g = Directed(_zero_edge_g!),
    outsym = [${", ".join(f':{s}' for s in _zero_outsym)}],
    name = :zero_coupling,
)
% else:
g = complete_graph(${n_nodes})
% endif
% endif

## ── Network ─────────────────────────────────────────────────────────────────
<%
default_coupling_name = next(iter(all_couplings.keys())) if all_couplings else (coupling.name if coupling else 'Diffusion')
has_weight_param = weight_sym is not None
needs_dealias = is_heterogeneous or has_events
# For single-node self-loop, use the zero edge to avoid coupling feedback
edge_var = 'edge_zero' if n_nodes == 1 else f'edge_{default_coupling_name}'
%>
% if is_heterogeneous:
## Heterogeneous network: different vertex models per node
vertex_array = VertexModel[vertex_${model.name} for _ in 1:${n_nodes}]
% for node in nodes:
% if node_dynamics_map.get(node.id, model.name) != model.name:
vertex_array[${node.id + 1}] = vertex_${node_dynamics_map[node.id]}
% endif
% endfor

nw = Network(g, vertex_array, ${edge_var}; dealias=true)
% elif needs_dealias:
nw = Network(g, vertex_${model.name}, ${edge_var}; dealias=true)
% else:
nw = Network(g, vertex_${model.name}, ${edge_var})
% endif
% if has_edge_matrix and has_weight_param:

## Set per-edge weights from connectivity matrix
p = NWParameter(nw)
p.e[1:ne(g), :${weight_sym}] = edge_weights
% endif

## ── Per-node defaults (set on network before NWState) ───────────────────────
% if find_fixpoint and is_heterogeneous:
## Set per-node parameter defaults for find_fixpoint (heterogeneous)
% for node in nodes:
<%
    dyn_name = node_dynamics_map.get(node.id, model.name)
    dyn = dynamics_dict[dyn_name]
    node_idx = node.id + 1
    node_params = parse_node_parameters(node)
%>
% for p_name, p_val in node_params.items():
set_default!(nw, VIndex(${node_idx}, :${p_name}), ${p_val})
% endfor
% endfor
% elif find_fixpoint and not is_heterogeneous:
## Set per-node parameter defaults for find_fixpoint (homogeneous)
% for node in nodes:
<%
    node_idx = node.id + 1
    node_params = parse_node_parameters(node)
%>
% for p_name, p_val in node_params.items():
set_default!(nw, VIndex(${node_idx}, :${p_name}), ${p_val})
% endfor
% endfor
% endif

## ── Find fixpoint (steady-state initial conditions) ─────────────────────────
% if find_fixpoint:

## Use ND.jl find_fixpoint for steady-state initial conditions
u0 = find_fixpoint(nw)
set_defaults!(nw, u0)
% endif

## ── Initial state ───────────────────────────────────────────────────────────
<%
from tvbo.templates.base.utils import sample_expression, collect_param_distributions
%>
s = NWState(nw)
% if needs_random:
using Random
rng = MersenneTwister(${dist_seed})
% endif
% if is_heterogeneous:
## Heterogeneous initial conditions: set per-node based on dynamics type
% for node in nodes:
<%
    dyn_name = node_dynamics_map.get(node.id, model.name)
    dyn = dynamics_dict[dyn_name]
    node_idx = node.id + 1
    node_state = getattr(node, 'state', None) or []
    state_items = node_state.values() if isinstance(node_state, dict) else node_state
    state_map = {}
    for state_entry in state_items:
        if isinstance(state_entry, dict):
            state_name = state_entry.get('name')
            state_value = state_entry.get('value')
        else:
            state_name = getattr(state_entry, 'name', None)
            state_value = getattr(state_entry, 'value', None)
        if state_name is not None and state_value is not None:
            state_map[str(state_name)] = state_value
    node_init = [state_map.get(sv_name, None) for sv_name in (dyn.state_variables or {}).keys()]
    if not any(v is not None for v in node_init):
        node_init = getattr(node, 'initial_state', None) or []
    node_params = parse_node_parameters(node)
%>
% for i, sv in enumerate((dyn.state_variables or {}).values()):
<%
    d = getattr(sv, 'distribution', None)
    has_dist = d and getattr(d, 'domain', None)
    init_val = node_init[i] if i < len(node_init) else None
%>
% if init_val is not None:
s.v[${node_idx}, :${sv.name}] = ${init_val}
% elif has_dist:
s.v[${node_idx}, :${sv.name}] = ${sample_expression(d, 'julia')}
% else:
s.v[${node_idx}, :${sv.name}] = ${_initial_value(sv)}
% endif
% endfor
% if not find_fixpoint:
% for p_name in list((dyn.parameters or {}).keys()):
<%
    p_val = node_params.get(p_name, None)
%>
% if p_val is not None:
s.p.v[${node_idx}, :${p_name}] = ${p_val}
% endif
% endfor
% for p_name, p_obj, d in dist_info.get(dyn_name, {}).get('param', []):
% if p_name not in node_params:
s.p.v[${node_idx}, :${p_name}] = ${sample_expression(d, 'julia')}
% endif
% endfor
% endif
% endfor
% elif not find_fixpoint:
## Homogeneous initial conditions
% for i, sv in enumerate(model.state_variables.values()):
<%
    d = getattr(sv, 'distribution', None)
    has_dist = d and getattr(d, 'domain', None)
%>
% if has_dist:
for node in 1:nv(g)
    s.v[node, :${sv.name}] = ${sample_expression(d, 'julia')}
end
% else:
s.v[1:nv(g), :${sv.name}] .= ${_initial_value(sv)}
% endif
% endfor
% for p_name, p_obj, d in collect_param_distributions(model):
for node in 1:nv(g)
    s.p.v[node, :${p_name}] = ${sample_expression(d, 'julia')}
end
% endfor
% endif
<%
# Collect per-edge parameter overrides (skip 'weight' — handled by connectivity matrix)
# IMPORTANT: SimpleGraph stores edges sorted by (min, max), so s.p.e[i, :param]
# corresponds to the i-th edge in sorted order, not YAML insertion order.
# We must sort edges the same way to assign parameters to the correct edges.
edge_param_lines = []

# Build list of (sorted_key, original_edge) to match SimpleGraph ordering
edges_with_keys = []
for edge in edges_list:
    s_node = min(edge.source, edge.target) + 1
    t_node = max(edge.source, edge.target) + 1
    edges_with_keys.append(((s_node, t_node), edge))

# Sort by (min_node, max_node) — same order as SimpleGraph edges()
edges_with_keys.sort(key=lambda x: x[0])

for i, (key, edge) in enumerate(edges_with_keys):
    eparams = getattr(edge, 'parameters', None) or {}
    if isinstance(eparams, dict):
        for p_name, p_obj in eparams.items():
            if p_name == 'weight':
                continue
            val = getattr(p_obj, 'value', None) if hasattr(p_obj, 'value') else p_obj
            if val is not None:
                edge_param_lines.append(f's.p.e[{i + 1}, :{p_name}] = {val}')
    else:
        for p in eparams:
            ep = parse_node_parameters({'parameters': [p]}) if not isinstance(p, dict) else p
            for p_name, p_val in (ep if isinstance(ep, dict) else {}).items():
                if p_name == 'weight' or p_val is None:
                    continue
                edge_param_lines.append(f's.p.e[{i + 1}, :{p_name}] = {p_val}')
%>
% if edge_param_lines:

## Per-edge parameter overrides
% for line in edge_param_lines:
${line}
% endfor
% endif

## ── Events / Callbacks ──────────────────────────────────────────────────────
% if has_events:
<%
    # Separate events by type
    continuous_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'continuous']
    preset_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'preset_time']
    discrete_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'discrete']
%>
% for ev, ev_src in continuous_events:
<%
    cond_syms = ', '.join(f':{s}' for s in (ev.condition_states or []))
    cond_psyms = ', '.join(f':{p}' for p in (ev.condition_parameters or []))
    affect_syms = '[]'
    affect_psyms = ', '.join(f':{p}' for p in (ev.affect_parameters or []))
%>

## Continuous callback: ${ev.name}
${ev.name}_cond = ComponentCondition([${cond_syms}], [${cond_psyms}]) do u, p, t
    ${ev.condition.rhs}
end
${ev.name}_affect = ComponentAffect([], [${affect_psyms}]) do u, p, ctx
    ${ev.affect.rhs}
end
${ev.name}_cb = ContinuousComponentCallback(${ev.name}_cond, ${ev.name}_affect)
% if ev.target_component == 'all_edges':
for i in 1:ne(g)
    set_callback!(nw[EIndex(i)], ${ev.name}_cb)
end
% elif ev.target_component and ev.target_component.startswith('edge_'):
<%
    edge_idx = int(ev.target_component.split('_')[1])
%>
set_callback!(nw[EIndex(${edge_idx})], ${ev.name}_cb)
% endif
% endfor
% for ev, ev_src in preset_events:
<%
    affect_psyms = ', '.join(f':{p}' for p in (ev.affect_parameters or []))
    times = ', '.join(str(t) for t in ev.trigger_times) if ev.trigger_times else '0.0'
%>

## Preset-time callback: ${ev.name}
<%
    # Can reuse existing affect if same affect body
    # Check if an identical affect was already defined
    reuse_affect = None
    for prev_ev, _ in continuous_events:
        if str(prev_ev.affect.rhs) == str(ev.affect.rhs) and \
           list(prev_ev.affect_parameters or []) == list(ev.affect_parameters or []):
            reuse_affect = prev_ev.name + '_affect'
            break
%>
% if reuse_affect:
${ev.name}_cb = PresetTimeComponentCallback(${times}, ${reuse_affect})
% else:
${ev.name}_affect = ComponentAffect([], [${affect_psyms}]) do u, p, ctx
    ${ev.affect.rhs}
end
${ev.name}_cb = PresetTimeComponentCallback(${times}, ${ev.name}_affect)
% endif
% if ev.target_component and ev.target_component.startswith('edge_'):
<%
    edge_idx = int(ev.target_component.split('_')[1])
%>
add_callback!(nw[EIndex(${edge_idx})], ${ev.name}_cb)
% elif ev.target_component == 'all_edges':
for i in 1:ne(g)
    add_callback!(nw[EIndex(i)], ${ev.name}_cb)
end
% endif
% endfor
% endif

## ── Problem + solve ─────────────────────────────────────────────────────────
tspan = (0.0, ${duration})
% if is_stochastic:
<%
sigma_vals = get_noise_sigmas(model)
%>

function nw_noise!(du, u, p, t)
    sigma = [${", ".join(str(s) for s in sigma_vals)}]
    for node in 1:${n_nodes}
        for i in eachindex(sigma)
            du[(node - 1) * ${n_sv} + i] = sigma[i]
        end
    end
    nothing
end

prob = SDEProblem(nw, nw_noise!, uflat(s), tspan, pflat(s))
sol = solve(prob, EulerHeun(); dt=${dt}, saveat=${dt})
% elif find_fixpoint:
## ODEProblem from NWState: auto-extracts initial state, parameters, and callbacks
u0 = NWState(nw)
prob = ODEProblem(nw, u0, tspan)
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); ${'dt=%s, ' % dt if fixed_step else ''}saveat=${dt})
% elif has_edge_matrix and has_weight_param:
prob = ODEProblem(nw, uflat(s), tspan, pflat(p))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); ${'dt=%s, ' % dt if fixed_step else ''}saveat=${dt})
% else:
prob = ODEProblem(nw, uflat(s), tspan, pflat(s))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); ${'dt=%s, ' % dt if fixed_step else ''}saveat=${dt})
% endif

## ── Graph data (extracted by Python adapter) ───────────────────────────────
adj_matrix = Float64.(adjacency_matrix(g))

## Spring layout for visualization (deterministic seed)
using Random: MersenneTwister
function spring_layout(g; seed=42, iterations=50, k=1.0)
    rng = MersenneTwister(seed)
    n = nv(g)
    pos = randn(rng, n, 2)
    for _ in 1:iterations
        disp = zeros(n, 2)
        for i in 1:n, j in (i+1):n
            d = pos[i, :] - pos[j, :]
            dist = max(norm(d), 0.01)
            rep = k^2 / dist
            disp[i, :] .+= d / dist * rep
            disp[j, :] .-= d / dist * rep
        end
        for e in edges(g)
            i, j = src(e), dst(e)
            d = pos[j, :] - pos[i, :]
            dist = max(norm(d), 0.01)
            att = dist^2 / k
            disp[i, :] .+= d / dist * att
            disp[j, :] .-= d / dist * att
        end
        for i in 1:n
            dl = max(norm(disp[i, :]), 0.01)
            pos[i, :] .+= disp[i, :] / dl * min(dl, 0.1)
        end
    end
    return pos
end
using LinearAlgebra: norm
node_positions = spring_layout(g)

## ── Plot ────────────────────────────────────────────────────────────────────
using Plots
plot(sol; ylabel="state", xlabel="time", title="${model.name} on ${n_nodes}-node network")
