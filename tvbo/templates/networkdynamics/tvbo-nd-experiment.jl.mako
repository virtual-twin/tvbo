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

Every graph is a SimpleDiGraph with the Directed edge model, and every edge carries the connection weight, so node i receives post(sum_j w_ij * pre(x_i, x_j)), the coupling every other backend integrates.

Context: Pre-computed dict from NetworkDynamicsAdapter.prepare_context()
</%doc>
## Template args: all pre-computed by NetworkDynamicsAdapter.prepare_context()
<%page args="experiment, model, network, integration, \
dynamics_dict, node_dynamics_map, all_couplings, coupling, \
coupling_vars, outdim, outsym_names, \
n_nodes, nodes, graph_gen, weight_matrix, \
coupling_splits, edge_split, graph_form, graph_edges, edge_weights, edge_parameters, event_edges, edge_event_names, line_partners, node_rows, dealias, \
sv_names, n_sv, is_heterogeneous, is_stochastic, \
dt, duration, solver_method, solve_kwargs, needs_stiff, \
dist_info, needs_random, dist_seed, \
all_events, has_events, coupling_observed, find_fixpoint, \
is_static, parse_node_parameters, get_noise_sigmas, graph_generator_call"/>
<%!
from tvbo.adapters.julia_model import (
    equation_rhs_text, julia_ode_package, needs_nanmath, needs_special_functions,
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
<%
# SpecialFunctions for erf/erfc/…, NaNMath for domain-restricted powers in a Piecewise, each sniffed once per dynamics.
_rhs_texts = [equation_rhs_text(dyn) for dyn in dynamics_dict.values()]
_needs_special = any(needs_special_functions(text) for text in _rhs_texts)
_needs_nanmath = any(needs_nanmath(text) for text in _rhs_texts)
%>
% if _needs_special:
using SpecialFunctions
% endif
% if _needs_nanmath:
import NaNMath
% endif

<%namespace name="edge" file="/tvbo-nd-edge.jl.mako"/>\
% if any(split['post'] is not None for split in coupling_splits.values()):
## ── Coupling post-expressions, each applied by a vertex to its summed edge input ──
% for c_name in all_couplings:
${edge.post_function(coupling_splits[c_name])}\
% endfor

% endif
## ── Vertex models (node dynamics) ───────────────────────────────────────────
% for dyn_name, dyn in dynamics_dict.items():
% if is_static(dyn):
## Static vertex: no dynamics, outputs parameter values
<%
    static_params = list((dyn.parameters or {}).keys())
    # A static vertex outputs the coupling variables of the default model, the symbols the edges expect.
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
<%include file="/tvbo-nd-vertex.jl.mako" args="model=dyn, all_couplings=all_couplings, outdim=outdim, split=edge_split" />

% endif
% endfor

## ── Edge models (coupling) ──────────────────────────────────────────────────
% for c_name, c in all_couplings.items():
<%include file="/tvbo-nd-edge.jl.mako" args="coupling=c, split=coupling_splits[c_name], outdim=outdim, outsym_names=outsym_names" />

% endfor

## ── Graph ───────────────────────────────────────────────────────────────────
% if graph_form == "generator":
g = SimpleDiGraph(${graph_generator_call(graph_gen, n_nodes, 'julia')})
% elif graph_form == "matrix":
using SimpleWeightedGraphs
W = [${'; '.join(' '.join(f'{weight_matrix[i, j]:.6g}' for j in range(n_nodes)) for i in range(n_nodes))}]
g = SimpleDiGraph(SimpleWeightedDiGraph(W))
edge_weights = Float64[${", ".join(repr(w) for w in edge_weights)}]
% elif graph_form == "listed":
## Each (source, target) edge in the order edges(g) visits them, the order edge_weights follows
g = SimpleDiGraph(${n_nodes})
% for i, j in graph_edges:
add_edge!(g, ${i + 1}, ${j + 1})
% endfor
edge_weights = Float64[${", ".join(repr(w) for w in edge_weights)}]
% elif graph_form == "single":
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
g = complete_digraph(${n_nodes})
% endif

## ── Network ─────────────────────────────────────────────────────────────────
<%
default_coupling_name = next(iter(all_couplings.keys())) if all_couplings else (coupling.name if coupling else 'Diffusion')
# The single-node graph's self-loop carries the zero edge, so the node gets no coupling feedback
edge_var = 'edge_zero' if graph_form == "single" else f'edge_{default_coupling_name}'
%>
% if is_heterogeneous:
## Heterogeneous network: different vertex models per node
vertex_array = VertexModel[vertex_${model.name} for _ in 1:${n_nodes}]
% for node in nodes:
% if node_dynamics_map.get(node.id, model.name) != model.name:
vertex_array[${node_rows[node.id] + 1}] = vertex_${node_dynamics_map[node.id]}
% endif
% endfor

nw = Network(g, vertex_array, ${edge_var}; dealias=true)
% elif dealias:
nw = Network(g, vertex_${model.name}, ${edge_var}; dealias=true)
% else:
nw = Network(g, vertex_${model.name}, ${edge_var})
% endif

## ── Per-component defaults a fixpoint search starts from (set on network before NWState) ──
% if find_fixpoint:
% for node in nodes:
% for p_name, p_val in parse_node_parameters(node).items():
set_default!(nw, VIndex(${node_rows[node.id] + 1}, :${p_name}), ${p_val})
% endfor
% endfor
${edge_values(fixpoint=True)}\
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
    node_idx = node_rows[node.id] + 1
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
<%def name="on_line(ev)">\
## The line-partner copy an edge event's affect ends with, where the graph carries undirected lines
% if line_partners and str(ev.name) in edge_event_names:
    on_line!(p, ctx, (${"".join(f":{name}, " for name in (ev.affect_parameters or []))}))
% endif
</%def>
<%def name="edge_values(fixpoint)">\
## Each edge's weight, on the coupling's weight parameter, and declared parameters, at the position edges(g) visits it in: network defaults for a fixpoint search, the state's parameters otherwise.
% if edge_weights is not None and edge_split is not None:
% if fixpoint:
for (k, w) in enumerate(edge_weights)
    set_default!(nw, EIndex(k, :${edge_split['weight']}), w)
end
% else:
s.p.e[1:ne(g), :${edge_split['weight']}] = edge_weights
% endif
% endif
% for k, p_name, p_val in edge_parameters:
% if fixpoint:
set_default!(nw, EIndex(${k}, :${p_name}), ${p_val})
% else:
s.p.e[${k}, :${p_name}] = ${p_val}
% endif
% endfor
</%def>
% if not find_fixpoint and (edge_weights is not None or edge_parameters):
${edge_values(fixpoint=False)}\
% endif

## ── Events / Callbacks ──────────────────────────────────────────────────────
% if has_events:
<%
    # Separate events by type
    continuous_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'continuous']
    preset_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'preset_time']
    discrete_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'discrete']
%>
% if line_partners:

## An undirected line is two directed edges: an edge event's parameter changes are copied onto the line's other direction
line_partner = Dict{Int, Int}(${", ".join(f"{a} => {b}" for a, b in line_partners.items())})
function on_line!(p, ctx, psyms)
    k = get(line_partner, ctx.eidx, 0)
    k == 0 && return nothing
    q = NWParameter(ctx.integrator)
    for name in psyms
        q.e[k, name] = p[name]
    end
    nothing
end
% endif
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
${on_line(ev)}\
end
${ev.name}_cb = ContinuousComponentCallback(${ev.name}_cond, ${ev.name}_affect)
% if ev.target_component == 'all_edges':
for i in 1:ne(g)
    set_callback!(nw[EIndex(i)], ${ev.name}_cb)
end
% elif ev.target_component in event_edges:
% for k in event_edges[ev.target_component]:
set_callback!(nw[EIndex(${k})], ${ev.name}_cb)
% endfor
% endif
% endfor
% for ev, ev_src in preset_events:
<%
    affect_psyms = ', '.join(f':{p}' for p in (ev.affect_parameters or []))
    times = ', '.join(str(t) for t in ev.trigger_times) if ev.trigger_times else '0.0'
%>

## Preset-time callback: ${ev.name}
<%
    # An identical affect body already defined is reused.
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
${on_line(ev)}\
end
${ev.name}_cb = PresetTimeComponentCallback(${times}, ${ev.name}_affect)
% endif
% if ev.target_component in event_edges:
% for k in event_edges[ev.target_component]:
add_callback!(nw[EIndex(${k})], ${ev.name}_cb)
% endfor
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
% else:
% if find_fixpoint:
## ODEProblem from NWState: auto-extracts initial state, parameters, and callbacks
u0 = NWState(nw)
prob = ODEProblem(nw, u0, tspan)
% else:
prob = ODEProblem(nw, uflat(s), tspan, pflat(s))
% endif
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); ${solve_kwargs})
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
