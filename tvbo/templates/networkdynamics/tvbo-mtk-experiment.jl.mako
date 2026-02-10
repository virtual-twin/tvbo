## -*- coding: utf-8 -*-
<%doc>
MTK full experiment template for NetworkDynamics.jl.

Generates a self-contained Julia script using @component syntax:
  1. Package imports (ModelingToolkit + NetworkDynamics)
  2. @component vertex definitions (one per unique dynamics)
  3. @component edge definitions (one per unique coupling)
  4. Graph construction
  5. VertexModel/EdgeModel wrapping + Network assembly
  6. NWState + initial conditions + ODEProblem + solve
  7. Plot

Context: Pre-computed dict from BaseAdapter.prepare_context()
</%doc>
<%page args="experiment, model, network, integration, \
dynamics_dict, node_dynamics_map, all_couplings, coupling, \
coupling_vars, outdim, outsym_names, \
n_nodes, nodes, graph_gen, edges_list, emf_names, \
has_edge_matrix, has_explicit_edges, is_directed, \
sv_names, n_sv, is_heterogeneous, is_stochastic, \
dt, duration, solver_method, needs_stiff, needs_weighted, \
weight_matrix, weight_sym, \
dist_info, needs_random, dist_seed, \
all_events, has_events, coupling_observed, find_fixpoint, \
is_static, parse_node_parameters, get_noise_sigmas, graph_generator_call, \
tstops, vertex_dv_names"/>

## ── Packages ────────────────────────────────────────────────────────────────
using Graphs
using NetworkDynamics
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt
% if is_stochastic:
using StochasticDiffEq
% else:
using OrdinaryDiffEqTsit5
% endif
% if needs_stiff:
using OrdinaryDiffEqSDIRK
% endif
% if needs_weighted:
using DelimitedFiles
using SimpleWeightedGraphs
% endif

## ── Vertex models (@component) ───────────────────────────────────────────────
<%!
from tvbo.export.code import render_expression
%>
% for dyn_name, dyn in dynamics_dict.items():
% if is_static(dyn):
## Static vertex: algebraic constraints only, no dynamics
<%
    static_params = list((dyn.parameters or {}).keys())
    static_ct = list((dyn.coupling_terms or {}).keys())
    static_obs = list((dyn.observed or {}).values()) if getattr(dyn, 'observed', None) else []
    static_dvs = list((dyn.derived_variables or {}).values()) if getattr(dyn, 'derived_variables', None) else []
    # For static vertex, coupling_variable params serve as output
    static_coupling_vars = [name for name, sv in (dyn.state_variables or {}).items()
                            if getattr(sv, 'coupling_variable', False)]
    # If no state variables, check parameters for output
    static_output = static_coupling_vars if static_coupling_vars else list((dyn.parameters or {}).keys())[:1]
    all_sym = static_params + static_ct + [obs.name for obs in static_obs] + [dv.name for dv in static_dvs]
    juliacode_static = lambda expr: render_expression(expr, format='mtk', parameters=all_sym)
%>
@component function ${dyn.name}(; name)
% if static_params:
    @parameters begin
% for p_name in static_params:
<%
    p = dyn.parameters[p_name]
    p_val = p.value if p.value is not None else ''
    default_str = f' = {p_val}' if p_val != '' else ''
%>
        ${p_name}${default_str}
% endfor
    end
% endif
    @variables begin
% for ct_name in static_ct:
        ${ct_name}(t), [input=true]
% endfor
% for p_name in static_output:
        ${p_name}(t), [output=true]
% endfor
% for obs in static_obs:
        ${obs.name}(t)
% endfor
% for dv in static_dvs:
        ${dv.name}(t)
% endfor
    end
    eqs = [
% for sv_name, sv in (dyn.state_variables or {}).items():
        ${sv_name} ~ ${juliacode_static(sv.equation.rhs)},
% endfor
% for dv in static_dvs:
        ${dv.name} ~ ${juliacode_static(dv.equation.rhs)},
% endfor
% for obs in static_obs:
        ${obs.name} ~ ${juliacode_static(obs.equation.rhs)},
% endfor
    ]
    return System(eqs, t; name)
end

% else:
<%include file="/tvbo-mtk-vertex.jl.mako" args="model=dyn, all_couplings=all_couplings" />

% endif
% endfor

## ── Edge models (@component) ─────────────────────────────────────────────────
% for c_name, c in all_couplings.items():
<%include file="/tvbo-mtk-edge.jl.mako" args="coupling=c, is_directed=is_directed, outdim=outdim, outsym_names=outsym_names" />

% endfor

## ── Graph ───────────────────────────────────────────────────────────────────
<%
MATRIX_THRESHOLD = 50
use_matrix = has_explicit_edges and len(edges_list) > MATRIX_THRESHOLD
%>
% if has_edge_matrix:
G = readdlm("${emf_names[0]}", ',', Float64, '\n')
g_weighted = SimpleWeightedDiGraph(G)
edge_weights = getfield.(collect(edges(g_weighted)), :weight)
g = SimpleDiGraph(g_weighted)
% elif graph_gen:
g = ${graph_generator_call(graph_gen, n_nodes, 'julia')}
% elif use_matrix and weight_matrix is not None:
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
g = complete_graph(${n_nodes})
% endif

## ── VertexModel / EdgeModel wrappers ────────────────────────────────────────
<%
default_coupling_name = next(iter(all_couplings.keys())) if all_couplings else coupling.name
default_coupling = next(iter(all_couplings.values())) if all_couplings else coupling
# Determine input/output symbol names for vertex wrapping
# Inputs: coupling term names from the default dynamics
default_dyn = model
default_ct_names = list(default_dyn.coupling_terms.keys()) if default_dyn.coupling_terms else ['c_in']
default_coupling_vars_list = [name for name, sv in default_dyn.state_variables.items()
                              if getattr(sv, 'coupling_variable', False)]
default_output_vars = default_coupling_vars_list if default_coupling_vars_list else list(default_dyn.state_variables.keys())
%>
% if is_heterogeneous:
## Heterogeneous: instantiate each vertex with per-node parameters
<%
vertex_instances = []
for node in nodes:
    dyn_name = node_dynamics_map.get(node.id, model.name)
    dyn = dynamics_dict[dyn_name]
    node_params = parse_node_parameters(node)
    ct = list((dyn.coupling_terms or {}).keys()) if dyn.coupling_terms else default_ct_names
    cvars = [name for name, sv in (dyn.state_variables or {}).items()
             if getattr(sv, 'coupling_variable', False)]
    out = cvars if cvars else list((dyn.state_variables or {}).keys())
    # For static, output might be a parameter
    if is_static(dyn) and not out:
        out = list((dyn.parameters or {}).keys())[:1]
    vertex_instances.append((node, dyn_name, dyn, node_params, ct, out))
%>
% for node, dyn_name, dyn, node_params, ct, out in vertex_instances:
<%
    inst_name = f"v{node.id + 1}"
    param_kwargs = ", ".join(f"{k}={v}" for k, v in node_params.items()) if node_params else ""
    param_str = f"({param_kwargs})" if param_kwargs else "()"
%>
@named ${inst_name}_mtk = ${dyn_name}${param_str}
${inst_name} = VertexModel(${inst_name}_mtk, [:${", :".join(ct)}], [:${", :".join(out)}]; vidx=${node.id + 1})
% endfor

nw = Network([${", ".join(f"v{node.id + 1}" for node in nodes)}], edge_${default_coupling_name}; dealias=true)
% else:
## Homogeneous: single vertex model for all nodes
@named vertex_mtk = ${model.name}()
vertex_${model.name} = VertexModel(vertex_mtk, [:${", :".join(default_ct_names)}], [:${", :".join(default_output_vars)}])

nw = Network(g, vertex_${model.name}, edge_${default_coupling_name})
% endif

## ── Per-node defaults ───────────────────────────────────────────────────────
<%
# Collect per-edge parameter overrides
edge_param_entries = []
edges_with_keys = []
for edge in edges_list:
    s_node = min(edge.source, edge.target) + 1
    t_node = max(edge.source, edge.target) + 1
    edges_with_keys.append(((s_node, t_node), edge))
edges_with_keys.sort(key=lambda x: x[0])
for i, (key, edge) in enumerate(edges_with_keys):
    eparams = getattr(edge, 'parameters', None) or {}
    if isinstance(eparams, dict):
        for p_name, p_obj in eparams.items():
            if p_name == 'weight':
                continue
            val = getattr(p_obj, 'value', None) if hasattr(p_obj, 'value') else p_obj
            if val is not None:
                edge_param_entries.append((i + 1, p_name, val))
%>
% if find_fixpoint and is_heterogeneous:
## Set per-node parameter defaults for find_fixpoint
% for node in nodes:
<%
    node_idx = node.id + 1
    node_params = parse_node_parameters(node)
%>
% for p_name, p_val in node_params.items():
set_default!(nw, VIndex(${node_idx}, :${p_name}), ${p_val})
% endfor
% endfor
% elif find_fixpoint and not is_heterogeneous:
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
% if find_fixpoint and edge_param_entries:
## Set per-edge parameter defaults for find_fixpoint
% for eidx, p_name, val in edge_param_entries:
set_default!(nw, EIndex(${eidx}, :${p_name}), ${val})
% endfor
% endif

## ── Find fixpoint ───────────────────────────────────────────────────────────
% if find_fixpoint:
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
## Heterogeneous initial conditions
% for node in nodes:
<%
    dyn_name = node_dynamics_map.get(node.id, model.name)
    dyn = dynamics_dict[dyn_name]
    node_idx = node.id + 1
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
% elif sv.initial_value is not None:
s.v[${node_idx}, :${sv.name}] = ${sv.initial_value}
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
% elif sv.initial_value is not None:
s.v[1:nv(g), :${sv.name}] .= ${sv.initial_value}
% endif
% endfor
% for p_name, p_obj, d in collect_param_distributions(model):
for node in 1:nv(g)
    s.p.v[node, :${p_name}] = ${sample_expression(d, 'julia')}
end
% endfor
% endif
% if not find_fixpoint and edge_param_entries:

## Per-edge parameter overrides
% for eidx, p_name, val in edge_param_entries:
s.p.e[${eidx}, :${p_name}] = ${val}
% endfor
% endif

## ── Events / Callbacks ──────────────────────────────────────────────────────
% if has_events:
<%
    continuous_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'continuous']
    preset_events = [(ev, src) for ev, src in all_events if str(getattr(ev.event_type, 'text', ev.event_type)) == 'preset_time']
%>
% for ev, ev_src in continuous_events:
<%
    cond_syms = ', '.join(f':{s}' for s in (ev.condition_states or []))
    cond_psyms = ', '.join(f':{p}' for p in (ev.condition_parameters or []))
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
    reuse_affect = None
    for prev_ev, _ in continuous_events:
        if str(prev_ev.affect.rhs) == str(ev.affect.rhs) and \
           list(prev_ev.affect_parameters or []) == list(ev.affect_parameters or []):
            reuse_affect = prev_ev.name + '_affect'
            break
%>

## Preset-time callback: ${ev.name}
% if reuse_affect:
${ev.name}_cb = PresetTimeComponentCallback(${times}, ${reuse_affect})
% else:
${ev.name}_affect = ComponentAffect([], [${affect_psyms}]) do u, p, ctx
    ${ev.affect.rhs}
end
${ev.name}_cb = PresetTimeComponentCallback(${times}, ${ev.name}_affect)
% endif
% if ev.target_component and ev.target_component.startswith('vertex_'):
<%
    vidx = int(ev.target_component.split('_')[1])
%>
add_callback!(nw[VIndex(${vidx})], ${ev.name}_cb)
% elif ev.target_component and ev.target_component.startswith('edge_'):
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
<%
    tstops_str = ""
    if tstops:
        tstops_str = ", tstops=[" + ", ".join(str(t) for t in tstops) + "]"
%>\
% if find_fixpoint:
u0 = NWState(nw)
prob = ODEProblem(nw, u0, tspan)
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); saveat=${dt}${tstops_str})
% elif has_edge_matrix and (weight_sym is not None):
prob = ODEProblem(nw, uflat(s), tspan, pflat(p))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); saveat=${dt}${tstops_str})
% else:
prob = ODEProblem(nw, uflat(s), tspan, pflat(s))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); saveat=${dt}${tstops_str})
% endif

## ── Graph data (extracted by Python adapter) ────────────────────────────────
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
