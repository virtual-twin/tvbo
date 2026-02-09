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
  - Static vertices: dynamics with system_type="static" or no state_variables

Context: experiment (SimulationExperiment instance)
</%doc>
<%page args="experiment"/>
<%
from collections import OrderedDict

model = experiment.local_dynamics
network = experiment.network
integration = experiment.integration

# ── Build dynamics library ──────────────────────────────────────────────────
# Collect all unique dynamics models from the experiment
dynamics_dict = OrderedDict()
# Always include the default model first
if model:
    dynamics_dict[model.name] = model
# Add any additional dynamics from the library
if hasattr(experiment, 'dynamics') and isinstance(experiment.dynamics, dict) and experiment.dynamics:
    for name, dyn in experiment.dynamics.items():
        if name not in dynamics_dict:
            dynamics_dict[name] = dyn

# Check if network has nodes with per-node dynamics assignments
nodes = getattr(network, 'nodes', None) or []
has_heterogeneous_nodes = False
node_dynamics_map = {}  # node_id → dynamics_name
if nodes:
    for node in nodes:
        dyn_name = None
        if hasattr(node, 'dynamics') and node.dynamics:
            dyn_name = str(node.dynamics)
        node_dynamics_map[node.id] = dyn_name or model.name
        if dyn_name and dyn_name != model.name:
            has_heterogeneous_nodes = True

is_heterogeneous = has_heterogeneous_nodes and len(dynamics_dict) > 1

# ── Resolve couplings ──────────────────────────────────────────────────────
# Collect all unique coupling models
nw_couplings = getattr(network, 'coupling', None) or {}
if isinstance(nw_couplings, dict) and nw_couplings:
    coupling = next(iter(nw_couplings.values()))
    all_couplings = OrderedDict(nw_couplings)
else:
    coupling = experiment.coupling
    all_couplings = OrderedDict({coupling.name: coupling}) if coupling else OrderedDict()

# ── Compute coupling output dimension ──────────────────────────────────────
# The edge outdim matches the number of coupling variables (vertex outputs)
# For homogeneous networks, use the default model. For heterogeneous, use
# the model that has the most coupling outputs.
def _get_outdim(dyn):
    """Get coupling output dimension from a Dynamics model."""
    if not dyn or not dyn.state_variables:
        return 1
    coupling_vars = [name for name, sv in dyn.state_variables.items()
                     if getattr(sv, 'coupling_variable', False)]
    return len(coupling_vars) if coupling_vars else len(dyn.state_variables)

def _get_outsym_names(dyn, outdim):
    """Get output symbol names for edge model."""
    if outdim == 1:
        return ['coupling']
    coupling_vars = [name for name, sv in dyn.state_variables.items()
                     if getattr(sv, 'coupling_variable', False)]
    if coupling_vars:
        return [f'flow_{v}' for v in coupling_vars]
    return [f'flow_{name}' for name in list(dyn.state_variables.keys())[:outdim]]

outdim = _get_outdim(model)
outsym_names = _get_outsym_names(model, outdim)

sv_names = list(model.state_variables.keys())
n_sv = len(sv_names)
n_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)

dt = integration.step_size if integration else 0.01
duration = integration.duration if integration else 1000.0

# Detect stochastic: any state variable with noise intensity > 0
is_stochastic = False
for dyn in dynamics_dict.values():
    for sv in (dyn.state_variables or {}).values():
        n = getattr(sv, 'noise', None)
        if n and getattr(getattr(n, 'intensity', None), 'value', None):
            try:
                if float(n.intensity.value) > 0:
                    is_stochastic = True
                    break
            except Exception:
                pass
    if is_stochastic:
        break

# Detect static vertex models (no state variables, only output function)
def _is_static(dyn):
    """Check if dynamics is a static/algebraic model (no differential equations)."""
    sys_type = getattr(dyn, 'system_type', None)
    if sys_type and 'static' in str(sys_type).lower():
        return True
    return not dyn.state_variables
%>

## ── Packages ────────────────────────────────────────────────────────────────
using Graphs
using NetworkDynamics
% if is_stochastic:
using StochasticDiffEq
% else:
using OrdinaryDiffEqTsit5
% endif
<%
solver_method = getattr(integration, 'method', 'Tsit5') if integration else 'Tsit5'
needs_stiff = solver_method and 'auto' in str(solver_method).lower()
needs_weighted = bool(getattr(network, 'edge_matrix_files', None))
graph_gen = getattr(network, 'graph_generator', None)
%>
% if needs_stiff:
using OrdinaryDiffEqSDIRK
% endif
% if needs_weighted:
using DelimitedFiles
using SimpleWeightedGraphs
% endif
<%
# Directed graph: edge_matrix_files produce SimpleWeightedDiGraph → SimpleDiGraph,
# or GraphGenerator.directed flag
is_directed = needs_weighted or (graph_gen and getattr(graph_gen, 'directed', False))
%>

## ── Vertex models (node dynamics) ───────────────────────────────────────────
% for dyn_name, dyn in dynamics_dict.items():
% if _is_static(dyn):
## Static vertex: no dynamics, outputs parameter value
<%
    static_params = list((dyn.parameters or {}).keys())
    # Output symbols: must match the coupling variable names of the default model
    # so that ND.jl can wire the static output to the same edge input
    if coupling_vars:
        static_outsyms = coupling_vars[:1]  # static node outputs first coupling var
    else:
        static_outsyms = list((dyn.output or {}).keys()) if getattr(dyn, 'output', None) else \
                         list((dyn.state_variables or {}).keys()) if dyn.state_variables else \
                         [sv_names[0]] if sv_names else ['out']
%>
function ${dyn.name}_g!(out, x, p, t)
% if static_params:
    (${", ".join(static_params)},) = p
    out[1] = ${static_params[0]}
% else:
    out[1] = 0.0
% endif
    nothing
end
vertex_${dyn.name} = VertexModel(;
    g = ${dyn.name}_g!,
    outsym = [${", ".join(f':{s}' for s in static_outsyms)}],
% if static_params:
    psym = [${", ".join(f':{p} => {dyn.parameters[p].value}' for p in static_params)}],
% endif
    ff = NoFeedForward(),
    name = :${dyn.name},
)

% else:
<%include file="/tvbo-nd-vertex.jl.mako" args="model=dyn" />

% endif
% endfor

## ── Edge models (coupling) ──────────────────────────────────────────────────
% for c_name, c in all_couplings.items():
<%include file="/tvbo-nd-edge.jl.mako" args="coupling=c, is_directed=is_directed, outdim=outdim, outsym_names=outsym_names" />

% endfor

## ── Graph ───────────────────────────────────────────────────────────────────
<%
import numpy as np
from tvbo.templates.base.utils import graph_generator_call

emf_list = getattr(network, 'edge_matrix_files', None) or []
# Extract filename string from File objects
emf_names = []
for f in emf_list:
    if hasattr(f, 'name') and f.name:
        emf_names.append(str(f.name))
    elif isinstance(f, str):
        emf_names.append(f)
    else:
        fname = getattr(f, 'file_name', None) or getattr(f, 'path', None)
        if fname:
            emf_names.append(str(fname))

has_edge_matrix = len(emf_names) > 0

# Build graph from explicit edges
edges_list = getattr(network, 'edges', None) or []
has_explicit_edges = len(edges_list) > 0

# For large networks, build a weight matrix instead of individual add_edge! calls
MATRIX_THRESHOLD = 50
use_matrix = has_explicit_edges and len(edges_list) > MATRIX_THRESHOLD

if use_matrix:
    W = np.zeros((n_nodes, n_nodes))
    for e in edges_list:
        w = 1.0
        params = getattr(e, 'parameters', None) or []
        for p in params:
            pname = getattr(p, 'name', None) or (p.get('name') if isinstance(p, dict) else None)
            if pname == 'weight':
                w = float(getattr(p, 'value', None) or (p.get('value') if isinstance(p, dict) else 1.0))
        W[e.source, e.target] = w
%>
% if has_edge_matrix:
G = readdlm("${emf_names[0]}", ',', Float64, '\n')
g_weighted = SimpleWeightedDiGraph(G)
edge_weights = getfield.(collect(edges(g_weighted)), :weight)
g = SimpleDiGraph(g_weighted)
% elif graph_gen:
g = ${graph_generator_call(graph_gen, n_nodes, 'julia')}
% elif use_matrix:
using SimpleWeightedGraphs
W = [${'; '.join(' '.join(f'{W[i,j]:.6g}' for j in range(n_nodes)) for i in range(n_nodes))}]
g = SimpleDiGraph(SimpleWeightedDiGraph(W))
% elif has_explicit_edges:
g = SimpleDiGraph(${n_nodes})
    % for e in edges_list:
add_edge!(g, ${e.source + 1}, ${e.target + 1})
    % endfor
% else:
g = complete_graph(${n_nodes})
% endif

## ── Network ─────────────────────────────────────────────────────────────────
<%
# Determine the default coupling name (first one)
default_coupling_name = next(iter(all_couplings.keys())) if all_couplings else coupling.name

# Detect if coupling has a 'weight' parameter that should be set from edge_matrix_files
cparam_names_list = list((coupling.parameters or {}).keys())
has_weight_param = 'w' in cparam_names_list or 'weight' in cparam_names_list
weight_sym = 'w' if 'w' in cparam_names_list else ('weight' if 'weight' in cparam_names_list else None)
%>
% if is_heterogeneous:
## Heterogeneous network: different vertex models per node
vertex_array = VertexModel[vertex_${model.name} for _ in 1:${n_nodes}]
% for node in nodes:
% if node_dynamics_map.get(node.id, model.name) != model.name:
vertex_array[${node.id + 1}] = vertex_${node_dynamics_map[node.id]}
% endif
% endfor

nw = Network(g, vertex_array, edge_${default_coupling_name}; dealias=true)
% else:
nw = Network(g, vertex_${model.name}, edge_${default_coupling_name})
% endif
% if has_edge_matrix and weight_sym:

## Set per-edge weights from connectivity matrix
p = NWParameter(nw)
p.e[1:ne(g), :${weight_sym}] = edge_weights
% endif

## ── Initial state ───────────────────────────────────────────────────────────
<%
from tvbo.templates.base.utils import (
    collect_sv_distributions, collect_param_distributions,
    has_distributions, get_distribution_seed, sample_expression,
)
# Collect distributions from ALL dynamics models
all_sv_dists = {}
all_param_dists = {}
needs_random = False
dist_seed = 42
for dyn_name, dyn in dynamics_dict.items():
    if has_distributions(dyn):
        needs_random = True
        dist_seed = get_distribution_seed(dyn)
    all_sv_dists[dyn_name] = collect_sv_distributions(dyn)
    all_param_dists[dyn_name] = collect_param_distributions(dyn)
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
    node_idx = node.id + 1  # Julia 1-indexed
    # Per-node initial_state overrides (list of floats matching state var order)
    node_init = getattr(node, 'initial_state', None) or []
    node_sv_names = list((dyn.state_variables or {}).keys())
    # Per-node parameter overrides
    node_params = {}
    for p in (getattr(node, 'parameters', None) or []):
        if isinstance(p, dict):
            node_params[p.get('name', '')] = p.get('value')
        elif hasattr(p, 'name') and not isinstance(p, str):
            node_params[getattr(p, 'name', '')] = getattr(p, 'value', None)
        elif isinstance(p, str):
            # ParameterName is a string subclass with dict repr: "{'name': ..., 'value': ...}"
            import ast
            try:
                d = ast.literal_eval(str(p))
                if isinstance(d, dict) and 'name' in d:
                    node_params[d['name']] = d.get('value')
            except (ValueError, SyntaxError):
                pass
%>
% for i, sv in enumerate((dyn.state_variables or {}).values()):
<%
    # Priority: node.initial_state[i] > sv.initial_value > distribution
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
% for p_name in list((dyn.parameters or {}).keys()):
<%
    # Per-node parameter override (from YAML nodes section)
    p_val = node_params.get(p_name, None)
%>
% if p_val is not None:
s.p.v[${node_idx}, :${p_name}] = ${p_val}
% endif
% endfor
% for p_name, p, d in all_param_dists.get(dyn_name, []):
% if p_name not in node_params:
s.p.v[${node_idx}, :${p_name}] = ${sample_expression(d, 'julia')}
% endif
% endfor
% endfor
% else:
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
% for p_name, p, d in collect_param_distributions(model):
for node in 1:nv(g)
    s.p.v[node, :${p_name}] = ${sample_expression(d, 'julia')}
end
% endfor
% endif

## ── Problem + solve ─────────────────────────────────────────────────────────
tspan = (0.0, ${duration})
% if is_stochastic:
<%
sigma_vals = []
for sv in model.state_variables.values():
    n = getattr(sv, 'noise', None)
    val = 0.0
    if n and getattr(getattr(n, 'intensity', None), 'value', None):
        try:
            val = float(n.intensity.value)
        except Exception:
            pass
    sigma_vals.append(val)
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
% elif has_edge_matrix and weight_sym:
prob = ODEProblem(nw, uflat(s), tspan, pflat(p))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); saveat=${dt})
% else:
prob = ODEProblem(nw, uflat(s), tspan, pflat(s))
sol = solve(prob, ${solver_method}(${'TRBDF2()' if needs_stiff else ''}); saveat=${dt})
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
