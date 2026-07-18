# -*- coding: utf-8 -*-
<%doc>TVB-Optim Experiment Template. Context: experiment (SimulationExperiment).</%doc>
<%namespace name="fn" file="/base/function-def.mako"/>
<%namespace name="const" file="/base/constants.mako"/>
<%namespace name="search" file="tvbo-tvboptim-search.py.mako"/>
<%namespace name="sweep" file="tvbo-tvboptim-sweep.py.mako"/>
<%namespace name="lyap" file="tvbo-tvboptim-lyapunov.py.mako"/>\
<%
from tvbo.codegen import render_expression
from tvbo.templates.tvboptim.utils import (
    safe_name, iter_parameter_values, as_list, get_attr, is_network_observation, obs_has_all_args,
    get_observation_refs, parse_loss_function, parse_free_param, get_domain_bounds,
    parse_exploration, get_param_info, get_node_param_overrides,
    normalize_coupling_aliases, resolve_coupling_input_map,
    get_node_state_overrides, render_jax_default, get_mode_layout,
    get_all_observations_from_algo, network_axis_leaf, graph_selection,
)
import numpy as np
import re

# Must have experiment
assert 'experiment' in context.keys(), "experiment required for experiment template"

# Direct references to experiment components (LinkML guarantees these exist)
model = experiment.dynamics
integration = experiment.integration
network = experiment.network

# Collect user-defined functions from experiment.functions
# These are functions defined in YAML that need to be recognized by the code printer
exp_functions = experiment.functions or {}
if hasattr(exp_functions, 'items'):
    user_functions = {str(fname): str(fname) for fname in exp_functions.keys()}
else:
    user_functions = {}

# JAX code generation helpers
jaxcode = lambda expr, params=None: render_expression(expr, format='jax', user_functions=user_functions, parameters=params)
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Extract key metadata from model. For number_of_modes>1 the per-node mode axis is
# folded into the state axis: state_names is the solver's flat (variable, mode) slot
# ordering, while var_names keeps the original variables (used for the result mode dim).
n_modes, state_names, var_slots = get_mode_layout(model)
var_names = list(model.state_variables.keys())
if n_modes > 1:
    # Informational reliability notice on the run logger, not warnings.warn: it fires
    # once per render and is a heads-up, not an API misuse. Users still see it at
    # WARNING level; it stays out of pytest's warnings summary.
    import logging as _logging
    _logging.getLogger("tvbo.run").warning(
        "number_of_modes=%s (mode-coupled model '%s') on the tvboptim backend is "
        "EXPERIMENTAL: the per-node mode axis is folded into the state axis (each "
        "variable occupies n_modes scalar slots). Validated against TVB to machine "
        "precision for the Stefanescu-Jirsa ReducedSet models; other multi-mode "
        "coupling topologies may not be faithful. Use the tvb backend for reference "
        "results.",
        n_modes, getattr(model, "name", "?"),
    )
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Model output variables (from model.output attribute)
# These are the variables the model defines as its primary output (e.g., v_pyr = y1 - y2)
model_output_vars = getattr(model, 'output', None) or []
if isinstance(model_output_vars, str):
    model_output_vars = [model_output_vars]
model_derived_outputs = [v for v in model_output_vars if v in (model.derived_variables or {})]
model_state_outputs = [v for v in model_output_vars if v in state_names]
has_model_output = bool(model_output_vars)

# Extract state variable bounds (for BoundedSolver)
# Uses SymPy oo so code printers emit the correct backend literal
from tvbo.templates.tvboptim.utils import get_state_bounds, format_bounds_array
state_bounds_lo, state_bounds_hi, has_state_bounds = get_state_bounds(model)
state_bounds_lo_str = format_bounds_array(state_bounds_lo, 'jax')
state_bounds_hi_str = format_bounds_array(state_bounds_hi, 'jax')

# Build coupling_inputs dict from model.coupling_inputs
coupling_inputs_dict = {}
coupling_keys = {}  # ci_name -> list of key names

if model.coupling_inputs:
    for ci_name, ci in model.coupling_inputs.items():
        coupling_inputs_dict[ci_name] = ci.dimension or 1
        if ci.keys:
            coupling_keys[ci_name] = list(ci.keys)
elif model.coupling_terms:
    for ct_name in model.coupling_terms.keys():
        coupling_inputs_dict[ct_name] = 1

# First coupling input key (for parameter access) - None for uncoupled models
first_coupling_key = list(coupling_inputs_dict.keys())[0] if coupling_inputs_dict else None
has_coupling = bool(coupling_inputs_dict)

# Build all_couplings dict from network.coupling (keyed by function name — schema convention)
# Fall back to experiment-level coupling if network.coupling is empty.
all_couplings = dict(network.coupling.items()) if network.coupling else {}
if not all_couplings and getattr(experiment, 'coupling', None):
    _exp_c = experiment.coupling
    if hasattr(_exp_c, 'items'):
        all_couplings = dict(_exp_c.items())
    else:
        all_couplings = {_exp_c.name or 'coupling': _exp_c}
all_couplings = normalize_coupling_aliases(all_couplings, model)

# Coupling-input → coupling-function mapping (+ local-term drop) resolved in the
# tvboptim Python layer, not here — see resolve_coupling_input_map.
ci_coupling_map, func_to_first_ci = resolve_coupling_input_map(model, all_couplings, coupling_inputs_dict)

# Translate function-name coupling key to ci name for tvboptim state access
_to_ci_key = lambda k: func_to_first_ci.get(k, k) if k else None

# Check if any coupling has delays
has_delay = any(c.delayed for c in all_couplings.values() if c)
# Differentiable (interpolated) delays are OPT-IN: only experiments whose
# coupling sets `interpolate_delays: true` use the decoupled-max_delay graph
# API (which needs the differentiable-delays tvboptim build). Everything else
# uses the stock delay graph that derives max_delay from the delays.
interpolate_delays = any(bool(getattr(c, 'interpolate_delays', False)) for c in all_couplings.values() if c)

# tract lengths → DenseLengthGraph, explicit per-edge delays → DenseDelayGraph.
use_length_graph, use_delay_graph = graph_selection(network, has_delay)

# Sparse coupling opt-in (Network.graph_representation: sparse): sparsify the weight matrix to
# BCOO so the `pre @ weights` reduction runs as an edge-sum (O(nnz)) instead of a dense NxN
# matmul. Requires EVERY coupling to be instantaneous AND vectorized (source-only `pre`, same
# rule as resolve_coupling_spec): a per-edge `pre` (e.g. sin(x_j - x_i)) or a delayed one would
# hit jsparse.sparsify on a nonlinear term and crash, so such networks fall back to dense.
_is_vectorized_coupling = lambda c: bool(getattr(c, 'vectorized', False)) or (
    bool(getattr(c, 'local_states', None)) and not getattr(c, 'incoming_states', None))
use_sparse = (
    str(getattr(network, 'graph_representation', 'auto') or 'auto') == 'sparse'
    and not has_delay
    and all(_is_vectorized_coupling(c) for c in all_couplings.values() if c)
)

# Collect all coupling parameters (for optimization)
all_coupling_params = {}  # (coupling_key, param_name) -> param_obj
all_coupling_param_shapes = {}  # (coupling_key, param_name) -> shape_str
coupling_param_names = set()  # Simple set of param names for quick lookup
for ck, cobj in all_couplings.items():
    if cobj and cobj.parameters:
        for p in cobj.parameters.values():
            all_coupling_params[(_to_ci_key(ck), p.name)] = p
            coupling_param_names.add(p.name)
            if p.shape and 'n_nodes' in str(p.shape):
                all_coupling_param_shapes[(_to_ci_key(ck), p.name)] = str(p.shape)

# Integration metadata
SOLVER_MAP = {'euler': 'Euler', 'heun': 'Heun', 'heunstochastic': 'Heun', 'rk4': 'RungeKutta4', 'rungekutta4thorder': 'RungeKutta4', 'runge_kutta': 'RungeKutta4', 'rungekutta': 'RungeKutta4'}
method = (integration.method or 'euler').lower()
solver_class = SOLVER_MAP.get(method)
assert solver_class, f"Unknown solver method: {method}. Valid: {list(SOLVER_MAP.keys())}"
dt = float(integration.step_size)
# Seconds per model time unit (ms -> 0.001). The Jacobian A inherits the model's native
# rate units (per-ms by tvbo convention); analytic-frequency diagnostics (PSD in Hz)
# convert A to per-second with this factor so the physical frequency axis is consistent.
from tvbo.utils.units import unit_to_si_factor
time_si_factor = unit_to_si_factor(getattr(integration, 'unit', None) or 'ms')

# Differentiation strategy -> native-solver kwargs, resolved in the tvboptim Python
# layer (shared with the solver template) rather than duplicated across mako blocks.
from tvbo.templates.tvboptim.utils import resolve_solver_kwargs, resolve_optimizer_mode, render_analysis_observations, render_recorded_observable, render_inference, render_adiabatic_signal, resolve_reduction, edge_label, edge_const, node_label, node_const
solver_kwargs_str = resolve_solver_kwargs(integration, dt)
# Warm-start/adiabatic scans run a plain forward solver but must honour the
# coupling_evaluation choice (recompute_coupling_per_stage); gradient kwargs
# (grad_horizon/block_size) don't apply to a forward scan, so pass only this.
_ce = getattr(integration, 'coupling_evaluation', None) if integration else None
warmstart_solver_kwargs = 'recompute_coupling_per_stage=True' if str(_ce) == 'per_stage' else ''
opt_mode = resolve_optimizer_mode(integration)

# Noise configuration from state_variables or integration.
# tvboptim's AdditiveGaussianNoise expects sigma = standard deviation
# of the per-step Wiener increment (increment = sigma * sqrt(dt) * N(0,1)).
# TVB's convention stores nsig = D = sigma^2 / 2 in `noise.intensity`.
# So when the YAML provides `intensity` (TVB style), convert via
# sigma = sqrt(2 * intensity). When it provides `sigma` directly, use as-is.
import math as _math
def _noise_sigma(noise_obj):
    if not noise_obj:
        return 0.0
    sp = (noise_obj.parameters or {}).get('sigma') if noise_obj.parameters else None
    if sp is not None:
        return float(sp.value if sp.value is not None else sp)
    intens = getattr(noise_obj, 'intensity', None)
    if intens is not None:
        D = float(intens.value if intens.value is not None else intens)
        return _math.sqrt(2.0 * D) if D > 0 else 0.0
    return 0.0

noise_sigma_per_state = []
noise_targets = []
for sv_name, sv in model.state_variables.items():
    sigma = _noise_sigma(getattr(sv, 'noise', None))
    if sigma > 0:
        noise_targets.append(sv_name)
    noise_sigma_per_state.append(sigma)

# Integration-level noise applies to all states if no per-state noise
if not any(s > 0 for s in noise_sigma_per_state):
    sigma = _noise_sigma(getattr(integration, 'noise', None))
    if sigma > 0:
        noise_sigma_per_state = [sigma] * len(model.state_variables)
        noise_targets = list(model.state_variables.keys())

has_noise = any(s > 0 for s in noise_sigma_per_state)
noise_sigma = noise_sigma_per_state if len(set(noise_sigma_per_state)) > 1 else [noise_sigma_per_state[0]] if noise_sigma_per_state else [0.0]
# For targeted noise (apply_to), extract the sigma for the targeted states only
# AdditiveNoise takes a single scalar sigma, so all targeted states share the same sigma
noise_sigma_targeted = [s for s in noise_sigma_per_state if s > 0]
noise_sigma_value = noise_sigma_targeted[0] if noise_sigma_targeted else 0.0

# Network metadata
n_nodes = N_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
_cs = getattr(network, 'conduction_speed', None)
conduction_speed = float(_cs.value if hasattr(_cs, 'value') else _cs) if _cs is not None else 1.0

# Network.transforms is applied once at resolution time by
# Network._apply_transform, so by the time `weights` reaches the
# generated `run_experiment` it is already the transformed matrix.
# No runtime inlining is needed.
has_weight_transforms = False
weight_transform_jax = []

# Simulation parameters
assert integration.duration, "integration.duration required in YAML"
t1_default = float(integration.duration)
transient_time = float(integration.transient_time) if integration.transient_time else 0.0
has_transient = transient_time > 0

# Execution config
exec_config = experiment.execution
n_workers = int(exec_config.n_workers) if exec_config and exec_config.n_workers else 1
n_threads = int(exec_config.n_threads) if exec_config and exec_config.n_threads else -1
precision = str(exec_config.precision) if exec_config and exec_config.precision else 'float64'
accelerator = str(exec_config.accelerator) if exec_config and exec_config.accelerator else 'cpu'
enable_x64 = precision == 'float64'
random_seed = int(exec_config.random_seed) if exec_config and exec_config.random_seed else 0

# Build observations dict from experiment.observations (analysis observations are
# handled by their own path, not the raw/network monitor categorisation).
observations_dict = {n: o for n, o in experiment.observations.items() if getattr(o, 'analysis', None) is None} if experiment.observations else {}

# Categorize observations using utils
network_observation_names, observation_names = get_observation_refs(observations_dict)

# Class name from model
dynamics_class = model.name.replace(' ', '').replace('-', '') if model.name else 'GeneratedDynamics'

# Dynamics parameter info (shared utility)
dyn_param_names, dyn_param_defaults, dyn_param_shapes = get_param_info(model.parameters)

# Per-node parameter overrides from network.nodes[].parameters
# If nodes define e.g. B=17.6 on node 1, auto-promote B to heterogeneous array
node_param_overrides = get_node_param_overrides(network, n_nodes, dyn_param_defaults)
for _np_name in node_param_overrides:
    if _np_name not in dyn_param_shapes:
        dyn_param_shapes[_np_name] = '(n_nodes,)'

# Per-node initial state overrides from node ``state:`` entries
# e.g. nodes[0].state = {theta: 0.8} → overrides default initial_value per node
_default_init = [
    (float(sv.initial_value) if sv.initial_value is not None else 0.0)
    for sv in model.state_variables.values()
    for _ in range(n_modes)  # one entry per (variable, mode) solver slot
]
node_state_overrides = get_node_state_overrides(network, n_nodes, state_names, _default_init)

# initial_state.method == from_working_point: a warm-start ramp (reusing ExplorationAxis:
# parameter + Range domain) that reaches a working point (operating branch) and seeds this
# run's IC from the settled endpoint. None unless declared. Per-step settle = the
# experiment's transient (else its main duration) — no separate settle concept.
from_working_point = None
_ini = getattr(experiment, 'initial_state', None)
if _ini is not None and str(getattr(_ini, 'method', '') or '') == 'from_working_point':
    _rax = _ini.ramp
    assert _rax is not None and getattr(_rax, 'parameter', None), \
        "initial_state.method=from_working_point requires ramp.parameter + ramp.domain"
    _rdom = _rax.domain
    _rlo = float(getattr(_rdom, 'lo', 0.0) or 0.0)
    _rhi = float(_rdom.hi)
    _rn = getattr(_rdom, 'n', None)
    _rnpts = int(_rn) if _rn else int(round((_rhi - _rlo) / float(_rdom.step))) + 1
    _rtr = float(getattr(integration, 'transient_time', 0.0) or 0.0)
    from_working_point = {
        'path': 'dynamics.%s' % str(_rax.parameter).rsplit('.', 1)[-1],
        'lo': _rlo, 'hi': _rhi, 'n': _rnpts,
        'settle': _rtr if _rtr > 0 else float(integration.duration),
    }

# initial_state.method == from_experiment with source_point == 'branch': the source run's
# WHOLE recorded branch is a per-cell seed. An independent exploration then restarts an
# analysis (Lyapunov, ...) at every branch point in parallel — shardable across HPC array
# tasks, unlike the sequential scan that produced the branch. The per-cell (parameter value,
# settled state) pairs arrive at runtime as branch_seed (Experiment._resolve_from_experiment_
# branch) → the _BRANCH_SEED module global. None unless declared.
from_experiment_branch = (
    _ini is not None
    and str(getattr(_ini, 'method', '') or '') == 'from_experiment'
    and str(getattr(_ini, 'source_point', '') or 'endpoint') == 'branch'
)

# Detect parameters with distribution.axis == 'time' — these are stochastic
# time-varying inputs pre-generated as arrays and indexed per integration step.
# Not regular params: excluded from DEFAULT_PARAMS, trajectories injected after prepare().
stochastic_param_names = set()
stochastic_param_info = {}  # name -> {dist, lo, hi, seed, default}
for pname in list(dyn_param_names):
    p_obj = (model.parameters[pname] if pname in model.parameters else None) if model.parameters else None
    if p_obj and getattr(p_obj, 'distribution', None):
        dist = p_obj.distribution
        axis = str(getattr(dist, 'axis', 'space'))
        if axis == 'time' or 'time' in axis:
            stochastic_param_names.add(pname)
            domain = getattr(dist, 'domain', None)
            dist_name = str(getattr(dist, 'name', 'Uniform')).lower()
            # Explicit mean/std (sigma) from the distribution parameters take
            # precedence over the domain; the domain is only a sampling-bounds
            # fallback (mean<-value, std<-(hi-lo)/4) when they are not given.
            _dmean = _dstd = None
            _dparams = getattr(dist, 'parameters', None) or {}
            for _dp in (_dparams.values() if hasattr(_dparams, 'values') else _dparams):
                _dn = str(getattr(_dp, 'name', ''))
                _dv = getattr(_dp, 'value', None)
                if _dv is None:
                    continue
                if _dn in ('mean', 'mu'):
                    _dmean = float(_dv)
                elif _dn in ('std', 'sigma', 'sd'):
                    _dstd = float(_dv)
            _lo = float(getattr(domain, 'lo', 0)) if domain else 0.0
            _hi = float(getattr(domain, 'hi', 1)) if domain else 1.0
            stochastic_param_info[pname] = {
                'dist': dist_name,
                'lo': _lo,
                'hi': _hi,
                'default': float(p_obj.value) if p_obj.value is not None else 0.0,
                'mean': _dmean,
                'std': _dstd,
                'seed': int(getattr(dist, 'seed', None) or 42),
                'shape': str(getattr(p_obj, 'shape', '')) if getattr(p_obj, 'shape', None) else '',
            }
# Remove stochastic params from dynamics params (trajectories injected after prepare)
dyn_param_names = [p for p in dyn_param_names if p not in stochastic_param_names]

# Detect state variables with distributions for IC-based trials
# When n_trials > 1 and state variables have distributions, each trial
# samples different initial conditions from those distributions.
sv_distribution_info = {}
for sv_name, sv in model.state_variables.items():
    dist = getattr(sv, 'distribution', None)
    if dist:
        # Fallback chain: distribution.domain → sv.domain → default
        domain = getattr(dist, 'domain', None)
        if not domain:
            domain = getattr(sv, 'domain', None)
        lo = float(domain.lo) if domain and domain.lo is not None else -1.0
        hi = float(domain.hi) if domain and domain.hi is not None else 1.0
        sv_distribution_info[sv_name] = {
            'dist': str(getattr(dist, 'name', 'Uniform')).lower(),
            'lo': lo,
            'hi': hi,
            'idx': state_names.index(sv_name if n_modes == 1 else f"{sv_name}__mode0"),
            'seed': int(getattr(dist, 'seed', None) or 42),
        }

# === Events metadata (stimuli and other time-dependent inputs) ===
# Schema: experiment.events is multivalued dict of Event objects
# Each stimulus-type event becomes an AbstractExternalInput, available as a variable in dfun
events_list = list(experiment.events.values()) if experiment.events else []
# Events that become an AbstractExternalInput (a variable available in the dfun):
# open-loop 'stimulus'/'stimulation' time functions AND closed-loop 'continuous'
# events, whose onset is triggered by a state condition crossing zero (a stateful
# ExternalInput that arms on the crossing and then emits its affect waveform).
def _is_external_input_event(ev):
    et = str(getattr(ev, 'event_type', 'stimulus'))
    return ('stimul' in et) or (et in ('continuous', 'discrete'))
stimulus_events = [ev for ev in events_list if _is_external_input_event(ev)]
has_stimulus_events = len(stimulus_events) > 0

# A stimulus event whose signal is an iid per-step draw (an event parameter with
# distribution.axis == 'time') needs the same step-time freeze as stochastic
# dynamics params: multi-stage solvers (Heun/RK4) must see one sample per step,
# not advance the step index at the t+dt sub-evaluation.
def _event_is_stochastic(ev):
    params = dict(ev.parameters) if getattr(ev, 'parameters', None) else {}
    for pobj in params.values():
        dist = getattr(pobj, 'distribution', None)
        if dist is not None and 'time' in str(getattr(dist, 'axis', 'space')):
            return True
    return False
has_stochastic_stimulus = any(_event_is_stochastic(ev) for ev in stimulus_events)

# External-input scope keys (stimulus event names) for the shared dotted-ref resolver:
# `<event>.<param>` -> `external.<event>.<param>` (e.g. stimulus.amplitude).
external_input_keys = {str(ev.name) for ev in stimulus_events}

# === Optimization metadata ===
# Schema: experiment.optimizations is multivalued dict, opt.stages is inlined_as_list
optim_list = list(experiment.optimizations.values()) if experiment.optimizations else []
has_optimization = len(optim_list) > 0

# === Bayesian inference metadata ===
inference_list = list(experiment.inferences.values()) if getattr(experiment, 'inferences', None) else []
has_inference = len(inference_list) > 0

# === Algorithm metadata (FIC, etc.) ===
# Schema: experiment.algorithms is multivalued dict
algorithms_list = list(experiment.algorithms.values()) if experiment.algorithms else []
has_algorithms = len(algorithms_list) > 0

# Extract optimizable parameters from optimization stages
optim_param_info = {}

# optimization.stages is always a list (inlined_as_list: true)
# If no stages, fall back to optimization-level free_parameters (flat mode)
for opt in optim_list:
    stages = opt.stages or []
    if not stages and opt.free_parameters:
        stages = [opt]  # Treat opt itself as a single stage
    for stage in stages:
        for fp in (stage.free_parameters or []):
            if isinstance(fp, str):
                ref = fp
                is_hetero = False
            else:
                # FreeParameter wrapper: .parameter is dotted ref
                ref = str(getattr(fp, 'parameter', '') or getattr(fp, 'name', ''))
                is_hetero = bool(getattr(fp, 'heterogeneous', False)) or (
                    getattr(fp, 'shape', None) and 'n_nodes' in str(fp.shape)
                )
            pname = ref.rsplit('.', 1)[-1] if '.' in ref else ref
            optim_param_info[pname] = {'heterogeneous': bool(is_hetero)}

# Collect param objects with heterogeneous info
# Separate dynamics vs coupling parameters
optim_params = []  # Dynamics parameters
optim_coupling_params = []  # Coupling parameters

for name, param in model.parameters.items():
    if str(name) in optim_param_info:
        param._optim_heterogeneous = optim_param_info[str(name)]['heterogeneous']
        optim_params.append(param)

# Check coupling parameters from all_couplings
for coupling_key, coupling_obj in all_couplings.items():
    if coupling_obj and coupling_obj.parameters:
        for name, param in coupling_obj.parameters.items():
            pname = str(name)
            if pname in optim_param_info or param.free:
                is_hetero = optim_param_info.get(pname, {}).get('heterogeneous', False)
                if not is_hetero:
                    is_hetero = param.heterogeneous or param.shape
                param._optim_heterogeneous = bool(is_hetero)
                param._coupling_key = _to_ci_key(coupling_key)
                optim_coupling_params.append(param)

coupling_optim_params = optim_coupling_params  # Alias for backwards compatibility

# =============================================================================
# Parse ALL optimization stages into structured list
# =============================================================================
# Coupling keys for parse_free_param
coupling_keys = set(all_couplings.keys())

# Wrapper for parse_free_param that passes model context
_parse_free_param = lambda fp: parse_free_param(fp, coupling_keys, model, all_couplings)

optimization_stages = []
for opt in optim_list:
    # Schema: opt.stages is always a list (inlined_as_list: true)
    # If no stages, fall back to optimization-level free_parameters (flat mode)
    stages_raw = opt.stages or []
    if not stages_raw and opt.free_parameters:
        stages_raw = [opt]  # Treat opt itself as a single stage

    for stage in stages_raw:
        # warmup_from only exists on OptimizationStage, not Optimization (flat mode)
        warmup_from = getattr(stage, 'warmup_from', None)
        stage_info = {
            'name': str(stage.name) if stage.name else f'stage_{len(optimization_stages)}',
            'label': str(stage.label) if stage.label else '',
            'algorithm': str(stage.algorithm) if stage.algorithm else 'adam',
            'learning_rate': float(stage.learning_rate) if stage.learning_rate else 0.01,
            'max_iterations': int(stage.max_iterations) if stage.max_iterations else 100,
            'warmup_from': str(warmup_from) if warmup_from else None,
            'free_parameters': [],
            'hyperparameters': {},
        }

        # Schema: free_parameters and hyperparameters are lists (inlined_as_list: true)
        for fp in (stage.free_parameters or []):
            parsed = _parse_free_param(fp)
            if parsed:
                stage_info['free_parameters'].append(parsed)

        # Filter out non-optax hyperparameters (has_aux is determined automatically)
        for hp in (stage.hyperparameters or []):
            hp_name = hp.name
            hp_value = hp.value
            # Skip non-optax hyperparameters
            if hp_name in ('has_aux',):
                continue
            if hp_name and hp_value is not None:
                stage_info['hyperparameters'][str(hp_name)] = float(hp_value)

        optimization_stages.append(stage_info)

# For single-stage or default case, extract settings from first stage
optimizer_name = optimization_stages[0]['algorithm'] if optimization_stages else 'adam'
learning_rate = optimization_stages[0]['learning_rate'] if optimization_stages else 0.01
max_steps = optimization_stages[0]['max_iterations'] if optimization_stages else 100
optimizer_hyperparams = optimization_stages[0]['hyperparameters'] if optimization_stages else {}

# Optimization integration settings (overrides experiment defaults if specified)
# If optimization has its own integration, we need fresh prepare() before optimization
opt_integration = None
opt_has_custom_integration = False
opt_solver_class = solver_class  # Default to experiment-level
opt_dt = dt
opt_t1 = t1_default
opt_has_state_bounds = has_state_bounds
opt_state_bounds_lo = state_bounds_lo
opt_state_bounds_hi = state_bounds_hi

if optim_list and optim_list[0].integration:
    opt_integration = optim_list[0].integration
    opt_has_custom_integration = True
    # Override integration settings from optimization.integration
    opt_method = (opt_integration.method or method).lower()
    opt_solver_class = SOLVER_MAP.get(opt_method, solver_class)
    opt_dt = float(opt_integration.step_size) if opt_integration.step_size else dt
    opt_t1 = float(opt_integration.duration) if opt_integration.duration else t1_default

# Check if optimization depends on an algorithm (copy that algorithm's result state)
# If no depends_on, optimization starts from FRESH network defaults (not algorithm results)
opt_depends_on = None
if optim_list:
    opt_depends_on = getattr(optim_list[0], 'depends_on', None)

# Schema provides ifabsent defaults, so these should always be populated
# Only assert if optimization is requested but values somehow missing
if has_optimization:
    assert optimizer_name, "optimization.algorithm not found (schema default: 'adam')"
    assert learning_rate is not None, "optimization.learning_rate not found (schema default: 0.001)"
    assert max_steps is not None, "optimization.max_iterations not found (schema default: 100)"

# === Observations metadata ===
# Split experiment.observations into raw vs derived views based on
# whether each Observation's `source` references another observation
# in the same experiment.
from tvbo.codegen.templater import is_derived as _is_derived
_all_observations = dict(experiment.observations) if experiment.observations else {}
# Analysis observations operate on the solve/loss (gradient, finite-difference,
# Lyapunov, ...) — handled by a dedicated path, not the raw/derived pipelines.
analysis_observations_dict = {n: o for n, o in _all_observations.items() if getattr(o, 'analysis', None) is not None}
analysis_observation_names = set(analysis_observations_dict.keys())
has_lyapunov = any(str(getattr(o.analysis, 'type', '') or '') == 'lyapunov' for o in analysis_observations_dict.values())

def _lyap_meta(_rn, _ctx):
    """Resolve a recorded Lyapunov analysis observation to backend-agnostic metadata.

    Shared by the two per-cell restart paths — the warm-start scan's post-scan pass and
    the from_experiment:branch restart — so both read segment_time / n_steps / n_exponents
    identically. ``_ctx`` names the caller for the error message.
    """
    _an = analysis_observations_dict[_rn].analysis
    _atype = str(getattr(_an, 'type', '') or '')
    assert _atype == 'lyapunov', (
        f"{_ctx} records analysis observation '{_rn}' of type '{_atype}'; only 'lyapunov' "
        "is restartable per branch point (it is seeded from each point's settled state).")
    _ap = {str(k): (v.value if hasattr(v, 'value') else v)
           for k, v in (getattr(_an, 'parameters', None) or {}).items()}
    return {'name': _rn, 'type': _atype,
            'segment_time': float(_ap.get('segment_time', 1.0)),
            'n_steps': int(_ap.get('n_steps', _ap.get('n', 10))),
            'n_exponents': int(_ap.get('n_exponents', _ap.get('k', 1)))}
observations = {n: o for n, o in _all_observations.items() if not _is_derived(o, experiment) and n not in analysis_observation_names}
derived_observations_dict = {n: o for n, o in _all_observations.items() if _is_derived(o, experiment) and n not in analysis_observation_names}
derived_observation_names = set(derived_observations_dict.keys())

def get_obs(name):
    """Look up observation by name from observations dict."""
    return observations.get(name)

def get_pipeline_output_key(obs_name):
    """Extract the last pipeline step's output key for an observation.

    Returns None if no explicit output is defined (caller should use .data or the value directly).
    """
    # Check regular observations first
    obs_obj = get_obs(obs_name)
    # Also check derived observations
    if not obs_obj:
        obs_obj = derived_observations_dict.get(obs_name)
    if obs_obj and obs_obj.pipeline:
        # Schema: pipeline is always a list (inlined_as_list: true)
        last_step = obs_obj.pipeline[-1]
        if last_step.output:
            # Handle multi-output (comma-separated) - take the last one as the "main" output
            outputs = [o.strip() for o in str(last_step.output).split(',')]
            return outputs[-1]
    # No explicit output - return None so callers use .data or direct value
    return None

# === Exploration metadata ===
# Schema: experiment.explorations is a multivalued dict, but users sometimes
# assign a single Exploration / list directly (which clobbers the container
# with a JsonObj that has no .values()). Coerce defensively.
_expl = experiment.explorations
if not _expl:
    exploration_list = []
elif hasattr(_expl, 'values') and callable(_expl.values):
    exploration_list = list(_expl.values())
elif isinstance(_expl, (list, tuple)):
    exploration_list = list(_expl)
else:
    exploration_list = [_expl]
has_explorations = len(exploration_list) > 0

# Parse explorations - uses schema ifabsent defaults
# Schema defaults: n_parallel=1, mode='product'
explorations = []
for expl in exploration_list:
    assert expl.name, "exploration.name required in YAML"
    exp_info = {
        # Sanitize: name is used as a Python function identifier in generated code.
        'name': safe_name(expl.name),
        'label': expl.label or '',
        # mode has schema ifabsent: string(product)
        'mode': expl.mode or 'product',
        # n_parallel is the backend-agnostic batch size: how many sweep cells the
        # backend processes as one vectorised chunk. The template translates this to
        # tvboptim internals (n_vmap, n_pmap) without exposing those names.
        'n_parallel': int(expl.n_parallel) if expl.n_parallel is not None else 1,
        # n_trials has schema ifabsent: integer(1)
        'n_trials': int(expl.n_trials) if expl.n_trials is not None else 1,
        # average: 'trials' to average over n_trials, None for individual results
        'average': str(expl.average) if expl.average else None,
        # parallel_mode: vmap | lax_map | pmap | auto. Defaults to auto (=lax_map at codegen).
        'parallel_mode': str(expl.parallel_mode) if getattr(expl, 'parallel_mode', None) else 'auto',
        'parallel_batch_size': int(expl.parallel_batch_size) if getattr(expl, 'parallel_batch_size', None) else None,
        'axes': [],
        # Observations to compute + stack per grid point (derived + `analysis` diagnostics).
        # NOTE: this block duplicates utils.parse_exploration — should be consolidated onto it.
        'record': [str(r) for r in (getattr(expl, 'record', None) or [])],
    }
    # Schema: space is keyed by parameter (optional for trial-only explorations)
    axes_list = as_list(expl.space)
    def _resolve_n(domain):
        """Compute n from domain: prefer n, else compute from step, else default 50."""
        if domain.n:
            return int(domain.n)
        if domain.step and domain.lo is not None and domain.hi is not None:
            return int(round((float(domain.hi) - float(domain.lo)) / float(domain.step))) + 1
        return 50
    for axis in axes_list:
        domain = axis.domain
        explored_values = axis.explored_values
        _el_domains = getattr(axis, 'element_domains', None) or []
        _builder = getattr(axis, 'builder', None)
        # ExplorationAxis.reduce: collapse this axis by a statistic in the result
        # container instead of keeping it as a grid dim (e.g. an execution.random_seed
        # trial ensemble → a mean/sem observation). Carried through into the axis
        # metadata so ExplorationResult knows which named dim to reduce, and how.
        _reduce = getattr(axis, 'reduce', None)
        _reduce_stat = (str(getattr(_reduce, 'statistic', None) or 'mean')
                        if _reduce is not None else None)
        # element_domains satisfy the axis whether they carry explored_values OR
        # lo/hi/n bounds — the hetero auto-expansion below reads either per element.
        assert domain or explored_values or _el_domains or _builder is not None or from_experiment_branch, \
            (f"exploration axis requires domain, explored_values, element_domains, or builder for "
             f"{axis.parameter} (or initial_state source_point='branch', which supplies the axis values)")
        pname = str(axis.parameter)
        # Dotted reference (== the Exploration.space key); the ExplorationResult axis
        # label uses this so grid coords are named consistently across backends, while
        # the grid state path below uses the bare `pname`.
        _axis_label = pname
        # Check for dotted notation: ClassName.param_name
        # If prefix matches a coupling key → coupling param, else dynamics param
        source_key = None
        is_coupling_param = False
        is_network_param = False
        graph_leaf = None
        if pname.startswith('network.'):
            # `network` is the reserved singleton-network scope (one Network per
            # experiment): `network.conduction_speed`, `network.edges.<label>`.
            # Split on the FIRST dot so the remainder stays a full attribute path
            # — rsplit would turn `network.edges.weight` into prefix
            # 'network.edges', which no longer matches the scope and would fall
            # through to the dynamics branch as a silent wrong-scope write.
            # network_axis_leaf() resolves the path to the graph leaf the axis
            # sweeps (and raises on an unsweepable attribute).
            is_network_param = True
            graph_leaf = network_axis_leaf(pname)
            # The grid path is the resolved leaf, so `pname` only names the axis'
            # `n_<name>` override kwarg; sanitize the attribute path into an
            # identifier (`edges.weight` -> `edges_weight`, leaving the existing
            # `conduction_speed` untouched). `_axis_label` keeps the declared
            # dotted path, so grid coords stay named as written in the recipe.
            pname = re.sub(r'\W', '_', pname[len('network.'):])
        elif '.' in pname:
            prefix, pname = pname.rsplit('.', 1)
            is_coupling_param = (prefix in all_couplings)
            source_key = _to_ci_key(prefix) if is_coupling_param else prefix
        # from_experiment:branch axis — the swept-parameter VALUES come from the source run's
        # recorded branch (loaded at runtime as _BRANCH_SEED), not from a domain here. This axis
        # carries only its identity (parameter path); the branch-analysis body binds each cell's
        # value + settled state. Keeps the recipe's single source of truth for the branch: the
        # source exploration, so the analysis restarts on exactly the points that were computed.
        if from_experiment_branch and not (domain or explored_values or _el_domains or _builder is not None):
            exp_info['axes'].append({
                'name': pname,
                'label': _axis_label,
                'is_coupling': is_coupling_param,
                'coupling_key': source_key if is_coupling_param else None,
                'dynamics_key': source_key if (not is_coupling_param and source_key) else None,
                'element_idx': None,
                'is_branch': True,
                'reduce': _reduce_stat,
            })
            continue
        # Builder axis: values materialized at runtime by a callable (ExplorationAxis.builder) —
        # e.g. per-count control-gain vectors chosen from a runtime solitary-node ordering. The
        # callable returns the stacked sweep values (leading axis = points); each value may be a
        # whole vector. Routed through the normal grid path as a DataAxis (Space gathers array-
        # valued axes per cell), so it inherits sharding / batching / as_grid with no special
        # path. Bypasses the hetero auto-expansion below: the builder supplies whole per-node
        # vectors, not per-element scalars. Arguments resolve to a literal or a base-sim
        # observation (`observations.<name>` -> `_bov('<name>')`, wired in the exploration fn).
        if _builder is not None:
            _bc = getattr(_builder, 'callable', None)
            assert _bc is not None and getattr(_bc, 'module', None) and getattr(_bc, 'name', None), \
                f"builder for exploration axis '{axis.parameter}' requires callable: {{name, module}}"
            import json as _json
            _arg_strs = []
            for _an, _arg in (_builder.arguments.items() if getattr(_builder, 'arguments', None) else []):
                _av = _arg.value if hasattr(_arg, 'value') else _arg
                if isinstance(_av, str) and 'observations.' in _av:
                    _arg_strs.append("%s=_bov(%r)" % (str(_an), _av.split('observations.', 1)[1]))
                else:
                    try:
                        _lit = repr(_json.loads(_json.dumps(_av)))   # coerce JsonObj -> plain literal
                    except Exception:
                        _lit = repr(_av)
                    _arg_strs.append("%s=%s" % (str(_an), _lit))
            exp_info['axes'].append({
                'name': pname,
                'label': _axis_label,
                'is_coupling': is_coupling_param,
                'is_network': is_network_param,
                'graph_leaf': graph_leaf,
                'coupling_key': source_key if is_coupling_param else None,
                'dynamics_key': source_key if not is_coupling_param and source_key else None,
                'element_idx': None,
                'builder_expr': "%s.%s(%s)" % (_bc.module, _bc.name, ", ".join(_arg_strs)),
                'reduce': _reduce_stat,
            })
            continue
        # `execution.random_seed` → a per-cell NOISE-SEED axis. Each grid cell reseeds
        # the stochastic solver's PRNG key (config.noise.key, a runtime leaf), so a
        # random-seed sweep becomes a real per-trial noise ensemble rather than a no-op
        # parameter. Values are integer seeds; the wrapper below turns each into a key.
        if source_key == 'execution' and pname == 'random_seed':
            if explored_values:
                _seed_vals = [int(v) for v in explored_values]
            else:
                _slo = int(domain.lo) if (domain and domain.lo is not None) else 0
                _seed_vals = [_slo + _i for _i in range(_resolve_n(domain))]
            exp_info['axes'].append({
                'name': 'random_seed',
                'label': _axis_label,
                'is_seed': True,
                'is_coupling': False,
                'coupling_key': None,
                'element_idx': None,
                'values': _seed_vals,
                'n': len(_seed_vals),
                'reduce': _reduce_stat,
            })
            continue
        # Auto-expand heterogeneous parameters: if pname matches a dynamics param
        # with shape containing 'n_nodes', expand to n_nodes element axes automatically.
        # e.g., K with shape "(n_nodes,)" → K_el0, K_el1, ... K_el(n_nodes-1)
        # A `network.`-scoped axis is never a dynamics parameter, so it must not be
        # expanded here even when a Dynamics happens to declare a same-named
        # per-node parameter — the axis sweeps the graph leaf, not the model.
        is_hetero_param = (not is_coupling_param and not is_network_param
                           and pname in dyn_param_shapes
                           and 'n_nodes' in dyn_param_shapes[pname])
        if is_hetero_param:
            _n = n_nodes
            _el_domains = getattr(axis, 'element_domains', None) or []
            # Build element→domain lookup: keyed by Range.element if set, else positional
            _el_dom_map = {}
            for _edi, _ed in enumerate(_el_domains):
                _ek = getattr(_ed, 'element', None)
                _ek = _ek if _ek is not None else _edi
                _el_dom_map[_ek] = _ed
            for _ei in range(_n):
                ax_entry = {
                    'name': pname,
                    'label': _axis_label,
                    'is_coupling': False,
                    'coupling_key': None,
                    'dynamics_key': source_key if source_key else None,
                    'element_idx': _ei,
                    'reduce': _reduce_stat,
                }
                # Per-element explored_values from element_domains override shared ones
                _dom_ei = _el_dom_map.get(_ei, None)
                _el_ev = getattr(_dom_ei, 'explored_values', None) if _dom_ei else None
                if _el_ev:
                    vals = [float(v) for v in _el_ev]
                    ax_entry['values'] = vals
                    ax_entry['n'] = len(vals)
                elif explored_values:
                    vals = [float(v) for v in explored_values]
                    ax_entry['values'] = vals
                    ax_entry['n'] = len(vals)
                else:
                    # Use per-element domain if available (by element key), else shared domain
                    _dom = _el_dom_map.get(_ei, domain)
                    assert _dom.lo is not None, f"domain.lo required for {axis.parameter}[{_ei}]"
                    assert _dom.hi is not None, f"domain.hi required for {axis.parameter}[{_ei}]"
                    n = _resolve_n(_dom)
                    ax_entry['lo'] = float(_dom.lo)
                    ax_entry['hi'] = float(_dom.hi)
                    ax_entry['n'] = n
                exp_info['axes'].append(ax_entry)
        else:
            if explored_values:
                vals = [float(v) for v in explored_values]
                exp_info['axes'].append({
                    'name': pname,
                    'label': _axis_label,
                    'values': vals,
                    'n': len(vals),
                    'is_coupling': is_coupling_param,
                    'is_network': is_network_param,
                    'graph_leaf': graph_leaf,
                    'coupling_key': source_key if is_coupling_param else None,
                    'dynamics_key': source_key if not is_coupling_param and source_key else None,
                    'element_idx': None,
                    'reduce': _reduce_stat,
                })
            else:
                assert domain.lo is not None, f"domain.lo required for {axis.parameter}"
                assert domain.hi is not None, f"domain.hi required for {axis.parameter}"
                n = _resolve_n(domain)
                if getattr(domain, 'log_scale', False):
                    # Honor log_scale: emit explicit log-spaced values (DataAxis).
                    import numpy as _np
                    _lo, _hi = float(domain.lo), float(domain.hi)
                    assert _lo > 0, f"log_scale requires domain.lo > 0 for {axis.parameter}"
                    _vals = [float(v) for v in _np.logspace(_np.log10(_lo), _np.log10(_hi), n)]
                    exp_info['axes'].append({
                        'name': pname,
                        'label': _axis_label,
                        'values': _vals,
                        'n': n,
                        'is_coupling': is_coupling_param,
                        'is_network': is_network_param,
                        'graph_leaf': graph_leaf,
                        'coupling_key': source_key if is_coupling_param else None,
                        'dynamics_key': source_key if not is_coupling_param and source_key else None,
                        'element_idx': None,
                        'reduce': _reduce_stat,
                    })
                else:
                    exp_info['axes'].append({
                        'name': pname,
                        'label': _axis_label,
                        'lo': float(domain.lo),
                        'hi': float(domain.hi),
                        'n': n,
                        'is_coupling': is_coupling_param,
                        'is_network': is_network_param,
                        'graph_leaf': graph_leaf,
                        'coupling_key': source_key if is_coupling_param else None,
                        'dynamics_key': source_key if (not is_coupling_param and not is_network_param and source_key) else None,
                        'element_idx': None,
                        'reduce': _reduce_stat,
                    })
    observable = expl.observable
    if observable:
        # FunctionCall: function attribute references the function
        func = observable.function
        func_name = func.name if hasattr(func, 'name') else str(func) if func else None
        args = observable.arguments or {}

        if args:
            # FunctionCall with arguments (e.g., rmse(fc.data, target)). arguments is a
            # dict keyed by name; the key IS the argument name.
            exp_info['observable_type'] = 'function_call'
            exp_info['observable_func'] = func_name
            exp_info['observable_args'] = []
            for arg_name, arg in args.items():
                arg_value = arg.value if hasattr(arg, 'value') else None
                if arg_value:
                    # Value references observation.output (e.g., "fc.data")
                    if '.' in str(arg_value):
                        obs_ref, output_key = str(arg_value).split('.', 1)
                        exp_info['observable_args'].append({'name': arg_name, 'obs': obs_ref, 'key': output_key})
                    else:
                        exp_info['observable_args'].append({'name': arg_name, 'obs': str(arg_value), 'key': 'data'})
                else:
                    # No value = runtime input (target_data)
                    exp_info['observable_args'].append({'name': arg_name, 'obs': None, 'key': None})
        else:
            # Simple observation reference (function: obs_name, no arguments)
            exp_info['observable_type'] = 'observation'
            exp_info['observable'] = func_name
            exp_info['output_key'] = get_pipeline_output_key(func_name) if func_name else None

    # Algorithms explicitly wired to this exploration (Exploration.algorithms).
    # Each runs AT EACH sweep point (sequentially) before the observable is
    # computed — e.g. FIC re-tuning J_i at every E/I ratio. Fully declarative:
    # name + n_iterations + hyperparameters are read from experiment.algorithms.
    exp_info['algorithms'] = []
    _exp_algos = dict(experiment.algorithms.items()) if experiment.algorithms else {}
    for _alg_name in (list(expl.algorithms) if getattr(expl, 'algorithms', None) else []):
        _alg = _exp_algos.get(_alg_name) or _exp_algos.get(safe_name(_alg_name))
        assert _alg is not None, f"exploration '{exp_info['name']}' wires unknown algorithm '{_alg_name}'"
        _hp = {}
        for _h in (getattr(_alg, 'hyperparameters', None) or []):
            _hp[str(_h.name)] = float(_h.value) if getattr(_h, 'value', None) is not None else 0.0
        _nit = getattr(_alg, 'n_iterations', None)
        assert _nit is not None, f"algorithm '{_alg_name}' missing n_iterations"
        # External inputs the run-func requires (same classification as the flat path,
        # ~2919-2934) so the exploration call site can forward them:
        #   input_names        - observations with a data_source (file)
        #   network_obs_inputs - network.observations.* / dataset.subject.*, bound as
        #                        module-level network-observation globals
        _alg_inp, _alg_netobs = [], []
        for _on in get_all_observations_from_algo(_alg, _exp_algos):
            _od = observations_dict.get(_on)
            if _od is None:
                continue
            if getattr(_od, 'data_source', None) is not None:
                _alg_inp.append(_on)
                continue
            _s = getattr(_od, 'source', None)
            if isinstance(_s, (list, tuple)):
                _s = _s[0] if _s else None
            if _s is not None and hasattr(_s, 'name'):
                _s = _s.name
            if _s and (str(_s).startswith('network.observations.')
                       or str(_s).startswith('dataset.subject')):
                _alg_netobs.append(_on)
        exp_info['algorithms'].append({
            'name': safe_name(_alg_name),
            'n_iterations': int(_nit),
            'hyperparams': _hp,
            'input_names': _alg_inp,
            'network_obs_inputs': _alg_netobs,
        })

    # Search strategy: 'grid' (default, exhaustive) or 'nsga2' (pymoo multi-objective).
    exp_info['strategy'] = str(getattr(expl, 'strategy', None) or 'grid')
    exp_info['objectives'] = [str(o) for o in (getattr(expl, 'objectives', None) or [])]
    # Warm-start controls (orthogonal to the search strategy): from_previous seeds each point
    # from the preceding point's settled state — a quasi-static branch-following sweep rendered
    # onto the tvboptim adiabatic_scan primitive; sweep_direction sets the traversal order.
    exp_info['sweep_seeding'] = str(getattr(expl, 'sweep_seeding', None) or 'independent')
    exp_info['sweep_direction'] = str(getattr(expl, 'sweep_direction', None) or 'up')
    if exp_info['strategy'] == 'nsga2':
        assert exp_info['objectives'], f"nsga2 exploration '{exp_info['name']}' requires objectives"
        # Resolve each decision axis to a tvboptim state path (+ optional log10 decode).
        _nsga_axes = []
        for _axis in axes_list:
            _apn = str(_axis.parameter); _apref = None
            if '.' in _apn:
                _apref, _apn = _apn.rsplit('.', 1)
            _adom = _axis.domain
            assert _adom is not None and _adom.lo is not None and _adom.hi is not None, \
                f"nsga2 axis '{_axis.parameter}' requires domain lo/hi"
            if _apref and _apref in all_couplings:
                _apath = f"coupling.{_to_ci_key(_apref)}.{_apn}"
            elif _apref in ('noise', 'AdditiveNoise', 'Noise'):
                _apath = f"noise.{_apn}"
            else:
                _apath = f"dynamics.{_apn}"
            _nsga_axes.append({
                'path': _apath, 'lo': float(_adom.lo), 'hi': float(_adom.hi),
                'transform': str(getattr(_axis, 'transform', None) or 'none'),
            })
        exp_info['nsga2_axes'] = _nsga_axes
        # GA hyperparameters, keyed by name in Exploration.parameters.
        _default_pop = n_workers if 'n_workers' in dir() else 8
        _ga = {'population_size': _default_pop, 'num_generations': 40, 'seed': 42,
               'reference_point': [1.0e6] * len(exp_info['objectives'])}
        for _gpn, _gpv in iter_parameter_values(expl.parameters):
            if _gpn == 'reference_point':
                _ga['reference_point'] = [float(v) for v in _gpv]
            elif _gpn in ('population_size', 'num_generations', 'seed'):
                _ga[_gpn] = int(_gpv)
            else:
                _ga[_gpn] = float(_gpv)
        exp_info['ga'] = _ga
        # Parallelism for the per-generation candidate evaluation (pmap devices),
        # baked as a literal like the refine stage's n_workers.
        exp_info['n_workers'] = n_workers if 'n_workers' in dir() else 1
    elif exp_info['strategy'] == 'adiabatic_scan':
        # Adiabatic bifurcation scan: one swept axis (from `space`) plus an observed-signal
        # expression and envelope settings carried on Exploration.parameters (Parameter.value
        # accepts strings and numbers). Delegates to tvboptim's adiabatic_scan at codegen.
        assert exp_info['axes'], f"adiabatic_scan exploration '{exp_info['name']}' requires one space axis"
        from tvbo.templates.tvboptim.utils import get_recorded_variable_names as _grvn_adia
        _ap = dict(iter_parameter_values(expl.parameters))
        _asig = _ap.get('signal')
        assert _asig, f"adiabatic_scan '{exp_info['name']}' requires a 'signal' parameter (e.g. signal: {{value: 'y1 - y2'}})"
        _, _, _adia_vars = _grvn_adia(model, experiment)
        exp_info['adiabatic'] = {
            'axis': exp_info['axes'][0],
            'signal_code': render_adiabatic_signal(str(_asig), _adia_vars),
            'segment_time': float(_ap.get('segment_time', 2000.0)),
            'skip': float(_ap.get('transient_time', 1000.0)),
            'bothways': bool(int(float(_ap.get('bothways', 1)))),
        }
        # adiabatic_scan is a preset over the orthogonal warm-start controls: a from_previous
        # sweep whose traversal is bidirectional when bothways is set and whose observable is the
        # signal envelope. Normalise onto sweep_seeding/sweep_direction so the shared warm-start
        # partial (sweep.warmstart_sweep_body) drives it — one renderer, one runtime primitive.
        exp_info['sweep_seeding'] = 'from_previous'
        exp_info['sweep_direction'] = 'bidirectional' if exp_info['adiabatic']['bothways'] else 'up'
    # General record-based warm-start (sweep_seeding=from_previous, no envelope preset): each
    # recorded observation must be a single-source, single-step trajectory reduction — it
    # becomes an adiabatic_scan statistic over the settled rollout of its source state variable.
    # The reduction callables are backend-agnostic, so they run under adiabatic_scan's jit; the
    # per-point rollout / transient come from the experiment integration.
    if exp_info['sweep_seeding'] == 'from_previous' and 'adiabatic' not in exp_info:
        assert len(exp_info['axes']) == 1, \
            f"warm-start '{exp_info['name']}' requires exactly one swept axis"
        assert exp_info['record'], \
            f"warm-start '{exp_info['name']}' (sweep_seeding=from_previous) requires record: [...]"
        _ws_records = []
        _ws_analysis = []
        for _rn in exp_info['record']:
            _obs = _all_observations.get(_rn)
            assert _obs is not None, \
                f"warm-start '{exp_info['name']}' records unknown observation '{_rn}'"
            # Analysis observations (Lyapunov, ...) ride the warm-start scan as a post-scan
            # pass: each swept value's carried settled state seeds the analysis solve, so
            # lambda_1 / xi_i are measured on the continued branch (unreachable from a cold
            # start). Resolved here to backend-agnostic metadata; the sweep partial lays out
            # the per-value map.
            if _rn in analysis_observation_names:
                _ws_analysis.append(_lyap_meta(_rn, f"warm-start '{exp_info['name']}'"))
                continue
            _src = [str(s) for s in (getattr(_obs, 'source', None) or [])]
            _pipe = list(getattr(_obs, 'pipeline', None) or [])
            _cal = getattr(_pipe[0], 'callable', None) if _pipe else None
            assert len(_src) == 1 and len(_pipe) == 1 and _cal is not None and getattr(_cal, 'module', None), \
                (f"warm-start record '{_rn}' must be a single-source, single-callable trajectory "
                 "reduction with a module, or an analysis observation (analysis rides the scan as a "
                 "post-scan pass; multi-source pipeline observations are not supported on the scan)")
            _ws_records.append({'name': _rn, 'call': "%s.%s" % (_cal.module, _cal.name),
                                'var_idx': var_names.index(_src[0])})
        exp_info['warmstart_records'] = _ws_records
        exp_info['warmstart_analysis'] = _ws_analysis
        exp_info['warmstart_segment'] = float(getattr(experiment.integration, 'duration', 1000.0))
        exp_info['warmstart_skip'] = float(getattr(experiment.integration, 'transient_time', 0.0) or 0.0)
    # from_experiment:branch restart — an independent exploration over the source run's recorded
    # branch: each cell restarts a per-point analysis (Lyapunov) from that point's swept value +
    # settled state. Independent per cell, so it shards across HPC array tasks (unlike the
    # sequential scan that produced the branch). Reuses the warm-start Lyapunov post-scan pass,
    # just sourcing (value, seed) from the loaded branch instead of an in-process scan.
    if from_experiment_branch:
        exp_info['branch_seed'] = True
        assert len(exp_info['axes']) == 1 and exp_info['axes'][0].get('is_branch'), \
            (f"branch exploration '{exp_info['name']}' requires exactly one axis whose values are "
             "supplied by the from_experiment branch (declare `space: [{parameter: <the source's "
             "swept parameter>}]` with no domain)")
        assert exp_info['record'], \
            f"branch exploration '{exp_info['name']}' requires record: [...] (the analysis to restart)"
        _b_analysis = []
        for _rn in exp_info['record']:
            assert _rn in analysis_observation_names, \
                (f"branch exploration '{exp_info['name']}' records '{_rn}'; only analysis observations "
                 "(e.g. lyapunov) can be restarted per branch point. Trajectory-reduction observations "
                 "belong on the source scan that produced the branch.")
            _b_analysis.append(_lyap_meta(_rn, f"branch exploration '{exp_info['name']}'"))
        exp_info['warmstart_analysis'] = _b_analysis
    explorations.append(exp_info)

# Optimizations that depend on an Exploration front → per-seed parallel refinement.
# Build the refine metadata (seed axes from the exploration, free-param paths from the
# optimization) so render_refine can emit the ParallelExecution-over-the-front body.
import math as _math
refine_infos = {}
_expl_by_name = {e['name']: e for e in explorations}
for _opt in optim_list:
    _do = getattr(_opt, 'depends_on', None)
    _do_name = safe_name(str(_do)) if _do else None
    _expl = _expl_by_name.get(_do_name) if _do_name else None
    if not _expl or _expl.get('strategy') != 'nsga2':
        continue
    _seed_axes = [{'path': ax['path'], 'transform': ax['transform'], 'col': _i}
                  for _i, ax in enumerate(_expl['nsga2_axes'])]
    _seed_paths = set(ax['path'] for ax in _seed_axes)
    _fps = []
    for _fp in (_opt.free_parameters or []):
        _fpn = str(_fp.parameter); _fpref = None
        if '.' in _fpn:
            _fpref, _fpn = _fpn.rsplit('.', 1)
        if _fpref and _fpref in all_couplings:
            _fpath = f"coupling.{_to_ci_key(_fpref)}.{_fpn}"
        elif _fpref in ('noise', 'AdditiveNoise', 'Noise'):
            _fpath = f"noise.{_fpn}"
        else:
            _fpath = f"dynamics.{_fpn}"
        _dom = getattr(_fp, 'domain', None)
        def _bnd(v):
            if v is None:
                return None
            fv = float(v)
            return 'jnp.inf' if _math.isinf(fv) and fv > 0 else ('-jnp.inf' if _math.isinf(fv) else fv)
        _fps.append({
            'name': _fpn, 'path': _fpath,
            'hetero': bool(getattr(_fp, 'heterogeneous', False)),
            'lo': _bnd(_dom.lo) if _dom else None,
            'hi': _bnd(_dom.hi) if _dom else None,
            'seeded': _fpath in _seed_paths,
        })
    # Metric observations from the loss arguments (fc-correlation and freq-gradient terms).
    _loss_obs = [str(k) for k in ((getattr(_opt, 'loss', None) and _opt.loss.arguments) or {}).keys()]
    _fc_obs = next((o for o in _loss_obs if 'fc' in o.lower()), _loss_obs[0] if _loss_obs else 'fc_corr_val')
    _freq_obs = next((o for o in _loss_obs if 'freq' in o.lower() or 'grad' in o.lower()),
                     _loss_obs[-1] if _loss_obs else 'freq_grad_corr')
    refine_infos[safe_name(_opt.name)] = {
        'name': safe_name(_opt.name), 'exploration': _expl['name'],
        'seed_axes': _seed_axes, 'free_params': _fps, 'objectives': list(_expl['objectives']),
        'optimizer': optimization_stages[0]['algorithm'] if optimization_stages else 'adam',
        'learning_rate': optimization_stages[0]['learning_rate'] if optimization_stages else 0.001,
        'max_steps': optimization_stages[0]['max_iterations'] if optimization_stages else 200,
        'opt_mode': opt_mode, 'n_nodes': n_nodes, 'n_workers': n_workers,
        'fc_obs': _fc_obs, 'freq_obs': _freq_obs,
    }
# Search-family codegen flags. `has_nsga2` gates the pymoo import + the nsga2 partial;
# a refine optimization (depends_on an Exploration front) replaces the standard
# single-state stage loop with a per-seed parallel sweep over the Pareto front.
has_nsga2 = any(e.get('strategy') == 'nsga2' for e in explorations)
has_warmstart = any(e.get('sweep_seeding') == 'from_previous' for e in explorations)
has_refine = len(refine_infos) > 0
refine_info = list(refine_infos.values())[0] if refine_infos else None

has_observations = len(observations) > 0

# Parse observations - these only have source (state variable), no derived observations
def _first_source_name(obs):
    src = getattr(obs, 'source', None)
    if not src:
        return None
    if isinstance(src, (list, tuple)):
        src = src[0] if src else None
    if src is None:
        return None
    return src.name if hasattr(src, 'name') and src.name else str(src)

obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': obs.label or '',
        'description': obs.description or '',
        'source': _first_source_name(obs),
        'equation': obs.equation.rhs if obs.equation else None,
    }
    obs_list.append(obs_info)

# Collect modules to import from derived observation pipelines
derived_obs_modules = set()
for dobs_name, dobs in derived_observations_dict.items():
    if dobs.pipeline:
        for stage in dobs.pipeline:
            c = getattr(stage, 'callable', None)
            if c:
                call_module = getattr(c, 'module', None)
                if call_module:
                    derived_obs_modules.add(call_module)

# First coupling name for docstring
first_coupling_name = list(all_couplings.keys())[0] if all_couplings else 'None'
%>
"""${dynamics_class} tvboptim Experiment."""
import os
import copy
import functools  # render_expression emits functools.reduce for Min/Max over lists
import logging

# Progress is logged, not printed: this generated script shares the ``tvbo``
# logger hierarchy, so ``TVBO_LOG_LEVEL`` / ``tvbo.set_log_level`` control it the
# same way in-process and standalone (see the ``__main__`` block below).
logger = logging.getLogger("tvbo.run")

import jax
% if enable_x64:
jax.config.update("jax_enable_x64", True)  # Required for stable gradient computation
% endif
import jax.numpy as jnp
import jax.scipy.signal
import equinox as eqx
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling, DelayedCoupling
% if has_stimulus_events:
from tvboptim.experimental.network_dynamics.external_input.base import AbstractExternalInput
% endif
% if has_delay:
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseLengthGraph, SparseDelayGraph
% else:
from tvboptim.experimental.network_dynamics.graph import DenseGraph, SparseGraph
% endif
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}, BoundedSolver
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif

% if has_optimization:
import optax
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, SavingLossCallback, SavingParametersCallback
from tvbo.templates.tvboptim.callbacks import LoggingProgressCallback
% endif
% if has_explorations:
from tvboptim.types import Space, GridAxis, DataAxis
from tvboptim.execution import ParallelExecution, SequentialExecution
from tvbo.templates.tvboptim.callbacks import progress_ticker   # grid-batch / lyapunov sweep progress
% endif
% if has_nsga2:
# Multi-objective search (Exploration.strategy == 'nsga2') + Pareto-seeded refinement.
import numpy as _np
from pymoo.core.problem import Problem as _Problem
from pymoo.algorithms.moo.nsga2 import NSGA2 as _NSGA2
from pymoo.optimize import minimize as _pymoo_minimize
from pymoo.indicators.hv import HV as _HV
% endif
% if has_warmstart or from_working_point:
# Warm-started (from_previous) parameter sweep — shared runtime primitive. Also used by
# initial_state.from_working_point to ramp to a working point and seed the run's IC.
from tvboptim.experimental.network_dynamics.analysis import adiabatic_scan as _adiabatic_scan
% endif
% for mod in sorted(derived_obs_modules):
import ${mod}
% endfor

# Result classes from tvbo
from tvbo.data.types import SimulationResult, AlgorithmResult, OptimizationResult, ExplorationResult
% if has_inference:
from tvbo.data.types import InferenceResult
import numpyro
import numpyro.distributions as dist
% endif
% if has_explorations:
from tvbo.data.types import _stacked_to_dataarray as _stacked_to_dataarray
% endif

% if stochastic_param_info:

def _inject_stochastic_trajectories(state, t1, dt, key=None):
    """Pre-generate stochastic parameter trajectories and inject into state.dynamics.

    Pre-generated arrays are indexed per integration step inside dynamics(),
    avoiding per-step RNG calls. Pure jnp.ndarray — safe for vmap/pmap.
    """
    if key is None:
        key = jax.random.key(0)
    n_steps = int(t1 / dt) + 2  # +2 for rounding safety
    % for sp_name, sp_info in stochastic_param_info.items():
<%
    # Determine noise shape: (n_steps,) for scalar, (n_steps, n_nodes) for per-node
    _sp_shape = sp_info.get('shape', '')
    if _sp_shape and 'n_nodes' in _sp_shape:
        _noise_shape = '(n_steps, n_nodes)'
    else:
        _noise_shape = '(n_steps,)'
    # Effective mean/std: explicit distribution parameters win; the domain
    # (mean<-value, std<-(hi-lo)/4) is only the fallback.
    _mu = sp_info['mean'] if sp_info.get('mean') is not None else sp_info['default']
    _sigma = sp_info['std'] if sp_info.get('std') is not None else (sp_info['hi'] - sp_info['lo']) / 4.0
%>\
    key, _subkey = jax.random.split(key)
    % if sp_info['dist'] == 'uniform':
    state.dynamics._stoch_${sp_name} = jax.random.uniform(_subkey, ${_noise_shape}, minval=${sp_info['lo']}, maxval=${sp_info['hi']})
    % elif sp_info['dist'] in ('gaussian', 'normal'):
    state.dynamics._stoch_${sp_name} = ${_mu} + ${_sigma} * jax.random.normal(_subkey, ${_noise_shape})
    % elif sp_info['dist'] in ('truncated_normal', 'truncatednormal'):
    _raw = jax.random.truncated_normal(_subkey, lower=${(sp_info['lo'] - _mu) / max(_sigma, 1e-6)}, upper=${(sp_info['hi'] - _mu) / max(_sigma, 1e-6)}, shape=${_noise_shape})
    state.dynamics._stoch_${sp_name} = ${_mu} + ${_sigma} * _raw
    % else:
    # Unsupported distribution '${sp_info['dist']}', using uniform fallback
    state.dynamics._stoch_${sp_name} = jax.random.uniform(_subkey, ${_noise_shape}, minval=${sp_info['lo']}, maxval=${sp_info['hi']})
    % endif
    % endfor
    return state

% endif

% if stochastic_param_info or has_stochastic_stimulus:
def _freeze_step_time(solver):
    """Patch solver to freeze t for all sub-evaluations within a step.

    Multi-stage solvers (RK4, Heun) evaluate dynamics at sub-step times
    (t, t+dt/2, t+dt). Time-indexed stochastic inputs (pre-generated arrays
    indexed by t) should be constant per integration step — the input is
    sampled once per step, not interpolated across sub-steps. This covers
    both stochastic dynamics params and iid per-step stimulus events (whose
    external-input compute also reads the frozen step time).

    This patches the solver's step method so all dynamics evaluations within
    a single step see the same time value (the step-start time t), preventing
    the k4 evaluation at t+dt from reading the next noise sample.
    """
    _original_step = solver.step

    def _frozen_step(dynamics_fn, t, state, dt, params, noise_sample=0.0):
        def frozen_dynamics(t_sub, state_sub, params_sub):
            return dynamics_fn(t, state_sub, params_sub)
        return _original_step(frozen_dynamics, t, state, dt, params, noise_sample)

    solver.step = _frozen_step
    return solver

% endif

def get_solver(block_size=None):
    """Configured solver. ``block_size`` (native solvers only) sets the nested-block-scan
    granularity so a streaming reduction (``prepare(reduce=...)``) folds the observable
    in-carry instead of materializing the trajectory; ``None`` keeps the single scan."""
    _solver_kwargs = dict(${solver_kwargs_str})
    if block_size is not None:
        _solver_kwargs['block_size'] = block_size
    base_solver = ${solver_class}(**_solver_kwargs)
% if has_state_bounds:
    solver = BoundedSolver(
        base_solver,
        low=jnp.array(${state_bounds_lo_str})[:, None],
        high=jnp.array(${state_bounds_hi_str})[:, None]
    )
% else:
    solver = base_solver
% endif
% if stochastic_param_info or has_stochastic_stimulus:
    solver = _freeze_step_time(solver)
% endif
    return solver

<%include file="/tvboptim/tvbo-tvboptim-dfun.py.mako" />

## Bind the dynamics class to an alias now, before the coupling classes are
## defined: a coupling may share the model's name (e.g. TVB's ``Linear`` model
## and ``Linear`` coupling), and the later ``class Linear(...Coupling)`` would
## otherwise shadow the model class so ``dynamics = Linear(**model_params)``
## would wrongly instantiate the coupling.
_TVBO_DYNAMICS_CLS = ${dynamics_class}

<%include file="tvbo-tvboptim-cfun.py.mako" />

% if has_stimulus_events:
<%include file="tvbo-tvboptim-stimulus.py.mako" />
% endif


def create_network(
    weights: jnp.ndarray,
    % if use_length_graph:
    distances: jnp.ndarray = None,
    % elif use_delay_graph:
    delays: jnp.ndarray = None,
    % endif
    region_labels: list = None,
    dynamics_params: dict = None,
    coupling_params: dict = None,
    noise_sigma: float = ${noise_sigma_value},
    % if has_delay:
    max_delay: float = None,
    % endif
) -> Network:
% if has_weight_transforms:
    # Weight transforms
    W = weights
    W_min, W_max = jnp.min(W), jnp.max(W)
    M = W
    M_min, M_max = W_min, W_max
    % for expr in weight_transform_jax:
    weights = ${expr}
    W = weights
    M = W
    W_min, W_max = jnp.min(W), jnp.max(W)
    M_min, M_max = W_min, W_max
    % endfor
% endif

    % if use_length_graph:
    # delays = lengths / speed each forward pass, so speed stays a live graph leaf.
    # max_delay_bound sizes the static history buffer (default: the build-speed max delay).
    if distances is None:
        distances = jnp.zeros_like(weights)
    _speed = ${conduction_speed}
    _max_delay_bound = max_delay if max_delay is not None else (float(jnp.max(distances)) / _speed if _speed > 0 else 0.0)
    graph = DenseLengthGraph(weights, distances, speed=_speed, region_labels=region_labels, max_delay_bound=_max_delay_bound)
    % elif use_delay_graph:
    # Per-edge delays used directly; non-edge entries arrive as NaN, so zero-fill first.
    if delays is None:
        delays = jnp.zeros_like(weights)
    delays = jnp.nan_to_num(delays)
    graph = DenseDelayGraph(weights, delays, region_labels=region_labels, max_delay_bound=max_delay)
    % elif use_sparse:
    # Sparse coupling: the weight matrix is sparsified to BCOO so the vectorized
    # `pre @ weights` reduction is an O(nnz) edge-sum, not a dense NxN matmul.
    graph = SparseGraph(weights, region_labels=region_labels)
    % else:
    graph = DenseGraph(weights, region_labels=region_labels)
    % endif

    n_nodes = weights.shape[0]

    _dynamics_params = {
        % for name in dyn_param_names:
        % if name in node_param_overrides:
        '${name}': jnp.array([${', '.join(str(v) for v in node_param_overrides[name])}]),
        % elif name in dyn_param_shapes:
        '${name}': jnp.full(${dyn_param_shapes[name]}, ${dyn_param_defaults.get(name, 1.0)}),
        % else:
        '${name}': ${render_jax_default(dyn_param_defaults.get(name, 1.0))},
        % endif
        % endfor
    }
    if dynamics_params:
        _dynamics_params.update(dynamics_params)
    dynamics = _TVBO_DYNAMICS_CLS(**_dynamics_params)

    coupling_dict = {}

    ## Build coupling parameter dicts per function
    % for coupling_key, coupling_obj in all_couplings.items():
<%
    # Class name = coupling key (cleaned), same as in cfun template
    c_class_name = coupling_key.replace(' ', '').replace('-', '')
    c_param_names, c_param_defaults, c_param_shapes = get_param_info(coupling_obj.parameters if hasattr(coupling_obj, 'parameters') else None)
%>
    _${coupling_key}_params = {
        % for name in c_param_names:
        % if name in c_param_shapes:
        '${name}': jnp.full(${c_param_shapes[name]}, ${c_param_defaults.get(name, 1.0)}),
        % else:
        '${name}': ${render_jax_default(c_param_defaults.get(name, 1.0))},
        % endif
        % endfor
    }
    if coupling_params and '${coupling_key}' in coupling_params:
        _${coupling_key}_params.update(coupling_params['${coupling_key}'])
    % endfor

    ## Assign coupling instances keyed by coupling_input name (tvboptim requirement)
    % for ci_name, (func_name, _ci_obj) in ci_coupling_map.items():
<%  c_class_name = func_name.replace(' ', '').replace('-', '') %>\
    coupling_dict['${ci_name}'] = ${c_class_name}(**dict(_${func_name}_params))
    % endfor

    % if has_noise:
    % if noise_targets:
    noise = AdditiveNoise(sigma=noise_sigma, apply_to=${noise_targets}, key=jax.random.key(${random_seed})) if noise_sigma > 0 else None
    % else:
    noise = AdditiveNoise(sigma=noise_sigma, key=jax.random.key(${random_seed})) if noise_sigma > 0 else None
    % endif
    % else:
    noise = None
    % endif

    % if has_stimulus_events:
    external_input = {
        % for ev in stimulus_events:
        '${ev.name}': ${ev.name}Input(),
        % endfor
    }
    % endif

    return Network(
        dynamics=dynamics,
        coupling=coupling_dict,
        graph=graph,
        noise=noise,
        % if has_stimulus_events:
        external_input=external_input,
        % endif
    )

% if sv_distribution_info:
def _sample_initial_conditions(state, key=None):
    """Sample initial conditions from state variable distributions.

    Each state variable with a ``distribution`` slot is replaced per-node by a
    draw from that distribution. The default RNG seed is taken from the
    distribution's ``seed`` slot (falls back to 42).
    """
    if key is None:
        key = jax.random.key(${list(sv_distribution_info.values())[0]['seed']})
    _keys = jax.random.split(key, ${len(sv_distribution_info)})
    ic = state.initial_state.dynamics  # (n_states,) broadcast or (n_states, n_nodes)
    # Ensure per-node shape so sampling produces independent values per node
    if ic.ndim == 1:
        ic = jnp.broadcast_to(ic[:, None], (ic.shape[0], n_nodes)).copy()
% for _si, (_sv_name, _sv_info) in enumerate(sv_distribution_info.items()):
% if _sv_info['dist'] in ('gaussian', 'normal'):
    ic = ic.at[${_sv_info['idx']}].set(${(_sv_info['lo'] + _sv_info['hi']) / 2} + ${(_sv_info['hi'] - _sv_info['lo']) / 4.0} * jax.random.normal(_keys[${_si}], (n_nodes,)))
% else:
    ic = ic.at[${_sv_info['idx']}].set(jax.random.uniform(_keys[${_si}], (n_nodes,), minval=${_sv_info['lo']}, maxval=${_sv_info['hi']}))
% endif
% endfor
    state.initial_state.dynamics = ic
    return state
% endif

% if network_observation_names:
# ── Network observations (empirical targets carried by the Network) ──────────
# Declared in YAML via `source: [network.observations.<measure>]`. The name->
# measure mapping is resolved in Python (SimulationExperiment.
# network_observation_measures) and passed in as `network_obs_measures`;
# values are materialized at run_experiment() time from the network (or a
# `network_observations` override).
_NETWORK_OBS_MEASURES = {${', '.join("'%s': '%s'" % (k, v) for k, v in network_obs_measures.items())}}
% for _on in sorted(network_observation_names):
${_on} = None  # network observation <- ${network_obs_measures[_on]}
% endfor

def _bind_network_observations(network_observations=None):
    """Materialize module-level network-observation constants from the given
    dict (keyed by observation name). Mirrors how `weights`/`distances` flow
    into the experiment; raises a clear error if a declared one is missing."""
    network_observations = network_observations or {}
% for _on in sorted(network_observation_names):
    global ${_on}
    if '${_on}' in network_observations and network_observations['${_on}'] is not None:
        ${_on} = jnp.asarray(network_observations['${_on}'])
    if ${_on} is None:
        raise ValueError(
            "Network observation '${_on}' (measure '${network_obs_measures[_on]}') "
            "was not provided. Pass it via "
            "run_experiment(network_observations={'${_on}': <matrix>}), or ensure "
            "the network supplies observational_measures=['${network_obs_measures[_on]}']."
        )
% endfor

% endif
<% _ds_recon_idx = context.get('dataset_reconcile_indices') or {} %>
% if _ds_recon_idx:
# ── Node reconciliation (keyed by label, never positional) ───────────────────
# Positions of the labels shared between the model network and each by_label
# empirical target. The loss gathers the simulated observable onto these nodes so
# it aligns, label for label, with the reconciled target.
_DATASET_RECON_IDX = {
% for _tname, _idx in _ds_recon_idx.items():
    '${_tname}': jnp.array(${_idx}),
% endfor
}

def _gather2d(matrix, idx):
    """Select the shared nodes on both axes of a (node, node) matrix by index."""
    return matrix[jnp.ix_(idx, idx)]
% endif
# ── Initial-condition construction (shared across every IC site) ─────────────
# The transient, the main run, and each exploration base build their IC the same
# way — sampled defaults, then per-node declared overrides, then an optional
# externally supplied operating point — so a sweep's points start from the same
# declared state as the main run rather than a cold sample. Every per-variable
# override is applied BY NAME through _STATE_INDEX (never by positional order).
_STATE_INDEX = {${', '.join("'%s': %d" % (n, i) for i, n in enumerate(state_names))}}

def _set_rows(state, name_to_vals):
    """Set per-node values of named state variables in initial_state.dynamics
    ([n_states, n_nodes]), placing each by its canonical row index _STATE_INDEX[name]."""
    for _name, _vals in name_to_vals.items():
        state.initial_state.dynamics = state.initial_state.dynamics.at[_STATE_INDEX[_name]].set(jnp.asarray(_vals))
    return state

# Per-node initial state declared via node ``state:`` YAML entries, keyed by name.
% if node_state_overrides:
_NODE_STATE_OVERRIDES = {
% for sv_name, sv_vals in node_state_overrides.items():
    '${sv_name}': jnp.array([${', '.join(str(v) for v in sv_vals)}]),
% endfor
}
% else:
_NODE_STATE_OVERRIDES = {}
% endif

def _apply_node_overrides(state):
    return _set_rows(state, _NODE_STATE_OVERRIDES)

# Runtime IC seed (InitialState.from_experiment): the settled operating point another
# experiment already reached, keyed by state-variable name -> (n_nodes,), handed in as
# run_experiment(seed_dynamics=...). None (the default) is a no-op.
_SEED_DYNAMICS = None

def _apply_seed_dynamics(state):
    return state if _SEED_DYNAMICS is None else _set_rows(state, _SEED_DYNAMICS)

# Runtime BRANCH seed (InitialState.from_experiment, source_point='branch'): the source run's
# whole recorded branch — {axis_name, axis_values (n_cells,), seeds {sv: (n_cells, n_nodes)},
# n_cells} — handed in as run_experiment(branch_seed=...). A branch-restart exploration reads
# it to build per-cell (swept value, settled state) pairs. None (the default) is a no-op.
_BRANCH_SEED = None

# Runtime PARAMETER seed (from_experiment Parameter.measure): model-parameter values
# loaded from the source run's operating point — a recorded observation (e.g. a control
# mask g) or a tuned free parameter (estimate__<param>, e.g. wLRE/wFFI). Keyed by
# parameter name; per-node vectors AND per-edge matrices. Handed in as
# run_experiment(seed_params=...).
_SEED_PARAMS = None
<%
    _seed_coupling_home = {}
    for _ck, _cobj in (all_couplings or {}).items():
        _cparams = getattr(_cobj, "parameters", None) or {}
        _pnames = list(_cparams.keys()) if hasattr(_cparams, "keys") \
            else [getattr(_p, "name", None) for _p in _cparams]
        for _pn in _pnames:
            if _pn:
                _seed_coupling_home[_pn] = _ck
%>\
% if _seed_coupling_home:

# Coupling home for each seedable coupling parameter (dynamics params route to dynamics).
_SEED_PARAM_COUPLING = {
% for _pn, _ck in _seed_coupling_home.items():
    ${repr(_pn)}: ${repr(_ck)},
% endfor
}

def _apply_seed_params(state):
    if _SEED_PARAMS is not None:
        for _name, _vals in _SEED_PARAMS.items():
            _v = jnp.asarray(_vals)
            _ck = _SEED_PARAM_COUPLING.get(_name)
            if _ck is not None:
                setattr(state.coupling[_ck], _name, _v)   # per-edge coupling param (e.g. wLRE)
            else:
                state.dynamics[_name] = _v                 # per-node dynamics param (e.g. g)
    return state
% else:

def _apply_seed_params(state):
    if _SEED_PARAMS is not None:
        for _name, _vals in _SEED_PARAMS.items():
            state.dynamics[_name] = jnp.asarray(_vals)
    return state
% endif

def run_simulation(
    network: Network,
    t1: float = ${t1_default},
    dt: float = ${dt},
    t0: float = 0.0,
    t_transient: float = ${transient_time},
    run_main: bool = True,
    **kwargs,
) -> Bunch:
    solver = get_solver()
    result_transient = None
    % if has_noise:
    # config.noise.key is a live runtime PRNG leaf (tvboptim reads it per solve), so honour
    # a runtime random_seed — run(random_seed=N) or a trial ensemble — instead of the
    # codegen-baked key. Unset → the experiment's execution.random_seed (${random_seed}),
    # i.e. byte-identical to the previously baked jax.random.key(${random_seed}).
    # jnp.asarray (not int()) so an array- or tracer-valued seed — a trial ensemble
    # binding one seed per cell — coerces instead of raising.
    _rs = kwargs.get('random_seed')
    _noise_key = jax.random.key(jnp.asarray(${random_seed} if _rs is None else _rs, dtype=jnp.uint32))
    % endif

    % if has_transient:
    # Run transient simulation to settle network dynamics
    if t_transient > 0:
        model_fn_init, state_init = prepare(network, solver, t0=t0, t1=t_transient, dt=dt)
        % if sv_distribution_info:
        # Sample initial conditions from state variable distributions
        state_init = _sample_initial_conditions(state_init)
        % endif
        # Per-node declared overrides, then an optional from_experiment seed
        state_init = _apply_seed_params(_apply_seed_dynamics(_apply_node_overrides(state_init)))
        % if stochastic_param_info:
        _inject_stochastic_trajectories(state_init, t_transient, dt, key=jax.random.key(${list(stochastic_param_info.values())[0]['seed']}))
        % endif
        % if has_noise:
        if getattr(state_init, 'noise', None) is not None:
            state_init.noise.key = _noise_key
        % endif
        result_transient = model_fn_init(state_init)
        # tvboptim >= 0.2.7: NativeSolution carries variable_names; update_history
        # slices state columns by name, so we can hand the solution over directly.
        network.update_history(result_transient)
    % endif

    # Main sim chains onto the transient: solver runs from t=t_transient to
    # t=t_transient + t1, so its time coord continues where the transient left
    # off. The caller still passes t1 as the main-sim duration.
    model_fn, state = prepare(network, solver, t0=t0 + t_transient, t1=t0 + t_transient + t1, dt=dt)
    % if sv_distribution_info:
    # Sample initial conditions from state variable distributions
    state = _sample_initial_conditions(state)
    % endif
    # Per-node initial state overrides (from node ``state:`` YAML entries)
    state = _apply_node_overrides(state)
    % if from_working_point:
    # initial_state.from_working_point: reach the operating branch by ramping the parameter
    # quasi-statically 0→target (warm-start, from_previous), then seed this run's IC from the
    # settled endpoint — not the cold IC above. Reuses the adiabatic_scan warm-start primitive.
    _wp_scan = _adiabatic_scan(
        network, solver,
        accessor=lambda _c: _c.${from_working_point['path']},
        low=${from_working_point['lo']}, high=${from_working_point['hi']}, n=${from_working_point['n']},
        t=${from_working_point['settle']}, skip=0, dt=dt, bothways=False,
        observe=lambda _r: _r.ys,
        statistics={'_endpoint': (lambda _a: _a[-1])},
    )
    state.initial_state.dynamics = jnp.asarray(_wp_scan.stats['_endpoint'])[-1]
    % endif
    # from_experiment: a supplied operating point is the final word on the main IC
    # (when there is no transient; with one, the transient's settled state below wins)
    state = _apply_seed_dynamics(state)
    state = _apply_seed_params(state)
    % if stochastic_param_info:
    _inject_stochastic_trajectories(state, t1, dt, key=jax.random.key(${list(stochastic_param_info.values())[0]['seed']}))
    % endif

    % if has_transient:
    # Initialize state variables from end of transient (settled dynamics)
    if result_transient is not None:
        _final = result_transient.data[-1]  # (n_states,) or (n_states, n_nodes)
        % for i, sv_name in enumerate(state_names):
        state.dynamics.${sv_name} = _final[${i}]
        % endfor
    % endif

    result = None
    observations = None
    if run_main:
        % if has_noise:
        if getattr(state, 'noise', None) is not None:
            state.noise.key = _noise_key
        % endif
        result = model_fn(state)
        observations = Bunch()
% for obs_name in observation_names:
% if obs_name in network_observation_names:
        observations.${obs_name} = ${obs_name}
% elif obs_name in derived_observation_names:
% else:
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
        observations.${obs_name} = ${obs_class}(history=result_transient)(result)
% endif
% endfor

        # Compute derived observations
        _all_obs = compute_all_observations(result, state, result_transient)
% for obs_name in sorted(derived_observation_names):
        observations.${obs_name} = _all_obs.${obs_name}
% endfor

        # Analysis observations (operate on the solve/loss, not result.data)
% if analysis_observations_dict:
        for _an_name, _an_val in compute_analysis_observations(state, network, result_transient).items():
            observations[_an_name] = _an_val
% endif

    return Bunch(
        model_fn=model_fn,
        state=state,
        result=result,
        result_transient=result_transient,
        observations=observations,
    )

<%include file="tvbo-tvboptim-observation.py.mako" />

<%
from tvbo.codegen import render_expression

# Schema: experiment.functions is multivalued dict
exp_funcs = dict(experiment.functions) if experiment.functions else {}

# Collect all function names for user_functions mapping
all_func_names = {str(fname): str(fname) for fname in exp_funcs.keys()}

def is_simple_callable(fdef, fname):
    """Check if function is a simple callable (just import, no wrapper needed).

    Simple = has callable + no apply_on_dimension + no equation + no source_code.
    Argument defaults in YAML are just documentation, not code to generate.
    """
    if not fdef.callable:
        return False
    # No apply_on_dimension (needs vmap wrapper)
    if fdef.apply_on_dimension:
        return False
    # No equation (hybrid callable+equation not supported as simple)
    if fdef.equation:
        return False
    # No source_code
    if fdef.source_code:
        return False
    return True

# Classify callables: simple (just import) vs complex (need wrapper)
simple_callable_imports = {}  # {(module, cname): fname} - direct import
complex_callable_imports = {}  # {(module, cname): _callable_cname} - prefixed import
funcs_needing_def = []  # Functions that need actual definition

for fname, fdef in exp_funcs.items():
    fname = str(fname)
    c = fdef.callable
    if c:
        module = c.module
        cname = c.name or c.qualname
        if module and cname:
            if is_simple_callable(fdef, fname):
                # Just import directly as the function name
                simple_callable_imports[(module, cname)] = fname
            else:
                # Need wrapper, import with prefix
                complex_callable_imports[(module, cname)] = f"_callable_{cname}"
                funcs_needing_def.append((fname, fdef))
    else:
        funcs_needing_def.append((fname, fdef))
%>
# Simple callable imports (direct, no wrapper needed)
% for (module, cname), local_name in sorted(simple_callable_imports.items()):
% if local_name != cname:
from ${module} import ${cname} as ${local_name}
% else:
from ${module} import ${cname}
% endif
% endfor

# Complex callable imports (prefixed, wrapper will be generated)
% for (module, cname), local_name in sorted(complex_callable_imports.items()):
from ${module} import ${cname} as ${local_name}
% endfor

# User-defined functions (generated via base function-def.mako)
% for fname, fdef in funcs_needing_def:
${fn.function_def(fdef, format='jax', user_functions=all_func_names)}
% endfor

<%
# Pre-compute loss function info for inline generation
loss_functions = [parse_loss_function(opt) for opt in optim_list]
loss_functions = [lf for lf in loss_functions if lf]

runtime_kwargs_needed = set()
for lf in loss_functions:
    for arg in lf['args']:
        if arg['type'] == 'runtime':
            runtime_kwargs_needed.add(arg['kwarg_name'])

# Pre-compute observation categorization for loss function
if loss_functions:
    _lf = loss_functions[0]
    _lf_func_name = _lf['func_name']
    _lf_args = _lf['args']
    _lf_obs_refs = _lf['obs_refs']
    _lf_agg_over = _lf['agg_over']
    _lf_agg_type = _lf['agg_type']
    _lf_agg_axis = 0 if _lf_agg_over == 'node' else (1 if _lf_agg_over == 'time' else None)
    _lf_agg_func = {'mean': 'mean', 'sum': 'sum', 'max': 'max', 'min': 'min'}.get(_lf_agg_type, 'mean')

    _lf_simulated_obs = [o for o in _lf_obs_refs if o in observation_names and o not in network_observation_names and o not in derived_observation_names]
    _lf_derived_obs = [o for o in _lf_obs_refs if o in derived_observation_names]

    _lf_source_obs_for_derived = set()
    _lf_derived_info = {}
    for dobs_name in _lf_derived_obs:
        dobs_def = derived_observations_dict.get(dobs_name)
        if dobs_def:
            sources = []
            for src in [_s for _s in (dobs_def.source or []) if (getattr(_s, 'name', None) or _s) in _all_observations]:
                src_name = str(src) if not hasattr(src, 'name') else str(src.name)
                sources.append(src_name)
                if src_name in observation_names and src_name not in network_observation_names and src_name not in derived_observation_names:
                    _lf_source_obs_for_derived.add(src_name)
            pipeline_call = None
            pipeline_args = []
            if dobs_def.pipeline:
                first_stage = dobs_def.pipeline[0]
                c = getattr(first_stage, 'callable', None)
                if c:
                    call_module = getattr(c, 'module', None)
                    call_name = getattr(c, 'name', None) or getattr(c, 'qualname', None)
                    if call_module and call_name:
                        pipeline_call = f"{call_module}.{call_name}"
                if hasattr(first_stage, 'arguments') and first_stage.arguments:
                    for arg_name, arg in first_stage.arguments.items():
                        arg_value = getattr(arg, 'value', None)
                        if arg_name and arg_value is not None:
                            pipeline_args.append((arg_name, arg_value))
            _lf_derived_info[dobs_name] = {
                'sources': sources,
                'callable': pipeline_call,
                'args': pipeline_args
            }

    _lf_all_simulated = sorted(set(_lf_simulated_obs) | _lf_source_obs_for_derived)
else:
    _lf_all_simulated = []
    _lf_derived_obs = []
    _lf_derived_info = {}
    _lf_func_name = None
    _lf_args = []
    _lf_agg_over = None
    _lf_agg_axis = None
    _lf_agg_func = 'mean'
%>

<%
# Observation dependency ordering lives in the tvboptim adapter (utils), harmonized
# with the derived-variable/parameter dependency graph — the template only calls it.
# toposort_observations emits any observation that lists another as a `source` AFTER
# that source; independents keep their input order.
from tvbo.templates.tvboptim.utils import toposort_observations

sorted_observation_names = list(observation_names)
sorted_derived_obs_names = toposort_observations(sorted(derived_observation_names), derived_observations_dict, _all_observations)
%>

def _obs_data(_o):
    """Underlying array of an observation value. Monitor results wrap the array in
    ``.data``; a bare array (numpy/jax — has ``.dtype``) is returned as-is, since its
    own ``.data`` would be a raw buffer, not the array."""
    return _o if hasattr(_o, 'dtype') else getattr(_o, 'data', _o)


def _windowed_corr(_reduce, _ts, **_kw):
    """Guard a windowed correlation reducer (e.g. compute_fc) against a degenerate
    window. Pearson correlation is undefined over fewer than two retained
    timepoints, where jnp.corrcoef collapses to a 0-d scalar that then crashes the
    diagonal write. A window like this arises when a derived FC observation is
    materialized on a short simulation (e.g. one BOLD sample) — the value is not
    meaningful there, so return a NaN (n, n) matrix instead of aborting the whole
    observation pipeline. Windows with >= 2 retained samples delegate to ``_reduce``
    unchanged, so full FC stays byte-identical."""
    if _ts.shape[0] - int(_kw.get('skip_t', 0)) < 2:
        _n = _ts.shape[-1]
        return jnp.full((_n, _n), jnp.nan).at[jnp.diag_indices(_n)].set(0)
    return _reduce(_ts, **_kw)


def compute_all_observations(result, state, result_transient=None, only=None, network_obs=None):
    # ``only`` (a set of observation names) restricts computation to those observations
    # plus, by the caller's closure, whatever they derive from. Default None computes
    # every declared observation (unchanged behaviour). The jitted per-grid-point
    # observable passes it so non-recorded observations — e.g. a non-jittable
    # ``solitary`` needed only by the base run — never execute inside the trace.
    obs = Bunch()

    # Network observations (empirical targets carried by the Network). A
    # `network_obs` entry wins over the module-level constant, so a caller
    # scoring against its own target — one subject of a batched cohort, whose
    # target is a traced leaf rather than a process-wide value — is not
    # silently scored against whatever `_bind_network_observations` last bound.
    _no = network_obs or {}
% for obs_name in sorted(network_observation_names):
    obs.${obs_name} = _no['${obs_name}'] if '${obs_name}' in _no else ${obs_name}
% endfor

    # Simulated observations (computed from result) - these derive from simulation state
% for obs_name in sorted_observation_names:
<%
    if obs_name in network_observation_names:
        continue  # Skip network observations, already handled above
    if obs_name in derived_observation_names:
        continue  # Derived observations are emitted in the dedicated loop below.

    obs_def = observations_dict.get(obs_name)
    has_pipeline = obs_def and obs_def.pipeline if obs_def else False

    # Regular observations derive from simulation state (via source attribute)
    # They do NOT have source_observation - that's only for DerivedObservation
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))

    # Get pipeline info
    pipeline_call = None
    if has_pipeline:
        first_stage = obs_def.pipeline[0] if obs_def.pipeline else None
        if first_stage:
            c = getattr(first_stage, 'callable', None)
            if c:
                call_module = getattr(c, 'module', None)
                call_name = getattr(c, 'name', None) or getattr(c, 'qualname', None)
                if call_module and call_name:
                    pipeline_call = f"{call_module}.{call_name}"
            else:
                fname = getattr(first_stage, 'function', None) or getattr(first_stage, 'name', None)
                pipeline_call = str(fname) if fname else None
%>
% if obs_name not in network_observation_names:
    if only is None or '${obs_name}' in only:
        # ${obs_name}: observation derived from simulation state
        _${obs_name}_monitor = ${obs_class}(history=result_transient)
        _${obs_name}_result = _${obs_name}_monitor(result)
        # Keep full result to preserve named outputs (e.g., .psd, .frequencies)
        obs.${obs_name} = _${obs_name}_result
% endif
% endfor

    # Derived observations (from derived_observations in schema)
% for dobs_name, dobs in derived_observations_dict.items():
<%
    # Source names of this derived observation, filtered to entries that
    # name another observation in the experiment.
    src_obs_list = []
    for so in (dobs.source or []):
        _so_name = str(so) if not hasattr(so, 'name') else str(so.name)
        if _so_name in _all_observations:
            src_obs_list.append(_so_name)

    # Get pipeline callable
    pipeline_call = None
    pipeline_equation = None       # equation-based derived obs (rhs over other observations)
    pipeline_equation_params = {}  # local equation constants
    pipeline_args = []
    positional_args = []  # Track positional args from source_observations
    if dobs.pipeline:
        first_stage = dobs.pipeline[0]
        c = getattr(first_stage, 'callable', None)
        if c:
            call_module = getattr(c, 'module', None)
            call_name = getattr(c, 'name', None) or getattr(c, 'qualname', None)
            if call_module and call_name:
                pipeline_call = f"{call_module}.{call_name}"
        if pipeline_call is None:
            # function-based derived observation: a YAML-defined function rendered
            # as a module-level helper. Preferred over library callables — it is
            # backend-independent (each backend renders the same function).
            _fn = getattr(first_stage, 'function', None)
            if _fn is not None:
                pipeline_call = str(_fn) if not hasattr(_fn, 'name') else str(_fn.name)
        if pipeline_call is None:
            # equation-based derived observation: an `equation` over other
            # observations (no callable/function). Rendered inline below via
            # render_expression, with each source observation bound to a local.
            _eq = getattr(first_stage, 'equation', None)
            if _eq is not None:
                pipeline_equation = getattr(_eq, 'rhs', None)
                pipeline_equation_params = dict(iter_parameter_values(getattr(_eq, 'parameters', None)))
        # Extract arguments from pipeline stage (callable/function path only; the
        # equation path binds its source observations directly at emit time below).
        if pipeline_call and hasattr(first_stage, 'arguments') and first_stage.arguments:
            for arg_name, arg in first_stage.arguments.items():
                arg_value = getattr(arg, 'value', None)
                # Only include arguments that have explicit values (not just names/descriptions)
                if arg_name and arg_value is not None:
                    val_str = str(arg_value)
                    # Check if value is an observation reference vs a literal
                    if val_str in src_obs_list or val_str in observation_names or val_str in derived_observation_names:
                        # Simple observation reference → its data array. Observations are stored
                        # as the full monitor result (a NativeSolution, to keep named outputs like
                        # .psd); a plain positional reference wants the underlying array, so unwrap
                        # `.data` (no-op when it is already a bare array). Dotted references below
                        # keep the named-output attribute instead.
                        positional_args.append(f"_obs_data(obs.{val_str})")
                    elif val_str.replace('.', '').replace('-', '').isdigit():
                        # Numeric literal - use as keyword arg
                        pipeline_args.append(f"{arg_name}={val_str}")
                    elif val_str.startswith('network.') and (_edge_lab := edge_label(val_str)):
                        # network.weight(s)/length(s) or network.edges.<label> → the
                        # connectome matrix embedded as a module constant by the included
                        # observation template (utils.collect_network_edge_arrays), NOT a
                        # string literal. Keeps derived observations (source = another
                        # observation) consistent with the non-derived source path.
                        pipeline_args.append(f"{arg_name}={edge_const(_edge_lab)}")
                    elif val_str.startswith('network.') and (_node_lab := node_label(val_str)):
                        # network.positions/instrength → the per-node vector embedded as a
                        # module constant by the included observation template
                        # (utils.collect_network_node_arrays), the node-level analogue of
                        # the edge-matrix branch above.
                        pipeline_args.append(f"{arg_name}={node_const(_node_lab)}")
                    elif '.' in val_str:
                        prefix = val_str.split('.')[0]
                        if prefix in (src_obs_list + list(observation_names) + list(derived_observation_names)):
                            # Dotted observation reference (e.g., avg_spectrum.avg_psd) - add as keyword
                            pipeline_args.append(f"{arg_name}=obs.{val_str}")
                        else:
                            # Unknown dotted reference - pass as string
                            pipeline_args.append(f"{arg_name}='{val_str}'")
                    else:
                        # String literal or other - use as keyword arg
                        pipeline_args.append(f"{arg_name}='{val_str}'" if isinstance(arg_value, str) else f"{arg_name}={val_str}")
        # If no explicit args were parsed, use source_observations positionally
        if pipeline_call and not positional_args and not pipeline_args:
            positional_args = [f"_obs_data(obs.{s})" for s in src_obs_list]

    # Build final args: positional first, then keyword
    all_args = positional_args + pipeline_args
    _args = ', '.join(all_args)
    # A windowed correlation reducer is undefined over a <2-sample window; route it
    # through _windowed_corr so a degenerate window (e.g. a derived FC materialized on
    # a single-BOLD-sample simulation) returns NaN instead of crashing. The FC-family
    # set is the streaming-reducer registry (shared with the streaming path), not a
    # hand-maintained list. Every other reducer emits a plain call, byte-identical.
    from tvbo.codegen.streaming_reducers import is_windowed_reducer
    _reduce_mod, _, _reduce_name = pipeline_call.rpartition('.') if pipeline_call else ('', '', '')
    _pipeline_emit = (
        f"_windowed_corr({pipeline_call}, {_args})"
        if pipeline_call and is_windowed_reducer(_reduce_mod or None, _reduce_name)
        else f"{pipeline_call}({_args})"
    )
%>
% if pipeline_call and src_obs_list:
    # ${dobs_name}: derived from ${', '.join(src_obs_list)}
    if (only is None or '${dobs_name}' in only) and all(hasattr(obs, _src) for _src in [${', '.join(f"'{s}'" for s in src_obs_list)}]):
        obs.${dobs_name} = ${_pipeline_emit}
% elif pipeline_equation and src_obs_list:
    # ${dobs_name}: equation over ${', '.join(src_obs_list)}
    if (only is None or '${dobs_name}' in only) and all(hasattr(obs, _src) for _src in [${', '.join(f"'{s}'" for s in src_obs_list)}]):
% for _src in src_obs_list:
        ${_src} = _obs_data(obs.${_src})
% endfor
        obs.${dobs_name} = ${jaxcode(pipeline_equation, pipeline_equation_params)}
% endif
% endfor

    return obs


% if analysis_observations_dict:
% if has_lyapunov:
${lyap.benettin_function()}

% endif
def compute_analysis_observations(state, network, result_transient=None):
    """Compute the declarative ``analysis`` observations — diagnostics that ANALYZE the
    solve/loss (Lyapunov spectrum, autodiff and finite-difference gradients) rather than
    transforming ``result.data``. Factored out so the main run and any exploration that
    records diagnostics per grid point share one implementation, keeping a G-sweep of the
    diagnostics fully metadata-derived. Analysis solves use a plain solver — the truncation
    window is an optimization knob, not part of these diagnostics."""
    obs = Bunch()
<%
    # A tuning algorithm with an activity-target objective (Deco FIC) defines the operating point by
    # a CONSTRAINT: the objective's target_variable = target_value, with the update rule's
    # target_parameter free. For the LINEAR-RESPONSE analysis obs (which linearise around the
    # deterministic fixed point), solve that constraint deterministically — the paper's fsolve on the
    # steady state — rather than the stochastic tuning loop. `None` => plain noise-off settle. The
    # stochastic algorithm itself is untouched (still used for the actual BOLD/rate simulations).
    _op_constraint = None
    for _alg in (experiment.algorithms.values() if experiment.algorithms else []):
        _obj = getattr(_alg, 'objective', None)
        _rules = list(getattr(_alg, 'update_rules', None) or [])
        if _obj is not None and getattr(_obj, 'target_variable', None) is not None and getattr(_obj, 'target_value', None) is not None and _rules:
            _tp = _rules[0].target_parameter
            _op_constraint = {
                'constraint_variable': str(_obj.target_variable),
                'target': float(_obj.target_value),
                'free_parameter': str(getattr(_tp, 'name', _tp)),
            }
            break
%>
${render_analysis_observations(analysis_observations_dict, coupling_keys, solver_class, transient_time, t1_default, dt, solver_kwargs_str, model=model, time_si_factor=time_si_factor, events=(dict(experiment.events) if experiment.events else {}), op_constraint=_op_constraint)}
    return obs
% endif


<%include file="tvbo-tvboptim-algorithm.py.mako" />


% if has_optimization:
<%
# Build a lookup dict for all known parameters (dynamics + coupling)
all_dynamics_params = {str(p.name): p for p in optim_params}
# For coupling params, store (param, coupling_key) so we know where to access them
all_coupling_params = {str(p.name): (p, getattr(p, '_coupling_key', first_coupling_key)) for p in optim_coupling_params} if first_coupling_key else {}
%>

def unwrap_all_parameters(state):
    """Convert all Parameter objects to plain values (freeze all)."""
    import jax.tree_util as jtu
    def unwrap(x):
        if isinstance(x, Parameter):
            return x.value
        return x
    return jtu.tree_map(unwrap, state, is_leaf=lambda x: isinstance(x, Parameter))


% for stage_idx, stage in enumerate(optimization_stages):
<%
stage_name = stage['name']
stage_free_params = stage['free_parameters']
stage_lr = stage['learning_rate']
stage_max_iter = stage['max_iterations']
stage_algorithm = stage['algorithm']
stage_hyperparams = stage['hyperparameters']
stage_warmup_from = stage['warmup_from']
%>

def mark_parameters_${stage_name}(state, n_nodes: int = ${n_nodes}):
    """Mark free parameters: ${', '.join(p['name'] for p in stage_free_params)}."""
    init_state = unwrap_all_parameters(copy.deepcopy(state))
% for fp in stage_free_params:
<%
fp_name = fp['name']
fp_hetero = fp['heterogeneous']
fp_shape = fp.get('shape', None)
fp_lo = fp.get('lower_bound', None)
fp_hi = fp.get('upper_bound', None)
has_bounds = fp_lo is not None or fp_hi is not None
# Optimizer start value (FreeParameter.initial_value): if given, the marked Parameter
# wraps this value instead of the base config's, so the descent begins from the declared
# point (e.g. G_START) while the base/warm-up config keeps its own value.
fp_init = fp.get('initial_value', None)
# Coupling key is explicitly set via dotted notation (e.g., FastLinearCoupling.G)
# Translate function name to ci name for tvboptim state access
coupling_key_for_param = fp.get('coupling_key', None)
if coupling_key_for_param:
    coupling_key_for_param = _to_ci_key(coupling_key_for_param)
is_coupling = coupling_key_for_param is not None
# Format bounds for code generation (None -> jnp.inf)
lo_str = f'{fp_lo}' if fp_lo is not None else '-jnp.inf'
hi_str = f'{fp_hi}' if fp_hi is not None else 'jnp.inf'
# Convert shape string to Python tuple (e.g., "(n_nodes, n_nodes)" -> (n_nodes, n_nodes))
# If shape is None, default to (n_nodes,) for heterogeneous params
if fp_shape:
    shape_str = fp_shape.strip('()').replace(' ', '')
    shape_code = '(' + shape_str + (',' if ',' not in shape_str else '') + ')'
else:
    shape_code = '(n_nodes,)'
c_wrap = f"jnp.asarray({fp_init})" if fp_init is not None else f"init_state.coupling.{coupling_key_for_param}.{fp_name}"
d_wrap = f"jnp.asarray({fp_init})" if fp_init is not None else f"init_state.dynamics.{fp_name}"
%>
% if is_coupling:
    # ${fp_name} - coupling parameter (${coupling_key_for_param})${ ' (bounded: ' + str(fp_lo) + ' to ' + str(fp_hi) + ')' if has_bounds else ''}
% if has_bounds:
    init_state.coupling.${coupling_key_for_param}.${fp_name} = BoundedParameter(
        ${c_wrap},
        low=${lo_str},
        high=${hi_str},
    )
% else:
    init_state.coupling.${coupling_key_for_param}.${fp_name} = Parameter(${c_wrap})
% endif
% if fp_hetero:
    init_state.coupling.${coupling_key_for_param}.${fp_name}.shape = ${shape_code}
% endif
% else:
    # ${fp_name} - dynamics parameter${ ' (bounded: ' + str(fp_lo) + ' to ' + str(fp_hi) + ')' if has_bounds else ''}
% if has_bounds:
    init_state.dynamics.${fp_name} = BoundedParameter(
        ${d_wrap},
        low=${lo_str},
        high=${hi_str},
    )
% else:
    init_state.dynamics.${fp_name} = Parameter(${d_wrap})
% endif
% if fp_hetero:
    init_state.dynamics.${fp_name}.shape = ${shape_code}
% endif
% endif
% endfor

    return init_state


def run_stage_${stage_name}(
    init_state,
    loss_fn,
    max_steps: int = ${stage_max_iter},
    learning_rate: float = ${stage_lr},
    **kwargs,
):
    """Run optimization stage: ${stage_name} (${stage_algorithm}, lr=${stage_lr})."""
    marked_state = mark_parameters_${stage_name}(init_state)
    opt_kwargs = {**kwargs}
% for hp_name, hp_value in stage_hyperparams.items():
    opt_kwargs.setdefault('${hp_name}', ${hp_value})
% endfor

    opt = create_optimizer(
        loss_fn,
        optimizer="${stage_algorithm}",
        learning_rate=learning_rate,
        max_steps=max_steps,
        **opt_kwargs
    )
    fitted_params, fitting_data = opt.run(marked_state, max_steps=max_steps, mode="${opt_mode}")
    return fitted_params, fitting_data

% endfor

def _smart_interval(n):
    """Compute smart interval: 1 for 0-10, 10 for 10-100, 100 for 100-1000, etc."""
    if n <= 10:
        return 1
    return 10 ** (len(str(n)) - 2)

def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    max_steps: int = ${max_steps},
    callback = None,
    print_every: int = None,
    save_every: int = None,
    **opt_kwargs,
):
    """Create configured optax optimizer."""
    optimizers = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "adamax": optax.adamax,
        "adamaxw": optax.adamaxw,
        "sgd": optax.sgd,
    }
    opt_fn = optimizers.get(optimizer, optax.adamaxw)

    # Build optimizer kwargs (hyperparameters like b1, b2)
    optimizer_kwargs = {**opt_kwargs}
% if optimizer_hyperparams:
    # Default hyperparameters from YAML
% for hp_name, hp_value in optimizer_hyperparams.items():
    optimizer_kwargs.setdefault('${hp_name}', ${hp_value})
% endfor
% endif

    # Smart defaults for callback intervals based on max_steps
    if print_every is None:
        print_every = _smart_interval(max_steps)
    if save_every is None:
        save_every = _smart_interval(max_steps)

    # Default callback: log progress + save loss + save state at smart intervals
    if callback is None:
        callback = MultiCallback([
            LoggingProgressCallback(every=print_every, total=max_steps),
            SavingLossCallback(every=save_every),
            SavingParametersCallback(every=save_every),
        ])

    return OptaxOptimizer(loss_fn, opt_fn(learning_rate, **optimizer_kwargs), callback=callback, has_aux=False)


def run_optimization(
    init_state,
    loss_fn,
    max_steps: int = ${max_steps},
    learning_rate: float = ${learning_rate},
    optimizer: str = "${optimizer_name}",
    callback = None,
    print_every: int = None,
    save_every: int = None,
    **kwargs,
):
    """Run gradient-based optimization."""
    opt = create_optimizer(
        loss_fn, optimizer=optimizer, learning_rate=learning_rate,
        max_steps=max_steps, callback=callback, print_every=print_every,
        save_every=save_every, **kwargs
    )
    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps, mode="${opt_mode}")
    return fitted_params, fitting_data
% endif


% if has_explorations:

% for expl in explorations:
<%
    total_points = 1
    for ax in expl['axes']:
        total_points *= (ax.get('n') or 1)   # builder axes have a runtime-only size (unknown here)
    has_axes = len(expl['axes']) > 0
    obs_type = expl.get('observable_type', 'observation')
    obs_func = expl.get('observable_func', '')
    obs_args = expl.get('observable_args', [])
    obs_name = expl.get('observable', '')
    output_key = expl.get('output_key')
    grid_desc = ' x '.join([f"{ax['name']}[{ax.get('n', '?')}]" for ax in expl['axes']]) if has_axes else f"{expl.get('n_trials', 1)} trials"
    # When the YAML declares observations, the JIT'd observable_fn returns
    # only the reduced observation values (no trajectory). This supersedes
    # the legacy model-output extraction path (which would have returned a
    # sliced trajectory) — the model.output declaration is only used for
    # model-output extraction when no observations are declared.
    bundles_observations = (
        obs_type != 'function_call'
        and not obs_name
        and bool(observation_names or derived_observation_names)
    )
    # Streaming-reduction fast path: when every recorded (non-analysis) observable is an
    # Observation.dynamics observer, fold each into the integrator carry via
    # prepare(reduce=...) so the trajectory is never materialized (peak memory drops from
    # O(batch·n_time·n_node) to O(batch·block·n_node)). This IS the memory win for the
    # trial-ensemble sweeps: the streaming observable is the base that the IC-trial vmap
    # maps over, so each of the N trials streams instead of materializing its trajectory.
    # A transient is handled inside the stream (skip=n_transient over the full window), so
    # no trim wrapper is needed. Element-slot axes, stochastic noise injection, and wired
    # algorithms still need the trajectory model_fn, so any of them forces the post-scan path.
    _rec_stream = [r for r in expl['record'] if r not in analysis_observation_names]
    _all_recorded_streaming = bool(_rec_stream) and all(
        resolve_reduction(_all_observations.get(r)) is not None for r in _rec_stream
    )
    _element_axes_present = any(ax.get('element_idx') is not None for ax in expl['axes'])
    _seed_axis_present = any(ax.get('is_seed') for ax in expl['axes'])
    _use_stream = (
        bundles_observations
        and _all_recorded_streaming
        and not stochastic_param_info
        and not _element_axes_present
        and not _seed_axis_present
        and not expl.get('algorithms')
    )
    _stream_names = _rec_stream
    _stream_bs = expl.get('block_size') or 1000
    _stream_skip = int(round(transient_time / dt)) if has_transient else 0
    _stream_t1 = (transient_time + t1_default) if has_transient else t1_default
    # Network-scope axes (e.g. `network.conduction_speed`): the base graph is a
    # DenseLengthGraph, so the axis sweeps its live `speed` leaf directly. _v_min
    # (the slowest swept speed) sizes the max_delay_bound history buffer.
    _network_axes = [ax for ax in expl['axes'] if ax.get('is_network')]
    _has_network_axis = bool(_network_axes)
    _v_min = None
    if _network_axes:
        _vvals = []
        for _nax in _network_axes:
            _vvals.extend(_nax['values'] if 'values' in _nax else [_nax['lo']])
        _v_min = min(_vvals)
%>
def ${expl['name']}(state, model_fn, result_transient=None, **kwargs):
    """${expl['label']} - ${grid_desc}."""
    _network = kwargs.get('network')
% if _has_network_axis:
    if _network is not None and hasattr(_network.graph, 'lengths'):
        # Rebuild the base DenseLengthGraph once (outside jit/vmap) so its buffer is
        # sized for the slowest swept speed; the conduction_speed axis then sweeps its
        # live `speed` leaf.
        _v_build = ${conduction_speed}
        _lengths = _network.graph.lengths
        _length_graph = DenseLengthGraph(
            _network.graph.weights, _lengths, speed=_v_build,
            region_labels=_network.graph.region_labels,
            # min(): the binding speed is the build speed when every swept speed is faster.
            max_delay_bound=float(jnp.max(_lengths)) / min(_v_build, ${_v_min}),
        )
        _network = type(_network)(
            _network.dynamics, _network.coupling, _length_graph, noise=_network.noise,
        )
% endif
% if any(ax.get('builder_expr') for ax in expl['axes']):
    # Builder-axis support: resolve a base-sim observation named in a builder argument.
    # `base_observations` is the Bunch of observations the main run computed before this
    # exploration; `_bov(name)` returns one (unwrapping a monitor result's `.data`).
    _base_obs = kwargs.get('base_observations') or Bunch()
    def _bov(_name):
        assert _name in _base_obs, (
            f"builder for '${expl['name']}' references base observation '{_name}', which the "
            "main run did not compute (run mode='all' so base observations are available)")
        _o = _base_obs[_name]
        return _o.data if hasattr(_o, 'data') else _o
% endif
    if _network is not None:
        _solver = get_solver()
% if has_transient:
        # Each grid point runs its own transient + main simulation.
        # This ensures each parameter combination settles to its own steady state.
        _t_transient = ${transient_time}
        _t_total = _t_transient + ${t1_default}
        _n_transient = int(_t_transient / ${dt})
        _expl_model_fn_raw, _expl_state = prepare(_network, _solver, t0=0.0, t1=_t_total, dt=${dt})
        _expl_state = copy.deepcopy(_expl_state)  # isolate from shared network params
        # Same IC construction as the main run: declared per-node overrides, then an
        # optional from_experiment operating-point seed (else the sweep cold-starts).
        _expl_state = _apply_seed_params(_apply_seed_dynamics(_apply_node_overrides(_expl_state)))
        % if stochastic_param_info:
        _inject_stochastic_trajectories(_expl_state, _t_total, ${dt}, key=jax.random.key(${list(stochastic_param_info.values())[0]['seed']}))
        % endif
        % if has_stimulus_events:
        # Offset event t0 by transient time (events are defined relative to main sim)
        for _ext_key in list(_expl_state.external.keys()):
            if hasattr(_expl_state.external[_ext_key], 't0'):
                _expl_state.external[_ext_key].t0 = _expl_state.external[_ext_key].t0 + _t_transient
        % endif
        # Wrap model_fn to trim transient — downstream observable code sees main sim only
        def _expl_model_fn(s):
            result = _expl_model_fn_raw(s)
            return NativeSolution(
                result.ts[_n_transient:], result.data[_n_transient:],
                dt=${dt}, variable_names=getattr(result, 'variable_names', None),
            )
% else:
        _expl_model_fn, _expl_state = prepare(_network, _solver, t0=0.0, t1=${t1_default}, dt=${dt})
        _expl_state = copy.deepcopy(_expl_state)  # isolate from shared network params
        # Same IC construction as the main run: declared per-node overrides, then an
        # optional from_experiment operating-point seed (else the sweep cold-starts).
        _expl_state = _apply_seed_params(_apply_seed_dynamics(_apply_node_overrides(_expl_state)))
        % if stochastic_param_info:
        _inject_stochastic_trajectories(_expl_state, ${t1_default}, ${dt}, key=jax.random.key(${list(stochastic_param_info.values())[0]['seed']}))
        % endif
% endif
    else:
        _expl_model_fn = model_fn
        _expl_state = state
% if expl['strategy'] == 'nsga2':
${search.nsga2_body(expl)}\
% elif expl.get('branch_seed'):
${sweep.branch_analysis_body(expl, solver_class, dt, warmstart_solver_kwargs)}\
% elif expl['sweep_seeding'] == 'from_previous':
${sweep.warmstart_sweep_body(expl, solver_class, dt, warmstart_solver_kwargs)}\
% else:
% if has_axes:
    grid_state = copy.deepcopy(_expl_state)
    % for ax in expl['axes']:
    % if ax.get('builder_expr'):
    ## Builder axis: materialize the sweep values from a callable, then sweep as a DataAxis.
    ## Values may be whole per-node vectors (array-valued axis). Product mode meshgrids the
    ## axes, which is 1-D only, so an array-valued axis is given a singleton group: Space's
    ## grouped path index-gathers it, carrying one whole vector per cell. Scalar axes (1-D
    ## values) stay ungrouped, so their behaviour is unchanged.
    _axisvals_${ax['name']} = jnp.asarray(${ax['builder_expr']})
    _grp_${ax['name']} = "${ax['name']}" if _axisvals_${ax['name']}.ndim > 1 else None
    % if ax.get('is_coupling'):
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']})
    % elif ax.get('is_network'):
    grid_state.graph.${ax['graph_leaf']} = DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']})
    % else:
    grid_state.dynamics.${ax['name']} = DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']})
    % endif
    % elif ax.get('is_seed'):
    ## Noise-seed axis: a dummy scalar slot Space sweeps; the wrapper below turns
    ## each cell's integer seed into config.noise.key, so every cell/trial draws an
    ## independent noise realization (a real per-trial ensemble, not a no-op).
    grid_state.dynamics._noise_seed = DataAxis(jnp.asarray(${ax['values']}, dtype=jnp.uint32))
    % elif ax.get('element_idx') is not None:
    ## Element-indexed parameter: create dummy scalar slot for Space discovery
    ## e.g., K[0] → grid_state.dynamics._K_el0 = GridAxis(...)
    % if 'values' in ax:
    grid_state.dynamics._${ax['name']}_el${ax['element_idx']} = DataAxis(${ax['values']})
    % else:
    grid_state.dynamics._${ax['name']}_el${ax['element_idx']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}_${ax['element_idx']}', ${ax['n']}))
    % endif
    % elif ax.get('is_coupling'):
    % if 'values' in ax:
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = DataAxis(${ax['values']})
    % else:
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}))
    % endif
    % elif ax.get('is_network'):
    ## Network-scope axis: sweep the graph's live `${ax['graph_leaf']}` leaf
    ## directly (resolved from the declared path by network_axis_leaf). Every
    ## dependent quantity is recomputed each forward pass — delays = lengths /
    ## speed, couplings from weights — so there is no per-cell graph or Network
    ## rebuild, and the leaf stays a differentiable pytree leaf.
    % if 'values' in ax:
    grid_state.graph.${ax['graph_leaf']} = DataAxis(${ax['values']})
    % else:
    grid_state.graph.${ax['graph_leaf']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}))
    % endif
    % else:
    % if 'values' in ax:
    grid_state.dynamics.${ax['name']} = DataAxis(${ax['values']})
    % else:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}))
    % endif
    % endif
    % endfor
    grid = Space(grid_state, mode="${expl['mode']}")
    # HPC sharding (tvboptim-native): `shard=(i, N)` runs only this array task's
    # slice of the sweep via Space's strided slice — the cells where j % N == i,
    # still vectorised, with fewer cells → bounded memory per task. The full grid
    # is reassembled by parameter value downstream (two-stage HPC pattern).
    _shard = kwargs.get('shard')
    if _shard is not None:
        _shard_i, _shard_n = int(_shard[0]), int(_shard[1])
        grid = grid[_shard_i::_shard_n]
% endif

    # Create observation monitors ONCE with history baked in (optimized pattern)
% if _use_stream:
    # Streaming reductions: every recorded observable is an Observation.dynamics observer,
    # so fold each into the integrator carry via prepare(reduce=...). The trajectory is
    # never materialized — peak memory is O(batch·block·n_node), which is what lets the
    # whole grid vmap on one device. Falls back to the post-scan path when the network is
    # not rebuildable (a passed-in model_fn) so the value is still produced.
    if _network is not None:
        # Stream over the FULL window [0, transient + observable]; the reducer's
        # skip=${_stream_skip} folds only the post-transient samples, so no trim wrapper
        # and no separate settle pass is needed — the settle happens inside the scan.
        _stream_model_fn, _ = prepare(
            _network, get_solver(block_size=${_stream_bs}),
            t0=0.0, t1=${_stream_t1}, dt=${dt},
            reduce=_compose_reducers(*[
                _STREAMING_REDUCERS[_n][0](_STREAMING_REDUCERS[_n][1], ${dt}, skip=${_stream_skip})
                for _n in ${repr(_stream_names)}
            ]),
        )
        @jax.jit
        def observable_fn(s):
            _vals = _stream_model_fn(s)
            return Bunch(**{_n: _v for _n, _v in zip(${repr(_stream_names)}, _vals)})
    else:
        @jax.jit
        def observable_fn(s):
            result = _expl_model_fn(s)
            return compute_all_observations(result, s, result_transient, only=${repr(set(_stream_names))})
% elif expl.get('record'):
<%
    # Closure of observations the jitted per-cell observable must compute: the recorded
    # (non-analysis) ones plus every observation they transitively depend on — through
    # `source` AND through pipeline-argument references (e.g. solitary's `omega_profile`
    # arg). Anything else — e.g. a non-jittable `solitary` the base run/builder needs but
    # this sweep does not record — is skipped so it never traces inside the observable.
    _all_obs_names = set(_all_observations.keys())
    def _obs_deps(_o):
        _deps = set()
        for _s in (getattr(_o, 'source', None) or []):
            _sn = str(_s) if not hasattr(_s, 'name') else str(_s.name)
            if _sn in _all_obs_names:
                _deps.add(_sn)
        for _st in (getattr(_o, 'pipeline', None) or []):
            for _an, _arg in ((getattr(_st, 'arguments', None) or {}).items()):
                _av = getattr(_arg, 'value', None)
                if _av is None:
                    continue
                _avs = str(_av)
                _base = _avs.split('.', 1)[0]   # bare name or dotted named-output ref
                if _avs in _all_obs_names:
                    _deps.add(_avs)
                elif _base in _all_obs_names:
                    _deps.add(_base)
        return _deps
    _need = set()
    _stack = [r for r in expl['record'] if r not in analysis_observation_names]
    while _stack:
        _n = _stack.pop()
        if _n in _need:
            continue
        _need.add(_n)
        _od = _all_observations.get(_n)
        if _od is not None:
            for _sn in _obs_deps(_od):
                if _sn not in _need:
                    _stack.append(_sn)
    _only_list = sorted(_need)
%>
    # Record a declared list of observations per grid point (derived via
    # compute_all_observations, `analysis` diagnostics via compute_analysis_observations),
    # stacked over the sweep into one array per name.
    @jax.jit
    def observable_fn(s):
${render_recorded_observable(expl['record'], derived_observation_names, network_observation_names, list(analysis_observations_dict.keys()), only_obs=_only_list)}
% elif obs_type == 'function_call':
<%
    # Collect unique observations used - categorize by type
    obs_used = set(a['obs'] for a in obs_args if a.get('obs'))
    # Simulated observations: in observation_names but NOT network or derived
    simulated_obs = [o for o in obs_used if o in observation_names and o not in network_observation_names and o not in derived_observation_names]
    # Network observations: external data (use module-level constant or kwargs)
    network_obs = [o for o in obs_used if o in network_observation_names]
    # Derived observations: computed from other observations
    derived_obs = [o for o in obs_used if o in derived_observation_names]
    # Runtime inputs: not defined as observations at all (passed via kwargs)
    runtime_obs = [o for o in obs_used if o not in observation_names and o not in derived_observation_names]
    needs_all_obs = len(derived_obs) > 0
%>
% for obs in sorted(simulated_obs):
<%
    obs_class = ''.join(word.capitalize() for word in obs.split('_'))
%>
    _${obs}_monitor = ${obs_class}(history=result_transient)
% endfor

    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
% for obs in sorted(simulated_obs):
        _${obs} = _${obs}_monitor(result)
% endfor
% if needs_all_obs:
        # Compute all observations to get derived observations
        _all_obs = compute_all_observations(result, s, result_transient)
% endif
<%
    # Build args list by observation type
    args_list = []
    for a in obs_args:
        if a.get('obs'):
            obs_name = a['obs']
            if obs_name in derived_observation_names:
                # Derived observation: from compute_all_observations
                args_list.append(f"getattr(_all_obs, '{obs_name}').data if hasattr(getattr(_all_obs, '{obs_name}', None), 'data') else getattr(_all_obs, '{obs_name}')")
            elif obs_name in network_observation_names:
                # Network observation: kwargs override, else module-level constant (from BIDS)
                args_list.append(f"kwargs.get('{obs_name}', {obs_name})")
            elif obs_name in observation_names:
                # Simulated observation: from monitor
                args_list.append(f"_{obs_name}.data")
            else:
                # Runtime input not defined as observation (must be in kwargs)
                args_list.append(f"kwargs['{obs_name}']")
        else:
            args_list.append(f"kwargs['{a['name']}']")
%>
        return ${obs_func}(${', '.join(args_list)})
% else:
<%
    # Check if this is a derived observation (no class exists - computed from other obs)
    is_derived_obs = obs_name in derived_observation_names if obs_name else False
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_')) if obs_name else ''
%>
% if not obs_name:
<%
    n_aux = len([v for v in (model_output_vars or []) if v in (model.derived_variables or {}).keys()])
    if not n_aux and model.derived_variables:
        n_aux = len(model.derived_variables)
%>
% if bundles_observations:
    # Observations declared: observable_fn returns only the reduced
    # observation values per grid point (no trajectory). Output size is
    # the sum of declared observation shapes — typically per-node or
    # per-pair statistics — rather than (T, n_states, n_nodes), so trial
    # vmaps and grid axes stay tractable.
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return compute_all_observations(result, s, result_transient)
% elif has_model_output and n_aux == 1:
    # Single model output — extract as (T, n_nodes) dropping the variable dimension
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return result.data[:, ${len(state_names)}, ...]
% elif has_model_output and n_aux > 1:
    # Multiple model outputs — keep variable dimension (T, n_out, n_nodes)
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return result.data[:, ${len(state_names)}:, ...]
% else:
    # No observable specified, no model output, no observations — return full simulation data
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return result.data
% endif
% elif is_derived_obs:
    # ${obs_name} is a derived observation - use compute_all_observations
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        all_obs = compute_all_observations(result, s, result_transient)
% if output_key:
        obs_result = getattr(all_obs, '${obs_name}', None)
        if hasattr(obs_result, '${output_key}'):
            return obs_result.${output_key}
        elif isinstance(obs_result, dict):
            return obs_result['${output_key}']
        else:
            return obs_result
% else:
        obs_result = getattr(all_obs, '${obs_name}', None)
        return obs_result.data if hasattr(obs_result, 'data') else obs_result
% endif
% else:
    _${obs_name}_monitor = ${obs_class}(history=result_transient)

    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        obs_result = _${obs_name}_monitor(result)
% if output_key:
        return obs_result['${output_key}'] if isinstance(obs_result, dict) else obs_result.data
% else:
        return obs_result.data
% endif
% endif
% endif

<%
    element_axes = [ax for ax in expl['axes'] if ax.get('element_idx') is not None]
%>
% if element_axes:
    ## Wrap observable_fn to reconstruct array parameters from element slots.
    ## Space sweeps dummy scalar slots (_K_el0, _K_el1), which we pack back
    ## into the original array parameter before running the simulation.
    _element_base_fn = observable_fn
    def observable_fn(s):
        % for ax in element_axes:
        s.dynamics.${ax['name']} = s.dynamics.${ax['name']}.at[${ax['element_idx']}].set(s.dynamics._${ax['name']}_el${ax['element_idx']})
        % endfor
        return _element_base_fn(s)
% endif

% if has_noise and any(ax.get('is_seed') for ax in expl['axes']):
    ## Per-cell noise reseeding: turn each cell's swept integer seed into
    ## config.noise.key so every cell/trial draws an independent noise realization.
    ## config.noise.key is a live runtime PRNG leaf (tvboptim's solve reads it per
    ## call), so varying it per cell needs no re-prepare — it composes with the vmap.
    _seed_base_fn = observable_fn
    def observable_fn(s):
        s.noise.key = jax.random.key(jnp.asarray(s.dynamics._noise_seed, dtype=jnp.uint32))
        return _seed_base_fn(s)
% endif

% if expl.get('n_trials', 1) > 1 and stochastic_param_info:
<%
    _sp_names = list(stochastic_param_info.keys())
    _n_sp = len(_sp_names)
%>\
    # === Trial parallelization: ${expl['n_trials']} trials via jax.vmap ===
    # Each trial uses a different random noise realization for stochastic parameters.
    # vmap maps the observable over all trials in parallel on the same device.
    _n_trials = ${expl['n_trials']}
    % if has_transient:
    _n_steps_stoch = int(_t_total / ${dt}) + 2
    % else:
    _n_steps_stoch = int(${t1_default} / ${dt}) + 2
    % endif
    _trial_keys = jax.random.split(jax.random.key(${stochastic_param_info[_sp_names[0]]['seed']}), _n_trials)
    % for _sp_idx, _sp_name in enumerate(_sp_names):
<%
    _sp_info = stochastic_param_info[_sp_name]
    _sp_shape = _sp_info.get('shape', '')
    if _sp_shape and 'n_nodes' in _sp_shape:
        _trial_noise_shape = f'(_n_steps_stoch, {n_nodes})'
    else:
        _trial_noise_shape = '(_n_steps_stoch,)'
    # Distribution-specific noise generation. Explicit mean/std (sigma) from the
    # distribution parameters win; the domain is only a fallback.
    if _sp_info['dist'] in ('gaussian', 'normal'):
        _mean = _sp_info['mean'] if _sp_info.get('mean') is not None else _sp_info['default']
        _std = _sp_info['std'] if _sp_info.get('std') is not None else (_sp_info['hi'] - _sp_info['lo']) / 4.0
        _noise_gen = f'{_mean} + {_std} * jax.random.normal(k, {_trial_noise_shape})'
    else:
        # Default: uniform
        _noise_gen = f"jax.random.uniform(k, {_trial_noise_shape}, minval={_sp_info['lo']}, maxval={_sp_info['hi']})"
    # For multiple stochastic params, split each trial key for independent sub-keys
    if _n_sp > 1:
        _noise_gen = _noise_gen.replace('(k,', f"(jax.random.split(k, {_n_sp + 1})[{_sp_idx + 1}],")
%>\
    _trial_noises_${_sp_name} = jax.vmap(lambda k: ${_noise_gen})(_trial_keys)
    % endfor

    _base_trial_observable = observable_fn

<%
    _pmode_s = str(expl.get('parallel_mode') or 'auto').lower()
    _pbatch_s = expl.get('parallel_batch_size')
%>\
    @jax.jit
    def observable_fn(s):
        def _run_trial(${', '.join(f'_tn_{sp}' for sp in _sp_names)}):
    % for _sp_name in _sp_names:
            s.dynamics._stoch_${_sp_name} = _tn_${_sp_name}
    % endfor
            return _base_trial_observable(s)
        # See the IC-trial branch for parallel_mode semantics.
% if _pmode_s == 'vmap':
        trial_results = jax.vmap(_run_trial)(${', '.join(f'_trial_noises_{sp}' for sp in _sp_names)})
% elif _pmode_s == 'pmap':
        trial_results = jax.pmap(_run_trial)(${', '.join(f'_trial_noises_{sp}' for sp in _sp_names)})
% elif len(_sp_names) == 1:
        trial_results = jax.lax.map(_run_trial, _trial_noises_${_sp_names[0]}, batch_size=${_pbatch_s if _pbatch_s else 1})
% else:
        trial_results = jax.lax.map(lambda args: _run_trial(*args), (${', '.join(f'_trial_noises_{sp}' for sp in _sp_names)}), batch_size=${_pbatch_s if _pbatch_s else 1})
% endif
    % if expl.get('average') == 'trials':
        return jnp.mean(trial_results, axis=0)
    % else:
        return trial_results
    % endif
% endif

% if expl.get('n_trials', 1) > 1 and sv_distribution_info:
<%
    _sv_names = list(sv_distribution_info.keys())
    _n_sv = len(_sv_names)
%>\
    # === IC-based trial parallelization: ${expl['n_trials']} trials ===
    # Each trial samples different initial conditions from state variable distributions.
    _n_trials = ${expl['n_trials']}
    _trial_key = jax.random.key(${random_seed})
    _trial_keys = jax.random.split(_trial_key, _n_trials)

    def _sample_ics(key):
        keys = jax.random.split(key, ${_n_sv + 1})
        ic = _expl_state.initial_state.dynamics  # (n_states, n_nodes)
    % for _si, (_sv_name, _sv_info) in enumerate(sv_distribution_info.items()):
    % if _sv_info['dist'] in ('gaussian', 'normal'):
        ic = ic.at[${_sv_info['idx']}].set(${(_sv_info['lo'] + _sv_info['hi']) / 2} + ${(_sv_info['hi'] - _sv_info['lo']) / 4.0} * jax.random.normal(keys[${_si + 1}], ic[${_sv_info['idx']}].shape))
    % else:
        ic = ic.at[${_sv_info['idx']}].set(jax.random.uniform(keys[${_si + 1}], ic[${_sv_info['idx']}].shape, minval=${_sv_info['lo']}, maxval=${_sv_info['hi']}))
    % endif
    % endfor
        return ic

    _trial_ics = jax.vmap(_sample_ics)(_trial_keys)  # (n_trials, n_states, n_nodes)

    _base_ic_observable = observable_fn

<%
    _pmode = str(expl.get('parallel_mode') or 'auto').lower()
    _pbatch = expl.get('parallel_batch_size')
%>\
    @jax.jit
    def observable_fn(s):
        def _run_trial(ic):
            s.initial_state.dynamics = ic
            return _base_ic_observable(s)
        # Trial-axis parallelism is picked per the Exploration.parallel_mode
        # slot. ``vmap`` batches all trials (fast, peak memory n_trials ×
        # per-trial working set); ``lax_map`` runs them sequentially with
        # peak memory bounded by one trial. ``auto`` defaults to lax_map
        # at batch_size=1 — safe for any n_trials × n_nodes.
% if _pmode == 'vmap':
        trial_results = jax.vmap(_run_trial)(_trial_ics)
% elif _pmode == 'pmap':
        trial_results = jax.pmap(_run_trial)(_trial_ics)
% else:
        trial_results = jax.lax.map(_run_trial, _trial_ics, batch_size=${_pbatch if _pbatch else 1})
% endif
    % if expl.get('average') == 'trials':
        return jnp.mean(trial_results, axis=0)
    % else:
        return trial_results
    % endif
% endif

% if expl['algorithms']:
    # ── Algorithm-wired exploration (Exploration.algorithms) ─────────────────
    # The wired algorithm chain runs AT EACH sweep point before the observable.
    # Algorithms are iterative/stateful (Python loops, e.g. FIC) — NOT vmappable
    # — so points execute SEQUENTIALLY. Each grid point's state arrives with its
    # swept params already applied; we run the chain to tune it, then observe.
    def _algo_point_fn(_pt_state):
        import copy as _copy
        _ps = _copy.deepcopy(_pt_state)
% for _algo in expl['algorithms']:
        _algo_res_${_algo['name']} = run_${_algo['name']}(
            _ps, _expl_model_fn, jax.random.key(${random_seed}),
            n_iterations=${_algo['n_iterations']},
% for _hp_name, _hp_val in _algo['hyperparams'].items():
            ${_hp_name}=${_hp_val},
% endfor
## External inputs the run-func requires (data-source files from kwargs; network/
## dataset targets from their module-level globals) — mirrors the flat call site.
% for _inp_name in _algo.get('input_names', []):
            ${_inp_name}=kwargs.get('${_inp_name}'),
% endfor
% for _net_obs_name in _algo.get('network_obs_inputs', []):
            ${_net_obs_name}=${_net_obs_name},
% endfor
            history=result_transient, verbose=False,
        )
        _ps = _algo_res_${_algo['name']}.state
% endfor
        return observable_fn(_ps)

% if has_axes:
    exec_runner = SequentialExecution(_algo_point_fn, grid)
    _grid_outputs = list(exec_runner.run())
% else:
    _grid_outputs = [_algo_point_fn(_expl_state)]
% endif
% else:
% if has_axes:
    import jax as _jax
    # Stream "grid batch i/N" through the tvbo.run logger from inside the jitted
    # lax.map (JAX-native jax.debug.callback, fires once per n_vmap batch) so a long
    # grid does not go silent after "STEP 2 > ...". Identity wrap when INFO is off.
    _n_batches = max(1, -(-grid.N // ${expl['n_parallel']}))
    exec_runner = ParallelExecution(
        progress_ticker(_n_batches, label="grid batch")(observable_fn),
        grid, n_pmap=_jax.device_count(), n_vmap=${expl['n_parallel']},
    )
    _grid_outputs = list(exec_runner.run())
% else:
    # Trial-only exploration — no parameter grid
    _grid_outputs = [observable_fn(_expl_state)]
% endif
% endif
    # Tree-aware stack: works for both array returns and pytree returns
    # (e.g. Bunch with data + observations when no observable is specified).
    _stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *_grid_outputs)

    # Build axes info for ExplorationResult. The point count mirrors the grid's own
    # ``kwargs.get('n_<axis>', <default>)`` so a runtime n override stays consistent
    # with the recorded coordinate (otherwise the stacked result and its coord disagree).
    _axes_info = [
% for ax in expl['axes']:
<%
    _nkey = f"n_{ax['name']}_{ax['element_idx']}" if ax.get('element_idx') is not None else f"n_{ax['name']}"
%>
        Bunch(
% if ax.get('element_idx') is not None:
            name='${ax.get('label', ax['name'])}[${ax['element_idx']}]',
% else:
            name='${ax.get('label', ax['name'])}',
% endif
% if ax.get('builder_expr'):
            ## Builder axis: values are runtime whole-vectors; the coordinate is the point index.
            explored_values=jnp.arange(len(_axisvals_${ax['name']})),
            n=len(_axisvals_${ax['name']}),
% elif 'values' in ax:
            explored_values=jnp.array(${ax['values']}),
            n=${ax['n']},
% else:
            lo=${ax['lo']},
            hi=${ax['hi']},
            explored_values=jnp.linspace(${ax['lo']}, ${ax['hi']}, kwargs.get('${_nkey}', ${ax['n']})),
            n=kwargs.get('${_nkey}', ${ax['n']}),
% endif
% if ax.get('is_coupling'):
            is_coupling=True,
            coupling_key='${ax['coupling_key']}',
% endif
% if ax.get('element_idx') is not None:
            element_idx=${ax['element_idx']},
% endif
% if ax.get('reduce'):
            reduce='${ax['reduce']}',
% endif
        ),
% endfor
    ]

    # Sharded run: read each retained cell's actual parameter values back from
    # the sliced grid itself, so the coordinates track the grid's own cell order
    # regardless of how the Space orders axes internally (avoids positional
    # mislabelling). Relabel to the exploration's axis names where unambiguous,
    # else keep the grid keypath. The flat per-shard result is then
    # self-describing and reassembles by value across tasks.
    _cell_coords = None
% if has_axes:
    _shard = kwargs.get('shard')
    if _shard is not None:
        _df = grid.to_dataframe()
        _bare_to_label, _network_label = {}, None
        for _a in _axes_info:
            _bare_to_label.setdefault(str(_a.name).rsplit('.', 1)[-1], str(_a.name))
            if str(_a.name).startswith('network.'):
                _network_label = str(_a.name)   # network-scope axis (e.g. conduction_speed)
        _cell_coords, _used = {}, set()
        for _col in _df.columns:
            _label = _bare_to_label.get(str(_col).rsplit('.', 1)[-1], None)
            if _label is None and _network_label is not None and str(_col).startswith('graph.'):
                # The network.conduction_speed axis sweeps the DenseLengthGraph's `speed`
                # leaf, which flattens to a keypath like "graph.2" whose bare name ("2")
                # does not match the axis label — restore the friendly network name.
                _label = _network_label
            if _label is None:
                _label = str(_col)
            if _label in _used:
                _label = str(_col)  # disambiguate a bare-name collision with the keypath
            _used.add(_label)
            _cell_coords[_label] = np.asarray(_df[_col].to_numpy())
% endif

% if bundles_observations:
    # observable_fn returned a Bunch of reduced observations.
    # No raw trajectory to attach; wrap each observation as xr.DataArray.
    _stacked_results = None
    _stacked_ts = None
    _observations_xr = {}
    for _obs_key, _obs_val in _stacked.items():
        if str(_obs_key).startswith('_'):
            continue
        _arr = getattr(_obs_val, 'ys', getattr(_obs_val, 'data', _obs_val))
        _ts = getattr(_obs_val, 'ts', None)
        _observations_xr[_obs_key] = _stacked_to_dataarray(
            _arr, _axes_info, intrinsic_ts=_ts,
            n_trials=${expl.get('n_trials', 1)}, name=str(_obs_key),
            cell_coords=_cell_coords,
        )
% else:
    _stacked_results = _stacked
    _stacked_ts = None
    _observations_xr = {}
% endif

    return ExplorationResult(
        name='${expl['name']}',
% if has_axes:
        grid=grid,
% endif
        results=_stacked_results,
        axes=_axes_info,
% if has_axes:
        cell_coords=_cell_coords,
% endif
<% _obs_label = obs_name if obs_name else (', '.join(model_output_vars) if has_model_output else obs_func) %>\
        observable='${_obs_label}',
        dt=${dt},
        output_names=${model_output_vars if has_model_output and not obs_name else []},
        observations=_observations_xr,
% if expl.get('n_trials', 1) > 1:
        n_trials=${expl['n_trials']},
% if expl.get('average'):
        average='${expl["average"]}',
% endif
% endif
    )
% endif


% endfor
% endif


${const.all_constants(experiment)}


def run_experiment(
    weights: jnp.ndarray,
    distances: jnp.ndarray = None,
    delays: jnp.ndarray = None,
    region_labels: list = None,
    mode: str = "all",
    stage: str = None,
    state: Bunch = None,
    seed_dynamics=None,
    seed_params=None,
    branch_seed=None,
% if network_observation_names:
    network_observations: Dict[str, Any] = None,
% endif
    **kwargs,
) -> Dict[str, Any]:
    """Run complete experiment workflow. Mode: simulation, optimization, exploration, algorithms, or all.

    seed_dynamics: optional {state_var_name: (n_nodes,)} operating point (InitialState.
    from_experiment) that overrides the sampled IC at every construction site.
    seed_params: optional {param_name: (n_nodes,)} per-node model parameters
    (InitialState.seed_parameters) loaded from the source run, e.g. a control mask.
    branch_seed: optional whole recorded branch (InitialState.from_experiment,
    source_point='branch') a branch-restart exploration replays per cell.
    """
    global _SEED_DYNAMICS, _SEED_PARAMS, _BRANCH_SEED
    _SEED_DYNAMICS = seed_dynamics
    _BRANCH_SEED = branch_seed
    _SEED_PARAMS = seed_params

    weights = jnp.array(weights)
    # Progress flows through the ``tvbo.run`` logger; ``quiet=True`` forces
    # silence for this call regardless of the configured level (back-compat with
    # ``run(..., quiet=True)``), otherwise ``TVBO_LOG_LEVEL`` / the tvbo logger
    # level decides what is shown.
    _quiet = kwargs.pop("quiet", False)

    def _log(msg):
        if not _quiet:
            logger.info(msg)
% if network_observation_names:
    # Materialize network-observation constants (empirical targets) from the
    # supplied matrices, keyed by observation name (e.g. {'fc_target': FC}).
    _bind_network_observations(network_observations)
% endif

    _log("STEP 1: Running simulation...")

    % if use_length_graph:
    # tract lengths → DenseLengthGraph derives delays = lengths / conduction_speed.
    network = create_network(weights, distances=distances, region_labels=region_labels, noise_sigma=${noise_sigma_value})
    % elif use_delay_graph:
    # explicit per-edge delays → DenseDelayGraph over the delay matrix directly.
    if delays is None:
        delays = jnp.array(distances) / ${conduction_speed} if (distances is not None and ${conduction_speed} > 0) else jnp.zeros_like(weights)
    else:
        delays = jnp.array(delays)
    network = create_network(weights, delays=delays, region_labels=region_labels, noise_sigma=${noise_sigma_value})
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=${noise_sigma_value})
    % endif

    # Determine if we need to run main simulation or just transient.
    # For algorithm/optimization/exploration modes, we only need transient - main simulation runs after
    run_main = mode in ('simulation', 'all', None)

    # Run simulation to get model_fn and state (includes transient settling if configured)
    sim_result = run_simulation(network, t1=${t1_default}, dt=${dt}, t_transient=${transient_time}, run_main=run_main, random_seed=kwargs.get('random_seed'))
    model_fn = sim_result.model_fn
    default_state = sim_result.state
    # Raw transient result for observation monitors (HRF warmup)
    transient = sim_result.result_transient
    _log(f"  Simulation period: ${t1_default} ms, dt: ${dt} ms")
    _log(f"  Transient period: ${transient_time} ms")

    # Use custom state if provided (e.g., from previous optimization)
    if state is not None:
        # Merge custom state parameters into the default state structure
        # This preserves internal state (_internal, coupling history, etc.)
        # while using the custom dynamics/coupling parameters
        use_state = copy.deepcopy(default_state)

        # Copy dynamics parameters from custom state
        if hasattr(state, 'dynamics'):
            for key in state.dynamics.keys():
                if not key.startswith('_'):
                    val = state.dynamics[key]
                    # Extract value from Parameter if needed
                    if hasattr(val, 'value'):
                        val = val.value
                    use_state.dynamics[key] = val

        # Copy coupling parameters from custom state
        if hasattr(state, 'coupling'):
            for coupling_name in state.coupling.keys():
                if not coupling_name.startswith('_'):
                    src_coupling = state.coupling[coupling_name]
                    dst_coupling = use_state.coupling[coupling_name]
                    for key in src_coupling.keys():
                        if not key.startswith('_'):
                            val = src_coupling[key]
                            # Extract value from Parameter if needed
                            if hasattr(val, 'value'):
                                val = val.value
                            dst_coupling[key] = val

        # Re-run simulation with custom parameters (only if main simulation was requested)
        if run_main:
            result = model_fn(use_state)
        else:
            result = None
        state = use_state
    else:
        state = default_state
        # Use raw result directly (run_simulation now returns raw results)
        result = sim_result.result

    # Compute observations only if main simulation was run
    if run_main and result is not None:
        observations = Bunch()
% for obs_name in observation_names:
% if obs_name in network_observation_names:
        observations.${obs_name} = ${obs_name}
% elif obs_name in derived_observation_names:
% else:
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
        observations.${obs_name} = ${obs_class}(history=transient)(result)
% endif
% endfor

        _all_obs = compute_all_observations(result, state, transient)
% for obs_name in sorted(derived_observation_names):
        observations.${obs_name} = _all_obs.${obs_name}
% endfor

        # Analysis observations (operate on the solve/loss, not result.data)
% if analysis_observations_dict:
        for _an_name, _an_val in compute_analysis_observations(state, network, transient).items():
            observations[_an_name] = _an_val
% endif
    else:
        observations = None

    # Save initial state from simulation (before any algorithms/optimization modify it)
    # This is the starting point for optimization unless depends_on is specified
    initial_state = copy.deepcopy(state)

<%
    # Result labels + record=True output channels are resolved in Python (the
    # tvboptim utils layer), not here — the template only emits from clean context.
    # The solver records ALL states (VARIABLES_OF_INTEREST); the user-facing result
    # presents only sv.record=True states (+ recorded aux), matching the tvb backend.
    # The full trajectory is kept intact above for observations and the warmup; the
    # record filter is applied as a channel slice on the presented result only.
    from tvbo.templates.tvboptim.utils import get_recorded_variable_names as _grvn, get_output_channels
    _, _requested_aux, result_var_names = _grvn(model, experiment)
    _output_idx, _output_names, _record_subset = get_output_channels(model, experiment)
%>
    % if _record_subset:
    # sv.record filters the presented channels; the full result/transient are kept
    # intact above for observations and warmup. Rebuild the NativeSolution on the
    # record=True channels (preserving its time axis), or slice a raw array.
    _record_idx = ${_output_idx}
    def _select_channels(_res):
        if _res is None:
            return None
        if hasattr(_res, "ys"):
            return type(_res)(_res.ts, _res.ys[:, _record_idx], dt=getattr(_res, "dt", None), variable_names=${tuple(_output_names)})
        return _res[:, _record_idx]
    _main_sel = _select_channels(result)
    _transient_sel = _select_channels(transient)
    transient_result = SimulationResult(result=_transient_sel, state_names=${_output_names}, nodes=region_labels) if _transient_sel is not None else None
    main_result = SimulationResult(result=_main_sel, observations=observations, state_names=${_output_names}, nodes=region_labels, transient=transient_result) if _main_sel is not None else None
    % else:
    transient_result = SimulationResult(result=transient, state_names=${result_var_names}, nodes=region_labels) if transient is not None else None
    main_result = SimulationResult(result=result, observations=observations, state_names=${result_var_names}, nodes=region_labels, transient=transient_result) if result is not None else None
    % endif

    results = Bunch(
        # Core simulation infrastructure (always present)
        model_fn=model_fn,
        state=state,
        network=network,

        # Integration result (mirrors integration section in YAML)
        # Access: results.integration.get_state(...), results.integration.transient
        integration=main_result,

    )
    _log("  Simulation complete.")

    % if has_explorations:
    if mode in ('exploration', 'all'):
        _log("STEP 2: Running explorations...")
        exploration_result = Bunch()

        % for expl in explorations:
        _log(f"  > ${expl['name']}")
        exploration_result.${expl['name']} = ${expl['name']}(
            state, model_fn,
            result_transient=transient,
            network=network,
            base_observations=observations,  # base-sim observations for builder-axis arguments
            **kwargs,  # Pass runtime kwargs (e.g., target data for correlation-based observables)
        )
        % endfor

        results.explorations = exploration_result
        _log("  Explorations complete.")
    % endif

    % if has_algorithms:
    if mode in ('algorithm', 'algorithms', 'all'):
        _log("STEP 3: Running algorithms...")
        # Determine if running all algorithms or just one
        algorithm_name = kwargs.get('name', kwargs.get('algorithm_name', None))
        run_all_algorithms = (mode in ('algorithms', 'all')) or (algorithm_name is None and mode == 'algorithm')

        if not run_all_algorithms and algorithm_name is None:
            available_algorithms = [${', '.join(f"'{safe_name(getattr(algo, 'name', 'algo'))}'" for algo in algorithms_list)}]
            raise ValueError(f"mode='algorithm' requires 'name' parameter. Available: {available_algorithms}")

        # Default random key from experiment-level execution.random_seed (can be overridden)
        default_algo_seed = kwargs.pop('seed', ${random_seed})
        algo_verbose = kwargs.pop('verbose', True)  # verbose is a display option, ok to default
        # Per-algorithm seeds (from algorithm.execution.random_seed if specified)
<%
    algo_seeds = {}
    for a in algorithms_list:
        aname = safe_name(getattr(a, 'name', 'algo'))
        algo_exec = getattr(a, 'execution', None)
        if algo_exec and hasattr(algo_exec, 'random_seed') and algo_exec.random_seed is not None:
            algo_seeds[aname] = int(algo_exec.random_seed)
        else:
            algo_seeds[aname] = None  # Use default
%>
        algo_seed_overrides = {${', '.join(f"'{k}': {v}" for k, v in algo_seeds.items())}}

        # Storage for algorithm results when running all
        algorithms_results = Bunch()

        # Run the specified algorithm(s)
        algo_result = None
<%
    # Build algorithms dict for looking up included algorithms
    algorithms_dict = {safe_name(getattr(a, 'name', 'algo')): a for a in algorithms_list}

    # Build dependency info for algorithms
    algorithms_deps = {}
    for a in algorithms_list:
        aname = safe_name(getattr(a, 'name', 'algo'))
        deps = getattr(a, 'depends_on', None) or []
        if isinstance(deps, str):
            deps = [deps]
        algorithms_deps[aname] = list(deps)

    # Get algorithms in dependency order (topological sort)
    # ALL algorithms run - order determined by depends_on declarations
    def get_sorted_algorithms():
        """Return ALL algorithm names in dependency order."""
        sorted_names = []
        remaining = set(algorithms_deps.keys())
        while remaining:
            # Find algorithms with all dependencies satisfied
            ready = [n for n in remaining if all(d in sorted_names or d not in remaining for d in algorithms_deps[n])]
            if not ready:
                # Circular dependency or missing dep - just add remaining
                ready = list(remaining)
            # Sort ready algorithms alphabetically for deterministic order among equals
            ready = sorted(ready)
            for n in ready:
                sorted_names.append(n)
                remaining.discard(n)
        return sorted_names

    sorted_algo_names = get_sorted_algorithms()

    def get_include_info(inc):
        """Extract algorithm name and argument overrides from AlgorithmInclude."""
        if hasattr(inc, 'algorithm'):
            algo_name = str(inc.algorithm.name) if hasattr(inc.algorithm, 'name') else str(inc.algorithm)
            args = {}
            inc_args = getattr(inc, 'arguments', None) or []
            if hasattr(inc_args, 'values'):
                inc_args = list(inc_args.values())
            for arg in inc_args:
                args[str(getattr(arg, 'name', ''))] = getattr(arg, 'value', None)
            return algo_name, args
        return str(inc), {}

    def get_all_hyperparams_exp(algo, alg_dict):
        """Get all hyperparameters including from included algorithms.
        Returns list of (name, value) tuples.
        """
        all_hp = {}
        # First, add hyperparameters from COMBINED included algorithms (with overrides).
        # Nested includes are skipped — their hyperparameters are passed inside the
        # generated run_<outer>() to the inner run_<inner>() call, not on the outer
        # signature, so exposing them here would pass an unexpected kwarg.
        for inc in (getattr(algo, 'includes', None) or []):
            if str(getattr(inc, 'mode', 'combined') or 'combined') == 'nested':
                continue
            inc_name, arg_overrides = get_include_info(inc)
            inc_algo = alg_dict.get(inc_name)
            if inc_algo:
                inc_hp = getattr(inc_algo, 'hyperparameters', None) or []
                if hasattr(inc_hp, 'values'):
                    inc_hp = list(inc_hp.values())
                for hp in inc_hp:
                    hp_name = str(getattr(hp, 'name', ''))
                    # Use override if present, else use original value
                    if hp_name in arg_overrides:
                        all_hp[hp_name] = arg_overrides[hp_name]
                    else:
                        all_hp[hp_name] = getattr(hp, 'value', None)
        # Then add this algorithm's own hyperparameters (override included)
        direct_hp = getattr(algo, 'hyperparameters', None) or []
        if hasattr(direct_hp, 'values'):
            direct_hp = list(direct_hp.values())
        for hp in direct_hp:
            all_hp[str(getattr(hp, 'name', ''))] = getattr(hp, 'value', None)
        return all_hp
%>
        # Define which algorithms to run
        if run_all_algorithms:
            # All algorithms run in dependency order (topological sort)
            algorithms_to_run = [${', '.join(f"'{n}'" for n in sorted_algo_names)}]
            _log(f"  Algorithms to run (dependency order): {algorithms_to_run}")
        else:
            algorithms_to_run = [algorithm_name]

        # Run algorithms in order
        for _algo_name_to_run in algorithms_to_run:
            algorithm_name = _algo_name_to_run
            # Reset random key for each algorithm (using per-algo seed if specified, else default)
            _algo_seed = algo_seed_overrides.get(algorithm_name, None)
            if _algo_seed is None:
                _algo_seed = default_algo_seed
            algo_key = jax.random.key(_algo_seed)  # Use newer key() API for consistency
            if algo_verbose:
                _log(f"\\n>>> Running algorithm: {algorithm_name} (seed={_algo_seed})")
            algo_result = None

% for algo in algorithms_list:
<%
    algo_name = safe_name(getattr(algo, 'name', 'algorithm'))

    # Get ALL hyperparameters including from included algorithms
    hyperparams_dict = get_all_hyperparams_exp(algo, algorithms_dict)
    n_iterations = getattr(algo, 'n_iterations', None)
    if n_iterations is None:
        raise ValueError(f"Algorithm '{algo_name}' missing required 'n_iterations' in YAML")

    # Get simulation_period from algorithm
    algo_sim_period = getattr(algo, 'simulation_period', None)
    if algo_sim_period is None:
        raise ValueError(f"Algorithm '{algo_name}' requires 'simulation_period' in YAML")

    # Observations - include from this algorithm AND any included algorithms
    def get_obs_names_with_includes(alg):
        """Get observation names from algorithm and all its includes."""
        obs_set = set()
        # This algorithm's observations
        obs_raw = getattr(alg, 'observations', None) or []
        if hasattr(obs_raw, '__iter__') and not isinstance(obs_raw, str):
            for o in obs_raw:
                obs_set.add(str(o))
        elif obs_raw:
            obs_set.add(str(obs_raw))
        # Included algorithms' observations (combined-mode only; nested includes
        # compute their observations inside their own inner loop, and their
        # external inputs are passed there — not on the outer signature).
        for inc in (getattr(alg, 'includes', None) or []):
            if str(getattr(inc, 'mode', 'combined') or 'combined') == 'nested':
                continue
            inc_algo_name = str(inc.algorithm.name) if hasattr(inc, 'algorithm') and hasattr(inc.algorithm, 'name') else str(getattr(inc, 'algorithm', inc))
            inc_algo = algorithms_dict.get(inc_algo_name)
            if inc_algo:
                obs_set.update(get_obs_names_with_includes(inc_algo))
        return obs_set

    obs_names = list(get_obs_names_with_includes(algo))

    # Determine which observations require external data:
    # 1. Observations with data_source (external file)
    # 2. Network observations (source starts with 'network.observations.')
    input_names = []
    network_obs_inputs = []  # Network observations that are module-level constants
    for obs_name in obs_names:
        obs_def = observations_dict.get(obs_name)
        if obs_def:
            # Check for data_source (external file)
            if hasattr(obs_def, 'data_source') and obs_def.data_source is not None:
                input_names.append(obs_name)
            else:
                # Check for network observation (from BIDS). `source` is
                # multivalued; for raw observations there is exactly one entry.
                _src = getattr(obs_def, 'source', None)
                if isinstance(_src, (list, tuple)):
                    _src = _src[0] if _src else None
                if _src is not None and hasattr(_src, 'name'):
                    _src = _src.name
                # `network.observations.*` (materialized from the Network/BIDS) and
                # `dataset.subject.*` (a per-subject target injected at run time via
                # run(active_subject=...)) are BOTH bound as module-level network-
                # observation globals by _bind_network_observations, so both must be
                # forwarded to the algorithm run-func as external inputs.
                if _src and (str(_src).startswith('network.observations.')
                             or str(_src).startswith('dataset.subject')):
                    network_obs_inputs.append(obs_name)

    # Get dependencies for this algorithm
    algo_deps = algorithms_deps.get(algo_name, [])
    has_deps = len(algo_deps) > 0
%>
            if algorithm_name == '${algo_name}':
                # Create algorithm-specific model_fn with simulation_period
                # Use get_solver() to ensure consistent solver config (with BoundedSolver if needed)
                algo_model_fn, algo_state = prepare(network, get_solver(), t1=${float(algo_sim_period)}, dt=${dt})

                # Create post-tuning model_fn/state using experiment-level integration duration
                # (needed for full-length BOLD simulation for FC computation)
                post_model_fn, post_state = prepare(network, get_solver(), t1=${t1_default}, dt=${dt})

                # Determine source state: depends_on result or initial_state
% if has_deps:
                # This algorithm depends on: ${algo_deps}
                # Copy from last dependency's result state (or initial if not yet run)
                _dep_name = '${algo_deps[-1]}'  # Use last dependency
                if _dep_name in algorithms_results and hasattr(algorithms_results[_dep_name], 'state'):
                    _source_state = algorithms_results[_dep_name].state
                    if algo_verbose:
                        _log(f"    (using state from dependency: {_dep_name})")
                else:
                    _source_state = initial_state
                    if algo_verbose:
                        _log(f"    (dependency {_dep_name} not yet run, using initial state)")
% else:
                # No dependencies - use initial state
                _source_state = initial_state
% endif

                # Copy PARAMETER VALUES from source state (dynamics, coupling params)
                for key in _source_state.dynamics.keys():
                    if not key.startswith('_'):
                        algo_state.dynamics[key] = _source_state.dynamics[key]
                for coupling_name in _source_state.coupling.keys():
                    if not coupling_name.startswith('_'):
                        for key in _source_state.coupling[coupling_name].keys():
                            if not key.startswith('_'):
                                algo_state.coupling[coupling_name][key] = _source_state.coupling[coupling_name][key]
                algo_state.initial_state.dynamics = _source_state.initial_state.dynamics

                # NOTE: Do NOT copy noise_samples - let prepare() create fresh noise.
                # The algorithm loop will update noise with key=jax.random.key(seed) anyway.

% for inp_name in input_names:
                # Validate required input: ${inp_name}
                if '${inp_name}' not in kwargs:
                    raise ValueError("Algorithm '${algo_name}' requires '${inp_name}' input (passed via kwargs)")
% endfor
<%
    # Detect if this algorithm uses sliding window and needs buffer inputs
    # Use hyperparams_dict which already includes hyperparams from included algorithms
    algo_has_window_size = 'window_size' in hyperparams_dict

    # Source observations needed for derived ones in this algorithm.
    algo_source_obs_needed = set()
    for obs_name in obs_names:
        dobs_def = derived_observations_dict.get(obs_name)
        if dobs_def and dobs_def.source:
            for src_obs in dobs_def.source:
                src_name = src_obs.name if hasattr(src_obs, 'name') else str(src_obs)
                if src_name in _all_observations:
                    algo_source_obs_needed.add(src_name)
    algo_needs_buffers = algo_has_window_size and len(algo_source_obs_needed) > 0

    # Multi-stage schedule (Algorithm.stages): resolve per-stage n_iterations +
    # hyperparameter values (stage overrides fall back to algo defaults).
    algo_stages = list(getattr(algo, 'stages', None) or [])
    has_stages = len(algo_stages) > 0
    stage_defs = []
    for _st in algo_stages:
        _over = {}
        for _arg in (getattr(_st, 'arguments', None) or []):
            _over[str(_arg.name)] = _arg.value
        _sd = {'n_iterations': int(_st.n_iterations)}
        for _hp_name, _hp_val in hyperparams_dict.items():
            _sd[_hp_name] = _over.get(_hp_name, _hp_val)
        stage_defs.append(_sd)
%>
% if has_stages:
                # ── Multi-stage schedule ──────────────────────────────────────
                # Run the algorithm body once per stage, IN ORDER, carrying
                # trajectory state + FC window buffer + monitors forward so the
                # stages form ONE continuous online tuning run. Per-stage eta /
                # window_size from YAML (Schirner 2023: eta halves, window doubles).
                _stage_defs = [
% for sd in stage_defs:
                    ${repr(sd)},
% endfor
                ]
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
% if has_deps:
                _stage_${src_obs}_buffer = (algorithms_results.get('${algo_deps[-1]}', Bunch()).get('${src_obs}_buffer', None) if '${algo_deps[-1]}' in algorithms_results else kwargs.get('${src_obs}_buffer', None))
% else:
                _stage_${src_obs}_buffer = kwargs.get('${src_obs}_buffer', None)
% endif
% endfor
% endif
% if has_deps:
                _stage_monitors = (algorithms_results.get('${algo_deps[-1]}', Bunch()).get('monitors', None) if '${algo_deps[-1]}' in algorithms_results else kwargs.get('monitors', None))
% else:
                _stage_monitors = kwargs.get('monitors', None)
% endif
                _stage_state = algo_state
                algo_result = None
                _stage_post_fc = []   # per-stage post-tuning FC matrices (r-trajectory)
                _stage_conv = []      # per-stage convergence Bunch (working-point trajectory)
                for _si, _sd in enumerate(_stage_defs):
                    if algo_verbose:
                        _log(f"  [{algorithm_name} stage {_si+1}/{len(_stage_defs)}] {_sd}")
                    algo_result = run_${algo_name}(
                        state=_stage_state,
                        model_fn=algo_model_fn,
                        key=jax.random.fold_in(algo_key, _si),
                        n_iterations=_sd['n_iterations'],
% for hp_name in hyperparams_dict.keys():
                        ${hp_name}=_sd['${hp_name}'],
% endfor
% for inp_name in input_names:
                        ${inp_name}=kwargs.get('${inp_name}'),
% endfor
% for net_obs_name in network_obs_inputs:
                        ${net_obs_name}=${net_obs_name},  # Module-level constant from BIDS
% endfor
                        post_model_fn=post_model_fn,
                        post_state=post_state,
                        history=transient,
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
                        ${src_obs}_buffer=_stage_${src_obs}_buffer,
% endfor
                        resync_every=kwargs.get('${algo_name}_resync_every', kwargs.get('resync_every', None)),
% endif
                        monitors=_stage_monitors,
                        run_post_tuning=True,   # validate after EACH stage (r-trajectory)
% if observation_ref:
                        observation_monitor=observations.${observation_ref},
% endif
                        verbose=algo_verbose,
                    )
                    _stage_state = algo_result.state
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
                    _stage_${src_obs}_buffer = algo_result.get('${src_obs}_buffer', _stage_${src_obs}_buffer)
% endfor
% endif
                    _stage_monitors = algo_result.get('monitors', _stage_monitors)
                    try:
                        _pt = algo_result.post_tuning.observations
                        _stage_post_fc.append(_pt['fc'] if 'fc' in _pt else _pt.get('fc'))
                    except Exception:
                        _stage_post_fc.append(None)
                    try:
                        _stage_conv.append(dict(algo_result.convergence))
                    except Exception:
                        _stage_conv.append(None)
                if algo_result is not None:
                    algo_result._extras['stage_post_fc'] = _stage_post_fc
                    algo_result._extras['stage_conv'] = _stage_conv
% else:
                algo_result = run_${algo_name}(
                    state=algo_state,
                    model_fn=algo_model_fn,
                    key=algo_key,
                    n_iterations=kwargs.get('${algo_name}_n_iterations', kwargs.get('n_iterations', ${n_iterations})),
                    print_every=kwargs.get('${algo_name}_print_every', kwargs.get('print_every', None)),
                    save_every=kwargs.get('${algo_name}_save_every', kwargs.get('save_every', None)),
% for hp_name, hp_val in hyperparams_dict.items():
<%
    if hp_val is None:
        raise ValueError(f"Hyperparameter '{hp_name}' in algorithm '{algo_name}' missing required 'value' in YAML")
%>
                    ${hp_name}=kwargs.get('${algo_name}_${hp_name}', kwargs.get('${hp_name}', ${hp_val})),
% endfor
% for inp_name in input_names:
                    ${inp_name}=kwargs.get('${inp_name}'),
% endfor
% for net_obs_name in network_obs_inputs:
                    ${net_obs_name}=${net_obs_name},  # Module-level constant from BIDS
% endfor
                    post_model_fn=post_model_fn,
                    post_state=post_state,
                    history=transient,
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
% if has_deps:
                    # Pass buffer from dependency if available
                    ${src_obs}_buffer=(algorithms_results.get('${algo_deps[-1]}', Bunch()).get('${src_obs}_buffer', None) if '${algo_deps[-1]}' in algorithms_results else kwargs.get('${src_obs}_buffer', None)),
% else:
                    ${src_obs}_buffer=kwargs.get('${src_obs}_buffer', None),  # Optional: pass from previous algorithm
% endif
% endfor
                    resync_every=kwargs.get('${algo_name}_resync_every', kwargs.get('resync_every', None)),
% endif
% if has_deps:
                    # Pass monitors from dependency for hemodynamic continuity
                    monitors=(algorithms_results.get('${algo_deps[-1]}', Bunch()).get('monitors', None) if '${algo_deps[-1]}' in algorithms_results else kwargs.get('monitors', None)),
% else:
                    monitors=kwargs.get('monitors', None),  # Optional: pass from previous algorithm
% endif
% if observation_ref:
                    observation_monitor=observations.${observation_ref},
% endif
                    verbose=algo_verbose,
                )
% endif
% endfor

            # After trying all algorithm blocks, check if one matched and store result
            if algo_result is not None:
                # Store result for this algorithm
                algorithms_results[algorithm_name] = algo_result
                # Results are stored; dependent algorithms will look them up via algorithms_results

        # End of algorithms_to_run loop

        # Error if no algorithm matched
        if len(algorithms_results) == 0:
            available_algorithms = [${', '.join(f"'{safe_name(getattr(algo, 'name', 'algo'))}'" for algo in algorithms_list)}]
            raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available_algorithms}")

        # Expose results
        if run_all_algorithms:
            # Running all: store all results, also expose last result at top level
            results['algorithms'] = algorithms_results
            # Expose each algorithm result by name for easy access: results.fic, results.fic_eib
            for _alg_name, _alg_result in algorithms_results.items():
                results[_alg_name] = _alg_result
            # Use last algorithm's result as the "main" result
            last_algo_name = algorithms_to_run[-1]
            _log(f"  Algorithms complete. Results: {list(algorithms_results.keys())}")
        else:
            # Running single: expose result at top level
            results['algorithms'] = {algorithm_name: algo_result}
            results[algorithm_name] = algo_result
            results['algorithm'] = Bunch(name=algorithm_name)
% if analysis_observations_dict:

        # Re-evaluate the operating-point analysis observations at the TUNED state. A tuning
        # algorithm (e.g. FIC) moves the operating point, so the base-run diagnostics — computed
        # at the pre-tuning state — are stale. Write onto main_result.observations (the Bunch that
        # result.observations exposes; SimulationResult may hold its own copy) so it reflects the
        # operating point the experiment actually settled to. The exploration path already observes
        # the tuned state.
        if main_result is not None and getattr(main_result, 'observations', None) is not None \
                and algo_result is not None and getattr(algo_result, 'state', None) is not None:
            for _an_name, _an_val in compute_analysis_observations(algo_result.state, network, transient).items():
                main_result.observations[_an_name] = _an_val
% endif
    % endif

    % if has_optimization:
    if mode in ('optimization', 'all'):
        _log("STEP 4: Running optimization...")
        _missing_inputs = []
% for kwarg_name in sorted(runtime_kwargs_needed) if runtime_kwargs_needed else []:
        if '${kwarg_name}' not in kwargs:
            _missing_inputs.append('${kwarg_name}')
% endfor
        if _missing_inputs:
            if mode == 'optimization':
                raise ValueError(f"Optimization requires these inputs via kwargs: {_missing_inputs}")
            else:
                # mode='all' - skip optimization if missing inputs
                _log(f"  Skipping optimization (missing: {_missing_inputs})")
        else:
            # Stage results storage (use Bunch for dot-notation access)
            stage_results = Bunch()

% if has_refine:
            # Refine reuses the shared base warm-up (model_fn/state/transient) — the same
            # settled state Stage 0 evaluated from — so the loss is byte-identical.
            _opt_model_fn = model_fn
            _opt_transient = transient
% else:
% if opt_has_custom_integration:
            # Prepare fresh model_fn and state for optimization
            # Optimization has custom integration settings: ${opt_solver_class} dt=${opt_dt} t1=${opt_t1}
            _log(f"  Preparing optimization model (t1=${opt_t1}ms, dt=${opt_dt}ms, solver=${opt_solver_class})")
% if opt_depends_on:
            # Use existing network (with history updated from algorithms)
            opt_model_fn, opt_state = prepare(network, get_solver(), t1=${opt_t1}, dt=${opt_dt})
            # Use existing transient for BOLD history
            opt_transient = transient
            # Copy parameter values from initial_state (result of algorithms or simulation)
            # optimization.depends_on: ${opt_depends_on}
            current_state = copy.deepcopy(opt_state)
            for key in initial_state.dynamics.keys():
                if not key.startswith('_'):
                    current_state.dynamics[key] = initial_state.dynamics[key]
            for coupling_name in initial_state.coupling.keys():
                if not coupling_name.startswith('_'):
                    for key in initial_state.coupling[coupling_name].keys():
                        if not key.startswith('_'):
                            current_state.coupling[coupling_name][key] = initial_state.coupling[coupling_name][key]
% else:
            # No depends_on: start from FRESH network (not modified by algorithms)
            # Create fresh network and run fresh transient for BOLD history
            % if use_length_graph:
            opt_network = create_network(weights, distances=distances, region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
            % elif use_delay_graph:
            opt_network = create_network(weights, delays=delays, region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
            % else:
            opt_network = create_network(weights, region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
            % endif
            opt_model_init, opt_state_init = prepare(opt_network, get_solver(), t1=${opt_t1}, dt=${opt_dt})
            opt_transient = opt_model_init(opt_state_init)  # Fresh BOLD history
            # Prepare optimization state from fresh network
            opt_model_fn, opt_state = prepare(opt_network, get_solver(), t1=${opt_t1}, dt=${opt_dt})
            current_state = copy.deepcopy(opt_state)
% endif
            _opt_model_fn = opt_model_fn
            _opt_transient = opt_transient
% else:
            _opt_model_fn = model_fn
            current_state = initial_state
            _opt_transient = transient
% endif
% endif

% if runtime_kwargs_needed:
% for kwarg_name in sorted(runtime_kwargs_needed):
            if '${kwarg_name}' not in kwargs:
                raise ValueError("Optimization loss requires '${kwarg_name}' input (passed via kwargs)")
            ${kwarg_name} = kwargs['${kwarg_name}']
% endfor
% endif

% if loss_functions:
            # Loss via the declarative observation pipeline. The objective is built from the
            # SAME compute_all_observations path as the diagnostics, so it is byte-identical to
            # the `loss` observation and stays backend-independent (no monitor-class references).
            def loss_fn(state):
                _obs = compute_all_observations(_opt_model_fn(state), state, _opt_transient)
<%
    _recon_idx = context.get('dataset_reconcile_indices') or {}
    # A by_label empirical target in this loss carries a keyed gather; the simulated
    # observables it is compared against are gathered onto the same shared nodes.
    _recon_target = next((a['obs_name'] for a in _lf_args
                          if a['type'] == 'observation' and a['obs_name'] in _recon_idx), None)
    loss_arg_exprs = []
    for a in _lf_args:
        if a['type'] == 'observation':
            obs_name_arg = a['obs_name']
            # Unwrap `.data` when the observation is an ObservationResult (derived
            # pipelines), else use the bare value — evaluates identically for the
            # bare case, so existing loss functions stay byte-identical.
            _acc = (f"(getattr(_obs, '{obs_name_arg}').data if hasattr(getattr(_obs, "
                    f"'{obs_name_arg}', None), 'data') else getattr(_obs, '{obs_name_arg}'))")
            if obs_name_arg in network_observation_names:
                # Empirical target: allow a runtime override, else the loaded constant.
                loss_arg_exprs.append(f"kwargs.get('{obs_name_arg}', {_acc})")
            elif _recon_target is not None:
                # Simulated observable: gather onto the target's shared nodes (keyed).
                loss_arg_exprs.append(f"_gather2d({_acc}, _DATASET_RECON_IDX['{_recon_target}'])")
            else:
                loss_arg_exprs.append(_acc)
        elif a['type'] == 'constant':
            loss_arg_exprs.append(str(a['value']))
        elif a['type'] == 'runtime':
            loss_arg_exprs.append(a['kwarg_name'])
%>
% if _lf_agg_over and _lf_agg_axis is not None:
                per_element_loss = jax.vmap(${_lf_func_name})(${', '.join(loss_arg_exprs)})
                return per_element_loss.${_lf_agg_func}()
% else:
                return ${_lf_func_name}(${', '.join(loss_arg_exprs)})
% endif
% else:
            def loss_fn(state):
                raise ValueError("No loss functions defined in optimization metadata.")
% endif

% if has_refine:
${search.refine_body(refine_info)}\
            results['optimizations'] = stage_results
            for _rk in list(stage_results.keys()):
                if not str(_rk).startswith('_'):
                    results[_rk] = stage_results[_rk]
            _log(f"  Refinement complete: {list(stage_results.keys())}")
% else:
% if len(optimization_stages) > 1:
            # Multi-stage optimization with optional stage filtering
            all_stage_names = [${', '.join(f"'{s['name']}'" for s in optimization_stages)}]

            if stage is not None:
                if stage not in all_stage_names:
                    raise ValueError(f"Unknown stage '{stage}'. Available stages: {all_stage_names}")
                stages_to_run = [stage]
                _log(f"  Running single stage: {stage}")
            else:
                stages_to_run = all_stage_names
                _log(f"  Multi-stage optimization: ${len(optimization_stages)} stages")

% for stage_idx, stage in enumerate(optimization_stages):
<%
stage_name = stage['name']
stage_warmup_from = stage['warmup_from']
stage_max_iter = stage['max_iterations']
stage_lr = stage['learning_rate']
%>
        # Stage ${stage_idx + 1}: ${stage_name}
        if '${stage_name}' in stages_to_run:
            _log(f"\n>>> Stage ${stage_idx + 1}/${len(optimization_stages)}: ${stage_name}")
            _log(f"    Free parameters: ${', '.join(p['name'] for p in stage['free_parameters'])}")
% if stage_warmup_from:
            _log(f"    Warmup from: ${stage_warmup_from}")
            # Use fitted_params from warmup_from stage (or from kwargs if running single stage)
            if '${stage_warmup_from}' in stage_results:
                current_state = stage_results['${stage_warmup_from}'].fitted_params
            elif 'warmup_state' in kwargs:
                # Allow passing in state from previous run
                current_state = kwargs['warmup_state']
                _log(f"    Using warmup_state from kwargs")
            elif stage is not None:
                # Running single stage without warmup - use initial state with warning
                _log(f"    WARNING: warmup_from='${stage_warmup_from}' not available, using initial state")
            else:
                raise ValueError(f"warmup_from stage '${stage_warmup_from}' not found in completed stages: {list(stage_results.keys())}")
% endif

            _fitted_${stage_name}, _history_${stage_name} = run_stage_${stage_name}(
                current_state,
                loss_fn,
                max_steps=kwargs.get('max_steps_${stage_name}', kwargs.get('max_steps', ${stage_max_iter})),
                learning_rate=kwargs.get('learning_rate_${stage_name}', ${stage_lr}),
            )

            # Run simulation with fitted parameters from this stage
            _post_${stage_name} = model_fn(_fitted_${stage_name})
            _post_${stage_name}_obs = compute_all_observations(_post_${stage_name}, _fitted_${stage_name}, transient)

            # Use OptimizationResult for each stage
            _stage_hyperparams = Bunch(
                learning_rate=kwargs.get('learning_rate_${stage_name}', ${stage_lr}),
                max_steps=kwargs.get('max_steps_${stage_name}', kwargs.get('max_steps', ${stage_max_iter})),
            )
            stage_results['${stage_name}'] = OptimizationResult(
                name='${stage_name}',
                state=_fitted_${stage_name},
                history=_history_${stage_name},
                simulation=SimulationResult(result=_post_${stage_name}, observations=_post_${stage_name}_obs, state_names=${state_names}),
                n_steps=kwargs.get('max_steps_${stage_name}', kwargs.get('max_steps', ${stage_max_iter})),
                hyperparameters=_stage_hyperparams,
            )
            current_state = _fitted_${stage_name}  # Chain to next stage

% endfor
        if stage is None:
            _log("  Multi-stage optimization complete")

        # Final results: last stage's fitted_params + per-stage access via dot notation
        results['fitted_params'] = current_state
        results['fitting_data'] = stage_results  # Bunch of all stage histories
        # Store under results.optimizations for consistent access
        results['optimizations'] = stage_results
        # Add each stage directly to results for easy access: results.global_optimization.fitted_params
        for _stage_name, _stage_result in stage_results.items():
            results[_stage_name] = _stage_result

% else:
            # Single-stage optimization
% if optimization_stages:
            init_state = mark_parameters_${optimization_stages[0]['name']}(current_state)
% else:
            init_state = copy.deepcopy(current_state)
% endif

            fitted_params, fitting_data = run_optimization(
                init_state,
                loss_fn,
                max_steps=kwargs.get('max_steps', ${max_steps}),
                learning_rate=kwargs.get('learning_rate', ${learning_rate}),
                optimizer=kwargs.get('optimizer', '${optimizer_name}'),
            )

            # Run final simulation with fitted parameters
            post_optimization = model_fn(fitted_params)

            # Compute ALL observations from post-optimization simulation
            post_optimization_observations = compute_all_observations(post_optimization, fitted_params, transient)

            # Store optimization result using OptimizationResult class
            _opt_name = '${loss_functions[0]["opt_name"] if loss_functions else "optimization"}'
            _opt_hyperparams = Bunch(
                learning_rate=kwargs.get('learning_rate', ${learning_rate}),
                optimizer=kwargs.get('optimizer', '${optimizer_name}'),
                max_steps=kwargs.get('max_steps', ${max_steps}),
            )
            _opt_result = OptimizationResult(
                name=_opt_name,
                state=fitted_params,
                history=fitting_data,
                simulation=SimulationResult(result=post_optimization, observations=post_optimization_observations, state_names=${state_names}),
                n_steps=kwargs.get('max_steps', ${max_steps}),
                hyperparameters=_opt_hyperparams,
            )

            # Store under results.optimizations.{name} for consistent structure
            results['optimizations'] = Bunch(**{_opt_name: _opt_result})
            results[_opt_name] = _opt_result  # Also at top level for convenience
            _log("  Optimization complete.")
% endif
% endif
    % endif

    % if has_inference:
    if mode in ('inference', 'all'):
        _log("STEP 5: Running Bayesian inference (MCMC)...")
% for _inf in inference_list:
${render_inference(_inf, coupling_keys, external_input_keys, set(derived_observation_names), set(network_observation_names))}
% endfor
        _log(f"  Inference complete. Posteriors: {list(results.get('inferences', Bunch()).keys())}")
    % endif

    _log("Experiment complete.")

    return results

<%
from pathlib import Path as _Path

# Check if network has BIDS configuration
has_bids = network.bids_dir is not None
if has_bids:
    # Resolve relative path using experiment's source file location
    _bids_path = _Path(network.bids_dir)
    if not _bids_path.is_absolute():
        _source_file = getattr(experiment, '_source_file', None)
        if _source_file:
            _bids_path = (_Path(_source_file).parent / _bids_path).resolve()
        else:
            _bids_path = _bids_path.resolve()
    bids_dir = str(_bids_path)
else:
    bids_dir = None
structural_measures = list(network.structural_measures) if network.structural_measures else None
observational_measures = list(network.observational_measures) if network.observational_measures else None
%>

if __name__ == "__main__":
    import pickle
    from pathlib import Path
    from tvbo.data.types import ExperimentResult
    from tvbo.log import configure_logging

    # Standalone run: surface progress on stderr, controlled by TVBO_LOG_LEVEL
    # (default INFO) — the same switch as ``tvbo run`` and ``exp.run(...)``.
    configure_logging()

    logger.info("${dynamics_class} Experiment - Standalone Execution")

% if has_bids:
    # Load network from BIDS (BEP017)
    from tvbo import Network as TVBONetwork
    logger.info("Loading network from BIDS: ${bids_dir}")
    _network = TVBONetwork.from_bids(
        "${bids_dir}",
% if structural_measures:
        structural_measures=${structural_measures},
% endif
% if observational_measures:
        observational_measures=${observational_measures},
% endif
    )
    weights = _network.weights_matrix
    distances = _network.lengths_matrix
    # Get region labels safely (may not be available in all BIDS datasets)
    try:
        region_labels = list(_network.labels.keys()) if _network.labels else None
    except (AttributeError, TypeError):
        region_labels = None
    logger.info("  Loaded network with %d nodes", weights.shape[0])
% else:
    # No BIDS directory configured - check if weights available
    if 'weights' not in dir() or weights is None:
        logger.error(
            "Network weights not defined. Either configure network.bids_dir in "
            "YAML or call run_experiment() with weights."
        )
        import sys
        sys.exit(1)
    distances = distances if 'distances' in dir() else None
    region_labels = region_labels if 'region_labels' in dir() else None
% endif

    # Run the experiment
    # Order: 1) Simulation → 2) Explorations → 3) Algorithms → 4) Optimization
% if network_observation_names:
    # Network observations (empirical targets) keyed by observation name,
    # resolved from the loaded network via the obs->measure mapping.
    _net_obs = {}
    if '_network' in dir() and _network is not None:
        _net_obs_data = _network.observations
        _net_obs = {name: _net_obs_data[measure]
                    for name, measure in _NETWORK_OBS_MEASURES.items()
                    if measure in _net_obs_data}
% endif
    raw_results = run_experiment(
        weights,
        distances=distances,
        region_labels=region_labels,
% if network_observation_names:
        network_observations=_net_obs,
% endif
        mode="all",
    )

    # Wrap in ExperimentResult for consistent access and export
    results = ExperimentResult(raw_results, experiment_name="${safe_name(experiment.label or 'experiment')}")

    # Save results
    output_path = Path(__file__).parent / "${safe_name(experiment.label or 'experiment')}_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info("Results saved to: %s", output_path)

    # Export BIDS-compatible output
    bids_output = Path(__file__).parent / "derivatives"
    results.export(bids_output, description="${safe_name(experiment.label or 'tvbsim')}")
    logger.info("BIDS output: %s", bids_output)

    # Summary — the results object itself is the script's stdout payload.
    logger.info("Results summary:")
    print(results)

