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
    parse_exploration, normalize_n_parallel, get_param_info, get_node_param_overrides,
    materialise_lazy_params,
    normalize_coupling_aliases, resolve_coupling_input_map,
    get_node_state_overrides, render_jax_default, get_mode_layout,
    get_all_observations_from_algo, network_axis_leaf, network_leaf_is_matrix,
    initial_conditions_axis_sv, noise_axis_param,
    graph_selection, observation_dims, parameter_keypath,
    has_host_pipeline, pipeline_stage_is_host, data_source_arrays,
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
# Functions inlined so the post-solve aux recompute is self-contained.
realign_render = lambda obj: model.render_equation(obj, format='jax', inline_functions=True)

# Extract key metadata from model. For number_of_modes>1 the per-node mode axis is
# folded into the state axis: state_names is the solver's flat (variable, mode) slot
# ordering, while var_names keeps the original variables (used for the result mode dim).
n_modes, state_names, var_slots = get_mode_layout(model)
var_names = list(model.state_variables.keys())
if n_modes > 1:
    # warnings.warn (dedups per process); CI-filtered in pyproject on number_of_modes=/EXPERIMENTAL, keep in sync.
    import warnings as _warnings
    _warnings.warn(
        f"number_of_modes={n_modes} (mode-coupled model '{getattr(model, 'name', '?')}') "
        "on the tvboptim backend is EXPERIMENTAL: the per-node mode axis is folded into "
        "the state axis (each variable occupies n_modes scalar slots). Validated against "
        "TVB to machine precision for the Stefanescu-Jirsa ReducedSet models; other "
        "multi-mode coupling topologies may not be faithful. Use the tvb backend for "
        "reference results.",
        stacklevel=2,
    )
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Model output variables (from model.output attribute)
# These are the variables the model defines as its primary output (e.g., v_pyr = y1 - y2)
model_output_vars = getattr(model, 'output', None) or []
if isinstance(model_output_vars, str):
    model_output_vars = [model_output_vars]
has_model_output = bool(model_output_vars)
# Each declared output resolves to its channel in the recorded ordering, so the position follows the layout rather than the kind.
from tvbo.templates.tvboptim.utils import (
    resolve_model_output_indices, format_channel_index, get_recorded_variable_names,
    state_only_recorded_aux, state_only_derived_var_names,
)
model_output_indices, model_output_names = resolve_model_output_indices(model, experiment)
_, _, _recorded_var_names = get_recorded_variable_names(model, experiment)
# State-only recorded derived variables to realign post-solve (single-mode only).
_state_only_aux = state_only_recorded_aux(model, experiment) if n_modes == 1 else []
# Bound as locals in the realign, in dependency order, so a recorded auxiliary can reach the intermediates it is built from.
_state_only_derived = state_only_derived_var_names(model) if _state_only_aux else []
model_output_channel_index = (
    format_channel_index(model_output_indices, len(_recorded_var_names))
    if model_output_indices else ''
)

# Extract state variable bounds (for BoundedSolver)
# Uses SymPy oo so code printers emit the correct backend literal
from tvbo.templates.tvboptim.utils import get_state_bounds, format_bounds_array, get_noise_covariance
state_bounds_lo, state_bounds_hi, has_state_bounds = get_state_bounds(model)
state_bounds_lo_str = format_bounds_array(state_bounds_lo, 'jax')
state_bounds_hi_str = format_bounds_array(state_bounds_hi, 'jax')

# A declared covariance wraps the solver in CorrelatedNoiseSolver, as finite clamped bounds wrap it in BoundedSolver.
noise_cov = get_noise_covariance(model, experiment)

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

# Resolved by TvboptimAdapter.resolve_couplings, in Python — see tvbo/adapters/tvboptim.py.
all_couplings = context['all_couplings']

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

# Stores the connectome as BCOO so each reduction is an O(nnz) edge-sum; tract lengths are the exception, needing DenseLengthGraph's live `speed` leaf, and are rejected below rather than downgraded.
use_sparse = str(getattr(network, 'graph_representation', 'auto') or 'auto') == 'sparse'
if use_sparse and use_length_graph:
    raise ValueError(
        "network.graph_representation: sparse is not available for a network with tract "
        "lengths: delays are derived as lengths / conduction_speed on every forward pass, "
        "which requires DenseLengthGraph's live `speed` leaf (swept by a "
        "`network.conduction_speed` axis and differentiable), and tvboptim has no sparse "
        "length-graph counterpart. Either drop graph_representation to keep the live "
        "conduction speed, or supply explicit per-edge `delay` edge attributes instead of "
        "lengths, which sparse does support (SparseDelayGraph)."
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
# Seconds per model time unit, which puts analytic-frequency diagnostics on a physical Hz axis.
from tvbo.utils.units import time_unit_of, unit_to_si_factor
time_unit = time_unit_of(integration, experiment)
time_si_factor = unit_to_si_factor(time_unit)

# Differentiation strategy -> native-solver kwargs, resolved in the tvboptim Python
# layer (shared with the solver template) rather than duplicated across mako blocks.
from tvbo.templates.tvboptim.utils import resolve_solver_kwargs, resolve_optimizer_mode, render_analysis_observations, render_recorded_observable, render_inference, render_adiabatic_signal, resolve_reduction, streaming_post_eval_plan, edge_label, edge_const, node_label, node_const
solver_kwargs_str = resolve_solver_kwargs(integration, dt)
# A forward scan honours coupling_evaluation but not the gradient kwargs, so only this one is passed.
_ce = getattr(integration, 'coupling_evaluation', None) if integration else None
warmstart_solver_kwargs = 'recompute_coupling_per_stage=True' if str(_ce) == 'per_stage' else ''
opt_mode = resolve_optimizer_mode(integration)

# Noise configuration from state_variables or integration.
# sigma is the standard deviation of the per-step Wiener increment, read through the shared reader so this template and the adapter cannot drift.
from tvbo.utils import noise_sigma as _shared_noise_sigma
def _noise_sigma(noise_obj):
    return _shared_noise_sigma(noise_obj) or 0.0

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

# Over the state axis the amplitude varies along the mixed axis, where `L diag(sigma)` is not the declared `diag(sigma) C diag(sigma)`, so fold the amplitudes in and drive every state at unit amplitude.
noise_cov_fold = bool(noise_cov) and str(noise_cov['axis']) == 'state'
if noise_cov_fold:
    noise_sigma_value = 1.0
    noise_targets = list(model.state_variables.keys())

# Network metadata
n_nodes = N_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
_cs = getattr(network, 'conduction_speed', None)
conduction_speed = float(_cs.value if hasattr(_cs, 'value') else _cs) if _cs is not None else 1.0

# `transforms:` -> JAX, applied in create_network on the RAW weights, so the kit is self-contained.
from tvbo.templates.tvboptim.utils import weight_transform_codegen as _weight_transform_codegen
weight_transform_jax, weight_transform_const_env, weight_transform_needs_lengths = _weight_transform_codegen(network)
has_weight_transforms = bool(weight_transform_jax)
weight_transform_distances_arg = "distances=distances, " if weight_transform_needs_lengths else ""

# Simulation parameters
assert integration.duration, "integration.duration required in YAML"
t1_default = float(integration.duration)
transient_time = float(integration.transient_time) if integration.transient_time else 0.0
has_transient = transient_time > 0

def event_clock_wrap(ax):
    """The transient offset a swept event onset inherits, as the ``wrap=`` that applies it.

    A fixed ``t0`` is declared relative to the main simulation and shifted onto the padded clock before the run; a swept one means the same thing. As a ``wrap`` the shift lands on the value substituted into the leaf and not on the axis's own points, so the grid coordinate stays the onset the recipe wrote.
    """
    return f", wrap=lambda _v: _v + {transient_time}" if (has_transient and ax['name'] == 't0') else ""

# Execution config
exec_config = experiment.execution
n_workers = int(exec_config.n_workers) if exec_config and exec_config.n_workers else 1
n_threads = int(exec_config.n_threads) if exec_config and exec_config.n_threads else -1
precision = str(exec_config.precision) if exec_config and exec_config.precision else 'float64'
accelerator = str(exec_config.accelerator) if exec_config and exec_config.accelerator else 'auto'
# accelerator -> JAX_PLATFORMS: 'auto' delegates to JAX's own device detection (None here).
from tvbo.templates.tvboptim.utils import jax_platform as _jax_platform_of
jax_platform = _jax_platform_of(accelerator)
enable_x64 = precision == 'float64'
random_seed = int(exec_config.random_seed) if exec_config and exec_config.random_seed else 0

# experiment.observations minus the ones with their own path: `analysis` diagnostics and cross-trial `reduce: trials` reductions are not raw/network monitors.
observations_dict = {n: o for n, o in experiment.observations.items()
                     if getattr(o, 'analysis', None) is None
                     and str(getattr(o, 'reduce', '') or '') != 'trials'} if experiment.observations else {}

# Categorize observations using utils
network_observation_names, observation_names = get_observation_refs(observations_dict)

# Class name from model
dynamics_class = model.name.replace(' ', '').replace('-', '') if model.name else 'GeneratedDynamics'

from tvbo.utils import initial_value as _initial_value

# Dynamics parameter info (shared utility)
dyn_param_names, dyn_param_defaults, dyn_param_shapes = get_param_info(model.parameters)
dyn_param_lazy = materialise_lazy_params(model.parameters, experiment)
# Couplings resolve their sourced/produced parameters the same way, hoisted here because the `_load_param` helper is emitted from this scope and a coupling-only lazy parameter must still get it.
coupling_param_lazy = {k: materialise_lazy_params(getattr(c, 'parameters', None), experiment) for k, c in all_couplings.items()}
any_coupling_lazy = any(coupling_param_lazy.values())

# Per-node parameter overrides from network.nodes[].parameters
# If nodes define e.g. B=17.6 on node 1, auto-promote B to heterogeneous array
node_param_overrides = get_node_param_overrides(network, n_nodes, dyn_param_defaults)
for _np_name in node_param_overrides:
    if _np_name not in dyn_param_shapes:
        dyn_param_shapes[_np_name] = '(n_nodes,)'

# Per-node initial state overrides from node ``state:`` entries
# e.g. nodes[0].state = {theta: 0.8} → overrides default initial_value per node
_default_init = [
    _initial_value(sv)
    for sv in model.state_variables.values()
    for _ in range(n_modes)  # one entry per (variable, mode) solver slot
]
node_state_overrides = get_node_state_overrides(network, n_nodes, state_names, _default_init)

# A warm-start ramp that seeds this run's IC from its settled endpoint, each step settling for the experiment's own transient.
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
        'path': parameter_keypath(_rax.parameter, couplings=all_couplings, coupling_key=_to_ci_key),
        'lo': _rlo, 'hi': _rhi, 'n': _rnpts,
        'settle': _rtr if _rtr > 0 else float(integration.duration),
    }

# The source run's whole recorded branch becomes a per-cell seed, so an analysis restarts at every branch point in parallel and shards, unlike the sequential scan that produced it.
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
                # A distribution's own seed overrides execution.random_seed, which defaults to 0.
                'seed': int(dist.seed) if getattr(dist, 'seed', None) is not None else random_seed,
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
            # A distribution's own seed overrides execution.random_seed, which defaults to 0.
            'seed': int(dist.seed) if getattr(dist, 'seed', None) is not None else random_seed,
        }

# === Events metadata (stimuli and other time-dependent inputs) ===
# Each active stimulus/continuous event becomes an AbstractExternalInput in the dfun; the shared resolver drops fisher-analysis target events, which are linear-response metadata and never integrated.
from tvbo.templates.tvboptim.utils import active_stimulus_events
stimulus_events = active_stimulus_events(experiment)
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

# A `subset` weight_distribution pre-samples one random-region mask per trial; the trial ensemble selects a row by writing state.external.<name>.trial.
subset_mask_events = [ev for ev in stimulus_events
                      if getattr(ev, 'weight_distribution', None) is not None
                      and str(getattr(ev.weight_distribution, 'name', '') or '').lower() == 'subset']

# External-input scope keys for the shared dotted-ref resolver AND for exploration/free-parameter axes: a swept `<event>.<param>` writes to `state.external.<event>.<param>`, where the emitted ExternalInput reads it, not to `state.dynamics`.
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

# The per-subject target becomes a leading-axis vmap batch rather than one workflow job per subject.
_dataset_on_device = bool(getattr(experiment, 'dataset_on_device', lambda: False)())
_dataset_target_names = set(getattr(experiment, 'dataset_observation_targets', None) or {})
try:
    _cohort_subject_ids = list(experiment.dataset_subject_ids()) if _dataset_on_device else []
except Exception:
    _cohort_subject_ids = []
# dataset.batch_size: subjects per on-device batch (None = size against the memory budget).
_cohort_batch_size = experiment.dataset_batch_size() if _dataset_on_device else None

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
# `reduce: trials` consumes another observation's TRIAL-STACKED output (n_trials, ...) host-side after the ensemble map, so it is excluded from the per-solve observers.
trial_reduced_dict = {n: o for n, o in _all_observations.items()
                      if str(getattr(o, 'reduce', '') or '') == 'trials'}
trial_reduced_names = set(trial_reduced_dict.keys())
for _trn, _tro in trial_reduced_dict.items():
    # as_list: `source` tolerates the scalar form, and iterating a bare string yields one "source" per character.
    _tr_sources = [str(getattr(_s, 'name', None) or _s) for _s in as_list(getattr(_tro, 'source', None) or [])]
    if not _tr_sources or any(_s not in _all_observations for _s in _tr_sources):
        raise ValueError(
            f"observation {_trn!r} declares reduce: trials, so every `source` must name another "
            f"observation (its per-trial values are what gets stacked); got {_tr_sources!r}. "
            f"Sourcing a raw state variable would stack full trajectories across all trials."
        )
    if len(_tr_sources) != 1 or len(list(getattr(_tro, 'pipeline', None) or [])) != 1:
        raise ValueError(
            f"observation {_trn!r} (reduce: trials) supports exactly one source observation and "
            f"one pipeline stage."
        )

observations = {n: o for n, o in _all_observations.items() if not _is_derived(o, experiment) and n not in analysis_observation_names and n not in trial_reduced_names}
derived_observations_dict = {n: o for n, o in _all_observations.items() if _is_derived(o, experiment) and n not in analysis_observation_names and n not in trial_reduced_names}
derived_observation_names = set(derived_observations_dict.keys())
# `record: false` marks an observation the recipe computes but does not keep: evaluated per grid point for its dependents, dropped before the sweep stacks the bundle.
unrecorded_observation_names = {n for n, o in _all_observations.items() if getattr(o, 'record', None) is False}

# True when an observation reaches HOST (non-JAX) pipeline code, which the exploration observable can only run outside jit; a jax-native callable traces and must not cost the sweep its vmap (see utils.pipeline_stage_is_host).
has_host_pipeline_obs = has_host_pipeline(list(observations.values()) + list(derived_observations_dict.values()))

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
# Schema defaults: n_parallel='auto', mode='product'
explorations = []
# Modules an ExplorationAxis.builder calls by dotted name; imported with the emit.
exploration_builder_modules = set()
for expl in exploration_list:
    assert expl.name, "exploration.name required in YAML"
    exp_info = {
        # Sanitize: name is used as a Python function identifier in generated code.
        'name': safe_name(expl.name),
        'label': expl.label or '',
        # mode has schema ifabsent: string(product)
        'mode': expl.mode or 'product',
        # How many sweep cells the backend vectorises at once: 'auto' defers the width to runtime, an explicit int passes through.
        'n_parallel': normalize_n_parallel(expl),
        # n_trials has schema ifabsent: integer(1)
        'n_trials': int(expl.n_trials) if expl.n_trials is not None else 1,
        # average: 'trials' to average over n_trials, None for individual results
        'average': str(expl.average) if expl.average else None,
        # parallel_mode: vmap | lax_map | pmap | auto. Defaults to auto (=lax_map at codegen).
        'parallel_mode': str(expl.parallel_mode) if getattr(expl, 'parallel_mode', None) else 'auto',
        'parallel_batch_size': int(expl.parallel_batch_size) if getattr(expl, 'parallel_batch_size', None) else None,
        'block_size': int(expl.block_size) if getattr(expl, 'block_size', None) else None,   # streaming-fold block granularity → get_solver(block_size=); bounds the per-block batched update's memory
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
        # Collapses the axis by a statistic instead of keeping it as a grid dim, carried into the axis metadata so the result knows which dim to reduce and how.
        _reduce = getattr(axis, 'reduce', None)
        _reduce_stat = (str(getattr(_reduce, 'statistic', None) or 'mean')
                        if _reduce is not None else None)
        # element_domains satisfy the axis with either explored_values or lo/hi/n bounds, the expansion below reading either.
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
        is_external_param = False
        is_network_param = False
        graph_leaf = None
        is_ic = False
        ic_row = None
        is_noise_param = False
        if pname.startswith('noise.'):
            # `noise.sigma` binds the amplitude leaf on the noise object directly — no wrapper, and `_axis_label` keeps the declared path so grid coords stay named as written.
            is_noise_param = True
            pname = noise_axis_param(pname)
            if not has_noise:
                raise ValueError(
                    f"exploration axis '{axis.parameter}': this experiment declares no "
                    f"noise, so there is no amplitude to sweep. Declare a noise sigma on "
                    f"a state variable (or on the integration) to sweep it."
                )
            if len(set(noise_sigma_targeted)) > 1:
                raise ValueError(
                    f"exploration axis '{axis.parameter}': this experiment declares a "
                    f"HETEROGENEOUS noise amplitude ({noise_sigma_per_state}), and one swept "
                    f"scalar would overwrite that whole per-state profile — every targeted "
                    f"state would be driven at the same amplitude, silently answering a "
                    f"different question than the one declared. Sweep a scale factor, or "
                    f"declare one sigma shared by every targeted state variable."
                )
        elif pname.startswith('network.'):
            # Split on the first dot so the remainder stays a full attribute path; rsplit would leave a prefix that no longer matches the scope and fall through as a wrong-scope write.
            is_network_param = True
            graph_leaf = network_axis_leaf(pname)
            # `pname` only names the axis' override kwarg, so sanitize it to an identifier; `_axis_label` keeps the declared dotted path for the grid coords.
            pname = re.sub(r'\W', '_', pname[len('network.'):])
        elif pname.startswith('initial_conditions.'):
            # A deterministic IC ensemble, one trajectory per swept value, distinct from the stochastic n_trials ensemble.
            is_ic = True
            _ic_sv = initial_conditions_axis_sv(axis.parameter)
            _ic_state_key = _ic_sv if n_modes == 1 else f"{_ic_sv}__mode0"
            assert _ic_state_key in state_names, (
                f"exploration axis '{axis.parameter}': unknown state variable "
                f"'{_ic_sv}' (state variables: {', '.join(state_names)})."
            )
            # A distributed SV is resampled every run, which would overwrite the swept value and degenerate the ensemble.
            if _ic_sv in sv_distribution_info:
                raise ValueError(
                    f"exploration axis '{axis.parameter}': state variable '{_ic_sv}' "
                    f"also declares a distribution, which resamples its initial value "
                    f"per run and would overwrite the swept value. Drop the distribution "
                    f"to sweep the initial condition deterministically, or drop the axis "
                    f"to keep the stochastic n_trials ensemble."
                )
            ic_row = state_names.index(_ic_state_key)
            pname = re.sub(r'\W', '_', _ic_sv)
        elif '.' in pname:
            prefix, pname = pname.rsplit('.', 1)
            is_coupling_param = (prefix in all_couplings)
            is_external_param = (not is_coupling_param and prefix in external_input_keys)
            # An unrecognised scope would fall through to the dynamics path with the prefix DISCARDED, so `nosie.sigma` silently sweeps a model's own `sigma` instead of the noise.
            _known_dyn = {str(_n).lower() for _n in (model.name, getattr(model, 'label', None), dynamics_class) if _n}
            if not is_coupling_param and not is_external_param and prefix.lower() not in _known_dyn and prefix != 'execution':
                raise ValueError(
                    f"exploration axis '{axis.parameter}': unknown scope '{prefix}'. The reserved "
                    f"scopes are 'noise.', 'network.', 'initial_conditions.' and "
                    f"'execution.random_seed'; otherwise a dotted parameter names this experiment's "
                    f"dynamics ('{model.name}'), one of its couplings "
                    f"({', '.join(sorted(all_couplings)) if all_couplings else 'none declared'}), "
                    f"or one of its external inputs "
                    f"({', '.join(sorted(external_input_keys)) if external_input_keys else 'none declared'})."
                )
            source_key = _to_ci_key(prefix) if is_coupling_param else prefix
        # Which grid sub-object this axis binds on and the leaf within it, stated ONCE for every append site below, so two axis shapes cannot disagree about scope.
        _scope_keys = {
            'is_coupling': is_coupling_param,
            'is_network': is_network_param,
            'graph_leaf': graph_leaf,
            'is_noise': is_noise_param,
            'is_external': is_external_param,
            'coupling_key': source_key if is_coupling_param else None,
            'external_key': source_key if is_external_param else None,
            'dynamics_key': source_key if (not is_coupling_param and not is_network_param and not is_external_param and source_key) else None,
            'element_idx': None,
            'reduce': _reduce_stat,
        }
        # The values come from the source run's recorded branch rather than a domain here, so the analysis restarts on exactly the points that were computed.
        if from_experiment_branch and not (domain or explored_values or _el_domains or _builder is not None):
            exp_info['axes'].append({
                'name': pname,
                'label': _axis_label,
                'is_branch': True,
                **_scope_keys,
            })
            continue
        # Before the builder branch: a seed axis bakes its integers into the grid at CODEGEN, and the generic parameter path would leave every cell identical under a real-looking ensemble dimension.
        if source_key == 'execution' and pname == 'random_seed' and _builder is not None:
            _bc = getattr(_builder, 'callable', None)
            _bargs = dict(_builder.arguments.items()) if getattr(_builder, 'arguments', None) else {}
            _deferred = [str(_n) for _n, _a in _bargs.items()
                         if getattr(_a, 'used', None) is not None
                         or (isinstance(getattr(_a, 'value', _a), str)
                             and 'observations.' in str(getattr(_a, 'value', _a)))]
            if _deferred:
                raise ValueError(
                    "exploration axis 'execution.random_seed' has a builder whose "
                    "argument(s) %s resolve at run time, but the seed values are baked "
                    "into the grid at codegen. Give the builder literal arguments, or "
                    "state the seeds with `explored_values:`/`domain:`." % _deferred
                )
            import importlib as _importlib
            import json as _json
            _bkwargs = {}
            for _an, _arg in _bargs.items():
                _av = _arg.value if hasattr(_arg, 'value') else _arg
                try:
                    _bkwargs[str(_an)] = _json.loads(_json.dumps(_av))
                except Exception:
                    _bkwargs[str(_an)] = _av
            explored_values = [int(_v) for _v in
                               getattr(_importlib.import_module(_bc.module), _bc.name)(**_bkwargs)]
            _builder = None

        # A callable materialises the stacked values at runtime, routed through the normal grid path as a DataAxis so it inherits sharding and batching with no special case.
        if _builder is not None:
            _bc = getattr(_builder, 'callable', None)
            assert _bc is not None and getattr(_bc, 'module', None) and getattr(_bc, 'name', None), \
                f"builder for exploration axis '{axis.parameter}' requires callable: {{name, module}}"
            import json as _json
            _arg_strs = []
            for _an, _arg in (_builder.arguments.items() if getattr(_builder, 'arguments', None) else []):
                # Resolved on the Python side and looked up at runtime, so a cross-experiment argument is never inlined into the code.
                if getattr(_arg, 'used', None) is not None:
                    _arg_strs.append("%s=_bdv(%r)" % (str(_an), "%s::%s" % (str(axis.parameter), str(_an))))
                    continue
                _av = _arg.value if hasattr(_arg, 'value') else _arg
                if isinstance(_av, str) and 'observations.' in _av:
                    _arg_strs.append("%s=_bov(%r)" % (str(_an), _av.split('observations.', 1)[1]))
                else:
                    try:
                        _lit = repr(_json.loads(_json.dumps(_av)))   # coerce JsonObj -> plain literal
                    except Exception:
                        _lit = repr(_av)
                    _arg_strs.append("%s=%s" % (str(_an), _lit))
            exploration_builder_modules.add(_bc.module)
            exp_info['axes'].append({
                'name': pname,
                'label': _axis_label,
                'builder_expr': "%s.%s(%s)" % (_bc.module, _bc.name, ", ".join(_arg_strs)),
                **_scope_keys,
            })
            continue
        # Each cell reseeds the solver's PRNG key, so a random-seed sweep is a real per-trial noise ensemble rather than a no-op parameter.
        if source_key == 'execution' and pname == 'random_seed':
            # With no noise, or a strategy whose body never reaches the grid binding, every cell comes out identical under a genuine-looking ensemble dimension, so fail rather than ship a fake one.
            _expl_strategy = str(getattr(expl, 'strategy', None) or 'grid')
            _seeding = str(getattr(expl, 'sweep_seeding', None) or '')
            _bypasses_grid = (
                _expl_strategy != 'grid'
                or _seeding == 'from_previous'
                or bool(getattr(expl, 'branch_seed', None))
            )
            if not has_noise or _bypasses_grid:
                _why = (
                    "this experiment's integration declares no noise (every state's "
                    "sigma is 0)" if not has_noise else
                    "this exploration's strategy (%r) never reaches the per-cell grid "
                    "binding that applies the seed" % _expl_strategy
                )
                raise ValueError(
                    "exploration axis 'execution.random_seed' has no consumer in "
                    "exploration %r: the seed reseeds the stochastic solver's PRNG key, "
                    "but %s, so every cell would produce an identical result. Either "
                    "make the seed reachable (give the integration noise, and use the "
                    "default grid strategy), or drop the axis and vary the ensemble "
                    "through a mechanism that does apply here (e.g. Exploration.n_trials "
                    "with a StateVariable.distribution for an initial-condition "
                    "ensemble)."
                    % (str(getattr(expl, 'name', None) or '<unnamed>'), _why)
                )
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
        # The grid binding writes the value into a dummy slot and the wrapper injects it into the state variable's row, so each cell starts from its own IC.
        if is_ic:
            if explored_values:
                _ic_vals = [float(v) for v in explored_values]
                exp_info['axes'].append({
                    'name': pname, 'label': _axis_label,
                    'is_ic': True, 'ic_row': ic_row,
                    'values': _ic_vals, 'n': len(_ic_vals),
                    'is_coupling': False, 'element_idx': None,
                    'reduce': _reduce_stat,
                })
            else:
                assert domain and domain.lo is not None and domain.hi is not None, \
                    (f"initial_conditions axis '{axis.parameter}' requires explored_values "
                     f"or a domain with lo/hi")
                _ic_n = _resolve_n(domain)
                exp_info['axes'].append({
                    'name': pname, 'label': _axis_label,
                    'is_ic': True, 'ic_row': ic_row,
                    'lo': float(domain.lo), 'hi': float(domain.hi), 'n': _ic_n,
                    'is_coupling': False, 'element_idx': None,
                    'reduce': _reduce_stat,
                })
            continue
        # A per-node dynamics parameter fans out to one element axis per node (K → K_el0…); a SCOPED axis never does, since its leaf lives on the graph, the noise or an external input, not on the model that happens to declare the same name.
        is_hetero_param = (not is_coupling_param and not is_network_param and not is_noise_param
                           and not is_external_param
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
                    **_scope_keys,
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
                        **_scope_keys,
                    })
                else:
                    exp_info['axes'].append({
                        'name': pname,
                        'label': _axis_label,
                        'lo': float(domain.lo),
                        'hi': float(domain.hi),
                        'n': n,
                        **_scope_keys,
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
        # Classified as the flat path does, so the exploration call site forwards the same inputs.
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
        _sp = getattr(_alg, 'simulation_period', None)
        if _sp is None:
            raise ValueError(f"Algorithm '{_alg_name}' requires 'simulation_period' in YAML")
        exp_info['algorithms'].append({
            'name': safe_name(_alg_name),
            'n_iterations': int(_nit),
            'hyperparams': _hp,
            'input_names': _alg_inp,
            'network_obs_inputs': _alg_netobs,
            'simulation_period': float(_sp),
        })

    # Search strategy: 'grid' (default, exhaustive) or 'nsga2' (pymoo multi-objective).
    exp_info['strategy'] = str(getattr(expl, 'strategy', None) or 'grid')
    exp_info['objectives'] = [str(o) for o in (getattr(expl, 'objectives', None) or [])]
    # from_previous seeds each point from the preceding point's settled state, and sweep_direction sets the traversal order.
    exp_info['sweep_seeding'] = str(getattr(expl, 'sweep_seeding', None) or 'independent')
    exp_info['sweep_direction'] = str(getattr(expl, 'sweep_direction', None) or 'up')
    if exp_info['strategy'] == 'nsga2':
        assert exp_info['objectives'], f"nsga2 exploration '{exp_info['name']}' requires objectives"
        # Resolve each decision axis to a tvboptim state path (+ optional log10 decode).
        _nsga_axes = []
        for _axis in axes_list:
            _adom = _axis.domain
            assert _adom is not None and _adom.lo is not None and _adom.hi is not None, \
                f"nsga2 axis '{_axis.parameter}' requires domain lo/hi"
            _apath = parameter_keypath(_axis.parameter, couplings=all_couplings, coupling_key=_to_ci_key)
            _nsga_axes.append({
                'path': _apath, 'lo': float(_adom.lo), 'hi': float(_adom.hi),
                'transform': str(getattr(_axis, 'transform', None) or 'none'),
            })
        exp_info['nsga2_axes'] = _nsga_axes
        # GA hyperparameters, keyed by name in Exploration.parameters.
        _default_pop = context.get('n_workers', 8)
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
        # Baked as a literal, like the refine stage's n_workers.
        exp_info['n_workers'] = context.get('n_workers', 1)
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
        # A preset over the warm-start controls, normalised onto sweep_seeding and sweep_direction so one renderer drives both.
        exp_info['sweep_seeding'] = 'from_previous'
        exp_info['sweep_direction'] = 'bidirectional' if exp_info['adiabatic']['bothways'] else 'up'
    # Each recorded observation must be a single-source, single-step trajectory reduction, becoming a statistic over the settled rollout of its source.
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
            # Each swept value's carried settled state seeds the analysis solve, so the exponents are measured on the continued branch rather than a cold start.
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
    # Each cell restarts the analysis from one branch point, so it shards across array tasks, reusing the warm-start post-scan pass with values from the loaded branch.
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
        _fpn = str(_fp.parameter).rsplit('.', 1)[-1]
        _fpath = parameter_keypath(_fp.parameter, couplings=all_couplings, coupling_key=_to_ci_key)
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

# Modules the emitted code calls by dotted name; nothing else in the emit would pull them in, so an unimported one is a NameError at the first swept cell.
derived_obs_modules = set()
for dobs_name, dobs in list(derived_observations_dict.items()) + list(trial_reduced_dict.items()):
    if dobs.pipeline:
        for stage in dobs.pipeline:
            c = getattr(stage, 'callable', None)
            if c:
                call_module = getattr(c, 'module', None)
                if call_module:
                    derived_obs_modules.add(call_module)
derived_obs_modules |= exploration_builder_modules

# First coupling name for docstring
first_coupling_name = list(all_couplings.keys())[0] if all_couplings else 'None'
%>
"""${dynamics_class} tvboptim Experiment."""
import os
import copy
import functools  # render_expression emits functools.reduce for Min/Max over lists
import logging
import time  # per-phase elapsed in the run log (tuning vs post-tuning wall-time)

# Shares the ``tvbo`` logger hierarchy, so ``TVBO_LOG_LEVEL`` controls it the same way in-process and standalone.
logger = logging.getLogger("tvbo.run")

% if jax_platform:
os.environ.setdefault("JAX_PLATFORMS", "${jax_platform}")  # from execution.accelerator=${accelerator}; 'auto' would let JAX detect
% endif
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
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% if has_state_bounds:
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver
% endif
% if noise_cov:
from tvbo.classes.correlated_noise import CorrelatedNoiseSolver, covariance_factor, fold_amplitudes
% endif
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
from tvboptim.types import Space, GridAxis, DataAxis, AbstractAxis
from tvboptim.execution import ParallelExecution, SequentialExecution
from tvbo.templates.tvboptim.callbacks import point_indices, progress_ticker, resolve_exploration_n_pmap, resolve_exploration_n_vmap   # array-axis cell → point index; grid-batch progress; n_parallel → vmap width and replica count
% endif
% if _dataset_on_device:
from tvbo.templates.tvboptim.callbacks import resolve_cohort_batch_size   # dataset.batch_size → subjects per on-device batch
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
# The shared warm-start primitive, also used by from_working_point to ramp to a working point.
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

% if dyn_param_lazy or any_coupling_lazy or (noise_cov and noise_cov['lazy']):
def _load_param(path, key, device=True):
    """Read a sourced or produced array from its content-addressed artifact.

    Materialised at codegen time so an operator of any size never enters the generated
    source. Read once when the network is built, not per step. ``device=False`` keeps the
    array in NumPy at its stored precision — what a host-side consumer needs, since
    ``jnp.asarray`` silently truncates float64 to float32 whenever x64 is off.

    A packed kit stages these artifacts by basename, so an absent author path is resolved
    against ``$TVBO_CONSTANTS_DIR`` or the run directory's ``constants/``.
    """
    from tvbo.data.matrix_io import LazyArrayStore, resolve_staged_path
    _arr = LazyArrayStore(resolve_staged_path(path), {}).read_dataset(key)
    return jnp.asarray(_arr) if device else _arr


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
% if noise_cov:
    # Factorise the declared covariance (correlated_over: ${noise_cov['axis']}) once here, not per step.
% if noise_cov['lazy']:
    _covariance = _load_param(${repr(noise_cov['lazy'][0])}, ${repr(noise_cov['lazy'][1])}, device=False)
% else:
    _covariance = ${repr(noise_cov['value'])}
% endif
% if noise_cov_fold:
    # Amplitudes folded in; the increment above is driven at unit amplitude.
    _covariance = fold_amplitudes(_covariance, ${repr(noise_sigma_per_state)})
% endif
    solver = CorrelatedNoiseSolver(
        solver, covariance_factor(_covariance), axis=${repr(noise_cov['axis'])}
    )
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
    % if use_length_graph or weight_transform_needs_lengths:
    distances: jnp.ndarray = None,
    % endif
    % if use_delay_graph and not use_length_graph:
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
    # Declared weight `transforms:` applied to the raw weights (kit stays self-contained).
% if weight_transform_needs_lengths:
    if distances is None:
        # Zero-filling here would silently normalise by the wrong denominator.
        raise ValueError("a declared weight transform reads the tract lengths L; pass distances= to create_network")
% endif
% for _line in weight_transform_const_env:
    ${_line}
% endfor
% for expr, matrix_env in weight_transform_jax:
% for _line in matrix_env:
    ${_line}
% endfor
    weights = ${expr}
% endfor
% endif

    % if use_length_graph:
    # Recomputed each forward pass so speed stays a live graph leaf, with max_delay_bound sizing the static history buffer.
    if distances is None:
        distances = jnp.zeros_like(weights)
    _speed = ${conduction_speed}
    # Measured as DenseLengthGraph does, elementwise then max: the other order differs by a float32 ULP and trips its strict bound check.
    _max_delay_bound = max_delay if max_delay is not None else (float(jnp.max(distances / _speed)) * (1.0 + 1e-4) if _speed > 0 else 0.0)
    graph = DenseLengthGraph(weights, distances, speed=_speed, region_labels=region_labels, max_delay_bound=_max_delay_bound)
    % elif use_delay_graph and use_sparse:
    # Weights and delays share one sparsity pattern so the gather runs per edge; non-edge entries arrive as NaN and are zero-filled first.
    if delays is None:
        delays = jnp.zeros_like(weights)
    delays = jnp.nan_to_num(delays)
    graph = SparseDelayGraph(weights, delays, region_labels=region_labels, max_delay_bound=max_delay)
    % elif use_delay_graph:
    # Per-edge delays used directly; non-edge entries arrive as NaN, so zero-fill first.
    if delays is None:
        delays = jnp.zeros_like(weights)
    delays = jnp.nan_to_num(delays)
    graph = DenseDelayGraph(weights, delays, region_labels=region_labels, max_delay_bound=max_delay)
    % elif use_sparse:
    # Stored as BCOO so the reduction is an O(nnz) edge-sum rather than a dense NxN matmul.
    graph = SparseGraph(weights, region_labels=region_labels)
    % else:
    graph = DenseGraph(weights, region_labels=region_labels)
    % endif

    n_nodes = weights.shape[0]

    _dynamics_params = {
        % for name in dyn_param_names:
        % if name in node_param_overrides:
        '${name}': jnp.array([${', '.join(str(v) for v in node_param_overrides[name])}]),
        % elif name in dyn_param_lazy:
        '${name}': _load_param(${repr(dyn_param_lazy[name][0])}, ${repr(dyn_param_lazy[name][1])}),
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
    # A sourced/produced coupling parameter resolves from storage like a dynamics one; without this the `.get(name, 1.0)` fallback below would silently emit a per-edge matrix as jnp.full(shape, 1.0).
    c_param_lazy = coupling_param_lazy.get(coupling_key, {})
%>
    _${coupling_key}_params = {
        % for name in c_param_names:
        % if name in c_param_lazy:
        '${name}': _load_param(${repr(c_param_lazy[name][0])}, ${repr(c_param_lazy[name][1])}),
        % elif name in c_param_shapes:
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

    Each state variable with a ``distribution`` is redrawn per-node from it, keyed by that
    distribution's OWN resolved seed (``distribution.seed`` overriding ``execution.random_seed``)
    — so every distributed variable honours its own seed, not just the first. An explicit ``key``
    overrides the per-distribution seeds (folded per variable so the variables stay decorrelated).
    """
    ic = state.initial_state.dynamics  # (n_states,) broadcast or (n_states, n_nodes)
    # Ensure per-node shape so sampling produces independent values per node
    if ic.ndim == 1:
        ic = jnp.broadcast_to(ic[:, None], (ic.shape[0], n_nodes)).copy()
% for _si, (_sv_name, _sv_info) in enumerate(sv_distribution_info.items()):
    _k${_si} = jax.random.fold_in(key, ${_si}) if key is not None else jax.random.key(${_sv_info['seed']})
% if _sv_info['dist'] in ('gaussian', 'normal'):
    ic = ic.at[${_sv_info['idx']}].set(${(_sv_info['lo'] + _sv_info['hi']) / 2} + ${(_sv_info['hi'] - _sv_info['lo']) / 4.0} * jax.random.normal(_k${_si}, (n_nodes,)))
% else:
    ic = ic.at[${_sv_info['idx']}].set(jax.random.uniform(_k${_si}, (n_nodes,), minval=${_sv_info['lo']}, maxval=${_sv_info['hi']}))
% endif
% endfor
    state.initial_state.dynamics = ic
    return state
% endif

% if network_observation_names:
<%
    # Measure-bound observations (network.observations.* / dataset.subject.*) are materialized at run time; a connectome-matrix source is embedded by the observation template and only needs an alias, and every emit site reads both as module-level globals.
    _measure_bound_obs = sorted(n for n in network_observation_names if n in network_obs_measures)

    def _obs_edge_label(_n):
        _o = _all_observations.get(_n)
        for _s in as_list(getattr(_o, 'source', None) or []):
            _lab = edge_label(str(getattr(_s, 'name', None) or _s))
            if _lab:
                return _lab
        return None
    _edge_bound_obs = sorted((n, _obs_edge_label(n)) for n in network_observation_names if n not in network_obs_measures)
    _unbound_obs = [n for n, lab in _edge_bound_obs if lab is None]
    if _unbound_obs:
        raise ValueError(
            f"network observations {_unbound_obs!r} are neither measure-bound "
            f"(network.observations.* / dataset.subject.*) nor a connectome matrix "
            f"(network.weight / network.edges.<label>), so nothing would define them."
        )
%>
# ── Network observations (empirical targets carried by the Network) ──────────
# Declared in YAML via `source: [network.observations.<measure>]`. The name->
# measure mapping is resolved in Python (SimulationExperiment.
# network_observation_measures) and passed in as `network_obs_measures`;
# values are materialized at run_experiment() time from the network (or a
# `network_observations` override).
_NETWORK_OBS_MEASURES = {${', '.join("'%s': '%s'" % (k, v) for k, v in network_obs_measures.items())}}
% for _on in _measure_bound_obs:
${_on} = None  # network observation <- ${network_obs_measures[_on]}
% endfor

def _bind_network_observations(network_observations=None):
    """Materialize module-level network-observation constants from the given
    dict (keyed by observation name). Mirrors how `weights`/`distances` flow
    into the experiment; raises a clear error if a declared one is missing."""
    network_observations = network_observations or {}
% for _on in _measure_bound_obs:
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
# Positions of the labels shared with each by_label target, so the loss gathers the simulated observable label for label.
_DATASET_RECON_IDX = {
% for _tname, _idx in _ds_recon_idx.items():
    '${_tname}': jnp.array(${_idx}),
% endfor
}

def _gather2d(matrix, idx):
    """Select the shared nodes on both axes of a (node, node) matrix by index."""
    return matrix[jnp.ix_(idx, idx)]
% endif
<% _ds_arrays = data_source_arrays(experiment) %>
% if _ds_arrays:
# What each `data_source.<key>` pipeline argument points at; `_bind_data_sources` turns it into numbers.
_DATA_SOURCE_SPECS = {
% for _k, _spec in _ds_arrays.items():
    ${repr(_k)}: {'path': ${repr(_spec["path"])}, 'edge': ${repr(_spec["edge"])}},
% endfor
}
_DATA_SOURCES = {}


def _bind_data_sources(spec_dir=None):
    """Load every declared `data_source` array into `_DATA_SOURCES`.

    Bound once, so each array is a traced constant inside the observable and no pipeline step
    opens a file per grid cell. `spec_dir` overrides where the companion networks are looked
    for; by default the `spec/` directory beside this module is used, which is where an emitted
    kit puts them.

    A kit puts the companions in `spec/` beside the script, while a study run execs this module
    with no `__file__` at all and the spec sits under the working directory, so every base is
    tried rather than making the caller know which shape it is in. The declared path is tried
    first, then its bare file name: a spec frozen into a kit carries the bundled name while the
    module beside it may still carry the path the recipe declared.
    """
    import pathlib

    from tvbo.classes.network import Network

    _f = globals().get('__file__')
    _here = pathlib.Path(_f).resolve().parent if _f else pathlib.Path.cwd()
    bases = [pathlib.Path(spec_dir)] if spec_dir else [
        _here / 'spec', pathlib.Path.cwd() / 'spec', _here.parent / 'spec', pathlib.Path.cwd(),
    ]
    for _key, _spec in _DATA_SOURCE_SPECS.items():
        _declared = pathlib.Path(_spec['path'])
        _candidates = [_declared] if _declared.is_absolute() else [b / _declared for b in bases]
        _candidates += [b / _declared.name for b in bases]
        _path = next((c for c in _candidates if c.exists()), _candidates[0])
        if not _path.exists():
            raise FileNotFoundError(
                f"data_source {_spec['path']!r} for '{_key}' was not found "
                f"(looked in {', '.join(str(b) for b in bases)}); re-emit the kit so the "
                "network travels with it, or pass spec_dir."
            )
        _matrix = Network.from_file(str(_path)).matrix(_spec['edge'])
        if _matrix is None:
            raise KeyError(f"{_spec['path']!r} carries no {_spec['edge']!r} edge.")
        _DATA_SOURCES[_key] = jnp.asarray(_matrix)


_bind_data_sources()

% endif
# Every IC site builds the same way — sampled defaults, declared per-node overrides, then an optional supplied operating point — applied by name through _STATE_INDEX.
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

# The settled operating point another experiment reached, keyed by state-variable name; None is a no-op.
_SEED_DYNAMICS = None

def _apply_seed_dynamics(state):
    return state if _SEED_DYNAMICS is None else _set_rows(state, _SEED_DYNAMICS)

# The source run's whole recorded branch, which a branch-restart exploration reads for its per-cell value and state pairs; None is a no-op.
_BRANCH_SEED = None

# Model-parameter values from the source run's operating point, keyed by parameter name, holding per-node vectors and per-edge matrices alike.
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

% if _state_only_aux:
def _realign_state_auxiliaries(sol, network):
    """Recompute recorded state-only derived variables from the recorded post-step
    state so they no longer lag it by one step. Coupling-dependent auxiliaries are
    left to the solver.
    """
    _ys = sol.ys
    _p = network.params.dynamics
    t = sol.ts.reshape(-1, 1)
    % for _name in dyn_param_names:
    ${_name} = _p.${_name}
    % endfor
    % for _i, _sname in enumerate(state_names):
    ${_sname} = _ys[:, ${_i}, :]
    % endfor
    % for _dp in (model.derived_parameters.values() if model.derived_parameters else []):
    ${_dp.name} = ${realign_render(_dp)}
    % endfor
    ## Bound in dependency order so a recorded auxiliary can reference the intermediates it is built from.
    % for _name in _state_only_derived:
    ${_name} = ${realign_render(model.derived_variables[_name])}
    % endfor
    % for _name, _offset in _state_only_aux:
<%    _ch = len(state_names) + _offset %>\
    _ys = _ys.at[:, ${_ch}, :].set(
        jnp.broadcast_to(jnp.atleast_1d(${_name}), _ys[:, ${_ch}, :].shape))
    % endfor
    return NativeSolution(sol.ts, _ys, dt=sol.dt, variable_names=sol.variable_names)


% endif
<%doc>
    Whether a plain forward simulation folds its observations in-carry instead of materialising the whole trajectory. An observation that reports one sample in k otherwise pays for all of them: a 1,200-frame BOLD slice out of a 1.93M-step trajectory costs about 3 GB.

    The gate is deliberately narrow. Stream only when every raw observation is itself a streaming reduction and every derived observation is computable from the streamed values alone, so that nothing needs `result` and nothing is lost by never forming it. A single non-streaming observation keeps the whole experiment on the materialise path byte for byte.
</%doc>
<%
    from tvbo.templates.tvboptim.utils import streaming_post_eval_plan as _spep
    _base_plan = _spep(experiment)
    _base_stream_names = _base_plan['names']
    _raw_obs = [n for n in observation_names
                if n not in network_observation_names and n not in derived_observation_names]
    # `reduce` rides the native block scan, so the solver family is part of the gate rather than an assumption.
    _base_stream = (bool(_base_stream_names)
                    and str(solver_class) in ('Euler', 'Heun', 'RungeKutta4')
                    and set(_raw_obs) == set(_base_stream_names)
                    and set(derived_observation_names) <= set(_base_plan['deliverables']))
    _base_bs = _base_plan['period_in_steps'] or 1000
    # Axis names for EVERY observation, from the reduction each one declares — independent of which reducers stream, so a materialised observer is labelled too.
    _obs_dims = observation_dims(experiment) or {}
%>
# observation name -> the axis names its reduction declares (utils.reduction_dims).
_OBSERVATION_DIMS = ${repr(_obs_dims)}


def _run_compiled(fn, state):
    """Execute a prepared solve under ``jax.jit``.

    ``prepare()`` hands back an UN-jitted callable that dispatches op by op for every step of
    the solve instead of running compiled — about 4.5x the wall time of the same call under
    ``jax.jit``. Every host-side evaluation of a prepared callable belongs here.

    *fn* itself is never rebound, because the same callables reach places that must keep the raw
    identity: a tuning core takes one as a STATIC argument to key its own jit cache on, and a
    loss function is traced by the optimizer.
    """
    return jax.jit(fn)(state)


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
    # A live runtime PRNG leaf, so a runtime random_seed wins over the codegen default ${random_seed}; jnp.asarray coerces an array- or tracer-valued seed instead of raising.
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
        result_transient = _run_compiled(model_fn_init, state_init)
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
    # Ramps the parameter quasi-statically to its target and seeds the IC from the settled endpoint rather than the cold one above.
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
    # A supplied operating point is the final word on the main IC, unless a transient follows and its settled state wins.
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
% if _base_stream:
    _stream_fn = None   # bound below only when run_main; keep the return's stream_fn= safe otherwise
% endif
    if run_main:
        % if has_noise:
        if getattr(state, 'noise', None) is not None:
            state.noise.key = _noise_key
        % endif
% if _base_stream:
        _stream_fn, _ = prepare(   # folds in-carry over ${_base_bs}-step blocks; no trajectory
            network, get_solver(block_size=${_base_bs}),
            t0=t0 + t_transient, t1=t0 + t_transient + t1, dt=dt,
            reduce=_compose_reducers(*[
                _STREAMING_REDUCERS[_n][0](
                    _STREAMING_REDUCERS[_n][1], dt,
                    warm_history=(None if result_transient is None
                                  else (result_transient.data if hasattr(result_transient, 'data')
                                        else result_transient)[:, _STREAMING_REDUCERS[_n][1], :]),
                )
                for _n in ${repr(_base_stream_names)}
            ]),
        )
        _stream_vals = dict(zip(${repr(_base_stream_names)}, _run_compiled(_stream_fn, state)))
        observations = Bunch(**_stream_vals)
        _all_obs = compute_all_observations(
            None, state, result_transient,
            only=${repr(sorted(derived_observation_names))}, precomputed=_stream_vals)
% for obs_name in sorted(derived_observation_names):
        observations.${obs_name} = _all_obs.${obs_name}
% endfor
% for obs_name in observation_names:
% if obs_name in network_observation_names:
        observations.${obs_name} = ${obs_name}
% endif
% endfor
% else:
        result = _run_compiled(model_fn, state)
        % if _state_only_aux:
        result = _realign_state_auxiliaries(result, network)
        % endif
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
% endif

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
% if _base_stream:
        stream_fn=_stream_fn,   # re-fold a caller's own state without materialising
% endif
    )

<%include file="tvbo-tvboptim-observation.py.mako" />

% if network_observation_names and _edge_bound_obs:
# A connectome-matrix network observation aliases the constant the observation module just embedded, under its own name — emitted after the include, where that constant comes into existence.
% for _on, _lab in _edge_bound_obs:
${_on} = ${edge_const(_lab)}
% endfor
% endif

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
from tvbo.templates.tvboptim.utils import toposort_observations, derived_equation_sample_period

sorted_observation_names = list(observation_names)
sorted_derived_obs_names = toposort_observations(sorted(derived_observation_names), derived_observations_dict, _all_observations)
%>

def _rebuilt_on(network, graph):
    """The same network over a rebuilt graph, keeping everything the graph does not carry.

    prepare() sizes the delay history buffer once from the graph it is handed, so an axis that
    can lengthen a delay is served by replacing the graph before the run rather than per cell.
    """
    return type(network)(network.dynamics, network.coupling, graph, noise=network.noise)


def _obs_data(_o):
    """Underlying array of an observation value. Monitor results wrap the array in
    ``.data``; a bare array (numpy/jax — has ``.dtype``) is returned as-is, since its
    own ``.data`` would be a raw buffer, not the array."""
    return _o if hasattr(_o, 'dtype') else getattr(_o, 'data', _o)


def _windowed_corr(_reduce, *_args, **_kw):
    """Guard a windowed correlation reducer (e.g. compute_fc) against a degenerate
    window. Pearson correlation is undefined over fewer than two retained
    timepoints, where jnp.corrcoef collapses to a 0-d scalar that then crashes the
    diagonal write. A window like this arises when a derived FC observation is
    materialized on a short simulation (e.g. one BOLD sample) — the value is not
    meaningful there, so return a NaN (n, n) matrix instead of aborting the whole
    observation pipeline. Windows with >= 2 retained samples are passed through to
    ``_reduce`` exactly as they arrived, so full FC stays byte-identical.

    The timeseries is whichever argument is an array, because a recipe binds a
    reducer's arguments by the callable's own parameter names: ``compute_fc`` takes
    its window as ``timeseries=``, and a guard that insisted on a positional one
    would only ever see the reducers that happen to be called that way."""
    _ts = _args[0] if _args else next((_v for _v in _kw.values() if hasattr(_v, 'shape')), None)
    if _ts is not None and _ts.shape[0] - int(_kw.get('skip_t', 0)) < 2:
        _n = _ts.shape[-1]
        return jnp.full((_n, _n), jnp.nan).at[jnp.diag_indices(_n)].set(0)
    return _reduce(*_args, **_kw)


UNRECORDED_OBSERVATIONS = ${repr(sorted(unrecorded_observation_names))}


def keep_recorded(obs):
    """Drop the observations declared ``record: false`` from one grid point's bundle.

    They are computed, because what the recipe does keep is derived from them, but they are
    not stacked over the sweep: an intermediate trajectory is typically an order of magnitude
    larger than every deliverable together, and the sweep returns one per cell. Filtering
    here rather than at save time is what keeps it out of the gather. A filter that would
    empty the bundle is ignored, so a recipe marking everything unrecorded still returns
    something to package.
    """
    if not UNRECORDED_OBSERVATIONS:
        return obs
    kept = {k: v for k, v in obs.items() if k not in UNRECORDED_OBSERVATIONS}
    return Bunch(**kept) if kept else obs


def compute_all_observations(result, state, result_transient=None, only=None, network_obs=None, precomputed=None, analysis_names=None, network=None):
    # ``only`` restricts computation to the named observations, keeping a non-jittable one out of the trace; ``precomputed`` seeds values folded in-carry elsewhere so derived observations need no trajectory.
    obs = Bunch()

    # A `network_obs` entry wins over the module-level constant, so a caller scoring against its own target is not scored against whatever was last bound.
    _no = network_obs or {}
% for obs_name in sorted(network_observation_names):
    obs.${obs_name} = _no['${obs_name}'] if '${obs_name}' in _no else ${obs_name}
% endfor

    # Seeded from the in-carry fold, so derived observations need no materialised trajectory.
    for _pk, _pv in (precomputed or {}).items():
        obs[_pk] = _pv

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
    if (only is None or '${obs_name}' in only) and '${obs_name}' not in (precomputed or {}):
        # ${obs_name}: observation derived from simulation state
        _${obs_name}_monitor = ${obs_class}(history=result_transient)
        _${obs_name}_result = _${obs_name}_monitor(result)
        # Keep full result to preserve named outputs (e.g., .psd, .frequencies)
        obs.${obs_name} = _${obs_name}_result
% endif
% endfor

    # Derived observations in dependency order: one whose source is itself derived must follow it, or its `hasattr(obs, src)` guard is false and it is silently skipped.
% for dobs_name in sorted_derived_obs_names:
<% dobs = derived_observations_dict[dobs_name] %>\
<%
    # Source names of this derived observation, filtered to entries that
    # name another observation in the experiment.
    src_obs_list = []
    src_node_arrays = {}   # equation symbol -> embedded per-node constant (network.nodes.<attr>)
    for so in (dobs.source or []):
        _so_name = str(so) if not hasattr(so, 'name') else str(so.name)
        if _so_name in _all_observations:
            src_obs_list.append(_so_name)
        elif node_label(_so_name):
            # Bound under its bare attribute name, so the equation reads `(I_E - I_E_range_lo)` rather than a generated constant identifier.
            src_node_arrays[node_label(_so_name)] = node_const(node_label(_so_name))

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
            # An `equation` over other observations rather than a callable, rendered inline with each source bound to a local.
            _eq = getattr(first_stage, 'equation', None)
            if _eq is not None:
                pipeline_equation = getattr(_eq, 'rhs', None)
                pipeline_equation_params = dict(iter_parameter_values(getattr(_eq, 'parameters', None)))
                _sample_dt = derived_equation_sample_period(dobs, _all_observations, dt)
                if _sample_dt is not None and 'dt' not in pipeline_equation_params:
                    pipeline_equation_params['dt'] = _sample_dt
        # The callable path only; the equation path binds its source observations directly at emit time below.
        if pipeline_call and hasattr(first_stage, 'arguments') and first_stage.arguments:
            for arg_name, arg in first_stage.arguments.items():
                arg_value = getattr(arg, 'value', None)
                # Only include arguments that have explicit values (not just names/descriptions)
                if arg_name and arg_value is not None:
                    val_str = str(arg_value)
                    # Check if value is an observation reference vs a literal
                    if val_str in src_obs_list or val_str in observation_names or val_str in derived_observation_names:
                        # Bound by name, since `arguments:` is keyed by the callable's parameter; `.data` unwraps the stored monitor result, while a dotted reference keeps its named output.
                        pipeline_args.append(f"{arg_name}=_obs_data(obs.{val_str})")
                    elif val_str.replace('.', '').replace('-', '').isdigit():
                        # Numeric literal - use as keyword arg
                        pipeline_args.append(f"{arg_name}={val_str}")
                    elif val_str.startswith('network.') and (_edge_lab := edge_label(val_str)):
                        # The embedded connectome constant rather than a string literal, keeping a derived observation consistent with the non-derived source path.
                        pipeline_args.append(f"{arg_name}={edge_const(_edge_lab)}")
                    elif val_str.startswith('network.') and (_node_lab := node_label(val_str)):
                        # The node-level analogue of the edge-matrix branch above.
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
    # A windowed correlation is undefined over a window under two samples, so route it through _windowed_corr to return NaN rather than crash; the family comes from the reducer registry, not a hand-kept list.
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
    # ${dobs_name}: equation over ${', '.join(list(src_obs_list) + sorted(src_node_arrays))}
    if (only is None or '${dobs_name}' in only) and all(hasattr(obs, _src) for _src in [${', '.join(f"'{s}'" for s in src_obs_list)}]):
% for _src in src_obs_list:
        ${_src} = _obs_data(obs.${_src})
% endfor
% for _sym, _const in sorted(src_node_arrays.items()):
        ${_sym} = ${_const}    # per-node array carried by the network
% endfor
        obs.${dobs_name} = ${jaxcode(pipeline_equation, pipeline_equation_params)}
% endif
% endfor
% if analysis_observations_dict:

    # Evaluated at this call's state, so a per-cell observable records the diagnostic at each swept operating point.
    if analysis_names:
        for _an_name, _an_val in compute_analysis_observations(state, network, result_transient).items():
            if _an_name in analysis_names:
                obs[_an_name] = _an_val
% endif

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
    # An activity-target objective defines the operating point by a constraint, which the linear-response observations solve deterministically rather than through the stochastic tuning loop.
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
# FreeParameter.initial_value: the marked Parameter wraps this instead of the base config's value, so the descent starts from the declared point while the base/warm-up config keeps its own.
fp_init = fp.get('initial_value', None)
# State keypath the parameter is marked on, resolved from its declared scope (the parser split the reference on its last dot, so scope + name recovers it losslessly).
_fp_scope = fp.get('coupling_key', None) or fp.get('dynamics_key', None)
fp_path = parameter_keypath(f"{_fp_scope}.{fp_name}" if _fp_scope else fp_name,
                            couplings=all_couplings, coupling_key=_to_ci_key)
fp_scope_name = fp_path.rsplit('.', 1)[0]
# Format bounds for code generation (None -> jnp.inf)
lo_str = f'{fp_lo}' if fp_lo is not None else '-jnp.inf'
hi_str = f'{fp_hi}' if fp_hi is not None else 'jnp.inf'
# Declared shape as a Python tuple ("(n_nodes, n_nodes)" -> (n_nodes, n_nodes)); a heterogeneous parameter with none declared is per-node.
if fp_shape:
    shape_str = fp_shape.strip('()').replace(' ', '')
    shape_code = '(' + shape_str + (',' if ',' not in shape_str else '') + ')'
else:
    shape_code = '(n_nodes,)'
fp_wrap = f"jnp.asarray({fp_init})" if fp_init is not None else f"init_state.{fp_path}"
%>
    # ${fp_name} - ${fp_scope_name} parameter${ ' (bounded: ' + str(fp_lo) + ' to ' + str(fp_hi) + ')' if has_bounds else ''}
% if has_bounds:
    init_state.${fp_path} = BoundedParameter(
        ${fp_wrap},
        low=${lo_str},
        high=${hi_str},
    )
% else:
    init_state.${fp_path} = Parameter(${fp_wrap})
% endif
% if fp_hetero:
    init_state.${fp_path}.shape = ${shape_code}
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
    # Every path returns a Bunch, so both must be unpacked at packaging time rather than passed as a raw array.
    returns_bunch = bundles_observations or bool(expl.get('record'))
    # When every recorded observable is a dynamics observer they fold into the integrator carry, dropping peak memory from O(batch·n_time·n_node) to O(batch·block·n_node); an element-slot axis, injected noise or a wired algorithm forces the post-scan path.
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
    # An exploration bundling every declared observation streams only when all of them are trajectory-free; one that needs the raw trajectory keeps the whole set on the materialise path.
    _bundle_plan = streaming_post_eval_plan(experiment) if bundles_observations else {'names': [], 'deliverables': [], 'period_in_steps': None}
    _bundle_stream_names = _bundle_plan['names']
    _bundled_all = set(observation_names) | set(derived_observation_names)
    _bundle_covered = set(_bundle_stream_names) | set(_bundle_plan['deliverables']) | set(network_observation_names)
    _bundle_fully_stream = (
        bundles_observations
        and bool(_bundle_stream_names)
        and not stochastic_param_info
        and not _element_axes_present
        and not analysis_observation_names
        and _bundled_all <= _bundle_covered
    )
    _bundle_bs = _bundle_plan['period_in_steps'] or 1000
    # prepare() sizes the delay buffer once from the base graph, so every axis that can lengthen a delay is read here, outside jit. A swept weight feeds no delay.
    _speed_axes = [ax for ax in expl['axes'] if ax.get('is_network') and ax.get('graph_leaf') == 'speed']
    _length_axes = [ax for ax in expl['axes'] if ax.get('is_network') and ax.get('graph_leaf') == 'lengths']
    _delay_axes = [ax for ax in expl['axes'] if ax.get('is_network') and ax.get('graph_leaf') == 'delays']

    def _swept_bound(axes, base):
        """Largest value a per-edge leaf reaches over the sweep, as the emitted expression.

        A leaf's points are scalars written across the graph's edges, so their extreme is known
        here; only a builder's may be whole matrices, and only those need a runtime max. The
        base graph's own leaf is always in the running, since an axis may sweep below it.
        """
        terms = [f"float(jnp.max({base}))"]
        terms += [repr(max(ax['values']) if 'values' in ax else float(ax['hi']))
                  for ax in axes if not ax.get('builder_expr')]
        terms += [f"float(jnp.max(_axisvals_{ax['name']}))" for ax in axes if ax.get('builder_expr')]
        return f"max({', '.join(terms)})" if len(terms) > 1 else terms[0]

    _v_min = min([v for ax in _speed_axes for v in (ax['values'] if 'values' in ax else [ax['lo']])], default=None)
    _v_bound = f"min(_v_build, {_v_min})" if _v_min is not None else "_v_build"
    # Reset per exploration; the `record:` branch narrows it to that sweep's recorded closure. Every jit/vmap decision here reads `_rec_host or has_host_pipeline_obs`, so an un-jitted observable can never reach ParallelExecution.
    _rec_host = False
%>
def ${expl['name']}(state, model_fn, result_transient=None, **kwargs):
    """${expl['label']} - ${grid_desc}."""
    _network = kwargs.get('network')
% if any(ax.get('builder_expr') for ax in expl['axes']):
    # The observations the main run computed before this exploration, defined ahead of the graph rebuild that sizes the delay buffer from a builder-produced length axis.
    _base_obs = kwargs.get('base_observations') or Bunch()
    def _bov(_name):
        assert _name in _base_obs, (
            f"builder for '${expl['name']}' references base observation '{_name}', which the "
            "main run did not compute (run mode='all' so base observations are available)")
        _o = _base_obs[_name]
        return _o.data if hasattr(_o, 'data') else _o
    # The cross-experiment counterpart of _bov, resolved by run() against results_root and keyed axis::arg.
    _builder_data = kwargs.get('builder_data') or {}
    def _bdv(_key):
        assert _key in _builder_data, (
            f"builder for '${expl['name']}' sources argument {_key!r} from another experiment "
            "via a used: DataRef, but it was not resolved — run() resolves builder_data before "
            "the run; ensure the source experiment has run and results_root points at it")
        return _builder_data[_key]
% endif
% for _lax in _length_axes + _delay_axes:
% if _lax.get('builder_expr'):
    # Materialised here, ahead of the graph rebuild that sizes the delay buffer from it, so the builder is called ONCE: the grid binding below reuses this value rather than re-evaluating an expression that may read base observations or cross-experiment data.
    _axisvals_${_lax['name']} = jnp.asarray(${_lax['builder_expr']})
% endif
% endfor
% if _speed_axes or _length_axes:
    if _network is not None and hasattr(_network.graph, 'lengths'):
        # Rebuilt once outside jit/vmap so the buffer covers the longest delay any cell can reach - the longest swept tract over the slowest swept speed; the axes then sweep the live `speed` / `lengths` leaves, which prepare() no longer re-reads.
        _v_build = ${conduction_speed}
        _lengths = _network.graph.lengths
        _length_graph = DenseLengthGraph(
            _network.graph.weights, _lengths, speed=_v_build,
            region_labels=_network.graph.region_labels,
            # A hair of headroom, so a float32 ULP never lands the buffer under the graph's own max(delay).
            max_delay_bound=${_swept_bound(_length_axes, '_lengths')} / ${_v_bound} * (1.0 + 1e-4),
        )
        _network = _rebuilt_on(_network, _length_graph)
% endif
% if _delay_axes:
    if _network is not None:
        assert hasattr(_network.graph, 'delays') and not hasattr(_network.graph, 'lengths'), (
            "a `network.edges.delay` axis sweeps the graph's `delays` leaf, which only a "
            "delay graph carries; this network measures tract lengths, so its delays are "
            "lengths / conduction_speed - sweep `network.conduction_speed` instead")
        # Same rule for the other per-edge leaf that feeds a delay, on the graph that states it directly.
        _delays = _network.graph.delays
        _delay_graph = DenseDelayGraph(
            _network.graph.weights, _delays, region_labels=_network.graph.region_labels,
            max_delay_bound=${_swept_bound(_delay_axes, '_delays')} * (1.0 + 1e-4),
        )
        _network = _rebuilt_on(_network, _delay_graph)
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
        # The main run's IC construction, so the sweep starts from the declared state rather than cold.
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
        # The main run's IC construction, so the sweep starts from the declared state rather than cold.
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
    _axis_label_by_id = {}

    def _ax(label, axis):
        """Bind an axis, remembering the path the recipe declared it as.

        `Space` names its dataframe columns after each swept leaf's pytree keypath, from which the
        declared path cannot be recovered (`network.conduction_speed` comes back as `graph.2`), so
        the label travels with the object instead of being guessed back from the column name.
        """
        _axis_label_by_id[id(axis)] = label
        return axis

    # Points of an axis that may be ARRAY-valued. Its grid coordinate is the point index (an xarray coord holds scalars), while its per-cell column holds whole arrays, so the cells are converted with the points that only exist here.
    _array_axis_points = {}
<%
    _matrix_axes = [ax for ax in expl['axes']
                    if ax.get('is_network') and network_leaf_is_matrix(ax.get('graph_leaf'))]
%>\
    % if _matrix_axes:
    # The graph's own topology, read once before any axis is bound so a second per-edge axis cannot see the first one's DataAxis in place of the weights.
    _edge_pattern = jnp.asarray(grid_state.graph.weights) != 0
    _across_edges = lambda _v: jnp.where(_edge_pattern, _v, 0.0)
    % endif
    % for ax in expl['axes']:
<% _lbl = ax.get('label', ax['name']) + (f"[{ax['element_idx']}]" if ax.get('element_idx') is not None else '') %>\
    % if ax.get('builder_expr'):
    ## An array-valued axis gets a singleton group, since product mode meshgrids only 1-D; a length axis was materialised above to size the delay buffer, so reuse it rather than re-call.
    % if not (ax.get('is_network') and ax.get('graph_leaf') == 'lengths'):
    _axisvals_${ax['name']} = jnp.asarray(${ax['builder_expr']})
    % endif
    _grp_${ax['name']} = "${ax['name']}" if _axisvals_${ax['name']}.ndim > 1 else None
    _array_axis_points["${_lbl}"] = _axisvals_${ax['name']}
    % if ax.get('is_external'):
    grid_state.external.${ax['external_key']}.${ax['name']} = _ax('${_lbl}', DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']}${event_clock_wrap(ax)}))
    % elif ax.get('is_coupling'):
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = _ax('${_lbl}', DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']}))
    % elif ax.get('is_network'):
    % if network_leaf_is_matrix(ax.get('graph_leaf')):
    # A builder may hand over one scalar per point or whole per-edge matrices; only the former is written across the edges.
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', DataAxis(
        _axisvals_${ax['name']}, group=_grp_${ax['name']},
        wrap=_across_edges if _axisvals_${ax['name']}.ndim == 1 else None))
    % else:
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']}))
    % endif
    % elif ax.get('is_noise'):
    grid_state.noise.${ax['name']} = _ax('${_lbl}', DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']}))
    % else:
    grid_state.dynamics.${ax['name']} = _ax('${_lbl}', DataAxis(_axisvals_${ax['name']}, group=_grp_${ax['name']}))
    % endif
    % elif ax.get('is_seed'):
    ## A dummy scalar slot the wrapper below turns into config.noise.key, so every cell draws an independent noise realization.
    grid_state.dynamics._noise_seed = _ax('${_lbl}', DataAxis(jnp.asarray(${ax['values']}, dtype=jnp.uint32)))
    % elif ax.get('is_ic'):
    ## A dummy scalar slot the wrapper below writes into the swept variable's row, so every cell integrates from its own IC.
    % if 'values' in ax:
    grid_state.dynamics._ic_${ax['name']} = _ax('${_lbl}', DataAxis(jnp.asarray(${ax['values']})))
    % else:
    grid_state.dynamics._ic_${ax['name']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})))
    % endif
    % elif ax.get('is_noise'):
    ## Noise-amplitude axis: a parameter leaf on the noise params, bound directly — no dummy slot, no wrapper.
    % if 'values' in ax:
    grid_state.noise.${ax['name']} = _ax('${_lbl}', DataAxis(jnp.asarray(${ax['values']})))
    % else:
    grid_state.noise.${ax['name']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})))
    % endif
    % elif ax.get('element_idx') is not None:
    ## Element-indexed parameter: create dummy scalar slot for Space discovery
    ## e.g., K[0] → grid_state.dynamics._K_el0 = GridAxis(...)
    % if 'values' in ax:
    grid_state.dynamics._${ax['name']}_el${ax['element_idx']} = _ax('${_lbl}', DataAxis(${ax['values']}))
    % else:
    grid_state.dynamics._${ax['name']}_el${ax['element_idx']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}_${ax['element_idx']}', ${ax['n']})))
    % endif
    % elif ax.get('is_external'):
    ## A `t0` is declared against the main simulation, so it rides the padded clock as a wrap and the coordinate stays the time the recipe wrote.
    % if 'values' in ax:
    grid_state.external.${ax['external_key']}.${ax['name']} = _ax('${_lbl}', DataAxis(jnp.asarray(${ax['values']}, dtype=float)${event_clock_wrap(ax)}))
    % else:
    grid_state.external.${ax['external_key']}.${ax['name']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})${event_clock_wrap(ax)}))
    % endif
    % elif ax.get('is_coupling'):
    % if 'values' in ax:
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = _ax('${_lbl}', DataAxis(${ax['values']}))
    % else:
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})))
    % endif
    % elif ax.get('is_network'):
    ## Sweeps the graph's live `${ax['graph_leaf']}` leaf directly, so every dependent quantity is recomputed each pass and the leaf stays differentiable; a swept scalar on a per-edge leaf is written across the edges as a wrap, leaving the axis 1-D.
    % if network_leaf_is_matrix(ax.get('graph_leaf')):
    % if 'values' in ax:
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', DataAxis(jnp.asarray(${ax['values']}, dtype=float), wrap=_across_edges))
    % else:
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}), wrap=_across_edges))
    % endif
    % elif 'values' in ax:
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', DataAxis(${ax['values']}))
    % else:
    grid_state.graph.${ax['graph_leaf']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})))
    % endif
    % else:
    % if 'values' in ax:
    grid_state.dynamics.${ax['name']} = _ax('${_lbl}', DataAxis(${ax['values']}))
    % else:
    grid_state.dynamics.${ax['name']} = _ax('${_lbl}', GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']})))
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
    # Every recorded observable folds into the integrator carry, so peak memory is O(batch·block·n_node) and the whole grid vmaps on one device; a passed-in model_fn falls back to the post-scan path.
    if _network is not None:
        # Streams the full window, the reducer's skip=${_stream_skip} folding only post-transient samples, so the settle happens inside the scan.
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
    # The recorded observations and everything they transitively depend on through `source` or a pipeline argument; anything else is skipped so it never traces inside the observable.
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
    # Jit the whole observable only when no recorded observation reaches a host callable.
    _rec_host = any(
        pipeline_stage_is_host(_st)
        for _n in _only_list
        for _st in (getattr(_all_observations.get(_n), 'pipeline', None) or [])
    )
%>
    # Record a declared list of observations per grid point (derived via
    # compute_all_observations, `analysis` diagnostics via compute_analysis_observations),
    # stacked over the sweep into one array per name.
% if _rec_host:
    # Host pipeline callables cannot trace under jit: jit only the solve.
    _expl_model_fn = jax.jit(_expl_model_fn)
% else:
    @jax.jit
% endif
    def observable_fn(s):
${render_recorded_observable(expl['record'], derived_observation_names, network_observation_names, list(analysis_observations_dict.keys()), only_obs=_only_list, recorded_var_names=_recorded_var_names)}
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
    # A sweep naming its outputs takes the `record:` branch above, where render_recorded_observable already evaluates the `analysis` diagnostics per cell; reaching here means `record:` is empty, so no analysis list is threaded.
    _an_arg = ""
%>
% if bundles_observations and _bundle_fully_stream:
    # Every bundled observation is trajectory-free, so the streamable ones fold into the carry and the deliverables come from the streamed values; a passed-in model_fn falls back to the materialise path.
    if _network is not None:
        _bundle_model_fn, _ = prepare(
            _network, get_solver(block_size=${_bundle_bs}),
            t0=0.0, t1=${_stream_t1}, dt=${dt},
            reduce=_compose_reducers(*[
                _STREAMING_REDUCERS[_n][0](_STREAMING_REDUCERS[_n][1], ${dt}, skip=${_stream_skip})
                for _n in ${repr(_bundle_stream_names)}
            ]),
        )
        @jax.jit
        def observable_fn(s):
            _vals = _bundle_model_fn(s)
            _pre = {_n: _v for _n, _v in zip(${repr(_bundle_stream_names)}, _vals)}
            return keep_recorded(compute_all_observations(None, s, result_transient, precomputed=_pre${_an_arg}))
    else:
% if has_host_pipeline_obs or _rec_host:
        # Host pipeline callables cannot trace under jit: jit only the solve.
        _expl_model_fn_jit = jax.jit(_expl_model_fn)
        def observable_fn(s):
            result = _expl_model_fn_jit(s)
            return keep_recorded(compute_all_observations(result, s, result_transient${_an_arg}))
% else:
        @jax.jit
        def observable_fn(s):
            result = _expl_model_fn(s)
            return keep_recorded(compute_all_observations(result, s, result_transient${_an_arg}))
% endif
% elif bundles_observations:
    # Observations declared: observable_fn returns only the reduced
    # observation values per grid point (no trajectory). Output size is
    # the sum of declared observation shapes — typically per-node or
    # per-pair statistics — rather than (T, n_states, n_nodes), so trial
    # vmaps and grid axes stay tractable.
% if has_host_pipeline_obs or _rec_host:
    # Host pipeline callables cannot trace under jit: jit only the solve.
    _expl_model_fn_jit = jax.jit(_expl_model_fn)
    def observable_fn(s):
        result = _expl_model_fn_jit(s)
        return keep_recorded(compute_all_observations(result, s, result_transient${_an_arg}))
% else:
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return keep_recorded(compute_all_observations(result, s, result_transient${_an_arg}))
% endif
% elif has_model_output and model_output_indices:
    # ``model_output_channel_index`` is a scalar for one output, dropping the variable dim, or a slice for several.
    @jax.jit
    def observable_fn(s):
        result = _expl_model_fn(s)
        return result.data[:, ${model_output_channel_index}, ...]
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
    ## config.noise.key is a live runtime leaf read per solve, so varying it per cell needs no re-prepare and composes with the vmap.
    _seed_base_fn = observable_fn
    def observable_fn(s):
        s.noise.key = jax.random.key(jnp.asarray(s.dynamics._noise_seed, dtype=jnp.uint32))
        return _seed_base_fn(s)
% endif

<%
    ic_axes = [ax for ax in expl['axes'] if ax.get('is_ic')]
%>
% if ic_axes:
    ## Each cell's swept value is written into the variable's row; the swept SV carries no distribution, so nothing resamples it after.
    _ic_base_fn = observable_fn
    def observable_fn(s):
        % for ax in ic_axes:
        s.initial_state.dynamics = s.initial_state.dynamics.at[${ax['ic_row']}].set(s.dynamics._ic_${ax['name']})
        % endfor
        return _ic_base_fn(s)
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
% if has_host_pipeline_obs or _rec_host:
    # A host pipeline callable cannot trace under jit, so trials run in a host loop with each solve still jitted.
    def observable_fn(s):
        def _run_trial(${', '.join(f'_tn_{sp}' for sp in _sp_names)}):
    % for _sp_name in _sp_names:
            s.dynamics._stoch_${_sp_name} = _tn_${_sp_name}
    % endfor
            return _base_trial_observable(s)
        _trial_outs = [_run_trial(${', '.join(f'_trial_noises_{sp}[_ti]' for sp in _sp_names)}) for _ti in range(_n_trials)]
        trial_results = jax.tree.map(lambda *_xs: jnp.stack([jnp.asarray(_x) for _x in _xs]), *_trial_outs)
% else:
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
% endif
    % if expl.get('average') == 'trials':
        return jax.tree.map(lambda _l: jnp.mean(_l, axis=0), trial_results)  # per-leaf so a Bunch of streamed observables averages over trials (jnp.mean fails on a Bunch)
    % else:
        return trial_results
    % endif
% endif

% if expl.get('n_trials', 1) > 1 and sv_distribution_info:
    # === IC-based trial parallelization: ${expl['n_trials']} trials ===
    # Keyed fold_in(fold_in(key(seed), sv_index), i): the variable index first decorrelates variables sharing a seed, the trial index keeps each IC independent of n_trials.
    _n_trials = ${expl['n_trials']}

    def _sample_ics(i):
        ic = _expl_state.initial_state.dynamics  # (n_states, n_nodes)
    % for _si, (_sv_name, _sv_info) in enumerate(sv_distribution_info.items()):
        _k${_si} = jax.random.fold_in(jax.random.fold_in(jax.random.key(${_sv_info['seed']}), ${_si}), i)
    % if _sv_info['dist'] in ('gaussian', 'normal'):
        ic = ic.at[${_sv_info['idx']}].set(${(_sv_info['lo'] + _sv_info['hi']) / 2} + ${(_sv_info['hi'] - _sv_info['lo']) / 4.0} * jax.random.normal(_k${_si}, ic[${_sv_info['idx']}].shape))
    % else:
        ic = ic.at[${_sv_info['idx']}].set(jax.random.uniform(_k${_si}, ic[${_sv_info['idx']}].shape, minval=${_sv_info['lo']}, maxval=${_sv_info['hi']}))
    % endif
    % endfor
        return ic

    _trial_ics = jax.vmap(_sample_ics)(jnp.arange(_n_trials))  # (n_trials, n_states, n_nodes)

    _base_ic_observable = observable_fn

<%
    _pmode = str(expl.get('parallel_mode') or 'auto').lower()
    _pbatch = expl.get('parallel_batch_size')
%>\
% if has_host_pipeline_obs or _rec_host:
    # A host pipeline callable cannot trace under jit, so trials run in a host loop with each solve still jitted.
    def observable_fn(s):
        def _run_trial(ic):
            s.initial_state.dynamics = ic
            return _base_ic_observable(s)
        _trial_outs = [_run_trial(_trial_ics[_ti]) for _ti in range(_n_trials)]
        trial_results = jax.tree.map(lambda *_xs: jnp.stack([jnp.asarray(_x) for _x in _xs]), *_trial_outs)
% else:
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
% endif
    % if expl.get('average') == 'trials':
        return jax.tree.map(lambda _l: jnp.mean(_l, axis=0), trial_results)  # per-leaf so a Bunch of streamed observables averages over trials (jnp.mean fails on a Bunch)
    % else:
        return trial_results
    % endif
% endif

<%
    # Only the host trial loop below writes a subset event's trial index; the stochastic-parameter and IC-distribution branches map over arrays, so every trial would silently reuse mask row 0.
    if subset_mask_events and expl.get('n_trials', 1) > 1 and (stochastic_param_info or sv_distribution_info):
        raise ValueError(
            f"exploration {expl['name']!r} draws its trials from a stochastic parameter / "
            f"initial-condition distribution, which cannot also advance the per-trial mask of "
            f"subset events {[str(e.name) for e in subset_mask_events]!r}. Declare the ensemble "
            f"as a trial-only exploration (n_trials over solver noise) instead."
        )
%>\
% if expl.get('n_trials', 1) > 1 and not stochastic_param_info and not sv_distribution_info and (has_noise or subset_mask_events):
    # ${expl['n_trials']} trials differing through live state leaves, run host-side because a per-trial observable may end in numpy, each solve still jitted.
    _n_trials = ${expl['n_trials']}

    _base_trial_observable = observable_fn

    def observable_fn(s):
        % if has_noise:
        _base_key = s.noise.key
        % endif
        def _run_trial(_ti):
            % if has_noise:
            s.noise.key = jax.random.fold_in(_base_key, _ti)
            % endif
            % for _sev in subset_mask_events:
            s.external.${_sev.name}.trial = float(_ti)
            % endfor
            return _base_trial_observable(s)
        # Runs OUTSIDE jit, so it writes the caller's live state: restore the leaves or the next cell's ensemble folds on top of trial n-1's key.
        try:
            _trial_outs = [_run_trial(_ti) for _ti in range(_n_trials)]
        finally:
            % if has_noise:
            s.noise.key = _base_key
            % endif
            % for _sev in subset_mask_events:
            s.external.${_sev.name}.trial = 0.0
            % endfor
            pass
        trial_results = jax.tree.map(lambda *_xs: jnp.stack([jnp.asarray(_x) for _x in _xs]), *_trial_outs)
    % if expl.get('average') == 'trials':
        return jax.tree.map(lambda _l: jnp.mean(_l, axis=0), trial_results)  # per-leaf so a Bunch of streamed observables averages over trials (jnp.mean fails on a Bunch)
    % else:
        return trial_results
    % endif
% endif

% if expl['algorithms']:
    # ── Algorithm-wired exploration (Exploration.algorithms) ─────────────────
    # The wired algorithm chain runs AT EACH sweep point before the observable.
    # Algorithms are iterative/stateful (Python loops, e.g. FIC) — NOT vmappable
    # Tuning integrates each algorithm's own simulation_period, then the full-duration observable runs from the point's declared ICs; `_network` is used so the delay buffer is sized from the same graph the sweep runs.
    if _network is None:
        raise ValueError(
            "algorithm-wired exploration needs a rebuildable network (run_experiment passes "
            "network=...); a passed-in model_fn carries no graph to build the tuning model from."
        )
% for _algo in expl['algorithms']:
    _tune_model_fn_${_algo['name']}, _tune_state_${_algo['name']} = prepare(_network, get_solver(), t1=${float(_algo['simulation_period'])}, dt=${dt})
% endfor
    def _algo_point_fn(_pt_state):
        import copy as _copy
        _ps = _copy.deepcopy(_pt_state)
% for _algo in expl['algorithms']:
        _ts = _copy.deepcopy(_tune_state_${_algo['name']})
        for _k in _ps.dynamics.keys():
            if not _k.startswith('_'):
                _ts.dynamics[_k] = _ps.dynamics[_k]
        for _cn in _ps.coupling.keys():
            for _k in _ps.coupling[_cn].keys():
                if not _k.startswith('_'):
                    _ts.coupling[_cn][_k] = _ps.coupling[_cn][_k]
        _ts.initial_state.dynamics = _ps.initial_state.dynamics
        _algo_res_${_algo['name']} = run_${_algo['name']}(
            _ts, _tune_model_fn_${_algo['name']}, jax.random.key(${random_seed}),
            n_iterations=${_algo['n_iterations']},
% for _hp_name, _hp_val in _algo['hyperparams'].items():
            ${_hp_name}=${_hp_val},
% endfor
## Mirrors the flat call site: data-source files from kwargs, network and dataset targets from their module-level globals.
% for _inp_name in _algo.get('input_names', []):
            ${_inp_name}=kwargs.get('${_inp_name}'),
% endfor
% for _net_obs_name in _algo.get('network_obs_inputs', []):
            ${_net_obs_name}=${_net_obs_name},
% endfor
            history=result_transient, verbose=False,
            run_post_tuning=False,
        )
        _rs = _algo_res_${_algo['name']}.state
        for _k in _rs.dynamics.keys():
            if not _k.startswith('_'):
                _ps.dynamics[_k] = _rs.dynamics[_k]
        for _cn in _rs.coupling.keys():
            for _k in _rs.coupling[_cn].keys():
                if not _k.startswith('_'):
                    _ps.coupling[_cn][_k] = _rs.coupling[_cn][_k]
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
% if has_host_pipeline_obs or _rec_host:
    # A host pipeline callable cannot trace under jit, so grid cells run sequentially with each solve still jitted.
    exec_runner = SequentialExecution(progress_ticker(grid.N, label="grid cell")(observable_fn), grid)
    _grid_outputs = list(exec_runner.run())
% else:
    _n_vmap = resolve_exploration_n_vmap(${repr(expl['n_parallel'])}, grid.N, observable_fn, _expl_state)
    _n_pmap = resolve_exploration_n_pmap(grid.N, _n_vmap)
    # Batch count for the i/N progress line: n_pmap devices × ceil(cells/n_vmap) chunks.
    _n_map = max(1, -(-grid.N // _n_pmap))
    _n_batches = max(1, _n_pmap * -(-_n_map // _n_vmap))
    exec_runner = ParallelExecution(
        progress_ticker(_n_batches, label="grid batch")(observable_fn),
        grid, n_pmap=_n_pmap, n_vmap=_n_vmap,
    )
    _grid_outputs = list(exec_runner.run())
% endif
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
            ## A vector-valued point has no scalar to key on, so the coordinate is its index; a scalar-valued one keys on the value, which an index would replace with 0..n-1 downstream.
            explored_values=(jnp.arange(len(_axisvals_${ax['name']}))
                             if _axisvals_${ax['name']}.ndim > 1 else _axisvals_${ax['name']}),
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

    # Each cell's actual parameter values, so coordinates follow the grid's OWN cell order rather than a positional reshape that assumes the declared axis order: a Space emits cells in pytree-leaf order, which differs whenever the swept axes live on different state sub-objects.
    _cell_coords = None
% if has_axes:
    _df = grid.to_dataframe()
    # Both sequences come from one flatten of one tree, so rank pairs each column with the axis that bound it and any mismatch raises.
    _bound = [_l for _l in jax.tree.leaves(grid_state, is_leaf=lambda _x: isinstance(_x, AbstractAxis)) if isinstance(_l, AbstractAxis)]
    _cell_coords = {}
    for _leaf, _col in zip(_bound, _df.columns, strict=True):
        _label = _axis_label_by_id[id(_leaf)]
        _vals = np.asarray(_df[_col].to_numpy())
        # An array-valued axis coordinates on the point INDEX, so the cells are converted here, where the materialised points are still in scope.
        _pts = _array_axis_points.get(_label)
        if _pts is not None and getattr(_pts, "ndim", 1) > 1:
            _vals = point_indices(_vals, _pts)
        _cell_coords[_label] = _vals
% endif

% if returns_bunch:
    # observable_fn returned a Bunch of reduced observations.
    # No raw trajectory to attach; wrap each observation as xr.DataArray.
    _stacked_results = None
    _stacked_ts = None
    _observations_xr = {}
    # Node labels ride along so a swept observation's node axis is selectable by label, exactly as an unswept one's is.
    _node_labels = getattr(getattr(_network, 'graph', None), 'region_labels', None)
    for _obs_key, _obs_val in _stacked.items():
        if str(_obs_key).startswith('_'):
            continue
        _arr = getattr(_obs_val, 'ys', getattr(_obs_val, 'data', _obs_val))
        _ts = getattr(_obs_val, 'ts', None)
        _observations_xr[_obs_key] = _stacked_to_dataarray(
            _arr, _axes_info, intrinsic_ts=_ts,
            n_trials=${expl.get('n_trials', 1)}, name=str(_obs_key),
            cell_coords=_cell_coords,
            dims=_OBSERVATION_DIMS.get(str(_obs_key)),
            nodes=_node_labels,
        )
% if trial_reduced_dict:
<%
    if has_axes:
        raise ValueError(
            "reduce: trials is not supported together with a parameter grid yet — "
            "declare the ensemble as a trial-only exploration (n_trials without axes)."
        )
    if expl.get('n_trials', 1) < 2:
        raise ValueError(
            f"observations {sorted(trial_reduced_dict)!r} declare reduce: trials, but "
            f"exploration {expl['name']!r} has no trial ensemble to reduce over (n_trials="
            f"{expl.get('n_trials', 1)})."
        )
%>\
    # The pipeline runs host-side on the trial-stacked source, the wrapping living in tvbo.data.types.
    from tvbo.data.types import _host_reduced_to_observations
% for _trn, _tro in trial_reduced_dict.items():
<%
    _tr_src_obj = as_list(_tro.source)[0]
    _tr_src = str(getattr(_tr_src_obj, 'name', None) or _tr_src_obj)
    _tr_stage = list(_tro.pipeline)[0]
    _tr_c = getattr(_tr_stage, 'callable', None)
    _tr_mod = getattr(_tr_c, 'module', None)
    _tr_name = getattr(_tr_c, 'name', None) or getattr(_tr_c, 'qualname', None)
    if not _tr_mod or not _tr_name:
        raise ValueError(
            f"observation {_trn!r} (reduce: trials) needs its single pipeline stage to name a "
            f"`callable` with both `module` and `name`; the stage runs host-side by dotted name."
        )
    _tr_args = getattr(_tr_stage, 'arguments', None) or {}
    _tr_items = list(_tr_args.items()) if hasattr(_tr_args, 'items') else []
    _tr_src_arg = None
    _tr_extra = []
    for _an, _ag in _tr_items:
        _av = getattr(_ag, 'value', _ag)
        if str(_av) == _tr_src:
            _tr_src_arg = str(_an)
        elif _av is not None:
            _tr_extra.append((str(_an), _av))
    _tr_kw = ''.join(f", {an}={av!r}" for an, av in _tr_extra)
%>\
    # Drops only the axis-less exploration's leading cell, so a one-node source keeps the stage's (n_trials, ...) contract.
    _tr_src_arr = np.asarray(_observations_xr['${_tr_src}'].data)
    if _tr_src_arr.ndim > 1 and _tr_src_arr.shape[0] == 1:
        _tr_src_arr = _tr_src_arr[0]
    _tr_out = ${_tr_mod}.${_tr_name}(${(_tr_src_arg + '=') if _tr_src_arg else ''}_tr_src_arr${_tr_kw})
    _observations_xr.update(_host_reduced_to_observations('${_trn}', _tr_out))
% endfor
% endif
% else:
<%
    # The stage reads its source out of _observations_xr, which only the returns_bunch path fills.
    if trial_reduced_dict:
        raise ValueError(
            f"observations {sorted(trial_reduced_dict)!r} declare reduce: trials, but "
            f"exploration {expl['name']!r} returns a raw trajectory rather than a bundle of "
            f"observations, so there is no per-trial source to stack. Give the exploration an "
            f"observation bundle (declare `record:` or observations without an `observable`)."
        )
%>\
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
        # The whole axis table: the container keeps the ARRAY-valued entries as keyed sidecars and drops the scalar ones.
        axis_points=_array_axis_points,
        # Guarded with the slicing, since an axis-less exploration is never sliced and calling it sharded would suppress the provenance sidecar.
        is_shard=kwargs.get('shard') is not None,
% endif
<% _obs_label = obs_name if obs_name else (', '.join(model_output_names) if has_model_output else obs_func) %>\
        observable='${_obs_label}',
        dt=${dt},
        output_names=${model_output_names if has_model_output and not obs_name else []},
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
    # ``quiet=True`` silences this call whatever the configured level; otherwise the tvbo logger level decides.
    _quiet = kwargs.pop("quiet", False)

    _run_t0 = time.perf_counter()

    def _log(msg):
        if not _quiet:
            logger.info("[+%.0fs] %s" % (time.perf_counter() - _run_t0, msg))
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
    network = create_network(weights, ${weight_transform_distances_arg}delays=delays, region_labels=region_labels, noise_sigma=${noise_sigma_value})
    % else:
    network = create_network(weights, ${weight_transform_distances_arg}region_labels=region_labels, noise_sigma=${noise_sigma_value})
    % endif

    # Determine if we need to run main simulation or just transient.
    # For algorithm/optimization/exploration modes, we only need transient - main simulation runs after
<%doc>
    ## An algorithm (e.g. FIC/EIB tuning) runs its own simulations and IS the
    ## experiment's deliverable, so a full-length base forward-sim before it is
    ## spurious — it integrates the UNTUNED operating point whose observations no
    ## one consumes, yet materializes the whole trajectory. At Schirner's fitting
    ## length (10 h biological time, 36M steps) that base sim alone is ~440 GB and
    ## OOMs before tuning even starts. `run_simulation` still returns model_fn/state
    ## for the algorithm when run_main is False; only the materialized `result` is
    ## skipped. 'simulation' mode still forces it (an explicit forward-sim request).
</%doc>
    % if has_algorithms:
    run_main = mode in ('simulation',)
    % else:
    run_main = mode in ('simulation', 'all', None)
    % endif

    # Run simulation to get model_fn and state (includes transient settling if configured)
    sim_result = run_simulation(network, t1=${t1_default}, dt=${dt}, t_transient=${transient_time}, run_main=run_main, random_seed=kwargs.get('random_seed'))
    model_fn = sim_result.model_fn
    default_state = sim_result.state
    # Raw transient result for observation monitors (HRF warmup)
    transient = sim_result.result_transient
    _log(f"  Simulation period: ${t1_default} ${time_unit}, dt: ${dt} ${time_unit}")
    _log(f"  Transient period: ${transient_time} ${time_unit}")

    # Use custom state if provided (e.g., from previous optimization)
    if state is not None:
        # Merge custom state parameters into the default state structure
        # This preserves internal state (_internal, coupling history, etc.)
        # while using the custom dynamics/coupling parameters
        use_state = copy.deepcopy(default_state)

        # `getattr(...) is not None` rather than `hasattr`, which a Bunch always answers True, so a partial custom state does not reach `None.keys()` below.
        if getattr(state, 'dynamics', None) is not None:
            for key in state.dynamics.keys():
                if not key.startswith('_'):
                    val = state.dynamics[key]
                    # Extract value from Parameter if needed
                    if hasattr(val, 'value'):
                        val = val.value
                    use_state.dynamics[key] = val

        # Copy coupling parameters from custom state (partial-state-safe, see above).
        if getattr(state, 'coupling', None) is not None:
            for coupling_name in state.coupling.keys():
                if not coupling_name.startswith('_') and coupling_name in use_state.coupling:
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
% if _base_stream:
        # Re-folded through the same reducers, since materialising here would defeat the reduction and pair a custom-state trajectory with the default state's observations.
        _custom_stream = _run_compiled(sim_result.stream_fn, use_state) if run_main else None
        result = None
% else:
        _custom_stream = None
        if run_main:
            result = _run_compiled(model_fn, use_state)
        else:
            result = None
% endif
        state = use_state
    else:
        _custom_stream = None
        state = default_state
        # Use raw result directly (run_simulation now returns raw results)
        result = sim_result.result

    # Compute observations only if main simulation was run
% if _base_stream:
    # Every observation folded in-carry, so the run_simulation Bunch is the answer, or the re-fold computed above for a caller-supplied state.
    if not run_main:
        observations = None
    elif _custom_stream is not None:
        observations = Bunch(**dict(zip(${repr(_base_stream_names)}, _custom_stream)))
    else:
        observations = sim_result.observations
% else:
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
% endif

## Consumed by every algorithm, and by optimization unless it refines or integrates its own.
<% _needs_initial_state = bool(algorithms_list) or (has_optimization and not has_refine and (opt_depends_on or not opt_has_custom_integration)) %>\
% if _needs_initial_state:
    # Starting point for the algorithms and optimization below, before either modifies `state`.
    initial_state = copy.deepcopy(state)
% endif

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
    main_result = SimulationResult(result=_main_sel, observations=observations, state_names=${_output_names}, nodes=region_labels, observation_dims=_OBSERVATION_DIMS, transient=transient_result) if (_main_sel is not None or observations is not None) else None
    % else:
    transient_result = SimulationResult(result=transient, state_names=${result_var_names}, nodes=region_labels) if transient is not None else None
    main_result = SimulationResult(result=result, observations=observations, state_names=${result_var_names}, nodes=region_labels, observation_dims=_OBSERVATION_DIMS, transient=transient_result) if (result is not None or observations is not None) else None
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
            _algo_wall0 = time.perf_counter()

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
                # Both forms are bound as module-level globals by _bind_network_observations, so both must be forwarded as external inputs.
                if _src and (str(_src).startswith('network.observations.')
                             or str(_src).startswith('dataset.subject')):
                    network_obs_inputs.append(obs_name)

    # Get dependencies for this algorithm
    algo_deps = algorithms_deps.get(algo_name, [])
    has_deps = len(algo_deps) > 0

    # Shared with the algorithm template so the two sides cannot drift: non-empty `names` means the post-tuning model folds in-carry and materialises no trajectory.
    _pp = streaming_post_eval_plan(experiment)
    _pp_names = _pp['names']
    _pp_bs = _pp['period_in_steps']
%>
            if algorithm_name == '${algo_name}':
                # Create algorithm-specific model_fn with simulation_period
                # Use get_solver() to ensure consistent solver config (with BoundedSolver if needed)
                algo_model_fn, algo_state = prepare(network, get_solver(), t1=${float(algo_sim_period)}, dt=${dt})

% if _pp_names:
                # Folds ${', '.join(_pp_names)} into the carry so the ${t1_default}ms trajectory is never materialised; block size ${_pp_bs} is a multiple of the reducer period, aligning TR boundaries to block boundaries.
                post_model_fn, post_state = prepare(
                    network, get_solver(block_size=${_pp_bs}), t1=${t1_default}, dt=${dt},
                    reduce=_compose_reducers(*[
                        _STREAMING_REDUCERS[_n][0](
                            _STREAMING_REDUCERS[_n][1], ${dt},
                            warm_history=(None if transient is None
                                          else (transient.data if hasattr(transient, 'data') else transient)[:, _STREAMING_REDUCERS[_n][1], :]),
                            # A single non-vmapped long fold, so progress streams live from inside the scan.
                            progress=True,
                        )
                        for _n in ${repr(_pp_names)}
                    ]),
                )
% else:
                # Create post-tuning model_fn/state using experiment-level integration duration
                # (needed for full-length BOLD simulation for FC computation)
                post_model_fn, post_state = prepare(network, get_solver(), t1=${t1_default}, dt=${dt})
% endif

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
        _sd = {'n_iterations': int(_st.n_iterations),
               'reset_state': bool(getattr(_st, 'reset_state', False) or False)}
        for _hp_name, _hp_val in hyperparams_dict.items():
            _sd[_hp_name] = _over.get(_hp_name, _hp_val)
        stage_defs.append(_sd)
    any_stage_resets = any(sd.get('reset_state') for sd in stage_defs)
    # Algorithm.evaluate: False keeps only the tuned state, skipping a full-duration post-fold.
    _ev = getattr(algo, 'evaluate', None)
    algo_evaluate = True if _ev is None else bool(_ev)

    # Own rules plus combined includes (a nested inner tunes inside the outer call, not across stages); each entry is (param_name, coupling_key or None), where None lives on state.dynamics.
    reset_targets = []
    if any_stage_resets:
        _cp2k = {}
        _net_r = experiment.network
        if _net_r and getattr(_net_r, 'coupling', None):
            for _ck_r, _co_r in _net_r.coupling.items():
                for _pn_r in ((getattr(_co_r, 'parameters', None) or {})).keys():
                    _cp2k[str(_pn_r)] = _to_ci_key(_ck_r)
        _rule_algos = [algo]
        for inc in (getattr(algo, 'includes', None) or []):
            if str(getattr(inc, 'mode', 'combined') or 'combined') == 'nested':
                continue
            _inc_name, _ = get_include_info(inc)
            _inc_algo = algorithms_dict.get(_inc_name)
            if _inc_algo is not None:
                _rule_algos.append(_inc_algo)
        _dyn_params_r = set((getattr(getattr(experiment, 'dynamics', None), 'parameters', None) or {}).keys())
        _seen_t = set()
        for _ra in _rule_algos:
            for _rule in (getattr(_ra, 'update_rules', None) or []):
                _tp = getattr(_rule, 'target_parameter', None)
                _tn_r = str(getattr(_tp, 'name', _tp))
                if _tn_r and _tn_r not in _seen_t:
                    _seen_t.add(_tn_r)
                    _gk_r = _cp2k.get(_tn_r)
                    # Without a coupling key the target is grafted off state.dynamics, so a name in neither namespace would emit a tree_at on a path that does not exist.
                    if _gk_r is None and _dyn_params_r and _tn_r not in _dyn_params_r:
                        raise ValueError(
                            f"algorithm {algo_name!r} declares reset_state and an update rule "
                            f"targeting {_tn_r!r}, which is neither a network coupling parameter "
                            f"nor a dynamics parameter, so it has no state leaf to carry across "
                            f"the stage boundary."
                        )
                    reset_targets.append((_tn_r, _gk_r))

    # A varying window_size sizes the buffer at the largest stage so the scan compiles once; a constant one keeps the contiguous per-stage path.
    _stage_ws = [int(sd['window_size']) for sd in stage_defs if sd.get('window_size') is not None]
    _has_varying_window = len(set(_stage_ws)) > 1
    _max_window = max(_stage_ws) if (_has_varying_window and _stage_ws) else None

    # Splits the network-obs inputs into the per-subject targets, which are the vmap axis, and the cohort-shared ones.
    cohort_batched_inputs = [i for i in network_obs_inputs if i in _dataset_target_names]
    cohort_shared_inputs = [i for i in network_obs_inputs if i not in _dataset_target_names]
    use_cohort = _dataset_on_device and len(cohort_batched_inputs) > 0
%>
% if use_cohort:
                # One vectorised ${algo_name} fit over the whole target batch instead of a workflow job per subject, the host unstacking it after run().
                algo_result = Bunch(
                    name=algorithm_name,
                    cohort_state=run_cohort_${algo_name}(
                        algo_state, algo_model_fn, algo_key,
% for _bi in cohort_batched_inputs:
                        ${_bi}=${_bi},
% endfor
% for _si in cohort_shared_inputs:
                        ${_si}=${_si},
% endfor
                        save_every=kwargs.get('${algo_name}_save_every', kwargs.get('save_every', None)),
                        resync_every=kwargs.get('${algo_name}_resync_every', kwargs.get('resync_every', None)),
                        batch_size=kwargs.get('cohort_batch_size', ${repr(_cohort_batch_size)}),
                    ),
                    subject_ids=${repr(_cohort_subject_ids)},
                )
% elif has_stages:
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
% if any_stage_resets:
                _stage_monitors0 = _stage_monitors
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
                _stage_${src_obs}_buffer0 = _stage_${src_obs}_buffer
% endfor
% endif

                def _carry_tuned_${algo_name}(_base, _tuned):
                    """Rebuild the stage's start state: the run's initial conditions, carrying only the tuned parameters.

                    AlgorithmStage.reset_state restarts the simulation each stage, so the state
                    variables, delay history and observation buffers come from ``_base`` (the
                    algorithm's entry state) while every update rule's target is taken from
                    ``_tuned`` (the previous stage's endpoint).
                    """
                    _out = _base
% for _tn, _gk in reset_targets:
% if _gk is not None:
                    _out = eqx.tree_at(lambda _s: _s.coupling.${_gk}.${_tn}, _out, _tuned.coupling.${_gk}.${_tn})
% else:
                    _out = eqx.tree_at(lambda _s: _s.dynamics.${_tn}, _out, _tuned.dynamics.${_tn})
% endif
% endfor
                    return _out
% endif
                for _si, _sd in enumerate(_stage_defs):
                    if algo_verbose:
                        _log(f"  [{algorithm_name} stage {_si+1}/{len(_stage_defs)}] {_sd}")
% if any_stage_resets:
                    if _si > 0 and _sd.get('reset_state'):
                        _stage_state = _carry_tuned_${algo_name}(algo_state, _stage_state)
                        _stage_monitors = _stage_monitors0
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
                        _stage_${src_obs}_buffer = _stage_${src_obs}_buffer0
% endfor
% endif
                        if algo_verbose:
                            _log(f"    reset_state: restarted from the initial conditions, carrying the tuned parameters")
% endif
                    algo_result = run_${algo_name}(
                        state=_stage_state,
                        model_fn=algo_model_fn,
% if any_stage_resets:
                        key=(algo_key if _sd.get('reset_state') else jax.random.fold_in(algo_key, _si)),
% else:
                        key=jax.random.fold_in(algo_key, _si),
% endif
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
% if _has_varying_window:
                        max_window_size=${_max_window},  # fixed ring size -> tuning scan compiles once across stage window sizes
% endif
% endif
                        monitors=_stage_monitors,
                        run_post_tuning=${algo_evaluate} and (_si == len(_stage_defs) - 1),   # Algorithm.evaluate, folded once after the last stage rather than per stage
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
                    run_post_tuning=${algo_evaluate},   # Algorithm.evaluate
                    verbose=algo_verbose,
                )
% endif
% endfor

            # After trying all algorithm blocks, check if one matched and store result
            if algo_result is not None:
                # Store result for this algorithm
                algorithms_results[algorithm_name] = algo_result
                if algo_verbose:
                    _log(f"  {algorithm_name} done: {time.perf_counter() - _algo_wall0:.1f}s wall (tuning + post-tuning eval)")
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

        # Tuning moves the operating point, so the base-run diagnostics are stale; re-evaluating here makes the persisted result reflect the point the experiment settled to.
        if algo_result is not None and getattr(algo_result, 'state', None) is not None:
            if main_result is None:
                # With no base result, a shell carries the tuned diagnostics so they persist like any other observation.
                main_result = SimulationResult(result=None, state_names=${result_var_names}, nodes=region_labels, observation_dims=_OBSERVATION_DIMS, transient=None)
                results.integration = main_result
            if getattr(main_result, 'observations', None) is None:
                main_result.observations = Bunch()
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
% if has_refine or len(optimization_stages) > 1:
            # Stage results storage (use Bunch for dot-notation access)
            stage_results = Bunch()
% endif

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
            opt_network = create_network(weights, ${weight_transform_distances_arg}delays=delays, region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
            % else:
            opt_network = create_network(weights, ${weight_transform_distances_arg}region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
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
    # A by_label target carries a keyed gather, so the simulated observables are gathered onto the same shared nodes.
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
            _post_${stage_name} = _run_compiled(model_fn, _fitted_${stage_name})
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
                simulation=SimulationResult(result=_post_${stage_name}, observations=_post_${stage_name}_obs, state_names=${state_names}, nodes=region_labels, observation_dims=_OBSERVATION_DIMS),
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
            post_optimization = _run_compiled(model_fn, fitted_params)

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
                simulation=SimulationResult(result=post_optimization, observations=post_optimization_observations, state_names=${state_names}, nodes=region_labels, observation_dims=_OBSERVATION_DIMS),
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

    # Progress on stderr under TVBO_LOG_LEVEL, the same switch as ``tvbo run``.
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
    # weights RAW — create_network applies the declared transforms.
    weights = _network.matrix("weight", apply_transforms=False)
    distances = _network.lengths_matrix
    # Get region labels safely (may not be available in all BIDS datasets)
    try:
        region_labels = list(_network.labels.keys()) if _network.labels else None
    except (AttributeError, TypeError):
        region_labels = None
    logger.info("  Loaded network with %d nodes", weights.shape[0])
% else:
    # No BIDS dir: the caller injects these (weights RAW — create_network applies transforms).
    weights = globals().get("weights")
    if weights is None:
        logger.error(
            "Network weights not defined. Either configure network.bids_dir in "
            "YAML or call run_experiment() with weights."
        )
        import sys
        sys.exit(1)
    distances = globals().get("distances")
    region_labels = globals().get("region_labels")
% endif

    # Run the experiment
    # Order: 1) Simulation → 2) Explorations → 3) Algorithms → 4) Optimization
% if network_observation_names:
    # Network observations (empirical targets) keyed by observation name,
    # resolved from the loaded network via the obs->measure mapping.
    _net_obs = {}
## `_network` is bound by the bids_dir branch alone, so whether the lookup can run is decided here rather than probed at runtime.
% if bids_dir:
    if _network is not None:
        _net_obs_data = _network.observations
        _net_obs = {name: _net_obs_data[measure]
                    for name, measure in _NETWORK_OBS_MEASURES.items()
                    if measure in _net_obs_data}
% endif
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

