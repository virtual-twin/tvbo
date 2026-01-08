# -*- coding: utf-8 -*-
<%doc>TVB-Optim Experiment Template. Context: experiment (SimulationExperiment).</%doc>
<%namespace name="fn" file="/base/function-def.mako"/>
<%namespace name="const" file="/base/constants.mako"/>
<%
from tvbo.export.code import render_expression
from tvbo.templates.tvboptim.utils import (
    safe_name, as_list, get_attr, is_network_observation, obs_has_all_args,
    get_observation_refs, parse_loss_function, parse_free_param, get_domain_bounds,
    parse_exploration, get_param_info
)
import numpy as np

# Must have experiment
assert 'experiment' in context.keys(), "experiment required for experiment template"

# Direct references to experiment components (LinkML guarantees these exist)
model = experiment.local_dynamics
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

# Extract key metadata from model
state_names = list(model.state_variables.keys())
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Extract state variable bounds (for BoundedSolver)
# Collect lo/hi from state_variable.domain if present
state_bounds_lo = []
state_bounds_hi = []
for sv_name, sv in model.state_variables.items():
    lo, hi = None, None
    if hasattr(sv, 'domain') and sv.domain:
        lo = getattr(sv.domain, 'lo', None)
        hi = getattr(sv.domain, 'hi', None)
    state_bounds_lo.append(float(lo) if lo is not None else float('-inf'))
    state_bounds_hi.append(float(hi) if hi is not None else float('inf'))

# Check if any state has finite bounds (needs BoundedSolver)
has_state_bounds = any(lo != float('-inf') for lo in state_bounds_lo) or any(hi != float('inf') for hi in state_bounds_hi)

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

# First coupling input key (for parameter access)
first_coupling_key = list(coupling_inputs_dict.keys())[0] if coupling_inputs_dict else None
assert first_coupling_key, "Model must have at least one coupling_input or coupling_term"

# Build all_couplings dict from network.coupling
all_couplings = dict(network.coupling.items()) if network.coupling else {}

# Check if any coupling has delays
has_delay = any(c.delayed for c in all_couplings.values() if c)

# Collect all coupling parameters (for optimization)
all_coupling_params = {}  # (coupling_key, param_name) -> param_obj
all_coupling_param_shapes = {}  # (coupling_key, param_name) -> shape_str
coupling_param_names = set()  # Simple set of param names for quick lookup
for ck, cobj in all_couplings.items():
    if cobj and cobj.parameters:
        for p in cobj.parameters.values():
            all_coupling_params[(ck, p.name)] = p
            coupling_param_names.add(p.name)
            if p.shape and 'n_nodes' in str(p.shape):
                all_coupling_param_shapes[(ck, p.name)] = str(p.shape)

# Integration metadata
SOLVER_MAP = {'euler': 'Euler', 'heun': 'Heun', 'heunstochastic': 'Heun', 'rk4': 'RungeKutta4'}
method = (integration.method or 'euler').lower()
solver_class = SOLVER_MAP.get(method)
assert solver_class, f"Unknown solver method: {method}. Valid: {list(SOLVER_MAP.keys())}"
dt = float(integration.step_size)

# Noise configuration from state_variables or integration
noise_sigma_per_state = []
noise_targets = []
for sv_name, sv in model.state_variables.items():
    sigma = 0.0
    if sv.noise and sv.noise.parameters:
        sigma_param = sv.noise.parameters.get('sigma')
        if sigma_param:
            sigma = float(sigma_param.value if sigma_param.value is not None else sigma_param)
        if sigma > 0:
            noise_targets.append(sv_name)
    noise_sigma_per_state.append(sigma)

# Integration-level noise applies to all states if no per-state noise
if not any(s > 0 for s in noise_sigma_per_state) and integration.noise and integration.noise.parameters:
    sigma_param = integration.noise.parameters.get('sigma')
    if sigma_param:
        sigma = float(sigma_param.value if sigma_param.value is not None else sigma_param)
        noise_sigma_per_state = [sigma] * len(model.state_variables)
        noise_targets = list(model.state_variables.keys())

has_noise = any(s > 0 for s in noise_sigma_per_state)
noise_sigma = noise_sigma_per_state if len(set(noise_sigma_per_state)) > 1 else [noise_sigma_per_state[0]] if noise_sigma_per_state else [0.0]

# Network metadata
n_nodes = N_nodes = network.number_of_regions
assert network.conduction_speed is not None, "network.conduction_speed required in YAML"
conduction_speed = float(network.conduction_speed.value if hasattr(network.conduction_speed, 'value') else network.conduction_speed)

# Normalization (optional)
_norm = getattr(network, 'normalization', None)
has_normalization = _norm is not None and hasattr(_norm, 'rhs') and _norm.rhs
normalization_jax = jaxcode(_norm.rhs) if has_normalization else None

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

# Build observations dict from experiment.observations
observations_dict = dict(experiment.observations.items()) if experiment.observations else {}

# Categorize observations using utils
network_observation_names, observation_names = get_observation_refs(observations_dict)

# Class name from model
dynamics_class = model.name.replace(' ', '').replace('-', '') if model.name else 'GeneratedDynamics'

# Dynamics parameter info (shared utility)
dyn_param_names, dyn_param_defaults, dyn_param_shapes = get_param_info(model.parameters)

# === Optimization metadata ===
# Schema: experiment.optimization is multivalued dict, opt.stages is inlined_as_list
optim_list = list(experiment.optimization.values()) if experiment.optimization else []
has_optimization = len(optim_list) > 0

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
                # Simple string reference: global parameter
                optim_param_info[fp] = {'heterogeneous': False}
            else:
                # Parameter object
                pname = str(fp.name)
                # Heterogeneous if explicitly set or shape contains n_nodes
                is_hetero = fp.heterogeneous or (fp.shape and 'n_nodes' in str(fp.shape))
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
                param._coupling_key = coupling_key
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
# Schema: experiment.observations is multivalued dict
observations = dict(experiment.observations) if experiment.observations else {}

# Derived observations from schema (explicit, separate slot) - define early for use in get_pipeline_output_key
derived_observations_dict = dict(experiment.derived_observations) if experiment.derived_observations else {}
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
# Schema: experiment.explorations is multivalued dict
exploration_list = list(experiment.explorations.values()) if experiment.explorations else []
has_explorations = len(exploration_list) > 0

# Parse explorations - uses schema ifabsent defaults
# Schema defaults: n_parallel=1, mode='product'
explorations = []
for expl in exploration_list:
    assert expl.name, "exploration.name required in YAML"
    exp_info = {
        'name': expl.name,
        'label': expl.label or '',
        # mode has schema ifabsent: string(product)
        'mode': expl.mode or 'product',
        # n_parallel has schema ifabsent: integer(1)
        'n_parallel': int(expl.n_parallel) if expl.n_parallel is not None else 1,
        'axes': [],
    }
    # Schema: parameters is multivalued dict
    params = expl.parameters
    assert params, f"exploration.parameters required in YAML for {expl.name}"
    for param in params.values():
        domain = param.domain
        assert domain, f"exploration parameter domain required for {param.name}"
        assert domain.lo is not None, f"domain.lo required for {param.name}"
        assert domain.hi is not None, f"domain.hi required for {param.name}"
        assert domain.n, f"domain.n required for {param.name}"
        pname = str(param.name)
        # Check for dotted notation: ClassName.param_name
        # If prefix matches a coupling key → coupling param, else dynamics param
        source_key = None
        is_coupling_param = False
        if '.' in pname:
            prefix, pname = pname.rsplit('.', 1)
            is_coupling_param = prefix in all_couplings
            source_key = prefix
        exp_info['axes'].append({
            'name': pname,
            'lo': float(domain.lo),
            'hi': float(domain.hi),
            'n': int(domain.n),
            'is_coupling': is_coupling_param,
            'coupling_key': source_key if is_coupling_param else None,
            'dynamics_key': source_key if not is_coupling_param and source_key else None,
        })
    observable = expl.observable
    if observable:
        # FunctionCall: function attribute references the function
        func = observable.function
        func_name = func.name if hasattr(func, 'name') else str(func) if func else None
        args = observable.arguments or []

        if args:
            # FunctionCall with arguments (e.g., rmse(fc.data, target))
            exp_info['observable_type'] = 'function_call'
            exp_info['observable_func'] = func_name
            exp_info['observable_args'] = []
            for arg in args:
                arg_name = arg.name if hasattr(arg, 'name') else str(arg)
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
    explorations.append(exp_info)

has_observations = len(observations) > 0

# Parse observations - these only have source (state variable), no derived observations
obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': obs.label or '',
        'description': obs.description or '',
        'source': obs.source.name if obs.source and hasattr(obs.source, 'name') else str(obs.source) if obs.source else None,
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

import jax
% if enable_x64:
jax.config.update("jax_enable_x64", True)  # Required for stable gradient computation
% endif
import jax.numpy as jnp
import jax.scipy.signal
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling, DelayedCoupling
% if has_delay:
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
% else:
from tvboptim.experimental.network_dynamics.graph import DenseGraph
% endif
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}, BoundedSolver
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif
% if has_optimization:
import optax
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback, SavingParametersCallback
% endif
% if has_explorations:
from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution
% endif
% for mod in derived_obs_modules:
import ${mod}
% endfor

# Result classes from tvbo
from tvbo.data.types import SimulationResult, AlgorithmResult, OptimizationResult, ExplorationResult


def get_solver():
    base_solver = ${solver_class}()
% if has_state_bounds:
    return BoundedSolver(
        base_solver,
        low=jnp.array(${state_bounds_lo})[:, None],
        high=jnp.array(${state_bounds_hi})[:, None]
    )
% else:
    return base_solver
% endif

<%include file="tvbo-tvboptim-dfun.py.mako" />

<%include file="tvbo-tvboptim-cfun.py.mako" />

def create_network(
    weights: jnp.ndarray,
    % if has_delay:
    delays: jnp.ndarray = None,
    % endif
    region_labels: list = None,
    dynamics_params: dict = None,
    coupling_params: dict = None,
    noise_sigma: float = ${noise_sigma[0]},
) -> Network:
% if has_normalization:
    # Normalization: ${_norm.rhs}
    W = weights
    W_min, W_max = jnp.min(W), jnp.max(W)
    weights = ${normalization_jax}
% endif

    % if has_delay:
    if delays is None:
        delays = jnp.zeros_like(weights)
    graph = DenseDelayGraph(weights, delays, region_labels=region_labels)
    % else:
    graph = DenseGraph(weights, region_labels=region_labels)
    % endif

    n_nodes = weights.shape[0]

    _dynamics_params = {
        % for name in dyn_param_names:
        % if name in dyn_param_shapes:
        '${name}': jnp.full(${dyn_param_shapes[name]}, ${dyn_param_defaults.get(name, 1.0)}),
        % else:
        '${name}': ${dyn_param_defaults.get(name, 1.0)},
        % endif
        % endfor
    }
    if dynamics_params:
        _dynamics_params.update(dynamics_params)
    dynamics = ${dynamics_class}(**_dynamics_params)

    coupling_dict = {}

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
        '${name}': ${c_param_defaults.get(name, 1.0)},
        % endif
        % endfor
    }
    if coupling_params and '${coupling_key}' in coupling_params:
        _${coupling_key}_params.update(coupling_params['${coupling_key}'])
    coupling_dict['${coupling_key}'] = ${c_class_name}(**_${coupling_key}_params)
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

    return Network(
        dynamics=dynamics,
        coupling=coupling_dict,
        graph=graph,
        noise=noise,
    )

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

    % if has_transient:
    # Run transient simulation to settle network dynamics
    if t_transient > 0:
        model_fn_init, state_init = prepare(network, solver, t0=t0, t1=t_transient, dt=dt)
        result_transient = model_fn_init(state_init)
        network.update_history(result_transient)
    % endif

    model_fn, state = prepare(network, solver, t0=t0, t1=t1, dt=dt)

    result = None
    observations = None
    if run_main:
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
% for obs_name in derived_observation_names:
        observations.${obs_name} = _all_obs.${obs_name}
% endfor

    return Bunch(
        model_fn=model_fn,
        state=state,
        result=result,
        result_transient=result_transient,
        observations=observations,
    )

<%include file="tvbo-tvboptim-observation.py.mako" />

<%
from tvbo.export.code import render_expression

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
            for src in (dobs_def.source_observations or []):
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
                    for arg in first_stage.arguments:
                        arg_name = getattr(arg, 'name', None)
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
def get_observation_dependencies(obs_name, derived_obs_dict):
    deps = set()
    dobs_def = derived_obs_dict.get(obs_name)
    if dobs_def:
        for src in (dobs_def.source_observations or []):
            src_name = str(src) if not hasattr(src, 'name') else str(src.name)
            deps.add(src_name)
    return deps

def toposort_observations(obs_names, derived_obs_dict):
    sorted_obs = []
    visited = set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        deps = get_observation_dependencies(name, derived_obs_dict)
        for dep in deps:
            if dep in obs_names:
                visit(dep)
        sorted_obs.append(name)
    for name in obs_names:
        visit(name)
    return sorted_obs

sorted_observation_names = list(observation_names)
sorted_derived_obs_names = toposort_observations(list(derived_observation_names), derived_observations_dict)
%>

def compute_all_observations(result, state, result_transient=None):
    obs = Bunch()

    # Network observations (static data from BIDS)
% for obs_name in network_observation_names:
    obs.${obs_name} = ${obs_name}  # Module-level constant
% endfor

    # Simulated observations (computed from result) - these derive from simulation state
% for obs_name in sorted_observation_names:
<%
    if obs_name in network_observation_names:
        continue  # Skip network observations, already handled above

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
% if has_pipeline:
    # ${obs_name}: pipeline-based observation
    _${obs_name}_monitor = ${obs_class}(history=result_transient)
    _${obs_name}_result = _${obs_name}_monitor(result)
    # Keep full result to preserve named outputs (e.g., .psd, .frequencies)
    obs.${obs_name} = _${obs_name}_result
% endif
% endif
% endfor

    # Derived observations (from derived_observations in schema)
% for dobs_name, dobs in derived_observations_dict.items():
<%
    # Get source_observations (multivalued, required)
    src_obs_list = []
    for so in (dobs.source_observations or []):
        src_obs_list.append(str(so) if not hasattr(so, 'name') else str(so.name))

    # Get pipeline callable
    pipeline_call = None
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
        # Extract arguments from pipeline stage
        # Handle explicit argument values with proper observation reference resolution
        if hasattr(first_stage, 'arguments') and first_stage.arguments:
            for arg in first_stage.arguments:
                arg_name = getattr(arg, 'name', None)
                arg_value = getattr(arg, 'value', None)
                # Only include arguments that have explicit values (not just names/descriptions)
                if arg_name and arg_value is not None:
                    val_str = str(arg_value)
                    # Check if value is an observation reference vs a literal
                    if val_str in src_obs_list or val_str in observation_names or val_str in derived_observation_names:
                        # Simple observation reference - add as positional
                        positional_args.append(f"obs.{val_str}")
                    elif val_str.replace('.', '').replace('-', '').isdigit():
                        # Numeric literal - use as keyword arg
                        pipeline_args.append(f"{arg_name}={val_str}")
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
        if not positional_args and not pipeline_args:
            positional_args = [f"obs.{s}" for s in src_obs_list]

    # Build final args: positional first, then keyword
    all_args = positional_args + pipeline_args
%>
% if pipeline_call and src_obs_list:
    # ${dobs_name}: derived from ${', '.join(src_obs_list)}
    if all(hasattr(obs, _src) for _src in [${', '.join(f"'{s}'" for s in src_obs_list)}]):
        obs.${dobs_name} = ${pipeline_call}(${', '.join(all_args)})
% endif
% endfor

    return obs


<%include file="tvbo-tvboptim-algorithm.py.mako" />


% if has_optimization:
<%
# Build a lookup dict for all known parameters (dynamics + coupling)
all_dynamics_params = {str(p.name): p for p in optim_params}
# For coupling params, store (param, coupling_key) so we know where to access them
all_coupling_params = {str(p.name): (p, getattr(p, '_coupling_key', first_coupling_key)) for p in optim_coupling_params}
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
# Coupling key is explicitly set via dotted notation (e.g., FastLinearCoupling.G)
coupling_key_for_param = fp.get('coupling_key', None)
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
%>
% if is_coupling:
    # ${fp_name} - coupling parameter (${coupling_key_for_param})${ ' (bounded: ' + str(fp_lo) + ' to ' + str(fp_hi) + ')' if has_bounds else ''}
% if has_bounds:
    init_state.coupling.${coupling_key_for_param}.${fp_name} = BoundedParameter(
        init_state.coupling.${coupling_key_for_param}.${fp_name},
        low=${lo_str},
        high=${hi_str},
    )
% else:
    init_state.coupling.${coupling_key_for_param}.${fp_name} = Parameter(init_state.coupling.${coupling_key_for_param}.${fp_name})
% endif
% if fp_hetero:
    init_state.coupling.${coupling_key_for_param}.${fp_name}.shape = ${shape_code}
% endif
% else:
    # ${fp_name} - dynamics parameter${ ' (bounded: ' + str(fp_lo) + ' to ' + str(fp_hi) + ')' if has_bounds else ''}
% if has_bounds:
    init_state.dynamics.${fp_name} = BoundedParameter(
        init_state.dynamics.${fp_name},
        low=${lo_str},
        high=${hi_str},
    )
% else:
    init_state.dynamics.${fp_name} = Parameter(init_state.dynamics.${fp_name})
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
        **opt_kwargs
    )
    fitted_params, fitting_data = opt.run(marked_state, max_steps=max_steps)
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

    # Default callback: print + save loss + save state at smart intervals
    if callback is None:
        callback = MultiCallback([
            DefaultPrintCallback(every=print_every),
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
    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps)
    return fitted_params, fitting_data
% endif


% if has_explorations:

% for expl in explorations:
<%
    total_points = 1
    for ax in expl['axes']:
        total_points *= ax['n']
    obs_type = expl.get('observable_type', 'observation')
    obs_func = expl.get('observable_func', '')
    obs_args = expl.get('observable_args', [])
    obs_name = expl.get('observable', '')
    output_key = expl.get('output_key')
    grid_desc = ' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])
%>
def ${expl['name']}(state, model_fn, result_transient=None, n_pmap: int = ${n_workers}, **kwargs):
    """${expl['label']} - Grid: ${grid_desc} = ${total_points} points."""
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    % if ax.get('is_coupling'):
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}))
    % else:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=kwargs.get('n_${ax['name']}', ${ax['n']}))
    % endif
    % endfor
    grid = Space(grid_state, mode="${expl['mode']}")

    # Create observation monitors ONCE with history baked in (optimized pattern)
% if obs_type == 'function_call':
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
        result = model_fn(s)
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
    is_derived_obs = obs_name in derived_observation_names
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
% if is_derived_obs:
    # ${obs_name} is a derived observation - use compute_all_observations
    @jax.jit
    def observable_fn(s):
        result = model_fn(s)
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
        result = model_fn(s)
        obs_result = _${obs_name}_monitor(result)
% if output_key:
        return obs_result['${output_key}'] if isinstance(obs_result, dict) else obs_result.data
% else:
        return obs_result.data
% endif
% endif
% endif

    exec_runner = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec_runner.run()

    # Build axes info for ExplorationResult
    _axes_info = [
% for ax in expl['axes']:
        Bunch(
            name='${ax['name']}',
            lo=${ax['lo']},
            hi=${ax['hi']},
            n=${ax['n']},
            values=jnp.linspace(${ax['lo']}, ${ax['hi']}, ${ax['n']}),
% if ax.get('is_coupling'):
            is_coupling=True,
            coupling_key='${ax['coupling_key']}',
% endif
        ),
% endfor
    ]

    return ExplorationResult(
        name='${expl['name']}',
        grid=grid,
        results=jnp.stack(results),
        axes=_axes_info,
        observable='${obs_name if obs_name else obs_func}',
    )


% endfor
% endif


${const.all_constants(experiment)}


def run_experiment(
    weights: jnp.ndarray,
    distances: jnp.ndarray = None,
    region_labels: list = None,
    mode: str = "all",
    stage: str = None,
    state: Bunch = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run complete experiment workflow. Mode: simulation, optimization, exploration, algorithms, or all."""

    weights = jnp.array(weights)

    print("\n" + "=" * 60)
    print("STEP 1: Running simulation...")
    print("=" * 60)

    % if has_delay:
    delays = jnp.array(distances) / ${conduction_speed} if distances is not None else jnp.zeros_like(weights)
    network = create_network(weights, delays, region_labels=region_labels, noise_sigma=${noise_sigma[0]})
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=${noise_sigma[0]})
    % endif

    # Determine if we need to run main simulation or just transient
    # For algorithm/optimization/exploration modes, we only need transient - main simulation runs after
    run_main = mode in ('simulation', 'all', None)

    # Run simulation to get model_fn and state (includes transient settling if configured)
    sim_result = run_simulation(network, t1=${t1_default}, dt=${dt}, t_transient=${transient_time}, run_main=run_main)
    model_fn = sim_result.model_fn
    default_state = sim_result.state
    # Raw transient result for observation monitors (HRF warmup)
    transient = sim_result.result_transient
    print(f"  Simulation period: ${t1_default} ms, dt: ${dt} ms")
    print(f"  Transient period: ${transient_time} ms")

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
% for obs_name in derived_observation_names:
        observations.${obs_name} = _all_obs.${obs_name}
% endfor
    else:
        observations = None

    # Save initial state from simulation (before any algorithms/optimization modify it)
    # This is the starting point for optimization unless depends_on is specified
    initial_state = copy.deepcopy(state)

    main_result = SimulationResult(result=result, observations=observations) if result is not None else None
    transient_result = SimulationResult(result=transient) if transient is not None else None

    results = Bunch(
        # Core simulation infrastructure (always present)
        model_fn=model_fn,
        state=state,
        network=network,

        # Integration results (mirrors integration section in YAML)
        integration=Bunch(
            main=main_result,
            transient=transient_result,
        ),

    )
    print("  Simulation complete.")

    % if has_explorations:
    if mode in ('exploration', 'all'):
        print("\n" + "=" * 60)
        print("STEP 2: Running explorations...")
        print("=" * 60)
        exploration_result = Bunch()

        % for expl in explorations:
        print(f"  > ${expl['name']}")
        exploration_result.${expl['name']} = ${expl['name']}(
            state, model_fn,
            result_transient=transient,
            **kwargs,  # Pass runtime kwargs (e.g., target data for correlation-based observables)
        )
        % endfor

        results.exploration = exploration_result
        print("  Explorations complete.")
    % endif

    % if has_algorithms:
    if mode in ('algorithm', 'algorithms', 'all'):
        print("\n" + "=" * 60)
        print("STEP 3: Running algorithms...")
        print("=" * 60)
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
        # First, add hyperparameters from included algorithms (with argument overrides)
        for inc in (getattr(algo, 'includes', None) or []):
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
            print(f"  Algorithms to run (dependency order): {algorithms_to_run}")
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
                print(f"\\n>>> Running algorithm: {algorithm_name} (seed={_algo_seed})")
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
        # Included algorithms' observations
        for inc in (getattr(alg, 'includes', None) or []):
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
            # Check for network observation (from BIDS)
            elif hasattr(obs_def, 'source') and obs_def.source and str(obs_def.source).startswith('network.observations.'):
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
                        print(f"    (using state from dependency: {_dep_name})")
                else:
                    _source_state = initial_state
                    if algo_verbose:
                        print(f"    (dependency {_dep_name} not yet run, using initial state)")
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

    # Find source observations needed (derived observations depend on source observations)
    # With DerivedObservation, look in derived_observations_dict for source_observations
    algo_source_obs_needed = set()
    for obs_name in obs_names:
        # Check if this is a derived observation
        dobs_def = derived_observations_dict.get(obs_name)
        if dobs_def and dobs_def.source_observations:
            for src_obs in dobs_def.source_observations:
                src_name = src_obs.name if hasattr(src_obs, 'name') else str(src_obs)
                algo_source_obs_needed.add(src_name)
    algo_needs_buffers = algo_has_window_size and len(algo_source_obs_needed) > 0
%>

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
            if last_algo_name in algorithms_results:
                results.update(algorithms_results[last_algo_name])
            print("\n" + "=" * 60)
            print(f"  Algorithms complete. Results: {list(algorithms_results.keys())}")
            print("=" * 60)
        else:
            # Running single: expose result at top level
            results.update(algo_result)
            results['algorithm'] = Bunch(name=algorithm_name)
    % endif

    % if has_optimization:
    if mode in ('optimization', 'all'):
        print("\n" + "=" * 60)
        print("STEP 4: Running optimization...")
        print("=" * 60)
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
                print(f"  Skipping optimization (missing: {_missing_inputs})")
        else:
            # Stage results storage (use Bunch for dot-notation access)
            stage_results = Bunch()

% if opt_has_custom_integration:
            # Prepare fresh model_fn and state for optimization
            # Optimization has custom integration settings: ${opt_solver_class} dt=${opt_dt} t1=${opt_t1}
            print(f"  Preparing optimization model (t1=${opt_t1}ms, dt=${opt_dt}ms, solver=${opt_solver_class})")
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
            opt_network = create_network(weights, region_labels=region_labels, noise_sigma=${getattr(network, 'noise_sigma', 0.01) or 0.01})
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

% if runtime_kwargs_needed:
% for kwarg_name in sorted(runtime_kwargs_needed):
            if '${kwarg_name}' not in kwargs:
                raise ValueError("Optimization loss requires '${kwarg_name}' input (passed via kwargs)")
            ${kwarg_name} = kwargs['${kwarg_name}']
% endfor
% endif

% if loss_functions:
            # Loss function with observation monitors
% for obs_name in _lf_all_simulated:
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
            _${obs_name}_monitor = ${obs_class}(history=_opt_transient)
% endfor

            def loss_fn(state):
                result = _opt_model_fn(state)
% for obs_name in _lf_all_simulated:
                _${obs_name} = _${obs_name}_monitor(result)
% endfor
% for dobs_name in _lf_derived_obs:
<%
    dinfo = _lf_derived_info.get(dobs_name, {})
    dcall = dinfo.get('callable')
    dargs = dinfo.get('args', [])
    dsources = dinfo.get('sources', [])
    positional = [f"__{s}" for s in dsources]
    keywords = [f"{name}={val}" for name, val in dargs if str(val) not in dsources]
%>
% if dcall:
% for src in dsources:
                __${src} = _${src}.data if hasattr(_${src}, 'data') else _${src}
% endfor
                _${dobs_name} = ${dcall}(${', '.join(positional + keywords)})
% endif
% endfor
<%
    loss_arg_exprs = []
    for a in _lf_args:
        if a['type'] == 'observation':
            obs_name_arg = a['obs_name']
            if obs_name_arg in network_observation_names:
                loss_arg_exprs.append(f"kwargs.get('{obs_name_arg}', {obs_name_arg})")
            elif obs_name_arg in derived_observation_names:
                loss_arg_exprs.append(f"__{obs_name_arg}" if f"__{obs_name_arg}" in ''.join([f"__{s}" for d in _lf_derived_info.values() for s in d.get('sources', [])]) else f"_{obs_name_arg}")
            else:
                if a.get('output_key'):
                    loss_arg_exprs.append(f"_{obs_name_arg}.{a['output_key']}")
                else:
                    loss_arg_exprs.append(f"_{obs_name_arg}.data")
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

% if len(optimization_stages) > 1:
            # Multi-stage optimization with optional stage filtering
            all_stage_names = [${', '.join(f"'{s['name']}'" for s in optimization_stages)}]

            if stage is not None:
                if stage not in all_stage_names:
                    raise ValueError(f"Unknown stage '{stage}'. Available stages: {all_stage_names}")
                stages_to_run = [stage]
                print(f"  Running single stage: {stage}")
            else:
                stages_to_run = all_stage_names
                print(f"  Multi-stage optimization: ${len(optimization_stages)} stages")

% for stage_idx, stage in enumerate(optimization_stages):
<%
stage_name = stage['name']
stage_warmup_from = stage['warmup_from']
stage_max_iter = stage['max_iterations']
stage_lr = stage['learning_rate']
%>
        # Stage ${stage_idx + 1}: ${stage_name}
        if '${stage_name}' in stages_to_run:
            print(f"\n>>> Stage ${stage_idx + 1}/${len(optimization_stages)}: ${stage_name}")
            print(f"    Free parameters: ${', '.join(p['name'] for p in stage['free_parameters'])}")
% if stage_warmup_from:
            print(f"    Warmup from: ${stage_warmup_from}")
            # Use fitted_params from warmup_from stage (or from kwargs if running single stage)
            if '${stage_warmup_from}' in stage_results:
                current_state = stage_results['${stage_warmup_from}'].fitted_params
            elif 'warmup_state' in kwargs:
                # Allow passing in state from previous run
                current_state = kwargs['warmup_state']
                print(f"    Using warmup_state from kwargs")
            elif stage is not None:
                # Running single stage without warmup - use initial state with warning
                print(f"    WARNING: warmup_from='${stage_warmup_from}' not available, using initial state")
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
                simulation=SimulationResult(result=_post_${stage_name}, observations=_post_${stage_name}_obs),
                n_steps=kwargs.get('max_steps_${stage_name}', kwargs.get('max_steps', ${stage_max_iter})),
                hyperparameters=_stage_hyperparams,
            )
            current_state = _fitted_${stage_name}  # Chain to next stage

% endfor
        if stage is None:
            print("\n" + "=" * 60)
            print("  Multi-stage optimization complete")
            print("=" * 60)

        # Final results: last stage's fitted_params + per-stage access via dot notation
        results['fitted_params'] = current_state
        results['fitting_data'] = stage_results  # Bunch of all stage histories
        # Store under results.optimization for consistent access
        results['optimization'] = stage_results
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
                simulation=SimulationResult(result=post_optimization, observations=post_optimization_observations),
                n_steps=kwargs.get('max_steps', ${max_steps}),
                hyperparameters=_opt_hyperparams,
            )

            # Store under results.optimization.{name} for consistent structure
            results['optimization'] = Bunch(**{_opt_name: _opt_result})
            results[_opt_name] = _opt_result  # Also at top level for convenience
            print("  Optimization complete.")
% endif
    % endif

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)

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

    print("=" * 60)
    print("${dynamics_class} Experiment - Standalone Execution")
    print("=" * 60)

% if has_bids:
    # Load network from BIDS (BEP017)
    from tvbo import Network as TVBONetwork
    print("Loading network from BIDS: ${bids_dir}")
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
    print(f"  Loaded network with {weights.shape[0]} nodes")
% else:
    # No BIDS directory configured - check if weights available
    if 'weights' not in dir() or weights is None:
        print("ERROR: Network weights not defined.")
        print("Either configure network.bids_dir in YAML or call run_experiment() with weights.")
        import sys
        sys.exit(1)
    distances = distances if 'distances' in dir() else None
    region_labels = region_labels if 'region_labels' in dir() else None
% endif

    # Run the experiment
    # Order: 1) Simulation → 2) Explorations → 3) Algorithms → 4) Optimization
    results = run_experiment(
        weights,
        distances=distances,
        region_labels=region_labels,
        mode="all",
    )

    # Save results
    output_path = Path(__file__).parent / "${safe_name(experiment.label or 'experiment')}_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Keys: {list(results.keys())}")
    if hasattr(results, 'observations'):
        print(f"  Observations: {list(results.observations.keys())}")
    if hasattr(results, 'algorithms'):
        print(f"  Algorithms: {list(results.algorithms.keys())}")
    if hasattr(results, 'fitted_params'):
        print(f"  Optimization: Complete")

