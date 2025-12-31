# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Complete Experiment Template
======================================

Generates a complete tvboptim experiment workflow including:
- Dynamics and Coupling classes (via includes)
- Network setup and simulation
- Observations and target generation
- Parameter exploration (grid search)
- Optimization pipeline

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Complete Python module for running the full experiment
</%doc>
<%namespace name="fn" file="/base/function-def.mako"/>
<%
from tvbo.export.code import render_expression
import numpy as np

# Must have experiment
assert 'experiment' in context.keys(), "experiment required for experiment template"

model = experiment.local_dynamics
coupling = experiment.coupling
integration = experiment.integration
network = experiment.network

# Collect user-defined functions from experiment.functions
# These are functions defined in YAML (e.g., correlation, cauchy_pdf) that need to be
# recognized by the code printer. Map function name -> function name (identity mapping)
# so the printer emits them as-is rather than raising PrintMethodNotImplementedError.
_exp_functions = getattr(experiment, 'functions', None) or {}
if hasattr(_exp_functions, 'items'):
    user_functions = {str(fname): str(fname) for fname in _exp_functions.keys()}
elif hasattr(_exp_functions, '__iter__'):
    user_functions = {str(getattr(f, 'name', f)): str(getattr(f, 'name', f)) for f in _exp_functions}
else:
    user_functions = {}

# JAX code generation helpers
# (Array function mappings like sum->jnp.sum are built into the printers)
# Pass user_functions so custom functions (correlation, cauchy_pdf, etc.) are recognized
jaxcode = lambda expr, params=None: render_expression(expr, format='jax', user_functions=user_functions, parameters=params)
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Extract key metadata
state_names = list(model.state_variables.keys())
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []
coupling_terms_raw = list(model.coupling_terms.keys()) if model.coupling_terms else ['default']

# Map TVBO coupling term names to tvboptim convention
# TVBO uses c_instant/c_delayed, tvboptim uses instant/delayed
def to_tvboptim_coupling_name(name):
    """Strip 'c_' prefix for tvboptim compatibility."""
    if name.startswith('c_'):
        return name[2:]  # Remove 'c_' prefix
    return name

coupling_terms = [to_tvboptim_coupling_name(ct) for ct in coupling_terms_raw]

# Coupling metadata
has_delay = hasattr(coupling, 'delayed') and coupling.delayed

# Select the appropriate coupling term based on delay status
# Use tvboptim convention: 'delayed' or 'instant' (without c_ prefix)
if has_delay:
    # Prefer coupling term containing 'delayed', otherwise use first
    target_coupling_term = next((ct for ct in coupling_terms if 'delayed' in ct.lower()), coupling_terms[0])
else:
    # Prefer coupling term containing 'instant', otherwise use first
    target_coupling_term = next((ct for ct in coupling_terms if 'instant' in ct.lower()), coupling_terms[0])
coupling_class = coupling.name.replace(' ', '').replace('-', '')
coupling_param_names = [p.name for p in coupling.parameters.values()] if hasattr(coupling, 'parameters') and coupling.parameters else []
coupling_param_defaults = {p.name: float(p.value) for p in coupling.parameters.values() if p.value is not None} if hasattr(coupling, 'parameters') and coupling.parameters else {}
incoming_states = list(getattr(coupling, 'incoming_states', None) or [])
local_states = list(getattr(coupling, 'local_states', None) or [])

# Integration metadata - uses schema ifabsent defaults where available
SOLVER_MAP = {'euler': 'Euler', 'heun': 'Heun', 'heunstochastic': 'Heun', 'rk4': 'RungeKutta4'}
# method has schema ifabsent: string(euler)
method = (integration.method or 'euler').lower()
solver_class = SOLVER_MAP.get(method)
assert solver_class, f"Unknown solver method: {method}. Valid: {list(SOLVER_MAP.keys())}"
# step_size has schema ifabsent: float(0.01220703125)
dt = float(integration.step_size)

# Noise configuration - read directly from metadata
# Priority: 1) state_variable.noise.parameters.sigma  2) integration.noise.parameters.sigma
noise_sigma_per_state = []
noise_targets = []  # State variable names that receive noise
for sv_name, sv in model.state_variables.items():
    sigma = 0.0
    if hasattr(sv, 'noise') and sv.noise is not None:
        sv_noise = sv.noise
        if hasattr(sv_noise, 'parameters') and sv_noise.parameters:
            params = sv_noise.parameters
            if isinstance(params, dict) and 'sigma' in params:
                sigma_param = params['sigma']
                sigma = float(sigma_param.value) if hasattr(sigma_param, 'value') else float(sigma_param)
        if sigma > 0:
            noise_targets.append(sv_name)
    noise_sigma_per_state.append(sigma)

# Fallback: integration-level noise applies to all states
if not any(s > 0 for s in noise_sigma_per_state) and integration.noise is not None:
    integ_noise = integration.noise
    if hasattr(integ_noise, 'parameters') and integ_noise.parameters:
        params = integ_noise.parameters
        if isinstance(params, dict) and 'sigma' in params:
            sigma_param = params['sigma']
            sigma = float(sigma_param.value) if hasattr(sigma_param, 'value') else float(sigma_param)
            noise_sigma_per_state = [sigma] * len(model.state_variables)
            noise_targets = list(model.state_variables.keys())  # All states

has_noise = any(s > 0 for s in noise_sigma_per_state)
# For single-state models or uniform noise, use scalar sigma
noise_sigma = noise_sigma_per_state if len(set(noise_sigma_per_state)) > 1 else [noise_sigma_per_state[0]] if noise_sigma_per_state else [0.0]

# Network metadata
n_nodes = network.number_of_regions
_cs = getattr(network, 'conduction_speed', None)
# conduction_speed is simulation-specific, require explicit specification
assert _cs is not None, "network.conduction_speed required in YAML"
conduction_speed = float(_cs.value if hasattr(_cs, 'value') else _cs)

# Normalization metadata (optional) - rendered via jaxcode like DerivedVariables
_norm = getattr(network, 'normalization', None)
has_normalization = _norm is not None and hasattr(_norm, 'rhs') and _norm.rhs
normalization_jax = jaxcode(_norm.rhs) if has_normalization else None

# Simulation parameters
assert integration.duration, "integration.duration required in YAML"
t1_default = float(integration.duration)
# transient_time has schema ifabsent: float(0)
transient_time = float(integration.transient_time) if integration.transient_time else 0.0
has_transient = transient_time > 0

# Execution config - parallelization settings (generic, mapped to JAX)
_exec = getattr(experiment, 'execution', None)
n_workers = int(getattr(_exec, 'n_workers', 1) or 1) if _exec else 1
n_threads = int(getattr(_exec, 'n_threads', -1) or -1) if _exec else -1
precision = str(getattr(_exec, 'precision', 'float64') or 'float64') if _exec else 'float64'
accelerator = str(getattr(_exec, 'accelerator', 'cpu') or 'cpu') if _exec else 'cpu'
enable_x64 = precision == 'float64'
random_seed = int(getattr(_exec, 'random_seed', 0) or 0) if _exec else 0

# Observation names (for computing all observations in run_simulation)
# Include all observations that have complete argument specifications
def obs_has_all_args(obs):
    """Check if observation has all required arguments satisfied.

    First step's data/input argument is implicitly satisfied by observation source.
    """
    pipeline = getattr(obs, 'pipeline', None) or []
    has_source = getattr(obs, 'source', None) or getattr(obs, 'source_observation', None)

    for step_idx, func in enumerate(pipeline):
        is_first_step = step_idx == 0
        args = getattr(func, 'arguments', None) or []
        if hasattr(args, '__iter__'):
            for arg in args:
                arg_name = getattr(arg, 'name', None)
                arg_value = getattr(arg, 'value', None)
                if arg_name and arg_value is None:
                    # First step's data-like args are satisfied by source
                    if is_first_step and has_source and arg_name in ('data', 'X', 'x', 'input', 'timeseries', 'a'):
                        continue  # Implicitly satisfied
                    return False  # Argument without value = requires runtime input
    return True

_observations = getattr(experiment, 'observations', None) or []
if hasattr(_observations, 'items'):
    _obs_list = list(_observations.items())
elif hasattr(_observations, '__iter__'):
    _obs_list = [(getattr(o, 'name', f'obs_{i}'), o) for i, o in enumerate(_observations)]
else:
    _obs_list = []

# Include all observations that have all required arguments satisfied
# Note: We include ALL observations, not just "leaf" ones, because users expect
# to access any observation they define (e.g., simulated_bold even if simulated_fc uses it)
observation_names = [
    name for name, obs in _obs_list
    if obs_has_all_args(obs)
]

# Class names
dynamics_class = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'

# === Optimization metadata ===
optim_raw = getattr(experiment, 'optimization', None) or []
if isinstance(optim_raw, dict):
    optim_list = list(optim_raw.values())
elif isinstance(optim_raw, list):
    optim_list = optim_raw
else:
    optim_list = [optim_raw] if optim_raw else []

has_optimization = len(optim_list) > 0


# Extract optimizable parameters from optimization stages
# Track parameter info: {name: {'heterogeneous': bool}}
optim_param_info = {}

# 1. optimization.stages.free_parameters (primary source)
for opt in optim_list:
    stages = getattr(opt, 'stages', None) or []
    if hasattr(stages, 'values'):
        stages = list(stages.values())
    for stage in stages:
        free_params = getattr(stage, 'free_parameters', None) or []
        if hasattr(free_params, 'values'):
            free_params = list(free_params.values())
        for fp in free_params:
            if isinstance(fp, str):
                # Simple string: global parameter
                optim_param_info[fp] = {'heterogeneous': False}
            elif hasattr(fp, 'name'):
                # Object with name attribute
                pname = str(fp.name)
                # Check for heterogeneous flag or shape
                is_hetero = getattr(fp, 'heterogeneous', False)
                shape = getattr(fp, 'shape', None)
                if shape and 'n_nodes' in str(shape):
                    is_hetero = True
                optim_param_info[pname] = {'heterogeneous': is_hetero}
            elif isinstance(fp, dict) and 'name' in fp:
                # Dict with 'name' key
                pname = str(fp['name'])
                is_hetero = fp.get('heterogeneous', False)
                shape = fp.get('shape', None)
                if shape and 'n_nodes' in str(shape):
                    is_hetero = True
                optim_param_info[pname] = {'heterogeneous': is_hetero}

# 2. fallback: param.free (legacy) or param.heterogeneous
for name, param in model.parameters.items():
    if getattr(param, 'free', False) and str(name) not in optim_param_info:
        is_hetero = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
        optim_param_info[str(name)] = {'heterogeneous': bool(is_hetero)}

# Now collect the actual param objects with heterogeneous info
# Separate dynamics vs coupling parameters
optim_params = []  # Dynamics parameters
optim_coupling_params = []  # Coupling parameters

for name, param in model.parameters.items():
    if str(name) in optim_param_info:
        # Attach heterogeneous info to param for template use
        param._optim_heterogeneous = optim_param_info[str(name)]['heterogeneous']
        optim_params.append(param)

# Check coupling parameters
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        pname = str(name)
        if pname in optim_param_info or getattr(param, 'free', False):
            is_hetero = optim_param_info.get(pname, {}).get('heterogeneous', False)
            if not is_hetero:
                is_hetero = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
            param._optim_heterogeneous = bool(is_hetero)
            optim_coupling_params.append(param)

# Legacy support: coupling params with free=True that weren't in free_parameters
coupling_optim_params = optim_coupling_params  # Alias for backwards compatibility

# =============================================================================
# Parse ALL optimization stages into structured list
# =============================================================================
import ast  # For safely parsing stringified dicts

def parse_free_param(fp):
    """Parse a free_parameter entry which could be:
    - str: simple param name like 'w'
    - str: stringified dict like "{'name': 'w', 'heterogeneous': True}"
    - dict: actual dict with 'name' key
    - object: with .name attribute
    """
    if isinstance(fp, str):
        # Check if it looks like a stringified dict
        stripped = fp.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict) and 'name' in parsed:
                    return {
                        'name': str(parsed['name']),
                        'heterogeneous': bool(parsed.get('heterogeneous', False)),
                    }
            except (ValueError, SyntaxError):
                pass
        # Simple string param name
        return {'name': fp, 'heterogeneous': False}
    elif hasattr(fp, 'name'):
        return {
            'name': str(fp.name),
            'heterogeneous': bool(getattr(fp, 'heterogeneous', False)),
        }
    elif isinstance(fp, dict) and 'name' in fp:
        return {
            'name': str(fp['name']),
            'heterogeneous': bool(fp.get('heterogeneous', False)),
        }
    return None

optimization_stages = []
for opt in optim_list:
    stages_raw = getattr(opt, 'stages', None) or []
    if hasattr(stages_raw, 'values'):
        stages_raw = list(stages_raw.values())

    for stage in stages_raw:
        stage_info = {
            'name': str(getattr(stage, 'name', f'stage_{len(optimization_stages)}')),
            'label': str(getattr(stage, 'label', '')),
            'algorithm': str(getattr(stage, 'algorithm', 'adam')),
            'learning_rate': float(getattr(stage, 'learning_rate', 0.01) or 0.01),
            'max_iterations': int(getattr(stage, 'max_iterations', 100) or 100),
            'warmup_from': str(getattr(stage, 'warmup_from', '')) if getattr(stage, 'warmup_from', None) else None,
            'free_parameters': [],
            'hyperparameters': {},
        }

        # Parse free_parameters
        free_params = getattr(stage, 'free_parameters', None) or []
        if hasattr(free_params, 'values'):
            free_params = list(free_params.values())
        for fp in free_params:
            parsed = parse_free_param(fp)
            if parsed:
                stage_info['free_parameters'].append(parsed)

        # Parse hyperparameters
        hyperparams = getattr(stage, 'hyperparameters', None) or []
        if hasattr(hyperparams, 'values'):
            hyperparams = list(hyperparams.values())
        for hp in hyperparams:
            hp_name = getattr(hp, 'name', None)
            hp_value = getattr(hp, 'value', None)
            if hp_name and hp_value is not None:
                stage_info['hyperparameters'][str(hp_name)] = float(hp_value)

        optimization_stages.append(stage_info)

# For single-stage or default case, extract settings from first stage
optimizer_name = optimization_stages[0]['algorithm'] if optimization_stages else 'adam'
learning_rate = optimization_stages[0]['learning_rate'] if optimization_stages else 0.01
max_steps = optimization_stages[0]['max_iterations'] if optimization_stages else 100
optimizer_hyperparams = optimization_stages[0]['hyperparameters'] if optimization_stages else {}

# Schema provides ifabsent defaults, so these should always be populated
# Only assert if optimization is requested but values somehow missing
if has_optimization:
    assert optimizer_name, "optimization.algorithm not found (schema default: 'adam')"
    assert learning_rate is not None, "optimization.learning_rate not found (schema default: 0.001)"
    assert max_steps is not None, "optimization.max_iterations not found (schema default: 100)"

# === Observations metadata ===
observations_raw = getattr(experiment, 'observations', None) or {}
if hasattr(observations_raw, 'values'):
    observations = dict(observations_raw.items()) if hasattr(observations_raw, 'items') else {}
elif hasattr(observations_raw, '__iter__') and not isinstance(observations_raw, dict):
    observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations_raw)}
else:
    observations = {}

# Helper: Get observation object by name
def get_obs(name):
    """Look up observation by name from observations dict."""
    obs = observations.get(name) if hasattr(observations, 'get') else None
    if obs is None:
        for o in (observations.values() if hasattr(observations, 'values') else observations):
            if getattr(o, 'name', None) == name:
                return o
    return obs

# Helper: Get the "main" output key from an observation's pipeline
def get_pipeline_output_key(obs_name):
    """Extract the last pipeline step's output key for an observation.

    Defaults to observation name if last step has no explicit output.
    """
    obs_obj = get_obs(obs_name)
    if obs_obj:
        pipeline = getattr(obs_obj, 'pipeline', None) or []
        if pipeline:
            last_step = pipeline[-1] if hasattr(pipeline, '__getitem__') else list(pipeline)[-1]
            last_output = getattr(last_step, 'output', None)
            if last_output:
                # Handle multi-output (comma-separated) - take the last one as the "main" output
                outputs = [o.strip() for o in str(last_output).split(',')]
                return outputs[-1]
            # Default: last step output is the observation name
            return obs_name
    return obs_name

# === Exploration metadata ===
exploration_dict = getattr(experiment, 'explorations', None) or {}
if isinstance(exploration_dict, dict):
    exploration_list = list(exploration_dict.values())
elif isinstance(exploration_dict, list):
    exploration_list = exploration_dict
else:
    exploration_list = [exploration_dict] if exploration_dict else []

has_explorations = len(exploration_list) > 0

# Parse explorations - uses schema ifabsent defaults
# Schema defaults: n_parallel=1, mode='product'
explorations = []
for expl in exploration_list:
    assert hasattr(expl, 'name') and expl.name, "exploration.name required in YAML"
    exp_info = {
        'name': expl.name,
        'label': getattr(expl, 'label', ''),
        # mode has schema ifabsent: string(product)
        'mode': getattr(expl, 'mode', None) or 'product',
        # n_parallel has schema ifabsent: integer(1)
        'n_parallel': int(expl.n_parallel) if expl.n_parallel is not None else 1,
        'axes': [],
    }
    params = getattr(expl, 'parameters', None)
    assert params, f"exploration.parameters required in YAML for {expl.name}"
    param_iter = params.values() if hasattr(params, 'values') else params
    for param in param_iter:
        domain = getattr(param, 'domain', None)
        assert domain, f"exploration parameter domain required in YAML for {getattr(param, 'name', param)}"
        assert domain.lo is not None, f"domain.lo required for {getattr(param, 'name', param)}"
        assert domain.hi is not None, f"domain.hi required for {getattr(param, 'name', param)}"
        assert hasattr(domain, 'n') and domain.n, f"domain.n required for {getattr(param, 'name', param)}"
        pname = getattr(param, 'name', str(param))
        # Determine if this is a dynamics or coupling parameter
        is_coupling_param = pname in coupling_param_names
        exp_info['axes'].append({
            'name': pname,
            'lo': float(domain.lo),
            'hi': float(domain.hi),
            'n': int(domain.n),
            'is_coupling': is_coupling_param,
            'coupling_key': target_coupling_term if is_coupling_param else None,
        })
    observable = getattr(expl, 'observable', None)
    if observable:
        # FunctionCall: always has 'function' attribute
        func = getattr(observable, 'function', None)
        func_name = getattr(func, 'name', str(func)) if func else None
        args = getattr(observable, 'arguments', None) or []

        if args:
            # FunctionCall with arguments (e.g., rmse(fc.data, target))
            exp_info['observable_type'] = 'function_call'
            exp_info['observable_func'] = func_name
            exp_info['observable_args'] = []
            for arg in args:
                arg_name = getattr(arg, 'name', str(arg))
                arg_value = getattr(arg, 'value', None)
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

# Parse observations
obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': getattr(obs, 'label', ''),
        'description': getattr(obs, 'description', ''),
        'source': None,
        'source_observation': None,
        'equation': None,
    }
    if hasattr(obs, 'source') and obs.source:
        obs_info['source'] = getattr(obs.source, 'name', str(obs.source))
    if hasattr(obs, 'source_observation') and obs.source_observation:
        src_obs = obs.source_observation
        obs_info['source_observation'] = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)
    if hasattr(obs, 'equation') and obs.equation:
        obs_info['equation'] = getattr(obs.equation, 'rhs', None)
    obs_list.append(obs_info)
%>
"""
${dynamics_class} tvboptim Experiment
${'=' * (len(dynamics_class) + 20)}

Auto-generated from TVBO SimulationExperiment specification.

Experiment: ${experiment.label if hasattr(experiment, 'label') else 'Generated'}
Model: ${model.name if hasattr(model, 'name') else 'Generated'}
Coupling: ${coupling.name if hasattr(coupling, 'name') else 'Generated'}
Nodes: ${n_nodes}
Integration: ${solver_class} (dt=${dt}ms)
Stochastic: ${has_noise}
Delayed: ${has_delay}

Workflows:
- Simulation: run_simulation()
- Observations: ${len(obs_list)} defined
- Explorations: ${len(explorations)} defined
- Optimization: ${'enabled' if has_optimization else 'disabled'}
"""

# =============================================================================
# Imports
# =============================================================================
import os
import copy

%if accelerator == 'cpu':
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=${n_workers}'
% endif

import jax
% if enable_x64:
jax.config.update("jax_enable_x64", True)  # Required for stable gradient computation
% endif
import jax.numpy as jnp
import jax.scipy.signal
import numpy as np
from typing import Tuple, Dict, Any, Optional

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
% if has_delay:
from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
% else:
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
% endif
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif
% if has_optimization:
import optax
from tvboptim.types import Parameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingCallback
% endif
% if has_explorations:
from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution
% endif


# =============================================================================
# Dynamics Model
# =============================================================================

<%include file="tvbo-tvboptim-dfun.py.mako" />


# =============================================================================
# Coupling Function
# =============================================================================

<%include file="tvbo-tvboptim-cfun.py.mako" />


# =============================================================================
# Network Setup
# =============================================================================

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
    """Create configured Network instance."""
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

    dynamics = ${dynamics_class}(**(dynamics_params or {}))

    # Note: incoming_states and local_states are hardcoded in the coupling class __init__
    default_coupling_params = {
        % for name in coupling_param_names:
        '${name}': ${coupling_param_defaults.get(name, 1.0)},
        % endfor
    }
    default_coupling_params.update(coupling_params or {})
    coupling = ${coupling_class}(**default_coupling_params)

    % if has_noise:
    % if noise_targets:
    # Noise applied to states: ${noise_targets}
    noise = AdditiveNoise(sigma=noise_sigma, apply_to=${noise_targets}, key=jax.random.key(${random_seed})) if noise_sigma > 0 else None
    % else:
    # Noise applied to all states (integration-level noise without targets)
    noise = AdditiveNoise(sigma=noise_sigma, key=jax.random.key(${random_seed})) if noise_sigma > 0 else None
    % endif
    % else:
    noise = None
    % endif

    return Network(
        dynamics=dynamics,
        coupling={'${target_coupling_term}': coupling},
        graph=graph,
        noise=noise,
    )


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(
    network: Network,
    t1: float = ${t1_default},
    dt: float = ${dt},
    t0: float = 0.0,
    t_transient: float = ${transient_time},
    **kwargs,
) -> Bunch:
    """Run network simulation with optional transient settling.

    If t_transient > 0, runs a transient simulation first to settle the network,
    then updates the network's initial conditions before the main simulation.

    Parameters
    ----------
    network : Network
        Configured network instance
    t1 : float
        Main simulation duration in ms
    dt : float
        Integration timestep in ms
    t0 : float
        Start time
    t_transient : float
        Transient settling duration in ms (0 = no transient)

    Returns
    -------
    Bunch
        Contains model_fn, state, result, and optionally result_transient
    """
    solver = ${solver_class}()
    result_transient = None

    % if has_transient:
    # Run transient simulation to settle network dynamics
    if t_transient > 0:
        model_fn_init, state_init = prepare(network, solver, t0=t0, t1=t_transient, dt=dt)
        result_transient = model_fn_init(state_init)

        # Update network with final state as new initial conditions
        network.update_history(result_transient)
    % endif

    # Main simulation
    model_fn, state = prepare(network, solver, t0=t0, t1=t1, dt=dt)
    result = model_fn(state)

    # Compute all observations
    # Pass result and result_transient for observations to use
    # - result: pre-computed simulation (avoids re-running)
    # - result_transient: HRF warmup history for BOLD observations
    observations = Bunch()
    obs_kwargs = dict(kwargs)
    obs_kwargs['result'] = result
    obs_kwargs['result_transient'] = result_transient
% for obs_name in observation_names:
    observations.${obs_name} = ${obs_name}(model_fn, state, **obs_kwargs)
% endfor

    return Bunch(model_fn=model_fn, state=state, result=result, transient=result_transient, observations=observations)


# =============================================================================
# Observable Functions (Generated from Pipeline Metadata)
# =============================================================================

<%include file="tvbo-tvboptim-observation.py.mako" />


# =============================================================================
# User-Defined Functions (from experiment.functions)
# =============================================================================
<%
from tvbo.export.code import render_expression

# Get user-defined functions
exp_funcs_raw = getattr(experiment, 'functions', None) or {}
if hasattr(exp_funcs_raw, 'items'):
    exp_funcs = dict(exp_funcs_raw.items())
else:
    exp_funcs = {}

# Collect all function names for user_functions mapping
all_func_names = {str(fname): str(fname) for fname in exp_funcs.keys()}

def is_simple_callable(fdef, fname):
    """Check if function is a simple callable (just import, no wrapper needed).

    Simple = has callable + no apply_on_dimension + no equation + no source_code.
    Argument defaults in YAML are just documentation, not code to generate.
    """
    c = getattr(fdef, 'callable', None)
    if not c:
        return False
    # No apply_on_dimension (needs vmap wrapper)
    if getattr(fdef, 'apply_on_dimension', None):
        return False
    # No equation (hybrid callable+equation not supported as simple)
    if getattr(fdef, 'equation', None):
        return False
    # No source_code
    if getattr(fdef, 'source_code', None):
        return False
    return True

# Classify callables: simple (just import) vs complex (need wrapper)
simple_callable_imports = {}  # {(module, cname): fname} - direct import
complex_callable_imports = {}  # {(module, cname): _callable_cname} - prefixed import
funcs_needing_def = []  # Functions that need actual definition

for fname, fdef in exp_funcs.items():
    fname = str(fname)
    c = getattr(fdef, 'callable', None)
    if c:
        module = getattr(c, 'module', None)
        cname = getattr(c, 'name', None) or getattr(c, 'qualname', None)
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

# Initialize precomputed constants (kernel generators, etc.)
# These are computed once at module load, not on every observation call
if '_init_precomputed' in dir():
    _init_precomputed()


# =============================================================================
# Loss Functions (Generated from Metadata)
# =============================================================================
<%
# Extract loss functions from optimization metadata
# Loss is now a FunctionCall - it references a function, not defines one
# Argument value patterns:
#   - observations.simulated_psd.psd  -> call simulated_psd(), get ['psd']
#   - observations.simulated_psd      -> call simulated_psd(), get primary output
#   - (no value)                      -> runtime input (target_data)
# Aggregate patterns:
#   - aggregate.over=node, aggregate.type=mean -> vmap over axis 0, then .mean()
loss_functions = []
for opt in optim_list:
    loss_call = getattr(opt, 'loss', None)
    if loss_call:
        # FunctionCall has 'function' (reference) or 'callable' (inline)
        func_ref = getattr(loss_call, 'function', None)
        callable_ref = getattr(loss_call, 'callable', None)

        # Determine the function name to call
        if func_ref:
            func_name = str(func_ref) if isinstance(func_ref, str) else getattr(func_ref, 'name', str(func_ref))
        elif callable_ref:
            func_name = getattr(callable_ref, 'name', None) or getattr(callable_ref, 'qualname', 'loss')
        else:
            func_name = 'loss'

        # Parse aggregate specification
        aggregate = getattr(loss_call, 'aggregate', None)
        agg_over = None
        agg_type = None
        if aggregate:
            agg_over_raw = getattr(aggregate, 'over', None)
            agg_type_raw = getattr(aggregate, 'type', None)
            # Handle enum values (e.g., DimensionType.node -> 'node')
            agg_over = str(agg_over_raw).split('.')[-1] if agg_over_raw else None
            agg_type = str(agg_type_raw).split('.')[-1] if agg_type_raw else 'mean'

        # Parse arguments: value = observation reference, no value = runtime input
        loss_args = getattr(loss_call, 'arguments', []) or []
        parsed_args = []
        obs_refs = set()  # Track which observations we need to call
        for arg in loss_args:
            arg_name = getattr(arg, 'name', None)
            arg_value = getattr(arg, 'value', None)
            if arg_name:
                if arg_value:
                    val_str = str(arg_value)
                    # Parse: observations.obs_name.output_key or observations.obs_name
                    if val_str.startswith('observations.'):
                        parts = val_str.split('.', 2)  # ['observations', 'obs_name', 'output_key']
                        obs_name = parts[1] if len(parts) > 1 else None
                        output_key = parts[2] if len(parts) > 2 else None
                        if obs_name:
                            obs_refs.add(obs_name)
                            parsed_args.append({
                                'name': arg_name,
                                'type': 'observation',
                                'obs_name': obs_name,
                                'output_key': output_key,
                            })
                    else:
                        # Fallback: treat as literal or old-style obs_name.key
                        if '.' in val_str:
                            obs_name, output_key = val_str.split('.', 1)
                            obs_refs.add(obs_name)
                            parsed_args.append({
                                'name': arg_name,
                                'type': 'observation',
                                'obs_name': obs_name,
                                'output_key': output_key,
                            })
                        else:
                            # Just observation name - use primary output
                            obs_refs.add(val_str)
                            parsed_args.append({
                                'name': arg_name,
                                'type': 'observation',
                                'obs_name': val_str,
                                'output_key': None,
                            })
                else:
                    # No value = runtime input (target_data)
                    parsed_args.append({
                        'name': arg_name,
                        'type': 'runtime',
                    })

        loss_functions.append({
            'opt_name': getattr(opt, 'name', 'loss'),
            'func_name': func_name,
            'args': parsed_args,
            'obs_refs': obs_refs,
            'agg_over': agg_over,
            'agg_type': agg_type,
        })
%>
def make_loss_fn(model_fn, target_data, result_transient=None, loss_type: str = None):
    """Create a loss function closure for optimization.

    Loss functions are generated from optimization metadata.
    Each loss MUST specify equation, source_code, or callable.

    IMPORTANT: Observation monitors are created ONCE here with history baked in,
    then reused in the inner loss function. This matches the exploration pattern
    and is critical for proper JAX differentiation.

    Args:
        model_fn: Compiled model function
        target_data: Target data for fitting (e.g., empirical FC)
        result_transient: Transient simulation result for HRF/BOLD pipeline warmup
        loss_type: Which loss function to use (defaults to first available)
    """
% if loss_functions:
    # Available loss functions from metadata: ${', '.join([lf['opt_name'] for lf in loss_functions])}
    if loss_type is None:
        loss_type = "${loss_functions[0]['opt_name']}"
% for loss_fn in loss_functions:
<%
    func_name = loss_fn['func_name']
    opt_name = loss_fn['opt_name']
    args = loss_fn['args']
    obs_refs = loss_fn['obs_refs']
    agg_over = loss_fn['agg_over']
    agg_type = loss_fn['agg_type']
    # Map dimension name to axis (node=0 for arrays shaped (n_nodes, ...))
    agg_axis = 0 if agg_over == 'node' else (1 if agg_over == 'time' else None)
    # Map reduction type to JAX function
    agg_func = {'mean': 'mean', 'sum': 'sum', 'max': 'max', 'min': 'min'}.get(agg_type, 'mean')
%>
    ${'if' if loop.first else 'elif'} loss_type == "${opt_name}":
        # Pre-create observation monitors ONCE (optimized pattern for JAX differentiation)
% for obs_name in sorted(obs_refs):
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
        _${obs_name}_monitor = ${obs_class}(history=result_transient)
% endfor

        def loss_${opt_name}(state):
            """Loss function calling ${func_name}

            Observations used: ${', '.join(sorted(obs_refs)) or '(none)'}
% if agg_over:
            Aggregation: ${agg_type} over ${agg_over} (axis ${agg_axis})
% endif
            """
            # Run simulation
            result = model_fn(state)

            # Apply pre-created observation monitors
% for obs_name in sorted(obs_refs):
            _${obs_name} = _${obs_name}_monitor(result)
% endfor

            # Prepare loss function arguments
% for arg in args:
% if arg['type'] == 'observation':
% if arg['output_key']:
            ${arg['name']} = _${arg['obs_name']}.${arg['output_key']}
% else:
            ${arg['name']} = _${arg['obs_name']}.data
% endif
% else:
            ${arg['name']} = target_data
% endif
% endfor

            # Compute loss
% if agg_over and agg_axis is not None:
            # Apply ${func_name} per-${agg_over} via vmap, then aggregate with ${agg_type}
            per_element_loss = jax.vmap(${func_name})(${', '.join([a['name'] for a in args])})
            loss_value = per_element_loss.${agg_func}()
% else:
            loss_value = ${func_name}(${', '.join([a['name'] for a in args])})
% endif
            return loss_value

        return loss_${opt_name}
% endfor
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: ${', '.join([lf['opt_name'] for lf in loss_functions])}")
% else:
    raise ValueError("No loss functions defined in optimization metadata. Each optimization must specify a loss with equation, source_code, or callable.")
% endif


% if has_optimization:
# =============================================================================
# Optimization
# =============================================================================

<%
# Build a lookup dict for all known parameters (dynamics + coupling)
all_dynamics_params = {str(p.name): p for p in optim_params}
all_coupling_params = {str(p.name): p for p in optim_coupling_params}
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
    """Mark parameters as optimizable for stage: ${stage_name}

    Free parameters: ${', '.join(p['name'] for p in stage_free_params)}
    """
    # Start by unwrapping all Parameters to plain values (freeze all)
    init_state = unwrap_all_parameters(copy.deepcopy(state))

    # Now mark only this stage's free parameters as optimizable
% for fp in stage_free_params:
<%
fp_name = fp['name']
fp_hetero = fp['heterogeneous']
# Check if it's a dynamics or coupling param
is_coupling = fp_name in all_coupling_params and fp_name not in all_dynamics_params
# If in both, prefer dynamics
%>
% if is_coupling:
    # ${fp_name} - coupling parameter
    init_state.coupling.${target_coupling_term}.${fp_name} = Parameter(init_state.coupling.${target_coupling_term}.${fp_name})
% if fp_hetero:
    init_state.coupling.${target_coupling_term}.${fp_name}.shape = (n_nodes,)
% endif
% else:
    # ${fp_name} - dynamics parameter
    init_state.dynamics.${fp_name} = Parameter(init_state.dynamics.${fp_name})
% if fp_hetero:
    init_state.dynamics.${fp_name}.shape = (n_nodes,)
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
    """Run optimization for stage: ${stage_name}

    Algorithm: ${stage_algorithm}
    Learning rate: ${stage_lr}
    Max iterations: ${stage_max_iter}
% if stage_hyperparams:
    Hyperparameters: ${stage_hyperparams}
% endif
    """
    # Mark this stage's parameters
    marked_state = mark_parameters_${stage_name}(init_state)

    # Build optimizer kwargs
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

# Legacy single-stage function for backwards compatibility
def mark_parameters_optimizable(state, n_nodes: int = ${n_nodes}):
    """Mark parameters as optimizable - uses first stage's configuration."""
% if optimization_stages:
    return mark_parameters_${optimization_stages[0]['name']}(state, n_nodes)
% else:
    return copy.deepcopy(state)
% endif


def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    print_every: int = 10,
    has_aux: bool = False,
    **opt_kwargs,
):
    """Create configured optimizer."""
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
    # Default hyperparameters from YAML (first stage)
% for hp_name, hp_value in optimizer_hyperparams.items():
    optimizer_kwargs.setdefault('${hp_name}', ${hp_value})
% endfor
% endif

    callback = MultiCallback([
        DefaultPrintCallback(every=print_every),
        SavingCallback(key="state", save_fun=lambda *args: args[1])  # Save updated state each step
    ])
    return OptaxOptimizer(loss_fn, opt_fn(learning_rate, **optimizer_kwargs), callback=callback, has_aux=has_aux)


def run_optimization(
    init_state,
    loss_fn,
    max_steps: int = ${max_steps},
    learning_rate: float = ${learning_rate},
    optimizer: str = "${optimizer_name}",
    **kwargs,
):
    """Run gradient-based optimization."""
    opt = create_optimizer(loss_fn, optimizer=optimizer, learning_rate=learning_rate, **kwargs)
    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps)
    return fitted_params, fitting_data
% endif


% if has_explorations:
# =============================================================================
# Parameter Exploration
# =============================================================================

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
%>
def ${expl['name']}(state, model_fn, target_data=None, result_transient=None, n_pmap: int = ${n_workers}):
    """${expl['label']} - Parameter exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    N_PMAP: Auto-detected from available devices (default: ${n_workers})
% if obs_type == 'function_call':
    Observable: ${obs_func}(${', '.join([a['name'] for a in obs_args])})
% else:
    Observable: ${obs_name}${"['" + output_key + "']" if output_key else ""}
% endif
    """
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    % if ax.get('is_coupling'):
    grid_state.coupling.${ax['coupling_key']}.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % else:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % endif
    % endfor
    grid = Space(grid_state, mode="${expl['mode']}")

    # Create observation monitors ONCE with history baked in (optimized pattern)
% if obs_type == 'function_call':
<%
    # Collect unique observations used
    obs_used = set(a['obs'] for a in obs_args if a.get('obs'))
%>
% for obs in sorted(obs_used):
<%
    obs_class = ''.join(word.capitalize() for word in obs.split('_'))
%>
    _${obs}_monitor = ${obs_class}(history=result_transient)
% endfor

    @jax.jit
    def observable_fn(s):
        result = model_fn(s)
% for obs in sorted(obs_used):
        _${obs} = _${obs}_monitor(result)
% endfor
        return ${obs_func}(${', '.join([('_' + a['obs'] + '.data') if a['obs'] else 'target_data' for a in obs_args])})
% else:
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
%>
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

    exec_runner = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec_runner.run()
    return Bunch(grid=grid, results=jnp.stack(results))


% endfor
% endif


# =============================================================================
# Constants
# =============================================================================

CONDUCTION_SPEED = ${conduction_speed}
N_NODES = ${n_nodes}
DT = ${dt}
T1 = ${t1_default}
T_TRANSIENT = ${transient_time}
NOISE_SIGMA = ${noise_sigma[0]}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_experiment(
    weights: jnp.ndarray,
    distances: jnp.ndarray = None,
    target_data: jnp.ndarray = None,
    region_labels: list = None,
    mode: str = "all",
    stage: str = None,
    state: Bunch = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run the complete ${dynamics_class} experiment workflow.

    Parameters
    ----------
    weights : jnp.ndarray
        Connectivity weight matrix (n_nodes x n_nodes)
    distances : jnp.ndarray, optional
        Tract length matrix for delay computation (n_nodes x n_nodes)
    target_data : jnp.ndarray, optional
        Target data for optimization (e.g., target PSDs, empirical FC)
        Required when mode includes 'optimization' or 'all'
    region_labels : list, optional
        Labels for brain regions
    mode : str
        Workflow mode: 'simulation', 'optimization', 'exploration', or 'all' (default)
    stage : str, optional
        Name of specific optimization stage to run. If None, runs all stages.
        Only used when mode='optimization' and multi-stage optimization is configured.
    state : Bunch, optional
        Pre-configured state (e.g., from previous optimization). If provided,
        uses these parameters for simulation instead of defaults.

    Returns
    -------
    Bunch
        Results Bunch containing:
        - model_fn: Compiled model function
        - state: Initial state
        - result: Simulation result
        - transient: Transient simulation result (if t_transient > 0)
        - network: Network instance
        - observations: Computed observations (Bunch)
        - fitted_params: Optimized parameters (if mode='optimization')
        - fitting_data: Optimization history (if mode='optimization')
        - explorations: Grid search results as Bunch (if mode='exploration')
    """


    weights = jnp.array(weights)

    # Setup network
    % if has_delay:
    delays = jnp.array(distances) / CONDUCTION_SPEED if distances is not None else jnp.zeros_like(weights)
    network = create_network(weights, delays, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % endif

    # Run simulation to get model_fn and state (includes transient settling if configured)
    sim_result = run_simulation(network, t1=T1, dt=DT, t_transient=T_TRANSIENT)
    model_fn = sim_result.model_fn
    default_state = sim_result.state
    transient = sim_result.transient

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

        # Re-run simulation with custom parameters
        result = model_fn(use_state)
        state = use_state
    else:
        state = default_state
        result = sim_result.result

    # Compute observations using the (potentially custom) result
    observations = Bunch()
    obs_kwargs = dict(kwargs)
    obs_kwargs['result'] = result
    obs_kwargs['result_transient'] = transient
% for obs_name in observation_names:
    observations.${obs_name} = ${obs_name}(model_fn, state, **obs_kwargs)
% endfor

    results = Bunch(
        model_fn=model_fn,
        state=state,
        result=result,
        transient=transient,
        network=network,
        observations=observations,
    )

    % if has_optimization:
    # Optimization workflow - multi-stage support
    if mode in ('optimization', 'all'):
        if target_data is None:
            raise ValueError("target_data is required for optimization mode")

        # Create loss function with target data and transient (for HRF/BOLD pipeline)
        loss_type = kwargs.get('loss_type', None)
        loss_fn = make_loss_fn(model_fn, target_data, result_transient=transient, loss_type=loss_type)

        # Stage results storage (use Bunch for dot-notation access)
        stage_results = Bunch()
        current_state = state  # Start with initial state

% if len(optimization_stages) > 1:
        # Multi-stage optimization with optional stage filtering
        all_stage_names = [${', '.join(f"'{s['name']}'" for s in optimization_stages)}]

        if stage is not None:
            if stage not in all_stage_names:
                raise ValueError(f"Unknown stage '{stage}'. Available stages: {all_stage_names}")
            stages_to_run = [stage]
            print(f"Running single stage: {stage}")
        else:
            stages_to_run = all_stage_names
            print("=" * 60)
            print("Multi-stage optimization: ${len(optimization_stages)} stages")
            print("=" * 60)

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
                max_steps=kwargs.get('max_steps_${stage_name}', ${stage_max_iter}),
                learning_rate=kwargs.get('learning_rate_${stage_name}', ${stage_lr}),
            )
            stage_results['${stage_name}'] = Bunch(
                fitted_params=_fitted_${stage_name},
                fitting_data=_history_${stage_name},
            )
            current_state = _fitted_${stage_name}  # Chain to next stage

% endfor
        if stage is None:
            print("\n" + "=" * 60)
            print("Multi-stage optimization complete")
            print("=" * 60)

        # Final results: last stage's fitted_params + per-stage access via dot notation
        results['fitted_params'] = current_state
        results['fitting_data'] = stage_results  # Bunch of all stage histories
        # Add each stage directly to results for easy access: results.global_optimization.fitted_params
        for _stage_name, _stage_result in stage_results.items():
            results[_stage_name] = _stage_result

% else:
        # Single-stage optimization
        init_state = mark_parameters_optimizable(state)

        fitted_params, fitting_data = run_optimization(
            init_state,
            loss_fn,
            max_steps=kwargs.get('max_steps', ${max_steps}),
            learning_rate=kwargs.get('learning_rate', ${learning_rate}),
            optimizer=kwargs.get('optimizer', '${optimizer_name}'),
        )

        results['fitted_params'] = fitted_params
        results['fitting_data'] = fitting_data
% endif
    % endif

    % if has_explorations:
    # Exploration workflow
    if mode in ('exploration', 'all'):
        explorations_result = Bunch()

        % for expl in explorations:
        explorations_result.${expl['name']} = ${expl['name']}(
            state, model_fn,
            target_data=target_data,
            result_transient=transient,
        )
        % endfor

        results.explorations = explorations_result
    % endif

    return results
