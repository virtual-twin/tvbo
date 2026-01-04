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
<%namespace name="const" file="/base/constants.mako"/>
<%
from tvbo.export.code import render_expression
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

# Helper function for safe Python identifiers
def safe_name(name):
    """Convert name to valid Python identifier."""
    return str(name).replace(' ', '_').replace('-', '_').lower()

# Extract key metadata from model
state_names = list(model.state_variables.keys())
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

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

# Observation names (for computing all observations in run_simulation)
def obs_has_all_args(obs):
    """Check if observation has all required arguments satisfied."""
    pipeline = obs.pipeline or []
    has_source = obs.source or obs.source_observation

    for step_idx, func in enumerate(pipeline):
        is_first_step = step_idx == 0
        args = func.arguments or []
        for arg in args:
            if arg.name and arg.value is None:
                # First step's data-like args are satisfied by source
                if is_first_step and has_source and arg.name in ('data', 'X', 'x', 'input', 'timeseries', 'a'):
                    continue  # Implicitly satisfied
                return False  # Argument without value = requires runtime input
    return True

def is_network_observation(obs):
    """Check if observation is a network observation (static data from BIDS)."""
    if not obs:
        return False
    source = getattr(obs, 'source', None)
    if source and str(source).startswith('network.observations'):
        return True
    return False

# Build observations dict from experiment.observations
observations_dict = dict(experiment.observations.items()) if experiment.observations else {}
_obs_list = list(observations_dict.items())

# Identify network observations (static data, not simulation-derived)
network_observation_names = set(name for name, obs in _obs_list if is_network_observation(obs))

# Include all observations that have all required arguments satisfied
observation_names = [name for name, obs in _obs_list if obs_has_all_args(obs)]

# Class name from model
dynamics_class = model.name.replace(' ', '').replace('-', '') if model.name else 'GeneratedDynamics'

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

# Fallback: param.free=True on model parameters (legacy)
for name, param in model.parameters.items():
    if param.free and str(name) not in optim_param_info:
        is_hetero = param.heterogeneous or param.shape
        optim_param_info[str(name)] = {'heterogeneous': bool(is_hetero)}

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
import ast  # For safely parsing stringified dicts

def get_domain_bounds(param_name):
    """Lookup domain bounds from model.parameters or coupling.parameters.
    Returns (lo, hi) tuple, where None means unbounded.
    """
    lo, hi = None, None

    def extract_bounds(param):
        """Extract (lo, hi) from param.domain if defined."""
        if param.domain:
            lo_val = param.domain.lo if param.domain.lo is not None else None
            hi_val = param.domain.hi if param.domain.hi is not None else None
            try:
                return (float(lo_val) if lo_val is not None else None,
                        float(hi_val) if hi_val is not None else None)
            except (TypeError, ValueError):
                pass
        return (None, None)

    # Check dynamics parameters
    if model.parameters and param_name in model.parameters:
        lo, hi = extract_bounds(model.parameters[param_name])

    # Check coupling parameters from all_couplings
    if lo is None and hi is None:
        for ck, cobj in all_couplings.items():
            if cobj.parameters and param_name in cobj.parameters:
                lo, hi = extract_bounds(cobj.parameters[param_name])
                break

    return (lo, hi)

def parse_free_param(fp):
    """Parse a free_parameter entry which could be:
    - str: simple param name like 'w' (dynamics param, infers local_dynamics)
    - str: dotted notation like 'ReducedWongWang.w' (dynamics param, explicit)
    - str: dotted notation like 'FastLinearCoupling.G' (coupling param)
    - str: stringified dict like "{'name': 'w', 'heterogeneous': True}"
    - dict: actual dict with 'name' key
    - object: with .name attribute

    Dotted notation: ClassName.param_name
    - If ClassName matches a coupling key → coupling param
    - Otherwise → dynamics param (ClassName is dynamics name)

    Returns dict with: name, heterogeneous, lower_bound, upper_bound,
                       coupling_key (if coupling), dynamics_key (if explicit dynamics)
    """
    # Get known coupling keys to distinguish coupling vs dynamics
    coupling_keys = set(all_couplings.keys())

    result = None
    source_key = None  # Will be set to coupling_key or dynamics_key
    is_coupling = False

    if isinstance(fp, str):
        # Check if it looks like a stringified dict
        stripped = fp.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict) and 'name' in parsed:
                    param_name = str(parsed['name'])
                    # Check for dotted notation in parsed name
                    if '.' in param_name:
                        prefix, param_name = param_name.rsplit('.', 1)
                        is_coupling = prefix in coupling_keys
                        source_key = prefix
                    result = {
                        'name': param_name,
                        'heterogeneous': bool(parsed.get('heterogeneous', False)),
                        'shape': parsed.get('shape', None),
                        'coupling_key': source_key if is_coupling else None,
                        'dynamics_key': source_key if not is_coupling and source_key else None,
                    }
            except (ValueError, SyntaxError):
                pass
        if result is None:
            # Check for dotted notation: ClassName.param_name
            if '.' in stripped:
                prefix, param_name = stripped.rsplit('.', 1)
                is_coupling = prefix in coupling_keys
                source_key = prefix
                result = {
                    'name': param_name,
                    'heterogeneous': False,
                    'shape': None,
                    'coupling_key': source_key if is_coupling else None,
                    'dynamics_key': source_key if not is_coupling else None,
                }
            else:
                # Simple string param name (dynamics, no explicit class)
                result = {'name': fp, 'heterogeneous': False, 'shape': None, 'coupling_key': None, 'dynamics_key': None}
    elif not isinstance(fp, (str, dict)):
        # Parameter object (most common case from LinkML)
        param_name = str(fp.name)
        # Check for dotted notation
        if '.' in param_name:
            prefix, param_name = param_name.rsplit('.', 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        result = {
            'name': param_name,
            'heterogeneous': bool(fp.heterogeneous),
            'shape': str(fp.shape) if fp.shape else None,
            'coupling_key': source_key if is_coupling else None,
            'dynamics_key': source_key if not is_coupling and source_key else None,
        }
        # Check if domain is specified directly on the free_parameter object
        if fp.domain:
            if fp.domain.lo is not None:
                try:
                    result['lower_bound'] = float(fp.domain.lo)
                except (TypeError, ValueError):
                    pass
            if fp.domain.hi is not None:
                try:
                    result['upper_bound'] = float(fp.domain.hi)
                except (TypeError, ValueError):
                    pass
    elif isinstance(fp, dict) and 'name' in fp:
        param_name = str(fp['name'])
        source_key = None
        is_coupling = False
        # Check for dotted notation
        if '.' in param_name:
            prefix, param_name = param_name.rsplit('.', 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        shape_str = fp.get('shape', None)
        result = {
            'name': param_name,
            'heterogeneous': bool(fp.get('heterogeneous', False)),
            'shape': str(shape_str) if shape_str else None,
            'coupling_key': source_key if is_coupling else None,
            'dynamics_key': source_key if not is_coupling and source_key else None,
        }
        # Check for domain in dict
        if 'domain' in fp:
            domain = fp['domain']
            if isinstance(domain, dict):
                if 'lo' in domain:
                    try:
                        result['lower_bound'] = float(domain['lo'])
                    except (TypeError, ValueError):
                        pass
                if 'hi' in domain:
                    try:
                        result['upper_bound'] = float(domain['hi'])
                    except (TypeError, ValueError):
                        pass

    if result is None:
        return None

    # Ensure all keys exist
    result.setdefault('coupling_key', None)
    result.setdefault('dynamics_key', None)

    # If no bounds from free_parameter, lookup from model/coupling parameters
    if 'lower_bound' not in result or 'upper_bound' not in result:
        model_lo, model_hi = get_domain_bounds(result['name'])
        if 'lower_bound' not in result and model_lo is not None:
            result['lower_bound'] = model_lo
        if 'upper_bound' not in result and model_hi is not None:
            result['upper_bound'] = model_hi

    # Set None for missing bounds (will become +/- inf)
    result.setdefault('lower_bound', None)
    result.setdefault('upper_bound', None)
    result.setdefault('shape', None)

    # AUTO-DETECT coupling parameters if not explicitly specified
    # Check if parameter exists in any coupling's parameters but not in dynamics
    if result.get('coupling_key') is None:
        param_name = result['name']
        # Check if it's NOT a dynamics parameter
        is_dynamics = model and hasattr(model, 'parameters') and param_name in model.parameters
        if not is_dynamics:
            # Check all couplings for this parameter
            for ck, cobj in all_couplings.items():
                if hasattr(cobj, 'parameters') and cobj.parameters and param_name in cobj.parameters:
                    result['coupling_key'] = ck
                    break

    return result

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
            parsed = parse_free_param(fp)
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

# Schema provides ifabsent defaults, so these should always be populated
# Only assert if optimization is requested but values somehow missing
if has_optimization:
    assert optimizer_name, "optimization.algorithm not found (schema default: 'adam')"
    assert learning_rate is not None, "optimization.learning_rate not found (schema default: 0.001)"
    assert max_steps is not None, "optimization.max_iterations not found (schema default: 100)"

# === Observations metadata ===
# Schema: experiment.observations is multivalued dict
observations = dict(experiment.observations) if experiment.observations else {}

def get_obs(name):
    """Look up observation by name from observations dict."""
    return observations.get(name)

def get_pipeline_output_key(obs_name):
    """Extract the last pipeline step's output key for an observation.

    Defaults to observation name if last step has no explicit output.
    """
    obs_obj = get_obs(obs_name)
    if obs_obj and obs_obj.pipeline:
        # Schema: pipeline is always a list (inlined_as_list: true)
        last_step = obs_obj.pipeline[-1]
        if last_step.output:
            # Handle multi-output (comma-separated) - take the last one as the "main" output
            outputs = [o.strip() for o in str(last_step.output).split(',')]
            return outputs[-1]
        return obs_name
    return obs_name

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

# Parse observations
obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': obs.label or '',
        'description': obs.description or '',
        'source': obs.source.name if obs.source and hasattr(obs.source, 'name') else str(obs.source) if obs.source else None,
        'source_observation': obs.source_observation.name if obs.source_observation and hasattr(obs.source_observation, 'name') else str(obs.source_observation) if obs.source_observation else None,
        'equation': obs.equation.rhs if obs.equation else None,
    }
    obs_list.append(obs_info)

# First coupling name for docstring
first_coupling_name = list(all_couplings.keys())[0] if all_couplings else 'None'
%>
"""
${dynamics_class} tvboptim Experiment
${'=' * (len(dynamics_class) + 20)}

Auto-generated from TVBO SimulationExperiment specification.

Experiment: ${experiment.label or 'Generated'}
Model: ${model.name or 'Generated'}
Coupling: ${first_coupling_name}
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
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif
% if has_optimization:
import optax
from tvboptim.types import Parameter, BoundedParameter
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

    # Create coupling functions for each entry in network.coupling
    # Key == class name (cleaned for Python identifier)
    n_nodes = weights.shape[0]
    coupling_dict = {}

    % for coupling_key, coupling_obj in all_couplings.items():
<%
    # Class name = coupling key (cleaned), same as in cfun template
    c_class_name = coupling_key.replace(' ', '').replace('-', '')
    c_params = list(coupling_obj.parameters.values()) if hasattr(coupling_obj, 'parameters') and coupling_obj.parameters else []
    c_param_names = [p.name for p in c_params]
    c_param_defaults = {p.name: float(p.value) if p.value is not None else 1.0 for p in c_params}
    c_param_shapes = {}
    for p in c_params:
        shape_str = getattr(p, 'shape', None)
        if shape_str and 'n_nodes' in str(shape_str):
            c_param_shapes[p.name] = str(shape_str)
%>
    # Coupling '${coupling_key}' -> ${c_class_name}
    _${coupling_key}_params = {
        % for name in c_param_names:
        % if name in c_param_shapes:
        '${name}': jnp.ones(${c_param_shapes[name].replace('n_nodes', 'n_nodes')}) * ${c_param_defaults.get(name, 1.0)},
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
        coupling=coupling_dict,
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
    # Network observations are module-level constants (already loaded from BIDS)
    observations = Bunch()
    obs_kwargs = dict(kwargs)
    obs_kwargs['result'] = result
    obs_kwargs['result_transient'] = result_transient
% for obs_name in observation_names:
% if obs_name in network_observation_names:
    observations.${obs_name} = ${obs_name}  # Static data from BIDS (module-level constant)
% else:
    observations.${obs_name} = ${obs_name}(model_fn, state, **obs_kwargs)
% endif
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
    loss_call = opt.loss
    if loss_call:
        # FunctionCall has 'function' (reference) or 'callable' (inline)
        func_ref = loss_call.function
        callable_ref = loss_call.callable

        # Determine the function name to call
        if func_ref:
            func_name = str(func_ref) if isinstance(func_ref, str) else (func_ref.name if hasattr(func_ref, 'name') else str(func_ref))
        elif callable_ref:
            func_name = callable_ref.name or callable_ref.qualname or 'loss'
        else:
            func_name = 'loss'

        # Parse aggregate specification
        aggregate = loss_call.aggregate
        agg_over = None
        agg_type = None
        if aggregate:
            # Handle enum values (e.g., DimensionType.node -> 'node')
            agg_over = str(aggregate.over).split('.')[-1] if aggregate.over else None
            agg_type = str(aggregate.type).split('.')[-1] if aggregate.type else 'mean'

        # Parse arguments: value = observation reference, no value = runtime input
        loss_args = loss_call.arguments or []
        parsed_args = []
        obs_refs = set()  # Track which observations we need to call
        for arg in loss_args:
            arg_name = arg.name
            arg_value = arg.value
            if arg_name:
                if arg_value is not None:
                    val_str = str(arg_value)
                    # Check if it's a scalar constant (numeric)
                    try:
                        float_val = float(arg_value)
                        # It's a numeric constant
                        parsed_args.append({
                            'name': arg_name,
                            'type': 'constant',
                            'value': arg_value,
                        })
                        continue
                    except (ValueError, TypeError):
                        pass
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
                    # No value = runtime input (passed via kwargs with same name)
                    parsed_args.append({
                        'name': arg_name,
                        'type': 'runtime',
                        'kwarg_name': arg_name,  # Use arg name as kwarg key
                    })

        loss_functions.append({
            'opt_name': opt.name or 'loss',
            'func_name': func_name,
            'args': parsed_args,
            'obs_refs': obs_refs,
            'agg_over': agg_over,
            'agg_type': agg_type,
        })

    # Collect all runtime kwargs needed for loss functions
    runtime_kwargs_needed = set()
    for lf in loss_functions:
        for arg in lf['args']:
            if arg['type'] == 'runtime':
                runtime_kwargs_needed.add(arg['kwarg_name'])
%>
def make_loss_fn(model_fn, result_transient=None, loss_type: str = None, **kwargs):
    """Create a loss function closure for optimization.

    Loss functions are generated from optimization metadata.
    Each loss MUST specify equation, source_code, or callable.

    Runtime inputs (observations with data_source) are passed via kwargs.
    Required kwargs: ${', '.join(sorted(runtime_kwargs_needed)) if runtime_kwargs_needed else '(none)'}

    IMPORTANT: Observation monitors are created ONCE here with history baked in,
    then reused in the inner loss function. This matches the exploration pattern
    and is critical for proper JAX differentiation.

    Args:
        model_fn: Compiled model function
        result_transient: Transient simulation result for HRF/BOLD pipeline warmup
        loss_type: Which loss function to use (defaults to first available)
        **kwargs: Runtime inputs (e.g., fc_target=empirical_fc_matrix)
    """
% if runtime_kwargs_needed:
    # Validate required runtime inputs
% for kwarg_name in sorted(runtime_kwargs_needed):
    if '${kwarg_name}' not in kwargs:
        raise ValueError("Optimization loss requires '${kwarg_name}' input (passed via kwargs)")
    ${kwarg_name} = kwargs['${kwarg_name}']
% endfor
% endif
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
        # Network observations are module-level constants (no monitor needed)
% for obs_name in sorted(obs_refs):
<%
    obs_class = ''.join(word.capitalize() for word in obs_name.split('_'))
    is_network_obs = obs_name in network_observation_names
%>
% if not is_network_obs:
        _${obs_name}_monitor = ${obs_class}(history=result_transient)
% endif
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

            # Apply observation monitors (simulation-derived observations only)
% for obs_name in sorted(obs_refs):
<%
    is_network_obs = obs_name in network_observation_names
%>
% if not is_network_obs:
            _${obs_name} = _${obs_name}_monitor(result)
% endif
% endfor

            # Prepare loss function arguments
% for arg in args:
% if arg['type'] == 'observation':
<%
    arg_obs_is_network = arg['obs_name'] in network_observation_names
%>
% if arg_obs_is_network:
% if arg['name'] != arg['obs_name']:
            # Network observation (module-level constant) with different arg name
            ${arg['name']} = ${arg['obs_name']}
% else:
            # Network observation: ${arg['obs_name']} (use module-level constant directly)
% endif
% elif arg['output_key']:
            ${arg['name']} = _${arg['obs_name']}.${arg['output_key']}
% else:
            ${arg['name']} = _${arg['obs_name']}.data
% endif
% elif arg['type'] == 'constant':
            ${arg['name']} = ${arg['value']}
% elif arg['type'] == 'runtime':
            # Runtime input from kwargs: ${arg['kwarg_name']}
            # (already validated and extracted at top of make_loss_fn)
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


# =============================================================================
# Iterative Algorithms (FIC, etc.)
# =============================================================================

<%include file="tvbo-tvboptim-algorithm.py.mako" />


% if has_optimization:
# =============================================================================
# Optimization
# =============================================================================

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
    **opt_kwargs,
):
    """Create configured optimizer.

    Note: has_aux is always False because our generated loss functions
    return only the loss value, not (loss, aux_data) tuples.
    """
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
    # has_aux=False: our loss functions return only loss value, not (loss, aux) tuples
    return OptaxOptimizer(loss_fn, opt_fn(learning_rate, **optimizer_kwargs), callback=callback, has_aux=False)


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
def ${expl['name']}(state, model_fn, result_transient=None, n_pmap: int = ${n_workers}, **kwargs):
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
        return ${obs_func}(${', '.join([('_' + a['obs'] + '.data') if a['obs'] else "kwargs['" + a['name'] + "']" for a in obs_args])})
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


${const.all_constants(experiment)}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_experiment(
    weights: jnp.ndarray,
    distances: jnp.ndarray = None,
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
    region_labels : list, optional
        Labels for brain regions
    mode : str
        Workflow mode: 'simulation', 'optimization', 'exploration', 'algorithms', or 'all' (default)
    stage : str, optional
        Name of specific optimization stage to run. If None, runs all stages.
        Only used when mode='optimization' and multi-stage optimization is configured.
    state : Bunch, optional
        Pre-configured state (e.g., from previous optimization). If provided,
        uses these parameters for simulation instead of defaults.
    **kwargs
        Runtime inputs for algorithms/optimization (e.g., fc_target for FC-based loss)

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
        - algorithms: Algorithm results (if mode='algorithms' or 'all')
    """

    weights = jnp.array(weights)

    # Setup network
    # -------------------------------------------------------------------------
    # STEP 1: Simulation (always runs first)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Running simulation...")
    print("=" * 60)

    % if has_delay:
    delays = jnp.array(distances) / ${conduction_speed} if distances is not None else jnp.zeros_like(weights)
    network = create_network(weights, delays, region_labels=region_labels, noise_sigma=${noise_sigma[0]})
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=${noise_sigma[0]})
    % endif

    # Run simulation to get model_fn and state (includes transient settling if configured)
    sim_result = run_simulation(network, t1=${t1_default}, dt=${dt}, t_transient=${transient_time})
    model_fn = sim_result.model_fn
    default_state = sim_result.state
    transient = sim_result.transient
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

        # Re-run simulation with custom parameters
        result = model_fn(use_state)
        state = use_state
    else:
        state = default_state
        result = sim_result.result

    # Compute observations using the (potentially custom) result
    # Network observations are module-level constants (already loaded from BIDS)
    observations = Bunch()
    obs_kwargs = dict(kwargs)
    obs_kwargs['result'] = result
    obs_kwargs['result_transient'] = transient
% for obs_name in observation_names:
% if obs_name in network_observation_names:
    observations.${obs_name} = ${obs_name}  # Static data from BIDS (module-level constant)
% else:
    observations.${obs_name} = ${obs_name}(model_fn, state, **obs_kwargs)
% endif
% endfor

    results = Bunch(
        model_fn=model_fn,
        state=state,
        result=result,
        transient=transient,
        network=network,
        observations=observations,
    )
    print("  Simulation complete.")

    % if has_explorations:
    # -------------------------------------------------------------------------
    # STEP 2: Explorations (parameter sweeps)
    # -------------------------------------------------------------------------
    if mode in ('exploration', 'all'):
        print("\n" + "=" * 60)
        print("STEP 2: Running explorations...")
        print("=" * 60)
        explorations_result = Bunch()

        % for expl in explorations:
        print(f"  > ${expl['name']}")
        explorations_result.${expl['name']} = ${expl['name']}(
            state, model_fn,
            result_transient=transient,
            **kwargs,  # Pass runtime kwargs (e.g., target data for correlation-based observables)
        )
        % endfor

        results.explorations = explorations_result
        print("  Explorations complete.")
    % endif

    % if has_algorithms:
    # -------------------------------------------------------------------------
    # STEP 3: Algorithms (FIC, EIB, etc.)
    # -------------------------------------------------------------------------
    # ALL parameters derived from YAML metadata
    #
    # Modes:
    #   - mode='algorithm': Run a single algorithm by name
    #   - mode='algorithms' or mode='all': Run ALL algorithms in dependency order
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

        # Random key from execution.random_seed in YAML (can be overridden)
        algo_seed = kwargs.pop('seed', ${random_seed})
        algo_key = kwargs.pop('key', jax.random.PRNGKey(algo_seed))
        algo_verbose = kwargs.pop('verbose', True)  # verbose is a display option, ok to default

        # Storage for algorithm results when running all
        algorithms_results = Bunch()
        current_state = state  # Track state through algorithm chain

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

    # Find algorithms that are included in other algorithms
    # These should NOT be run standalone when mode='all'
    included_algorithms = set()
    for a in algorithms_list:
        includes = getattr(a, 'includes', None) or []
        for inc in includes:
            if hasattr(inc, 'algorithm'):
                inc_name = safe_name(str(getattr(inc, 'algorithm', '')))
            else:
                inc_name = safe_name(str(inc))
            if inc_name:
                included_algorithms.add(inc_name)

    # Get algorithms in dependency order (topological sort)
    # Exclude algorithms that are included in other algorithms
    def get_sorted_algorithms():
        """Return algorithm names in dependency order, excluding included algorithms."""
        # Start with algorithms that are NOT included in others
        all_algos = set(algorithms_deps.keys())
        top_level_algos = all_algos - included_algorithms

        sorted_names = []
        remaining = set(top_level_algos)
        while remaining:
            # Find algorithms with all dependencies satisfied
            ready = [n for n in remaining if all(d in sorted_names or d not in remaining for d in algorithms_deps[n])]
            if not ready:
                # Circular dependency or missing dep - just add remaining
                ready = list(remaining)
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
            algorithms_to_run = [${', '.join(f"'{n}'" for n in sorted_algo_names)}]
            print(f"  Algorithms to run: {algorithms_to_run}")
        else:
            algorithms_to_run = [algorithm_name]

        # Run algorithms in order
        for _algo_name_to_run in algorithms_to_run:
            algorithm_name = _algo_name_to_run
            if algo_verbose:
                print(f"\\n>>> Running algorithm: {algorithm_name}")
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

    # Observation reference (deprecated - now use observations list)
    observation_ref = None
%>
            if algorithm_name == '${algo_name}':
                # Create algorithm-specific model_fn with simulation_period
                algo_model_fn, algo_state = prepare(network, Heun(), t1=${float(algo_sim_period)}, dt=${dt})

                # Copy PARAMETER VALUES from settled main state (dynamics, coupling params)
                for key in state.dynamics.keys():
                    if not key.startswith('_'):
                        algo_state.dynamics[key] = state.dynamics[key]
                for coupling_name in state.coupling.keys():
                    if not coupling_name.startswith('_'):
                        for key in state.coupling[coupling_name].keys():
                            if not key.startswith('_'):
                                algo_state.coupling[coupling_name][key] = state.coupling[coupling_name][key]
                algo_state.initial_state.dynamics = state.initial_state.dynamics

% for inp_name in input_names:
                # Validate required input: ${inp_name}
                if '${inp_name}' not in kwargs:
                    raise ValueError("Algorithm '${algo_name}' requires '${inp_name}' input (passed via kwargs)")
% endfor
<%
    # Detect if this algorithm uses sliding window and needs buffer inputs
    # Use hyperparams_dict which already includes hyperparams from included algorithms
    algo_has_window_size = 'window_size' in hyperparams_dict

    # Find source observations needed (observations with source_observation dependency)
    algo_source_obs_needed = set()
    for obs_name in obs_names:
        obs_def = observations_dict.get(obs_name)
        if obs_def:
            src_obs = getattr(obs_def, 'source_observation', None)
            if src_obs:
                algo_source_obs_needed.add(str(src_obs))
    algo_needs_buffers = algo_has_window_size and len(algo_source_obs_needed) > 0
%>

                algo_result = run_${algo_name}(
                    state=algo_state,
                    model_fn=algo_model_fn,
                    key=algo_key,
                    history=transient,
                    n_iterations=kwargs.get('n_iterations', ${n_iterations}),
% for hp_name, hp_val in hyperparams_dict.items():
<%
    if hp_val is None:
        raise ValueError(f"Hyperparameter '{hp_name}' in algorithm '{algo_name}' missing required 'value' in YAML")
%>
                    ${hp_name}=kwargs.get('${hp_name}', ${hp_val}),
% endfor
% for inp_name in input_names:
                    ${inp_name}=kwargs.get('${inp_name}'),
% endfor
% for net_obs_name in network_obs_inputs:
                    ${net_obs_name}=${net_obs_name},  # Module-level constant from BIDS
% endfor
% if algo_needs_buffers:
% for src_obs in algo_source_obs_needed:
                    ${src_obs}_buffer=kwargs.get('${src_obs}_buffer', None),  # Optional: pass from previous algorithm
% endfor
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
                # Update state for next algorithm in chain (if it depends on this one)
                if hasattr(algo_result, 'state'):
                    current_state = algo_result.state
                    # Also update state.dynamics/coupling for next algo_state creation
                    state = current_state

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
    # -------------------------------------------------------------------------
    # STEP 4: Optimization (gradient-based parameter fitting)
    # -------------------------------------------------------------------------
    # Runtime inputs for loss function are passed via kwargs (e.g., fc_target)
    if mode in ('optimization', 'all'):
        print("\n" + "=" * 60)
        print("STEP 4: Running optimization...")
        print("=" * 60)
        # Check if all required runtime inputs are provided
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
            # Create loss function with runtime inputs from kwargs
            loss_type = kwargs.get('loss_type', None)
            loss_fn = make_loss_fn(model_fn, result_transient=transient, loss_type=loss_type, **kwargs)

            # Stage results storage (use Bunch for dot-notation access)
            stage_results = Bunch()
            current_state = state  # Start with current state (may be updated by algorithms)

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
            print("  Multi-stage optimization complete")
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
            print("  Optimization complete.")
% endif
    % endif

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)

    return results


# =============================================================================
# Standalone Execution
# =============================================================================

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    print("=" * 60)
    print("${dynamics_class} Experiment - Standalone Execution")
    print("=" * 60)

    # Run the experiment
    # Order: 1) Simulation → 2) Explorations → 3) Algorithms → 4) Optimization
    results = run_experiment(
        weights,  # Uses module-level weights from network loading
        distances=distances if 'distances' in dir() else None,
        region_labels=region_labels if 'region_labels' in dir() else None,
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

