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
jaxcode = lambda expr: render_expression(expr, format='jax', user_functions=user_functions)
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
has_noise = integration.noise is not None
noise_sigma = np.asarray(experiment.noise_sigma_array).flatten().tolist() if has_noise else [0.0]

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

# Observation names (for computing all observations in run_simulation)
# Include all observations that have complete argument specifications
def obs_has_all_args(obs):
    """Check if observation has all required arguments satisfied."""
    pipeline = getattr(obs, 'pipeline', None) or []
    for func in pipeline:
        args = getattr(func, 'arguments', None) or []
        if hasattr(args, '__iter__'):
            for arg in args:
                if getattr(arg, 'name', None) and getattr(arg, 'value', None) is None:
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

# Extract optimizable parameters
optim_params = []
for name, param in model.parameters.items():
    if getattr(param, 'free', False):
        optim_params.append(param)


# Note: Coupling parameters with free=True are collected but not currently
# used for optimization because the state structure stores coupling params
# differently (under state.coupling[key].params). See mark_parameters_optimizable.
coupling_optim_params = []
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        if getattr(param, 'free', False):
            coupling_optim_params.append(param)

# Extract optimizer settings - uses schema ifabsent defaults
# Schema defaults: algorithm='adam', learning_rate=0.001, max_iterations=100
optimizer_name = None
learning_rate = None
max_steps = None

for opt in optim_list:
    if hasattr(opt, 'algorithm') and opt.algorithm:
        optimizer_name = str(opt.algorithm)
    if hasattr(opt, 'learning_rate') and opt.learning_rate is not None:
        learning_rate = float(opt.learning_rate)
    if hasattr(opt, 'max_iterations') and opt.max_iterations is not None:
        max_steps = int(opt.max_iterations)

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
    """Extract the last pipeline step's output key for an observation."""
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
    return None

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
        exp_info['axes'].append({
            'name': getattr(param, 'name', str(param)),
            'lo': float(domain.lo),
            'hi': float(domain.hi),
            'n': int(domain.n),
        })
    observable = getattr(expl, 'observable', None)
    if observable:
        obs_name = getattr(observable, 'name', str(observable))
        exp_info['observable'] = obs_name
        exp_info['output_key'] = get_pipeline_output_key(obs_name)
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

import copy
import jax
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
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback
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
    noise = AdditiveNoise(sigma=noise_sigma) if noise_sigma > 0 else None
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

# Collect callable imports: {module: [(qualname, alias), ...]}
callable_imports = {}
for fname, fdef in exp_funcs.items():
    callable_def = getattr(fdef, 'callable', None)
    if callable_def:
        module = getattr(callable_def, 'module', None)
        qualname = getattr(callable_def, 'qualname', None) or getattr(callable_def, 'name', None)
        if module and qualname:
            if module not in callable_imports:
                callable_imports[module] = []
            callable_imports[module].append((qualname, str(fname)))

def to_numeric(val):
    """Convert string to numeric if possible."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return int(val) if '.' not in val else float(val)
        except ValueError:
            return val
    return val

# Generate functions from source_code OR equation (skip callables - they're imported)
standalone_funcs = []
for fname, fdef in exp_funcs.items():
    # Skip callable functions - they're handled via imports
    callable_def = getattr(fdef, 'callable', None)
    if callable_def and (getattr(callable_def, 'module', None)):
        continue

    source_code = getattr(fdef, 'source_code', None)
    equation = getattr(fdef, 'equation', None)
    eq_rhs = getattr(equation, 'rhs', None) if equation else None
    time_range = getattr(fdef, 'time_range', None)

    # Get arguments for function signature
    args_raw = getattr(fdef, 'arguments', None) or []
    arg_list = []
    arg_names = []  # Just the names, for passing to parser
    if hasattr(args_raw, '__iter__'):
        for arg in args_raw:
            arg_name = getattr(arg, 'name', None)
            arg_value = getattr(arg, 'value', None)
            if arg_name:
                if arg_value is not None:
                    # Convert to proper numeric type
                    numeric_val = to_numeric(arg_value)
                    arg_list.append(f"{arg_name}={repr(numeric_val)}")
                else:
                    arg_list.append(str(arg_name))
                # Also collect just the names for parsing (to override SymPy builtins)
                arg_names.append(str(arg_name))

    if source_code:
        code_str = str(source_code).strip()
        lines = code_str.split('\n')
        standalone_funcs.append({
            'name': str(fname),
            'source_code': code_str,
            'equation': None,
            'time_range': None,
            'description': getattr(fdef, 'description', ''),
            'is_multiline': len(lines) > 1,
            'is_lambda': code_str.startswith('lambda '),
            'args': arg_list,
        })
    elif time_range and eq_rhs:
        # Kernel generator with time_range - generates t array and evaluates equation
        tr_lo = getattr(time_range, 'lo', 0)
        tr_hi = getattr(time_range, 'hi', 'duration')
        tr_step = getattr(time_range, 'step', 'dt')
        # Get equation parameters for local variable assignments
        eq_params = getattr(equation, 'parameters', None) or {}
        param_assigns = {}
        param_names_for_eq = ['t']  # t is always available in time_range equations
        if hasattr(eq_params, 'items'):
            for pname, pobj in eq_params.items():
                pval = getattr(pobj, 'value', None)
                if pval is not None:
                    param_assigns[str(pname)] = to_numeric(pval)
                param_names_for_eq.append(str(pname))
        # Render equation with t and equation parameters as symbols
        rendered = render_expression(str(eq_rhs), format='jax', user_functions=all_func_names, parameters=param_names_for_eq)
        standalone_funcs.append({
            'name': str(fname),
            'source_code': None,
            'equation': rendered,
            'time_range': {'lo': tr_lo, 'hi': str(tr_hi), 'step': str(tr_step)},
            'param_assigns': param_assigns,
            'description': getattr(fdef, 'description', ''),
            'is_multiline': False,
            'is_lambda': False,
            'args': arg_list,  # duration, dt as function arguments
        })
    elif eq_rhs:
        # Generate from equation - render to JAX code
        # Pass arg_names as parameters so they become Symbols (overriding SymPy functions like gamma)
        rendered = render_expression(str(eq_rhs), format='jax', user_functions=all_func_names, parameters=arg_names)
        standalone_funcs.append({
            'name': str(fname),
            'source_code': None,
            'equation': rendered,
            'time_range': None,
            'description': getattr(fdef, 'description', '') or getattr(equation, 'definition', ''),
            'is_multiline': False,
            'is_lambda': False,
            'args': arg_list,
        })
%>
# Callable imports (functions from external modules)
% for module, imports in callable_imports.items():
% for qualname, alias in imports:
% if qualname == alias:
from ${module} import ${qualname}
% else:
from ${module} import ${qualname} as ${alias}
% endif
% endfor
% endfor

% for func in standalone_funcs:
# ${func['description'] or func['name']}
<%
    if func.get('time_range'):
        # Kernel generator with time_range - generates t array and evaluates equation
        tr = func['time_range']
        args_str = ', '.join(func['args']) if func['args'] else ''
        # Build parameter assignments from equation.parameters
        param_assigns = func.get('param_assigns', {})
        param_lines = '\n    '.join([f"{k} = {v}" for k, v in param_assigns.items()])
        body_lines = []
        if param_lines:
            body_lines.append(param_lines)
        body_lines.append(f"t = jnp.arange({tr['lo']}, {tr['hi']}, {tr['step']})")
        body_lines.append(f"return {func['equation']}")
        body = '\n    '.join(body_lines)
        out_line = f"def {func['name']}({args_str}):\n    {body}"
    elif func['equation']:
        # Generated from equation.rhs - create a simple function
        # Extract free symbols from equation as parameters if no args specified
        args_str = ', '.join(func['args']) if func['args'] else 'x, y'
        out_line = f"def {func['name']}({args_str}):\n    return {func['equation']}"
    elif func['is_lambda'] and not func['is_multiline']:
        # Simple single-line lambda: name = lambda ...
        out_line = f"{func['name']} = {func['source_code']}"
    elif func['is_lambda'] and func['is_multiline']:
        # Multiline lambda - need to convert to def function
        code = func['source_code']
        import re
        lambda_pattern = r'^lambda\s+([^:]+):'
        match = re.match(lambda_pattern, code.split('\n')[0])
        if match:
            params = match.group(1).strip()
            first_line = code.split('\n')[0]
            colon_idx = first_line.find(':')
            first_body_part = first_line[colon_idx+1:].strip() if colon_idx >= 0 else ''
            remaining_lines = '\n'.join(code.split('\n')[1:])
            body_parts = []
            if first_body_part:
                body_parts.append(first_body_part)
            if remaining_lines.strip():
                body_parts.append(remaining_lines)
            body = '\n'.join(body_parts)
            body_lines = body.split('\n')
            if body_lines:
                min_indent = float('inf')
                for line in body_lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                if min_indent == float('inf'):
                    min_indent = 0
                body_lines = [line[min_indent:] if len(line) > min_indent else line.lstrip() for line in body_lines]
            has_return = any('return ' in line for line in body_lines)
            if not has_return and body_lines:
                for i, line in enumerate(body_lines):
                    if line.strip() and not line.strip().startswith('#'):
                        body_lines[i] = 'return ' + line
                        break
            body = '\n    '.join(body_lines)
            out_line = f"def {func['name']}({params}): {body}"
        else:
            out_line = f"{func['name']} = {func['source_code']}"
    else:
        # Plain source_code expression - wrap in function definition
        args_str = ', '.join(func['args']) if func['args'] else ''
        code = func['source_code']
        if '\n' in code or code.strip().startswith('def '):
            # Multi-line or already a function definition
            out_line = code
        else:
            # Single expression - wrap in function
            out_line = f"def {func['name']}({args_str}):\n    return {code}"
%>
${out_line}

% endfor


# =============================================================================
# Loss Functions (Generated from Metadata)
# =============================================================================
<%
# Extract loss functions from optimization metadata
# Loss is now a Function with source_code, equation, or callable options
loss_functions = []
for opt in optim_list:
    loss_fn = getattr(opt, 'loss', None)
    if loss_fn:
        # Function-based loss
        loss_info = {
            'opt_name': getattr(opt, 'name', 'loss'),
            'name': getattr(loss_fn, 'name', 'loss'),
            'label': getattr(loss_fn, 'label', '') or getattr(opt, 'label', 'Loss'),
            'source_code': getattr(loss_fn, 'source_code', None),
            'equation': None,
            'callable': None,
            'input_key': None,  # Which key from predicted observation to use
            'output_key': None,  # Name for loss output
        }

        # Check for equation (supports rhs or output_equation)
        eq = getattr(loss_fn, 'equation', None) or getattr(loss_fn, 'output_equation', None)
        if eq:
            loss_info['equation'] = getattr(eq, 'rhs', None)

        # Check for aggregate specification (LossFunction attribute)
        aggregate = getattr(loss_fn, 'aggregate', None)
        if aggregate:
            loss_info['aggregate'] = {
                'over': getattr(aggregate, 'over', None),
                'type': getattr(aggregate, 'type', 'mean'),
            }
        else:
            loss_info['aggregate'] = None

        # Check for callable
        callable_def = getattr(loss_fn, 'callable', None)
        if callable_def:
            loss_info['callable'] = {
                'module': getattr(callable_def, 'module', None),
                'qualname': getattr(callable_def, 'qualname', None),
            }

        # Extract variable names from arguments
        # First argument = predicted data key, 'target' argument = target data key
        loss_args = getattr(loss_fn, 'arguments', []) or []
        arg_names = []
        for arg in loss_args:
            arg_name = getattr(arg, 'name', None)
            if arg_name:
                arg_names.append(arg_name)
        # First non-target argument is the predicted key
        if arg_names:
            loss_info['input_key'] = arg_names[0]
            # Look for explicit 'target' argument, otherwise use second arg
            if 'target' in arg_names:
                loss_info['target_key'] = 'target'
            elif len(arg_names) > 1:
                loss_info['target_key'] = arg_names[1]

        # Get predicted_from and targets observations
        predicted_from = getattr(opt, 'predicted_from', None)
        if predicted_from:
            pred_name = getattr(predicted_from, 'name', str(predicted_from)) if hasattr(predicted_from, 'name') else str(predicted_from)
            loss_info['predicted_from'] = pred_name
            # If no explicit input_key, use last pipeline output from predicted_from
            if not loss_info['input_key']:
                loss_info['input_key'] = get_pipeline_output_key(pred_name)
        else:
            loss_info['predicted_from'] = None

        targets = getattr(opt, 'targets', [])
        target_names = [getattr(t, 'name', str(t)) if hasattr(t, 'name') else str(t) for t in targets] if targets else []
        loss_info['targets'] = target_names
        # Get target output key
        if target_names:
            loss_info['target_output_key'] = get_pipeline_output_key(target_names[0])

        loss_functions.append(loss_info)

# Helper: extract observation names that might be referenced in an equation
def extract_obs_refs_from_equation(eq_str, obs_names):
    """Find observation names referenced in an equation string."""
    if not eq_str:
        return []
    refs = []
    for obs_name in obs_names:
        if obs_name in eq_str:
            refs.append(obs_name)
    return refs
%>
% for loss_fn in loss_functions:
<%
    input_key = loss_fn.get('input_key')
    target_key = loss_fn.get('target_key')
    predicted_from = loss_fn.get('predicted_from')
    targets = loss_fn.get('targets', [])
    source_code = loss_fn.get('source_code')
    equation = loss_fn.get('equation')
    callable_info = loss_fn.get('callable')

    # Validate: loss MUST have equation, source_code, or callable
    has_loss_def = bool(source_code or equation or (callable_info and callable_info.get('qualname')))
    assert has_loss_def, f"Loss '{loss_fn['opt_name']}' must specify equation, source_code, or callable"

    # Default key names if not specified
    pred_var = input_key or 'predicted'
    targ_var = target_key or 'target'

    # Find observation names referenced in the equation
    # These observations need to be called and their outputs made available
    all_obs_names = list(observations.keys()) if observations else []
    obs_refs = extract_obs_refs_from_equation(equation, all_obs_names) if equation else []
%>

def loss_${loss_fn['opt_name']}(model_fn, state, target_data: jnp.ndarray = None):
    """${loss_fn['label']}

    Predicted from: ${predicted_from or 'first target'}
    Targets: ${', '.join(targets) if targets else 'None'}
    Input key: ${input_key or 'auto'} -> ${pred_var}
    Target key: ${target_key or 'auto'} -> ${targ_var}
% if obs_refs:
    Observations called: ${', '.join(obs_refs)}
% endif
    """
% if obs_refs:
    # Call observations referenced in the loss equation and extract their primary outputs
% for obs_name in obs_refs:
<%
    # Get the primary output key for this observation (last pipeline step output)
    obs_output_key = get_pipeline_output_key(obs_name)
    # Use _data suffix to avoid shadowing the observation function name
    var_name = obs_name + '_data'
%>
    _${obs_name}_result = ${obs_name}(model_fn, state)
% if obs_output_key:
    ${var_name} = _${obs_name}_result['${obs_output_key}']
% else:
    ${var_name} = _${obs_name}_result.get('data', next((v for v in _${obs_name}_result.values() if hasattr(v, 'shape')), _${obs_name}_result))
% endif
% endfor

<%
    # Replace observation.key references in equation with proper dict access
    # e.g., simulated_psd.psd -> _simulated_psd_result['psd']
    rendered_eq = str(equation)
    import re
    # Find all observation.key patterns and build variable mappings
    var_mappings = {}  # Maps variable names to their source expressions
    for obs_name in obs_refs:
        # Replace obs_name.key with variable name (for use in vmap)
        pattern = re.escape(obs_name) + r'\.(\w+)'
        matches = re.findall(pattern, rendered_eq)
        for key in matches:
            var_name = f"{obs_name}_{key}"
            var_mappings[var_name] = f"_{obs_name}_result['{key}']"
            rendered_eq = re.sub(re.escape(obs_name) + r'\.' + re.escape(key), var_name, rendered_eq)
        # Also replace bare obs_name (without .key) with obs_name_data
        rendered_eq = re.sub(r'\b' + re.escape(obs_name) + r'\b(?!_result|_)', obs_name + '_data', rendered_eq)

    # Get aggregate specification from loss function
    aggregate = loss_fn.get('aggregate')
    has_aggregation = aggregate is not None and aggregate.get('over')
    aggregate_over = aggregate.get('over') if aggregate else None
    # Handle enum values (may be PermissibleValue with .text attribute or string)
    _agg_type = aggregate.get('type', 'mean') if aggregate else 'mean'
    aggregate_type = getattr(_agg_type, 'text', str(_agg_type)) if _agg_type else 'mean'
%>
% if has_aggregation and var_mappings:
    # Aggregated loss: apply per-${aggregate_over}, then ${aggregate_type}
<%
    # Get the variable names for vmap arguments
    vmapped_vars = list(var_mappings.keys())
    vmapped_sources = [var_mappings[v] for v in vmapped_vars]
%>
    def _per_element_loss(${', '.join(vmapped_vars)}, _target):
        return ${rendered_eq.replace('target_data', '_target')}
    _per_element_losses = jax.vmap(_per_element_loss)(${', '.join(vmapped_sources)}, target_data)
% if aggregate_type == 'mean':
    loss_value = jnp.mean(_per_element_losses)
% elif aggregate_type == 'sum':
    loss_value = jnp.sum(_per_element_losses)
% elif aggregate_type == 'max':
    loss_value = jnp.max(_per_element_losses)
% elif aggregate_type == 'min':
    loss_value = jnp.min(_per_element_losses)
% else:
    loss_value = _per_element_losses  # No reduction
% endif
% else:
    # Compute loss directly (no aggregation)
    loss_value = ${rendered_eq}
% endif
    _aux = {${', '.join(["'" + obs + "': " + obs + "_data" for obs in obs_refs])}}


% elif predicted_from:
    # Run predicted observation
    _pred_result = ${predicted_from}(model_fn, state)
% if input_key:
    # Extract predicted data with exact key name for equation
    ${pred_var} = _pred_result['${input_key}']
% else:
    # Fallback: use 'data' key or first array value
    ${pred_var} = _pred_result.get('data', next(v for v in _pred_result.values() if hasattr(v, 'shape')))
% endif

% if equation:
    # Equation-based loss using exact keys: ${equation}
    # Apply vmap for per-row computation on 2D arrays, then average
    def _scalar_loss(${pred_var}, ${targ_var}):
        return ${jaxcode(equation)}
    # vmap over first axis (rows) and average the results
    loss_value = jnp.mean(jax.vmap(_scalar_loss)(${pred_var}, target_data))
% elif source_code:
    # Source code loss
    _loss_fn = ${source_code.strip()}
    loss_value = _loss_fn(${pred_var}, target_data)
% elif callable_info and callable_info.get('qualname'):
    # Callable loss: ${callable_info.get('module', '')}.${callable_info['name']}
    loss_value = ${callable_info['name']}(${pred_var}, target_data)
% endif
    _aux = _pred_result
% elif targets:
    # Fallback: use first target observation for predictions
    _pred_result = ${targets[0]}(model_fn, state)
% if input_key:
    ${pred_var} = _pred_result['${input_key}']
% else:
    ${pred_var} = _pred_result.get('data', next(v for v in _pred_result.values() if hasattr(v, 'shape')))
% endif
% if equation:
    def _scalar_loss(${pred_var}, ${targ_var}):
        return ${jaxcode(equation)}
    loss_value = jnp.mean(jax.vmap(_scalar_loss)(${pred_var}, target_data))
% elif source_code:
    _loss_fn = ${source_code.strip()}
    loss_value = _loss_fn(${pred_var}, target_data)
% endif
    _aux = _pred_result
% else:
    # No observation specified, run generic simulation
    _pred_result = model_fn(state)
% if input_key:
    ${pred_var} = _pred_result['${input_key}']
% else:
    ${pred_var} = _pred_result.get('data', next(v for v in _pred_result.values() if hasattr(v, 'shape')))
% endif
% if equation:
    def _scalar_loss(${pred_var}, ${targ_var}):
        return ${jaxcode(equation)}
    loss_value = jnp.mean(jax.vmap(_scalar_loss)(${pred_var}, target_data))
% elif source_code:
    _loss_fn = ${source_code.strip()}
    loss_value = _loss_fn(${pred_var}, target_data)
% endif
    _aux = _pred_result
% endif

    return loss_value, _aux

% endfor

def make_loss_fn(model_fn, target_data, loss_type: str = None):
    """Create a loss function closure for optimization.

    Loss functions are generated from optimization metadata.
    Each loss MUST specify equation, source_code, or callable.
    """
% if loss_functions:
    # Available loss functions from metadata: ${', '.join([lf['opt_name'] for lf in loss_functions])}
    if loss_type is None:
        loss_type = "${loss_functions[0]['opt_name']}"
% for loss_fn in loss_functions:
    ${'if' if loop.first else 'elif'} loss_type == "${loss_fn['opt_name']}":
        return lambda state: loss_${loss_fn['opt_name']}(model_fn, state, target_data)
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

def mark_parameters_optimizable(state, n_nodes: int = ${n_nodes}):
    """Mark parameters as optimizable and set their shapes."""
    init_state = copy.deepcopy(state)

    % for param in optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
%>
    init_state.dynamics.${param_name} = Parameter(init_state.dynamics.${param_name})
    % if is_heterogeneous:
    init_state.dynamics.${param_name}.shape = (n_nodes,)
    % endif
    % endfor

    # Note: Coupling parameters are stored under state.coupling[coupling_key].params
    # and require different handling than dynamics parameters.
    # For now, only dynamics parameters are marked as optimizable.
    # To optimize coupling parameters, access them via:
    #   init_state.coupling['${target_coupling_term}'].params.G = Parameter(...)

    return init_state


def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    print_every: int = 10,
    has_aux: bool = True,
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
    callback = MultiCallback([DefaultPrintCallback(every=print_every)])
    return OptaxOptimizer(loss_fn, opt_fn(learning_rate), callback=callback, has_aux=has_aux)


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
%>
<%
    output_key = expl.get('output_key')
%>
def ${expl['name']}(state, model_fn, n_pmap: int = ${expl['n_parallel']}):
    """${expl['label']} - Parameter exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    Observable: ${expl['observable']}${"['" + output_key + "']" if output_key else ""}
    """
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % endfor
    grid = Space(grid_state, mode="${expl['mode']}")
% if output_key:
    # Extract '${output_key}' from observation dict
    def observable_fn(s):
        result = ${expl['observable']}(model_fn, s)
        return result['${output_key}'] if isinstance(result, dict) else result
% else:
    observable_fn = lambda s: ${expl['observable']}(model_fn, s)
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
    state = sim_result.state
    result = sim_result.result
    transient = sim_result.transient
    observations = sim_result.observations

    results = Bunch(
        model_fn=model_fn,
        state=state,
        result=result,
        transient=transient,
        network=network,
        observations=observations,
    )

    % if has_optimization:
    # Optimization workflow
    if mode in ('optimization', 'all'):
        if target_data is None:
            raise ValueError("target_data is required for optimization mode")

        # Mark parameters as optimizable
        init_state = mark_parameters_optimizable(state)

        # Create loss function with target data (loss_type defaults to first from metadata)
        loss_type = kwargs.get('loss_type', None)
        loss_fn = make_loss_fn(model_fn, target_data, loss_type=loss_type)

        # Run optimization
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

    % if has_explorations:
    # Exploration workflow
    if mode in ('exploration', 'all'):
        explorations_result = Bunch()

        % for expl in explorations:
        explorations_result.${expl['name']} = ${expl['name']}(
            state, model_fn, n_pmap=kwargs.get('n_pmap', ${expl['n_parallel']})
        )
        % endfor

        results.explorations = explorations_result
    % endif

    return results
