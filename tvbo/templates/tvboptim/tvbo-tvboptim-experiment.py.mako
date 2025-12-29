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


# =============================================================================
# Loss Functions (Generated from Metadata)
# =============================================================================
<%
# Extract loss functions from optimization metadata
# Loss is now a FunctionCall - it references a function, not defines one
loss_functions = []
for opt in optim_list:
    loss_call = getattr(opt, 'loss', None)
    if loss_call:
        # FunctionCall has 'function' (reference) or 'callable' (inline)
        func_ref = getattr(loss_call, 'function', None)
        callable_ref = getattr(loss_call, 'callable', None)

        # Determine the function name to call
        if func_ref:
            # Reference to user-defined function (e.g., 'rmse')
            func_name = str(func_ref) if isinstance(func_ref, str) else getattr(func_ref, 'name', str(func_ref))
        elif callable_ref:
            func_name = getattr(callable_ref, 'name', None) or getattr(callable_ref, 'qualname', 'loss')
        else:
            func_name = 'loss'

        # Get argument names
        loss_args = getattr(loss_call, 'arguments', []) or []
        arg_names = [getattr(arg, 'name', None) for arg in loss_args if getattr(arg, 'name', None)]

        loss_functions.append({
            'opt_name': getattr(opt, 'name', 'loss'),
            'func_name': func_name,
            'arg_names': arg_names,
            'predicted_from': getattr(opt, 'predicted_from', None),
        })
%>
# Loss function wrappers (call observations, then referenced loss function)
% for loss_fn in loss_functions:
<%
    func_name = loss_fn['func_name']
    opt_name = loss_fn['opt_name']
    arg_names = loss_fn['arg_names']

    # Get predicted_from observation
    predicted_from = loss_fn.get('predicted_from')
    if predicted_from:
        pred_name = getattr(predicted_from, 'name', str(predicted_from)) if hasattr(predicted_from, 'name') else str(predicted_from)
    else:
        pred_name = None

    # First arg is typically simulated, second is empirical/target
    sim_arg = arg_names[0] if arg_names else 'simulated'
    emp_arg = arg_names[1] if len(arg_names) > 1 else 'empirical'
%>
def loss_${opt_name}(model_fn, state, target_data: jnp.ndarray = None):
    """Loss wrapper calling ${func_name}

    Calls observation '${pred_name or 'fc'}' and computes loss against target_data.
    """
% if pred_name:
    _obs_result = ${pred_name}(model_fn, state)
    ${sim_arg} = _obs_result[_obs_result._primary_key]
% else:
    # Default: call fc observation
    _obs_result = fc(model_fn, state)
    ${sim_arg} = _obs_result[_obs_result._primary_key]
% endif
    ${emp_arg} = target_data
    loss_value = ${func_name}(${sim_arg}, ${emp_arg})
    return loss_value, _obs_result

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
