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

jaxcode = lambda expr: render_expression(expr, format='jax')
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
# Only compute LEAF observations (those not used as source by others)
# This avoids redundant computation since derived obs call their parents
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

# Find which observations are used as sources by others
_used_as_source = set()
for name, obs in _obs_list:
    src_obs = getattr(obs, 'source_observation', None)
    if src_obs:
        src_name = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)
        _used_as_source.add(src_name)

# Only compute leaf observations (not used as source) that have all args
observation_names = [
    name for name, obs in _obs_list
    if obs_has_all_args(obs) and name not in _used_as_source
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
has_fc = any('fc' in str(getattr(obs, 'name', '')).lower() for obs in observations.values())

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
    observations = Bunch()
% for obs_name in observation_names:
    observations.${obs_name} = ${obs_name}(model_fn, state, **kwargs)
% endfor

    return Bunch(model_fn=model_fn, state=state, result=result, transient=result_transient, observations=observations)


# =============================================================================
# Observable Functions (Generated from Pipeline Metadata)
# =============================================================================

<%include file="tvbo-tvboptim-observation.py.mako" />


# =============================================================================
# Utility Functions
# =============================================================================

def cauchy_pdf(x: jnp.ndarray, x0: float, gamma: float = 1.0) -> jnp.ndarray:
    """Cauchy (Lorentzian) distribution for target spectra."""
    return 1.0 / (jnp.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))

# =============================================================================
# Loss Functions (Generated from Metadata)
# =============================================================================
<%
# Extract loss functions from optimization metadata
loss_functions = []
for opt in optim_list:
    loss_eq = getattr(opt, 'loss', None)
    if loss_eq:
        loss_rhs = getattr(loss_eq, 'rhs', None)
        loss_label = getattr(loss_eq, 'label', 'loss')
        targets = getattr(opt, 'targets', [])
        target_names = [getattr(t, 'name', str(t)) if hasattr(t, 'name') else str(t) for t in targets] if targets else []

        # Get output key from target observation's pipeline
        output_key = get_pipeline_output_key(target_names[0]) if target_names else None

        loss_functions.append({
            'name': getattr(opt, 'name', 'loss'),
            'rhs': loss_rhs,
            'label': loss_label,
            'targets': target_names,
            'output_key': output_key,
        })
%>
# Utility functions for loss computation
def correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Pearson correlation between two arrays."""
    return jnp.corrcoef(x.ravel(), y.ravel())[0, 1]


def rmse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Root mean squared error between two arrays."""
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def mse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Mean squared error between two arrays."""
    return jnp.mean((x - y) ** 2)

% for loss_fn in loss_functions:
<%
    output_key = loss_fn.get('output_key')
%>

def loss_${loss_fn['name']}(model_fn, state, target_data: jnp.ndarray):
    """${loss_fn['label']}: ${loss_fn['rhs'] or 'correlation-based loss'}"""
% if loss_fn['targets']:
    # Target observation: ${', '.join(loss_fn['targets'])}
    _obs_result = ${loss_fn['targets'][0]}(model_fn, state)
% else:
    _obs_result = spectrum(model_fn, state)
% endif
% if output_key:
    predicted = _obs_result['${output_key}']
% else:
    # Fallback: use 'data' key or first array value
    predicted = _obs_result.get('data', next(v for v in _obs_result.values() if hasattr(v, 'shape')))
% endif
% if loss_fn['rhs'] and 'corrcoef' in str(loss_fn['rhs']).lower():
    # Correlation-based loss: ${loss_fn['rhs']}
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted, target_data
    )
    loss_value = 1.0 - correlations.mean()
% elif loss_fn['rhs'] and 'mse' in str(loss_fn['rhs']).lower():
    # MSE loss
    loss_value = mse(predicted, target_data)
% elif loss_fn['rhs'] and 'rmse' in str(loss_fn['rhs']).lower():
    # RMSE loss
    loss_value = rmse(predicted, target_data)
% else:
    # Default: correlation loss
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted, target_data
    )
    loss_value = 1.0 - correlations.mean()
% endif
    return loss_value, _obs_result

% endfor
% if not loss_functions:
# No loss functions defined in metadata - providing basic loss function

def loss_default(model_fn, state, target_data: jnp.ndarray):
    """Default correlation-based loss function."""
    _obs_result = spectrum(model_fn, state)
    predicted = _obs_result.get('data', next(v for v in _obs_result.values() if hasattr(v, 'shape')))
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted, target_data
    )
    loss_value = 1.0 - correlations.mean()
    return loss_value, _obs_result

% endif
% if has_fc:

def functional_connectivity(model_fn, state, skip_t: int = 20):
    """Compute functional connectivity from simulation."""
    from tvboptim.observations.observation import compute_fc
    result = model_fn(state)
    return compute_fc(result, skip_t=skip_t)


def loss_fc_correlation(model_fn, state, target_fc: jnp.ndarray, skip_t: int = 20):
    """Loss based on correlation between simulated and target FC."""
    from tvboptim.observations.observation import fc_corr
    fc_sim = functional_connectivity(model_fn, state, skip_t)
    return 1.0 - fc_corr(fc_sim, target_fc), fc_sim
% endif


def make_loss_fn(model_fn, target_data, loss_type: str = None):
    """Create a loss function closure for optimization."""
% if loss_functions:
    # Available loss functions from metadata: ${', '.join([lf['name'] for lf in loss_functions])}
    if loss_type is None:
        loss_type = "${loss_functions[0]['name']}"
% for loss_fn in loss_functions:
    ${'if' if loop.first else 'elif'} loss_type == "${loss_fn['name']}":
        return lambda state: loss_${loss_fn['name']}(model_fn, state, target_data)
% endfor
% else:
    if loss_type is None:
        loss_type = "default"
    if loss_type == "default":
        return lambda state: loss_default(model_fn, state, target_data)
% endif
% if has_fc:
    elif loss_type == "fc_correlation":
        return lambda state: loss_fc_correlation(model_fn, state, target_data)
% endif
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: ${', '.join([lf['name'] for lf in loss_functions]) if loss_functions else 'default'}")


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
