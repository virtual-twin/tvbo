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
        exp_info['observable'] = getattr(observable, 'name', str(observable))
    explorations.append(exp_info)

# === Observations metadata ===
observations = getattr(experiment, 'observations', None) or {}
if hasattr(observations, 'values'):
    observations = dict(observations.items()) if hasattr(observations, 'items') else {}
elif hasattr(observations, '__iter__') and not isinstance(observations, dict):
    observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations)}

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
) -> tuple:
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
    return Bunch(model_fn=model_fn, state=state, result=result, result_transient=result_transient)


# =============================================================================
# Target Data Generation
# =============================================================================

def cauchy_pdf(x: jnp.ndarray, x0: float, gamma: float = 1.0) -> jnp.ndarray:
    """Cauchy (Lorentzian) distribution for target spectra."""
    return 1.0 / (np.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))


def gaussian_pdf(x: jnp.ndarray, mu: float, sigma: float = 1.0) -> jnp.ndarray:
    """Gaussian distribution for target spectra."""
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * np.pi))


def generate_target_peak_frequencies(
    lengths: jnp.ndarray,
    reference_idx: int = 0,
    f_min: float = 7.0,
    f_max: float = 11.0,
) -> jnp.ndarray:
    """Generate target peak frequencies from distance to reference region."""
    dist_from_ref = lengths[reference_idx, :]
    min_dist = dist_from_ref.min()
    max_dist = dist_from_ref.max()
    delta_f = (f_max - f_min) / (max_dist - min_dist + 1e-8)
    peak_freqs = f_max - delta_f * (dist_from_ref - min_dist)
    return peak_freqs


def generate_target_spectra(
    frequencies: jnp.ndarray,
    peak_freqs: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """Generate target PSDs from peak frequencies using Cauchy distribution."""
    target_psds = jax.vmap(lambda fp: cauchy_pdf(frequencies, fp, gamma))(peak_freqs)
    return target_psds


# =============================================================================
# Observable Functions (Generated from Metadata)
# =============================================================================
<%
# Build observation dependency graph
obs_by_name = {obs_name: obs for obs_name, obs in observations.items()}

def get_obs_equation(obs):
    """Extract equation RHS from observation."""
    if hasattr(obs, 'equation') and obs.equation:
        return getattr(obs.equation, 'rhs', None)
    return None

def get_obs_params(obs):
    """Extract parameters from observation."""
    params = {}
    if hasattr(obs, 'parameters') and obs.parameters:
        for pname, p in obs.parameters.items():
            if hasattr(p, 'value'):
                params[pname] = p.value
    return params
%>
% for obs_name, obs in observations.items():
<%
    eq_rhs = get_obs_equation(obs)
    obs_params = get_obs_params(obs)
    src_obs = getattr(obs, 'source_observation', None)
    if src_obs:
        src_obs_name = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)
    else:
        src_obs_name = None
%>

def ${obs_name}(model_fn, state, dt: float = ${dt}, downsample: int = 10, **kwargs):
    """${getattr(obs, 'description', obs_name) or obs_name}"""
% if eq_rhs and 'welch' in eq_rhs.lower():
    # Spectral analysis via Welch's method
    result = model_fn(state)
    fs = kwargs.get('fs', ${obs_params.get('fs', '1000.0 / (dt * downsample)')})
    data = result.data[::downsample, 0, :]
    f, Pxx = jax.scipy.signal.welch(data.T, fs=fs)
    return f, Pxx
% elif eq_rhs and 'mean' in eq_rhs.lower():
    # Averaging operation: ${eq_rhs}
    f, Pxx = ${src_obs_name if src_obs_name else 'spectrum'}(model_fn, state, dt=dt, downsample=downsample, **kwargs)
    return f, jnp.mean(Pxx, axis=0)
% elif eq_rhs and 'argmax' in eq_rhs.lower():
    # Peak extraction: ${eq_rhs}
    f, S = ${src_obs_name if src_obs_name else 'avg_spectrum'}(model_fn, state, dt=dt, downsample=downsample, **kwargs)
    return f[jnp.argmax(S)]
% elif eq_rhs and 'cauchy' in eq_rhs.lower():
    # Target distribution generation: ${eq_rhs}
    gamma = kwargs.get('gamma', ${obs_params.get('gamma', 1.0)})
    f = kwargs.get('frequencies')
    peak_freqs = kwargs.get('peak_freqs')
    return jax.vmap(lambda fp: cauchy_pdf(f, fp, gamma))(peak_freqs)
% elif src_obs_name and not eq_rhs:
    # Pure delegation to: ${src_obs_name}
    return ${src_obs_name}(model_fn, state, dt=dt, downsample=downsample, **kwargs)
% else:
    # Generic observation - return simulation result
    result = model_fn(state)
    return result.data
% endif

% endfor
% if not has_observations:
# No observations defined in metadata - providing basic spectral functions

def spectrum(model_fn, state, dt: float = ${dt}, downsample: int = 10, fs: float = None):
    """Compute power spectral density using Welch's method."""
    result = model_fn(state)
    if fs is None:
        fs = 1000.0 / (dt * downsample)
    data = result.data[::downsample, 0, :]
    f, Pxx = jax.scipy.signal.welch(data.T, fs=fs)
    return f, Pxx


def avg_spectrum(model_fn, state, **kwargs):
    """Compute average power spectrum across all regions."""
    f, Pxx = spectrum(model_fn, state, **kwargs)
    return f, jnp.mean(Pxx, axis=0)


def peak_frequency(model_fn, state, **kwargs):
    """Extract peak frequency from average spectrum."""
    f, S = avg_spectrum(model_fn, state, **kwargs)
    return f[jnp.argmax(S)]

% endif

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
        loss_functions.append({
            'name': getattr(opt, 'name', 'loss'),
            'rhs': loss_rhs,
            'label': loss_label,
            'targets': target_names,
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

def loss_${loss_fn['name']}(model_fn, state, target_data: jnp.ndarray):
    """${loss_fn['label']}: ${loss_fn['rhs'] or 'correlation-based loss'}"""
% if loss_fn['targets']:
    # Target observation: ${', '.join(loss_fn['targets'])}
    f, predicted = ${loss_fn['targets'][0]}(model_fn, state)
% else:
    f, predicted = spectrum(model_fn, state)
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
    return loss_value, predicted

% endfor
% if not loss_functions:
# No loss functions defined in metadata - providing basic loss function

def loss_default(model_fn, state, target_data: jnp.ndarray):
    """Default correlation-based loss function."""
    f, predicted = spectrum(model_fn, state)
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted, target_data
    )
    loss_value = 1.0 - correlations.mean()
    return loss_value, predicted

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
def ${expl['name']}(state, model_fn, n_pmap: int = ${expl['n_parallel']}):
    """${expl['label']} - Parameter exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    Observable: ${expl['observable']}
    """
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % endfor
    grid = Space(grid_state, mode="${expl['mode']}")
    observable_fn = lambda s: ${expl['observable']}(model_fn, s)
    exec_runner = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec_runner.run()
    return {'grid': grid, 'results': jnp.stack(results)}


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
    dict
        Results dictionary containing:
        - 'model_fn': Compiled model function
        - 'state': Initial state
        - 'result': Simulation result (if mode includes simulation)
        - 'network': Network instance
        - 'fitted_params': Optimized parameters (if mode='optimization')
        - 'fitting_data': Optimization history (if mode='optimization')
        - 'exploration_results': Grid search results (if mode='exploration')
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
    result_transient = sim_result.result_transient

    results = {
        'model_fn': model_fn,
        'state': state,
        'result': result,
        'result_transient': result_transient,
        'network': network,
    }

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
        exploration_results = {}

        % for expl in explorations:
        exploration_results['${expl['name']}'] = ${expl['name']}(
            state, model_fn, n_pmap=kwargs.get('n_pmap', ${expl['n_parallel']})
        )
        % endfor

        results['exploration_results'] = exploration_results
    % endif

    return results
