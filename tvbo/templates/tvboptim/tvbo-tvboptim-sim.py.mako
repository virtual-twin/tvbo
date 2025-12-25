# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Full Simulation Workflow Template
===========================================

Generates a complete tvboptim Network Dynamics simulation workflow
from a TVBO SimulationExperiment specification.

This template produces a standalone Python script that includes:
- All model components (dynamics, coupling, noise, solver)
- Network setup with graph
- Simulation execution using prepare() and model()
- Result handling and visualization

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Complete Python script for running a tvboptim simulation
</%doc>
<%
from tvbo.export.code import render_expression
import numpy as np

# Must have experiment
assert 'experiment' in context.keys(), "experiment required for simulation template"

model = experiment.local_dynamics
coupling = experiment.coupling
integration = experiment.integration
network = experiment.network

jaxcode = lambda expr: render_expression(expr, format='jax')
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# ============================================================================
# Extract all metadata
# ============================================================================

# Dynamics metadata
state_names = list(model.state_variables.keys())
initial_state = [float(sv.initial_value) if sv.initial_value is not None else 0.0
                 for sv in model.state_variables.values()]
param_names = [p.name for p in model.parameters.values()]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in model.parameters.values()}
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []
coupling_terms = list(model.coupling_terms.keys()) if model.coupling_terms else ['default']
aux_names = list(model.derived_variables.keys()) if model.derived_variables else []

# Coupling metadata
coupling_params = list(coupling.parameters.values()) if hasattr(coupling, 'parameters') else []
coupling_param_names = [p.name for p in coupling_params]
coupling_param_defaults = {p.name: float(p.value) if p.value is not None else 1.0 for p in coupling_params}
has_delay = hasattr(coupling, 'delayed') and coupling.delayed

# Integration metadata
dt = float(integration.step_size) if integration.step_size else 0.1
method = integration.method.lower() if hasattr(integration, 'method') else 'euler'
solver_map = {'euler': 'Euler', 'heun': 'Heun', 'heunstochastic': 'Heun',
              'rk4': 'RungeKutta4', 'rungekutta': 'RungeKutta4'}
solver_class = solver_map.get(method, 'Euler')

# Noise metadata
has_noise = integration.noise is not None
if has_noise:
    noise_sigma = np.asarray(experiment.noise_sigma_array).flatten().tolist() if hasattr(experiment, 'noise_sigma_array') else [0.1]
else:
    noise_sigma = [0.0]

# Network metadata
n_nodes = network.number_of_regions
region_labels = list(network.region_labels) if hasattr(network, 'region_labels') and network.region_labels else None
# Handle conduction_speed as Parameter object or float
_cs = getattr(network, 'conduction_speed', None)
conduction_speed = float(_cs.value if hasattr(_cs, 'value') else _cs) if _cs is not None else 3.0

# Extract connectivity matrices from network metadata
weights_array = network.weights if hasattr(network, 'weights') and network.weights is not None else None
_distances = getattr(network, 'distances', None)
distances_array = _distances if _distances is not None else getattr(network, 'tract_lengths', None)
network_name = getattr(network, 'name', None) or getattr(network, 'label', None)

# Simulation parameters
horizon = experiment.horizon if hasattr(experiment, 'horizon') else 1
t1_default = float(integration.duration) if hasattr(integration, 'duration') and integration.duration else 1000.0
transient_time = float(integration.transient_time) if hasattr(integration, 'transient_time') and integration.transient_time else 0.0

# Class names
dynamics_class = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'
coupling_class = coupling.name.replace(' ', '').replace('-', '') if hasattr(coupling, 'name') and coupling.name else 'GeneratedCoupling'

# Known tvboptim coupling classes that should be imported, not generated
KNOWN_TVBOPTIM_COUPLINGS = {
    'SigmoidalJansenRit', 'DelayedSigmoidalJansenRit',
    'LinearCoupling', 'DelayedLinearCoupling',
    'DifferenceCoupling', 'DelayedDifferenceCoupling',
    'FastLinearCoupling', 'SubspaceCoupling',
}
use_builtin_coupling = coupling_class in KNOWN_TVBOPTIM_COUPLINGS

# Coupling state specifications (needed for both builtin and generated couplings)
incoming_states = getattr(coupling, 'incoming_states', None) or []
if isinstance(incoming_states, str):
    incoming_states = [incoming_states]
incoming_states = list(incoming_states) if incoming_states else []
local_states = getattr(coupling, 'local_states', None) or []
if isinstance(local_states, str):
    local_states = [local_states]
local_states = list(local_states) if local_states else []

# Monitors
monitors = getattr(experiment, 'monitors', None)
if callable(monitors):
    monitors = monitors()
elif hasattr(monitors, 'values'):
    monitors = list(monitors.values())
has_bold = monitors and any('bold' in str(type(m).__name__).lower() for m in monitors)
%>
"""
${dynamics_class} tvboptim Network Dynamics Simulation
${'=' * (len(dynamics_class) + 40)}

Auto-generated from TVBO SimulationExperiment specification.

Model: ${model.name if hasattr(model, 'name') else 'Generated'}
Coupling: ${coupling.name if hasattr(coupling, 'name') else 'Generated'}
Nodes: ${n_nodes}
Integration: ${solver_class} (dt=${dt}ms)
Stochastic: ${has_noise}
Delayed: ${has_delay}
"""

# ============================================================================
# Imports
# ============================================================================

import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# TVB-Optim Network Dynamics imports
from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
% if has_delay:
% if use_builtin_coupling:
from tvboptim.experimental.network_dynamics.coupling import ${coupling_class}
% else:
from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling
% endif
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
% else:
% if use_builtin_coupling:
from tvboptim.experimental.network_dynamics.coupling import ${coupling_class}
% else:
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
% endif
from tvboptim.experimental.network_dynamics.graph import DenseGraph
% endif
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif

% if has_bold:
# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold
from tvboptim.observations.observation import compute_fc, fc_corr, rmse
% endif

# Module-level model function placeholder (set in run_experiment or run_simulation)
# Observation functions use this to run the simulation
model = None


# ============================================================================
# Dynamics Model: ${dynamics_class}
# ============================================================================

class ${dynamics_class}(AbstractDynamics):
    """${dynamics_class} neural mass model.

    ${model.description if hasattr(model, 'description') and model.description else ''}

    State variables: ${state_names}
    Parameters: ${param_names}
    """

    STATE_NAMES = ${tuple(state_names)}
    INITIAL_STATE = ${tuple(initial_state)}
    % if aux_names:
    AUXILIARY_NAMES = ${tuple(aux_names)}
    % else:
    AUXILIARY_NAMES = ()
    % endif

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
    )

    COUPLING_INPUTS = {
        % for ct in coupling_terms:
        '${ct}': 1,
        % endfor
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute ${dynamics_class} dynamics."""
        # Unpack parameters
        % for name in param_names:
        ${name} = params.${name}
        % endfor

        % if derived_param_names:
        # Derived parameters
        % for dp in model.derived_parameters.values():
        ${dp.name} = ${jaxcode_obj(dp)}
        % endfor
        % endif

        # Unpack state variables
        % for i, svar in enumerate(state_names):
        ${svar} = state[${i}]
        % endfor

        # Unpack coupling
        % for i, ct in enumerate(coupling_terms):
        ${ct} = coupling.${ct}[0] if hasattr(coupling, '${ct}') else 0.0
        % endfor

        % if model.functions:
        # Helper functions
        % for f in model.functions.values():
        def ${f.name}(${', '.join([arg.name if hasattr(arg, 'name') else str(arg) for arg in (f.arguments.values() if hasattr(f.arguments, 'values') else f.arguments)])}):
            return ${jaxcode_obj(f)}
        % endfor
        % endif

        % if model.derived_variables:
        # Derived variables
        % for dv in model.derived_variables.values():
        ${dv.name} = ${jaxcode_obj(dv)}
        % endfor
        % endif

        # State derivatives
        % for sv in model.state_variables.values():
        d${sv.name}_dt = ${jaxcode_obj(sv)}
        % endfor

        derivatives = jnp.array([
            % for sv in model.state_variables.values():
            d${sv.name}_dt,
            % endfor
        ])

        % if aux_names:
        auxiliaries = jnp.array([${', '.join(aux_names)}])
        % else:
        auxiliaries = jnp.array([])
        % endif

        return derivatives, auxiliaries


# ============================================================================
# Coupling Function: ${coupling_class}
# ============================================================================

% if not use_builtin_coupling:
% if has_delay:
class ${coupling_class}(DelayedCoupling):
% else:
class ${coupling_class}(InstantaneousCoupling):
% endif
    """${coupling_class} coupling function."""

    N_OUTPUT_STATES = ${len(coupling_terms)}
    DEFAULT_PARAMS = Bunch(
        % for name in coupling_param_names:
        ${name}=${coupling_param_defaults.get(name, 1.0)},
        % endfor
        % if not coupling_param_names:
        G=1.0,
        % endif
    )

    def __init__(self, **kwargs):
        % if incoming_states:
        super().__init__(
            incoming_states=${incoming_states},
            % if local_states:
            local_states=${local_states},
            % endif
            **kwargs
        )
        % else:
        super().__init__(**kwargs)
        % endif

    % if hasattr(coupling, 'pre_expression') and coupling.pre_expression:
    def pre(self, incoming_states, local_states, params):
        % for name in coupling_param_names:
        ${name} = params.${name}
        % endfor
        % for i, sname in enumerate(incoming_states):
        ${sname} = incoming_states[${i}]
        % endfor
        return ${jaxcode(coupling.pre_expression.rhs)}
    % endif

    def post(self, summed_inputs, local_states, params):
        % for name in coupling_param_names:
        ${name} = params.${name}
        % endfor
        gx = summed_inputs
        % if hasattr(coupling, 'post_expression') and coupling.post_expression:
        return ${jaxcode(coupling.post_expression.rhs)}
        % else:
        G = params.G if hasattr(params, 'G') else 1.0
        return G * gx
        % endif
% else:
# Using built-in tvboptim coupling: ${coupling_class}
# Imported from tvboptim.experimental.network_dynamics.coupling
% endif


# ============================================================================
# Network Setup
# ============================================================================

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
    """Create configured Network instance.

    Parameters
    ----------
    weights : jnp.ndarray
        Connectivity weights matrix [n_nodes, n_nodes]
    % if has_delay:
    delays : jnp.ndarray, optional
        Delay matrix [n_nodes, n_nodes] in ms
    % endif
    region_labels : list, optional
        Names of brain regions
    dynamics_params : dict, optional
        Override dynamics parameters
    coupling_params : dict, optional
        Override coupling parameters
    noise_sigma : float, optional
        Noise standard deviation (default: ${noise_sigma[0]})

    Returns
    -------
    Network
        Configured network instance
    """
    # Create graph
    % if has_delay:
    if delays is None:
        delays = jnp.zeros_like(weights)
    graph = DenseDelayGraph(weights, delays, region_labels=region_labels)
    % else:
    graph = DenseGraph(weights, region_labels=region_labels)
    % endif

    # Create dynamics
    dynamics = ${dynamics_class}(**(dynamics_params or {}))

    # Create coupling
    # Default coupling parameters from metadata (including incoming_states)
    default_coupling_params = {
        % if incoming_states:
        'incoming_states': ${incoming_states},
        % endif
        % if local_states:
        'local_states': ${local_states},
        % endif
        % for name in coupling_param_names:
        '${name}': ${coupling_param_defaults.get(name, 1.0)},
        % endfor
    }
    default_coupling_params.update(coupling_params or {})
    coupling = ${coupling_class}(**default_coupling_params)

    % if has_noise:
    # Create noise
    noise = AdditiveNoise(sigma=noise_sigma) if noise_sigma > 0 else None
    % else:
    noise = None
    % endif

    # Assemble network
    return Network(
        dynamics=dynamics,
        coupling={'${coupling_terms[0]}': coupling},
        graph=graph,
        noise=noise,
    )


# ============================================================================
# Simulation Functions
# ============================================================================

def run_simulation(
    network: Network,
    t1: float = ${t1_default},
    dt: float = ${dt},
    t0: float = 0.0,
) -> tuple:
    """Run network simulation.

    Parameters
    ----------
    network : Network
        Configured network instance
    t1 : float
        Simulation end time in ms (default: ${t1_default})
    dt : float
        Integration timestep in ms (default: ${dt})
    t0 : float
        Simulation start time in ms (default: 0.0)

    Returns
    -------
    tuple
        (model_fn, state, result) - compiled model, initial state, simulation result
    """
    # Prepare simulation
    solver = ${solver_class}()
    model_fn, state = prepare(network, solver, t0=t0, t1=t1, dt=dt)

    # Run simulation
    result = model_fn(state)

    return model_fn, state, result


% if has_bold:
def run_with_bold(
    network: Network,
    t1: float = 120_000.0,  # 2 minutes for BOLD
    dt: float = ${dt},
    bold_period: float = 1000.0,
    skip_t: int = 20,
):
    """Run simulation with BOLD signal computation.

    Parameters
    ----------
    network : Network
        Configured network instance
    t1 : float
        Simulation duration in ms
    dt : float
        Integration timestep
    bold_period : float
        BOLD sampling period (TR) in ms
    skip_t : int
        Initial TRs to skip for transient removal

    Returns
    -------
    tuple
        (model_fn, state, result, bold_result, fc)
    """
    # Initial transient run
    solver = ${solver_class}()
    model_fn, state = prepare(network, solver, t0=0, t1=t1, dt=dt)
    result_init = model_fn(state)

    # Update history and run again
    network.update_history(result_init)
    model_fn, state = prepare(network, solver, t0=0, t1=t1, dt=dt)
    result = model_fn(state)

    # BOLD transformation
    bold_monitor = Bold(
        period=bold_period,
        downsample_period=dt,
        voi=0,
        history=result_init,
    )
    bold_result = bold_monitor(result)

    # Compute FC
    fc = compute_fc(bold_result, skip_t=skip_t)

    return model_fn, state, result, bold_result, fc
% endif


<%
# ============================================================================
# EXPLORATION METADATA
# ============================================================================
exploration_dict = getattr(experiment, 'explorations', None) or {}
# Handle both dict and list formats
if isinstance(exploration_dict, dict):
    exploration_list = list(exploration_dict.values())
elif isinstance(exploration_dict, list):
    exploration_list = exploration_dict
else:
    exploration_list = [exploration_dict] if exploration_dict else []

explorations = []
for expl in exploration_list:
    exp_info = {
        'name': getattr(expl, 'name', 'exploration'),
        'label': getattr(expl, 'label', ''),
        'mode': getattr(expl, 'mode', 'product'),
        'n_parallel': getattr(expl, 'n_parallel', 8),
        'axes': [],
    }
    params = getattr(expl, 'parameters', None)
    if params:
        # Handle dict or list of parameters
        param_iter = params.values() if hasattr(params, 'values') else params
        for param in param_iter:
            domain = getattr(param, 'domain', None)
            if domain:
                # Handle lo/hi as strings or floats
                lo = float(domain.lo) if domain.lo is not None else 0.0
                hi = float(domain.hi) if domain.hi is not None else 1.0
                n = int(domain.n) if hasattr(domain, 'n') and domain.n else 32
                exp_info['axes'].append({
                    'name': getattr(param, 'name', str(param)),
                    'lo': lo,
                    'hi': hi,
                    'n': n,
                    'log_scale': bool(getattr(domain, 'log_scale', False)),
                })
    observable = getattr(expl, 'observable', None)
    if observable:
        exp_info['observable'] = getattr(observable, 'name', str(observable))
    explorations.append(exp_info)

# ============================================================================
# OPTIMIZATION METADATA
# ============================================================================
optim_list = getattr(experiment, 'optimization', None) or []
if not isinstance(optim_list, list):
    optim_list = [optim_list]

optim_params = []
for name, param in model.parameters.items():
    is_free = getattr(param, 'free', False)
    has_domain = hasattr(param, 'domain') and param.domain is not None
    if is_free or has_domain:
        optim_params.append(param)

coupling_optim_params = []
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        is_free = getattr(param, 'free', False)
        has_domain = hasattr(param, 'domain') and param.domain is not None
        if is_free or has_domain:
            coupling_optim_params.append(param)

optimizer_name = 'adamaxw'
learning_rate = 0.001
max_steps = 100

for opt in optim_list:
    if hasattr(opt, 'algorithm') and opt.algorithm:
        optimizer_name = str(opt.algorithm)
    if hasattr(opt, 'learning_rate') and opt.learning_rate:
        learning_rate = float(opt.learning_rate)
    if hasattr(opt, 'max_iterations') and opt.max_iterations:
        max_steps = int(opt.max_iterations)
    if hasattr(opt, 'free_parameters') and opt.free_parameters:
        for pref in opt.free_parameters:
            pname = pref if isinstance(pref, str) else getattr(pref, 'name', None)
            if pname and pname in model.parameters:
                p = model.parameters[pname]
                if p not in optim_params:
                    optim_params.append(p)

# ============================================================================
# OBSERVATION METADATA
# ============================================================================
observations = getattr(experiment, 'observations', None) or {}
if not isinstance(observations, dict):
    if hasattr(observations, 'items'):
        observations = dict(observations.items())
    elif hasattr(observations, '__iter__'):
        observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations)}
    else:
        observations = {}

# Build observation DAG - topologically sort by dependencies
def topo_sort_observations(obs_dict):
    """Sort observations so dependencies come before dependents."""
    # Build dependency graph
    deps = {}
    for name, obs in obs_dict.items():
        src_obs = getattr(obs, 'source_observation', None)
        if src_obs:
            src_name = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)
            deps[name] = src_name
        else:
            deps[name] = None

    # Topological sort
    sorted_names = []
    visited = set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        dep = deps.get(name)
        if dep and dep in obs_dict:
            visit(dep)
        sorted_names.append(name)

    for name in obs_dict:
        visit(name)
    return [(name, obs_dict[name]) for name in sorted_names]

sorted_observations = topo_sort_observations(observations)

# Parse observations into structured info
obs_list = []
for obs_name, obs in sorted_observations:
    obs_info = {
        'name': obs_name,
        'label': getattr(obs, 'label', ''),
        'description': getattr(obs, 'description', ''),
        'source': None,
        'source_observation': None,
        'params': {},
        'pipeline': [],
    }

    # Get source (state variable) or source_observation (another observation)
    if hasattr(obs, 'source') and obs.source:
        obs_info['source'] = getattr(obs.source, 'name', str(obs.source))
    if hasattr(obs, 'source_observation') and obs.source_observation:
        src_obs = obs.source_observation
        obs_info['source_observation'] = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)

    # Get observation-level parameters
    if hasattr(obs, 'parameters') and obs.parameters:
        for pname, param in (obs.parameters.items() if hasattr(obs.parameters, 'items') else [(p.name, p) for p in obs.parameters]):
            obs_info['params'][pname] = getattr(param, 'value', None)

    # Parse pipeline (DAG of functions)
    if hasattr(obs, 'pipeline') and obs.pipeline:
        pipeline_funcs = list(obs.pipeline.values()) if hasattr(obs.pipeline, 'values') else list(obs.pipeline)
        for func in pipeline_funcs:
            func_info = {
                'name': getattr(func, 'name', 'step'),
                'output': getattr(func, 'output', None),
                'input': str(getattr(func, 'input', None)) if getattr(func, 'input', None) else None,
                'equation': None,
                'callable': None,
                'arguments': {},
            }
            # Get equation
            if hasattr(func, 'equation') and func.equation:
                func_info['equation'] = getattr(func.equation, 'rhs', None) or getattr(func.equation, 'righthandside', None)
                # Get equation parameters
                if hasattr(func.equation, 'parameters') and func.equation.parameters:
                    for pname, param in (func.equation.parameters.items() if hasattr(func.equation.parameters, 'items') else []):
                        func_info['arguments'][pname] = getattr(param, 'value', param)
            # Get callable
            if hasattr(func, 'callable') and func.callable:
                c = func.callable
                func_info['callable'] = {
                    'module': getattr(c, 'module', None),
                    'name': getattr(c, 'name', getattr(c, 'qualname', None)),
                }
            # Get arguments
            if hasattr(func, 'arguments') and func.arguments:
                args_list = list(func.arguments.values()) if hasattr(func.arguments, 'values') else list(func.arguments)
                for arg in args_list:
                    arg_name = getattr(arg, 'name', None)
                    arg_value = getattr(arg, 'value', None)
                    if arg_name:
                        func_info['arguments'][arg_name] = arg_value

            obs_info['pipeline'].append(func_info)

    # If no pipeline but has equation, create single-step pipeline
    if not obs_info['pipeline'] and hasattr(obs, 'equation') and obs.equation:
        eq = obs.equation
        obs_info['pipeline'].append({
            'name': obs_name,
            'output': obs_name,
            'input': None,
            'equation': getattr(eq, 'rhs', None),
            'callable': None,
            'arguments': {},
        })

    obs_list.append(obs_info)
%>
% if obs_list:
# ============================================================================
# Observation Functions (DAG-based, matching JR.qmd pattern)
# ============================================================================
# Note: These functions take `state` and use the module-level `model` function.
# They form a DAG where source observations are called by dependent observations.

<%
# Build map of observation outputs for cross-observation references
obs_output_map = {obs['name']: obs['name'] for obs in obs_list}

# Identify which observations are "root" (have source state var) vs "derived" (have source_observation)
root_observations = [obs for obs in obs_list if obs['source'] and not obs['source_observation']]
derived_observations = [obs for obs in obs_list if obs['source_observation']]
%>
% for obs in obs_list:
<%
    obs_name = obs['name']
    obs_source = obs['source']
    obs_src_obs = obs['source_observation']
    obs_params = obs['params']
    pipeline = obs['pipeline']

    # Get equation from pipeline if present
    obs_eq = None
    if pipeline:
        obs_eq = pipeline[0].get('equation') if pipeline else None
%>
def ${obs_name}(state):
    """${obs['label'] or obs['name']}

    ${obs['description'] or 'Auto-generated observation function.'}
    """
    % if obs_src_obs:
    ## Derived observation - calls another observation function
    % if 'argmax' in str(obs_eq) or 'peak' in obs_name.lower():
    ## Peak frequency extraction pattern
    f, S = ${obs_src_obs}(state)
    idx = jnp.argmax(S)
    return f[idx]
    % elif 'mean' in str(obs_eq):
    ## Average spectrum pattern
    f, Pxx = ${obs_src_obs}(state)
    return f, jnp.mean(Pxx, axis=0)
    % else:
    return ${obs_src_obs}(state)
    % endif
    % else:
    ## Root observation - runs model and extracts state variable
    result = model(state)
    % if obs_source:
    state_idx = ${state_names.index(obs_source) if obs_source in state_names else 0}
    % else:
    state_idx = 0
    % endif
    % if 'welch' in str(obs_eq) or 'spectrum' in obs_name.lower():
    ## Spectrum computation pattern (matches JR.qmd exactly)
    # Subsample by 10 to get 100 Hz, compute Welch PSD
    f, Pxx = jax.scipy.signal.welch(result.data[::10, state_idx, :].T, fs=100.0)
    return f, Pxx
    % else:
    return result.data[:, state_idx, :]
    % endif
    % endif


% endfor
% endif
% if explorations:
# ============================================================================
# Parameter Exploration
# ============================================================================

import copy
from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution

% for expl in explorations:
<%
    n_axes = len(expl['axes'])
    total_points = 1
    for ax in expl['axes']:
        total_points *= ax['n']
%>
def setup_${expl['name']}_grid(state):
    """Configure grid for ${expl['name']} exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    Mode: ${expl['mode']}
    """
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % endfor
    return Space(grid_state, mode="${expl['mode']}")


def run_${expl['name']}_exploration(state, observable_fn, n_pmap: int = ${expl['n_parallel']}):
    """Run ${expl['name']} parameter exploration."""
    grid = setup_${expl['name']}_grid(state)
    exec = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec.run()
    return grid, jnp.stack(results)


% endfor
% endif
% if optim_params or coupling_optim_params:
# ============================================================================
# Optimization Configuration
# ============================================================================

import copy
import optax
from tvboptim.types import Parameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingCallback


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

    % for param in coupling_optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False)
%>
    init_state.coupling.${param_name} = Parameter(init_state.coupling.${param_name})
    % if is_heterogeneous:
    init_state.coupling.${param_name}.shape = (n_nodes,)
    % endif
    % endfor

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
# ============================================================================
# Configuration Constants (from metadata)
# ============================================================================

CONDUCTION_SPEED = ${conduction_speed}
N_NODES = ${n_nodes}
DT = ${dt}
T1 = ${t1_default}
NOISE_SIGMA = ${noise_sigma[0]}
MAX_STEPS = ${max_steps}
% if explorations:
N_PMAP = ${explorations[0]['n_parallel'] if explorations else 8}
% endif


# ============================================================================
# Run Experiment
# ============================================================================

def run_experiment(
    weights,
    distances=None,
    region_labels=None,
% if optim_params or coupling_optim_params:
    target_data=None,
% endif
):
    """Run the complete ${dynamics_class} simulation workflow.

    All simulation parameters (dt, t1, noise, conduction_speed, optimizer settings)
    are taken from the experiment metadata. Only data inputs are required.
% if explorations:

    Explorations defined in metadata will be executed automatically.
% endif
% if optim_params or coupling_optim_params:

    Optimization defined in metadata will be executed automatically if target_data is provided.
% endif

    Parameters
    ----------
    weights : array-like
        Connectivity weights matrix [n_nodes x n_nodes]
    distances : array-like, optional
        Distance matrix [n_nodes x n_nodes] (e.g., tract lengths in mm)
    region_labels : list, optional
        Names of brain regions
% if optim_params or coupling_optim_params:
    target_data : array-like, optional
        Target data for optimization (e.g., target PSDs)
% endif

    Returns
    -------
    dict
        Results dictionary containing:
        - 'model_fn': Compiled model function
        - 'state': Initial simulation state
        - 'result': Simulation result
% if explorations:
        - 'exploration': (grid, results) from parameter exploration
% endif
% if optim_params or coupling_optim_params:
        - 'optimization': (fitted_params, fitting_data) if target_data provided
% endif
    """
    # Convert inputs to jax arrays
    weights = jnp.array(weights)
    n_nodes = weights.shape[0]

    % if has_delay:
    if distances is not None:
        delays = jnp.array(distances) / CONDUCTION_SPEED
    else:
        delays = jnp.zeros_like(weights)
    network = create_network(weights, delays, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % endif

    # Run simulation
    model_fn, state, result = run_simulation(network, t1=T1, dt=DT)

    # Set module-level model for observation functions
    global model
    model = model_fn

    results = {
        'model_fn': model_fn,
        'state': state,
        'result': result,
        'network': network,
    }

    % if explorations:
    # Parameter exploration (defined in metadata)
    # Note: Observations use module-level 'model' set above
    grid, expl_results = run_${explorations[0]['name']}_exploration(state, ${explorations[0].get('observable', 'peak_freq')}, n_pmap=N_PMAP)
    results['exploration'] = (grid, expl_results)
    % endif

    % if optim_params or coupling_optim_params:
    # Optimization (defined in metadata)
    if target_data is not None:
        init_state = mark_parameters_optimizable(state, n_nodes=n_nodes)

        def loss_fn(s):
            f, Pxx = spectrum(s)
            # Correlation-based loss
            corr = jnp.mean(jax.vmap(lambda a, b: jnp.corrcoef(a, b)[0, 1])(Pxx, target_data))
            return 1.0 - corr, Pxx

        fitted_params, fitting_data = run_optimization(init_state, loss_fn, max_steps=MAX_STEPS)
        results['optimization'] = (fitted_params, fitting_data)
    % endif

    return results
