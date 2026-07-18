# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Full Simulation Workflow Template
===========================================

Generates a complete tvboptim Network Dynamics simulation workflow.
Uses <%include> to compose dfun, cfun templates.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Complete Python script for running a tvboptim simulation
</%doc>
<%
from tvbo.codegen import render_expression
import numpy as np

# Must have experiment
assert 'experiment' in context.keys(), "experiment required for simulation template"

model = experiment.dynamics
coupling = experiment.coupling
integration = experiment.integration
network = experiment.network

jaxcode = lambda expr: render_expression(expr, format='jax')
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Extract key metadata
state_names = list(model.state_variables.keys())
param_names = [p.name for p in model.parameters.values()]
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Build coupling_inputs dict:
# Each coupling_input name → its dimension (default 1)
coupling_inputs_dict = {}

if hasattr(model, 'coupling_inputs') and model.coupling_inputs:
    for ci_name, ci in model.coupling_inputs.items():
        dim = getattr(ci, 'dimension', 1) or 1
        coupling_inputs_dict[ci_name] = dim
elif hasattr(model, 'coupling_terms') and model.coupling_terms:
    for ct_name in model.coupling_terms.keys():
        coupling_inputs_dict[ct_name] = 1

# Coupling metadata
has_delay = hasattr(coupling, 'delayed') and coupling.delayed
# Differentiable (interpolated) delays — opt-in only (see experiment template).
interpolate_delays = bool(getattr(coupling, 'interpolate_delays', False))
coupling_class = coupling.name.replace(' ', '').replace('-', '') if hasattr(coupling, 'name') and coupling.name else 'GeneratedCoupling'
coupling_param_names = [p.name for p in coupling.parameters.values()] if hasattr(coupling, 'parameters') and coupling.parameters else []
coupling_param_defaults = {p.name: float(p.value) if p.value is not None else 1.0 for p in coupling.parameters.values()} if hasattr(coupling, 'parameters') and coupling.parameters else {}
incoming_states = list(getattr(coupling, 'incoming_states', None) or [])
local_states = list(getattr(coupling, 'local_states', None) or [])

# Delay-graph selection (see graph_selection): tract lengths → DenseLengthGraph
# (delays = lengths / conduction_speed, speed a live sweepable/differentiable leaf);
# a network carrying only explicit per-edge "delay" attributes → DenseDelayGraph.
from tvbo.templates.tvboptim.utils import graph_selection

use_length_graph, use_delay_graph = graph_selection(network, has_delay)

# Extract state variable bounds (for BoundedSolver)
from tvbo.templates.tvboptim.utils import get_state_bounds
state_bounds_lo, state_bounds_hi, has_state_bounds = get_state_bounds(model)

# Integration metadata
SOLVER_MAP = {'euler': 'Euler', 'heun': 'Heun', 'heunstochastic': 'Heun', 'rk4': 'RungeKutta4', 'rungekutta4thorder': 'RungeKutta4', 'runge_kutta': 'RungeKutta4', 'rungekutta': 'RungeKutta4'}
method = (integration.method or 'euler').lower()
solver_class = SOLVER_MAP.get(method, 'Euler')
dt = float(integration.step_size) if integration.step_size else 0.1
has_noise = integration.noise is not None
noise_sigma = np.asarray(experiment.noise_sigma_array).flatten().tolist() if hasattr(experiment, 'noise_sigma_array') else [0.1]

# Network metadata
n_nodes = N_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
_cs = getattr(network, 'conduction_speed', None)
conduction_speed = float(_cs.value if hasattr(_cs, 'value') else _cs) if _cs else 3.0

# Simulation parameters
t1_default = float(integration.duration) if hasattr(integration, 'duration') and integration.duration else 1000.0

# Class names
dynamics_class = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'

# Events metadata (stimuli and other time-dependent inputs)
events_list = list(experiment.events.values()) if experiment.events else []
stimulus_events = [ev for ev in events_list
                   if ('stimul' in str(getattr(ev, 'event_type', 'stimulus')))
                   or (str(getattr(ev, 'event_type', '')) in ('continuous', 'discrete'))]
has_stimulus_events = len(stimulus_events) > 0
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

# =============================================================================
# Imports
# =============================================================================

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from tvboptim.experimental.network_dynamics import Network, prepare, solve
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
% if has_stimulus_events:
from tvboptim.experimental.network_dynamics.external_input.base import AbstractExternalInput
% endif
% if has_delay:
from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseLengthGraph
% else:
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
% endif
% if has_noise:
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
% endif

# Module-level model function (set in run_simulation)
model = None


# =============================================================================
# Solver Configuration
# =============================================================================

<%include file="tvbo-tvboptim-solver.py.mako" />


# =============================================================================
# Dynamics Model
# =============================================================================

<%include file="/tvboptim/tvbo-tvboptim-dfun.py.mako" />


# =============================================================================
# Coupling Function
# =============================================================================

<%include file="tvbo-tvboptim-cfun.py.mako" />

% if has_stimulus_events:

# =============================================================================
# External Inputs (Events)
# =============================================================================

<%include file="tvbo-tvboptim-stimulus.py.mako" />
% endif


# =============================================================================
# Network Setup
# =============================================================================

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
    noise_sigma: float = ${noise_sigma[0]},
    % if has_delay:
    max_delay: float = None,
    % endif
) -> Network:
    """Create configured Network instance.

    ``max_delay`` (delayed coupling only) is the ``max_delay_bound``: a static
    history-buffer length decoupled from the delay values. Pass it when the delays
    vary differentiably (e.g. ``delays = tract_lengths / v`` with conduction speed
    ``v`` optimised) so the buffer keeps a fixed shape. When None it is derived from
    the build-speed max delay.
    """
    % if use_length_graph:
    # tract lengths → DenseLengthGraph derives delays = lengths / conduction_speed;
    # conduction speed stays a live, sweepable and differentiable graph leaf.
    if distances is None:
        distances = jnp.zeros_like(weights)
    _speed = ${conduction_speed}
    _max_delay_bound = max_delay if max_delay is not None else (float(jnp.max(distances)) / _speed if _speed > 0 else 0.0)
    graph = DenseLengthGraph(weights, distances, speed=_speed, region_labels=region_labels, max_delay_bound=_max_delay_bound)
    % elif use_delay_graph:
    # explicit per-edge delays (no tract lengths) → DenseDelayGraph directly.
    if delays is None:
        delays = jnp.zeros_like(weights)
    delays = jnp.nan_to_num(delays)
    graph = DenseDelayGraph(weights, delays, region_labels=region_labels, max_delay_bound=max_delay)
    % else:
    graph = DenseGraph(weights, region_labels=region_labels)
    % endif

    dynamics = ${dynamics_class}(**(dynamics_params or {}))

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
    noise = AdditiveNoise(sigma=noise_sigma) if noise_sigma > 0 else None
    % else:
    noise = None
    % endif

    # Build coupling dict: each coupling_input name maps to the coupling function
    coupling_dict = {
    % for ci_name in coupling_inputs_dict.keys():
        '${ci_name}': coupling,
    % endfor
    }

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


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(
    network: Network,
    t1: float = ${t1_default},
    dt: float = ${dt},
    t0: float = 0.0,
) -> tuple:
    """Run network simulation."""
    model_fn, state = prepare(network, get_solver(), t0=t0, t1=t1, dt=dt)
    result = model_fn(state)
    return model_fn, state, result


# =============================================================================
# Constants
# =============================================================================

CONDUCTION_SPEED = ${conduction_speed}
N_NODES = ${n_nodes}
DT = ${dt}
T1 = ${t1_default}
NOISE_SIGMA = ${noise_sigma[0]}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_experiment(weights, distances=None, delays=None, region_labels=None):
    """Run the complete ${dynamics_class} simulation workflow."""
    global model

    weights = jnp.array(weights)
    % if use_length_graph:
    # tract lengths → DenseLengthGraph derives delays = lengths / conduction_speed.
    network = create_network(weights, distances=distances, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % elif use_delay_graph:
    # explicit per-edge delays → DenseDelayGraph over the delay matrix directly.
    if delays is None:
        delays = jnp.array(distances) / CONDUCTION_SPEED if (distances is not None and CONDUCTION_SPEED > 0) else jnp.zeros_like(weights)
    else:
        delays = jnp.array(delays)
    network = create_network(weights, delays=delays, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % else:
    network = create_network(weights, region_labels=region_labels, noise_sigma=NOISE_SIGMA)
    % endif

    model_fn, state, result = run_simulation(network, t1=T1, dt=DT)
    model = model_fn

    return {
        'model_fn': model_fn,
        'state': state,
        'result': result,
        'network': network,
    }
