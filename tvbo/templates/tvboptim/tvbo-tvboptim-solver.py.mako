# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Solver/Integration Template
=====================================

Generates solver configuration for tvboptim.experimental.network_dynamics.

This template generates a `get_solver()` function that returns a properly configured
solver instance. If state variables have bounds, the solver is wrapped in BoundedSolver.

Context Variables:
- experiment: SimulationExperiment instance (optional, provides model + integration)
- integration: Integration instance (required if no experiment)
- model: Dynamics instance (optional, provides state variable bounds)

Output:
- Solver imports
- get_solver() function
- Module-level solver constant
</%doc>
<%
import jax.numpy as jnp

# Map method names to tvboptim solver classes
SOLVER_MAP = {
    'euler': 'Euler',
    'heun': 'Heun',
    'heunstochastic': 'Heun',
    'runge_kutta': 'RungeKutta4',
    'rungekutta': 'RungeKutta4',
    'rk4': 'RungeKutta4',
    'rungekutta4thorder': 'RungeKutta4',
    'dopri5': 'DiffraxSolver',
    'tsit5': 'DiffraxSolver',
    'adaptive': 'DiffraxSolver',
}

# Get integration and model from context
if 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.dynamics
else:
    integration = context.get('integration')
    model = context.get('model', None)

method = (integration.method or 'euler').lower() if integration else 'euler'
solver_class = SOLVER_MAP.get(method, 'Euler')
is_diffrax = solver_class == 'DiffraxSolver'
dt = float(integration.step_size) if integration and integration.step_size else 0.1

# Extract state variable bounds for BoundedSolver
from tvbo.templates.tvboptim.utils import get_state_bounds, format_bounds_array, resolve_solver_kwargs
state_bounds_lo, state_bounds_hi, has_state_bounds = get_state_bounds(model)
state_bounds_lo_str = format_bounds_array(state_bounds_lo, 'jax')
state_bounds_hi_str = format_bounds_array(state_bounds_hi, 'jax')

# Differentiation strategy -> native-solver kwargs, resolved in the tvboptim Python
# layer (shared with the experiment template). Diffrax has no such knobs.
solver_kwargs_str = resolve_solver_kwargs(integration, dt, is_diffrax=is_diffrax)
%>
% if is_diffrax:
import diffrax
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver, BoundedSolver
% else:
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}, BoundedSolver
% endif


def get_solver(block_size=None):
    """Get configured solver instance.

    Returns the solver specified in integration metadata.

    ``block_size`` (native solvers only) sets the nested-block-scan granularity so a
    streaming reduction (``prepare(reduce=...)``) folds the observable in-carry instead
    of materializing the trajectory; ``None`` keeps the default single scan. Ignored by
    Diffrax solvers, which have no such knob.
% if has_state_bounds:
    Wrapped in BoundedSolver to enforce state variable domain constraints:
    % for i, sv_name in enumerate(model.state_variables.keys()):
    - ${sv_name}: [${state_bounds_lo[i]}, ${state_bounds_hi[i]}]
    % endfor
% endif
    """
% if is_diffrax:
    % if method == 'dopri5':
    base_solver = DiffraxSolver(diffrax.Dopri5())
    % elif method == 'tsit5':
    base_solver = DiffraxSolver(diffrax.Tsit5())
    % else:
    base_solver = DiffraxSolver(diffrax.Dopri5())
    % endif
% else:
    _solver_kwargs = dict(${solver_kwargs_str})
    if block_size is not None:
        _solver_kwargs['block_size'] = block_size
    base_solver = ${solver_class}(**_solver_kwargs)
% endif
% if has_state_bounds:
    # Wrap solver with BoundedSolver to enforce state variable domain constraints
    # Shape (n_states, 1) broadcasts with state array (n_states, n_nodes)
    return BoundedSolver(
        base_solver,
        low=jnp.array(${state_bounds_lo_str})[:, None],
        high=jnp.array(${state_bounds_hi_str})[:, None]
    )
% else:
    return base_solver
% endif


# Module-level solver instance (use get_solver() for fresh instance)
solver = get_solver()
DT = ${dt}
