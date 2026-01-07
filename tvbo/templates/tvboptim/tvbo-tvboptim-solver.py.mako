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
    'dopri5': 'DiffraxSolver',
    'tsit5': 'DiffraxSolver',
    'adaptive': 'DiffraxSolver',
}

# Get integration and model from context
if 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.local_dynamics
else:
    integration = context.get('integration')
    model = context.get('model', None)

method = (integration.method or 'euler').lower() if integration else 'euler'
solver_class = SOLVER_MAP.get(method, 'Euler')
is_diffrax = solver_class == 'DiffraxSolver'
dt = float(integration.step_size) if integration and integration.step_size else 0.1

# Extract state variable bounds for BoundedSolver
state_bounds_lo = []
state_bounds_hi = []
if model and hasattr(model, 'state_variables') and model.state_variables:
    for sv_name, sv in model.state_variables.items():
        lo, hi = None, None
        if hasattr(sv, 'domain') and sv.domain:
            lo = getattr(sv.domain, 'lo', None)
            hi = getattr(sv.domain, 'hi', None)
        state_bounds_lo.append(float(lo) if lo is not None else float('-inf'))
        state_bounds_hi.append(float(hi) if hi is not None else float('inf'))

# Check if any state has finite bounds (needs BoundedSolver)
has_state_bounds = any(lo != float('-inf') for lo in state_bounds_lo) or any(hi != float('inf') for hi in state_bounds_hi)
%>
% if is_diffrax:
import diffrax
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver, BoundedSolver
% else:
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}, BoundedSolver
% endif


def get_solver():
    """Get configured solver instance.

    Returns the solver specified in integration metadata.
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
    base_solver = ${solver_class}()
% endif
% if has_state_bounds:
    # Wrap solver with BoundedSolver to enforce state variable domain constraints
    # Shape (n_states, 1) broadcasts with state array (n_states, n_nodes)
    return BoundedSolver(
        base_solver,
        low=jnp.array(${state_bounds_lo})[:, None],
        high=jnp.array(${state_bounds_hi})[:, None]
    )
% else:
    return base_solver
% endif


# Module-level solver instance (use get_solver() for fresh instance)
solver = get_solver()
DT = ${dt}
