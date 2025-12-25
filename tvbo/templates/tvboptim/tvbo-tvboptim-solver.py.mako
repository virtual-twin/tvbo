# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Solver/Integration Template
=====================================

Generates solver configuration for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- integration: Integration instance (required if no experiment)

Output:
- Solver getter function and constants
</%doc>
<%
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

# Get integration from context
if 'experiment' in context.keys():
    integration = experiment.integration
else:
    integration = context['integration']

method = integration.method.lower() if hasattr(integration, 'method') and integration.method else 'euler'
solver_class = SOLVER_MAP.get(method, 'Euler')
is_diffrax = solver_class == 'DiffraxSolver'
dt = float(integration.step_size) if hasattr(integration, 'step_size') and integration.step_size else 0.1
%>

% if is_diffrax:
import diffrax
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver
% else:
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% endif


def get_solver():
    """Get configured solver instance."""
% if is_diffrax:
    % if method == 'dopri5':
    return DiffraxSolver(diffrax.Dopri5())
    % elif method == 'tsit5':
    return DiffraxSolver(diffrax.Tsit5())
    % else:
    return DiffraxSolver(diffrax.Dopri5())
    % endif
% else:
    return ${solver_class}()
% endif


solver = get_solver()
dt = ${dt}
