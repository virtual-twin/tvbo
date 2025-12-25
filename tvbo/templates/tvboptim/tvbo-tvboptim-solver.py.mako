# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Solver/Integration Template
=====================================

Generates solver configuration for tvboptim.experimental.network_dynamics
from a TVBO Integration specification.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- integration: Integration instance (required if no experiment)

Output:
- Python code for solver instantiation (Euler, Heun, RK4, or Diffrax)
</%doc>
<%
import numpy as np

# Get integration from context
if 'experiment' in context.keys():
    integration = experiment.integration
else:
    integration = context['integration']

# Determine solver type from integration method
method = integration.method.lower() if hasattr(integration, 'method') else 'euler'

# Map TVBO method names to tvboptim solver classes
solver_map = {
    'euler': 'Euler',
    'heun': 'Heun',
    'heunstochastic': 'Heun',  # Stochastic handled by noise component
    'runge_kutta': 'RungeKutta4',
    'rungekutta': 'RungeKutta4',
    'rk4': 'RungeKutta4',
    'dopri5': 'DiffraxSolver',
    'tsit5': 'DiffraxSolver',
    'adaptive': 'DiffraxSolver',
}

solver_class = solver_map.get(method, 'Euler')
is_diffrax = solver_class == 'DiffraxSolver'

# Get timestep
dt = float(integration.step_size) if hasattr(integration, 'step_size') and integration.step_size else 0.1

# Stochastic configuration
is_stochastic = integration.noise is not None if hasattr(integration, 'noise') else False
%>
"""Solver configuration for tvboptim Network Dynamics.

Auto-generated from TVBO integration specification.
Method: ${method}
Timestep: ${dt}
Stochastic: ${is_stochastic}
"""

% if is_diffrax:
import diffrax
from tvboptim.experimental.network_dynamics.solvers import DiffraxSolver
% else:
from tvboptim.experimental.network_dynamics.solvers import ${solver_class}
% endif


def get_solver():
    """Get configured solver instance.

    Returns
    -------
    solver : NativeSolver or DiffraxSolver
        Configured solver for integration
    """
% if is_diffrax:
    # Diffrax adaptive solver (no delays supported)
    % if method == 'dopri5':
    diffrax_solver = diffrax.Dopri5()
    % elif method == 'tsit5':
    diffrax_solver = diffrax.Tsit5()
    % else:
    diffrax_solver = diffrax.Dopri5()  # Default adaptive
    % endif

    return DiffraxSolver(diffrax_solver)
% else:
    return ${solver_class}()
% endif


def get_integration_params():
    """Get integration parameters.

    Returns
    -------
    dict
        Integration parameters including dt, t0, t1
    """
    return {
        'dt': ${dt},
        't0': 0.0,
        't1': 1000.0,  # Override in simulation
    }


# Convenience: Export solver instance directly
solver = get_solver()
dt = ${dt}
