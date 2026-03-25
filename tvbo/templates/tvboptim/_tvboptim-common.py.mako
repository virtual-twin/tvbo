# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Common Template Utilities
===================================

Shared utility <%def> blocks for tvboptim templates. Import with:
    <%namespace name="common" file="_tvboptim-common.py.mako"/>

Usage:
    ${common.jaxcode(expr)}
    ${common.array_input(arr)}

Note: This file only contains utility functions. Each template (dfun, cfun, etc.)
is self-contained. The sim template uses <%include> to compose them.
</%doc>
<%!
import numpy as np
from tvbo.codegen import render_expression

# Solver method -> tvboptim class mapping
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

# Known tvboptim coupling classes (import rather than generate)
KNOWN_TVBOPTIM_COUPLINGS = {
    'SigmoidalJansenRit', 'DelayedSigmoidalJansenRit',
    'LinearCoupling', 'DelayedLinearCoupling',
    'DifferenceCoupling', 'DelayedDifferenceCoupling',
    'FastLinearCoupling', 'SubspaceCoupling',
}
%>

## =============================================================================
## Code Rendering Utilities
## =============================================================================

<%def name="jaxcode(expr)">${render_expression(expr, format='jax') if expr else 'None'}</%def>

<%def name="array_input(arr)">jnp.array(${arr.tolist() if hasattr(arr, 'tolist') else list(arr)})</%def>

<%def name="sanitize_name(name, default='Generated')">${name.replace(' ', '').replace('-', '').replace('_', '') if name else default}</%def>

## =============================================================================
## Solver Utilities
## =============================================================================

<%def name="get_solver_class(method)">${SOLVER_MAP.get(method.lower() if method else 'euler', 'Euler')}</%def>

<%def name="is_diffrax(method)">${SOLVER_MAP.get(method.lower() if method else 'euler', 'Euler') == 'DiffraxSolver'}</%def>

<%def name="is_builtin_coupling(class_name)">${class_name in KNOWN_TVBOPTIM_COUPLINGS}</%def>
