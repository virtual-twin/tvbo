# -*- coding: utf-8 -*-
"""
Utilities for base templates.

Extracts Python logic from Mako templates for cleaner, testable code.
"""


def get_coupling_terms(model):
    """Extract coupling terms from model, separating global from local.

    Returns:
        tuple: (all_terms, global_terms, has_local_coupling)
    """
    all_terms = list(model.coupling_terms.keys()) if hasattr(model, 'coupling_terms') and model.coupling_terms else []
    global_terms = [ct for ct in all_terms if ct != 'local_coupling']
    has_local = 'local_coupling' in all_terms
    return all_terms, global_terms, has_local


def get_func_name(model, override=None):
    """Get function name from model, with optional override."""
    if override:
        return override
    if hasattr(model, 'name') and model.name:
        return model.name.replace(' ', '').replace('-', '')
    return 'dfun'


def get_func_args(f):
    """Extract argument names from a function object.

    Handles both dict-like (f.arguments.values()) and list-like arguments.
    """
    args = f.arguments
    if hasattr(args, 'values'):
        return [arg.name if hasattr(arg, 'name') else str(arg) for arg in args.values()]
    return [arg.name if hasattr(arg, 'name') else str(arg) for arg in args]


def np_module(fmt):
    """Get numpy module name for format."""
    return 'jnp' if fmt == 'jax' else 'np'


# Special functions that require scipy.special (numpy) or jax.scipy.special (jax)
SCIPY_SPECIAL_FUNCTIONS = {'erfc', 'erf', 'gamma', 'gammaln', 'bessel', 'beta'}


def needs_scipy_special(model, fmt):
    """Check if model equations use scipy.special functions.
    
    Renders derived variables and state equations to detect scipy.special usage.
    Returns True if any equation contains scipy.special (for numpy) or jsp.special (for jax).
    """
    search_str = 'scipy.special' if fmt in ('numpy', 'scipy') else 'jsp.special'
    
    # Check derived variables
    for dv in (model.derived_variables or {}).values():
        code = model.render_equation(dv, format=fmt)
        if search_str in code:
            return True
    
    # Check state variable equations
    for sv in model.state_variables.values():
        code = model.render_equation(sv, format=fmt)
        if search_str in code:
            return True
    
    return False
