# -*- coding: utf-8 -*-
#
# Module: utils.py
#
# RateML Template Utilities
# =========================
#
# Minimal utilities for generating RateML-style CUDA/Python code from TVBO.
# The TVBO LinkML datamodel already has all metadata - we just need code printers.
#
"""
Usage in templates:
    <%
    from tvbo.templates.rateml.utils import cuda_code, python_code
    %>
    ${cuda_code(sv.equation.rhs)}
"""
import re
from typing import Optional, Dict, Any

import sympy.printing.c as spc
from sympy import Symbol, Function, Pow, Float
from sympy.printing.pycode import PythonCodePrinter
from tvbo.knowledge.simulation.equations import sympify as tvbo_sympify


# =============================================================================
# CUDA Code Printer
# =============================================================================

class CUDACodePrinter(spc.C99CodePrinter):
    """CUDA code printer extending SymPy's C99 printer.
    
    Handles:
    - powf instead of pow for floats
    - CUDA math functions (expf, sinf, etc.)
    - PI/INF macros
    """
    
    printmethod = "_cuda_code"
    
    _default_settings = {
        **spc.C99CodePrinter._default_settings,
        'precision': None,  # Use powf, expf, etc.
    }
    
    # CUDA single-precision math functions
    _cuda_functions = {
        'exp': 'expf',
        'log': 'logf',
        'sin': 'sinf',
        'cos': 'cosf',
        'tan': 'tanf',
        'asin': 'asinf',
        'acos': 'acosf',
        'atan': 'atanf',
        'sqrt': 'sqrtf',
        'abs': 'fabsf',
        'tanh': 'tanhf',
        'sinh': 'sinhf',
        'cosh': 'coshf',
    }
    
    def __init__(self, settings=None):
        super().__init__(settings or {})
        self.known_functions.update(self._cuda_functions)
    
    def _print_Pow(self, expr):
        """Use powf for float exponentiation."""
        base = self._print(expr.base)
        exp = self._print(expr.exp)
        
        # Special cases
        if expr.exp == -1:
            return f"(1.0f / {base})"
        if expr.exp == 0.5:
            return f"sqrtf({base})"
        if expr.exp == 2:
            return f"({base} * {base})"
        
        return f"powf({base}, {exp})"
    
    def _print_Float(self, expr):
        """Ensure floats have 'f' suffix for CUDA."""
        val = super()._print_Float(expr)
        if 'e' in val.lower() or '.' in val:
            if not val.endswith('f'):
                val = val + 'f'
        return val
    
    def _print_Integer(self, expr):
        """Keep integers as floats for consistency in CUDA."""
        return f"{int(expr)}.0f"
    
    def _print_Symbol(self, expr):
        """Handle special symbols."""
        name = super()._print_Symbol(expr)
        # Replace pi/inf with CUDA macros
        if name.lower() == 'pi':
            return 'PI'
        if name.lower() == 'inf':
            return 'INF'
        return name
    
    def _print_Piecewise(self, expr):
        """Convert Piecewise to ternary operators."""
        args = expr.args
        if len(args) == 2:
            # Simple if-else
            cond = self._print(args[0][1])
            val_true = self._print(args[0][0])
            val_false = self._print(args[1][0])
            return f"(({cond}) ? ({val_true}) : ({val_false}))"
        
        # Nested ternary for multiple conditions
        result = self._print(args[-1][0])  # Default (else)
        for val, cond in reversed(args[:-1]):
            cond_str = self._print(cond)
            val_str = self._print(val)
            result = f"(({cond_str}) ? ({val_str}) : ({result}))"
        return result


# =============================================================================
# Numba Code Printer (for TVB Python models)
# =============================================================================

class NumbaPrinter(PythonCodePrinter):
    """Python printer compatible with Numba's nopython mode.
    
    Uses numpy functions but avoids constructs Numba doesn't support.
    """
    
    def _print_Piecewise(self, expr):
        """Numba-compatible piecewise using if/else."""
        # For numba gufuncs, we need simple if/else, not np.where
        args = expr.args
        if len(args) == 2:
            cond = self._print(args[0][1])
            val_true = self._print(args[0][0])
            val_false = self._print(args[1][0])
            return f"({val_true} if {cond} else {val_false})"
        
        # Nested for multiple conditions
        result = self._print(args[-1][0])
        for val, cond in reversed(args[:-1]):
            cond_str = self._print(cond)
            val_str = self._print(val)
            result = f"({val_str} if {cond_str} else {result})"
        return result


# =============================================================================
# Convenience Functions for Templates
# =============================================================================

_cuda_printer = CUDACodePrinter()
_numba_printer = NumbaPrinter()


def cuda_code(expr, local_dict: Optional[Dict[str, Any]] = None) -> str:
    """Convert expression to CUDA code string.
    
    Args:
        expr: SymPy expression, Equation object, or string
        local_dict: Optional dict of symbols for parsing strings
    
    Returns:
        CUDA code string
    """
    if expr is None:
        return ""
    
    # Handle Equation objects
    if hasattr(expr, 'rhs'):
        expr = expr.rhs
    
    # Parse string expressions
    if isinstance(expr, str):
        try:
            expr = tvbo_sympify(expr, local_dict=local_dict)
        except Exception:
            # Fallback: just do basic string replacements
            return _string_to_cuda(expr)
    
    return _cuda_printer.doprint(expr)


def python_code(expr, local_dict: Optional[Dict[str, Any]] = None) -> str:
    """Convert expression to Python/Numba code string.
    
    Args:
        expr: SymPy expression, Equation object, or string
        local_dict: Optional dict of symbols for parsing strings
    
    Returns:
        Python code string
    """
    if expr is None:
        return ""
    
    # Handle Equation objects
    if hasattr(expr, 'rhs'):
        expr = expr.rhs
    
    # Parse string expressions
    if isinstance(expr, str):
        try:
            expr = tvbo_sympify(expr, local_dict=local_dict)
        except Exception:
            return expr  # Return as-is if parsing fails
    
    return _numba_printer.doprint(expr)


def _string_to_cuda(expr_str: str) -> str:
    """Simple string-based conversion to CUDA for expressions that fail parsing."""
    result = str(expr_str)
    
    # Power syntax: ** -> powf
    # Match x**y patterns
    power_pattern = r'(\w+|\([^)]+\))\s*\*\*\s*(\w+|\([^)]+\))'
    while re.search(power_pattern, result):
        result = re.sub(power_pattern, r'powf(\1, \2)', result)
    
    # pi -> PI, inf -> INF
    result = re.sub(r'\bpi\b', 'PI', result)
    result = re.sub(r'\binf\b', 'INF', result)
    
    # Common functions to float versions
    for fn in ['exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'tanh']:
        result = re.sub(rf'\b{fn}\s*\(', f'{fn}f(', result)
    
    return result


# =============================================================================
# Template Helpers (direct attribute access, minimal processing)
# =============================================================================

def has_boundaries(model) -> bool:
    """Check if any state variable has boundaries defined."""
    if not hasattr(model, 'state_variables') or not model.state_variables:
        return False
    return any(
        getattr(sv, 'boundaries', None) is not None
        for sv in model.state_variables.values()
    )


def get_initial_value(sv) -> float:
    """Get initial value for state variable, with sensible default."""
    if hasattr(sv, 'initial_value') and sv.initial_value is not None:
        return float(sv.initial_value)
    if hasattr(sv, 'domain') and sv.domain:
        lo = getattr(sv.domain, 'lo', 0.0) or 0.0
        hi = getattr(sv.domain, 'hi', 1.0) or 1.0
        return (lo + hi) / 2
    return 0.0


def get_domain_str(obj) -> str:
    """Get domain as 'lo, hi' string for state variable initialization."""
    domain = getattr(obj, 'domain', None)
    if domain:
        lo = getattr(domain, 'lo', 0.0) or 0.0
        hi = getattr(domain, 'hi', 1.0) or 1.0
        return f"{lo}, {hi}"
    if hasattr(obj, 'initial_value') and obj.initial_value is not None:
        v = float(obj.initial_value)
        return f"{v}, {v}"
    return "0.0, 1.0"


def get_boundary_str(sv) -> str:
    """Get boundaries as 'lo, hi' string for state variable clipping."""
    boundaries = getattr(sv, 'boundaries', None)
    if boundaries:
        lo = getattr(boundaries, 'lo', None)
        hi = getattr(boundaries, 'hi', None)
        if lo is not None and hi is not None:
            return f"{lo}, {hi}"
    return ""


def get_range_str(param) -> str:
    """Get parameter range as 'lo=x, hi=y, step=z' string for NArray domain."""
    domain = getattr(param, 'domain', None)
    if domain:
        lo = getattr(domain, 'lo', 0.0) or 0.0
        hi = getattr(domain, 'hi', 1.0) or 1.0
        step = getattr(domain, 'step', 0.01) or 0.01
        return f"lo={lo}, hi={hi}, step={step}"
    return ""



