"""SymPy expression printers that render symbolic model math to backend source code.

Each printer subclasses a SymPy code printer and layers on TVBO's array-function
vocabulary (defined in `tvbo.parse.expression`), the backend-abstracted array
primitives from `_ArrayFunctionPrinterMixin` (slicing, reductions, broadcasting),
and per-backend syntax fixes. `get_printer` selects a printer by target format
(`numpy`, `jax`, `julia`, `mtk`, `fortran`, `python`, `lems`, `sympy`), and
`render_expression` is the high-level entry point that parses a string or SymPy
expression and prints it for the chosen backend.
"""

import logging

import sympy.printing.julia as spj
import sympy.printing.numpy as spn
import sympy.printing.fortran as spf
from sympy.printing.pycode import PythonCodePrinter as _PythonCodePrinter
from sympy import Symbol, S, Function, preorder_traversal
from sympy import latex
from sympy.printing import StrPrinter
from tvbo.datamodel.schema import Equation
from tvbo.parse.expression import parse_eq

logger = logging.getLogger(__name__)


# =============================================================================
# Array Function Printer Mappings
# =============================================================================
# Maps the ARRAY_FUNCTIONS (defined in tvbo.parse.expression) to their target
# implementations for each output format. Printers use these via known_functions.

ARRAY_FUNCTION_MAPPINGS = {
    "jax": {
        # reductions
        "sum": "jnp.sum",
        "mean": "jnp.mean",
        "std": "jnp.std",
        "var": "jnp.var",
        "max": "jnp.max",
        "min": "jnp.min",
        "abs": "jnp.abs",
        "prod": "jnp.prod",
        "cumsum": "jnp.cumsum",
        "diff": "jnp.diff",
        "argmax": "jnp.argmax",
        "argmin": "jnp.argmin",
        # shape / construction / manipulation
        "concatenate": "jnp.concatenate",
        "stack": "jnp.stack",
        "vstack": "jnp.vstack",
        "hstack": "jnp.hstack",
        "pad": "jnp.pad",
        "roll": "jnp.roll",
        "flip": "jnp.flip",
        "reshape": "jnp.reshape",
        "transpose": "jnp.transpose",
        "expand_dims": "jnp.expand_dims",
        "squeeze": "jnp.squeeze",
        "take": "jnp.take",
        "tile": "jnp.tile",
        "repeat": "jnp.repeat",
        "arange": "jnp.arange",
        "zeros": "jnp.zeros",
        "ones": "jnp.ones",
        "where": "jnp.where",
        "clip": "jnp.clip",
        "dot": "jnp.dot",
        "matmul": "jnp.matmul",
    },
    "numpy": {
        "sum": "np.sum",
        "mean": "np.mean",
        "std": "np.std",
        "var": "np.var",
        "max": "np.max",
        "min": "np.min",
        "abs": "np.abs",
        "prod": "np.prod",
        "cumsum": "np.cumsum",
        "diff": "np.diff",
        "argmax": "np.argmax",
        "argmin": "np.argmin",
        "concatenate": "np.concatenate",
        "stack": "np.stack",
        "vstack": "np.vstack",
        "hstack": "np.hstack",
        "pad": "np.pad",
        "roll": "np.roll",
        "flip": "np.flip",
        "reshape": "np.reshape",
        "transpose": "np.transpose",
        "expand_dims": "np.expand_dims",
        "squeeze": "np.squeeze",
        "take": "np.take",
        "tile": "np.tile",
        "repeat": "np.repeat",
        "arange": "np.arange",
        "zeros": "np.zeros",
        "ones": "np.ones",
        "where": "np.where",
        "clip": "np.clip",
        "dot": "np.dot",
        "matmul": "np.matmul",
    },
    "julia": {
        "sum": "sum",
        "mean": "mean",
        "std": "std",
        "var": "var",
        "max": "maximum",
        "min": "minimum",
        "abs": "abs",
        "prod": "prod",
        "cumsum": "cumsum",
        "diff": "diff",
        "argmax": "argmax",
        "argmin": "argmin",
        "concatenate": "vcat",
        "stack": "stack",
        "vstack": "vcat",
        "hstack": "hcat",
        "flip": "reverse",
        "reshape": "reshape",
        "transpose": "transpose",
        "clip": "clamp",
        "dot": "dot",
        "matmul": "*",
    },
    "python": {
        "sum": "sum",
        "mean": "statistics.mean",
        "std": "statistics.stdev",
        "var": "statistics.variance",
        "max": "max",
        "min": "min",
        "abs": "abs",
        "prod": "math.prod",
        "concatenate": "list.__add__",
    },
}


def inline_functions(expr, func_defs):
    """
    Inline all function applications in an expression.

    Parameters
    ----------
    expr : sympy.Expr
        The expression containing function calls to inline.
    func_defs : dict
        Dictionary mapping function name -> (arg_names, body_expr)
        where arg_names is a list of argument names and body_expr is the
        sympy expression for the function body.

    Returns
    -------
    sympy.Expr
        Expression with all function calls replaced by their inlined bodies.

    Example
    -------
    >>> from sympy import symbols, Function
    >>> x, y, v = symbols('x y v')
    >>> # Define Sigm(v) = 2*e0/(1 + exp(r*(v0 - v)))
    >>> func_defs = {'Sigm': (['v'], 2*e0/(1 + exp(r*(v0 - v))))}
    >>> expr = A*Sigm(x - y)
    >>> inline_functions(expr, func_defs)
    2*A*e0/(1 + exp(r*(v0 - x + y)))
    """
    result = expr
    for func_name, (arg_names, body) in func_defs.items():
        F = Function(func_name)
        # Find all applications of this function and replace them
        for sub_expr in list(preorder_traversal(result)):
            if hasattr(sub_expr, "func") and sub_expr.func == F:
                # Get the actual arguments
                actual_args = sub_expr.args
                # Create substitution dict: formal arg -> actual arg
                subs = {Symbol(name): arg for name, arg in zip(arg_names, actual_args)}
                # Substitute into body
                inlined = body.subs(subs)
                # Replace in result
                result = result.subs(sub_expr, inlined)
    return result


def print_Piecewise(Printer, expr, verbose=False):
    """
    Print Piecewise expressions as nested np.where statements.
    """
    args = expr.args

    # Start with the default case (the last piece)
    default = Printer._print(args[-1][0])
    result = default  # Default fallback for np.where

    if verbose:
        logger.debug("Piecewise expr=%s args=%s default=%s", expr, args, default)

    # Iterate over conditions and expressions in reverse order (excluding the default)
    for value, condition in reversed(args[:-1]):
        if verbose:
            logger.debug("Piecewise branch: condition=%s value=%s", condition, value)
        condition_str = Printer._print(condition)
        value_str = Printer._print(value)
        # Build the nested conditional via the printer's backend-abstracted primitive
        # (numpy/jax -> ``<mod>.where(...)``; julia -> ``ifelse(...)``).
        result = Printer._where3(condition_str, value_str, result)

    if verbose:
        logger.debug("Piecewise result=%s", result)
    return result


# Array-manipulation primitives that SymPy cannot represent natively. Each entry
# maps a function name to a handler ``(printer, expr) -> code string`` that emits
# the right call for the printer's array module (``printer._module``: np / jnp).
# Shared by NumPyPrinter and JaxPrinter so a new primitive is one dict entry.
# Handlers stay backend-agnostic: they call the printer's rendering *primitives*
# (``_cat``, ``_render_index``, ``_slice_axis``, ``_transpose``, ``_reduce_axis``,
# ``_shape``, …), which each printer implements for its own conventions (numpy/jax:
# Python 0-based slicing; Julia: 1-based ``end``-relative). Adding an op is one handler
# plus, where a backend's syntax differs, one primitive override — never per-backend
# string-building duplicated across printers.
def _afp_concatenate(p, expr):
    args = list(expr.args)
    if args and args[-1].is_integer:
        axis, arrays = int(args[-1]), args[:-1]
    else:
        axis, arrays = 0, args
    return p._cat([p._print(a) for a in arrays], axis)


def _afp_window_mean(p, expr):
    return p._window_mean(p._print(expr.args[0]), p._print(expr.args[1]))


def _afp_subsample(p, expr):
    """Strided slice of the leading (time) axis.

    ``subsample(x, step)``        -> ``x[::step]``
    ``subsample(x, start, step)`` -> ``x[start::step]``
    The 3-arg form lets a strided downsample (e.g. tvboptim BOLD TR sampling
    ``data[step::step]``) be authored as a declarative equation, not ``source_code``.
    """
    a = expr.args
    if len(a) >= 3:
        return p._render_index(a[0], [(a[1], None, a[2])])
    return p._render_index(a[0], [(None, None, a[1])])


def _afp_slice_axis(p, expr):
    """``slice_axis(x, axis, start, stop[, step])`` -> bounded slice of one axis (keeps ndim)."""
    a = expr.args
    step = a[4] if len(a) > 4 else None
    return p._slice_axis(p._print(a[0]), int(a[1]), a[2], a[3], step)


def _afp_slice_from(p, expr):
    """``slice_from(x, axis, start)`` -> open-ended slice of one axis (to the end)."""
    a = expr.args
    return p._slice_from(p._print(a[0]), int(a[1]), a[2])


def _afp_shape(p, expr):
    """``shape(x, axis)`` -> length of ``x`` along ``axis`` (for open-ended slices etc.)."""
    return p._shape(p._print(expr.args[0]), int(expr.args[1]))


def _afp_global_mean(p, expr):
    return p._reduce_axis("mean", p._print(expr.args[0]), -2, keepdims=True)


def _afp_transpose(p, expr):
    return p._transpose(p._print(expr.args[0]))


def _afp_mode_dot(p, expr):
    """Contract X's mode axis with matrix M (TVB's ``numpy.dot(xi, A_ik)``)."""
    return p._mat_dot(p._print(expr.args[0]), p._print(expr.args[1]))


def _afp_mode_sum(p, expr):
    """Sum over the mode axis, broadcasting back (TVB's ``coupling[0].sum(axis=1)[:, None]``)."""
    return p._reduce_axis("sum", p._print(expr.args[0]), -1, keepdims=True)


_ARRAY_FUNCTION_PRINTERS = {
    "concatenate": _afp_concatenate,
    "window_mean": _afp_window_mean,
    "subsample": _afp_subsample,
    "slice_axis": _afp_slice_axis,
    "slice_from": _afp_slice_from,
    "shape": _afp_shape,
    "global_mean": _afp_global_mean,
    "transpose": _afp_transpose,
    "mode_dot": _afp_mode_dot,
    "mode_sum": _afp_mode_sum,
}

# Ops Julia renders natively (slice/stride family + shape, plus the mode-axis
# contractions used by multi-mode models); the rest defer to Julia's
# name-mappings (vcat/transpose/…) or graceful-degrade, so Julia output never regresses.
_JULIA_HANDLED_OPS = {"subsample", "slice_axis", "slice_from", "shape", "mode_dot", "mode_sum"}


class _ArrayFunctionPrinterMixin:
    """Shared printer hooks + backend-abstracted array primitives.

    Routes array primitives (``concatenate``, ``subsample``, ``mode_dot``, …) through
    ``_ARRAY_FUNCTION_PRINTERS`` and ``Piecewise`` through ``print_Piecewise``,
    deferring to the parent printer otherwise. Kept as a mixin listed first in the MRO
    so ``super()`` resolves to the concrete SymPy printer base.

    The ``_*`` primitive methods below emit the **numpy/jax** forms (Python, 0-based
    slicing). ``JuliaPrinter`` overrides the ones whose syntax differs (1-based,
    ``end``-relative indexing), so each array op is defined once as a backend-agnostic
    handler and only its differing syntax is overridden per backend.
    """

    # --- routing ---
    def _print_Piecewise(self, expr):
        return print_Piecewise(self, expr)

    def _print_Function(self, expr):
        handler = _ARRAY_FUNCTION_PRINTERS.get(expr.func.__name__)
        if handler is not None:
            return handler(self, expr)
        return super()._print_Function(expr)

    # --- backend-abstracted array primitives (numpy/jax defaults) ---
    def _afn(self, name):
        """Module-qualify an array function, e.g. ``jnp.take``."""
        return f"{self._module}.{name}"

    def _where3(self, cond, a, b):
        return f"{self._afn('where')}({cond}, {a}, {b})"

    def _cat(self, arrays, axis):
        return f"{self._afn('concatenate')}([{', '.join(arrays)}], axis={axis})"

    def _transpose(self, base):
        return f"{base}.T"

    def _mat_dot(self, a, b):
        return f"{self._afn('dot')}({a}, {b})"

    def _reduce_axis(self, fn, base, axis, keepdims=False):
        kw = ", keepdims=True" if keepdims else ""
        return f"{self._afn(fn)}({base}, axis={axis}{kw})"

    def _window_mean(self, X, w):
        return f"{self._afn('mean')}({X}.reshape(-1, {w}, *{X}.shape[1:]), axis=1)"

    def _shape(self, base, axis):
        return f"{base}.shape[{axis}]"

    def _render_index(self, base, specs):
        """Render ``base[...]`` with Python 0-based slices. ``specs`` is a per-axis list;
        each entry is ``None`` (full ``:``) or ``(start, stop, step)`` with ``None`` parts."""
        parts = []
        for s in specs:
            if s is None:
                parts.append(":")
                continue
            start, stop, step = s
            lo = "" if start is None else self._print(start)
            hi = "" if stop is None else self._print(stop)
            parts.append(f"{lo}:{hi}" if step is None else f"{lo}:{hi}:{self._print(step)}")
        return f"{self._print(base)}[{', '.join(parts)}]"

    def _slice_axis(self, base, axis, start, stop, step=None):
        rng = f"{self._afn('arange')}({self._print(start)}, {self._print(stop)}"
        rng += f", {self._print(step)})" if step is not None else ")"
        return f"{self._afn('take')}({base}, {rng}, axis={axis})"

    def _slice_from(self, base, axis, start):
        rng = f"{self._afn('arange')}({self._print(start)}, {self._shape(base, axis)})"
        return f"{self._afn('take')}({base}, {rng}, axis={axis})"


class NumPyPrinter(_ArrayFunctionPrinterMixin, spn.NumPyPrinter):
    """NumPy code printer for TVBO symbolic expressions.

    Extends SymPy's `NumPyPrinter` with the array-function vocabulary from
    `ARRAY_FUNCTION_MAPPINGS["numpy"]` and the backend-abstracted array
    primitives supplied by `_ArrayFunctionPrinterMixin`. Known functions and
    constants are module-qualified with `module`, and `erf`/`erfc` are routed to
    `scipy.special`.

    Args:
        settings: Printer settings forwarded to the SymPy base printer.
        module: Module prefix used to qualify NumPy names (e.g. `np`).
    """

    def __init__(self, settings=None, module="np"):
        self._module = module
        m = module + "."
        self._kf = {k: m + v for k, v in spn._known_functions_numpy.items()}
        self._kc = {k: m + v for k, v in spn._known_constants_numpy.items()}

        self._kf.update({"erfc": "scipy.special.erfc"})
        self._kf.update({"erf": "scipy.special.erf"})
        super().__init__(settings=settings)
        # Add array function mappings
        self.known_functions.update(ARRAY_FUNCTION_MAPPINGS["numpy"])


class JaxPrinter(_ArrayFunctionPrinterMixin, spn.JaxPrinter):
    """JAX code printer for TVBO symbolic expressions.

    Extends SymPy's `JaxPrinter` with the array-function vocabulary from
    `ARRAY_FUNCTION_MAPPINGS["jax"]` and the mixin's array primitives, routing
    `erf`/`erfc` to `jsp.special`. When broadcasting inference is enabled it
    analyzes the index usage of indexed subexpressions to insert explicit axes so
    that `jnp` operations broadcast correctly.

    Args:
        settings: Printer settings forwarded to the SymPy base printer.
        module: Module prefix used to qualify JAX names (e.g. `jnp`).
    """

    def __init__(self, settings=None, module="jnp"):
        self._module = module
        m = module + "."
        self._kf = {k: m + v for k, v in spn._known_functions_numpy.items()}
        self._kc = {k: m + v for k, v in spn._known_constants_numpy.items()}

        self._kf.update({"erfc": "jsp.special.erfc"})
        self._kf.update({"erf": "jsp.special.erf"})
        super().__init__(settings=settings)
        # Add array function mappings
        self.known_functions.update(ARRAY_FUNCTION_MAPPINGS["jax"])
        # Context for broadcasting inference
        self._index_context = None  # Set by top-level print to enable broadcasting

    def _analyze_indices(self, expr):
        """Analyze all indexed expressions to build index context.

        Returns a dict mapping index symbols to their position (axis),
        the maximum dimensionality found, and Sum reduction info.

        Example: For expr containing a[i,j], b[i,j], rmse[i]:
        - index_positions = {i: 0, j: 1}
        - max_dims = 2

        For Sum(f(a[i,j]), (j, 0, m-1)) - the result has only index i.
        """
        from sympy import preorder_traversal, Indexed, Sum

        index_positions = {}  # {index_symbol: axis_position}
        max_dims = 0
        sum_reduced_indices = set()  # Indices that are reduced by Sum

        for sub_expr in preorder_traversal(expr):
            if isinstance(sub_expr, Sum):
                # Track which indices are being summed over
                for limit in sub_expr.limits:
                    sum_reduced_indices.add(limit[0])
            elif isinstance(sub_expr, Indexed):
                indices = sub_expr.indices
                # Only count indices that are NOT reduced by a Sum
                [idx for idx in indices if idx not in sum_reduced_indices]
                max_dims = max(max_dims, len(indices))
                for pos, idx in enumerate(indices):
                    if idx not in index_positions:
                        index_positions[idx] = pos

        return {"positions": index_positions, "max_dims": max_dims, "reduced": sum_reduced_indices}

    def _print_with_broadcasting(self, expr):
        """Print expression with automatic broadcasting inference.

        Analyzes index usage across the entire expression and generates
        appropriate broadcasting (e.g., [:, None]) for lower-dimensional terms.
        """
        # Analyze the full expression to understand index context
        self._index_context = self._analyze_indices(expr)
        try:
            return self._print(expr)
        finally:
            self._index_context = None

    def _print_Indexed(self, expr):
        """Print indexed expression with automatic broadcasting.

        When index context is set:
        - a[i,j] in a 2D context -> a (no change needed)
        - rmse[i] in a 2D context -> rmse[:, None] (broadcast over missing j dimension)

        The key insight: if rmse[i] appears alongside a[i,j], then:
        - i is at axis 0 for both
        - j is at axis 1, but rmse doesn't have it
        - So rmse needs [:, None] to broadcast correctly
        """

        base_name = str(expr.base)
        indices = expr.indices

        # If no index context, just return the base name (default behavior)
        if self._index_context is None:
            return base_name

        max_dims = self._index_context["max_dims"]
        index_positions = self._index_context["positions"]

        # If this indexed expr has the same dimensionality as max, no broadcasting needed
        if len(indices) >= max_dims:
            return base_name

        # Need to add broadcasting dimensions
        # Figure out which axes this indexed expr covers
        covered_axes = set()
        for idx in indices:
            if idx in index_positions:
                covered_axes.add(index_positions[idx])

        # Build slice notation: [:, None, :, None, ...]
        # where : is for axes we have, None is for axes we're missing
        slices = []
        for axis in range(max_dims):
            if axis in covered_axes:
                slices.append(":")
            else:
                slices.append("None")

        return f"{base_name}[{', '.join(slices)}]"

    def _print_Sum(self, expr):
        """Convert SymPy Sum to jnp.sum for array operations.

        Handles patterns with single, multi-index, or nested sums:

        Single index (full reduction):
        - Sum(x[i], (i, 0, n-1)) -> jnp.sum(x)
        - Sum(x[i]*y[i], (i, 0, n-1)) -> jnp.sum(x*y)

        Multi-index (partial reduction - row/column-wise):
        - Sum(a[i,j], (j, 0, m-1)) -> jnp.sum(a, axis=1)  # sum over j (cols), keep i (rows)
        - Sum((a[i,j] - b[i,j])**2, (j, 0, m-1)) -> jnp.sum((a - b)**2, axis=1)

        Nested sums (full reduction over multiple indices):
        - Sum(a[i,j], (i, 0, n-1), (j, 0, m-1)) -> jnp.sum(a)  # full reduction

        The dummy index position determines the axis: for a[i,j], i=axis0, j=axis1.
        Summing over j means axis=1. The remaining index i yields the output shape.
        """
        from sympy import Indexed, preorder_traversal

        func = expr.function  # The expression being summed
        limits = expr.limits  # ((i, lower, upper), (j, lower, upper), ...)

        if not limits:
            # No limits - just print the function
            return f"{self._module}.sum({self._print(func)})"

        # Collect ALL dummy variables from all limits
        dummies = [lim[0] for lim in limits]

        # Find all indexed expressions and determine axes from index positions
        axes = []  # List of axes to reduce over
        max_indices = 0
        remaining_indices = set()  # Indices that remain after reduction

        for sub_expr in preorder_traversal(func):
            if isinstance(sub_expr, Indexed):
                indices = sub_expr.indices
                if len(indices) > max_indices:
                    max_indices = len(indices)
                # Find positions of all dummies in this indexed expression
                for idx, index_sym in enumerate(indices):
                    if index_sym in dummies:
                        if idx not in axes:
                            axes.append(idx)
                    else:
                        remaining_indices.add(index_sym)

        # Replace indexed expressions: remove the summed indices
        result = func
        for sub_expr in list(preorder_traversal(func)):
            if isinstance(sub_expr, Indexed):
                has_dummy = any(d in sub_expr.indices for d in dummies)
                if has_dummy:
                    # Replace indexed expr with its base array
                    result = result.subs(sub_expr, sub_expr.base)

        # Generate code with appropriate axis specification
        n_axes_to_reduce = len(axes)
        if n_axes_to_reduce == max_indices or n_axes_to_reduce == 0:
            # Full reduction: summing over ALL indices -> scalar
            return f"{self._module}.sum({self._print(result)})"
        elif n_axes_to_reduce == 1:
            # Partial reduction over single axis
            axis = axes[0]
            sum_code = f"{self._module}.sum({self._print(result)}, axis={axis})"

            # Check if we need to add broadcasting dimensions
            if self._index_context is not None:
                ctx_max_dims = self._index_context["max_dims"]
                ctx_positions = self._index_context["positions"]

                # The Sum result has (max_indices - 1) dimensions
                result_dims = max_indices - 1

                if result_dims < ctx_max_dims:
                    # Need to add broadcasting dimensions
                    # Figure out which axes the remaining indices cover
                    covered_axes = set()
                    for idx in remaining_indices:
                        if idx in ctx_positions:
                            covered_axes.add(ctx_positions[idx])

                    # Build slice notation
                    slices = []
                    for ax in range(ctx_max_dims):
                        if ax in covered_axes:
                            slices.append(":")
                        else:
                            slices.append("None")

                    sum_code = f"({sum_code})[{', '.join(slices)}]"

            return sum_code
        else:
            # Multiple axes but not all: reduce over multiple specific axes
            # Sort axes in descending order to reduce from back to front
            axes_tuple = tuple(sorted(axes))
            return f"{self._module}.sum({self._print(result)}, axis={axes_tuple})"


class JuliaPrinter(_ArrayFunctionPrinterMixin, spj.JuliaCodePrinter):
    """Julia code printer for TVBO symbolic expressions.

    Extends SymPy's `JuliaCodePrinter` with the `ARRAY_FUNCTION_MAPPINGS["julia"]`
    vocabulary and Julia-specific overrides of the mixin's array primitives, which
    use 1-based, `end`-relative indexing. Runs non-strict so unknown constructs
    print partially rather than raising, maps the legacy `atan2` name onto Julia's
    two-argument `atan`, and routes domain-restricted powers inside `Piecewise`
    branches through NaNMath.

    Args:
        settings: Printer settings forwarded to the SymPy base printer; `strict`
            defaults to `False`.
    """

    # Julia array functions are bare names (resolved via known_functions), not module-qualified.
    _module = ""

    def __init__(self, settings=None):
        settings = settings or {}
        # Be tolerant: allow partial printing instead of raising for unknown constructs.
        settings.setdefault("strict", False)
        super().__init__(settings=settings)
        # Add array function mappings
        self.known_functions.update(ARRAY_FUNCTION_MAPPINGS["julia"])
        # SymPy's Julia printer still emits the pre-0.7 name ``atan2``; modern Julia
        # spells the two-argument arctangent ``atan(y, x)`` (same arg order/semantics).
        self.known_functions["atan2"] = "atan"
        # Set while printing Piecewise branch bodies so domain-restricted powers are
        # routed through NaNMath (see _print_Pow / _print_Piecewise).
        self._in_piecewise = False

    # Route only the ops with a clean Julia form (slice/stride family + shape) through the
    # shared handlers — which then call the Julia-overridden primitives below. Everything else
    # (concatenate->vcat, transpose, reductions) defers to Julia's name-mappings, so existing
    # Julia output never regresses. NOTE: bypass the mixin's catch-all _print_Function (which
    # would route *every* registered op) by dispatching to the SymPy base directly.
    def _print_Function(self, expr):
        name = expr.func.__name__
        if name in _JULIA_HANDLED_OPS:
            return _ARRAY_FUNCTION_PRINTERS[name](self, expr)
        return spj.JuliaCodePrinter._print_Function(self, expr)

    def _print_Piecewise(self, expr):
        """Print a Piecewise, flagging its branch bodies as domain-unsafe.

        numpy/JAX evaluate every ``where`` branch and let out-of-domain ops yield
        NaN (harmlessly discarded by the select). Julia's ``ifelse`` is also eager,
        but its ``sqrt``/``^`` *throw* ``DomainError`` on a negative argument instead
        of returning NaN — so a dead branch (e.g. ``sqrt`` of a momentarily-negative
        discriminant) aborts the whole solve. Flag the bodies so ``_print_Pow`` routes
        domain-restricted powers through NaNMath and restores the numpy contract.
        """
        prev = self._in_piecewise
        self._in_piecewise = True
        try:
            return print_Piecewise(self, expr)
        finally:
            self._in_piecewise = prev

    def _print_Pow(self, expr):
        """Route domain-restricted powers inside Piecewise branches through NaNMath.

        Only fractional exponents are affected (``sqrt`` = exp 1/2, ``x^(1/3)`` …),
        and only within a Piecewise body; integer powers and every non-Piecewise
        expression keep SymPy's default Julia rendering, so non-branching models are
        byte-for-byte unchanged. NaNMath results equal ``Base`` results for in-domain
        inputs, so this never alters valid numerics.
        """
        from sympy.core.numbers import equal_valued

        if self._in_piecewise:
            exp = expr.exp
            if equal_valued(exp, 0.5):
                return "NaNMath.sqrt(%s)" % self._print(expr.base)
            if expr.is_commutative and equal_valued(exp, -0.5):
                sym = "/" if expr.base.is_number else "./"
                return "1 %s NaNMath.sqrt(%s)" % (sym, self._print(expr.base))
            if getattr(exp, "is_number", False) and not exp.is_integer:
                return "NaNMath.pow(%s, %s)" % (self._print(expr.base), self._print(exp))
        return spj.JuliaCodePrinter._print_Pow(self, expr)

    # SymPy's JuliaCodePrinter does not implement IndexedBase by default; our templates
    # occasionally introduce placeholder IndexedBase symbols (e.g. x_i, x_j) for clarity.
    # For code-generation these act like ordinary scalar symbols, so we just emit the name.
    def _print_IndexedBase(self, expr):  # noqa: D401
        return str(expr)

    # If an actual indexed object (e.g. A[i]) appears, convert to Julia's 1-based indexing.
    # We assume symbolic indices start at 0 if produced by Python-centric logic; without
    # concrete numeric indices we cannot safely +1 them, so leave symbolic indices unchanged.
    def _print_Indexed(self, expr):
        try:
            base = self._print(expr.base)
            inds = [self._print(i) for i in expr.indices]
            return f"{base}[{', '.join(inds)}]"
        except Exception:
            return str(expr)

    # --- Julia primitive overrides: 1-based, ``end``-relative, column-major indexing ---
    # ``Piecewise`` reuses the shared ``print_Piecewise`` via this ``_where3`` override
    # (numpy/jax -> ``<mod>.where``; Julia -> ``ifelse``), so no dedicated Julia Piecewise.
    def _where3(self, cond, a, b):
        return f"ifelse({cond}, {a}, {b})"

    def _shape(self, base, axis):
        return f"size({base}, {axis + 1})"

    def _mat_dot(self, a, b):
        # Multi-mode contraction ``numpy.dot(X, M)``: X is a length-n_modes vector and
        # M a Vector{Vector} of rows, so numpy's result[j] = sum_i X[i]*M[i][j].
        # ``reduce(hcat, M)`` stacks the rows as columns (== Mᵀ), so
        # ``reduce(hcat, M) * X`` reproduces that contraction as a mode vector.
        return f"(reduce(hcat, {b}) * {a})"

    def _reduce_axis(self, fn, base, axis, keepdims=False):
        # Mode-axis reduction (``mode_sum``) over a per-node mode vector reduces
        # the whole vector to a scalar that broadcasts back through ``.+``/``.-``.
        jl_fn = {"sum": "sum", "mean": "Statistics.mean"}.get(fn, fn)
        return f"{jl_fn}({base})"

    def _render_index(self, base, specs):
        parts = []
        for s in specs:
            if s is None:
                parts.append(":")
                continue
            start, stop, step = s
            lo = "1" if start is None else f"{self._print(start)} + 1"
            hi = "end" if stop is None else self._print(stop)
            parts.append(f"{lo}:{hi}" if step is None else f"{lo}:{self._print(step)}:{hi}")
        return f"{self._print(base)}[{', '.join(parts)}]"

    def _slice_axis(self, base, axis, start, stop, step=None):
        lo = f"{self._print(start)} + 1"
        hi = self._print(stop)
        rng = f"{lo}:{hi}" if step is None else f"{lo}:{self._print(step)}:{hi}"
        return f"selectdim({base}, {axis + 1}, {rng})"

    def _slice_from(self, base, axis, start):
        return f"selectdim({base}, {axis + 1}, {self._print(start)} + 1:{self._shape(base, axis)})"

    def _print_Add(self, expr, order=None):
        """Element-wise addition/subtraction (``.+`` / ``.-``).

        Unlike ``*``/``/``/``^`` (which the base Julia printer already emits as
        ``.*``/``./``/``.^``), scalar+array ``+``/``-`` is NOT auto-broadcast in
        Julia — ``[1,2] + 1`` raises ``MethodError``. Array-valued models
        (mode-coupling, quadrature vectors) mix scalar and vector terms, so we
        emit the dotted forms, which work uniformly for scalars and arrays.
        Mirrors the base printer's term ordering and sign handling.
        """
        from sympy.printing.precedence import precedence

        terms = self._as_ordered_terms(expr, order=order)
        prec = precedence(expr)
        parts = []
        for term in terms:
            t = self._print(term)
            if t.startswith("-") and not term.is_Add:
                sign = ".-"
                t = t[1:]
            else:
                sign = ".+"
            if precedence(term) < prec or term.is_Add:
                parts.extend([sign, "(%s)" % t])
            else:
                parts.extend([sign, t])
        sign = parts.pop(0)
        if sign == ".+":
            sign = ""
        elif sign == ".-":
            sign = "-"  # leading unary negation broadcasts fine in Julia
        return sign + " ".join(parts)


class MTKPrinter(JuliaPrinter):
    """Printer for ModelingToolkit.jl @mtkmodel equations.

    MTK equations are scalar symbolic, so we use plain ``+``, ``-``, ``*``,
    ``/``, ``^`` instead of Julia's element-wise ``.+``, ``.-``, ``.*``, ``./``,
    ``.^``.
    """

    def _print_Add(self, expr, order=None):
        # Scalar-symbolic: plain +/- (base CodePrinter behaviour, bypassing
        # JuliaPrinter's element-wise .+/.- override).
        return spj.JuliaCodePrinter._print_Add(self, expr, order=order)

    def _print_Pow(self, expr):
        # MTK builds Symbolics.jl expressions, which can't trace NaNMath.* calls;
        # keep the plain symbolic ``^`` (bypassing JuliaPrinter's NaNMath routing).
        return spj.JuliaCodePrinter._print_Pow(self, expr)

    def _print_Mul(self, expr):
        from sympy import S, Mul, Pow, Rational
        from sympy.printing.precedence import precedence

        if expr.is_number and expr.is_imaginary and expr.as_coeff_Mul()[0].is_integer:
            return "%sim" % self._print(-S.ImaginaryUnit * expr)

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            from sympy.core.mul import _keep_coeff

            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # numerator
        b = []  # denominator

        pow_paren = []
        if self.order not in ("old", "none"):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)

        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity and item.p == 1:
                b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]
        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # Always scalar: use * and / (never .* or ./)
        def multjoin(a_str):
            return " * ".join(a_str)

        if not b:
            return sign + multjoin(a_str)
        elif len(b) == 1:
            return "%s / %s" % (sign + multjoin(a_str), b_str[0])
        else:
            return "%s / (%s)" % (sign + multjoin(a_str), multjoin(b_str))

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued
        from sympy.printing.precedence import precedence

        PREC = precedence(expr)
        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)
        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                return "1 / sqrt(%s)" % self._print(expr.base)
            if equal_valued(expr.exp, -1):
                return "1 / %s" % self.parenthesize(expr.base, PREC)
        # Always scalar: use ^ (never .^)
        return "%s ^ %s" % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))


class FortranPrinter(spf.FCodePrinter):
    """Fortran code printer for TVBO symbolic expressions.

    Extends SymPy's `FCodePrinter` with free-form source, the Fortran 2003
    standard, and array contraction disabled. Symbolic constants (`pi`, `E`, …)
    are emitted as plain double-precision literals instead of `parameter`
    declarations, which would be invalid in an expression context.

    Args:
        settings: Printer settings forwarded to the SymPy base printer;
            `source_format`, `standard`, and `contract` are defaulted.
    """

    def __init__(self, settings=None):
        settings = settings or {}
        settings.setdefault("source_format", "free")
        settings.setdefault("standard", 2003)
        settings.setdefault("contract", False)
        super().__init__(settings=settings)

    # SymPy's FCodePrinter inlines symbolic constants like ``pi`` and ``E``
    # by emitting a ``parameter (pi = ...)`` declaration, which is invalid
    # inside an expression context (e.g. ``F(1) = parameter (pi=...) pi*r``).
    # Render them as plain double-precision literals instead.
    def _print_NumberSymbol(self, expr):
        return self._settings.get("precision_str", "%.17g") % float(expr) + "d0"

    _print_Catalan = _print_NumberSymbol
    _print_EulerGamma = _print_NumberSymbol
    _print_Exp1 = _print_NumberSymbol
    _print_GoldenRatio = _print_NumberSymbol
    _print_Pi = _print_NumberSymbol


class LEMSPrinter(StrPrinter):
    """Printer for LEMS (Low Entropy Model Specification) math expressions.

    Key differences from plain StrPrinter:
    - Powers use ``^`` instead of ``**``
    - Natural log is ``log`` (both SymPy and LEMS ``log`` are natural log)
    - ``abs`` instead of ``Abs``
    - ``sign(x)`` → ``(H(x) - H(-1*x))`` (Heaviside decomposition)
    - ``Mod(x, y)`` → ``(x + y*ceil(-(x/y)))``
    - Relational operators use LEMS dot-notation: ``.gt.``, ``.geq.``, ``.lt.``,
      ``.leq.``, ``.eq.``, ``.neq.``
    - Boolean operators: ``.and.``, ``.or.``, ``.not.``
    - ``Piecewise`` rendered via Heaviside trick (``H(cond)*val)``

    Parameters
    ----------
    settings : dict, optional
        Printer settings.  Recognised key:

        ``parameters`` : list of str
            Model symbol names.  When a SymPy ``Function`` whose name matches
            a parameter is encountered, it is printed as implicit multiplication
            (``gamma*x``) instead of a function call (``gamma(x)``).  This
            defends against symbols that were parsed without proper
            ``parameters=`` overrides.
    """

    # SymPy function name → LEMS function name
    _lems_functions = {
        "Abs": "abs",
        "ceiling": "ceil",
        "Heaviside": "H",
    }

    def __init__(self, settings=None):
        settings = dict(settings or {})
        params = settings.pop("parameters", [])
        super().__init__(settings)
        self._model_params = set(params)

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued
        from sympy.printing.precedence import precedence

        PREC = precedence(expr)
        if equal_valued(expr.exp, 0.5):
            return f"sqrt({self._print(expr.base)})"
        if equal_valued(expr.exp, -0.5):
            return f"1/sqrt({self._print(expr.base)})"
        if equal_valued(expr.exp, -1):
            return f"1/{self.parenthesize(expr.base, PREC)}"
        return f"{self.parenthesize(expr.base, PREC)}^{self.parenthesize(expr.exp, PREC)}"

    def _print_log(self, expr):
        # SymPy log() is natural log; LEMS log() is also natural log
        if len(expr.args) == 1:
            return f"log({self._print(expr.args[0])})"
        # log(x, base) — change of base via natural log
        return f"(log({self._print(expr.args[0])}) / log({self._print(expr.args[1])}))"

    def _print_sign(self, expr):
        # sign(x) = H(x) - H(-x) using LEMS built-in Heaviside
        arg = self._print(expr.args[0])
        return f"(H({arg}) - H(-1*{arg}))"

    def _print_Mod(self, expr):
        # Mod(x, y) = x + y*ceil(-(x/y))  [floor = -ceil(-x)]
        x = self._print(expr.args[0])
        y = self._print(expr.args[1])
        return f"({x} + {y}*ceil(-({x})/({y})))"

    def _print_Function(self, expr):
        name = expr.func.__name__
        # Safety net: if a model parameter was mis-parsed as a function call
        # (e.g. gamma(x) instead of gamma*x), treat as multiplication.
        if self._model_params and name in self._model_params:
            args = "*".join(self._print(a) for a in expr.args)
            return f"{name}*{args}" if args else name
        lems_name = self._lems_functions.get(name, name)
        args = ", ".join(self._print(a) for a in expr.args)
        return f"{lems_name}({args})"

    # ── Relational operators ───────────────────────────────────────────

    def _print_StrictGreaterThan(self, expr):
        return f"{self._print(expr.lhs)} .gt. {self._print(expr.rhs)}"

    def _print_GreaterThan(self, expr):
        return f"{self._print(expr.lhs)} .geq. {self._print(expr.rhs)}"

    def _print_StrictLessThan(self, expr):
        return f"{self._print(expr.lhs)} .lt. {self._print(expr.rhs)}"

    def _print_LessThan(self, expr):
        return f"{self._print(expr.lhs)} .leq. {self._print(expr.rhs)}"

    def _print_Equality(self, expr):
        return f"{self._print(expr.lhs)} .eq. {self._print(expr.rhs)}"

    def _print_Unequality(self, expr):
        return f"{self._print(expr.lhs)} .neq. {self._print(expr.rhs)}"

    # ── Boolean operators ──────────────────────────────────────────────

    def _print_And(self, expr):
        return " .and. ".join(self.parenthesize(a, 0) for a in expr.args)

    def _print_Or(self, expr):
        return " .or. ".join(self.parenthesize(a, 0) for a in expr.args)

    def _print_Not(self, expr):
        return f".not. {self.parenthesize(expr.args[0], 0)}"

    # ── Piecewise → Heaviside product ─────────────────────────────────

    def _print_Piecewise(self, expr):
        # LEMS has no ternary; use H(cond)*val summation as fallback.
        # True branch last (otherwise case).
        from sympy import S as sympy_S

        terms = []
        for val, cond in expr.args:
            if cond == sympy_S.true:
                terms.append(self._print(val))
            else:
                terms.append(f"H({self._print(cond)}) * {self._print(val)}")
        return " + ".join(terms)


class PythonCodePrinter(_PythonCodePrinter):
    """Plain-Python code printer for TVBO symbolic expressions.

    Extends SymPy's `PythonCodePrinter` to run non-strict (partial printing of
    unknown constructs) and adds `ceil`, `sign`, and the
    `ARRAY_FUNCTION_MAPPINGS["python"]` vocabulary. `Piecewise` is rendered as
    nested conditional expressions and `sign(x)` as an inline comparison, so the
    output depends only on `math` and the standard library.

    Args:
        settings: Printer settings forwarded to the SymPy base printer; `strict`
            defaults to `False`.
    """

    def __init__(self, settings=None):
        settings = settings or {}
        # Be lenient: allow partial printing for unknown constructs
        settings.setdefault("strict", False)
        super().__init__(settings=settings)

        # Add additional math functions not in the base printer
        self.known_functions.update(
            {
                "ceil": "math.ceil",
                "sign": "math.copysign(1, {0})",  # Python's math doesn't have sign directly
            }
        )
        # Add array function mappings
        self.known_functions.update(ARRAY_FUNCTION_MAPPINGS["python"])

    def _print_Piecewise(self, expr):
        # Basic nested conditional for plain Python
        args = expr.args
        default = self._print(args[-1][0])
        result = default

        for value, condition in reversed(args[:-1]):
            condition_str = self._print(condition)
            value_str = self._print(value)
            result = f"({value_str} if {condition_str} else {result})"

        return result

    def _print_sign(self, expr):
        # sign(x) -> (1 if x > 0 else (-1 if x < 0 else 0))
        arg = self._print(expr.args[0])
        return f"(1 if {arg} > 0 else (-1 if {arg} < 0 else 0))"


def get_printer(format, parameters=None, order=None):
    """Return a code printer instance for the given target format.

    Args:
        format: Target output format. One of `numpy`, `jax`, `julia`, `mtk`,
            `fortran`, `python`, `lems`, or `sympy`/`symbolic`/`pyrates`.
        parameters: Parameter names passed to `LEMSPrinter`; used only for the
            `lems` format.
        order: Term ordering passed to the printer; `none` preserves source term
            order. When omitted the printer's default ordering is used.

    Returns:
        A configured printer instance for `format`.

    Raises:
        ValueError: If `format` is not a supported output format.
    """
    # order='none' preserves source term order; default keeps prior behaviour.
    extra = {} if order is None else {"order": order}

    if format == "numpy":
        return NumPyPrinter(settings=extra) if extra else NumPyPrinter()
    elif format == "jax":
        return JaxPrinter(settings=extra) if extra else JaxPrinter()
    elif format == "julia":
        return JuliaPrinter(settings=extra) if extra else JuliaPrinter()
    elif format == "mtk":
        return MTKPrinter(settings=extra) if extra else MTKPrinter()
    elif format == "fortran":
        return FortranPrinter(settings=extra) if extra else FortranPrinter()
    elif format == "python":
        return PythonCodePrinter(settings=extra) if extra else PythonCodePrinter()
    elif format == "lems":
        return LEMSPrinter(settings={"parameters": parameters or [], **extra})
    elif format in ["sympy", "symbolic", "pyrates"]:
        return StrPrinter(settings=extra) if extra else StrPrinter()
    else:
        raise ValueError(f"Unsupported format: {format}")


def render_expression(
    expression,
    format="jax",
    user_functions={},
    parameters=None,
    infer_broadcasting=False,
    preserve_order=False,
):
    """Render a SymPy expression or string to target format code.

    Uses parse_eq for proper handling of indexed expressions and Sum/Product.

    Parameters
    ----------
    expression : str or sympy.Expr
        The expression to render.
    format : str
        Target format ('jax', 'numpy', 'julia', 'python', etc.)
    user_functions : dict
        Custom function name mappings for the printer. These are also passed
        to parse_eq so they're recognized as functions (not implicit multiplication).
    parameters : list of str, optional
        Parameter names to define as Symbols. These OVERRIDE SymPy built-in
        functions (e.g., 'gamma' becomes Symbol('gamma'), not the gamma function).
    infer_broadcasting : bool
        If True, analyze indexed expressions and automatically add broadcasting
        dimensions (e.g., rmse[i] -> rmse[:, None] when used with a[i,j]).
        This enables mathematically correct notation to generate correct array code.
    preserve_order : bool
        If True, keep the source term order (no SymPy Add/Mul canonicalization)
        so generated code matches reference code operation-for-operation.
    """
    if isinstance(expression, str):
        # Pass user_functions AND the array-op vocabulary to parse_eq so they're
        # recognized as functions (else implicit multiplication splits e.g.
        # pad(x) into pad*x).
        func_names = list(user_functions.keys()) if user_functions else []
        func_names += list(ARRAY_FUNCTION_MAPPINGS.get(format, {}).keys())
        # preserve_order: parse unevaluated + print order='none' so SymPy keeps
        # the authored term order (float +/* are non-associative).
        _po = {"evaluate": False} if preserve_order else {}
        expression = parse_eq(expression, parameters=parameters, functions=func_names, **_po)

    printer = get_printer(format, parameters=parameters, order="none" if preserve_order else None)
    # User functions extend built-in mappings (don't override if already mapped)
    if user_functions:
        for name, target in user_functions.items():
            if name not in printer.known_functions:
                printer.known_functions[name] = target

    # Use broadcasting-aware printing if requested and printer supports it
    if infer_broadcasting and hasattr(printer, "_print_with_broadcasting"):
        return printer._print_with_broadcasting(expression)

    return printer.doprint(expression)


def render_equation(
    equation: Equation,
    format="jax",
    local_dict={},
    user_functions={},
    replace=None,
    remove=None,
    inline_funcs=None,
    preserve_order=False,
    **kwargs,
):
    """
    Render an equation to a target format.

    Parameters
    ----------
    equation : Equation
        The equation to render.
    format : str
        Target format: 'jax', 'numpy', 'python', 'julia', 'fortran', 'latex'.
    local_dict : dict
        Dictionary of local symbols/functions for parsing.
    user_functions : dict
        Custom function mappings for the printer.
    replace : dict
        Symbol replacements {old_name: new_name}.
    remove : list
        Symbols to replace with zero.
    inline_funcs : dict, optional
        Dictionary mapping function name -> (arg_names, body_expr) for inlining
        custom functions. The body_expr should be a sympy expression.
        Example: {'Sigm': (['v'], 2*e0/(1 + exp(r*(v0 - v))))}
    preserve_order : bool
        If True, keep the source term order (no SymPy canonicalization).
    **kwargs
        Additional arguments passed to parse_eq.

    Returns
    -------
    str
        The rendered equation string.
    """
    # latex prints the raw parsed expression (before replace/remove), so it keeps
    # its own short path; every other format shares the route below.
    if format == "latex":
        if preserve_order:  # keep authored term order (see render_expression)
            kwargs.setdefault("evaluate", False)
        return latex(parse_eq(equation, local_dict=local_dict, **kwargs))

    expr, uf = _prepare_expr(
        equation, local_dict, user_functions, replace, remove, inline_funcs, preserve_order, kwargs
    )
    return _printer_for(format, uf, preserve_order).doprint(expr)


def _prepare_expr(
    equation, local_dict, user_functions, replace, remove, inline_funcs, preserve_order, kwargs
):
    """Parse ``equation`` and resolve its model-function set for printing.

    Shared by :func:`render_equation` and :func:`render_equation_cse` so both take
    one identical parse -> replace -> remove -> inline route (no drift). Returns
    ``(expr, uf)`` where ``uf`` maps model-defined function names for the printer.
    """
    if preserve_order:  # keep authored term order (see render_expression)
        kwargs.setdefault("evaluate", False)
    expr = parse_eq(equation, local_dict=local_dict, **kwargs)

    if replace:
        expr = expr.xreplace({Symbol(k): Symbol(v) for k, v in replace.items()})
    if remove:
        expr = expr.xreplace({Symbol(k): S.Zero for k in remove})
    if inline_funcs:
        expr = inline_functions(expr, inline_funcs)

    # Model-defined functions must be registered so the printer emits ``f(x)``;
    # printers already know ARRAY_FUNCTION_MAPPINGS.
    uf = dict(user_functions) if isinstance(user_functions, dict) else {}
    if isinstance(local_dict, dict) and local_dict:
        for name, obj in local_dict.items():
            if getattr(obj, "is_Function", False) and name not in uf:
                uf[str(name)] = str(name)
    return expr, uf


def _printer_for(format, uf, preserve_order):
    """Build a printer for ``format`` with model-defined functions registered."""
    printer = get_printer(format, order="none" if preserve_order else None)
    if uf:
        try:
            printer.known_functions.update(uf)
        except AttributeError:
            pass  # Some printers don't have known_functions
    return printer


def render_equation_cse(
    equation: Equation,
    format="numpy",
    local_dict={},
    user_functions={},
    replace=None,
    remove=None,
    inline_funcs=None,
    preserve_order=False,
    symbol_prefix="_cse",
    **kwargs,
):
    """Render ``equation`` as ``(setup, final)`` with common subexpressions hoisted.

    ``setup`` is a list of ``(name, expr_str)`` assignments (dependency order) and
    ``final`` is the return-expression string. Repeated subexpressions — notably
    repeated model-function calls such as ``muV(fe, fi, ...)`` — are computed once
    via :func:`sympy.cse`. Interpreted backends (numpy / TVB) would otherwise
    re-evaluate every occurrence; the jax path keeps the flat ``render_equation``
    form and leans on XLA's JIT-time CSE. ``setup`` is empty when nothing is shared.
    """
    from sympy import cse, numbered_symbols

    expr, uf = _prepare_expr(
        equation, local_dict, user_functions, replace, remove, inline_funcs, preserve_order, kwargs
    )
    printer = _printer_for(format, uf, preserve_order)

    replacements, reduced = cse(expr, symbols=numbered_symbols(symbol_prefix))
    setup = [(printer.doprint(sym), printer.doprint(sub)) for sym, sub in replacements]
    final = printer.doprint(reduced[0] if reduced else expr)
    return setup, final
