import sympy.printing.julia as spj
import sympy.printing.numpy as spn
import sympy.printing.fortran as spf
from sympy.printing.pycode import PythonCodePrinter as _PythonCodePrinter
from sympy import Symbol, S, Function, preorder_traversal
from sympy import latex
from sympy.printing import StrPrinter
from tvbo.datamodel.schema import Equation
from tvbo.parse.expression import parse_eq


# =============================================================================
# Array Function Printer Mappings
# =============================================================================
# Maps the ARRAY_FUNCTIONS (defined in tvbo.parse.expression) to their target
# implementations for each output format. Printers use these via known_functions.

ARRAY_FUNCTION_MAPPINGS = {
    "jax": {
        "sum": "jnp.sum",
        "mean": "jnp.mean",
        "std": "jnp.std",
        "var": "jnp.var",
        "max": "jnp.max",
        "min": "jnp.min",
        "abs": "jnp.abs",
        "prod": "jnp.prod",
        "concatenate": "jnp.concatenate",
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
        "concatenate": "np.concatenate",
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
        "concatenate": "vcat",
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
        print("expr:", expr)
        print("args:", args)
        print("default:", default)

    # Iterate over conditions and expressions in reverse order (excluding the default)
    for value, condition in reversed(args[:-1]):
        if verbose:
            print("condition:", condition)
            print("value:", value)
        condition_str = Printer._print(condition)
        value_str = Printer._print(value)
        # Build the nested np.where
        result = f"{Printer._module}.where({condition_str}, {value_str}, {result})"

    if verbose:
        print("result:", result)
        print()
    return result


class NumPyPrinter(spn.NumPyPrinter):
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

    def _print_Piecewise(self, expr):
        return print_Piecewise(self, expr)


class JaxPrinter(spn.JaxPrinter):
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

    def _print_Piecewise(self, expr):
        return print_Piecewise(self, expr)

    def _print_Function(self, expr):
        """Handle special array functions like concatenate."""
        func_name = expr.func.__name__
        if func_name == "concatenate":
            # concatenate(a, b, axis) -> jnp.concatenate([a, b], axis=axis)
            args = list(expr.args)
            if args and args[-1].is_integer:
                axis = int(args[-1])
                arrays = args[:-1]
            else:
                axis = 0
                arrays = args
            array_strs = ", ".join(self._print(a) for a in arrays)
            return f"jnp.concatenate([{array_strs}], axis={axis})"
        if func_name == "window_mean":
            # window_mean(X, step) -> jnp.mean(X.reshape(-1, step, *X.shape[1:]), axis=1)
            X_str = self._print(expr.args[0])
            w_str = self._print(expr.args[1])
            return f"jnp.mean({X_str}.reshape(-1, {w_str}, *{X_str}.shape[1:]), axis=1)"
        if func_name == "subsample":
            # subsample(X, step) -> X[::step]
            X_str = self._print(expr.args[0])
            s_str = self._print(expr.args[1])
            return f"{X_str}[::{s_str}]"
        if func_name == "global_mean":
            # global_mean(X) -> jnp.mean(X, axis=-2, keepdims=True)
            X_str = self._print(expr.args[0])
            return f"jnp.mean({X_str}, axis=-2, keepdims=True)"
        if func_name == "transpose":
            # transpose(X) -> X.T
            X_str = self._print(expr.args[0])
            return f"{X_str}.T"
        # Fall back to parent implementation
        return super()._print_Function(expr)

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


class JuliaPrinter(spj.JuliaCodePrinter):
    def __init__(self, settings=None):
        settings = settings or {}
        # Be tolerant: allow partial printing instead of raising for unknown constructs.
        settings.setdefault("strict", False)
        super().__init__(settings=settings)
        # Add array function mappings
        self.known_functions.update(ARRAY_FUNCTION_MAPPINGS["julia"])

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

    # Provide a basic Piecewise -> nested ifelse translation if needed later; keep simple now.
    def _print_Piecewise(self, expr):
        # Fallback: replicate numpy-style nesting using ifelse(cond, val, else_expr)
        args = expr.args
        default = self._print(args[-1][0])
        out = default
        for val, cond in reversed(args[:-1]):
            cond_s = self._print(cond)
            val_s = self._print(val)
            out = f"ifelse({cond_s}, {val_s}, {out})"
        return out


class MTKPrinter(JuliaPrinter):
    """Printer for ModelingToolkit.jl @mtkmodel equations.

    MTK equations are scalar symbolic, so we use plain ``*``, ``/``, ``^``
    instead of Julia's element-wise ``.*``, ``./``, ``.^``.
    """

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
        # Pass user_functions as functions to parse_eq so they're recognized
        # This prevents implicit multiplication from breaking function names
        func_names = list(user_functions.keys()) if user_functions else None
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
    # Ensure parsing knows about symbols and undefined functions from the model scope
    if preserve_order:  # keep authored term order (see render_expression)
        kwargs.setdefault("evaluate", False)
    expr = parse_eq(equation, local_dict=local_dict, **kwargs)

    if format == "latex":
        return latex(expr)

    if replace:
        symbol_map = {Symbol(k): Symbol(v) for k, v in replace.items()}
        expr = expr.xreplace(symbol_map)

    if remove:
        expr = expr.xreplace({Symbol(k): S.Zero for k in remove})

    # Inline custom functions if provided
    if inline_funcs:
        expr = inline_functions(expr, inline_funcs)

    # Build user_functions mapping for model-defined symbolic functions
    # Printers already have ARRAY_FUNCTION_MAPPINGS built-in
    uf = dict(user_functions) if isinstance(user_functions, dict) else {}

    # Auto-detect model-defined functions from local_dict
    if isinstance(local_dict, dict) and local_dict:
        for name, obj in local_dict.items():
            if getattr(obj, "is_Function", False) and name not in uf:
                uf[str(name)] = str(name)

    printer = get_printer(format, order="none" if preserve_order else None)
    # User functions take precedence over built-in mappings
    if uf:
        try:
            printer.known_functions.update(uf)
        except AttributeError:
            pass  # Some printers don't have known_functions

    return printer.doprint(expr)
