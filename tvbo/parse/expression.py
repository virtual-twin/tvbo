from sympy import parse_expr, Symbol, Function, IndexedBase, Sum, Product, sqrt
from sympy.parsing.sympy_parser import (
    standard_transformations,
    convert_xor,
    split_symbols_custom,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)

# Implicit multiplication WITHOUT split_symbols: multi-letter identifiers
# (e.g. "perturbation") stay as a single Symbol instead of being expanded into
# the product of their letters. Digit-prefix splitting like "2x" -> "2*x"
# is unaffected because that lives in `implicit_multiplication`.
_no_split_symbols = split_symbols_custom(lambda _name: False)
_implicit_mul_app_no_split = (
    _no_split_symbols,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)
from sympy.parsing.latex import parse_latex

from tvbo.datamodel.schema import Equation


# =============================================================================
# Custom SymPy Classes for Mathematical Aggregation
# =============================================================================


class Mean(Function):
    """Mean over indexed expression: Mean(f(x[i]), (i, 0, N-1)).

    Mathematical notation for averaging over a dimension. Translates to
    jnp.mean(jax.vmap(...)) or jnp.mean(...) depending on the inner function.

    Example:
        Mean(1 - correlation(x[i], y[i]), (i, 0, N-1))
        -> jnp.mean(jax.vmap(lambda x, y: 1 - correlation(x, y))(x, y))
    """

    @classmethod
    def eval(cls, *args):
        # Don't auto-evaluate; let the printer handle code generation
        return None


# =============================================================================
# Symbolic Summation Support
# =============================================================================
# SymPy's Sum requires explicit index variables: Sum(f(i), (i, a, b))
#
# For proper mathematical notation, use:
#   Sum(x[i]*y[i], (i, 0, n-1))  ->  translates to jnp.sum(x*y)
#
# Index variables are detected dynamically from Sum/Product limits.
# The code printers in tvbo.codegen.code handle the translation.


# =============================================================================
# Array Function Definitions (single source of truth for parsing)
# =============================================================================
# These are array reduction/aggregation functions that SymPy doesn't have natively.
# We define them as undefined SymPy Functions so they parse correctly (preventing
# implicit multiplication like 'mean(x)' -> 'm*e*a*n*(x)').
#
# NOTE: These use lowercase names to distinguish from SymPy's symbolic Sum/Product
# which require explicit index variables. Our versions are for array reduction
# operations (like numpy's sum/mean) that reduce over all elements.
#
# For printer mappings (jnp.sum, np.mean, etc.), see tvbo.codegen.code

ARRAY_FUNCTIONS = {
    "sum": Function("sum"),
    "mean": Function("mean"),
    "std": Function("std"),
    "var": Function("var"),
    "max": Function("max"),
    "min": Function("min"),
    "abs": Function("abs"),
    "prod": Function("prod"),
    "concatenate": Function("concatenate"),
    # Array-manipulation functions that carry Python-specific semantics.
    # Custom printer methods in tvbo.codegen.code expand these to the correct
    # JAX / NumPy calls (including .reshape() and keyword args that SymPy
    # cannot represent natively).
    "window_mean": Function("window_mean"),  # window_mean(X, w) → jnp.mean(X.reshape(-1, w, *X.shape[1:]), axis=1)
    "subsample": Function("subsample"),      # subsample(X, step) → X[::step]
    "global_mean": Function("global_mean"),  # global_mean(X) → jnp.mean(X, axis=-2, keepdims=True)
    "transpose": Function("transpose"),      # transpose(X) → X.T
}


def parse_eq(
    equation: Equation,
    parameters=None,
    **kwargs,
):
    """Parse the right-hand side of an equation or a raw expression string.

    Extends parsing with the ability to pass parameters, functions, symbols, and
    arbitrary SymPy objects commonly used in nonlinear systems dynamics.

    Parameters
    ----------
    equation : Equation | str
        An Equation from tvbo's datamodel or a raw expression string. If an
        `Equation` with `latex=True` is provided, LaTeX parsing is used.
    parameters : Iterable[str] | Mapping[str, object] | None
        Names or a mapping of parameter names to SymPy objects or numbers. If an
        iterable of strings is provided, they are created as SymPy Symbols and
        injected into the parsing context. If a mapping is provided, the values
        are injected as-is (Symbols, Functions, Expressions, numbers, etc.).

    Keyword-only enhancements (optional)
    ------------------------------------
    local_dict : dict
        Additional local names to inject into the parser (merged on top of defaults).
    functions : Iterable[str] | Mapping[str, object]
        Names or mapping for functions. String names are created as undefined
        SymPy functions, e.g., Function('f'). Mapping values are used as-is.
    symbols : Iterable[str] | Mapping[str, Symbol]
        Extra symbol names or mapping for state variables, etc. String names are
        created as SymPy Symbols. Mapping values are used as-is.
    objects : Mapping[str, object]
        Arbitrary additional objects (e.g., Heaviside, MatrixSymbol, IndexedBase,
        Derivative alias, etc.) to inject into the local namespace.
    extra_transformations : Iterable[callable]
        Extra SymPy parser transformations to augment the defaults.
    transformations : Iterable[callable]
        Full control over the transformation pipeline (overrides defaults if provided).

    Returns
    -------
    sympy.Expr
        Parsed SymPy expression.
    """

    # Start with user-provided locals (sympy's parse_expr handles pi, E, etc. by default)
    local_dict = dict(kwargs.pop("local_dict", {}))

    # Add SymPy's Sum, Product, IndexedBase for proper mathematical notation
    # These allow parsing expressions like Sum(x[i], (i, 0, n-1))
    local_dict.setdefault("Sum", Sum)
    local_dict.setdefault("Product", Product)
    local_dict.setdefault("IndexedBase", IndexedBase)
    local_dict.setdefault("Mean", Mean)  # Custom Mean function for indexed averaging
    local_dict.setdefault("sqrt", sqrt)

    # Merge array functions (user-provided local_dict takes precedence)
    for name, fn in ARRAY_FUNCTIONS.items():
        if name not in local_dict:
            local_dict[name] = fn

    # Helper to coerce iterables/mappings into name -> sympy object entries
    # IMPORTANT: User-defined parameters OVERRIDE SymPy built-ins (e.g., gamma, lambda)
    def _update_from_names_or_map(container, factory):
        if not container:
            return
        # Mapping case - user values override everything
        if isinstance(container, dict):
            local_dict.update(container)
            return
        # Iterable of names/objects
        try:
            for item in container:
                if isinstance(item, str):
                    obj = factory(item)
                    local_dict[item] = obj  # Override any existing (including SymPy functions)
                else:
                    # Allow passing actual SymPy objects; try to infer name
                    name = getattr(item, "name", None)
                    if name:
                        local_dict[name] = item
        except TypeError:
            # Not iterable -> ignore
            pass

    # Parameters: create Symbols for bare string names; accept mapping of name->object/value
    _update_from_names_or_map(parameters, Symbol)

    # Functions: create undefined SymPy Functions for bare string names; accept mapping
    _update_from_names_or_map(kwargs.pop("functions", None), lambda n: Function(n))

    # Symbols/variables: explicit symbols beyond parameters
    _update_from_names_or_map(kwargs.pop("symbols", None), Symbol)

    # Arbitrary additional objects (e.g., Derivative, Heaviside, MatrixSymbol)
    objs = kwargs.pop("objects", None) or kwargs.pop("extras", None)
    if isinstance(objs, dict):
        local_dict.update(objs)

    # Determine expression string to parse
    if isinstance(equation, str):
        expression = equation
    else:
        expression = equation.rhs

    # If it's already an Expr, return it directly
    if not isinstance(expression, str):
        return expression

    # LaTeX path
    if isinstance(equation, Equation) and getattr(equation, "latex", False):
        # parse_latex doesn't accept local_dict; it returns a SymPy Expr directly
        return parse_latex(expression, backend="lark")

    import re

    # Auto-detect indexed variables (e.g., x[i], y[j]) and create IndexedBase for them
    # This allows natural mathematical notation: Sum(x[i]*y[i], (i, 0, n-1))
    # NOTE: This MUST override any Symbol definitions (including from parameters)
    # because x[i] syntax requires IndexedBase, not Symbol
    indexed_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[")
    for match in indexed_pattern.finditer(expression):
        var_name = match.group(1)
        # Override even if already defined - indexed access requires IndexedBase
        local_dict[var_name] = IndexedBase(var_name)

    # Auto-detect index variables from Sum/Product limits: Sum(..., (i, a, b))
    # Pattern matches the first element in limit tuples like (i, 0, n-1)
    limit_pattern = re.compile(r"\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*[^,]+\s*,\s*[^)]+\)")
    for match in limit_pattern.finditer(expression):
        idx_name = match.group(1)
        # Create as Symbol if not already defined
        if idx_name not in local_dict:
            local_dict[idx_name] = Symbol(idx_name)

    # Build transformations pipeline
    extra_transformations = tuple(kwargs.pop("extra_transformations", ()))
    transformations = kwargs.pop(
        "transformations",
        standard_transformations + _implicit_mul_app_no_split + (convert_xor,) + extra_transformations,
    )

    # Remaining kwargs forwarded to parse_expr (e.g., evaluate=False, global_dict=...)
    return parse_expr(
        expression,
        local_dict=local_dict,
        transformations=transformations,
        **kwargs,
    )
