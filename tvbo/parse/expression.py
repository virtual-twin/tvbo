"""Parse TVBO equation strings into SymPy expressions.

Provides [`parse_eq`](expression.qmd#parse_eq) for turning an `Equation` (or a raw string) into a SymPy expression, along with the custom aggregation symbols and the `ARRAY_FUNCTIONS` registry of array reduction/manipulation functions (`sum`, `mean`, `slice_axis`, `mode_dot`, …) that the code printers in `tvbo.codegen.code` lower to backend-specific calls.

`ARRAY_FUNCTIONS` is the single source of truth for parsing. Each entry is an undefined SymPy `Function` — that is what makes `mean(x)` parse as a call rather than being split by implicit multiplication into `m*e*a*n*(x)`. The names are lowercase to keep them distinct from SymPy's own symbolic `Sum` and `Product`, which need explicit index variables where these reduce over whole arrays, numpy-style. Printer mappings live in `tvbo.codegen.code`.

**Every argument is positional.** SymPy's parser forwards `f(x, axis=0)` into `Basic`'s options and raises `ValueError: Unknown options`, so a primitive can never carry a keyword. Anything that would be one — a metric, an axis, a target range, a distribution, a seed — is a schema field on the DAG step and is lowered into a positional argument, which is why the `Procedural` graph-generator DAG is typed rather than free-form.

The registry covers array manipulation with Python-specific semantics (`window_mean`, `subsample`), structural slice and shape ops so a pipeline that selects a variable of interest, trims a transient or downsamples is authored as declarative equations rather than `source_code`, general ops for per-timestep detectors and permutation-significance tests (`take`, `sum_axis`, `pearson`), the graph-construction primitives a `Procedural` GraphGenerator lowers to, and the distribution samplers.

A sampler takes its PRNG state as the **first** argument, because JAX is functionally pure and a key cannot be threaded implicitly through a rendered expression; the trailing arguments are the sample shape. Draws are *not* bit-identical across backends — numpy's PCG64 is not jax's Threefry.

Symbolic summation uses SymPy's own `Sum`, which needs explicit index variables: `Sum(x[i]*y[i], (i, 0, n-1))`. Index variables are detected from the `Sum` and `Product` limits, and the code printers handle the translation.
"""

from sympy import parse_expr, Symbol, Function, IndexedBase, Sum, Product, sqrt
from sympy.parsing.sympy_parser import (
    standard_transformations,
    convert_xor,
    split_symbols_custom,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)

# Implicit multiplication WITHOUT split_symbols: multi-letter identifiers (e.g. "perturbation") stay as a single Symbol instead of being expanded into the product of their letters. Digit-prefix splitting like "2x" -> "2*x" is unaffected because that lives in `implicit_multiplication`.
_no_split_symbols = split_symbols_custom(lambda _name: False)
_implicit_mul_app_no_split = (
    _no_split_symbols,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)
from sympy.parsing.latex import parse_latex

from tvbo.datamodel.schema import Equation


# Custom SymPy Classes for Mathematical Aggregation


class Mean(Function):
    """Mean over indexed expression: Mean(f(x[i]), (i, 0, N-1)).

    Mathematical notation for averaging over a dimension. Translates to jnp.mean(jax.vmap(...)) or jnp.mean(...) depending on the inner function.

    Example:
        Mean(1 - correlation(x[i], y[i]), (i, 0, N-1))
        -> jnp.mean(jax.vmap(lambda x, y: 1 - correlation(x, y))(x, y))
    """

    @classmethod
    def eval(cls, *args):
        """Suppress automatic simplification so the symbol survives to codegen.

        SymPy calls this classmethod when a `Mean(...)` is constructed. Returning `None` signals that no closed-form evaluation should be performed, keeping the expression as an unevaluated `Mean` node that the code printers in [`tvbo.codegen.code`](../codegen/code.qmd) translate into the backend's mean/reduction call.

        Args:
            *args: The positional arguments the `Mean` was called with (the inner
                expression and index limit tuple). Left unused.

        Returns:
            Always `None`, leaving the `Mean` application unevaluated.
        """
        return None


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
    # Python-specific semantics; the printers expand these to the right JAX / NumPy call.
    "window_mean": Function("window_mean"),  # window_mean(X, w) → jnp.mean(X.reshape(-1, w, *X.shape[1:]), axis=1)
    "subsample": Function("subsample"),  # subsample(X, step[, start]) → X[start::step]
    # Structural slice/shape ops.
    "slice_axis": Function("slice_axis"),  # slice_axis(X, axis, start, stop[, step]) → bounded slice of one axis (keeps ndim)
    "slice_from": Function(
        "slice_from"
    ),  # slice_from(X, axis, start)               → open-ended slice of one axis (to the end)
    "shape": Function("shape"),  # shape(X, axis)                           → length of X along axis
    "global_mean": Function("global_mean"),  # global_mean(X) → jnp.mean(X, axis=-2, keepdims=True)
    "transpose": Function("transpose"),  # transpose(X) → X.T
    "mode_dot": Function("mode_dot"),  # mode_dot(X, M) → X·M contracted over the mode axis ({np,jnp}.dot)
    "mode_sum": Function(
        "mode_sum"
    ),  # mode_sum(X) → sum over the mode axis, keepdims ({np,jnp}.sum(X, axis=-1, keepdims=True))
    # Linear-algebra ops (e.g. streaming co-moment reducers, projections).
    "outer": Function("outer"),  # outer(a, b) → rank-1 outer product a_i b_j
    "diag": Function("diag"),  # diag(M) → the main diagonal of M as a vector
    "zero_diagonal": Function("zero_diagonal"),  # zero_diagonal(M) → M with its main diagonal set to 0
    "matmul": Function("matmul"),  # matmul(A, B) → ordinary matrix product A @ B
    "strided_convolve": Function(
        "strided_convolve"
    ),  # strided_convolve(X, k, s) → 'valid' conv of X⊛k evaluated only at the [s::s] output indices (fuses convolve+subsample; no full FFT)
    # A 2-D gather, a single-axis reduction, and Pearson correlation.
    "take": Function("take"),  # take(x, idx) → gather x by an int index array (result has idx.shape)
    "sum_axis": Function("sum_axis"),  # sum_axis(x, axis) → reduce one axis ({np,jnp}.sum(x, axis=..))
    "pearson": Function(
        "pearson"
    ),  # pearson(x, y) → Pearson r of two FLAT/1-D operands (reduces all elements; the per-step node-collapsing corr — NOT a columnwise 2-D corr, and distinct from the loss-helper `correlation`)
    "clip": Function("clip"),  # clip(x, lo, hi) → bound x to [lo, hi] ({np,jnp}.clip); e.g. clip(cos_sim, -1, 1) before acos
    "any": Function("any"),  # any(x) → True if any element is truthy ({np,jnp}.any); e.g. any(p_div <= sig)
    "all": Function("all"),  # all(x) → True if every element is truthy ({np,jnp}.all)
    # Graph-construction primitives: the vocabulary a `Procedural` GraphGenerator's DAG lowers to.
    "grid_positions": Function(
        "grid_positions"
    ),  # grid_positions(nx, ny, x_extent, y_extent) → [nx*ny, 2] regular-lattice node coordinates, x-major
    "pairwise_distance": Function(
        "pairwise_distance"
    ),  # pairwise_distance(pos) → [n,n] euclidean distances between rows of pos
    "fill_diagonal": Function(
        "fill_diagonal"
    ),  # fill_diagonal(M, v) → M with its main diagonal set to v (v=inf suppresses self-connections)
    "gaussian_pdf": Function(
        "gaussian_pdf"
    ),  # gaussian_pdf(pos, mean, cov) → isotropic multivariate-normal density at each row of pos
    "normalize": Function("normalize"),  # normalize(M, axis) → M divided by its sum along `axis` (axis must be a literal int)
    "minmax_rescale": Function(
        "minmax_rescale"
    ),  # minmax_rescale(x, lo, hi) → x affinely rescaled from its own min/max onto [lo, hi]
    "eigvals": Function("eigvals"),  # eigvals(M) → eigenvalues of M (e.g. spectral-radius rescaling)
    # One head per distribution, mirroring the backend sampler table in dev/GenericProcedureEngine.md §2.3.
    "sample_normal": Function("sample_normal"),  # sample_normal(key, mean, std, *shape)
    "sample_uniform": Function("sample_uniform"),  # sample_uniform(key, lo, hi, *shape)
    "sample_lognormal": Function("sample_lognormal"),  # sample_lognormal(key, mu, sigma, *shape)
    "sample_beta": Function("sample_beta"),  # sample_beta(key, a, b, *shape)
    "sample_exponential": Function("sample_exponential"),  # sample_exponential(key, scale, *shape)
}


def parse_eq(
    equation: Equation,
    parameters=None,
    **kwargs,
):
    """Parse the right-hand side of an equation or a raw expression string.

    Extends parsing with the ability to pass parameters, functions, symbols, and arbitrary SymPy objects commonly used in nonlinear systems dynamics.

    A user-defined parameter always overrides a SymPy built-in of the same name, so a model free to call something `gamma` or `lambda` gets its own symbol rather than the special function. Indexed variables are detected in the source text and bound as `IndexedBase`, which likewise overrides any `Symbol` of that name — `x[i]` cannot be parsed against a plain `Symbol`. Index variables are picked out of `Sum` and `Product` limits and bound as plain `Symbol`s where they are not already defined.

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

    local_dict.setdefault("Sum", Sum)
    local_dict.setdefault("Product", Product)
    local_dict.setdefault("IndexedBase", IndexedBase)
    local_dict.setdefault("Mean", Mean)  # Custom Mean function for indexed averaging
    local_dict.setdefault("sqrt", sqrt)

    # Merge array functions (user-provided local_dict takes precedence)
    for name, fn in ARRAY_FUNCTIONS.items():
        if name not in local_dict:
            local_dict[name] = fn

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

    indexed_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[")
    for match in indexed_pattern.finditer(expression):
        var_name = match.group(1)
        # Override even if already defined - indexed access requires IndexedBase
        local_dict[var_name] = IndexedBase(var_name)

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
