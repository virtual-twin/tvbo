from sympy import parse_expr, Symbol, Function
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.parsing.latex import parse_latex

from tvbo.datamodel.tvbo_datamodel import Equation

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

    # Start with user-provided locals (no hidden defaults)
    local_dict = dict(kwargs.pop("local_dict", {}))

    # Helper to coerce iterables/mappings into name -> sympy object entries
    def _update_from_names_or_map(container, factory):
        if not container:
            return
        # Mapping case
        if isinstance(container, dict):
            local_dict.update(container)
            return
        # Iterable of names/objects
        try:
            for item in container:
                if isinstance(item, str):
                    obj = factory(item)
                    local_dict[item] = obj
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

    # Build transformations pipeline
    extra_transformations = tuple(kwargs.pop("extra_transformations", ()))
    transformations = kwargs.pop(
        "transformations",
        standard_transformations + (implicit_multiplication_application,) + extra_transformations,
    )

    # Remaining kwargs forwarded to parse_expr (e.g., evaluate=False, global_dict=...)
    return parse_expr(
        expression,
        local_dict=local_dict,
        transformations=transformations,
        **kwargs,
    )
