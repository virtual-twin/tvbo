import sympy.printing.julia as spj
import sympy.printing.numpy as spn
import sympy.printing.fortran as spf
import sympy.printing.c as spc
from sympy.printing.pycode import PythonCodePrinter as _PythonCodePrinter
from sympy import IndexedBase, parse_expr, Symbol, S, Function, preorder_traversal
from sympy import latex
from sympy.printing import StrPrinter
from tvbo.datamodel.tvbo_datamodel import Equation
from tvbo.knowledge.simulation.equations import _clash1, sympify as tvbo_sympify
from tvbo.parse.expression import parse_eq


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
            if hasattr(sub_expr, 'func') and sub_expr.func == F:
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

    def _print_Piecewise(self, expr):
        return print_Piecewise(self, expr)


class JuliaPrinter(spj.JuliaCodePrinter):
    def __init__(self, settings=None):
        settings = settings or {}
        # Be tolerant: allow partial printing instead of raising for unknown constructs.
        settings.setdefault("strict", False)
        super().__init__(settings=settings)

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


class FortranPrinter(spf.FCodePrinter):
    def __init__(self, settings=None):
        settings = settings or {}
        settings.setdefault("source_format", "free")
        settings.setdefault("standard", 2003)
        settings.setdefault("contract", False)
        super().__init__(settings=settings)


class PythonCodePrinter(_PythonCodePrinter):
    def __init__(self, settings=None):
        settings = settings or {}
        # Be lenient: allow partial printing for unknown constructs
        settings.setdefault("strict", False)
        super().__init__(settings=settings)

        # Add additional math functions not in the base printer
        self.known_functions.update({
            "ceil": "math.ceil",
            "sign": "math.copysign(1, {0})",  # Python's math doesn't have sign directly
        })

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


def get_printer(format):

    if format == "numpy":
        return NumPyPrinter()
    elif format == "jax":
        return JaxPrinter()
    elif format == "julia":
        return JuliaPrinter()
    elif format == "fortran":
        return FortranPrinter()
    elif format == "python":
        return PythonCodePrinter()
    elif format in ["sympy", "symbolic", "pyrates"]:
        # Return StrPrinter for plain SymPy string output (exp, sin, etc. without prefix)
        # This is suitable for PyRates which uses SymPy's parser internally
        return StrPrinter()
    else:
        raise ValueError(f"Unsupported format: {format}")


def render_expression(
    expression,
    format="jax",
    # module="np",
    # fully_qualified_modules=False,
    user_functions={},
):
    indexed_symbols = {"x_j": IndexedBase("x_j"), "x_i": IndexedBase("x_i")}

    if isinstance(expression, str):
        try:
            expression = parse_expr(expression, {**_clash1, **indexed_symbols})
        except Exception:
            print("Failed to parse expression string; trying tvbo.sympify.")
            # Fallback: use sympify from equations.py when parse_expr cannot handle the input
            expression = tvbo_sympify(expression)
    printer = get_printer(format)
    printer.known_functions.update(user_functions)

    # Printer._settings.update({"fully_qualified_modules": fully_qualified_modules})
    return printer.doprint(expression)


def render_equation(
    equation: Equation,
    format="jax",
    local_dict={},
    user_functions={},
    replace=None,
    remove=None,
    inline_funcs=None,
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
    **kwargs
        Additional arguments passed to parse_eq.

    Returns
    -------
    str
        The rendered equation string.
    """
    # Ensure parsing knows about symbols and undefined functions from the model scope
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

    # Build a minimal user_functions mapping so printers emit f(x) for model-defined
    # symbolic functions. Prefer SymPy function heads (is_Function) and avoid tagging
    # arbitrary Python callables to reduce false-positives.
    uf = dict(user_functions) if isinstance(user_functions, dict) else {}
    if not uf and isinstance(local_dict, dict) and local_dict:
        for name, obj in local_dict.items():
            if getattr(obj, "is_Function", False):
                uf[str(name)] = str(name)

    printer = get_printer(format)
    # Update printer known functions if available (NumPy/JAX printers support this)
    try:
        printer.known_functions.update(uf)
    except Exception:
        pass
    return printer.doprint(expr)
