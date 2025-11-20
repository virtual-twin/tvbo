import sympy.printing.julia as spj
import sympy.printing.numpy as spn
import sympy.printing.fortran as spf
from sympy import IndexedBase, parse_expr, Symbol, S
from sympy import latex
from tvbo.datamodel.tvbo_datamodel import Equation
from tvbo.knowledge.simulation.equations import _clash1, sympify as tvbo_sympify
from tvbo.parse.expression import parse_eq


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


def get_printer(format):

    if format in ["python", "numpy"]:
        return NumPyPrinter()
    elif format == "jax":
        return JaxPrinter()
    elif format == "julia":
        return JuliaPrinter()
    elif format == "fortran":
        return FortranPrinter()
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
    **kwargs,
):
    # Ensure parsing knows about symbols and undefined functions from the model scope
    expr = parse_eq(equation, local_dict=local_dict, **kwargs)

    if format == "latex":
        return latex(expr)

    if replace:
        symbol_map = {Symbol(k): Symbol(v) for k, v in replace.items()}
        expr = expr.xreplace(symbol_map)

    if remove:
        expr = expr.xreplace({Symbol(k): S.Zero for k in remove})

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
