"""Read generated code back into SymPy and compare it with the specification.

The golden corpus freezes what each backend emits, so it reports that an output changed and says nothing about whether it was right. This module asks the other question: for one model and one backend, does every emitted right-hand side mean what the specification states? [`check_backend`](#tvbo.analysis.backend_check.check_backend) parses the emitted source, rebuilds each state variable's derivative as a SymPy expression over the model's own symbols, and compares it with the specification's, one verdict per state variable.

The readers interpret the emitted text and share nothing with the printers that wrote it, which is what makes agreement evidence: Python's `ast` for the JAX, NumPy, TVB and tvboptim outputs, tree-sitter for Julia, and the standard XML parser for LEMS. A reader refuses what it does not understand by raising [`Unreadable`](#tvbo.analysis.backend_check.Unreadable), and the verdict says so, because a construct skipped in silence would pass as agreement.

Two expressions are compared by subtracting them. A difference SymPy cancels to zero is a proof; otherwise both are evaluated at random points in 30-digit arithmetic, where agreement at every point is strong evidence and disagreement at one is a counterexample.
"""

from __future__ import annotations

import ast
import functools
import operator
import random
import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any
from xml.etree import ElementTree

import mpmath
import sympy as sp
from sympy.core.function import AppliedUndef

from tvbo.parse.expression import ARRAY_FUNCTIONS, function_bodies

EQUAL = "equal"
DIFFERS = "differs"
UNREAD = "unread"

SAMPLE_POINTS = 24
DRAWS_PER_POINT = 8
"""How many random points are drawn for each one that must evaluate to a real number on both sides."""
PRECISION = 30
TOLERANCE = 1e-12


class Unreadable(Exception):
    """The emitted source contains something the reader does not interpret."""


@dataclass
class Verdict:
    """Whether one state variable's emitted derivative matches the specification."""

    name: str
    status: str
    detail: str = ""
    specified: Any = None
    emitted: Any = None
    pinned: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status == EQUAL


@dataclass
class Readback:
    """What a reader recovered: each state variable's derivative, and the inputs the code fixes to a constant."""

    derivatives: dict
    pinned: dict = field(default_factory=dict)


def _piecewise(condition, when_true, when_false):
    return sp.Piecewise((when_true, condition), (when_false, True))


def _heaviside(argument):
    """LEMS's `H` as jLEMS and PyLEMS evaluate it: 1 above zero, 0 below, and 0.5 at zero."""
    if argument.is_Relational or isinstance(argument, sp.logic.boolalg.BooleanFunction):
        raise Unreadable(f"H({argument}) hands a condition to a function that takes a number")
    return sp.Piecewise((1, argument > 0), (sp.Rational(1, 2), sp.Eq(argument, 0)), (0, True))


def _reduce_over_modes(operand, **keywords):
    """`sum(x, axis=-1, keepdims=True)`, the only reduction the dynamics emitters write."""
    if keywords != {"axis": -1, "keepdims": True}:
        raise Unreadable(f"a sum with arguments {keywords} is not the mode sum")
    return ARRAY_FUNCTIONS["mode_sum"](operand)


def _conjunction(conditions, axis=0):
    """`all([a, b], axis=0)` and `logical_and.reduce((a, b))`: every listed condition holds."""
    return sp.And(*conditions)


FUNCTIONS = {
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pow": sp.Pow,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "arcsin": sp.asin,
    "asin": sp.asin,
    "arccos": sp.acos,
    "acos": sp.acos,
    "arctan": sp.atan,
    "atan": lambda *arguments: sp.atan(*arguments) if len(arguments) == 1 else sp.atan2(*arguments),
    "arctan2": sp.atan2,
    "atan2": sp.atan2,
    "abs": sp.Abs,
    "sign": sp.sign,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "mod": sp.Mod,
    "minimum": sp.Min,
    "min": sp.Min,
    "maximum": sp.Max,
    "max": sp.Max,
    "less": sp.Lt,
    "less_equal": sp.Le,
    "greater": sp.Gt,
    "greater_equal": sp.Ge,
    "equal": sp.Eq,
    "not_equal": sp.Ne,
    "logical_and": sp.And,
    "logical_or": sp.Or,
    "logical_not": sp.Not,
    "all": _conjunction,
    "logical_and.reduce": _conjunction,
    "where": _piecewise,
    "ifelse": _piecewise,
    "H": _heaviside,
    "dot": ARRAY_FUNCTIONS["mode_dot"],
    "sum": _reduce_over_modes,
}
"""Every function name a reader accepts, in any backend's spelling, as the SymPy object it means."""

PYTHON_CONSTANTS = {"pi": sp.pi, "inf": sp.oo, "e": sp.E}
"""The constants of a numeric module, read where the code names them as attributes (`np.pi`, `jnp.e`)."""

JULIA_CONSTANTS = {"pi": sp.pi, "π": sp.pi, "ℯ": sp.E, "Inf": sp.oo}
"""The constants Julia's `Base` defines; Euler's number is `ℯ`, so a bare `e` is an unbound name, as it is when the code runs."""

SHAPE_ONLY = {"asarray", "atleast_1d", "atleast_2d", "broadcast_to", "moveaxis", "full_like", "float", "squeeze"}
"""Calls that change an array's shape and leave its value alone, read as their first argument."""

STACKING = {"array", "stack", "broadcast_arrays", "concatenate"}
"""Calls that gather values into one array, read as the tuple of what they gather."""


class _Input:
    """A function argument the emitted code reads its inputs from."""

    def __init__(self, role: str):
        self.role = role


@dataclass
class _Local:
    """A function the emitted code defines beside its derivatives, called with the bindings in force at the call."""

    formals: list[str]
    body: Any

    def bind(self, arguments: list, environment: dict, where: str) -> dict:
        """*environment* with each formal bound to its argument; *where* names the call in the refusal."""
        if len(self.formals) != len(arguments):
            raise Unreadable(f"{where} does not bind the arguments of a local function one to one")
        return {**environment, **dict(zip(self.formals, arguments, strict=True))}


class _Scope:
    """The model's declared names, as the symbols its own equations were parsed against."""

    def __init__(self, model, modes_flattened: bool = False):
        from tvbo.templates.base.utils import get_coupling_terms

        self.stride = int(model.number_of_modes or 1) if modes_flattened else 1
        self.symbols = model.symbolic_system.scope()
        self.states = [str(name) for name in model.state_variables]
        self.inputs, self.coupling, _ = get_coupling_terms(model)

    def symbol(self, name: str):
        resolved = self.symbols.get(name)
        if not isinstance(resolved, sp.Basic):
            raise Unreadable(f"reads `{name}`, which the model does not declare")
        return resolved

    def positional(self, role: str, index: int):
        names = self.states if role == "state" else self.coupling
        index = index // self.stride if role == "state" else index
        if not 0 <= index < len(names):
            raise Unreadable(f"reads {role} entry {index}, and the model declares {len(names)}")
        return self.symbol(names[index])

    def pin(self, pinned: dict, name: str, value) -> None:
        """Record that the code fixes the coupling input *name* to a constant instead of reading it."""
        if name in self.inputs and isinstance(value, sp.Basic) and value.is_number:
            pinned[self.symbol(name)] = value


_DFUN = ("dfun", {"current_state": "state", "cX": "coupling", "_p": "named"})
PYTHON_ENTRY = {
    "jax": _DFUN,
    "numpy": _DFUN,
    "tvb": ("dfun", {"self": "named", "state_variables": "state", "coupling": "coupling"}),
    "tvboptim": ("dynamics", {"self": "named", "state": "state", "params": "named", "coupling": "named", "external": "named"}),
}
"""The function each Python format puts its derivatives in, and how each of its arguments is read: by position, or by attribute name."""

MODES_FLATTENED = {"tvboptim", "julia"}
"""Formats whose state vector holds one entry per mode of each state variable, variable by variable."""

CHECKED_FORMATS = (*PYTHON_ENTRY, "julia", "neuroml")
"""The code formats a reader covers."""

_ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "^": operator.pow}
_COMPARISONS = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge, "==": sp.Eq, "!=": sp.Ne}
_UNARY = {"-": operator.neg, "+": operator.pos, "!": sp.Not}
"""The operators both languages spell alike, by spelling."""

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: sp.Mod,
}
_COMPARE = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge, ast.Eq: sp.Eq, ast.NotEq: sp.Ne}
_PYTHON_UNARY = {ast.USub: _UNARY["-"], ast.UAdd: _UNARY["+"], ast.Not: _UNARY["!"], ast.Invert: _UNARY["!"]}


def dotted(node) -> str:
    """`jnp.special.erfc` for the attribute chain that spells it, or `""` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


class _PythonReader:
    """Interprets a Python expression tree over an environment of SymPy values."""

    def __init__(self, scope: _Scope, environment: dict | None = None, modules: frozenset = frozenset()):
        self.scope = scope
        self.environment = dict(environment or {})
        self.modules = modules
        self.pinned: dict = {}

    def run(self, statements):
        """Execute straight-line *statements* and return what they return."""
        for statement in statements:
            if isinstance(statement, ast.Return):
                return self.value(statement.value)
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            if isinstance(statement, ast.Assign):
                value = self.value(statement.value)
                for target in statement.targets:
                    self.assign(target, value)
                continue
            if isinstance(statement, ast.FunctionDef):
                self.environment[statement.name] = _Local([argument.arg for argument in statement.args.args], statement.body)
                continue
            if isinstance(statement, ast.If) and self.binds_nothing(statement):
                continue
            raise Unreadable(f"line {statement.lineno}: a {type(statement).__name__} statement")
        raise Unreadable("the entry point returns nothing")

    @staticmethod
    def binds_nothing(conditional: ast.If) -> bool:
        """Whether a conditional only stores attributes on the instance, which no derivative is read from."""
        return all(
            isinstance(statement, ast.Assign)
            and all(isinstance(target, ast.Attribute) and dotted(target).startswith("self.") for target in statement.targets)
            for statement in [*conditional.body, *conditional.orelse]
        )

    def assign(self, target, value) -> None:
        if isinstance(target, ast.Name):
            self.environment[target.id] = value
            self.scope.pin(self.pinned, target.id, value)
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple) and len(target.elts) == len(value):
            for element, item in zip(target.elts, value, strict=True):
                self.assign(element, item)
        elif not (isinstance(target, ast.Attribute) and dotted(target) == f"self.{target.attr}"):
            raise Unreadable(f"line {target.lineno}: an assignment to {ast.unparse(target)}")

    def value(self, node):  # noqa: C901 — one branch per expression node kind
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else sp.sympify(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise Unreadable(f"line {node.lineno}: `{node.id}` is read before anything binds it")
            return self.environment[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.value(element) for element in node.elts)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self.value(node.left), self.value(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _PYTHON_UNARY:
            return _PYTHON_UNARY[type(node.op)](self.value(node.operand))
        if isinstance(node, ast.BoolOp):
            return (sp.And if isinstance(node.op, ast.And) else sp.Or)(*[self.value(v) for v in node.values])
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            operands = [self.value(node.left), *[self.value(c) for c in node.comparators]]
            return sp.And(*[_COMPARE[type(op)](a, b) for op, (a, b) in zip(node.ops, pairwise(operands), strict=True)])
        if isinstance(node, ast.IfExp):
            condition = self.value(node.test)
            if condition is True or condition is False:
                return self.value(node.body if condition else node.orelse)
            return _piecewise(condition, self.value(node.body), self.value(node.orelse))
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        raise Unreadable(f"line {node.lineno}: the expression {ast.unparse(node)}")

    def attribute(self, node):
        module, _, name = dotted(node).rpartition(".")
        if module.split(".")[0] in self.modules:
            if name in PYTHON_CONSTANTS:
                return PYTHON_CONSTANTS[name]
            raise Unreadable(f"line {node.lineno}: the constant {ast.unparse(node)}")
        if node.attr == "shape":
            return None
        base = self.value(node.value)
        if isinstance(base, _Input) and base.role == "named":
            return self.scope.symbol(node.attr)
        raise Unreadable(f"line {node.lineno}: the attribute {ast.unparse(node)}")

    def subscript(self, node):
        base = self.value(node.value)
        index = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
            raise Unreadable(f"line {node.lineno}: the index in {ast.unparse(node)}")
        if isinstance(base, _Input) and base.role in ("state", "coupling"):
            return self.scope.positional(base.role, index.value)
        if isinstance(node.value, ast.Call) and dotted(node.value.func).endswith("atleast_1d") and index.value == 0:
            return base
        raise Unreadable(f"line {node.lineno}: the subscript {ast.unparse(node)}")

    def call(self, node):
        spelled = dotted(node.func)
        if isinstance(node.func, ast.Attribute) and spelled.split(".")[0] not in self.modules:
            raise Unreadable(f"line {node.lineno}: the method call {ast.unparse(node)}")
        name = next((spelling for spelling in FUNCTIONS if spelled.endswith(f".{spelling}")), spelled.rpartition(".")[2])
        if name == "hasattr":
            return True
        arguments = [self.value(argument) for argument in node.args]
        keywords = {keyword.arg: _literal(keyword) for keyword in node.keywords}
        local = self.environment.get(name)
        if isinstance(local, _Local):
            where = f"line {node.lineno}: the call {ast.unparse(node)}"
            if keywords:
                raise Unreadable(f"{where} passes keywords to a local function")
            return _PythonReader(self.scope, local.bind(arguments, self.environment, where), self.modules).run(local.body)
        if name in SHAPE_ONLY:
            return arguments[0]
        if name in STACKING:
            gathered = arguments[0] if len(arguments) == 1 and isinstance(arguments[0], tuple) else tuple(arguments)
            modes_of_one_variable = keywords.get("axis") == -1 and len(set(gathered)) == 1
            return gathered[0] if modes_of_one_variable else tuple(gathered)
        if name in FUNCTIONS:
            return _apply(name, arguments, keywords, f"line {node.lineno}: the call {ast.unparse(node)}")
        raise Unreadable(f"line {node.lineno}: the call {ast.unparse(node)}")


def _apply(name: str, arguments: list, keywords: dict, where: str):
    """`FUNCTIONS[name]` of *arguments*; arguments the function does not take make the call unreadable, named by *where*."""
    try:
        return FUNCTIONS[name](*arguments, **keywords)
    except TypeError as error:
        raise Unreadable(f"{where} does not take these arguments: {error}") from error


def _literal(keyword: ast.keyword):
    """The constant a call passes as *keyword*, such as `axis=-1`."""
    try:
        return ast.literal_eval(keyword.value)
    except ValueError as error:
        raise Unreadable(f"line {keyword.value.lineno}: the keyword value {ast.unparse(keyword)}") from error


def _imported_modules(tree) -> frozenset:
    """The names the module's imports bind, which an attribute chain may start from."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return frozenset(names)


def read_python(code: str, model, format: str) -> Readback:
    """Recover the derivatives a generated Python module computes.

    Finds the format's entry point, binds its arguments to the model's inputs, and interprets the body as straight-line assignments. State and coupling arrays are read by position against the model's declared order, so an emitter that swapped two state variables would produce a different expression rather than the same one under another name.
    """
    tree = ast.parse(code)
    entry, roles = PYTHON_ENTRY[format]
    function = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    if function is None:
        raise Unreadable(f"no function named `{entry}`")
    scope = _Scope(model, modes_flattened=format in MODES_FLATTENED)
    reader = _PythonReader(scope, modules=_imported_modules(tree))
    arguments = function.args.args
    defaults = dict(zip(reversed(arguments), reversed(function.args.defaults), strict=False))
    for argument in arguments:
        name = argument.arg
        if name in roles:
            reader.environment[name] = _Input(roles[name])
        elif name in scope.symbols or argument not in defaults:
            reader.environment[name] = scope.symbol(name)
        else:
            reader.assign(ast.Name(id=name), reader.value(defaults[argument]))
    returned = reader.run(function.body)
    derivatives = returned[0] if isinstance(returned, tuple) and returned and isinstance(returned[0], tuple) else returned
    if (
        not isinstance(derivatives, tuple)
        or len(derivatives) != len(scope.states)
        or not all(isinstance(derivative, sp.Basic) for derivative in derivatives)
    ):
        raise Unreadable(f"returns {derivatives!r}, not one derivative for each of {len(scope.states)} state variables")
    return Readback(dict(zip(scope.states, derivatives, strict=True)), reader.pinned)


_LEMS_OPERATORS = {
    "^": "**",
    ".lt.": "<",
    ".leq.": "<=",
    ".gt.": ">",
    ".geq.": ">=",
    ".eq.": "==",
    ".neq.": "!=",
    ".and.": " and ",
    ".or.": " or ",
}
_LEMS_OPERATOR = re.compile("|".join(re.escape(spelling) for spelling in _LEMS_OPERATORS))


class _LemsReader:
    """Resolves a LEMS component type's names on demand, since LEMS declares them in no particular order."""

    def __init__(self, scope: _Scope, dynamics):
        self.scope = scope
        self.definitions = {element.get("name"): element for element in dynamics if element.get("name")}
        self.resolved: dict = {"SEC": sp.Integer(1)}

    def expression(self, text: str):
        tree = ast.parse(_LEMS_OPERATOR.sub(lambda match: _LEMS_OPERATORS[match.group()], text).strip(), mode="eval")
        declared = self.definitions.keys() | self.scope.symbols.keys() | self.resolved.keys()
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - FUNCTIONS.keys()
        environment = {name: self.name(name) for name in names & declared}
        return _PythonReader(self.scope, environment).value(tree.body)

    def name(self, name: str):
        if name not in self.resolved:
            definition = self.definitions.get(name)
            self.resolved[name] = self.scope.symbol(name) if definition is None else self.derived(definition)
        return self.resolved[name]

    def derived(self, element):
        if element.tag == "DerivedVariable":
            return self.expression(element.get("value"))
        if element.tag == "ConditionalDerivedVariable":
            cases = [(self.expression(c.get("value")), c.get("condition")) for c in element.findall("Case")]
            return sp.Piecewise(*[(value, self.expression(when) if when else True) for value, when in cases])
        return self.scope.symbol(element.get("name"))


def read_lems(code: str, model, format: str) -> Readback:
    """Recover the time derivatives a generated LEMS component type declares.

    `SEC`, the constant the emitter divides by to convert model time to seconds, is read as one unit of model time. Regimes and event handlers are not read: this compares the flow between events.
    """
    component = ElementTree.fromstring(code).find("ComponentType")
    dynamics = component.find("Dynamics") if component is not None else None
    if dynamics is None:
        raise Unreadable("no ComponentType with a Dynamics block")
    if dynamics.find("Regime") is not None:
        raise Unreadable("the dynamics are split into regimes")
    reader = _LemsReader(_Scope(model), dynamics)
    derivatives = {d.get("variable"): reader.expression(d.get("value")) for d in dynamics.findall("TimeDerivative")}
    return Readback(derivatives)


_JULIA_BINARY = {
    **_ARITHMETIC,
    **_COMPARISONS,
    "//": operator.truediv,
    "%": lambda a, b: a - b * sp.sign(a / b) * sp.floor(sp.Abs(a / b)),  # Julia's `%` is `rem`, truncated toward zero
    "&&": sp.And,
    "||": sp.Or,
    "&": sp.And,
    "|": sp.Or,
}


def julia_parser():
    """A tree-sitter parser for Julia source; reading Julia needs `tvbo[julia]`."""
    try:
        import tree_sitter
        import tree_sitter_julia
    except ImportError as error:
        raise ImportError("Reading generated Julia needs tree-sitter: install `tvbo[julia]`.") from error
    return tree_sitter.Parser(tree_sitter.Language(tree_sitter_julia.language()))


class _JuliaReader:
    """Interprets a Julia function body over an environment of SymPy values."""

    def __init__(self, scope: _Scope, arguments: list[str]):
        self.scope = scope
        self.derivative_array, self.state_array, self.parameters = arguments[:3]
        self.environment: dict = {name: scope.symbol(name) for name in arguments[3:]}
        self.derivatives: dict = {}

    def run(self, statements) -> None:
        for statement in statements:
            if statement.type == "assignment" or (
                statement.type == "compound_assignment_expression" and statement.children[1].text == b".="
            ):
                target, _, value = statement.children
                self.assign(target, value)
            elif statement.type not in ("identifier", "line_comment"):
                raise Unreadable(f"line {statement.start_point[0] + 1}: a {statement.type} statement")

    def assign(self, target, value) -> None:
        text = value.text.decode()
        if target.type == "tuple_expression" and text == self.parameters:
            for name in self.names(target):
                self.environment[name] = self.scope.symbol(name)
        elif target.type in ("open_tuple", "tuple_expression") and text == self.state_array:
            for index, name in enumerate(self.names(target)):
                self.environment[name] = self.scope.positional("state", index)
        elif target.type == "index_expression" and target.children[0].text.decode() == self.derivative_array:
            self.derivatives[str(self.entry(target, self.derivative_array))] = self.value(value)
        elif target.type == "identifier":
            self.environment[target.text.decode()] = self.value(value)
        elif target.type == "call_expression":
            self.environment[target.children[0].text.decode()] = _Local(self.names(target.children[1]), value)
        else:
            raise Unreadable(f"line {target.start_point[0] + 1}: an assignment to {target.text.decode()}")

    @staticmethod
    def names(node) -> list[str]:
        return [child.text.decode() for child in node.children if child.type == "identifier"]

    def entry(self, node, array: str):
        """The state variable that `array[i]`, or the slice `array[first:last]` over its modes, belongs to."""
        where = f"line {node.start_point[0] + 1}: {node.text.decode()}"
        subscripts = node.children[1].named_children if node.type == "index_expression" else []
        if len(subscripts) != 1 or node.children[0].text.decode() != array:
            raise Unreadable(f"{where} is not an entry of `{array}`")
        if subscripts[0].type == "integer_literal":
            return self.scope.positional("state", int(subscripts[0].text) - 1)
        bounds = subscripts[0].named_children if subscripts[0].type == "range_expression" else []
        if len(bounds) == 2 and all(bound.type == "integer_literal" for bound in bounds):
            first, last = (int(bound.text) for bound in bounds)
            if last - first + 1 == self.scope.stride:
                return self.scope.positional("state", first - 1)
        raise Unreadable(f"{where} is not one state variable")

    def columns_of(self, node):
        """The mode-coupling matrix `A` where *node* is `reduce(hcat, A)`, else `None`.

        `reduce(hcat, A)` stacks the rows of `A` as columns, so `reduce(hcat, A) * x` is `x` contracted with the first index of `A`: the specification's `mode_dot(x, A)`.
        """
        while node.type == "parenthesized_expression" and node.named_child_count == 1:
            node = node.named_children[0]
        if node.type != "call_expression" or node.children[0].text != b"reduce":
            return None
        arguments = node.children[-1].named_children
        if len(arguments) != 2 or arguments[0].text != b"hcat":
            return None
        return self.value(arguments[1])

    def value(self, node):  # noqa: C901 — one branch per expression node kind
        text = node.text.decode()
        if node.type == "integer_literal":
            return sp.Integer(text)
        if node.type == "float_literal":
            return sp.Float(text)
        if node.type == "boolean_literal":
            return sp.true if text == "true" else sp.false
        if node.type == "identifier":
            if text in self.environment:
                return self.environment[text]
            if text in JULIA_CONSTANTS:
                return JULIA_CONSTANTS[text]
            raise Unreadable(f"line {node.start_point[0] + 1}: `{text}` is read before anything binds it")
        if node.type == "parenthesized_expression" and node.named_child_count == 1:
            return self.value(node.named_children[0])
        if node.type == "binary_expression":
            left, operator, right = node.children
            spelling = operator.text.decode().lstrip(".") or "."
            matrix = self.columns_of(left)
            if matrix is not None and operator.text == b"*":
                return ARRAY_FUNCTIONS["mode_dot"](self.value(right), matrix)
            if spelling in _JULIA_BINARY:
                return _JULIA_BINARY[spelling](self.value(left), self.value(right))
        if node.type == "unary_expression":
            operator, operand = node.children
            spelling = operator.text.decode().lstrip(".")
            if spelling in _UNARY:
                return _UNARY[spelling](self.value(operand))
        if node.type == "ternary_expression":
            condition, when_true, when_false = node.named_children
            return _piecewise(self.value(condition), self.value(when_true), self.value(when_false))
        if node.type == "index_expression":
            return self.entry(node, self.state_array)
        if node.type == "macrocall_expression" and node.children[0].text.decode() == "@view":
            return self.entry(node.children[1].named_children[0], self.state_array)
        if node.type in ("call_expression", "broadcast_call_expression"):
            name = node.children[0].text.decode().rpartition(".")[2]
            arguments = [self.value(argument) for argument in node.children[-1].named_children]
            local = self.environment.get(name)
            if isinstance(local, _Local):
                outer = self.environment
                self.environment = local.bind(arguments, outer, f"line {node.start_point[0] + 1}: {text}")
                try:
                    return self.value(local.body)
                finally:
                    self.environment = outer
            if name in ("-", "+") and len(arguments) == 1:
                return _UNARY[name](arguments[0])
            if name == "sum" and len(arguments) == 1:
                return ARRAY_FUNCTIONS["mode_sum"](arguments[0])  # a state is a vector over the modes
            if name in FUNCTIONS:
                return _apply(name, arguments, {}, f"line {node.start_point[0] + 1}: {text}")
        raise Unreadable(f"line {node.start_point[0] + 1}: the expression {text}")


def read_julia(code: str, model, format: str) -> Readback:
    """Recover the derivatives a generated Julia in-place function writes into its first argument.

    The function is the first one in the file whose name ends in `!`, with the signature `(dx, x, p, t)`. Julia's broadcasting dots are dropped, since `a .+ b` and `a + b` state the same arithmetic on the per-node values the specification describes.
    """
    tree = julia_parser().parse(code.encode())
    function = next((node for node in tree.root_node.children if node.type == "function_definition"), None)
    if function is None or tree.root_node.has_error:
        raise Unreadable("the source is not a Julia module with a function definition")
    signature = function.child_by_field_name("signature") or function.named_children[0]
    call = signature.named_children[0]
    arguments = [
        (argument.named_children[0] if argument.type == "named_argument" else argument).text.decode()
        for argument in call.named_children[1].named_children
    ]
    if len(arguments) < 3:
        raise Unreadable(f"the function takes {arguments}, not (dx, x, p, t)")
    scope = _Scope(model, modes_flattened=format in MODES_FLATTENED)
    reader = _JuliaReader(scope, arguments)
    reader.run([node for node in function.named_children if node != signature])
    return Readback(reader.derivatives)


READERS = {"python": read_python, "julia": read_julia, "xml": read_lems}
"""The reader for each output language a code format declares."""


def specified_derivatives(model) -> dict:
    """Each state variable's right-hand side as the specification states it, keyed by state variable.

    Derived parameters and derived variables stay as their symbols, and the comparison evaluates them from their own definitions. For a discrete system the right-hand side is the map itself.
    """
    form = model.symbolic_system.form()
    return {name: equation.rhs for name, equation in form["state-equations"].items()}


def _definitions(model) -> dict:
    """The model's derived quantities, `{symbol: expression}`, in an order where each follows what it reads."""
    form = model.symbolic_system.form()
    return {
        equation.lhs: equation.rhs
        for group in ("derived-parameters", "derived-variables")
        for equation in form[group].values()
    }


def _stand_in(application):
    """A concrete expression for a call SymPy cannot evaluate, the same for the same call on either side.

    `abs` is the one such head with a meaning of its own: the parser keeps it as an array function, and it states the absolute value.
    """
    if application.func.__name__ == "abs":
        return sp.Abs(*application.args)
    seed = sum(ord(character) for character in application.func.__name__)
    return sp.sin(seed + sum((position + sp.Rational(137, 100)) * a for position, a in enumerate(application.args)))


@functools.lru_cache(maxsize=4096)
def _compiled(concrete) -> tuple:
    """*concrete*'s free symbols in a fixed order, and the mpmath function of them it computes; the same for the same expression in any model."""
    symbols = tuple(sorted(concrete.free_symbols, key=str))
    return symbols, sp.lambdify(symbols, concrete, modules="mpmath", cse=True)


class _Comparison:
    """Compares expressions over one model's definitions at one fixed set of random points."""

    def __init__(self, model, pinned: dict):
        from tvbo.codegen.code import inline_functions

        self.inline_functions = inline_functions
        self.bodies = function_bodies(model)
        self.definitions = _definitions(model)
        self.pinned = pinned
        self.random = random.Random(0)
        self.points: list[dict] = []
        self.compiled: dict = {}

    def point(self, index: int, symbols) -> dict:
        """Sample point *index*, extended to cover *symbols*: inputs drawn at random, definitions evaluated from them.

        Odd points draw every input positive, where the roots and logarithms of a model written for positive rates and time constants stay real; even points draw both signs, which is what reaches both sides of a threshold.
        """
        while len(self.points) <= index:
            self.points.append({symbol: mpmath.mpmathify(value) for symbol, value in self.pinned.items()})
        point = self.points[index]
        for symbol in symbols:
            if symbol in point:
                continue
            if symbol in self.definitions:
                point[symbol] = self.evaluate(self.definitions[symbol], index)
            else:
                point[symbol] = mpmath.mpf(self.random.uniform(0.1 if index % 2 else -2, 2))
        return point

    def evaluate(self, expression, index: int):
        """The value of *expression* at sample point *index*."""
        if expression not in self.compiled:
            inlined = self.inline_functions(expression, self.bodies)
            self.compiled[expression] = _compiled(inlined.replace(lambda e: isinstance(e, AppliedUndef), _stand_in))
        symbols, function = self.compiled[expression]
        point = self.point(index, symbols)
        return function(*[point[symbol] for symbol in symbols])

    def real_value(self, expression, index: int):
        """The value of *expression* at sample point *index* when it is a finite real number, else `None`; a complex value whose imaginary part vanishes, as `sqrt(k/m)**2` at a negative `m`, is real."""
        try:
            value = mpmath.mpmathify(self.evaluate(expression, index))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if isinstance(value, mpmath.mpc) and abs(value.imag) <= TOLERANCE * max(1, abs(value.real)):
            value = value.real
        return value if isinstance(value, mpmath.mpf) and mpmath.isfinite(value) else None

    def verdict(self, name: str, specified, emitted) -> Verdict:
        """Equal when the two are one expression, or agree at every sample point where either is a finite real number.

        A point where only one side is a finite real number is a difference: an emitted `sqrt(x)` for a specified `sqrt(Abs(x))` differs wherever `x` is negative. Inputs the code fixes to a constant are fixed in the specification too, and named in the verdict.
        """
        common = {"name": name, "specified": specified, "emitted": emitted, "pinned": self.pinned}
        fixed = ", ".join(f"{symbol} = {value}" for symbol, value in self.pinned.items())
        but = f"; the code fixes {fixed}" if fixed else ""
        if specified.xreplace(self.pinned) == emitted:
            return Verdict(status=EQUAL, detail=f"the same expression{but}", **common)
        compared = 0
        with mpmath.workdps(PRECISION):
            for index in range(SAMPLE_POINTS * DRAWS_PER_POINT):
                if compared == SAMPLE_POINTS:
                    break
                expected, found = self.real_value(specified, index), self.real_value(emitted, index)
                if expected is None and found is None:
                    continue
                compared += 1
                if expected is None or found is None or abs(expected - found) > TOLERANCE * max(1, abs(expected)):
                    read = specified.free_symbols | emitted.free_symbols
                    at = {str(symbol): mpmath.nstr(value, 6) for symbol, value in self.points[index].items() if symbol in read}
                    spelled = ["no real number" if value is None else mpmath.nstr(value, 12) for value in (expected, found)]
                    detail = f"specified {spelled[0]}, emitted {spelled[1]} at {at}{but}"
                    return Verdict(status=DIFFERS, detail=detail, **common)
        if compared < SAMPLE_POINTS // 2:
            detail = f"only {compared} of {SAMPLE_POINTS * DRAWS_PER_POINT} random points evaluate to a real number"
            return Verdict(status=UNREAD, detail=detail, **common)
        return Verdict(status=EQUAL, detail=f"agrees at {compared} random points{but}", **common)


def check_backend(model, format: str, code: str | None = None) -> list[Verdict]:
    """Per-state-variable verdicts on whether *format*'s generated code states the model's equations.

    Args:
        model: The `Dynamics` whose specification is the reference.
        format: A code format `Dynamics.render_code` accepts, e.g. `"jax"`, `"tvb"`, `"julia"` or `"neuroml"`.
        code: Source to check in place of a fresh render, e.g. a frozen reference or a hand-edited file.

    Returns:
        One [`Verdict`](#tvbo.analysis.backend_check.Verdict) per state variable. `equal` means the emitted derivative matches the specification, `differs` carries a point where the two disagree, and `unread` means the reader met something it does not interpret, named in `detail`.

    Raises:
        ValueError: If *format* emits a language no reader covers, or Python whose entry point no reader knows.
    """
    from tvbo.codegen.templater import source_language

    language = source_language(format)
    if language not in READERS or language == "python" and format not in PYTHON_ENTRY:
        raise ValueError(f"Format '{format}' emits {language or 'unformatted'} output in a layout no reader covers.")
    specified = specified_derivatives(model)
    try:
        readback = READERS[language](model.render_code(format=format) if code is None else code, model, format)
    except (Unreadable, SyntaxError, ElementTree.ParseError) as error:
        return [Verdict(name, UNREAD, str(error), expression) for name, expression in specified.items()]
    comparison = _Comparison(model, readback.pinned)
    verdicts = []
    for name, expression in specified.items():
        if name not in readback.derivatives:
            verdicts.append(Verdict(name, DIFFERS, "the code computes no derivative for it", expression))
        else:
            verdicts.append(comparison.verdict(name, expression, readback.derivatives[name]))
    return verdicts
