"""The axes an expression's value carries, followed by name through the expression tree.

A result is keyed by its axis names wherever they are known, so a step that reshapes its input (``sum_axis``, indexing, ``welch``) must say which axes it keeps rather than leave the container to guess from a shape. This module is that statement for the array vocabulary of :mod:`tvbo.parse.expression`: :func:`expression_dims` propagates a mapping of symbol names to axis names through one expression, and :func:`resolve_axis` lets a reduction name the axis it drops (``sum_axis(x, node)``) as well as number it. Both are backend-neutral; the printers in :mod:`tvbo.codegen.code` call them at codegen and the tvboptim observation layer uses them to declare a derived observation's axes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import sympy as sp

Dims = tuple[str, ...]

AXIS_ARGUMENTS: dict[str, int] = {
    "sum_axis": 1,
    "rankdata": 1,
    "normalize": 1,
    "slice_axis": 1,
    "slice_from": 1,
    "shape": 1,
}
"""For each primitive that takes an axis, the position of that argument. The axis is an integer literal or the name of one of its operand's axes."""

FREQUENCY = "frequency"
"""The axis a spectrum lies along."""


def resolve_axis(axis, operand_dims: Dims | None, signature: str, name: str = "axis") -> int:
    """*axis* as an ``int``: an integer literal, or the name of one of the operand's *operand_dims*.

    An array axis is fixed when the code is compiled, so a traced or symbolic value is refused. A name is refused when the operand's axes are not known at this point, naming the way out: give the position, or declare ``dims:`` on the source so the name can be followed.
    """
    if getattr(axis, "is_Integer", False):
        return int(axis)
    if isinstance(axis, sp.Symbol):
        if operand_dims is not None and str(axis) in operand_dims:
            return operand_dims.index(str(axis))
        where = (
            "; its axes are not known here, so give the axis as a position or declare `dims:` on the source"
            if operand_dims is None
            else f"; its axes are {list(operand_dims)}"
        )
        raise ValueError(f"{signature}: {name} {str(axis)!r} is not an axis of the operand{where}.")
    raise ValueError(f"{signature}: {name} must be an integer literal or an axis name, got {axis!r}.")


def axis_symbols(expr) -> set[str]:
    """The names standing in an axis position anywhere in *expr*, which are axis names rather than free variables."""
    names: set[str] = set()
    for node in sp.preorder_traversal(expr):
        at = AXIS_ARGUMENTS.get(getattr(getattr(node, "func", None), "__name__", ""))
        if at is not None and len(node.args) > at and isinstance(node.args[at], sp.Symbol):
            names.add(str(node.args[at]))
    return names


def _broadcast(parts: list[Dims | None]) -> Dims | None:
    """The axes an elementwise combination of operands carrying *parts* has: a scalar takes the other's, equal axes stay, and a trailing subset (numpy's alignment, by name) takes the longer; anything else is unknown."""
    if any(p is None for p in parts):
        return None
    out: Dims = ()
    for p in parts:
        if len(p) > len(out):
            out, p = p, out
        if p != out[len(out) - len(p) :]:
            return None
    return out


def _drop(dims: Dims | None, axis: int) -> Dims | None:
    if dims is None or not -len(dims) <= axis < len(dims):
        return None
    at = axis % len(dims)
    return dims[:at] + dims[at + 1 :]


def _keep(d: Callable[[sp.Basic], Dims | None], args) -> Dims | None:
    return d(args[0]) if args else None


def _reduce(d, args, signature, name="axis"):
    operand = d(args[0])
    return _drop(operand, resolve_axis(args[1], operand, signature, name))


def _reversed(d, args):
    operand = d(args[0])
    return None if operand is None else tuple(reversed(operand))


def _matmul(d, args):
    a, b = d(args[0]), d(args[1])
    return None if a is None or b is None or not a or not b else a[:-1] + b[1:]


def _outer(d, args):
    a, b = d(args[0]), d(args[1])
    return None if a is None or b is None else a + b


def _welch(d, args):
    operand = d(args[0])
    return None if operand is None or not operand else (FREQUENCY,) + operand[1:]


def _same(d, args):
    parts = [d(a) for a in args if not getattr(a, "is_Integer", False)]
    return parts[0] if parts and all(p == parts[0] for p in parts) else None


_RULES: dict[str, Callable] = {
    "sum_axis": lambda d, args: _reduce(d, args, "sum_axis(x, axis)"),
    "rankdata": _keep,
    "normalize": _keep,
    "slice_axis": _keep,
    "slice_from": _keep,
    "window_mean": _keep,
    "subsample": _keep,
    "global_mean": _keep,
    "mode_sum": _keep,
    "mode_dot": _keep,
    "take": _keep,
    "zero_diagonal": _keep,
    "fill_diagonal": _keep,
    "minmax_rescale": _keep,
    "interp": _keep,
    "transpose": _reversed,
    "pinv": _reversed,
    "matmul": _matmul,
    "outer": _outer,
    "welch": _welch,
    "rfftfreq": lambda d, args: (FREQUENCY,),
    "pearson": lambda d, args: (),
    "shape": lambda d, args: (),
    "concatenate": _same,
}
"""How each array primitive's axes follow from its operands'. A primitive of the vocabulary with no rule (``upper_triangle``, ``diag``, ``eigvals``, ``hann``, the samplers, the graph constructors) has axes no name describes, so its value is unknown here and an observation built on it declares its ``dims:``."""


def expression_dims(expr, env: Mapping[str, Dims | None]) -> Dims | None:
    """The axes *expr*'s value carries, given *env*, the axes of each symbol it reads (``()`` for a scalar), or ``None`` where they cannot be followed.

    Numbers are scalars; an elementwise operation broadcasts its operands by name (:func:`_broadcast`); indexing drops one leading axis per index; the array primitives follow :data:`_RULES`; a symbol or a function the table does not know is unknown, as is any operand that is.
    """
    from tvbo.parse.expression import ARRAY_FUNCTIONS

    def d(node) -> Dims | None:
        if isinstance(node, sp.Symbol):
            return env.get(str(node))
        if getattr(node, "is_Number", False) or isinstance(node, sp.NumberSymbol) or node in (sp.true, sp.false):
            return ()
        if isinstance(node, sp.Indexed):
            base = env.get(str(node.base.label))
            if base is None or len(node.indices) > len(base) or not all(getattr(i, "is_Number", False) for i in node.indices):
                return None
            return base[len(node.indices) :]
        if isinstance(node, sp.Piecewise):
            return _broadcast([d(pair.expr) for pair in node.args] + [d(pair.cond) for pair in node.args])
        if isinstance(node, sp.Function):
            name = node.func.__name__
            rule = _RULES.get(name)
            if rule is not None:
                return rule(d, node.args)
            if name in ARRAY_FUNCTIONS or name in AXIS_ARGUMENTS:
                return None
        if not node.args:
            return None
        return _broadcast([d(a) for a in node.args])

    return d(expr)
