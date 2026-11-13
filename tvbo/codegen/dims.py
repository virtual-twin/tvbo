"""The axes an expression's value carries, followed by name through the expression tree.

A result is keyed by its axis names wherever they are known, so a step that reshapes its input (``sum_axis``, indexing, ``welch``) must say which axes it keeps rather than leave the container to guess from a shape. This module is that statement for the array vocabulary of :mod:`tvbo.parse.expression`: :func:`expression_dims` propagates a mapping of symbol names to axis names through one expression, and :func:`resolve_axis` lets a reduction name the axis it drops (``sum_axis(x, node)``) as well as number it. Both are backend-neutral; the printers in :mod:`tvbo.codegen.code` call them at codegen and the tvboptim observation layer uses them to declare a derived observation's axes.

It is also the one reader of a declared ``shape``: :func:`shape_terms` parses it, :func:`resolve_shape` sizes it, :func:`shape_code` writes it into generated code, :func:`dynamics_sizes` finds and checks the sizes a dynamics' shapes name, and :func:`per_node_dims` types a per-node equation by the element axes those shapes give its operands.
"""

from __future__ import annotations

import ast
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


NODE = "node"
"""The axis every per-node array carries, which the shape term ``n_nodes`` sizes."""

N_NODES = "n_nodes"
"""The shape term for the owning network's node count."""

REDUCTIONS = frozenset({"sum", "mean", "max", "min", "std", "var", "prod"})
"""The one-argument array functions that a per-node equation reads as a reduction over the element axis."""


def shape_terms(shape) -> tuple[int | str, ...] | None:
    """The terms of a declared shape: ``None`` when none is declared, else its integer literals and names in order.

    A shape is written as a Python tuple, ``"(n_nodes, n)"``, ``"(2, 2)"`` or ``"()"`` for a scalar; a single parenthesised term without its comma, ``"(n_nodes)"``, is read as the one-term tuple. Anything else is refused, naming the shape.
    """
    if shape is None or str(shape).strip() == "":
        return None
    text = str(shape).strip()
    try:
        body = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"shape {text!r} is not a Python tuple of integers and names, such as '(n_nodes, n)'") from exc
    terms: list[int | str] = []
    for element in body.elts if isinstance(body, ast.Tuple) else [body]:
        if isinstance(element, ast.Constant) and type(element.value) is int and element.value >= 0:
            terms.append(element.value)
        elif isinstance(element, ast.Name):
            terms.append(element.id)
        else:
            raise ValueError(
                f"shape {text!r}: a term is an integer, `n_nodes` or the name of a size, not {ast.unparse(element)!r}"
            )
    return tuple(terms)


def terms_code(terms, sizes: Mapping[str, int] | None = None) -> str:
    """Shape terms as the tuple a generated module writes, without spaces, each size replaced by its integer and ``n_nodes`` kept as the module's name for the node count."""
    parts = [str((sizes or {}).get(term, term)) for term in terms]
    return "(" + ",".join(parts) + ("," if len(parts) == 1 else "") + ")"


def shape_code(shape, sizes: Mapping[str, int] | None = None) -> str:
    """The declared shape written as :func:`terms_code` writes its terms."""
    return terms_code(shape_terms(shape) or (), sizes)


def resolve_shape(
    shape, *, owner: str, sizes: Mapping[str, int] | None = None, n_nodes: int | None = None
) -> tuple[int, ...] | None:
    """The shape's integer extents, ``None`` when none is declared.

    ``n_nodes`` takes *n_nodes*, a size its value in *sizes*. A term that is neither, or ``n_nodes`` where no node count is known, is refused, naming *owner*, the shape and the term.
    """
    terms = shape_terms(shape)
    if terms is None:
        return None
    sizes = sizes or {}
    extents = []
    for term in terms:
        if isinstance(term, int):
            extents.append(term)
        elif term == N_NODES and n_nodes is not None:
            extents.append(int(n_nodes))
        elif term in sizes:
            extents.append(int(sizes[term]))
        elif term == N_NODES:
            raise ValueError(f"{owner}: shape {shape!r} names `n_nodes`, and no network gives a node count here")
        else:
            raise ValueError(
                f"{owner}: shape {shape!r} names {term!r}, which is neither `n_nodes` nor a size; a shape term is an integer, `n_nodes`, or a parameter of the same dynamics that holds an integer"
            )
    return tuple(extents)


def shape_dims(shape) -> Dims:
    """The axis names a declared shape gives its array: ``node`` for ``n_nodes`` and ``element_<term>`` for an integer or a size."""
    return tuple(NODE if term == N_NODES else f"element_{term}" for term in shape_terms(shape) or ())


def dynamics_sizes(dynamics, experiment=None) -> dict[str, int]:
    """The sizes *dynamics* declares: each of its parameters that one of its shapes names, with its integer value.

    A shape here is a parameter's or a state variable's. A size fixes an array's extent when the code is compiled, so it must be a literal integer that nothing changes: one that is drawn, sourced, produced or not an integer is refused, and so, given *experiment*, is one that a fit, an update rule, a continuation or a sweep acts on, as :func:`tvbo.utils.parameters_acted_on` names them. Each refusal names the size and the parameter or state whose shape reads it.
    """
    from tvbo.utils import parameters_acted_on

    parameters = getattr(dynamics, "parameters", None) or {}
    owners = {
        **{f"parameter {name!r}": p for name, p in parameters.items()},
        **{f"state {name!r}": sv for name, sv in (getattr(dynamics, "state_variables", None) or {}).items()},
    }
    readers: dict[str, str] = {}
    for owner, element in owners.items():
        for term in shape_terms(getattr(element, "shape", None)) or ():
            if isinstance(term, str) and term in parameters:
                readers.setdefault(term, owner)
    acted = parameters_acted_on(experiment, dynamics) if experiment is not None else {}
    sizes = {}
    for name, reader in readers.items():
        parameter = parameters[name]
        value = getattr(parameter, "value", None)
        why = next(
            (
                reason
                for reason, applies in (
                    ("is drawn from a distribution", getattr(parameter, "distribution", None) is not None),
                    ("is sourced", any(getattr(parameter, slot, None) for slot in ("source", "measure", "used"))),
                    ("is produced", getattr(parameter, "producer", None) is not None),
                    (f"is {acted.get(name)}", name in acted),
                    (
                        f"holds {value!r}, not an integer",
                        not (isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer()),
                    ),
                )
                if applies
            ),
            None,
        )
        if why is not None:
            raise ValueError(
                f"size {name!r}, which the shape of {reader} names, {why}; a size fixes an array's extent when the code is compiled, so it is a literal integer"
            )
        sizes[name] = int(value)
    return sizes


def per_node_dims(expr, env: Mapping[str, Dims]) -> Dims:
    """The element axes *expr* carries in a per-node equation, given *env*, each symbol's element axes (``()`` for a scalar per node).

    The node axis is the equation's own batch axis, so it never appears here: a per-region readout of shape ``(n_nodes, n)`` reads as ``(element_n,)`` at its node. ``matmul`` contracts the left operand's last axis with the right one's first, which must be the same axis; ``sum``, ``mean``, ``max``, ``min``, ``std``, ``var`` and ``prod`` of one argument reduce its element axis; ``+ - * /`` and every other function combine operands of equal axes or a scalar. A mismatch, an index into an element and a symbol *env* does not know are refused, naming the operands.
    """

    def d(node) -> Dims:
        if isinstance(node, sp.Symbol):
            if str(node) not in env:
                raise ValueError(f"{expr}: {str(node)!r} has no declared axes here")
            return tuple(env[str(node)])
        if getattr(node, "is_Number", False) or isinstance(node, sp.NumberSymbol) or node in (sp.true, sp.false):
            return ()
        if isinstance(node, sp.Indexed):
            raise ValueError(f"{expr}: indexing an element, {node}, is not read in a per-node equation")
        name = node.func.__name__ if isinstance(node, sp.Function) else None
        if name == "matmul":
            a, b = d(node.args[0]), d(node.args[1])
            if not a or not b or a[-1] != b[0]:
                raise ValueError(f"{node}: {node.args[0]} {list(a)} @ {node.args[1]} {list(b)} contracts no shared axis")
            return a[:-1] + b[1:]
        if name in REDUCTIONS and len(node.args) == 1:
            d(node.args[0])
            return ()
        if not node.args:
            raise ValueError(f"{expr}: {node} has no declared axes here")
        parts = [(arg, d(arg)) for arg in node.args]
        shaped = [(arg, dims) for arg, dims in parts if dims]
        for arg, dims in shaped[1:]:
            if dims != shaped[0][1]:
                raise ValueError(
                    f"{node}: {shaped[0][0]} {list(shaped[0][1])} and {arg} {list(dims)} have different element axes"
                )
        return shaped[0][1] if shaped else ()

    return d(expr)
