"""Transform vocabulary: the network's own edge attributes, and masked reductions.

A ``transforms:`` entry is a ``Function`` whose equation is written over the network's
edge attributes — ``weight``, ``length``, or the canonical ``network.edges.<label>`` —
resolved by the same :func:`tvbo.utils.edge_label` that observation sources and
exploration axes go through. There is no second, invented vocabulary: a derived quantity
is spelled as the reduction it is (``max(weight)``), so nothing has to be declared twice
and no backend can be handed a name the runtime never defined.

A reduction may be scoped by a boolean mask, in either of two spellings, because the
notation people reach for differs and both are unambiguous:

.. code-block:: yaml

    rhs: "weight / mean(weight[weight > 0])"    # the boolean subscript
    rhs: "weight / mean(weight, weight > 0)"    # the predicate as an argument

Both normalise to one node, ``red(expr, predicate)``, lowered once into ``Piecewise``.
Each printer already turns that into its own ``where``/``ifelse``, so the mask is
backend-independent for free and no two backends can disagree about what it means. A
boolean subscript is only legal *inside* a reduction: on its own it has a data-dependent
output shape, so it cannot be jitted and is rejected.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from sympy import Basic, Function, Indexed, IndexedBase, Integer, Piecewise, oo, preorder_traversal
from sympy.logic.boolalg import Boolean

_SUBSCRIPTED = re.compile(r"([A-Za-z_]\w*)\s*\[")

REDUCTIONS: Dict[str, Basic] = {
    "sum": Integer(0),
    "mean": Integer(0),
    "min": oo,
    "nanmin": oo,
    "max": -oo,
    "nanmax": -oo,
}
"""Reduction head to the value a masked-out entry contributes.

``sum`` and ``mean`` fill with zero. An extremum fills with its identity, so a mask that
keeps nothing yields ±inf and poisons the result loudly, rather than returning the
plausible-looking extremum of the entries it was told to ignore. ``mean`` additionally
divides by the number of kept entries, which is the whole reason an unmasked
``mean(weight)`` over a sparse connectome is not the mean of its edges.
"""


def subscript_locals(source: str) -> Dict[str, IndexedBase]:
    """An ``IndexedBase`` for every name *source* subscripts, to hand the parser.

    ``parse_expr`` builds a plain ``Symbol`` for a name it has not been given, and a
    ``Symbol`` is not subscriptable — so ``mean(weight[weight > 0])`` would die with
    ``'Symbol' object is not subscriptable`` before anything could read the mask.
    """
    return {name: IndexedBase(name) for name in set(_SUBSCRIPTED.findall(source or ""))}


def _plain(expr):
    """*expr* with every ``IndexedBase`` collapsed to its bare symbol.

    ``parse_expr`` builds ``IndexedBase`` for any name it sees subscripted, including in
    the predicate, so the same edge attribute would otherwise reach the printer as two
    different objects depending on where it appeared.
    """
    bases = {b: b.label for b in expr.atoms(IndexedBase)}
    return expr.xreplace(bases) if bases else expr


def _split_mask(node):
    """A reduction node's ``(operand, predicate)``, for either inline spelling.

    Returns ``(operand, None)`` when the reduction declares no mask of its own.
    """
    args = node.args
    if len(args) == 2 and isinstance(args[1], Boolean):
        return args[0], args[1]
    if len(args) == 1 and isinstance(args[0], Indexed):
        indices = args[0].indices
        if len(indices) == 1 and isinstance(indices[0], Boolean):
            return args[0].base.label, indices[0]
    return args[0] if args else None, None


def canonical_reductions(expr):
    """Rewrite either mask spelling into the canonical ``red(operand, predicate)``."""
    replacements = {}
    for node in preorder_traversal(expr):
        head = getattr(getattr(node, "func", None), "__name__", None)
        if head not in REDUCTIONS or not node.args:
            continue
        operand, mask = _split_mask(node)
        if mask is None:
            continue
        replacements[node] = Function(head)(operand, mask)
    return expr.xreplace(replacements) if replacements else expr


def lower_reductions(expr):
    """Lower canonical masked reductions to ``Piecewise``, which every printer handles.

    ``mean`` becomes a kept-sum over a kept-count rather than a masked ``mean``, because
    an array library's ``mean`` divides by the full size no matter what it was handed.
    """
    replacements = {}
    for node in preorder_traversal(expr):
        head = getattr(getattr(node, "func", None), "__name__", None)
        if head not in REDUCTIONS or len(node.args) != 2 or not isinstance(node.args[1], Boolean):
            continue
        operand, mask = node.args
        fill = REDUCTIONS[head]
        kept = Piecewise((operand, mask), (fill, True))
        if head == "mean":
            counted = Function("sum")(Piecewise((Integer(1), mask), (Integer(0), True)))
            replacements[node] = Function("sum")(kept) / counted
        else:
            replacements[node] = Function(head)(kept)
    return expr.xreplace(replacements) if replacements else expr


def prepare(expr, what: str = "transform"):
    """Normalise, lower and validate a transform expression. The one entry point.

    Both the runtime and every emitter go through this, so a mask cannot mean one thing
    when evaluated and another when printed.

    Args:
        expr: The parsed transform expression.
        what: How to name the transform in an error.

    Returns:
        The lowered expression, ready for :func:`tvbo.codegen.code.render_expression`.

    Raises:
        ValueError: A boolean subscript survived outside a reduction. Its output shape
            depends on the data, so there is nothing static to emit.
    """
    lowered = _plain(lower_reductions(canonical_reductions(_plain(expr))))
    for node in preorder_traversal(lowered):
        if isinstance(node, Indexed) and any(isinstance(i, Boolean) for i in node.indices):
            raise ValueError(
                f"{what} subscripts {node.base} with a boolean outside a reduction. "
                f"A boolean subscript selects a data-dependent number of entries, so it has "
                f"no static shape to emit; only a reduction over it does. Write the reduction "
                f"explicitly, e.g. `mean({node.base}[{node.indices[0]}])`."
            )
    return lowered


def edge_symbols(expr) -> List[str]:
    """Names *expr* references, in sorted order, for the caller to resolve as edges."""
    return sorted({str(s) for s in _plain(expr).free_symbols})


def runtime_env(resolve, symbols: Sequence[str], jnp, jsp=None) -> Dict[str, object]:
    """Bind every symbol *expr* names to a live array.

    Args:
        resolve: Callable mapping an edge-attribute name to its matrix, or None.
        symbols: The names to bind, from :func:`edge_symbols`.
        jnp: The array module the lowered expression is evaluated against.
        jsp: Optional scipy namespace, for a transform equation that uses one.

    Returns:
        Mapping of each resolvable name to its array, plus the array modules.
    """
    env: Dict[str, object] = {}
    for name in symbols:
        value = resolve(name)
        if value is not None:
            env[name] = value
    env.update(jnp=jnp, np=jnp, jsp=jsp)
    return env


def emit_env(
    symbols: Sequence[str], resolve, target: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """Source lines binding the edge attributes *symbols* names, for an emitted script.

    Args:
        symbols: The names to bind, from :func:`edge_symbols`.
        resolve: Callable mapping an edge-attribute name to the expression the emitted
            code calls it, or None for a name that is not an edge attribute.
        target: The transform's own target. That attribute binds to the value flowing
            through the chain, so a second transform sees the first one's output; every
            other attribute binds once to the network's stored matrix.

    Returns:
        A ``(chained_lines, constant_lines)`` pair. ``chained_lines`` are re-emitted for
        each transform in a chain; ``constant_lines`` are bound once.
    """
    from tvbo.utils import edge_label

    target_label = edge_label(target) or target
    chained: List[str] = []
    constant: List[str] = []
    for name in dict.fromkeys(symbols):
        source = resolve(name)
        if source is None:
            continue
        label = edge_label(name) or name
        (chained if label == target_label else constant).append(f"{name} = {source}")
    return chained, constant
