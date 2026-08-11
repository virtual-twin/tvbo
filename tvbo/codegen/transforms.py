"""The matrix-transform vocabulary, declared once for the runtime and for codegen.

A `transforms:` entry is a `Function`, so it is either equation-based or callable-based, and
a symbolic one is written against a small vocabulary of primitives — `W`, `W_max`,
`W_rowsum_safe`, `L`, plus the network's per-node parameter vectors. Both the runtime
evaluator (`Network._apply_transform`) and the emitters that inline a transform into a
generated script need that vocabulary, and a second hand-written copy of it drifts: the
runtime is where a transform author adds a primitive, so the emitted kit is the side that
silently goes wrong.

Each primitive is therefore declared once, as a source expression over two base names —
`_M`, the matrix being transformed, and `_L`, the lengths. The runtime evaluates those
strings; an emitter prints them. Neither can define a primitive the other lacks.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

BASE_MATRIX = "_M"
BASE_LENGTHS = "_L"

PRELUDE: Tuple[Tuple[str, str], ...] = (
    ("_rs", f"{BASE_MATRIX}.sum(axis=1, keepdims=True)"),
    ("_cs", f"{BASE_MATRIX}.sum(axis=0, keepdims=True)"),
)

PRIMITIVES: Dict[str, str] = {
    "M": BASE_MATRIX,
    "W": BASE_MATRIX,
    "M_min": f"jnp.nanmin({BASE_MATRIX})",
    "W_min": f"jnp.nanmin({BASE_MATRIX})",
    "M_max": f"jnp.nanmax({BASE_MATRIX})",
    "W_max": f"jnp.nanmax({BASE_MATRIX})",
    "W_rowsum": "_rs",
    "W_colsum": "_cs",
    "W_rowsum_safe": "jnp.where(_rs > 0, _rs, 1.0)",
    "W_colsum_safe": "jnp.where(_cs > 0, _cs, 1.0)",
    "L": BASE_LENGTHS,
}

DATA_DERIVED = frozenset({"L"})
"""Primitives that do not depend on the matrix under transform.

An emitter binds these once; everything else is rebound per transform so a chain of
transforms sees the preceding one's output.
"""


MASKABLE_REDUCTIONS = frozenset({"mean", "sum"})
"""Reductions a declared `mask:` rewrites.

`min`/`max` are deliberately absent: masking them needs a ±infinity fill whose spelling
is backend-specific, and no recipe asks for it yet. A masked `min(...)` raises rather
than silently reducing over the unmasked matrix — the failure this whole mechanism
exists to prevent.
"""


def mask_reductions(expr, mask):
    """Rewrite every reduction in *expr* to reduce over *mask* only.

    `mean(W)` becomes `sum(where(mask, W, 0)) / sum(where(mask, 1, 0))` and `sum(W)`
    becomes `sum(where(mask, W, 0))`, built as sympy `Piecewise` so every backend prints
    it through its own `where` primitive rather than a numpy-shaped string.

    The rewrite is what makes a mask expressible at all: `mean(W[W > 0])` is boolean
    indexing, whose output shape depends on the data, so it is illegal under `jax.jit`
    and silently loses the mask through a printer that drops the subscript.

    Args:
        expr: The transform expression.
        mask: A sympy boolean expression over the same primitives, e.g. `W > 0`.

    Returns:
        *expr* with its reductions masked.

    Raises:
        NotImplementedError: A reduction outside `MASKABLE_REDUCTIONS` appears under a mask.
    """
    from sympy import Function, Integer, Piecewise, preorder_traversal

    counted = Function("sum")(Piecewise((Integer(1), mask), (Integer(0), True)))
    replacements = {}
    for node in preorder_traversal(expr):
        head = getattr(getattr(node, "func", None), "__name__", None)
        if head is None or not node.args:
            continue
        if head in MASKABLE_REDUCTIONS:
            kept = Function("sum")(Piecewise((node.args[0], mask), (Integer(0), True)))
            replacements[node] = kept / counted if head == "mean" else kept
        elif head in ("min", "max", "nanmin", "nanmax"):
            raise NotImplementedError(
                f"`mask:` cannot scope `{head}(...)` yet — it needs a backend-specific "
                f"infinity fill. Drop the mask, or reduce with mean/sum."
            )
    return expr.subs(replacements, simultaneous=True) if replacements else expr


def required_prelude(symbols: Iterable[str]) -> List[Tuple[str, str]]:
    """The prelude bindings *symbols* need, in declaration order.

    Dependencies are matched as whole identifiers, never as substrings: a primitive
    whose expression merely contains the text of a prelude name (``_rsq``, a name
    inside a string) must not drag that binding in, and a binding that is genuinely
    referenced must not be missed.
    """
    wanted = {s for s in symbols if s in PRIMITIVES}
    referenced: set[str] = set()
    for s in wanted:
        referenced |= set(_IDENTIFIER.findall(PRIMITIVES[s]))
    return [(name, expr) for name, expr in PRELUDE if name in referenced]


def runtime_env(matrix, lengths, jnp, jsp=None) -> Dict[str, object]:
    """Evaluate every primitive against live arrays.

    Args:
        matrix: The matrix under transform, bound to `M`/`W`.
        lengths: The network's length matrix, bound to `L`.
        jnp: The array module the primitive expressions are written against.
        jsp: Optional scipy namespace, exposed to transform equations that use it.

    Returns:
        Mapping of every primitive name to its value, plus the array modules.
    """
    scope: Dict[str, object] = {BASE_MATRIX: matrix, BASE_LENGTHS: lengths, "jnp": jnp}
    for name, expr in PRELUDE:
        scope[name] = eval(expr, dict(scope))
    env = {name: eval(expr, dict(scope)) for name, expr in PRIMITIVES.items()}
    env.update(jnp=jnp, np=jnp, jsp=jsp)
    return env


def emit_env(symbols: Sequence[str], matrix: str, lengths: str) -> Tuple[List[str], List[str]]:
    """Source lines binding the primitives *symbols* uses, for an emitted script.

    Args:
        symbols: Free symbols of the transform expression; anything outside
            `PRIMITIVES` is ignored here and handled by the caller.
        matrix: Expression the emitted code calls the matrix under transform.
        lengths: Expression the emitted code calls the length matrix.

    Returns:
        A `(matrix_lines, data_lines)` pair. `matrix_lines` must be re-emitted for
        each transform in a chain; `data_lines` are bound once.
    """
    used = [s for s in dict.fromkeys(symbols) if s in PRIMITIVES]
    if not used:
        return [], []

    def _bind(expr: str) -> str:
        return expr.replace(BASE_MATRIX, matrix).replace(BASE_LENGTHS, lengths)

    matrix_lines = [f"{name} = {_bind(expr)}" for name, expr in required_prelude(used)]
    data_lines: List[str] = []
    for s in used:
        line = f"{s} = {_bind(PRIMITIVES[s])}"
        (data_lines if s in DATA_DERIVED else matrix_lines).append(line)
    return matrix_lines, data_lines
