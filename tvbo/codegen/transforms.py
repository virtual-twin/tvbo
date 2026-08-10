"""The matrix-transform vocabulary, declared once for the runtime and for codegen.

A `transforms:` entry is a `Function`, so it is either equation-based or callable-based, and a symbolic one is written against a small vocabulary of primitives — `W`, `W_max`,
`W_rowsum_safe`, `L`, plus the network's per-node parameter vectors. Both the runtime evaluator (`Network._apply_transform`) and the emitters that inline a transform into a
generated script need that vocabulary, and a second hand-written copy of it drifts: the runtime is where a transform author adds a primitive, so the emitted kit is the side that
silently goes wrong.

Each primitive is therefore declared once, as a source expression over two base names —
`_M`, the matrix being transformed, and `_L`, the lengths. The runtime evaluates those strings; an emitter prints them. Neither can define a primitive the other lacks.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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


def required_prelude(symbols: Iterable[str]) -> List[Tuple[str, str]]:
    """The prelude bindings *symbols* transitively need, in declaration order."""
    wanted = {s for s in symbols if s in PRIMITIVES}
    exprs = " ".join(PRIMITIVES[s] for s in wanted)
    return [(name, expr) for name, expr in PRELUDE if name in exprs]


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
