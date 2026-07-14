"""General resolver: lower a declarative streaming-reducer spec to backend source.

A :class:`~tvbo.codegen.streaming_reducers.StreamingReducerSpec` authors a windowed
reducer as symbolic **assignment recurrences** (strings). This module parses each RHS
to sympy against the reducer's vocabulary and prints it with the target
:mod:`tvbo.codegen.code` printer, so the SAME spec emits jax / numpy / julia. There is
**no reducer-specific logic here** — the concrete recipe lives in the spec (metadata);
the algorithm template emits the resolved assignments into a backend-shaped scaffolding.
Verified byte-identical to ``compute_fc`` including a large DC baseline
(``scratchpad/proto_general.py``).
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from tvbo.codegen.code import get_printer
from tvbo.parse.expression import ARRAY_FUNCTIONS


def resolve_streaming_reducer(spec: Any, fmt: str = "jax") -> dict:
    """Lower a declarative reducer *spec* to printed per-assignment source for *fmt*.

    Returns ``{"state": [...], "add": [(lhs, rhs_src), ...], "evict": ..., "resync": ...,
    "emit": rhs_src}``. The reducer's symbolic vocabulary is the state variables plus the
    per-step sample ``v`` and the window ``x``; each assignment's ``lhs`` joins the
    vocabulary so later RHS may reference it (sequential data flow). State-var symbols win
    over same-named array functions (e.g. a state ``mean`` shadows the ``mean`` reducer).
    """
    printer = get_printer(fmt).doprint
    vocab = {
        **ARRAY_FUNCTIONS,
        "pi": sp.pi,
        **{name: sp.Symbol(name) for name in (*spec.state, "v", "x")},
    }

    def _print(rhs: str) -> str:
        return printer(sp.sympify(str(rhs), locals=vocab))

    def _block(assignments) -> list:
        out = []
        for lhs, rhs in assignments:
            out.append((lhs, _print(rhs)))
            vocab[lhs] = sp.Symbol(lhs)  # available to subsequent RHS
        return out

    return {
        "state": list(spec.state),
        "add": _block(spec.add),
        "evict": _block(spec.evict),
        "resync": _block(spec.resync),
        "emit": _print(spec.emit),
    }
