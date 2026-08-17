"""Codegen registry: pipeline reducer callable -> backend streaming reducer.

A *streaming reducer* replaces the ``O(window * N^2)``/step rolled-buffer recompute of a sliding-window source observation (e.g. ``compute_fc`` recomputed over the whole BOLD window every online-tuning iteration) with an incremental ``O(N^2)``/step reducer over the ``(add, evict, emit, resync)`` protocol. The reducer is authored declaratively (:class:`StreamingReducerSpec`) and lowered to backend source by :mod:`tvbo.codegen.reducers`, so tvbo emits the realization into the generated experiment — no backend ships one.

The registry is **per backend**: a backend that has a registered streaming form for a given pipeline reducer uses the streaming path in generated code; a backend that registers nothing (``tvb`` / ``pyrates`` / ``julia``) keeps the recompute path byte-for-byte unchanged. The whole mechanism is transparent to the recipe / schema — a study keeps declaring ``pipeline: [compute_fc]`` and the algorithm codegen swaps in the streaming reducer only where one is registered.

Registering a new streaming reducer (e.g. for dFC / metastability) is a new ``database/reducers/*.yaml`` recipe — state + add/evict/resync/emit assignment strings — loaded at import; no code change.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingReducerSpec:
    """Declarative, backend-agnostic spec of a windowed streaming reducer (metadata).

    The reducer's math is authored as symbolic **assignment recurrences** (strings over the state vars, the arriving/leaving sample ``v``, and the window ``x``); the general resolver in :mod:`tvbo.codegen.reducers` parses them and the sympy printers lower them per backend, so tvbo emits the reducer into the generated experiment and no backend ships a reducer realization (works on any backend/tvboptim version — nothing external to resolve). **Sequential reassignment encodes the data flow** — a later RHS sees the value assigned to a state var above it — so no ``new_*`` temporaries are needed. The recipe is data here; the template that emits it is fully general.

    Attributes:
        state: accumulator state variables, unpacked from ``acc`` and returned.
        add: ordered ``(lhs, rhs)`` assignments folding one arriving sample ``v``.
        evict: ordered assignments dropping one leaving sample ``v`` (sliding window).
        resync: ordered assignments rebuilding the state exactly from the window ``x``.
        resync_masked: optional assignments rebuilding the state from a max-sized ring
            buffer whose valid window is the last ``L`` rows, selected by the boolean
            mask ``m`` (row-vector over the full buffer) with count ``L``. Present so a
            reducer can be resynced over a fixed-shape buffer with a traced window
            length — the tuning scan then compiles once across varying window sizes.
            Empty when the reducer has no masked form (the masked path is not offered).
        emit: readout expression over the final state (the reduced value).
        emit_kind: ``"window"`` (emit every step; wired today) or ``"stride"`` (emit a
            per-window observation for a downstream reduction — the dFC / FCD
            follow-on; :func:`streaming_capable` returns ``False`` for it until the
            template's stride branch lands, recompute fallback meanwhile).
    """

    state: tuple
    add: tuple
    evict: tuple
    resync: tuple
    emit: str
    resync_masked: tuple = ()
    emit_kind: str = "window"


# backend -> {fully_qualified_reducer_name -> StreamingReducerSpec}
_REGISTRY: dict[str, dict[str, StreamingReducerSpec]] = {}


def register_streaming_reducer(backend: str, reducer: str, spec: StreamingReducerSpec) -> None:
    """Register *spec* as *backend*'s streaming form of the *reducer* callable.

    Args:
        backend: Codegen backend key (e.g. ``"tvboptim"``).
        reducer: Fully-qualified pipeline reducer callable the spec replaces
            (e.g. ``"tvboptim.observations.observation.compute_fc"``). The bare
            callable name is also matched as a fallback.
        spec: The streaming reducer descriptor.
    """
    _REGISTRY.setdefault(backend, {})[reducer] = spec


def lookup_streaming_reducer(backend: str, reducer_module: str | None, reducer_name: str):
    """Return the :class:`StreamingReducerSpec` for a pipeline reducer, or ``None``.

    Returns ``None`` (-> recompute fallback) when the backend registers nothing for this reducer, or when streaming is disabled via the ``TVBO_STREAMING_REDUCERS=0`` environment switch (used to codegen the recompute reference for validation and as an escape hatch). The reducer realization is tvbo-emitted, so there is no external factory to resolve.
    """
    if os.environ.get("TVBO_STREAMING_REDUCERS", "1") == "0":
        return None
    backend_map = _REGISTRY.get(backend)
    if not backend_map:
        return None
    qualified = f"{reducer_module}.{reducer_name}" if reducer_module else reducer_name
    return backend_map.get(qualified) or backend_map.get(reducer_name)


def streaming_capable(backend: str, reducer_module: str | None, reducer_name: str) -> bool:
    """True when a step-wise (``"window"``) streaming reducer is registered.

    The algorithm template gates its streaming branch on this; ``"stride"`` (dFC / FCD) specs return ``False`` here until the template's stride branch lands, so registering them is safe (recompute fallback stays in force).
    """
    spec = lookup_streaming_reducer(backend, reducer_module, reducer_name)
    return spec is not None and spec.emit_kind == "window"


def is_windowed_reducer(reducer_module: str | None, reducer_name: str, backend: str = "tvboptim") -> bool:
    """True when a pipeline reducer is a registered windowed reducer for *backend*.

    The registered reducer (``compute_fc`` for ``tvboptim``) is a windowed correlation reduction — undefined over a ``< 2``-sample window, where ``jnp.corrcoef`` collapses to a scalar — so codegen routes it through a degenerate-window guard. Unlike :func:`lookup_streaming_reducer`, this is a pure registry-membership test with no factory-availability or ``TVBO_STREAMING_REDUCERS`` gating: the guard applies whether or not the streaming *path* is active. It shares the one registered reducer set so the guard's routing and the streaming lookup cannot drift.
    """
    backend_map = _REGISTRY.get(backend)
    if not backend_map:
        return False
    qualified = f"{reducer_module}.{reducer_name}" if reducer_module else reducer_name
    return qualified in backend_map or reducer_name in backend_map


# Recipes are data in `tvbo/database/reducers/*.yaml`; a backend registering none recomputes instead.
_REDUCERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "database" / "reducers"


def _spec_from_metadata(meta: dict) -> StreamingReducerSpec:
    """Build a :class:`StreamingReducerSpec` from a parsed reducer YAML."""

    def _assigns(key):
        return tuple((str(lhs), str(rhs)) for lhs, rhs in meta.get(key, []))

    return StreamingReducerSpec(
        state=tuple(str(s) for s in meta["state"]),
        add=_assigns("add"),
        evict=_assigns("evict"),
        resync=_assigns("resync"),
        resync_masked=_assigns("resync_masked"),
        emit=str(meta["emit"]),
        emit_kind=str(meta.get("emit_kind", "window")),
    )


def _load_reducer_recipes() -> None:
    """Register every ``database/reducers/*.yaml`` recipe for its declared backends."""
    import yaml

    for yf in sorted(_REDUCERS_DIR.glob("*.yaml")):
        meta = yaml.safe_load(yf.read_text())
        spec = _spec_from_metadata(meta)
        for backend in meta.get("backends", []):
            for reducer_ref in meta.get("replaces", []):
                register_streaming_reducer(str(backend), str(reducer_ref), spec)


_load_reducer_recipes()
