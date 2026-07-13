"""Codegen registry: pipeline reducer callable -> backend streaming reducer.

A *streaming reducer* replaces the ``O(window * N^2)``/step rolled-buffer
recompute of a sliding-window source observation (e.g. ``compute_fc`` recomputed
over the whole BOLD window every online-tuning iteration) with an incremental
``O(N^2)``/step reducer exposing the ``(init, add, evict, emit, finalize,
resync)`` protocol (see ``tvboptim.observations.observation.StreamingReducer``).

The registry is **per backend**: a backend that has a registered streaming form
for a given pipeline reducer uses the streaming path in generated code; a backend
that registers nothing (``tvb`` / ``pyrates`` / ``julia``) keeps the recompute
path byte-for-byte unchanged. The whole mechanism is transparent to the recipe /
schema — a study keeps declaring ``pipeline: [compute_fc]`` and the algorithm
codegen swaps in the streaming reducer only where one is registered.

Registering a new streaming reducer (e.g. for dFC / metastability) is a single
:func:`register_streaming_reducer` call — no recipe change. See
``dev/streaming_reducers.md`` for the walkthrough.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingReducerSpec:
    """How a backend streams a windowed pipeline reducer.

    Attributes:
        factory: Fully-qualified factory that returns a ``StreamingReducer``
            (``init``/``add``/``evict``/``emit``/``finalize``/``resync``). Called
            in generated code as ``factory(s_var=<s_var>, skip_t=<skip_t>)``, so
            it must be importable in the emitted module (the tvboptim reducers
            live under ``tvboptim.observations.observation``, already imported by
            the generated experiment).
        emit_kind: ``"window"`` — emit the reduced value every step (the FC case;
            the algorithm template's streaming branch supports this today).
            ``"stride"`` — emit a per-window observation every ``stride`` steps
            into a downstream reduction (the dFC / FCD / metastability extension
            point). The registration hook + reducer exist for ``"stride"``; the
            template's stride branch is the documented follow-on, so
            :func:`streaming_capable` returns ``False`` for it until that lands
            (the recompute fallback stays in force meanwhile).
    """

    factory: str
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

    Returns ``None`` (-> recompute fallback) when the backend registers nothing
    for this reducer, or when streaming is disabled via the
    ``TVBO_STREAMING_REDUCERS=0`` environment switch (used to codegen the
    recompute reference for validation and as an escape hatch).
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

    The algorithm template gates its streaming branch on this; ``"stride"``
    (dFC / FCD) specs return ``False`` here until the template's stride branch
    lands, so registering them is safe (recompute fallback stays in force).
    """
    spec = lookup_streaming_reducer(backend, reducer_module, reducer_name)
    return spec is not None and spec.emit_kind == "window"


# ── Built-in registrations ────────────────────────────────────────────────────
# tvboptim / JAX: windowed Pearson-correlation FC — fully implemented + validated.
register_streaming_reducer(
    "tvboptim",
    "tvboptim.observations.observation.compute_fc",
    StreamingReducerSpec(
        factory="tvboptim.observations.observation.windowed_cov",
        emit_kind="window",
    ),
)
# Extension hook: dFC / FCD / metastability. The reducer exists
# (``windowed_fcd``); the template's ``"stride"`` emit branch is the documented
# follow-on (see dev/streaming_reducers.md). Registered so it is reachable, but
# ``streaming_capable`` reports False for it -> recompute fallback stays in force.
register_streaming_reducer(
    "tvboptim",
    "tvboptim.observations.observation.compute_fcd",
    StreamingReducerSpec(
        factory="tvboptim.observations.observation.windowed_fcd",
        emit_kind="stride",
    ),
)
# Other backends (tvb / pyrates / julia) register nothing -> recompute fallback.
