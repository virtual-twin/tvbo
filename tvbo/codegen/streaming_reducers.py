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

import importlib
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

# factory string -> whether it resolves in the current environment (memoised).
_FACTORY_AVAILABLE: dict[str, bool] = {}


def _factory_available(factory: str) -> bool:
    """True when the ``module.attr`` *factory* resolves in the current env.

    Streaming reducers live in ``tvboptim``; a tvboptim older than the reducer
    it names (e.g. a released wheel that predates ``windowed_cov``) won't expose
    it. Codegen runs in the same interpreter that executes the generated
    experiment, so resolving the factory here reliably predicts whether the
    emitted call would work — and lets an unresolved factory fall back to the
    recompute path (supported on every backend) instead of emitting a call that
    raises ``AttributeError`` at runtime.
    """
    cached = _FACTORY_AVAILABLE.get(factory)
    if cached is not None:
        return cached
    module_name, _, attr = factory.rpartition(".")
    available = False
    if module_name and attr:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            available = False
        else:
            available = hasattr(module, attr)
    _FACTORY_AVAILABLE[factory] = available
    return available


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
    for this reducer, when the registered factory does not resolve in the
    installed backend package (e.g. a tvboptim that predates the reducer), or
    when streaming is disabled via the ``TVBO_STREAMING_REDUCERS=0`` environment
    switch (used to codegen the recompute reference for validation and as an
    escape hatch).
    """
    if os.environ.get("TVBO_STREAMING_REDUCERS", "1") == "0":
        return None
    backend_map = _REGISTRY.get(backend)
    if not backend_map:
        return None
    qualified = f"{reducer_module}.{reducer_name}" if reducer_module else reducer_name
    spec = backend_map.get(qualified) or backend_map.get(reducer_name)
    if spec is None or not _factory_available(spec.factory):
        return None
    return spec


def streaming_capable(backend: str, reducer_module: str | None, reducer_name: str) -> bool:
    """True when a step-wise (``"window"``) streaming reducer is registered.

    The algorithm template gates its streaming branch on this; ``"stride"``
    (dFC / FCD) specs return ``False`` here until the template's stride branch
    lands, so registering them is safe (recompute fallback stays in force).
    """
    spec = lookup_streaming_reducer(backend, reducer_module, reducer_name)
    return spec is not None and spec.emit_kind == "window"


def is_windowed_reducer(reducer_module: str | None, reducer_name: str, backend: str = "tvboptim") -> bool:
    """True when a pipeline reducer is a registered windowed reducer for *backend*.

    The registered reducers (``compute_fc`` / ``compute_fcd`` for ``tvboptim``) are
    windowed correlation reductions — undefined over a ``< 2``-sample window, where
    ``jnp.corrcoef`` collapses to a scalar — so codegen routes them through a
    degenerate-window guard. Unlike :func:`lookup_streaming_reducer`, this is a pure
    registry-membership test with no factory-availability or
    ``TVBO_STREAMING_REDUCERS`` gating: the guard applies whether or not the
    streaming *path* is active. It shares the one registered FC-family set so the
    guard's routing and the streaming lookup cannot drift.
    """
    backend_map = _REGISTRY.get(backend)
    if not backend_map:
        return False
    qualified = f"{reducer_module}.{reducer_name}" if reducer_module else reducer_name
    return qualified in backend_map or reducer_name in backend_map


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
