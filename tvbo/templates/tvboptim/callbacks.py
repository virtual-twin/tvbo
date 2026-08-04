# Copyright Berlin Institute of Health / Charité University Medicine Berlin
# Department of Neurology and Experimental Neurology
# Brain Simulation Section

"""Runtime callbacks for generated tvboptim scripts.

Imported by the generated experiment/optimization scripts (which always run with
tvboptim available), so this module may depend on tvboptim. It routes optimizer
progress through the central ``tvbo.run`` logger (see :mod:`tvbo.log`), so one
switch — ``TVBO_LOG_LEVEL`` / ``tvbo.set_log_level`` / the CLI ``--quiet`` —
governs it exactly as it governs the rest of a run.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from tvboptim.optim.callbacks import AbstractCallback

logger = logging.getLogger("tvbo.run")


# n_parallel:auto bounds (see resolve_n_vmap / estimate_per_cell_bytes); explicit int bypasses.
AUTO_NVMAP_CAP = 64
AUTO_NVMAP_MEM_BUDGET_GB = 2.0


def _env_positive(name, cast, default):
    """Read env var ``name`` as ``cast`` if it is a positive value, else ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = cast(raw)
    except (ValueError, TypeError):
        val = 0
    if val > 0:
        return val
    logger.warning("Ignoring %s=%r (want a positive number); using default %s.", name, raw, default)
    return default


def auto_nvmap_cap() -> int:
    """Cell-count cap for ``n_parallel: auto``, overridable via ``TVBO_NVMAP_AUTO_CAP``.

    Read at call time so the environment variable takes effect per run. Falls back to
    :data:`AUTO_NVMAP_CAP` if the variable is unset or not a positive integer.
    """
    return int(_env_positive("TVBO_NVMAP_AUTO_CAP", int, AUTO_NVMAP_CAP))


def auto_nvmap_budget_bytes() -> int:
    """Batch working-memory budget for ``n_parallel: auto``, in bytes.

    Overridable via ``TVBO_NVMAP_MEM_BUDGET_GB`` (default :data:`AUTO_NVMAP_MEM_BUDGET_GB`).
    Bounds the live batch (``shared_ram_devices × n_vmap × per-cell-bytes``) so
    auto-vectorisation cannot blow up peak memory on a large-per-cell model.
    """
    return int(_env_positive("TVBO_NVMAP_MEM_BUDGET_GB", float, AUTO_NVMAP_MEM_BUDGET_GB) * (1024 ** 3))


def shared_ram_device_count() -> int:
    """Number of devices whose per-cell batches share one physical RAM pool.

    CPU host-replication (``xla_force_host_platform_device_count``) fans one host's RAM
    across N logical devices, so a pmapped batch holds ``N × n_vmap`` cells in the *same*
    RAM. Real GPU/TPU devices each have independent memory (only ``n_vmap`` cells apiece),
    so returns 1 there. Scales the ``n_parallel: auto`` memory budget (see
    :func:`resolve_exploration_n_vmap`). Returns 1 if JAX or the device list is unavailable.
    """
    try:
        import jax

        devices = jax.devices()
        if devices and all(getattr(d, "platform", None) == "cpu" for d in devices):
            return len(devices)
    except Exception:
        pass
    return 1


def estimate_per_cell_bytes(observable_fn, state) -> Optional[int]:
    """Best-effort per-cell peak working-memory estimate for ``n_parallel: auto``.

    Compiles the single-cell observable ahead-of-time and reads XLA's memory analysis
    (``temp + output + argument``), so the estimate includes the transient buffers the
    observable allocates and then reduces away — e.g. a BOLD trajectory and its FFT
    convolution behind a scalar loss. Summing only the output (as ``eval_shape`` would)
    under-counts such reduction observables, so a vmapped batch of them silently OOMs.
    Falls back to an output+input shape sum, then ``None``, so the caller degrades to
    the count-only cap.
    """
    try:
        import jax
    except Exception:
        return None

    try:
        stats = jax.jit(observable_fn).lower(state).compile().memory_analysis()
        if stats is not None:
            total = sum(
                int(getattr(stats, attr, 0) or 0)
                for attr in ("temp_size_in_bytes", "output_size_in_bytes", "argument_size_in_bytes")
            )
            if total > 0:
                return total
    except Exception:  # AOT compile / memory_analysis unsupported — fall through
        pass

    def _nbytes(tree):
        total = 0
        for leaf in jax.tree.leaves(tree):
            size, dtype = getattr(leaf, "size", None), getattr(leaf, "dtype", None)
            if size is not None and dtype is not None:
                total += int(size) * dtype.itemsize
        return total

    try:
        return _nbytes(jax.eval_shape(observable_fn, state)) + _nbytes(state)
    except Exception:  # abstract eval can fail on exotic observables — degrade gracefully
        return None


def resolve_cohort_batch_size(spec, n_subjects, fit_fn=None, example_args=None):
    """Resolve ``dataset.batch_size`` to a subject count per on-device batch.

    An explicit integer passes straight through (clamped to ``[1, n_subjects]``) —
    the caller opted in, so no memory bound applies. ``None`` requests automatic
    sizing: the whole cohort in one batch unless one lane's estimated peak memory
    (:func:`estimate_per_cell_bytes`, compiling *fit_fn* on *example_args*) times the
    shared-RAM device count would exceed :func:`auto_nvmap_budget_bytes`, in which
    case the batch is narrowed to fit. Without an estimate it degrades to the whole
    cohort, i.e. the un-chunked vmap. Unlike :func:`resolve_n_vmap`, no fixed count
    cap applies: a cohort's batch is bounded by memory, not an exploration-grid cap.
    """
    n = max(1, int(n_subjects))
    if spec is not None:
        return max(1, min(int(spec), n))
    per_lane = (
        estimate_per_cell_bytes(lambda a: fit_fn(*a), example_args)
        if fit_fn is not None and example_args is not None
        else None
    )
    if not per_lane or per_lane <= 0:
        return n
    n_pmap = max(1, shared_ram_device_count())
    width = auto_nvmap_budget_bytes() // (int(per_lane) * n_pmap)
    width = max(1, min(n, int(width)))
    logger.debug(
        "dataset.batch_size=auto -> %d/%d subjects/batch (per_lane=%s B, n_pmap=%d)",
        width, n, per_lane, n_pmap,
    )
    return width


def resolve_exploration_n_vmap(spec, grid_n, observable_fn, state):
    """Resolve ``Exploration.n_parallel`` to a vmap chunk width for a grid run.

    Composition both exploration templates call: for ``"auto"`` it estimates per-cell
    memory (:func:`estimate_per_cell_bytes`) and counts shared-RAM devices
    (:func:`shared_ram_device_count`) to bound the batch; an explicit integer skips the
    estimate (and its compile) and passes straight through :func:`resolve_n_vmap`.
    """
    per_cell = estimate_per_cell_bytes(observable_fn, state) if _is_auto(spec) else None
    return resolve_n_vmap(spec, grid_n, per_cell, n_pmap=shared_ram_device_count())


def _is_auto(spec) -> bool:
    """True when an ``n_parallel`` spec requests automatic vmap-width resolution."""
    return isinstance(spec, str) and spec.strip().lower() == "auto"


def resolve_n_vmap(spec, grid_n, per_cell_bytes=None, n_pmap=1):
    """Resolve an ``Exploration.n_parallel`` spec to a concrete vmap chunk width.

    Args:
        spec: An integer chunk size (``1`` = fully sequential), or the string
            ``"auto"`` to vectorise up to the count cap and memory budget.
        grid_n: Number of cells in the exploration grid.
        per_cell_bytes: Optional per-cell working-memory estimate (see
            :func:`estimate_per_cell_bytes`). When given, ``"auto"`` additionally caps
            ``n_vmap`` so the concurrently-held batch fits the budget.
        n_pmap: Number of devices whose batches share one RAM pool (see
            :func:`shared_ram_device_count`). The live batch is ``n_pmap × n_vmap`` cells
            in that shared RAM, so the budget bounds ``n_pmap × n_vmap × per_cell`` — not
            ``n_vmap`` alone (else forced host devices multiply the footprint past it).

    Returns:
        Positive integer vmap chunk width. An explicit integer is passed through
        unchanged (memory bounds do not apply — the caller opted in).
    """
    if not _is_auto(spec):
        return max(1, int(spec))
    n_pmap = max(1, int(n_pmap))
    width = min(int(grid_n), auto_nvmap_cap())
    if per_cell_bytes and per_cell_bytes > 0:
        width = min(width, auto_nvmap_budget_bytes() // (int(per_cell_bytes) * n_pmap))
    width = max(1, width)
    logger.debug(
        "n_parallel=auto -> n_vmap=%d (grid_n=%d, n_pmap=%d, per_cell_bytes=%s)",
        width, int(grid_n), n_pmap, per_cell_bytes,
    )
    return width


class LoggingProgressCallback(AbstractCallback):
    """Log optimization progress at INFO every ``every`` steps.

    A logging-native, drop-in replacement for tvboptim's print-based
    :class:`~tvboptim.optim.callbacks.DefaultPrintCallback`. When *total* is
    given the line reads ``step i/total``. Never signals a stop.

    Args:
        every: Emit one line every ``every`` steps (tvboptim gates the call).
        total: Total step count, shown as ``i/total`` when known.
    """

    def __init__(self, every: int = 1, total: Optional[int] = None) -> None:
        super().__init__(every)
        self.total = total

    def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
        if logger.isEnabledFor(logging.INFO):
            where = f"{i}/{self.total}" if self.total else str(i)
            logger.info("  step %s: loss=%.6g", where, float(loss_value))
        return False, diff_state, static_state


def progress_ticker(total: int, *, every: Optional[int] = None, label: str = "batch"):
    """Wrap a scanned/vmapped per-item function so it streams ``label i/total`` progress.

    The exploration / sweep grid runs as one JIT-compiled ``jax.lax.map``, so it prints
    ``STEP 2 > <exploration>`` and then nothing until it returns — the cluster "empty log"
    problem. This fires a JAX-native ``jax.debug.callback`` (no JIT break, vmap-safe) once
    per ``lax.map`` batch — a no-arg callback has no batched input to vectorise, so it runs
    once per scan step — ticking a host-side counter and logging through the central
    ``tvbo.run`` logger. The ``jax_tqdm`` pattern, reduced to the logging we already route.

    Args:
        total: Number of batches (``ceil(n_cells / n_vmap)``) for the ``i/total`` line.
        every: Log cadence in batches; defaults to ~25 evenly-spaced updates.
        label: Noun for the line, e.g. ``"grid batch"``.

    Returns:
        ``wrap(fn) -> fn`` — the identity when INFO is disabled, so there is zero runtime
        overhead under ``--quiet`` or a coarse ``TVBO_LOG_LEVEL``.
    """
    import jax
    from itertools import count

    total = max(1, int(total))
    every = every or max(1, total // 25)
    ticks = count(1)

    def _tick(*_):
        i = next(ticks)
        if i == total or i % every == 0:
            logger.info("  %s %d/%d (%d%%)", label, i, total, 100 * i // total)

    def wrap(fn):
        if not logger.isEnabledFor(logging.INFO):
            return fn

        def wrapped(*args, **kwargs):
            jax.debug.callback(_tick, ordered=False)
            return fn(*args, **kwargs)

        return wrapped

    return wrap
