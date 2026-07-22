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


# ``Exploration.n_parallel: auto`` vectorises the grid in chunks of n_vmap cells. It
# is bounded two ways (whichever is smaller):
#   * a cell COUNT cap (``AUTO_NVMAP_CAP``, env ``TVBO_NVMAP_AUTO_CAP``): per-cell
#     throughput saturates by ~32-64 cells, so this forfeits no meaningful speed;
#   * a MEMORY budget (``TVBO_NVMAP_MEM_BUDGET_GB``): vectorising N cells holds N ×
#     per-cell working set (output trajectory + live state, incl. delay history) at
#     once, so — unlike a sequential run, which holds one cell — auto can raise peak
#     memory. The budget bounds that batch footprint; when a per-cell estimate is
#     available (see :func:`estimate_per_cell_bytes`) auto shrinks n_vmap to fit it.
# An explicit integer ``n_parallel`` bypasses both bounds.
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
    Bounds ``n_vmap × per-cell-bytes`` so auto-vectorisation cannot blow up peak memory
    on a large-per-cell model (e.g. a whole-brain delay network).
    """
    return int(_env_positive("TVBO_NVMAP_MEM_BUDGET_GB", float, AUTO_NVMAP_MEM_BUDGET_GB) * (1024 ** 3))


def estimate_per_cell_bytes(observable_fn, state) -> Optional[int]:
    """Best-effort per-cell working-memory estimate for ``n_parallel: auto``.

    Sums the recorded output size (via ``jax.eval_shape`` — abstract, no compute) and
    the live per-cell input ``state`` (which carries any delay/history buffers). This
    is the buffer that a vmapped batch holds ``n_vmap`` copies of. Returns ``None`` if
    the shapes cannot be inferred, so the caller falls back to the count-only cap.
    """
    try:
        import jax

        def _nbytes(tree):
            total = 0
            for leaf in jax.tree.leaves(tree):
                size = getattr(leaf, "size", None)
                dtype = getattr(leaf, "dtype", None)
                if size is not None and dtype is not None:
                    total += int(size) * dtype.itemsize
            return total

        out = jax.eval_shape(observable_fn, state)
        return _nbytes(out) + _nbytes(state)
    except Exception:  # abstract eval can fail on exotic observables — degrade gracefully
        return None


def resolve_n_vmap(spec, grid_n, per_cell_bytes=None):
    """Resolve an ``Exploration.n_parallel`` spec to a concrete vmap chunk width.

    Args:
        spec: An integer chunk size (``1`` = fully sequential), or the string
            ``"auto"`` to vectorise up to the count cap and memory budget.
        grid_n: Number of cells in the exploration grid.
        per_cell_bytes: Optional per-cell working-memory estimate (see
            :func:`estimate_per_cell_bytes`). When given, ``"auto"`` additionally caps
            ``n_vmap`` at ``budget // per_cell_bytes`` so the batch fits the budget.

    Returns:
        Positive integer vmap chunk width. An explicit integer is passed through
        unchanged (memory bounds do not apply — the caller opted in).
    """
    if not (isinstance(spec, str) and spec.strip().lower() == "auto"):
        return max(1, int(spec))
    width = min(int(grid_n), auto_nvmap_cap())
    if per_cell_bytes and per_cell_bytes > 0:
        width = min(width, auto_nvmap_budget_bytes() // int(per_cell_bytes))
    return max(1, width)


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
