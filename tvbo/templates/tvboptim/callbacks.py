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
from typing import Optional

from tvboptim.optim.callbacks import AbstractCallback

logger = logging.getLogger("tvbo.run")


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
