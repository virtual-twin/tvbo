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
