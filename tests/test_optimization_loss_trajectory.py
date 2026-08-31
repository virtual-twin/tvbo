"""A gradient fit's loss trajectory belongs in the container.

tvboptim's callbacks write the history into a plain ``dict``, so probing it with ``hasattr(history, "loss")`` found nothing and both `loss_trajectory` and `final_loss` stayed None — for every optimization, in every container. A convergence panel had nothing to bind, and nothing failed to say so.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tvbo.data.types import ExperimentResult, OptimizationResult


def _history(losses):
    """The shape tvboptim's `SavingLossCallback` leaves behind: a dict of DataFrames keyed `step`/`save`."""
    frame = pd.DataFrame(columns=["step", "save"])
    for step, value in enumerate(losses):
        frame.loc[len(frame)] = [step, value]
    return {"loss": frame}


def test_the_trajectory_is_read_from_a_keyed_history():
    opt = OptimizationResult(name="fit", history=_history([0.32, 0.19, 0.09]))
    assert opt.loss_trajectory is not None
    assert float(opt.final_loss) == pytest.approx(0.09)


def test_it_reaches_the_container(tmp_path):
    """The binding a convergence panel declares: `optimization__<name>__loss_trajectory`."""
    xr = pytest.importorskip("xarray")
    source = SimpleNamespace(network=SimpleNamespace(node_labels=["n0"]), dynamics=None, coupling=None)
    opt = OptimizationResult(name="gradient_eib", history=_history([0.32, 0.19, 0.09]))
    written = ExperimentResult(optimizations={"gradient_eib": opt}, source=source).save(
        str(tmp_path), compress=False, record_only=False
    )
    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        assert np.asarray(ds["optimization__gradient_eib__loss_trajectory"]).tolist() == pytest.approx([0.32, 0.19, 0.09])
        assert float(ds["optimization__gradient_eib__final_loss"]) == pytest.approx(0.09)


def test_a_history_without_a_loss_leaves_it_absent():
    """Absent, not zero: an optimization that recorded no loss has no trajectory to report."""
    opt = OptimizationResult(name="fit", history={})
    assert opt.loss_trajectory is None and opt.final_loss is None
