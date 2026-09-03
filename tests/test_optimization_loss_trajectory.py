"""A gradient fit's loss trajectory belongs in the container.

tvboptim's callbacks write the history into a plain ``dict``, so probing it with ``hasattr(history, "loss")`` found nothing and both `loss_trajectory` and `final_loss` stayed None — for every optimization, in every container. A convergence panel had nothing to bind, and nothing failed to say so.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tvbo.data.types import ExperimentResult, OptimizationResult


def _history(losses, states=None):
    """The shape tvboptim's `SavingLossCallback` leaves behind: a dict of DataFrames keyed `step`/`save`.

    *states* adds what `SavingParametersCallback` records beside it — the saved state itself, one per recorded step.
    """
    frame = pd.DataFrame(columns=["step", "save"])
    for step, value in enumerate(losses):
        frame.loc[len(frame)] = [step, value]
    history = {"loss": frame}
    if states is not None:
        history["parameters"] = states
    return history


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


def _saved(tmp_path, opt, name="fit"):
    """Write a one-optimization container and hand back the opened dataset's path."""
    source = SimpleNamespace(network=SimpleNamespace(node_labels=["n0"]), dynamics=None, coupling=None)
    written = ExperimentResult(optimizations={name: opt}, source=source).save(str(tmp_path), compress=False, record_only=False)
    return [p for p in written if p.endswith(".h5")][0]


def test_a_parameter_trajectory_reaches_the_container(tmp_path):
    """The state a fit hands back is an object pytree rather than a dict, and an equinox module and this stand-in are descended by the same `vars()` branch, so a walk that knew only about dicts and lists wrote nothing at all here."""
    xr = pytest.importorskip("xarray")
    states = [SimpleNamespace(dynamics=SimpleNamespace(g=np.asarray(v))) for v in (1.0, 0.6, 0.4)]
    opt = OptimizationResult(name="fit", history=_history([0.32, 0.19, 0.09], states))
    with xr.open_dataset(_saved(tmp_path, opt), engine="h5netcdf") as ds:
        assert np.asarray(ds["optimization__fit__history__dynamics.g"]).tolist() == pytest.approx([1.0, 0.6, 0.4])


def test_the_parameter_trajectory_shares_the_loss_step_axis(tmp_path):
    """What lets a convergence panel draw loss and parameters against each other, with the step selectable by name rather than by position."""
    xr = pytest.importorskip("xarray")
    states = [{"g": np.asarray(v)} for v in (1.0, 0.6, 0.4)]
    opt = OptimizationResult(name="fit", history=_history([0.32, 0.19, 0.09], states))
    with xr.open_dataset(_saved(tmp_path, opt), engine="h5netcdf") as ds:
        assert ds["optimization__fit__history__g"].dims == ds["optimization__fit__loss_trajectory"].dims
        assert np.asarray(ds["step"]).tolist() == [0, 1, 2]


def test_solver_bookkeeping_is_not_written_as_a_result(tmp_path):
    """A solver carries its step size and stage indices beside the values it fitted, and writing those states them as outcomes of the run."""
    xr = pytest.importorskip("xarray")
    states = [{"g": np.asarray(v), "_step_size": np.asarray(0.01)} for v in (1.0, 0.6)]
    opt = OptimizationResult(name="fit", history=_history([0.3, 0.1], states))
    with xr.open_dataset(_saved(tmp_path, opt), engine="h5netcdf") as ds:
        assert "optimization__fit__history__g" in ds.data_vars
        assert not [v for v in ds.data_vars if "step_size" in v]


def test_a_parameter_that_changes_shape_does_not_cost_the_container(tmp_path):
    """A trajectory that cannot be stacked drops that one parameter; it must not take the fit's observations and fitted values down with it."""
    xr = pytest.importorskip("xarray")
    states = [{"g": np.zeros(2)}, {"g": np.zeros(3)}]
    opt = OptimizationResult(name="fit", history=_history([0.3, 0.1], states))
    with xr.open_dataset(_saved(tmp_path, opt), engine="h5netcdf") as ds:
        assert not [v for v in ds.data_vars if "history" in v]
        assert float(ds["optimization__fit__final_loss"]) == pytest.approx(0.1)
