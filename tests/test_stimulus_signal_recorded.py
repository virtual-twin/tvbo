"""A run keeps the stimulus it was driven with beside the trajectory it drove.

`ExperimentResult.save` writes every equation-defined stimulus event as `event__<name>`, evaluated on the recorded time axis, so a figure draws the drive and the response from one container. The signal must be the declared pulse exactly: zero before the onset, the amplitude while it is on, zero after.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.data.types import stimulus_signals, stimulus_waveforms  # noqa: E402

SPEC = """
label: Pulse into a leak
dynamics:
  name: Leak
  parameters:
    k: {value: 0.5}
    I: {value: 0.0}
  state_variables:
    x: {equation: {rhs: "-k*x + I"}, initial_value: 0.0}
network: {label: Solo, number_of_nodes: 1}
integration: {method: euler, step_size: 1.0, duration: 50.0, transient_time: 0.0}
events:
  I:
    event_type: stimulus
    equation:
      rhs: "Piecewise((amplitude, (t >= onset) & (t < onset + width)), (0.0, True))"
    parameters:
      onset: {value: 10.0}
      width: {value: 5.0}
      amplitude: {value: 2.0}
"""


def test_signal_is_the_declared_pulse():
    exp = SimulationExperiment.from_string(SPEC)
    t = np.arange(0.0, 30.0, 1.0)
    signal = stimulus_signals(exp, t)["event__I"]
    expected = np.where((t >= 10.0) & (t < 15.0), 2.0, 0.0)
    np.testing.assert_array_equal(signal.values, expected)
    assert signal.dims == ("time",)


def test_saved_container_carries_the_signal(tmp_path):
    exp = SimulationExperiment.from_string(SPEC)
    written = exp.run().save(tmp_path)
    ds = xr.open_dataset(next(p for p in written if str(p).endswith(".h5")))
    assert "event__I" in ds
    time = ds["integration"].coords["time"].values
    np.testing.assert_array_equal(ds["event__I"].values, np.where((time >= 10.0) & (time < 15.0), 2.0, 0.0))


def _stimulus(name):
    return SimpleNamespace(
        name=name,
        event_type="stimulus",
        equation=SimpleNamespace(rhs="amplitude*t"),
        parameters={"amplitude": SimpleNamespace(value=2.0)},
        dataLocation=None,
    )


def test_a_fisher_target_is_neither_written_nor_reported_as_a_drive():
    """The event a `fisher` analysis names as its target is never integrated, so the container holds no signal for it and the report keeps its effect in the event table instead of stating it as a drive."""
    from tvbo.utils.report import event_table

    events = {"I": _stimulus("I"), "probe": _stimulus("probe")}
    exp = SimpleNamespace(
        events=events, observations={"fim": SimpleNamespace(analysis=SimpleNamespace(type="fisher", target="probe"))}
    )
    shown = list(stimulus_waveforms(exp))
    assert shown == ["I"]
    assert list(stimulus_signals(exp, np.arange(3.0))) == ["event__I"]
    rows = {
        line.split("|")[1].strip(): line
        for line in event_table(events, shown_above=shown).splitlines()
        if line.startswith("| `")
    }
    assert "$amplitude t$" not in rows["`I`"] and "$amplitude t$" in rows["`probe`"]
