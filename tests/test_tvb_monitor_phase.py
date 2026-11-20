"""A periodic tvb monitor samples the run's own grid, whatever number of steps its delay history counts.

TVB counts the rows of a delay history as steps already taken, so the probe of ``test_delay_history_from_opening``, whose history has 5 rows, opens at step 4. A generated periodic monitor that counted its period on the absolute step sampled where that step is a multiple of the period: with a 3 ms period, 2, 5, 8 and 11 ms into the run, where the undelayed probe samples at 3, 6, 9 and 12 ms. The 2 ms subsample of ``test_tvb_delayed_clock`` divides 4 and so never showed it. ``y(t) = (settle + t) a W x(0)`` is linear in time: a subsample reads the closed form at its stamp, and a window's mean reads it at the mean of its samples' times, half a step after the centre it is stamped at.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tvb")

from tvbo import SimulationExperiment

from .test_delay_history_from_opening import DT, _assert_reads_its_opening
from .test_tvb_delayed_clock import SETTLES, _assert_on_the_measurement_clock, _settled, _undelayed

pytestmark = pytest.mark.backend_tvb

PERIOD = 3 * DT
"""A period that does not divide the 4 steps the probe's history counts."""

MONITORS = {
    "SubSample": {"iri": "tvbo:observation_models/subsample", "period": PERIOD},
    "TemporalAverage": {"iri": "tvbo:observation_models/temporal_average_tvb", "period": PERIOD},
}
CENTRE = {"SubSample": 0.0, "TemporalAverage": PERIOD / 2}
"""Where each monitor stamps a sample, behind the step that takes it: the subsample at that step, the average at the centre of its window."""
LAG = {"SubSample": 0.0, "TemporalAverage": DT / 2}
"""Where each sample reads the closed form, after its stamp: the subsample at the stamp, the average at the mean of its post-step samples' times, half a step after the window's centre."""
EACH_MONITOR = pytest.mark.parametrize("monitor", list(MONITORS))


def _observed(spec, monitor, settle):
    """*monitor* of *spec* run on tvb, checked to be on the run's grid of *PERIOD*, and its measured ``x``, ``y`` and the times their closed form is read at (`LAG`)."""
    spec["observations"] = {monitor: dict(MONITORS[monitor])}
    observation = SimulationExperiment(**spec).run("tvb").integration.observations[monitor]
    _assert_on_the_measurement_clock(observation, settle, period=PERIOD, centre=CENTRE[monitor])
    data = observation.data
    if "mode" in data.dims:
        data = data.isel(mode=0)
    data = data.transpose("time", "variable", "node")
    return data.sel(variable="x").values, data.sel(variable="y").values, data["time"].values + LAG[monitor]


@SETTLES
@EACH_MONITOR
def test_a_delayed_run_s_monitor_samples_the_run_s_grid(monitor, settle):
    """Counted from the step the run opened at, the delayed probe's 3 ms monitor samples at 3, 6, 9 and 12 ms, after a settle at -3 and 0 ms, and each sample reads the closed form at its stamp, an average half a step after it."""
    x, y, t = _observed(_settled(settle), monitor, settle)
    _assert_reads_its_opening(x, y, t, settle)


@EACH_MONITOR
def test_an_undelayed_run_s_monitor_keeps_its_grid(monitor):
    """An undelayed run opens at step 0, where the run's grid and the absolute one are the same."""
    spec = _undelayed()
    assert SimulationExperiment(**spec).execute("tvb").current_step == 0
    x, y, t = _observed(spec, monitor, 0.0)
    _assert_reads_its_opening(x, y, t, 0.0)
