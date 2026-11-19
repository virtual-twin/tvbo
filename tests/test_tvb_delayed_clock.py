"""A tvb run reports every monitor on the measurement clock every other backend uses, however long its delay history.

TVB stamps the sample of step k at ``k * dt`` and counts the rows of an initial history as steps already taken: a configure on a history of H rows advances ``current_step`` by H - 1, so a clock read off TVB's stamps starts H - 1 steps late, and a settle read off it is neither moved before ``t = 0`` nor counted. The probe is the one of ``test_delay_history_from_opening`` with a declared opening per node: ``x`` holds it, ``y`` integrates what the 4 ms delayed coupling delivers, so ``y(t) = (settle + t) a W x(0)`` on the measurement clock and a sample stamped on another clock is off that line.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest

pytest.importorskip("tvb")

from tvbo import ExperimentResult, SimulationExperiment

from .test_delay_history_from_opening import (
    DT,
    DURATION,
    RTOL,
    _assert_reads_its_opening,
    _declared_per_node,
    _trace,
)

pytestmark = pytest.mark.backend_tvb

SETTLE = 6.0
SETTLES = pytest.mark.parametrize("settle", [0.0, SETTLE], ids=["no-settle", "settle"])

MONITORS = {
    "Raw": {"iri": "tvbo:observation_models/raw"},
    "SubSample": {"iri": "tvbo:observation_models/subsample", "period": 2 * DT},
    "TemporalAverage": {"iri": "tvbo:observation_models/temporal_average_tvb", "period": DT},
}
"""Raw, a 2 ms subsample stamped at the step it takes, and a one-step average stamped at the centre of its window, half a step before the step that ends it; the probe's 5-row history counts 4 steps, an even number, so the subsample takes the run's even steps."""


def _settled(settle, observations=None):
    """The probe with each node's declared ``x``, a settle of *settle* ms and, if given, the monitors *observations*."""
    spec = _declared_per_node()
    spec["integration"]["transient_time"] = settle
    if observations is not None:
        spec["observations"] = copy.deepcopy(observations)
    return spec


def _undelayed():
    """The probe with zero-length tracts and an undelayed coupling, so its history is one row."""
    spec = _settled(0.0)
    for edge in spec["network"]["edges"]:
        edge["parameters"]["length"]["value"] = 0.0
    spec["network"]["coupling"]["LinearCoupling"]["delayed"] = False
    return spec


def _wrapped(spec, settle):
    """*spec* run on its configured simulator and handed to ``from_tvb`` with the step the simulator opened at."""
    sim = SimulationExperiment(**spec).execute("tvb")
    start_step = sim.current_step
    assert start_step > 0, "the configure on the delay history must have counted steps for the clock to need correcting"
    return ExperimentResult.from_tvb(sim, sim.run(), transient_time=settle, start_step=start_step)


def _assert_on_the_measurement_clock(result, settle, period=DT, centre=0.0):
    """*result*'s measured window is stamped ``period, 2 period, ...`` less *centre*, and a declared settle stays on ``.transient`` stamped the same way up to ``t = 0``."""
    stamps = period * np.arange(1, int(DURATION / period) + 1) - centre
    np.testing.assert_allclose(result.data["time"].values, stamps, rtol=0, atol=1e-9)
    if not settle:
        assert result.transient is None
        return
    assert result.transient is not None, "a declared settle must be kept on `.transient`"
    stamps = period * np.arange(1, int(settle / period) + 1) - settle - centre
    np.testing.assert_allclose(result.transient.data["time"].values, stamps, rtol=0, atol=1e-9)


@pytest.mark.backend_tvboptim
@SETTLES
def test_a_delayed_tvb_run_is_stamped_on_tvboptim_s_clock(settle):
    """The measured window of the same recipe carries the same stamps on tvb as on tvboptim, sample for sample, and the same numbers."""
    tvb, tvboptim = _trace(_settled(settle), "tvb"), _trace(_settled(settle), "tvboptim")
    np.testing.assert_array_equal(tvb["time"].values, tvboptim["time"].values)
    np.testing.assert_allclose(tvb.values, tvboptim.values, rtol=1e-8, atol=1e-8)


def test_a_delayed_tvb_run_counts_its_settle_on_the_corrected_clock():
    """The settle is the samples stamped at or before ``t = 0`` once the history's steps are taken off, and every sample of the whole run, settle included, is the closed form at its own stamp."""
    result = SimulationExperiment(**_settled(SETTLE)).run("tvb").integration
    _assert_on_the_measurement_clock(result, SETTLE)
    full = result.full.isel(mode=0).transpose("time", "variable", "node")
    x, y = full.sel(variable="x").values, full.sel(variable="y").values
    _assert_reads_its_opening(x, y, full["time"].values, SETTLE, RTOL["tvb"])


@SETTLES
@pytest.mark.parametrize("path", ["run", "from_tvb"])
def test_every_monitor_keeps_its_settle_off_its_data(settle, path):
    """Each extra monitor is on the measurement clock with its settle on ``.transient``, on the run path and through ``from_tvb`` alike. The subsample's settle samples are at -4, -2 and 0 ms; the average's first window inside the measured window is stamped at half a step, so it is measured, with a settle or without one."""
    spec = _settled(settle, MONITORS)
    if path == "run":
        observations = SimulationExperiment(**spec).run("tvb").integration.observations
    else:
        observations = _wrapped(spec, settle).integration.observations
    _assert_on_the_measurement_clock(observations["SubSample"], settle, period=2 * DT)
    _assert_on_the_measurement_clock(observations["TemporalAverage"], settle, centre=DT / 2)


@SETTLES
@pytest.mark.parametrize("handed", [True, False], ids=["handed", "run-here"])
def test_from_tvb_puts_a_delayed_run_on_the_measurement_clock(settle, handed):
    """The standalone script's wrapper reports the run path's clock, whether it is handed the run with the step the simulator opened at or runs the simulator itself."""
    if handed:
        result = _wrapped(_settled(settle), settle)
    else:
        sim = SimulationExperiment(**_settled(settle)).execute("tvb")
        assert sim.current_step > 0, "the configure on the delay history must have counted steps"
        result = ExperimentResult.from_tvb(sim, transient_time=settle)
    _assert_on_the_measurement_clock(result.integration, settle)


def test_from_tvb_refuses_a_run_without_the_step_it_opened_at():
    """A finished run's stamps do not say how many history rows TVB counted before it, so ``from_tvb`` is told rather than guessing."""
    sim = SimulationExperiment(**_settled(0.0)).execute("tvb")
    with pytest.raises(ValueError, match="start_step"):
        ExperimentResult.from_tvb(sim, sim.run())


def test_the_standalone_script_hands_from_tvb_the_step_it_opened_at():
    """Matched across whitespace, since the rendered script is laid out by black."""
    code = SimulationExperiment(**_settled(SETTLE)).render_code("tvb")
    assert re.search(r"start_step = sim\.current_step\s+raw = sim\.run\(\)", code)
    call = r"ExperimentResult\.from_tvb\(\s*sim,\s*raw,\s*transient_time=[^,]+,\s*start_step=start_step,?\s*\)"
    assert re.search(call, code)
    assert len(re.findall(r"\bsim(?:ulator)?\.configure\(\)", code)) == 1, "the script configures its simulator once"


def test_a_tvb_run_configures_its_simulator_once(monkeypatch):
    """``define_simulation`` builds the simulator unconfigured and the run configures it once, after it sets the opening, so no default history is drawn before it."""
    from tvb.simulator.simulator import Simulator

    configured = []
    configure = Simulator.configure

    def counted(self, *args, **kwargs):
        configured.append(self)
        return configure(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "configure", counted)
    SimulationExperiment(**_settled(SETTLE)).run("tvb")
    assert len(configured) == 1


def test_an_undelayed_tvb_run_keeps_its_clock():
    """A one-row history counts no step, so an undelayed run's stamps are TVB's own, ``dt, 2 dt, ...``."""
    assert SimulationExperiment(**_undelayed()).execute("tvb").current_step == 0
    _assert_on_the_measurement_clock(SimulationExperiment(**_undelayed()).run("tvb").integration, 0.0)
