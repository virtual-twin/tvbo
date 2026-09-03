"""An external input's own parameter is a sweepable axis, on the clock the recipe declares it in.

A `<event>.<param>` axis writes to ``state.external.<event>.<param>``, where the emitted ExternalInput reads it. A declared ``t0`` is on the measurement clock, which is also the clock the measured scan integrates on -- it opens at 0, the settle having run before it as a scan of its own -- so neither a fixed onset nor a swept one is shifted anywhere. The exploration's coordinate is that same declared time, and so is the time axis its recorded trajectory comes back on.
"""

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

_T0_SWEEP = """
id: 91
label: "Swept pulse onset"
dynamics:
  name: Generic2dOscillator
  iri: tvbo:Generic2dOscillator
  parameters:
    a: {value: -1.5}
    b: {value: -15.0}
    c: {value: 0.0}
    d: {value: 0.015}
    e: {value: 3.0}
    f: {value: 1.0}
    tau: {value: 4.0}
    I: {value: 0.1}
  state_variables:
    V:
      equation: {rhs: "d*tau*(I*gamma - V**3*f + V**2*e + V*g + V*local_coupling + W*alpha + c_glob*gamma) + stimulus"}
      coupling_variable: true
      initial_value: 0.0
    W:
      equation: {rhs: "d*(V**2*c + V*b - W*beta + a)/tau"}
      initial_value: 0.0
network:
  number_of_nodes: 1
  coupling:
    c_lin:
      iri: tvbo:Linear
      delayed: false
      parameters: {G: {value: 0.0}}
      incoming_states: [V]
integration:
  method: Heun
  step_size: 0.2
  duration: 150.0
  transient_time: 50.0
events:
  stimulus:
    event_type: stimulus
    equation:
      rhs: "Piecewise((amplitude, (t >= t0) & (t < t0 + duration)), (0.0, True))"
    parameters:
      amplitude: {value: 0.4}
      t0: {value: 10.0, unit: ms}
      duration: {value: 1.0, unit: ms}
    nodes: [0]
    weights: [1.0]
explorations:
  onset_sweep:
    mode: product
    space:
      - parameter: stimulus.t0
        explored_values: [10.0, 60.0]
"""
"""A quiescent single node pulsed once, sweeping the onset across a 50 ms transient.

The two onsets straddle the settle's own length, so a stray shift by it lands the first pulse where the second belongs and cannot be read as noise.
"""


def _experiment(tmp_path):
    spec = tmp_path / "t0_sweep.yaml"
    spec.write_text(_T0_SWEEP)
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    return exp


def _squeeze(code):
    """The emitted code with whitespace and quoting normalised, since the formatter wraps long calls."""
    return "".join(code.split()).replace('"', "'")


BINDING = "grid_state.external.stimulus.t0=_ax('stimulus.t0',DataAxis("
"""The emitted binding: the external slot, carrying the path the recipe declared it as."""


def test_the_axis_writes_the_external_input_slot(tmp_path):
    """A swept event parameter reaches `state.external`, not `state.dynamics`."""
    code = _experiment(tmp_path).render_code("tvboptim")
    assert BINDING in _squeeze(code)


def test_a_swept_onset_is_written_as_declared(tmp_path):
    """The axis carries the declared times, unshifted, exactly as a fixed onset does.

    The scan opens at ``-transient_time``, so the solver's own clock IS the measurement clock and an onset means what it says. A shift here would fire every cell one settle late.
    """
    squeezed = _squeeze(_experiment(tmp_path).render_code("tvboptim"))
    axis = squeezed.split(BINDING)[1].split("grid=Space(")[0]
    assert "jnp.asarray([10.0,60.0]" in axis, axis
    assert "+50.0" not in axis, axis


def test_every_cell_fires_at_the_onset_it_declares(tmp_path):
    """The pulse lands where the recipe puts it, and the coordinate and the time axis are that same declared time.

    Read as the first step where V departs, not as the largest one: the pulse runs for a millisecond and V is slow, so the steepest step sits somewhere inside it and says only which millisecond, where the departure names the sample.
    """
    sweep = _experiment(tmp_path).run("tvboptim").explorations["onset_sweep"]
    np.testing.assert_allclose(sweep.cell_coords["stimulus.t0"], [10.0, 60.0])
    V = sweep.results.isel(variable=0, node=0)
    t = np.asarray(sweep.results.coords["time"])
    onsets = []
    for i in range(2):
        step = np.abs(np.diff(np.asarray(V[i])))
        onsets.append(t[int(np.argmax(step > 0.5 * step.max()))])
    np.testing.assert_allclose(onsets, [10.0, 60.0])


def test_the_recorded_sweep_is_on_the_measurement_clock(tmp_path):
    """A cell's recorded trajectory is the MEASURED window, on the clock a declared onset is declared against.

    A sweep records the window it integrated, and the settle is no longer part of that window: it is its own scan, run once, whose endpoint the grid warm-starts from. So `results` is the analogue of `SimulationResult.data` rather than of `.full` — on the same clock as before, which is what lets a declared onset still be found at the time it declares.
    """
    result = _experiment(tmp_path).run("tvboptim")
    t = np.asarray(result.explorations["onset_sweep"].results.coords["time"])
    assert t[0] == pytest.approx(0.2), t[0]
    assert t[-1] == pytest.approx(150.0), t[-1]
    np.testing.assert_allclose(t, np.asarray(result.integration.data.coords["time"]))
