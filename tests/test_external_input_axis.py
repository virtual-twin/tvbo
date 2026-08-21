"""An external input's own parameter is a sweepable axis, on the clock the recipe declares it in.

A `<event>.<param>` axis writes to ``state.external.<event>.<param>``, where the emitted ExternalInput reads it. A fixed ``t0`` is declared relative to the MAIN simulation and shifted onto the padded clock before the run; a swept one means the same thing, so the axis carries the same shift and the exploration's coordinate carries the declared time back. Without the shift every cell fired one transient early -- inside the window the run discards -- and the sweep looked inert rather than wrong.
"""

import numpy as np
import pytest

from .tvboptim_capabilities import needs_axis_wrap

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

The two onsets straddle the transient: unshifted, the first fires inside the discarded window and never reaches the recorded trace at all.
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


def test_a_swept_onset_carries_the_transient_shift(tmp_path):
    """The axis is written on the padded clock, exactly as a fixed onset is."""
    squeezed = _squeeze(_experiment(tmp_path).render_code("tvboptim"))
    axis = squeezed.split(BINDING)[1].split("grid=Space(")[0]
    assert "+50.0" in axis, axis


@needs_axis_wrap
def test_every_cell_fires_at_the_onset_it_declares(tmp_path):
    """The pulse lands where the recipe puts it, and the coordinate is that same declared time."""
    sweep = _experiment(tmp_path).run("tvboptim").explorations["onset_sweep"]
    np.testing.assert_allclose(sweep.cell_coords["stimulus.t0"], [10.0, 60.0])
    V = sweep.results.isel(variable=0, node=0)
    t = np.asarray(sweep.results.coords["time"])
    onsets = [t[int(np.argmax(np.abs(np.diff(np.asarray(V[i]))))) + 1] for i in range(2)]
    np.testing.assert_allclose(onsets, [10.0, 60.0])
