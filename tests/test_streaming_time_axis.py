"""A streamed observable must say when each of its samples was taken.

An observation that declares ``reduce: streaming`` is folded into the integrator carry, so what comes back is the reducer's value rather than a materialised ``ObservationResult`` carrying a ``ts``. The container labels it from the axis names the reduction declares, which gives it a ``time`` DIMENSION — and then nothing puts a coordinate on that dimension, so the array announces a time axis it cannot describe. The values are right, which is what makes the loss silent: moving an observable onto the streaming path for memory reasons should not cost it its clock.

The stamping rule is the one #110 established by perturbation rather than by reading the emitting code: a sample covers the period that ENDS at its timestamp, so sample m sits at ``(m + 1) * period`` from the measured origin, whatever settle preceded the window.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: Streamed observable time axis
dynamics:
  name: Ramp
  parameters:
    k: {value: 1.0}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.0}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: %(duration)s, unit: ms%(transient)s}
observations:
  streamed:
    label: Running sum, streamed
    source: [x]
    period: %(period)s
    reduce: streaming
    dynamics:
      name: RunningSum
      system_type: discrete
      state_variables:
        acc:
          initial_value: 0.0
          equation_type: recurrence
          equation: {rhs: "acc + x"}
      derived_variables:
        total:
          record: true
          equation: {rhs: "acc"}
"""


def _streamed(duration=50.0, period=10.0, transient=0.0):
    spec = SPEC % {
        "duration": duration,
        "period": period,
        "transient": f", transient_time: {transient}" if transient else "",
    }
    return SimulationExperiment.from_string(spec).run("tvboptim").observations["streamed"]


def test_the_streamed_observable_declares_a_time_axis():
    """Precondition for the rest: the container labels it from the reduction, so `time` is a dim."""
    assert "time" in _streamed().dims


def test_the_time_axis_carries_a_coordinate():
    """The defect itself. A `time` dim with no coordinate cannot say what any of its values are."""
    da = _streamed()
    assert "time" in da.coords, f"time dim has no coordinate; coords are {list(da.coords)}"


def test_a_sample_is_stamped_at_the_end_of_the_period_it_covers():
    """The #110 convention: sample m spans (m*period, (m+1)*period] and is stamped at the last of them."""
    period = 10.0
    ts = np.asarray(_streamed(duration=50.0, period=period).coords["time"])
    assert np.allclose(ts, (np.arange(len(ts)) + 1) * period), f"not end-of-period: {ts}"


def test_the_axis_opens_at_t_zero_however_long_the_settle():
    """D8: t = 0 is the start of MEASUREMENT, so a declared settle moves no reported timestamp."""
    without = np.asarray(_streamed(transient=0.0).coords["time"])
    settled = np.asarray(_streamed(transient=30.0).coords["time"])
    assert np.allclose(without, settled), f"the settle shifted the axis: {without} vs {settled}"


def test_the_coordinate_is_one_stamp_per_sample():
    """A coordinate of the wrong length is worse than none: it would mislabel every sample."""
    da = _streamed(duration=50.0, period=10.0)
    assert len(da.coords["time"]) == da.sizes["time"] == 5


def test_stamping_did_not_disturb_the_values():
    """The reducer folds what it folded before; only the labelling is new."""
    da = _streamed(duration=50.0, period=10.0)
    # x ramps at k=1 and the observer sums it after each step, so the sample at the end of step n is 1+..+n.
    steps = (np.arange(5) + 1) * 10
    assert np.allclose(np.asarray(da)[:, 0], steps * (steps + 1) / 2.0)
