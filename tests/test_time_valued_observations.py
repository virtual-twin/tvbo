"""A time-valued observation must not restate the integration step.

Two slots used to force a recipe to hardcode ``dt``. ``aggregation: first_passage`` returns a sample *index*, so any equation turning it into a latency had to multiply by a literal step; and ``tail_samples`` states a trailing window as a sample count, so the same declaration covers half the duration at half the step. Both are silent: changing ``step_size`` rescales the reported number with no error anywhere.

``dt`` is now bound inside a derived observation's ``equation`` (to the sample period of its sources), and ``tail_duration`` states a window as a length of simulated time. Both are pinned here at three step sizes, because a single-step test would pass on a hardcoded literal too.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: Time-valued observations
dynamics:
  name: Ramp
  parameters:
    k: {value: 0.5}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: %(step)s, duration: 100.0, unit: ms}
observations:
  t_cross:
    label: First passage of x
    source: [x]
    aggregation: first_passage
    parameters: {threshold: {value: 5.0}}
  x_tail:
    label: Mean x over the last 20 ms
    source: [x]
    aggregation: mean
    tail_duration: 20.0
  latency:
    label: First-passage latency (ms)
    source: [t_cross]
    pipeline:
      - equation: {rhs: "t_cross * dt"}
"""


def render(step):
    return SimulationExperiment.from_string(SPEC % {"step": step}).render_code("tvboptim")


@pytest.mark.parametrize(
    "step,factor,samples",
    [(0.5, "0.5", 40), (1.0, "1.0", 20), (0.25, "0.25", 80)],
)
def test_dt_and_tail_duration_track_the_step(step, factor, samples):
    """One declaration, three steps: the latency factor and the window both follow ``step_size``."""
    code = render(step)
    assert f"obs.latency = {factor} * t_cross" in code
    assert f"_data = _data[-{samples}:]" in code


def test_tail_samples_still_states_a_raw_count():
    """The sample-count form is unchanged — it is correct when the window really is N samples."""
    spec = SPEC.replace("tail_duration: 20.0", "tail_samples: 7") % {"step": 0.5}
    assert "_data = _data[-7:]" in SimulationExperiment.from_string(spec).render_code("tvboptim")


def test_declaring_both_window_forms_is_refused():
    """They disagree the moment the step changes, so the render fails instead of picking one."""
    spec = SPEC.replace("tail_duration: 20.0", "tail_duration: 20.0\n    tail_samples: 7")
    with pytest.raises(ValueError, match="tail_samples"):
        SimulationExperiment.from_string(spec % {"step": 0.5}).render_code("tvboptim")


def test_a_window_shorter_than_one_sample_is_refused():
    """Rounding it to zero samples would hand the aggregation an empty slice."""
    spec = SPEC.replace("tail_duration: 20.0", "tail_duration: 0.2")
    with pytest.raises(ValueError, match="no samples"):
        SimulationExperiment.from_string(spec % {"step": 1.0}).render_code("tvboptim")
