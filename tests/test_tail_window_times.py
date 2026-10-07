"""A trailing-window observation (`tail_duration` / `tail_samples`) keeps the time stamps of the samples it kept.

The monitor used to label any output shorter than the recorded span by spreading it uniformly across that span, which is right for a subsampled series and wrong for a tail: a 200 ms tail of a 60 s run came back stamped from 1 ms to 60 000 ms, so a panel plotting it against time drew 200 samples across a minute.
"""

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

_SPEC = """
id: 1
dynamics:
  name: Leak
  label: Leak
  parameters:
    a: {value: 0.1}
  state_variables:
    x:
      initial_value: 1.0
      equation: {rhs: "-a * x"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 1
  nodes:
    - {id: 0, label: r0}
  edges: []
integration: {method: euler, step_size: 1.0, duration: 500.0, transient_time: 0.0, unit: ms}
observations:
  x_tail: {source: [x], tail_duration: 20.0}
execution: {random_seed: 0, precision: float64}
"""


@pytest.mark.slow
def test_a_tail_is_stamped_at_the_end_of_the_run():
    result = SimulationExperiment.from_string(_SPEC).run("tvboptim")
    tail = result.integration.observations["x_tail"]
    t = np.asarray(tail.ts, dtype=float)
    assert t.size == 20
    assert np.allclose(np.diff(t), 1.0)
    assert t[-1] == pytest.approx(500.0, abs=1.0)
