"""A tuning rule that steps a coupling parameter addresses it where tvboptim's state holds it: under the coupling-input name, not the coupling's own.

tvboptim keys `state.coupling` by coupling input, a recipe keys its couplings by function, and the algorithm template resolves the one to the other exactly as the experiment template does (`coupling_input_wiring`). The model feeds input ``c`` from coupling ``LinCoupling``, so an accessor built from the coupling's key names a branch the state does not have.
"""

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

pytestmark = pytest.mark.backend_tvboptim

_SPEC = """
id: 1
dynamics:
  name: CoupledLeak
  parameters:
    I: {value: 0.2}
  coupling_inputs:
    c: {source: LinCoupling}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "-x + I + c"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 1.0}
    - {source: 1, target: 0, weight: 1.0}
  coupling:
    LinCoupling:
      parameters:
        G: {value: 0.5, shape: "(n_nodes,)"}
      pre_expression: {rhs: x_j}
      post_expression: {rhs: G * gx}
      incoming_states: [x]
      local_states: [x]
integration: {method: euler, step_size: 0.1, duration: 20.0, transient_time: 0.0, unit: ms}
observations:
  mean_x: {source: [x], aggregation: mean}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.0}
    observations: [mean_x]
    update_rules:
      - name: G_update
        target_parameter: {name: G}
        equation: {rhs: "G - eta * mean_x"}
    hyperparameters:
      - {name: eta, value: 0.1}
    n_iterations: 3
    simulation_period: 10.0
execution: {random_seed: 0, precision: float64}
"""


@pytest.fixture(scope="module")
def experiment():
    return SimulationExperiment.from_string(_SPEC)


def test_the_rule_addresses_the_coupling_input(experiment):
    code = "".join(experiment.render_code("tvboptim").split())
    assert "state.coupling.c.G" in code
    assert "coupling.LinCoupling.G" not in code


def test_the_tuning_steps_the_gain_the_input_reads(experiment):
    tune = experiment.run("tvboptim", mode="algorithms", quiet=True).algorithms["tune"]
    tuned = np.asarray(tune.state.coupling.c.G)
    assert tuned.shape == (2,)
    assert np.all(tuned < 0.5)
