"""An algorithm wired into an exploration (`Exploration.algorithms`) tunes at every grid cell, and that tuning has to see the cell.

Four properties, each of which failed silently before: a swept stimulus parameter (an `<event>.<param>` axis) or noise amplitude (`noise.sigma`) must reach the tuning run and not only the measurement after it, or every cell is tuned against the default drive or noise; an `execution.random_seed` axis must seed the tuning noise too, or the "ensemble" shares one tuning realisation; and the value each update rule reached is a result of the cell, recorded beside its observations as `algorithm_<name>_<target>` on the axes its parameter declares.
"""

import re

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

# One noisy leaky node driven by a stimulus event, with a per-node offset `g` a homeostatic rule tunes until the node's mean is zero.
_SPEC = """
id: 1
dynamics:
  name: TunedLeak
  label: TunedLeak
  parameters:
    g: {value: 0.0, heterogeneous: true, free: true, shape: "(n_nodes,)"}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "-x + drive - g"}
      coupling_variable: true
      noise: {additive: true, parameters: {sigma: {value: 0.05}}}
  output: [x]
network:
  number_of_nodes: 1
  nodes:
    - {id: 0, label: r0}
  edges: []
integration: {method: euler, step_size: 1.0, duration: 50.0, transient_time: 0.0, unit: ms}
events:
  drive:
    event_type: stimulus
    equation: {rhs: "A"}
    parameters:
      A: {value: 0.0}
observations:
  mean_x: {source: [x], aggregation: mean}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.0}
    observations: [mean_x]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g + eta * mean_x"}
    hyperparameters:
      - {name: eta, value: 0.5}
    n_iterations: 30
    simulation_period: 20.0
explorations:
  sweep:
    algorithms: [tune]
    space:
      drive.A: {explored_values: [0.0, 1.0]}
      execution.random_seed: {explored_values: [1, 2]}
execution: {random_seed: 0, precision: float64}
"""


@pytest.fixture(scope="module")
def experiment():
    return SimulationExperiment.from_string(_SPEC)


def _code(experiment):
    """The rendered module with whitespace removed, so the formatter's wrapping cannot break a match."""
    return "".join(experiment.render_code("tvboptim").split())


def test_the_tuning_reads_the_cells_seed(experiment):
    assert "jax.random.key(jnp.asarray(_ps.dynamics._noise_seed" in _code(experiment)


def test_a_swept_stimulus_parameter_reaches_the_tuning(experiment):
    assert "_ts.external[_en][_k]=_ps.external[_en][_k]" in _code(experiment)


def test_a_swept_noise_amplitude_reaches_the_tuning():
    spec = _SPEC.replace(
        "      execution.random_seed: {explored_values: [1, 2]}", "      noise.sigma: {explored_values: [0.01, 0.1]}"
    )
    assert re.search(r"_ts\.noise\.sigma=\(?_ps\.noise\.sigma", _code(SimulationExperiment.from_string(spec)))


def test_the_tuned_target_is_recorded_on_its_node_axis(experiment):
    code = _code(experiment)
    assert '_point_out["algorithm_tune_g"]=_ps.dynamics["g"]' in code
    assert '"algorithm_tune_g":["node"]' in code


@pytest.mark.slow
def test_each_cell_tunes_against_its_own_drive_and_seed(experiment):
    """The tuned offset follows the swept drive (g settles near A) and differs between seeds at one drive."""
    result = experiment.run("tvboptim")
    g = result.explorations["sweep"].observations["algorithm_tune_g"]
    assert "node" in g.dims
    g = g.squeeze("node")
    by_drive = g.mean("execution.random_seed")
    assert float(by_drive.sel({"drive.A": 1.0})) - float(by_drive.sel({"drive.A": 0.0})) > 0.5
    seeds = g.sel({"drive.A": 1.0}).values
    assert not np.isclose(seeds[0], seeds[1])
