"""A per-edge conduction delay is a sweepable graph leaf.

A network that carries explicit per-edge ``delay`` attributes lowers onto a ``DenseDelayGraph``, whose ``delays`` are a live leaf, so ``network.edges.delay`` sweeps them the way ``network.conduction_speed`` sweeps a length graph's ``speed``. Before this the axis raised "no graph leaf to sweep" and varying a delay meant writing one experiment per value.

The history buffer is the trap: it is static, sized once outside jit, and a cell whose delay exceeds it reads the wrong history. These pin both halves -- the axis reaches the leaf, and the buffer is built for the longest delay the sweep can ask for.
"""

import numpy as np
import pytest

from .tvboptim_capabilities import needs_axis_wrap

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

_DELAY_AXIS_EXPERIMENT = """
id: 9
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
      distribution: {name: Uniform, domain: {lo: 0.0, hi: 6.283185}}
  output: [theta]
  number_of_modes: 1
execution: {random_seed: 0}
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - source: 0
      target: 1
      parameters:
        weight: {value: 0.5}
        delay: {value: 2.0}
    - source: 1
      target: 0
      parameters:
        weight: {value: 0.5}
        delay: {value: 2.0}
coupling:
  name: KuramotoCoupling
  label: KuramotoCoupling
  delayed: true
  parameters:
    a: {name: a, value: 0.05}
    N: {name: N, value: 1.0}
  pre_expression: {rhs: "sin(theta_j - theta_i)"}
  post_expression: {rhs: "a * gx / N"}
  incoming_states: [theta]
  local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 200.0
  step_size: 1.0
  transient_time: 0.0
explorations:
  sweep:
    name: sweep
    mode: product
    space:
      - parameter: network.edges.delay
        explored_values: [2.0, 40.0]
"""
"""A 2-node delayed Kuramoto whose exploration sweeps the per-edge delay itself.

The two swept values straddle the graph's own 2 ms delay by an order of magnitude, so a buffer sized only for the build-time delay would truncate the 40 ms cell and the two cells would agree. The initial phases are drawn from a Uniform under a fixed seed rather than left at the shared default, where ``sin(theta_j - theta_i)`` is identically zero and the coupling, delayed or not, does nothing.
"""


_ORDER_PARAMETER = """observations:
  order_parameter:
    label: "Kuramoto order parameter"
    source: [theta]
    pipeline:
      - function: kuramoto_order
        arguments:
          data:
            value: integration.result

"""
"""An observation on the sweep, so the run stacks a bundle and has to key it by the swept delay."""


def _experiment(tmp_path):
    spec = tmp_path / "delay_axis.yaml"
    spec.write_text(_DELAY_AXIS_EXPERIMENT)
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    return exp


def test_delay_axis_emits_the_delays_leaf(tmp_path):
    """The axis is classified network-scope, writes the graph's own leaf, and carries its label.

    A graph leaf's dataframe column is positional, so the declared path has to travel with the bound axis or the cell coordinates cannot be keyed on it.
    """
    code = _experiment(tmp_path).render_code("tvboptim")
    squeezed = "".join(code.split()).replace('"', "'")
    assert "grid_state.graph.delays=_ax('network.edges.delay',DataAxis(" in squeezed


def test_delay_axis_sizes_the_history_buffer_for_the_longest_delay(tmp_path):
    """The pre-jit rebuild covers the largest swept delay, not just the graph's own."""
    code = _experiment(tmp_path).render_code("tvboptim")
    rebuild = [ln for ln in code.splitlines() if "max_delay_bound=max(" in ln]
    assert rebuild, "no delay-graph rebuild sized by the swept maximum"
    assert "40.0" in rebuild[0], rebuild[0]


@needs_axis_wrap
def test_delay_axis_actually_changes_the_trajectory(tmp_path):
    """Two delays, two trajectories: the sweep must not be silently inert."""
    sweep = _experiment(tmp_path).run("tvboptim").explorations["sweep"]
    assert sweep.results.dims[0] == "network.edges.delay", sweep.results.dims
    theta = np.asarray(sweep.results)
    assert theta.shape[0] == 2, theta.shape
    assert not np.allclose(theta[0], theta[1]), "the swept delay left the trajectory unchanged"


@needs_axis_wrap
def test_delay_axis_coordinate_is_the_swept_delay(tmp_path):
    """The grid coordinate is the scalar that was swept, not the matrix it was written across.

    A per-edge leaf takes the whole N x N matrix per cell, so reading the grid back verbatim gave an object column of matrices and every keyed path raised on hashing one.
    """
    spec = tmp_path / "delay_obs.yaml"
    spec.write_text(_DELAY_AXIS_EXPERIMENT.replace("explorations:", _ORDER_PARAMETER + "explorations:"))
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    sweep = exp.run("tvboptim").explorations["sweep"]
    np.testing.assert_allclose(sweep.cell_coords["network.edges.delay"], [2.0, 40.0])
    obs = sweep.observations["order_parameter"]
    assert obs.dims[0] == "network.edges.delay", obs.dims
    np.testing.assert_allclose(obs.coords["network.edges.delay"].values, [2.0, 40.0])


def test_delay_axis_on_a_tract_length_network_is_refused(tmp_path):
    """Lengths win over edge delays, so sweeping `delays` there has no leaf to reach."""
    spec = tmp_path / "length_net.yaml"
    spec.write_text(_DELAY_AXIS_EXPERIMENT.replace("delay: {value: 2.0}", "distance: {value: 30.0}"))
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    code = exp.render_code("tvboptim")
    assert "sweep `network.conduction_speed` instead" in code
