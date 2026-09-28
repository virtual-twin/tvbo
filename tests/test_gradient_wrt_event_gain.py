"""A gradient observation taken with respect to a stimulus's per-node gain returns one sensitivity per node.

Three leaky nodes; the first and the third drive the second, with weights 1 and 1/4, and a stimulus with a separate gain at each node enters at the first and the third. The observation differentiates the second node's summed trace, a pipeline observation, with respect to those gains. The network is linear and starts at rest, so the exact answer is known without a reference run: the gain at the untargeted node has no effect, the first node's gain matters four times as much as the third's, and the gradient dotted with the gains returns the observation itself. Each case runs three times: in reverse mode, again with a checkpoint interval and a truncation window declared for a fit (the gradient keeps the checkpoints, which leave it unchanged, and ignores the truncation, which would cut the sums short), and in forward mode, which answers the same.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo.classes.experiment import SimulationExperiment

_SPEC = """
id: 1
functions:
  total:
    source_code: jnp.sum(data[:, 0, 1])
    arguments: {data: {}}
dynamics:
  name: Leak
  parameters:
    k: {value: 0.1}
  coupling_inputs:
    c: {}
  state_variables:
    x:
      equation: {rhs: -k * x + c}
      initial_value: 0.0
      coupling_variable: true
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
    - {id: 2, label: r2}
  edges:
    - {source: 0, target: 1, weight: 1.0}
    - {source: 2, target: 1, weight: 0.25}
  coupling:
    LinCoupling:
      parameters:
        G: {value: 0.05}
      pre_expression: {rhs: x_j}
      post_expression: {rhs: G * gx}
      incoming_states: [x]
      local_states: [x]
integration: {method: Euler, duration: 50.0, step_size: 0.1, transient_time: 0.0DIFFERENTIATION}
events:
  drive:
    event_type: stimulus
    equation: {rhs: amplitude}
    parameters:
      amplitude: {value: 1.0, shape: "(n_nodes,)"}
    target_variable: x
    target_regions: [r0, r2]
observations:
  trace:
    source: [x]
    aggregation: none
    record: true
  received:
    source: [x]
    pipeline:
      - output: received
        function: total
        arguments: {data: {value: trace}}
  sensitivity:
    analysis:
      type: gradient
      target: received
      wrt: [x.amplitude]
      parameters: {mode: {value: MODE}}
"""


CHECKPOINTED = ", differentiation: {checkpoint_interval: 10.0, truncation_window: 20.0}"


@pytest.fixture(
    scope="module",
    params=[("", "reverse"), (CHECKPOINTED, "reverse"), ("", "forward")],
    ids=["plain", "checkpointed", "forward"],
)
def observed(tmp_path_factory, request):
    differentiation, mode = request.param
    root = tmp_path_factory.mktemp("gain")
    spec = root / "gain.yaml"
    spec.write_text(_SPEC.replace("DIFFERENTIATION", differentiation).replace("MODE", mode))
    obs = SimulationExperiment.from_file(str(spec)).run(format="tvboptim", results_root=str(root)).observations
    return {name: np.asarray(getattr(getattr(obs, name), "data", getattr(obs, name))) for name in ("received", "sensitivity")}


def test_one_sensitivity_per_node(observed):
    assert observed["sensitivity"].shape == (3,)


def test_the_sensitivities_follow_the_wiring(observed):
    first, untargeted, third = observed["sensitivity"]
    assert untargeted == 0.0
    assert first > 0.0
    np.testing.assert_allclose(first, 4.0 * third, rtol=1e-9)


def test_the_gradient_dotted_with_the_gains_returns_the_observation(observed):
    np.testing.assert_allclose(observed["sensitivity"].sum(), observed["received"], rtol=1e-9)


def test_an_unknown_gradient_mode_is_refused(tmp_path):
    spec = tmp_path / "gain.yaml"
    spec.write_text(_SPEC.replace("DIFFERENTIATION", "").replace("MODE", "sideways"))
    with pytest.raises(ValueError, match="sideways"):
        SimulationExperiment.from_file(str(spec)).run(format="tvboptim", results_root=str(tmp_path))


@pytest.mark.parametrize(("mode", "transform"), [("reverse", "jax.grad"), ("forward", "jax.jacfwd")])
def test_the_declared_mode_picks_the_transform(tmp_path, mode, transform):
    """Both modes answer the same, so only the rendered source shows which one differentiated."""
    spec = tmp_path / "gain.yaml"
    spec.write_text(_SPEC.replace("DIFFERENTIATION", "").replace("MODE", mode))
    code = SimulationExperiment.from_file(str(spec)).render_code("tvboptim")
    assert f"obs.sensitivity = {transform}(_grad_of_sensitivity)" in code
