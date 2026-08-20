"""A stimulus weighting may be DERIVED, and then it must declare where it came from.

`Event.weights` is a literal list and `weight_distribution` samples one, but a real spatial stimulus is usually neither: it is an ROI mask projected through a basis, a retinotopic map, a measured stimulation field. Inlining a few hundred such numbers into the spec hides their provenance and makes them uncheckable, which is the exact thing `source:`/`producer:` exist to prevent for every other external array.

`Event.weight_parameter` closes that: the weighting is a Parameter, so it carries the same provenance triple, and it is resolved into the per-node weighting at load time — the codegen downstream still sees the plain array it already consumed.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from tvbo import SimulationExperiment

_SPEC = """
id: 3
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
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c + drive"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
events:
  drive:
    name: drive
    event_type: stimulus
    label: "Spatially patterned drive"
    equation: {rhs: "Piecewise((1.0, t >= 0.01), (0, True))"}
# WEIGHTS
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
    - {id: 2, label: r2}
  edges:
    - {source: 0, target: 1, weight: 0.5}
    - {source: 1, target: 0, weight: 0.5}
coupling:
  name: KuramotoCoupling
  label: KuramotoCoupling
  parameters:
    a: {name: a, value: 0.01}
    N: {name: N, value: 1.0}
  pre_expression: {rhs: "sin(theta_j - theta_i)"}
  post_expression: {rhs: "a * gx / N"}
  incoming_states: [theta]
  local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 20.0
  step_size: 1.0
  transient_time: 0.0
"""

_PRODUCER_MODULE = """
import numpy as np

def roi_pattern(scale=1.0):
    \"\"\"Stand-in for a mask projected through a basis: a derived per-node weighting.\"\"\"
    return np.array([0.25, 0.5, 1.0]) * float(scale)

def wrong_length():
    return np.array([1.0, 2.0])
"""


@pytest.fixture
def producer_module(tmp_path, monkeypatch):
    """Make a producer callable importable by bare module name, as `code/` would be."""
    (tmp_path / "wp_producer.py").write_text(textwrap.dedent(_PRODUCER_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("wp_producer", None)
    yield "wp_producer"
    sys.modules.pop("wp_producer", None)


def _experiment(tmp_path, weights_block):
    path = tmp_path / "spec.yaml"
    path.write_text(_SPEC.replace("# WEIGHTS\n", textwrap.indent(weights_block, "    ")))
    exp = SimulationExperiment.from_file(str(path))
    exp.configure()
    return exp


def _event(exp):
    events = exp.events or {}
    return next(iter(events.values()))


def _produced(module, name="roi_pattern", scale=None):
    args = f"\n    arguments:\n      scale: {{value: {scale}}}" if scale is not None else ""
    return (
        "weight_parameter:\n"
        "  name: v1_weighting\n"
        "  description: 'ROI mask projected into the basis'\n"
        "  producer:\n"
        f"    callable: {{module: {module}, name: {name}}}"
        f"{args}\n"
        "nodes: [0, 1, 2]\n"
    )


def test_produced_weighting_resolves_into_the_event(tmp_path, producer_module):
    """The declared producer runs and its array becomes the per-node weighting."""
    exp = _experiment(tmp_path, _produced(producer_module))
    np.testing.assert_allclose(_event(exp).weights, [0.25, 0.5, 1.0])


def test_producer_arguments_are_honoured(tmp_path, producer_module):
    exp = _experiment(tmp_path, _produced(producer_module, scale=4.0))
    np.testing.assert_allclose(_event(exp).weights, [1.0, 2.0, 4.0])


def test_weighting_length_must_match_the_targeted_nodes(tmp_path, producer_module):
    """A weighting is aligned with `nodes`; a mismatch is a spec error, not a broadcast."""
    block = _produced(producer_module, name="wrong_length")
    with pytest.raises(ValueError, match="targets 3 node"):
        _experiment(tmp_path, block)


def test_parameter_without_any_value_is_refused(tmp_path):
    block = "weight_parameter:\n  name: v1_weighting\n  description: 'declares nothing'\nnodes: [0, 1, 2]\n"
    with pytest.raises(ValueError, match="no `value`, `source` or `producer`"):
        _experiment(tmp_path, block)


def test_a_literal_weighting_still_wins(tmp_path, producer_module):
    """The explicit array is the most specific statement; it must not be overwritten."""
    block = _produced(producer_module) + "weights: [9.0, 8.0, 7.0]\n"
    exp = _experiment(tmp_path, block)
    np.testing.assert_allclose(_event(exp).weights, [9.0, 8.0, 7.0])


def test_a_literal_value_on_the_parameter_also_resolves(tmp_path):
    """`value:` is the degenerate case of the same slot — no provenance, but no special path."""
    block = "weight_parameter:\n  name: v1_weighting\n  value: [0.1, 0.2, 0.3]\nnodes: [0, 1, 2]\n"
    np.testing.assert_allclose(_event(_experiment(tmp_path, block)).weights, [0.1, 0.2, 0.3])


def test_weighting_defaults_to_every_node_when_none_are_named(tmp_path, producer_module):
    """With no `nodes:`, a produced weighting spans the network, like a sampled one does."""
    block = _produced(producer_module).replace("nodes: [0, 1, 2]\n", "")
    exp = _experiment(tmp_path, block)
    assert list(_event(exp).nodes) == [0, 1, 2]
    np.testing.assert_allclose(_event(exp).weights, [0.25, 0.5, 1.0])
