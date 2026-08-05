"""``execution.accelerator`` decides the JAX platform, and it must be pinned before import.

JAX fixes its platform the first time it initialises, so a generated script has exactly
one chance to honour the declaration: an ``os.environ.setdefault("JAX_PLATFORMS", ...)``
emitted above ``import jax``. That makes the line load-bearing rather than cosmetic —
if it is missing, or names the wrong platform, the run silently lands on whatever
device JAX picked and nothing downstream can move it.

The mapping itself lives in one helper (:func:`tvbo.templates.tvboptim.utils.jax_platform`)
because three call sites need it — both codegen templates and the in-process analysis
renderer — and a fourth spelling of ``gpu -> cuda`` is how they drift apart.
"""

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment
from tvbo.templates.tvboptim.utils import jax_platform

_SPEC = """
id: 7
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
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
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
# EXECUTION
"""


def _render(tmp_path, execution_block):
    spec = _SPEC.replace("# EXECUTION\n", execution_block)
    path = tmp_path / "spec.yaml"
    path.write_text(spec)
    exp = SimulationExperiment.from_file(str(path))
    exp.configure()
    return exp.render_code("tvboptim")


@pytest.mark.parametrize("accelerator,platform", [("cpu", "cpu"), ("gpu", "cuda"), ("tpu", "tpu")])
def test_declared_accelerator_pins_the_platform_before_import(tmp_path, accelerator, platform):
    """Assert the contract, not the spelling: the emitted source is formatter-wrapped."""
    code = _render(tmp_path, f"execution:\n  accelerator: {accelerator}\n")
    pin = code.index("JAX_PLATFORMS")
    assert "os.environ.setdefault(" in code[max(0, pin - 80):pin]
    assert f'"{platform}"' in code[pin:pin + 80]
    assert pin < code.index("import jax"), "pin must precede the import"


@pytest.mark.parametrize("execution_block", ["", "execution:\n  accelerator: auto\n"])
def test_auto_leaves_detection_to_jax(tmp_path, execution_block):
    """'auto' is the absence of a pin, not a platform named 'auto'."""
    assert "JAX_PLATFORMS" not in _render(tmp_path, execution_block)


def test_helper_is_the_single_source_of_the_mapping():
    assert jax_platform("gpu") == "cuda"
    assert jax_platform("auto") is None and jax_platform(None) is None
    assert jax_platform("cpu") == "cpu" and jax_platform("TPU") == "tpu"
