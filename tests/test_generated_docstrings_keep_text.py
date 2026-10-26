r"""Recipe prose reaches the generated module's docstrings as written.

Descriptions and labels carry LaTeX, and a generated string literal reads their backslashes as escape sequences: `\sin` is an invalid one that warns on every parse, and `\tau`, `\rho`, `\nabla`, `\bar` and `\frac` are valid ones that turn into control characters without a word. A double quote can close the literal. Each case below renders a module and checks that it parses with warnings as errors and that its string constants hold the prose unchanged.
"""

import ast
import warnings

import pytest

from tvbo import SimulationExperiment

PROSE = (
    r'carrier $\sin(2\pi f t)$ with $\tau$, $\rho$, $\nabla$, $\bar r$ and $\frac{a}{b}$, a "quoted" word and a closing quote"'
)

_SPEC = """
id: 1
dynamics:
  name: TunedLeak
  label: TunedLeak
  description: {prose}
  parameters:
    g: {{value: 0.0, description: {prose}}}
  state_variables:
    x:
      initial_value: 0.0
      equation: {{rhs: "-x + drive - g"}}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 1
  nodes:
    - {{id: 0, label: r0}}
  edges: []
integration: {{method: euler, step_size: 1.0, duration: 10.0, transient_time: 0.0, unit: ms}}
events:
  drive:
    description: {prose}
    event_type: stimulus
    equation: {{rhs: "A * sin(t)"}}
    parameters:
      A: {{value: 1.0}}
functions:
  thin:
    description: {prose}
    source_code: "data[::2, 0, 0]"
    arguments:
      data: {{}}
observations:
  mean_x: {{label: {prose}, description: {prose}, source: [x], aggregation: mean}}
  thinned:
    label: {prose}
    description: {prose}
    source: [x]
    pipeline:
      - function: thin
        arguments: {{data: x}}
explorations:
  g_sweep:
    label: {prose}
    space:
      TunedLeak.g: {{explored_values: [0.0, 1.0]}}
execution: {{random_seed: 0}}
"""


def _experiment():
    return SimulationExperiment.from_string(_SPEC.format(prose=f"'{PROSE}'"))  # single-quoted YAML keeps backslashes literal


def _strings(code):
    """Every string constant of the parsed module, parsed with every warning raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tree = ast.parse(code)
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_the_spec_carries_the_prose_unchanged():
    experiment = _experiment()
    assert experiment.events["drive"].description == PROSE


@pytest.mark.parametrize("backend", ["tvboptim", "tvb"])
def test_generated_docstrings_hold_the_prose_as_written(backend):
    pytest.importorskip(backend)
    carrying = [text for text in _strings(_experiment().render_code(backend)) if "carrier" in text]
    assert carrying and all(PROSE in text for text in carrying)
