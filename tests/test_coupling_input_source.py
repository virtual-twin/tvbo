"""A coupling input is fed by the coupling its `source` names, and the coupling class is rendered with that input's channels.

The network wiring and the coupling class read one pairing of coupling functions to coupling inputs (`resolve_coupling_input_map`). The wiring read the inputs as a name-to-dimension dict, so an explicit `CouplingInput.source` was never seen and the inputs were paired by position; the coupling class read a second copy of the pairing that honoured `source` but did not broadcast one coupling to several inputs (G291). Each case renders the experiment and runs it on tvboptim, which checks every coupling's output channels against the dynamics' ``COUPLING_INPUTS``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tvbo import SimulationExperiment

_SPEC = """
dynamics:
  name: TwoInputs
  parameters:
    tau: {name: tau, value: 10.0}
  coupling_inputs:
INPUTS
  state_variables:
    x:
      equation: {lhs: "Derivative(x, t)", rhs: "(-x + XDRIVE) / tau"}
      initial_value: 0.1
      coupling_variable: true
      variable_of_interest: true
    y:
      equation: {lhs: "Derivative(y, t)", rhs: "(-y + YDRIVE) / tau"}
      initial_value: 0.2
      coupling_variable: true
      variable_of_interest: true
  output: [x, y]
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5}
    - {source: 1, target: 0, weight: 0.5}
  coupling:
COUPLINGS
integration:
  method: Euler
  duration: 5.0
  step_size: 0.1
"""

_SCALAR = """    {name}:
      incoming_states: [{state}]
      pre_expression: {{rhs: "{state}_j"}}
      post_expression: {{rhs: "gx"}}
"""

_PAIR = """    {name}:
      incoming_states: [x]
      pre_expression: {{rhs: "[x_j, 2 * x_j]"}}
      post_expression: {{rhs: "gx"}}
"""


def _experiment(inputs, couplings, xdrive, ydrive):
    spec = _SPEC.replace("INPUTS", inputs).replace("COUPLINGS", couplings).replace("XDRIVE", xdrive).replace("YDRIVE", ydrive)
    return SimulationExperiment.from_string(spec)


def _declared(code):
    """``{input: channels}`` from the dynamics' ``COUPLING_INPUTS`` literal in *code*."""
    start = code.index("COUPLING_INPUTS = {")
    body = code[start : code.index("}", start)]
    return {name: int(n) for name, n in re.findall(r"['\"](\w+)['\"]:\s*(\d+)", body)}


def _wired(code):
    """``{input: coupling class}`` from the network's coupling dict in *code*."""
    return dict(re.findall(r"coupling_dict\[['\"](\w+)['\"]\] = (\w+)\(", code))


def _n_output_states(code, cls):
    """The ``N_OUTPUT_STATES`` the coupling class *cls* declares in *code*."""
    return int(re.search(rf"class {cls}\(.*?N_OUTPUT_STATES = (\d+)", code, re.S).group(1))


def _swapped():
    """Inputs ``a`` (two channels) and ``b`` (one) whose sources pair them against the couplings' declaration order: ``a`` from ``Cy``, ``b`` from ``Cx``."""
    inputs = "    a: {name: a, dimension: 2, keys: [a_e, a_i], source: Cy}\n    b: {name: b, dimension: 1, source: Cx}\n"
    couplings = _SCALAR.format(name="Cx", state="y") + _PAIR.format(name="Cy")
    return _experiment(inputs, couplings, "a_e + b", "a_i")


def _broadcast():
    """One coupling ``C`` of two channels and two inputs of two channels each, neither naming a source: ``C`` feeds both."""
    inputs = "    a: {name: a, dimension: 2, keys: [a_e, a_i]}\n    b: {name: b, dimension: 2, keys: [b_e, b_i]}\n"
    return _experiment(inputs, _PAIR.format(name="C"), "a_e + b_i", "a_i + b_e")


def test_an_input_is_wired_to_the_coupling_its_source_names():
    """``a`` receives ``Cy`` and ``b`` receives ``Cx``, as their sources say, not the couplings in declaration order."""
    code = _swapped().render_code("tvboptim")
    wired = _wired(code)
    assert set(wired) == {"a", "b"}, wired
    assert "Cy" in wired["a"] and "Cx" in wired["b"], wired


@pytest.mark.parametrize("build", [_swapped, _broadcast], ids=["swapped-sources", "broadcast"])
def test_each_input_declares_the_channels_of_the_coupling_it_receives(build):
    """Every wired input declares its own dimension, and the coupling class wired to it emits that many channels."""
    code = build().render_code("tvboptim")
    declared, wired = _declared(code), _wired(code)
    assert wired, "the render wires no coupling"
    assert (
        {ci: declared[ci] for ci in wired}
        == {ci: _n_output_states(code, cls) for ci, cls in wired.items()}
        == {"a": 2, "b": declared["b"]}
    )


@pytest.mark.parametrize("build", [_swapped, _broadcast], ids=["swapped-sources", "broadcast"])
def test_the_paired_couplings_run_on_tvboptim(build):
    """The tvboptim run builds the network, checking each coupling's channels against ``COUPLING_INPUTS``, and integrates it."""
    pytest.importorskip("tvboptim")

    result = build().run("tvboptim")
    data = np.asarray(result.integration.data)
    assert data.size and np.isfinite(data).all()
