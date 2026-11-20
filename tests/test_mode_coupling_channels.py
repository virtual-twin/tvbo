"""The generated tvboptim dynamics declares each coupling input with the channel count of the coupling the network wires to it.

tvboptim checks a coupling's output channels against the dynamics' ``COUPLING_INPUTS`` when it builds the network. Under the mode fold a coupling that reads a multi-mode coupling variable emits one channel per mode, while the dynamics declared each input by its ``dimension`` alone, so every curated multi-mode model raised on a tvboptim with that check. The test reads both numbers from the rendered experiment, so it needs no tvboptim and holds on every release.
"""

from __future__ import annotations

import re

import pytest

from tvbo import Dynamics, SimulationExperiment

MODE_MODELS = ["ReducedSetHindmarshRose", "ReducedSetFitzHughNagumo", "StefanescuJirsa2D", "StefanescuJirsa3D"]


def _declared(code, block):
    """``{name: int}`` from the first ``<block> = {...}`` literal in *code*."""
    body = code[code.index(f"{block} = {{") : code.index("}", code.index(f"{block} = {{"))]
    return {name: int(n) for name, n in re.findall(r"['\"](\w+)['\"]:\s*(\d+)", body)}


def _n_output_states(code, cls):
    """The ``N_OUTPUT_STATES`` the coupling class *cls* declares in *code*."""
    return int(re.search(rf"class {cls}\(.*?N_OUTPUT_STATES = (\d+)", code, re.S).group(1))


@pytest.mark.parametrize("name", MODE_MODELS)
def test_each_wired_input_declares_its_coupling_channels(name):
    """Every coupling input the network wires declares the ``N_OUTPUT_STATES`` of the coupling class it is wired to."""
    experiment = SimulationExperiment(dynamics=Dynamics.from_db(name), integration={"duration": 1.0, "step_size": 0.1})
    code = experiment.render_code("tvboptim")
    declared = _declared(code, "COUPLING_INPUTS")
    wired = dict(re.findall(r"coupling_dict\[['\"](\w+)['\"]\] = (\w+)\(", code))
    assert wired, f"{name}: the render wires no coupling"
    produced = {ci: _n_output_states(code, cls) for ci, cls in wired.items()}
    assert {ci: declared[ci] for ci in wired} == produced
