"""A kit freezes the script a run renders, so the frozen script integrates what ``run()`` integrates when every delay of the network vanishes.

A kit freezes its backend script with `_freeze_backend_script`, and ``tvbo run --rendered`` runs that script through `SimulationExperiment.run`, which configures the experiment first. When every delay vanishes, at an infinite conduction speed or on tract lengths of zero, configure turns the couplings' declared `delayed` off, so a run renders an instantaneous coupling. A script rendered without configuring keeps the declared delayed coupling at zero delay, which reads the incoming states from a history buffer that opens on the dynamics' default state, not on the drawn phases, so its first step couples different phases and the trajectories part from there. The two integrate different systems, against `run`'s promise that a frozen script and the from-spec render give identical results.

A script the freeze cannot render leaves the kit to run that experiment in spec mode, which the freeze says in a warning rather than an info line.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment
from tvbo.cli.workflow import _freeze_backend_script

pytest.importorskip("tvboptim")
pytestmark = pytest.mark.backend_tvboptim

N = 3
WEIGHTS = 0.5 * np.roll(np.eye(N), 1, axis=1) + 0.3 * np.roll(np.eye(N), -1, axis=1)
"""A ring with unequal weights each way, stored dense, so either script reads the weights the run hands it."""

VANISHING = {
    "infinite-speed": (float("inf"), 40.0 * (WEIGHTS > 0)),
    "zero-lengths": (3.0, np.zeros((N, N))),
}
"""The two ways every delay vanishes, as (conduction speed in mm/ms, tract lengths in mm)."""

SPEC = {
    "id": 1,
    "label": "three phase oscillators whose delays all vanish",
    "dynamics": {
        "name": "Kuramoto",
        "parameters": {"omega": {"value": 0.0628}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {
            "theta": {
                "equation": {"lhs": "Derivative(theta, t)", "rhs": "omega + c"},
                "variable_of_interest": True,
                "coupling_variable": True,
                "distribution": {"name": "Uniform", "domain": {"lo": 0.0, "hi": 6.283185307179586}},
            }
        },
        "output": ["theta"],
    },
    "network": {
        "number_of_nodes": N,
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": 3.0}},
        "coupling": {
            "KuramotoCoupling": {
                "delayed": True,
                "parameters": {"a": {"value": 0.2}, "N": {"value": 1.0}},
                "pre_expression": {"rhs": "sin(theta_j - theta_i)"},
                "post_expression": {"rhs": "a * gx / N"},
                "incoming_states": ["theta"],
                "local_states": ["theta"],
            }
        },
    },
    "integration": {"method": "RungeKutta4thOrder", "step_size": 0.5, "duration": 20.0, "transient_time": 0.0},
    "execution": {"random_seed": 0, "precision": "float64"},
}
"""Koller2024's zero-delay control in miniature: a Kuramoto coupling declared delayed, drawn phases and RK4, 40 steps."""


def _experiment(case):
    speed, lengths = VANISHING[case]
    spec = copy.deepcopy(SPEC)
    spec["network"]["parameters"]["conduction_speed"]["value"] = speed
    exp = SimulationExperiment(**spec)
    exp.network.set_matrix("weight", WEIGHTS)
    exp.network.set_matrix("length", lengths)
    return exp


def _frozen(case, kit):
    """The tvboptim script a kit under *kit* freezes for *case*, read back from the kit."""
    assert _freeze_backend_script(_experiment(case), kit, "tvboptim", "1") == "scripts/1.py"
    return (kit / "scripts" / "1.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("case", list(VANISHING))
def test_a_kit_freezes_the_script_a_run_renders_when_every_delay_vanishes(case, tmp_path):
    """The frozen script is the configured render, which builds no delayed graph, though the couplings declare themselves delayed."""
    assert "DenseLengthGraph(" in _experiment(case).render("tvboptim"), "the declared render must be delayed"
    frozen = _frozen(case, tmp_path)
    assert "DenseLengthGraph(" not in frozen
    assert frozen == _experiment(case).render_code("tvboptim")


@pytest.mark.parametrize("case", list(VANISHING))
def test_a_frozen_script_integrates_what_run_integrates_when_every_delay_vanishes(case, tmp_path):
    """Running the frozen script gives the trajectory ``run()`` gives, element for element."""
    frozen = _experiment(case).run("tvboptim", rendered_code=_frozen(case, tmp_path))
    reference = _experiment(case).run("tvboptim")
    data, expected = np.asarray(frozen.integration.data), np.asarray(reference.integration.data)
    assert np.ptp(expected[-1]) > 1e-3, "the phases must differ for the comparison to mean anything"
    np.testing.assert_array_equal(data, expected)


LAYERED = """
id: 1
label: Layered
dynamics:
  name: Leak
  parameters: {tau: {value: 5.0}}
  state_variables:
    x: {initial_value: 0.0, equation: {rhs: "-x / tau + c"}, coupling_variable: true}
  coupling_inputs: {c: {}}
  output: [x]
network:
  label: Regions
  number_of_nodes: 2
  node_template: {subnetwork: {label: Units, number_of_nodes: 3, node_template: {dynamics: Leak}}}
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x, target_var: c, directed: true}
integration: {method: euler, step_size: 0.1, duration: 1.0, unit: ms}
"""
"""An experiment no backend renders: its network places a subnetwork on every node, and tvbo builds no strategy that lowers it."""


def test_a_script_the_kit_cannot_freeze_is_reported_as_a_warning(tmp_path, monkeypatch):
    """A render that fails at freeze is a warning naming the experiment and the error, and saying the kit runs it in spec mode."""
    warned = []
    monkeypatch.setattr("tvbo.cli._common.warn", warned.append)
    assert _freeze_backend_script(SimulationExperiment.from_string(LAYERED), tmp_path, "tvboptim", "layered") is None
    (message,) = warned
    assert "experiment layered" in message
    assert "subnetworks" in message
    assert "its kit runs it in spec mode" in message
