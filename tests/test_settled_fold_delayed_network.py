"""A cell that folds its streamed observations after its own settle runs on a delayed network.

An exploration cell, and the evaluation after an algorithm has tuned, runs the declared settle under its own values and then folds its recorded observations in-carry over the measured window. The fold is prepared inside the traced cell, because a kernel's ring opens on the settle of the run it measures, and prepare() asks the network for its delay history. The network the fold was prepared on was the base run's, which holds the base run's settled history, and tvboptim reads a stored history through Python branches on its time stamps. Inside the trace those are tracers, so every such cell on a delayed network with a declared settle died with ``TracerBoolConversionError`` before it integrated a step. The delay lines a cell reads are the ones its own settle wrote, which its state carries, so the history the network holds is never what it reads. The reference is the experiment run alone at the run's values: the base run for a cell at the base values, and a run at the tuned value for the evaluation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

LENGTHS = [[0.0, 30.0, 0.0], [45.0, 0.0, 12.0], [21.0, 60.0, 0.0]]
"""Tract lengths in mm, zero where there is no edge, so at 3 mm/ms the delays span 4 to 20 ms."""

SPEC = {
    "id": 1,
    "label": "three delayed phase oscillators",
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
        "number_of_nodes": 3,
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": 3.0}},
        "nodes": [{"id": i, "label": f"r{i}"} for i in range(3)],
        "edges": [
            {"source": j, "target": i, "parameters": {"weight": {"value": 0.5}, "length": {"value": length}}}
            for i, row in enumerate(LENGTHS)
            for j, length in enumerate(row)
            if length
        ],
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
    "observations": {"mean_theta": {"source": ["theta"], "aggregation": "mean", "reduce": "streaming"}},
    "integration": {
        "method": "RungeKutta4thOrder",
        "step_size": 1.0,
        "duration": 40.0,
        "transient_time": 30.0,
        "block_size": 10,
    },
    "execution": {"random_seed": 0, "precision": "float64"},
}
"""Koller2024's cortical shape in miniature: a delayed Kuramoto network on tract lengths, a declared settle, and an observation reduced in-carry."""

TRIALS = {"trials": {"name": "trials", "record": ["mean_theta"], "n_trials": 3}}
SWEEP = {
    "sweep": {
        "name": "sweep",
        "mode": "product",
        "record": ["mean_theta"],
        "space": [{"parameter": "Kuramoto.omega", "explored_values": [0.0628, 0.2]}],
    }
}
TUNING = {
    "tune": {
        "objective": {"type": "activity_target", "target_variable": "theta", "target_value": 0.5},
        "observations": ["mean_theta"],
        "update_rules": [{"name": "omega_update", "target_parameter": {"name": "omega"}, "equation": {"rhs": "omega + eta"}}],
        "hyperparameters": [{"name": "eta", "value": 0.01}],
        "n_iterations": 2,
        "simulation_period": 20.0,
    }
}
"""A rule that moves omega off its declared value, so the evaluation after tuning is a run at values no base run has."""


def _noisy(spec):
    spec = copy.deepcopy(spec)
    spec["dynamics"]["state_variables"]["theta"]["noise"] = {
        "additive": True,
        "gaussian": True,
        "parameters": {"sigma": {"value": 0.01}},
    }
    spec["integration"]["method"] = "Heun"
    return spec


VARIANTS = {"deterministic": SPEC, "noisy": _noisy(SPEC)}


def _with(spec, **sections):
    """*spec* with the given top-level sections added."""
    spec = copy.deepcopy(spec)
    spec.update(copy.deepcopy(sections))
    return spec


def _cells(result, name):
    return np.asarray(result.explorations[name].observations["mean_theta"]).reshape(-1, 3)


def _values(observation):
    return np.asarray(getattr(observation, "data", observation)).reshape(-1)


@pytest.mark.parametrize(
    ("sections", "call"),
    [({"explorations": TRIALS}, "_settled_fold(_network"), ({"algorithms": TUNING}, "post_model_fn = _settled_fold(")],
    ids=["exploration", "evaluation"],
)
def test_the_cases_fold_through_the_settled_path(sections, call):
    """The cases below reach the fold prepared per call, the path the failure was on."""
    assert call in SimulationExperiment(**_with(SPEC, **sections)).render_code("tvboptim")


@pytest.mark.parametrize("variant", VARIANTS)
def test_trials_on_a_delayed_network_fold_after_their_own_settle(variant):
    """Each trial integrates its settle and folds its window: one finite row per trial, and the trials' random openings tell them apart."""
    cells = _cells(SimulationExperiment(**_with(VARIANTS[variant], explorations=TRIALS)).run("tvboptim"), "trials")
    assert cells.shape == (3, 3) and np.isfinite(cells).all()
    assert np.ptp(cells, axis=0).max() > 1e-3, "trials from different openings must differ"


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_swept_cell_at_the_base_values_folds_what_the_base_run_folds(variant):
    """The cell at the base run's values is the base run, so it folds the same observation, which reads delay lines its own settle wrote."""
    result = SimulationExperiment(**_with(VARIANTS[variant], explorations=SWEEP)).run("tvboptim")
    cells = _cells(result, "sweep")
    assert np.abs(cells[1] - cells[0]).max() > 1e-3, "the two cells must differ for the comparison to mean anything"
    np.testing.assert_allclose(cells[0], _values(result.observations.mean_theta), rtol=0, atol=1e-9)


def test_the_evaluation_after_tuning_folds_after_its_own_settle():
    """The evaluation after an algorithm has tuned folds through the same path, and is the experiment run alone at the tuned value."""
    tune = SimulationExperiment(**_with(SPEC, algorithms=TUNING)).run("tvboptim").algorithms["tune"]
    omega = float(_values(tune.state.dynamics.omega)[0])
    assert abs(omega - 0.0628) > 1e-3, "the rule must have moved omega for the comparison to mean anything"
    alone = copy.deepcopy(SPEC)
    alone["dynamics"]["parameters"]["omega"]["value"] = omega
    reference = _values(SimulationExperiment(**alone).run("tvboptim").observations.mean_theta)
    np.testing.assert_allclose(_values(tune.post_tuning.observations.mean_theta), reference, rtol=0, atol=1e-9)
