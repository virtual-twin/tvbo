"""A script frozen while its couplings were delayed runs once the bound network turns out to have no delays.

A kit freezes its backend script with `render`, which takes the declared `delayed` flags as they stand, and ``tvbo run --rendered`` configures the experiment before it runs that script. When the bound network gives every edge a zero delay (an infinite conduction speed, or tract lengths that read a reference no run had bound at freeze time), configure turns every coupling's `delayed` off, and the run path hands the script its weights in the form the graph it would emit now takes: a BCOO when the network stores them sparse. The script builds a dense length graph, and ``jnp.array`` walked the BCOO element by element (about 1.4 h on a 1000-node connectome) before raising ``setting an array element with a sequence``. Past that, the script spelled its infinite speed ``inf``, a name it does not define.

The frozen script integrates a delayed coupling whose delays are all zero, which is what it was rendered to integrate. The reference is the same experiment at a conduction speed fast enough that every delay rounds to zero steps and slow enough that configure keeps the couplings delayed: rendered and run in one go, it integrates by the same script on the dense weights the run path hands a delayed graph.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

N = 6
WEIGHTS = sparse.csr_matrix(0.5 * np.roll(np.eye(N), 1, axis=1) + 0.3 * np.roll(np.eye(N), -1, axis=1))
"""A ring with unequal weights each way, stored csr, so the run path hands a graph without delays a BCOO."""
LENGTHS = 40.0 * (WEIGHTS.toarray() > 0)
"""Tract lengths in mm on the ring's edges."""
FAST = 4000.0
"""A conduction speed in mm/ms whose delays (0.01 ms) lie above configure's zero tolerance and round to zero steps of 0.5 ms."""

SPEC = {
    "id": 1,
    "label": "six phase oscillators at an infinite conduction speed",
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
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": float("inf")}},
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
    "integration": {"method": "RungeKutta4thOrder", "step_size": 0.5, "duration": 40.0, "transient_time": 0.0},
    "execution": {"random_seed": 0, "precision": "float64"},
}
"""Koller2024's zero-delay control in miniature: a Kuramoto coupling declared delayed on a network whose conduction speed is infinite."""


def _experiment(speed=float("inf")):
    spec = copy.deepcopy(SPEC)
    spec["network"]["parameters"]["conduction_speed"]["value"] = speed
    exp = SimulationExperiment(**spec)
    exp.network.set_matrix("weight", WEIGHTS)
    exp.network.set_matrix("length", LENGTHS)
    return exp


def test_the_case_is_a_dense_script_handed_a_bcoo_against_a_delayed_reference():
    """The frozen script builds a dense length graph, the configured run path hands it a BCOO, and the reference keeps its couplings delayed."""
    from jax.experimental.sparse import BCOO

    from tvbo.adapters.tvboptim import solver_weights

    assert "DenseLengthGraph(" in _experiment().render("tvboptim")
    configured = _experiment()
    configured.configure()
    assert isinstance(solver_weights(configured.network), BCOO)
    assert "DenseLengthGraph(" in _experiment(FAST).render_code("tvboptim")


def test_a_frozen_script_runs_on_a_network_whose_delays_vanish():
    """The script frozen with the declared delays runs on the BCOO the run path hands it, and integrates what its delayed twin integrates in one go."""
    frozen = _experiment().run("tvboptim", rendered_code=_experiment().render("tvboptim"))
    reference = _experiment(FAST).run("tvboptim")
    data, expected = np.asarray(frozen.integration.data), np.asarray(reference.integration.data)
    assert data.shape == expected.shape
    assert np.ptp(expected[-1]) > 1e-3, "the phases must differ for the comparison to mean anything"
    np.testing.assert_allclose(data, expected, rtol=0, atol=1e-9)
