#
# Module: test_multiscale.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
"""Smoke tests for the reservoir multi-scale flattening (``tvbo.multiscale``).

The graph-generator engine (reservoir recurrence, SC loading, weight sampling)
is monkeypatched with tiny synthetic arrays so the lowering logic — Kronecker
assembly of ``W_global`` and the OOM size guard — is exercised without a real
network file.
"""
import numpy as np
import pytest

from tvbo import multiscale
from tvbo.graph_generators import engine

R = 3   # macro regions
N = 4   # reservoir units per region


def _minimal_experiment():
    """A minimal Experiment-A dict matching the pattern flatten_reservoir accepts."""
    return {
        "dynamics": {
            "name": "ESN",
            "parameters": {"tau": {"value": 2.0}},
            "state_variables": {
                "x": {"equation": {"rhs": "(1/tau) * (-x + tanh(W_int @ x))"}}
            },
        },
        "network": {
            "iri": "dummy://sc",
            "node_template": {
                "subnetwork": {
                    "graph_generator": {
                        "seed": 0,
                        "parameters": {
                            "n": {"value": N},
                            "weight_distribution": {
                                "distribution": {
                                    "name": "Normal",
                                    "parameters": {"mean": {"value": 0.0},
                                                   "std": {"value": 1.0}},
                                }
                            },
                        },
                    },
                    "edges": [],
                }
            },
            "coupling": {
                "Long": {
                    "parameters": {"kappa": {"value": 0.5}},
                    "post_expression": {"rhs": "kappa * gx"},
                }
            },
        },
        "integration": {"method": "Heun", "step_size": 1.0,
                        "duration": 10, "transient_time": 0},
    }


@pytest.fixture
def patched_engine(monkeypatch):
    """Replace the heavy engine calls with deterministic synthetic arrays."""
    def fake_run_generator(name, params, seed=None):
        n = int(params["n"])
        return {"weights": np.full((n, n), 0.1)}

    # SC with no self-connections
    sc = np.ones((R, R)) - np.eye(R)
    monkeypatch.setattr(engine, "run_generator", fake_run_generator)
    monkeypatch.setattr(engine, "load_matrix", lambda source: sc)
    monkeypatch.setattr(engine, "sample",
                        lambda dist, shape=None, seed=None: np.ones(shape))
    return sc


def test_flatten_reservoir_assembles_global_matrix(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment())

    assert fr.n_regions == R
    assert fr.n_units == N
    assert fr.tau == 2.0
    assert fr.activation == "tanh"
    # W_global is the dense (R·n)² Kronecker assembly
    assert fr.weights.shape == (R * N, R * N)
    assert np.isfinite(fr.weights).all()
    # flat scalar model wires coupling inside the activation
    assert "tanh(ReservoirInput)" in \
        fr.dynamics["state_variables"]["x"]["equation"]["rhs"]


def test_n_override_shrinks_reservoir(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment(), n_override=2)
    assert fr.n_units == 2
    assert fr.weights.shape == (R * 2, R * 2)


def test_size_guard_raises_before_allocation(patched_engine):
    # R·N = 12 flat nodes; cap below that must refuse to build W_global
    with pytest.raises(ValueError, match="max_flat_nodes"):
        multiscale.flatten_reservoir(_minimal_experiment(), max_flat_nodes=5)


def test_size_guard_disabled_with_none(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment(), max_flat_nodes=None)
    assert fr.weights.shape == (R * N, R * N)
