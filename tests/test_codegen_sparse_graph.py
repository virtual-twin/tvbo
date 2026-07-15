"""Codegen: ``Network.graph_representation: sparse`` selects a ``SparseGraph`` — but only
when every coupling is instantaneous AND vectorized (source-only ``pre``). A per-edge
``pre`` (e.g. ``sin(x_j - x_i)``) or a delayed coupling would hit ``jsparse.sparsify`` on a
nonlinear term and crash, so those networks must fall back to a dense graph.

These pin the ``use_sparse`` gate in ``tvbo-tvboptim-experiment.py.mako`` — ``render_code``
only needs the network metadata (the generated ``create_network`` takes ``weights`` as a
runtime argument), so no connectivity source is required to check which graph class is emitted.
"""
import copy

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment, database_path

EXPERIMENTS_DIR = database_path / "experiments"

# Minimal instantaneous Kuramoto with a VECTORIZED (source-only) grid coupling in the
# factored angle-addition form: sum_j A_ij sin(theta_j - theta_i)
#   = cos(theta_i)*(A @ sin theta) - sin(theta_i)*(A @ cos theta).
MINI_EXP = {
    "id": 1,
    "label": "sparse-codegen unit fixture",
    "dynamics": {
        "name": "MiniKuramoto",
        "system_type": "continuous",
        "output": ["theta"],
        "parameters": {"K": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "theta": {"equation": {"rhs": "K * c"}, "initial_value": 0.0, "coupling_variable": True},
        },
    },
    "network": {
        "number_of_nodes": 4,
        "graph_representation": "sparse",
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["theta"],
                "pre_expression": {"rhs": "[sin(theta), cos(theta)]"},
                "post_expression": {"rhs": "cos(theta_i)*gx_0 - sin(theta_i)*gx_1"},
            }
        },
    },
    "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0,
                    "transient_time": 0.0, "unit": "s"},
}


def _graph_line(spec):
    """The ``graph = ...Graph(...)`` line the tvboptim experiment codegen emits for ``spec``."""
    code = SimulationExperiment(**spec).render_code("tvboptim")
    return next((ln.strip() for ln in code.splitlines() if "graph = " in ln and "Graph(" in ln), "")


def test_sparse_vectorized_instantaneous_emits_sparsegraph():
    """graph_representation: sparse + a vectorized instantaneous coupling -> SparseGraph."""
    assert "SparseGraph(" in _graph_line(MINI_EXP)


def test_sparse_peredge_coupling_falls_back_to_dense():
    """A per-edge pre (sin(x_j - x_i)) is not a source-only mat-vec — sparsify would crash on
    it — so sparse must fall back to DenseGraph even though graph_representation is sparse."""
    spec = copy.deepcopy(MINI_EXP)
    spec["network"]["coupling"]["c"] = {
        "delayed": False,
        "incoming_states": ["theta"],
        "local_states": ["theta"],
        "pre_expression": {"rhs": "sin(x_j - x_i)"},
        "post_expression": {"rhs": "gx"},
    }
    line = _graph_line(spec)
    assert "DenseGraph(" in line and "SparseGraph(" not in line


def test_default_representation_is_dense():
    """No graph_representation (defaults to 'auto') keeps the pre-existing DenseGraph emit."""
    spec = copy.deepcopy(MINI_EXP)
    spec["network"].pop("graph_representation")
    line = _graph_line(spec)
    assert "DenseGraph(" in line and "SparseGraph(" not in line


def test_sparse_ignored_for_delayed_network():
    """Opting a DELAYED network into sparse must stay DenseDelayGraph (θ_j(t-τ_ij) is per-edge;
    sparsify can't apply a nonlinear pre to it)."""
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "JR_MEG_FrequencyGradient_Optimization.yaml"))
    exp.network.graph_representation = "sparse"
    code = exp.render_code("tvboptim")
    assert "DenseDelayGraph(" in code and "SparseGraph(" not in code
