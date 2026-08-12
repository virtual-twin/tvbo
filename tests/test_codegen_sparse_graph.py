"""Codegen: which graph class ``Network.graph_representation`` selects.

``sparse`` stores the connectome as BCOO so each coupling reduction is an O(nnz)
edge-sum (``segment_sum`` over prepared edges) rather than a dense NxN matmul. It
covers instantaneous networks (``SparseGraph``) and networks with explicit per-edge
delays (``SparseDelayGraph``), under any coupling form: tvboptim gathers per edge, so
a per-edge ``pre`` such as ``sin(x_j - x_i)`` is native. Nothing constructs a sparse
matrix, so the old "sparsify would crash on a nonlinear term" fallbacks are gone.

Tract lengths are the one combination sparse cannot express: delays there are
``lengths / conduction_speed`` recomputed each pass, which needs ``DenseLengthGraph``'s
live ``speed`` leaf (swept by a ``network.conduction_speed`` axis, and differentiable).
That is rejected at codegen instead of being silently downgraded.

These pin the ``use_sparse`` gate in ``tvbo-tvboptim-experiment.py.mako``. ``render_code``
only needs the network metadata (the generated ``create_network`` takes ``weights`` as a
runtime argument), so no connectivity source is required to check which class is emitted;
one test additionally runs the emitted module so sparse/dense agreement is checked, not
just the class name.
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
    "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0, "unit": "s"},
}


def _graph_line(spec):
    """The ``graph = ...Graph(...)`` line the tvboptim experiment codegen emits for ``spec``."""
    code = SimulationExperiment(**spec).render_code("tvboptim")
    return next((ln.strip() for ln in code.splitlines() if "graph = " in ln and "Graph(" in ln), "")


def test_sparse_vectorized_instantaneous_emits_sparsegraph():
    """graph_representation: sparse + a vectorized instantaneous coupling -> SparseGraph."""
    assert "SparseGraph(" in _graph_line(MINI_EXP)


def test_sparse_peredge_coupling_stays_sparse():
    """A per-edge pre (sin(x_j - x_i)) is gathered per edge, so it stays sparse.

    tvboptim's sparse path reads `incoming_states[:, source_e]` and passes
    `target_messages` for a pre that also reads the target, then reduces with
    segment_sum — no sparse matrix is ever built, so there is no nonlinear-term
    crash to avoid and no reason to downgrade to dense.
    """
    spec = copy.deepcopy(MINI_EXP)
    spec["network"]["coupling"]["c"] = {
        "delayed": False,
        "incoming_states": ["theta"],
        "local_states": ["theta"],
        "pre_expression": {"rhs": "sin(x_j - x_i)"},
        "post_expression": {"rhs": "gx"},
    }
    line = _graph_line(spec)
    assert "SparseGraph(" in line and "DenseGraph(" not in line


def test_default_representation_is_dense():
    """No graph_representation (defaults to 'auto') keeps the pre-existing DenseGraph emit."""
    spec = copy.deepcopy(MINI_EXP)
    spec["network"].pop("graph_representation")
    line = _graph_line(spec)
    assert "DenseGraph(" in line and "SparseGraph(" not in line


# Explicit per-edge delays (no tract lengths) — the delayed network shape sparse supports.
MINI_EXP_DELAYED = copy.deepcopy(MINI_EXP)
MINI_EXP_DELAYED["network"]["edges"] = [
    {"source": 0, "target": 1, "parameters": {"weight": {"value": 0.5}, "delay": {"value": 2.0}}},
    {"source": 1, "target": 0, "parameters": {"weight": {"value": 0.4}, "delay": {"value": 3.0}}},
]
MINI_EXP_DELAYED["network"]["coupling"]["c"]["delayed"] = True


# A DIRECT (non-factored) local-in-pre coupling: sin(x_j - x_i) reads the target-local
# x_i inside pre(), the exact form the extended grid model uses (E_j*sin(theta_j-theta_i)).
# Its per-edge message is 1-D (nnz) on a SparseGraph, so pre() must add the leading
# n_output axis rank-agnostically. Distinct initial phases are required: with all nodes
# at theta=0, sin(x_j - x_i)=0 makes the coupling inert and the run vacuous.
MINI_EXP_DIRECT_PRE = copy.deepcopy(MINI_EXP)
MINI_EXP_DIRECT_PRE["network"]["edges"] = [
    {"source": 0, "target": 1, "parameters": {"weight": {"value": 0.5}}},
    {"source": 1, "target": 0, "parameters": {"weight": {"value": 0.4}}},
    {"source": 2, "target": 3, "parameters": {"weight": {"value": 0.3}}},
    {"source": 3, "target": 2, "parameters": {"weight": {"value": 0.6}}},
]
MINI_EXP_DIRECT_PRE["network"]["coupling"]["c"] = {
    "delayed": False,
    "incoming_states": ["theta"],
    "local_states": ["theta"],
    "pre_expression": {"rhs": "sin(x_j - x_i)"},
    "post_expression": {"rhs": "gx"},
}
MINI_EXP_DIRECT_PRE["dynamics"]["state_variables"]["theta"] = {
    "equation": {"rhs": "K * c"},
    "coupling_variable": True,
    "distribution": {"name": "Uniform", "domain": {"lo": 0.0, "hi": 6.283185}},
}
MINI_EXP_DIRECT_PRE["execution"] = {"random_seed": 0}


def test_sparse_delayed_network_emits_sparsedelaygraph():
    """Explicit per-edge delays + sparse -> SparseDelayGraph.

    Weights and delays share one BCOO sparsity pattern and the delayed gather runs per
    edge, so a delayed network no longer has to fall back to dense.
    """
    line = _graph_line(MINI_EXP_DELAYED)
    assert "SparseDelayGraph(" in line and "DenseDelayGraph(" not in line


def test_delayed_network_without_sparse_stays_dense():
    """The default (auto) representation keeps the dense delayed graph."""
    spec = copy.deepcopy(MINI_EXP_DELAYED)
    spec["network"].pop("graph_representation")
    line = _graph_line(spec)
    assert "DenseDelayGraph(" in line and "SparseDelayGraph(" not in line


def test_sparse_delayed_run_matches_dense():
    """The emitted sparse delayed code must RUN and agree with the dense emit.

    Codegen picking the right class proves nothing on its own — this drives the
    generated module end to end so a sparse/dense divergence in tvbo's own emission
    (edge ordering, delay zero-fill, max_delay_bound) cannot pass silently.
    """
    import numpy as np

    sparse_spec = copy.deepcopy(MINI_EXP_DELAYED)
    dense_spec = copy.deepcopy(MINI_EXP_DELAYED)
    dense_spec["network"].pop("graph_representation")

    sparse_res = SimulationExperiment(**sparse_spec).run(format="tvboptim")
    dense_res = SimulationExperiment(**dense_spec).run(format="tvboptim")

    sparse_ys = np.asarray(sparse_res.integration.data)
    dense_ys = np.asarray(dense_res.integration.data)
    assert sparse_ys.shape == dense_ys.shape
    np.testing.assert_allclose(sparse_ys, dense_ys, rtol=1e-6, atol=1e-8)


def test_sparse_instantaneous_local_pre_run_matches_dense():
    """A single-term local-in-pre coupling (sin(x_j - x_i)) must RUN on a sparse graph.

    pre() reads the target-local x_i, so the emitted per-edge message is 1-D (nnz) on a
    SparseGraph; the leading n_output axis has to be added rank-agnostically. A fixed 2-D
    reshape (``coupling_term[:, :, None]``) indexes a 1-D array with three axes and
    crashes — the failure the extended grid model hit. The delayed variant of this branch
    was the only one executed; this pins the INSTANTANEOUS one and checks sparse agrees
    with the dense emit, so a revert to the fixed-rank reshape can't pass silently.
    """
    import numpy as np

    sparse_spec = copy.deepcopy(MINI_EXP_DIRECT_PRE)
    dense_spec = copy.deepcopy(MINI_EXP_DIRECT_PRE)
    dense_spec["network"].pop("graph_representation")

    sparse_res = SimulationExperiment(**sparse_spec).run(format="tvboptim")
    dense_res = SimulationExperiment(**dense_spec).run(format="tvboptim")

    sparse_ys = np.asarray(sparse_res.integration.data)
    dense_ys = np.asarray(dense_res.integration.data)
    assert sparse_ys.shape == dense_ys.shape
    # A degenerate all-equal run would pass a dense/sparse check vacuously; require the
    # coupling to have actually done something (distinct initial phases evolved).
    assert float(np.ptp(sparse_ys)) > 1e-6, "coupling inert — test is vacuous"
    np.testing.assert_allclose(sparse_ys, dense_ys, rtol=1e-6, atol=1e-8)


def test_sparse_with_tract_lengths_is_rejected():
    """Tract lengths need DenseLengthGraph's live `speed` leaf; there is no sparse counterpart.

    Rejected at codegen rather than silently downgraded, because a silent downgrade
    would quietly discard the memory characteristic the recipe explicitly asked for.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "JR_MEG_FrequencyGradient_Optimization.yaml"))
    exp.network.graph_representation = "sparse"
    with pytest.raises(ValueError, match="tract lengths"):
        exp.render_code("tvboptim")


def test_tract_length_network_is_dense_length_graph_by_default():
    """Without the sparse opt-in, a tract-length network emits DenseLengthGraph.

    DenseLengthGraph subclasses DenseDelayGraph; the distinction matters because only
    the length graph carries the live `speed` leaf a conduction_speed axis sweeps.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "JR_MEG_FrequencyGradient_Optimization.yaml"))
    code = exp.render_code("tvboptim")
    assert "DenseLengthGraph(" in code and "SparseGraph(" not in code
