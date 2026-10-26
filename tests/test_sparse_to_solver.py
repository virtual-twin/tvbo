"""A sparse connectome reaches the tvboptim solver as a BCOO, without a dense N x N ever being allocated.

`Network.matrix` serves the stored format; `adapters.tvboptim.solver_weights` is the one place a scipy matrix becomes a JAX one on this path and hands the emitted code a BCOO exactly when the emitted graph is sparse. ``auto`` follows the store: a csr companion is sparse unless tract lengths or weight transforms force the dense graph, and asking for sparse against either is refused by name. A run under a forbidden ``toarray`` is what "without densifying" means here.
"""

import copy

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("tvboptim")
from tvbo import SimulationExperiment  # noqa: E402
from tvbo.classes.network import Network  # noqa: E402

N = 6
RING = sparse.csr_matrix(np.roll(np.eye(N), 1, axis=1) + np.roll(np.eye(N), -1, axis=1))
"""An undirected ring, stored csr: 12 of 36 entries."""
RING_DENSE = RING.toarray()

SPEC = {
    "id": 1,
    "label": "sparse-to-solver fixture",
    "dynamics": {
        "name": "MiniKuramoto",
        "system_type": "continuous",
        "output": ["theta"],
        "parameters": {"K": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "theta": {
                "equation": {"rhs": "K * c"},
                "initial_value": 0.0,
                "coupling_variable": True,
                "distribution": {"name": "Uniform", "domain": {"lo": 0.0, "hi": 6.283185}},
            },
        },
    },
    "network": {
        "number_of_nodes": N,
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
    "execution": {"random_seed": 0},
}


def _experiment(representation=None, weights=RING, transform=None, **network):
    spec = copy.deepcopy(SPEC)
    if representation is not None:
        spec["network"]["graph_representation"] = representation
    spec["network"].update(network)
    exp = SimulationExperiment(**spec)
    exp.network.set_matrix("weight", weights)
    if transform:
        exp.network.add_transform("weight", transform)
    return exp


def _graph_line(exp):
    code = exp.render_code("tvboptim")
    return next((ln.strip() for ln in code.splitlines() if "graph = " in ln and "Graph(" in ln), "")


@pytest.fixture
def no_densify(monkeypatch):
    """Every scipy ``toarray`` raises, so a run that completes never held the dense matrix."""

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__}.toarray() densified the connectome")

    monkeypatch.setattr(sparse._base._spbase, "toarray", refuse)
    monkeypatch.setattr(sparse._compressed._cs_matrix, "toarray", refuse)


def test_stored_format_reads_the_header():
    assert Network.from_matrix(RING).stored_format("weight") == "csr"
    assert Network.from_matrix(RING).stored_format("weights") == "csr", "any spelling"
    assert Network.from_matrix(RING_DENSE).stored_format("weight") == "dense"
    assert Network(number_of_nodes=3).stored_format("weight") is None


def test_a_csr_store_reaches_the_solver_as_bcoo(no_densify):
    from jax.experimental.sparse import BCOO

    from tvbo.adapters.tvboptim import solver_weights

    weights = solver_weights(_experiment("sparse").network)
    assert isinstance(weights, BCOO) and weights.nse == RING.nnz
    np.testing.assert_array_equal(np.asarray(weights.todense()), RING_DENSE)


def test_the_sparse_run_matches_dense(no_densify):
    exp = _experiment("sparse")
    assert "SparseGraph(" in _graph_line(exp)
    sparse_ys = np.asarray(exp.run(format="tvboptim").integration.data)
    dense_ys = np.asarray(_experiment("dense", weights=RING_DENSE).run(format="tvboptim").integration.data)
    assert float(np.ptp(dense_ys)) > 1e-6, "coupling inert — test is vacuous"
    np.testing.assert_allclose(sparse_ys, dense_ys, rtol=1e-6, atol=1e-8)


def test_auto_follows_the_store():
    assert "SparseGraph(" in _graph_line(_experiment())
    assert "DenseGraph(" in _graph_line(_experiment(weights=RING_DENSE))
    assert "DenseGraph(" in _graph_line(_experiment("dense"))


def test_auto_never_overrides_lengths_or_transforms():
    from tvbo.adapters.tvboptim import solver_weights

    delayed = _experiment(coupling={"c": {**SPEC["network"]["coupling"]["c"], "delayed": True}})
    delayed.network.set_matrix("length", np.full((N, N), 10.0))
    assert "DenseLengthGraph(" in _graph_line(delayed)
    assert not hasattr(solver_weights(delayed.network), "nse"), "a dense graph takes a dense array"
    assert "DenseGraph(" in _graph_line(_experiment(transform="weight / 2"))


def test_sparse_against_lengths_or_transforms_is_refused_by_name():
    delayed = _experiment("sparse", coupling={"c": {**SPEC["network"]["coupling"]["c"], "delayed": True}})
    delayed.network.set_matrix("length", np.full((N, N), 10.0))
    with pytest.raises(ValueError, match="tract lengths"):
        delayed.render_code("tvboptim")
    with pytest.raises(ValueError, match="weight transforms"):
        _experiment("sparse", transform="weight / 2").render_code("tvboptim")


def test_the_standalone_graph_is_sparse_too(no_densify):
    from tvboptim.experimental.network_dynamics.graph import SparseGraph

    from tvbo.adapters.tvboptim import to_tvboptim

    net = Network.from_matrix(RING, graph_representation="sparse")
    assert isinstance(to_tvboptim(net, delays=False, return_type="graph"), SparseGraph)
