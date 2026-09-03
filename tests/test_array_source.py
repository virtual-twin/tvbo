"""The companion store reads one array at a time, and a metadata question reads none.

These pin the contract by counting value reads through the one seam every read in `tvbo.data.matrix_io` takes, `_read_values`, rather than by timing. A surface companion carries a mesh beside its connectivity, and a sensor companion carries a non-square leadfield; the reads a caller pays for are exactly the ones it names.
"""

import os

import numpy as np
import pytest
from scipy import sparse

from tvbo.data import matrix_io
from tvbo.data.matrix_io import LazyArrayStore, write_matrix

h5py = pytest.importorskip("h5py")


@pytest.fixture
def value_reads(monkeypatch):
    """Dataset paths whose values were read, in order."""
    reads: list[str] = []
    original = matrix_io._read_values

    def counting(dataset):
        reads.append(dataset.name)
        return original(dataset)

    monkeypatch.setattr(matrix_io, "_read_values", counting)
    return reads


@pytest.fixture
def companion(tmp_path):
    """A companion the way a surface-plus-sensors network would ship one: a sparse connectome, a dense leadfield, a mesh."""
    path = tmp_path / "net.h5"
    W = sparse.random(50, 50, density=0.05, format="csr", random_state=0)
    with h5py.File(path, "w") as f:
        write_matrix(f.create_group("edges/weight"), W, fmt="csr")
        write_matrix(f.create_group("edges/gain"), np.random.default_rng(0).random((4, 50), dtype="f4"), fmt="dense")
        f.create_dataset("mesh/vertices", data=np.zeros((50, 3), dtype="f4"))
        f.create_dataset("nodes/coordinates", data=np.zeros((50, 3), dtype="f4"))
    meta = {"edges": [{"label": "weight", "format": "csr"}, {"label": "gain", "format": "dense"}]}
    return path, meta, W


def test_construction_names_and_membership_read_no_values(companion, value_reads):
    path, meta, _ = companion
    store = LazyArrayStore(path, meta)
    assert store.names == ["weight", "gain"]
    assert "weight" in store
    assert "gain" in store
    assert "length" not in store
    assert value_reads == []


def test_info_reads_the_header_only(companion, value_reads):
    path, meta, W = companion
    store = LazyArrayStore(path, meta)
    w = store.info("weight")
    assert (w.path, w.shape, w.format) == ("edges/weight", (50, 50), "csr")
    assert w.nbytes == W.data.nbytes
    g = store.info("gain")
    assert (g.shape, g.format, g.dtype) == ((4, 50), "dense", np.dtype("f4"))
    m = store.info("mesh/vertices")
    assert (m.shape, m.format) == ((50, 3), "dataset")
    with pytest.raises(KeyError):
        store.info("length")
    assert value_reads == []


def test_reading_one_matrix_reads_that_matrix(companion, value_reads):
    path, meta, _ = companion
    store = LazyArrayStore(path, meta)
    store["gain"]
    assert value_reads == ["/edges/gain/data"]
    assert not store._loaded
    store["gain"]
    assert value_reads == ["/edges/gain/data"], "a second access is served from the cache"


def test_a_csr_companion_stays_csr(companion):
    path, meta, W = companion
    store = LazyArrayStore(path, meta)
    got = store["weight"]
    assert sparse.isspmatrix_csr(got)
    np.testing.assert_allclose(got.toarray(), W.toarray())


def test_names_are_discovered_from_the_file_when_the_sidecar_declares_none(companion, value_reads):
    path, _, _ = companion
    store = LazyArrayStore(path, {})
    assert sorted(store.names) == ["gain", "weight"]
    assert value_reads == []


def test_the_arrays_view_reads_per_key_and_dict_reads_all(companion, value_reads):
    path, meta, _ = companion
    store = LazyArrayStore(path, meta)
    view = store.arrays
    assert list(view) == ["weight", "gain"]
    assert len(view) == 2
    assert value_reads == []
    view["gain"]
    assert value_reads == ["/edges/gain/data"]
    everything = dict(store.arrays)
    assert set(everything) == {"weight", "gain"}
    assert "/edges/weight/data" in value_reads
    assert store._loaded


def test_a_held_handle_opens_the_file_once(companion, monkeypatch):
    path, meta, _ = companion
    opens: list[str] = []
    original = h5py.File.__init__

    def counting(self, name, *a, **k):
        if isinstance(name, (str, os.PathLike)):
            opens.append(str(name))
        original(self, name, *a, **k)

    monkeypatch.setattr(h5py.File, "__init__", counting)
    store = LazyArrayStore(path, meta)
    with store:
        store.info("weight")
        store["weight"]
        store["gain"]
        store.read_dataset("mesh/vertices")
    assert len(opens) == 1


def test_read_dataset_reads_that_dataset(companion, value_reads):
    path, meta, _ = companion
    store = LazyArrayStore(path, meta)
    got = store.read_dataset("mesh/vertices")
    assert got.shape == (50, 3)
    assert value_reads == ["/mesh/vertices"]
    with pytest.raises(KeyError):
        store.read_dataset("mesh/normals")


def test_edge_params_are_read_with_their_edge(tmp_path, value_reads):
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        write_matrix(f.create_group("edges/weight"), np.eye(3), fmt="dense")
        write_matrix(f.create_group("edges/weight/edge_parameters/length"), np.ones((3, 3)), fmt="dense")
    store = LazyArrayStore(path, {"edges": [{"label": "weight"}]})
    assert "weight" in store.edge_params
    assert value_reads == []
    assert set(store.edge_params["weight"]) == {"length"}
    assert value_reads == ["/edges/weight/data", "/edges/weight/edge_parameters/length/data"]


def test_a_missing_edge_is_a_key_error(companion):
    path, meta, _ = companion
    store = LazyArrayStore(path, meta)
    with pytest.raises(KeyError):
        store["length"]


class TestNetworkOverTheStore:
    @pytest.fixture
    def net(self, companion, tmp_path):
        from tvbo.data.network_io import load_network

        path, _, _ = companion
        sidecar = tmp_path / "net.yaml"
        sidecar.write_text(
            "tvbo_class: tvbo:Network\n"
            "label: surface-plus-sensors\n"
            "number_of_nodes: 50\n"
            f"data_file: {path.name}\n"
            "edges:\n"
            "  - label: weight\n"
            "    format: csr\n"
            "  - label: gain\n"
            "    format: dense\n"
        )
        return load_network(sidecar)

    def test_loading_a_network_reads_no_values(self, value_reads, net):
        assert value_reads == []

    def test_names_and_info_read_no_values(self, value_reads, net):
        assert net.matrix_names == ["weight", "gain"]
        assert net.matrix_info("gain").shape == (4, 50)
        assert net.matrix_info("mesh/vertices").shape == (50, 3)
        assert value_reads == []

    def test_matrix_returns_the_stored_format_and_dense_on_request(self, net, companion):
        _, _, W = companion
        got = net.matrix("weight")
        assert sparse.issparse(got)
        dense = net.matrix("weight", format="dense")
        assert isinstance(dense, np.ndarray)
        np.testing.assert_allclose(dense, W.toarray())

    def test_asking_for_weight_reads_only_weight(self, value_reads, net):
        net.matrix("weight")
        assert all(p.startswith("/edges/weight/") for p in value_reads)
        assert "/edges/gain/data" not in value_reads

    def test_a_non_square_matrix_is_served_by_name(self, net):
        assert net.matrix("gain").shape == (4, 50)


def test_a_sparse_network_still_counts_as_having_matrices(companion, tmp_path):
    """`np.asarray(csr).size` is 1, so a size test reported every sparse network as empty."""
    from tvbo.cli.workflow import _network_has_matrices
    from tvbo.data.network_io import load_network

    path, _, _ = companion
    sidecar = tmp_path / "s.yaml"
    sidecar.write_text(
        f"tvbo_class: tvbo:Network\nnumber_of_nodes: 50\ndata_file: {path.name}\nedges:\n  - label: weight\n    format: csr\n"
    )
    assert _network_has_matrices(load_network(sidecar))


class TestResidency:
    """`arrays` is what is resident, `materialize` is how it gets there, and `repr` says which is which."""

    @pytest.fixture
    def net(self, companion, tmp_path):
        from tvbo.data.network_io import load_network

        path, _, _ = companion
        sidecar = tmp_path / "net.yaml"
        sidecar.write_text(
            f"tvbo_class: tvbo:Network\nnumber_of_nodes: 50\ndata_file: {path.name}\nedges:\n  - label: weight\n    format: csr\n  - label: gain\n    format: dense\n"
        )
        return load_network(sidecar)

    def test_loading_leaves_nothing_resident_and_repr_lists_the_rest(self, net, value_reads):
        assert net.arrays == {}
        assert "lazy: edges/gain, edges/weight, mesh/vertices, nodes/coordinates" in repr(net)
        assert value_reads == []

    def test_materialize_reads_exactly_what_it_is_asked_for(self, net, value_reads):
        sim = net.materialize("weight", "mesh/vertices")
        assert set(sim.arrays) == {"edges/weight", "mesh/vertices"}
        assert set(value_reads) == {"/edges/weight/data", "/edges/weight/indices", "/edges/weight/indptr", "/mesh/vertices"}
        assert "resident: edges/weight, mesh/vertices" in repr(sim)
        assert "lazy: edges/gain, nodes/coordinates" in repr(sim)

    def test_materialize_returns_a_copy_with_its_own_residency(self, net):
        sim = net.materialize("gain")
        assert "edges/gain" in sim.arrays
        assert net.arrays == {}, "the original was not touched"
        assert sim._store is net._store
        sim2 = sim.materialize("weight")
        assert sim2.arrays["edges/gain"] is sim.arrays["edges/gain"], "already-resident arrays are shared, not copied"

    def test_materialize_refuses_a_path_nothing_holds(self, net):
        with pytest.raises(KeyError):
            net.materialize("length")

    def test_an_implicit_read_lands_in_the_same_place(self, net):
        net.matrix("gain")
        assert set(net.arrays) == {"edges/gain"}

    def test_the_mesh_is_not_read_by_loading_and_is_served_by_path(self, net, value_reads):
        assert value_reads == []
        assert net.array("mesh/vertices").shape == (50, 3)
        assert value_reads == ["/mesh/vertices"]
        assert net.array("mesh/normals") is None

    def test_set_array_is_resident_and_wins_over_the_companion(self, net, value_reads):
        user = np.ones((4, 50))
        net.set_array("gain", user)
        assert net.array("gain") is user or np.array_equal(net.array("gain"), user)
        assert value_reads == []

    def test_observations_are_edge_matrices(self, net):
        net.observational_measures = ["gain"]
        assert net.observations["gain"].shape == (4, 50)


def test_a_surface_network_round_trips_its_mesh_without_the_network_reading_it(tmp_path, value_reads):
    """A re-save copies the mesh through from the source companion; nothing about it was ever resident on the network."""
    from tvbo.classes.network import Network
    from tvbo.data.network_io import load_network, save_network

    net = Network.from_matrix(np.eye(4))
    net.set_array("mesh/vertices", np.zeros((9, 3), dtype="f4"))
    net.set_array("mesh/elements", np.zeros((8, 3), dtype="i4"))
    save_network(net, tmp_path / "surf.yaml", binary_format="h5")
    back = load_network(tmp_path / "surf.yaml")
    reads_before = list(value_reads)
    assert back.arrays == {}
    save_network(back, tmp_path / "again.yaml", binary_format="h5")
    twice = load_network(tmp_path / "again.yaml")
    assert twice.array("mesh/vertices").shape == (9, 3)
    assert twice.array("mesh/elements").shape == (8, 3)
    assert "/mesh/vertices" in value_reads[len(reads_before) :], "the re-save read the mesh to copy it through"


def test_edge_parameters_are_arrays_under_their_edge(tmp_path):
    from tvbo.classes.network import Network
    from tvbo.data.network_io import load_network, save_network

    net = Network.from_matrix(np.eye(3))
    net.set_matrix("streamlineCount", np.eye(3))
    net.set_array("edges/streamlineCount/edge_parameters/tractLength", np.full((3, 3), 7.0))
    assert net.edge_parameter_arrays() == {
        "streamlineCount": {"tractLength": net.arrays["edges/streamlineCount/edge_parameters/tractLength"]}
    }
    save_network(net, tmp_path / "ep.yaml", binary_format="h5")
    back = load_network(tmp_path / "ep.yaml")
    np.testing.assert_array_equal(back.edge_parameter_arrays()["streamlineCount"]["tractLength"], np.full((3, 3), 7.0))


class TestPytree:
    """`arrays` is the pytree: the key set is the treedef, every resident array is a leaf, the spec is static."""

    @pytest.fixture
    def net(self):
        from tvbo.classes.network import Network

        W = np.arange(9, dtype=float).reshape(3, 3)
        net = Network.from_matrix(W)
        net.set_array("gain", np.ones((2, 3)))
        net.set_array("mesh/vertices", np.zeros((5, 3)))
        net.set_array("mesh/elements", np.zeros((4, 3), dtype="int32"))
        return net

    def test_reordering_keys_does_not_retrace_but_changing_them_does(self, net):
        import jax
        import jax.numpy as jnp

        from tvbo.classes.network import Network

        traces = []

        @jax.jit
        def total(n):
            traces.append(1)
            return jnp.sum(n.arrays["edges/weight"])

        reordered = Network.from_matrix(np.zeros((3, 3)))
        for key in reversed(list(net.arrays)):
            reordered.set_array(key, net.arrays[key])
        total(net)
        total(reordered)
        assert len(traces) == 1, "the same key set in another order is the same treedef"
        total(net.materialize())
        assert len(traces) == 1
        smaller = Network.from_matrix(np.zeros((3, 3)))
        total(smaller)
        assert len(traces) == 2, "a different key set is a different treedef"

    def test_grad_returns_a_cotangent_for_every_float_leaf(self, net):
        import jax
        import jax.numpy as jnp

        def loss(n):
            a = n.arrays
            return jnp.sum(a["edges/weight"] ** 2) + 3.0 * jnp.sum(a["edges/gain"]) + jnp.sum(a["mesh/vertices"])

        g = jax.grad(loss, allow_int=True)(net)
        assert type(g) is type(net)
        assert set(g.arrays) == {"edges/weight", "edges/gain", "mesh/vertices", "mesh/elements"}
        np.testing.assert_array_equal(g.arrays["edges/weight"], 2 * net.arrays["edges/weight"])
        np.testing.assert_array_equal(g.arrays["edges/gain"], 3.0)
        np.testing.assert_array_equal(g.arrays["mesh/vertices"], 1.0)
        assert g.arrays["mesh/elements"].dtype == jax.dtypes.float0, "integer topology is a leaf with no cotangent"

    def test_the_spec_is_static_inside_a_trace(self, net):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def zeros_per_node(n):
            return jnp.zeros(n.number_of_nodes)

        assert zeros_per_node(net).shape == (3,)

    def test_a_traced_matrix_is_served_raw(self, net):
        import jax

        got = jax.jit(lambda n: n.matrix("weight"))(net)
        np.testing.assert_array_equal(got, net.arrays["edges/weight"])

    def test_flattening_reads_nothing_and_warns_when_nothing_is_materialised(
        self, companion, tmp_path, value_reads, monkeypatch
    ):
        import jax

        from tvbo.classes.network import LazyMaterializationWarning
        from tvbo.data.network_io import load_network

        path, _, _ = companion
        sidecar = tmp_path / "lazy.yaml"
        sidecar.write_text(
            f"tvbo_class: tvbo:Network\nnumber_of_nodes: 50\ndata_file: {path.name}\nedges:\n  - label: gain\n    format: dense\n"
        )
        lazy = load_network(sidecar)
        with pytest.warns(LazyMaterializationWarning, match="materialize"):
            jax.tree_util.tree_flatten(lazy)
        assert value_reads == []
        monkeypatch.setenv("TVBO_JAX_STRICT", "1")
        with pytest.raises(RuntimeError, match="materialize"):
            jax.tree_util.tree_flatten(lazy)
        leaves, _ = jax.tree_util.tree_flatten(lazy.materialize("gain"))
        assert len(leaves) == 1 and value_reads == ["/edges/gain/data"]

    def test_a_sparse_resident_array_is_refused_by_name(self):
        import jax

        from tvbo.classes.network import Network

        net = Network.from_matrix(np.eye(3))
        net.set_array("weight", sparse.eye(3, format="csr"))
        with pytest.raises(TypeError, match="edges/weight"):
            jax.tree_util.tree_flatten(net)
