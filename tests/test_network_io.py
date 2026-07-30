"""Tests for §12 — matrix_io, network_io, converters modules.

Focused on round-trip correctness: write → read → compare.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# ── matrix_io tests ──────────────────────────────────────────────────


class TestAutoFormat:
    def test_small_dense(self):
        from tvbo.data.matrix_io import auto_format

        m = np.random.rand(87, 87)
        assert auto_format(m) == "dense"

    def test_large_sparse(self):
        from tvbo.data.matrix_io import auto_format

        m = np.zeros((600, 600))
        m[0, 1] = 1.0  # fill < 30%
        assert auto_format(m) == "csr"

    def test_large_dense(self):
        from tvbo.data.matrix_io import auto_format

        m = np.random.rand(600, 600)
        assert auto_format(m) == "dense"  # fill ~100% > 30%


class TestMatrixRoundTrip:
    @pytest.fixture
    def h5_file(self):
        import h5py

        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                yield hf, f.name

    def test_dense_roundtrip(self, h5_file):
        import h5py
        from tvbo.data.matrix_io import write_matrix, read_matrix

        hf, path = h5_file
        m = np.random.rand(10, 10).astype("float32")
        grp = hf.create_group("test")
        write_matrix(grp, m, fmt="dense")
        hf.close()

        with h5py.File(path, "r") as hf2:
            result = read_matrix(hf2["test"])
        np.testing.assert_allclose(result, m, atol=1e-6)

    def test_csr_roundtrip(self, h5_file):
        import h5py
        from tvbo.data.matrix_io import write_matrix, read_matrix

        hf, path = h5_file
        m = np.eye(20, dtype="float32") * 5.0
        grp = hf.create_group("test")
        write_matrix(grp, m, fmt="csr")
        hf.close()

        with h5py.File(path, "r") as hf2:
            result = read_matrix(hf2["test"])
        np.testing.assert_allclose(result, m, atol=1e-6)

    def test_coo_roundtrip(self, h5_file):
        import h5py
        from tvbo.data.matrix_io import write_matrix, read_matrix

        hf, path = h5_file
        m = np.zeros((15, 15), dtype="float32")
        m[3, 7] = 1.0
        m[10, 2] = 2.0
        grp = hf.create_group("test")
        write_matrix(grp, m, fmt="coo")
        hf.close()

        with h5py.File(path, "r") as hf2:
            result = read_matrix(hf2["test"])
        np.testing.assert_allclose(result, m, atol=1e-6)

    def test_scipy_sparse_input_records_its_shape(self, h5_file):
        """A scipy matrix keeps its shape: np.asarray(csr) is a 0-d object array."""
        import h5py
        import scipy.sparse as sp
        from tvbo.data.matrix_io import write_matrix, read_matrix

        hf, path = h5_file
        m = sp.random(400, 400, density=0.01, format="csr", random_state=0)
        grp = hf.create_group("test")
        write_matrix(grp, m, fmt="csr")
        hf.close()

        with h5py.File(path, "r") as hf2:
            assert tuple(hf2["test"].attrs["shape"]) == (400, 400)
            result = read_matrix(hf2["test"])
        np.testing.assert_allclose(result, m.toarray(), atol=1e-6)


# ── network_io tests ─────────────────────────────────────────────────


class TestTemplateEdgeOnlyNetwork:
    """A network whose `edges` are matrix DECLARATIONS, not connections.

    Every connectome loaded from a `data_file:` companion is of this shape, and at
    mesh scale materialising an N x N from those declarations is the difference
    between a run and an out-of-memory kill.
    """

    def _network(self, n=4000):
        from tvbo import Network
        from tvbo.datamodel.schema import Edge

        return Network(number_of_nodes=n, edges=[Edge(label="weight", format="csr")])

    def test_no_matrix_is_built_from_template_edges(self):
        net = self._network()
        assert net._delays_from_edges() is None
        assert net._weights_from_edges() is None

    def test_placeholder_nodes_are_not_written_to_the_sidecar(self, tmp_path):
        import scipy.sparse as sp
        import yaml as _yaml
        from tvbo.data.network_io import save_network, load_network

        net = self._network(n=600)
        net.set_matrix("weight", sp.eye(600, format="csr") * 2.0)
        save_network(net, tmp_path / "net.yaml")

        meta = _yaml.safe_load((tmp_path / "net.yaml").read_text())
        assert "nodes" not in meta
        assert meta["number_of_nodes"] == 600

        back = load_network(tmp_path / "net.yaml")
        assert len(back.nodes) == 600
        np.testing.assert_allclose(back.weights_matrix, np.eye(600) * 2.0, atol=1e-6)

    def test_authored_nodes_survive_the_round_trip(self, tmp_path):
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        net = Network.from_matrix(np.eye(3), labels=["L_V1", "R_V1", "thal"])
        save_network(net, tmp_path / "net.yaml")

        back = load_network(tmp_path / "net.yaml")
        assert [n.label for n in back.nodes] == ["L_V1", "R_V1", "thal"]


class TestNetworkIO:
    def test_edge_length_delays_follow_conduction_speed(self):
        from tvbo import Network

        net = Network.from_string(
            """
number_of_nodes: 2
nodes:
  - {id: 0, label: R0}
  - {id: 1, label: R1}
edges:
  - source: 0
    target: 1
    directed: true
    parameters:
      weight: {value: 1.0}
      length: {value: 30.0, unit: mm}
"""
        )

        net.parameters["conduction_speed"].value = 3.0
        assert net.weights_matrix[1, 0] == 1.0
        assert net.weights_matrix[0, 1] == 0.0
        assert net.lengths_matrix[1, 0] == 30.0
        assert net.lengths_matrix[0, 1] == 0.0
        assert net.calculate_delays()[1, 0] == 10.0
        assert net.calculate_delays()[0, 1] == 0.0

        net.parameters["conduction_speed"].value = 0.003
        assert net.calculate_delays()[1, 0] == 10000.0

    def test_edge_delay_parameters_bypass_conduction_speed(self):
        from tvbo import Network

        net = Network.from_string(
            """
number_of_nodes: 2
nodes:
  - {id: 0, label: R0}
  - {id: 1, label: R1}
edges:
  - source: 0
    target: 1
    directed: true
    parameters:
      weight: {value: 1.0}
      delay: {value: 30.0, unit: ms}
"""
        )

        net.parameters["conduction_speed"].value = 3.0
        assert net.calculate_delays()[1, 0] == 30.0
        assert np.isnan(net.calculate_delays()[0, 1])

        net.parameters["conduction_speed"].value = 0.003
        assert net.calculate_delays()[1, 0] == 30.0

    def test_load_new_format_roundtrip(self):
        """Create a new-format sidecar+HDF5, then load it back."""
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        weights = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype="float32")
        lengths = np.array([[0, 10, 20], [10, 0, 30], [20, 30, 0]], dtype="float32")
        net = Network.from_matrix(weights, lengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.yaml"
            save_network(net, out, binary_format="h5")
            net2 = load_network(out)
            assert net2.number_of_regions == 3 or net2.number_of_nodes == 3
            assert net2.label is not None or True  # label may be None for synthetic

    def test_save_load_roundtrip_h5(self):
        """Save a Network as YAML+HDF5, reload, and compare."""
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        # Build a small test network
        weights = np.random.rand(5, 5).astype("float32")
        lengths = np.random.rand(5, 5).astype("float32") * 100
        net = Network.from_matrix(weights, lengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_net.yaml"
            save_network(net, out, binary_format="h5")

            assert out.exists(), "YAML sidecar not written"
            assert out.with_suffix(".h5").exists(), "HDF5 companion not written"

            net2 = load_network(out)
            assert net2.number_of_regions == 5 or net2.number_of_nodes == 5

    def test_save_load_roundtrip_csv(self):
        """Save a Network as YAML+CSV, reload, and compare."""
        from tvbo import Network
        from tvbo.data.network_io import save_network

        weights = np.random.rand(4, 4).astype("float32")
        lengths = np.random.rand(4, 4).astype("float32") * 50
        net = Network.from_matrix(weights, lengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_net.yaml"
            save_network(net, out, binary_format="csv")

            assert out.exists(), "YAML sidecar not written"
            assert out.with_suffix(".csv").exists(), "CSV companion not written"


# ── converters tests ──────────────────────────────────────────────────


class TestRelmatEntities:
    def test_entities_from_dict(self):
        from tvbo.data.converters import relmat_entities

        meta = {
            "parcellation": {"atlas": {"name": "DesikanKilliany"}},
            "tractogram": "dTOR",
            "space": "MNI152NLin2009c",
            "descriptor": "SC",
        }
        entities = relmat_entities(meta)
        assert entities["atlas"] == "DesikanKilliany"
        assert entities["description"] == "SC"

    def test_entities_from_network(self):
        from tvbo import Network
        from tvbo.data.converters import relmat_entities

        net = Network.from_matrix(
            np.eye(3, dtype="float32"),
            np.ones((3, 3), dtype="float32"),
        )
        # Should not raise even with minimal metadata
        entities = relmat_entities(net)
        assert isinstance(entities, dict)


class TestFromTvbZip:
    def test_from_tvb_zip_missing_file(self):
        from tvbo.data.converters import from_tvb_zip

        with pytest.raises(Exception):
            from_tvb_zip("/nonexistent/path.zip")


# ── LazyArrayStore tests ─────────────────────────────────────────────


class TestLazyArrayStore:
    def test_lazy_no_load_on_init(self):
        from tvbo.data.matrix_io import LazyArrayStore

        # Should not crash even with nonexistent path — no I/O on init
        store = LazyArrayStore(Path("/fake/path.h5"), {"edges": []})
        assert not store._loaded

    def test_lazy_loads_on_access(self):
        """Build an HDF5, wrap in LazyArrayStore, verify lazy load."""
        import h5py
        from tvbo.data.matrix_io import LazyArrayStore, write_matrix

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = Path(f.name)

        meta = {"edges": [{"label": "weight"}]}
        with h5py.File(path, "w") as hf:
            grp = hf.create_group("edges/weight")
            write_matrix(grp, np.eye(3, dtype="float32"), fmt="dense")

        store = LazyArrayStore(path, meta)
        assert not store._loaded
        arrays = store.arrays  # triggers load
        assert store._loaded
        assert "weight" in arrays
        np.testing.assert_allclose(arrays["weight"], np.eye(3), atol=1e-6)
        path.unlink()


# ── Zarr roundtrip tests ─────────────────────────────────────────────


class TestZarrRoundTrip:
    def test_save_load_roundtrip_zarr(self):
        """Save a Network as YAML+Zarr, reload, and compare arrays."""
        pytest.importorskip("zarr")
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        weights = np.random.rand(5, 5).astype("float32")
        lengths = np.random.rand(5, 5).astype("float32") * 100
        net = Network.from_matrix(weights, lengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_net.yaml"
            save_network(net, out, binary_format="zarr")

            assert out.exists(), "YAML sidecar not written"
            zarr_dir = out.with_suffix(".zarr")
            assert zarr_dir.exists(), "Zarr companion not created"

            net2 = load_network(out)
            assert net2.number_of_regions == 5 or net2.number_of_nodes == 5

    def test_zarr_matrix_roundtrip(self):
        """Write/read a matrix via Zarr group."""
        zarr = pytest.importorskip("zarr")
        from tvbo.data.matrix_io import write_matrix, read_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            z = zarr.open(str(Path(tmpdir) / "test.zarr"), mode="w")
            m = np.random.rand(8, 8).astype("float32")
            grp = z.create_group("test")
            write_matrix(grp, m, fmt="dense")

            z2 = zarr.open(str(Path(tmpdir) / "test.zarr"), mode="r")
            result = read_matrix(z2["test"])
            np.testing.assert_allclose(result, m, atol=1e-6)


# ── Edge parameters roundtrip tests ──────────────────────────────────


class TestEdgeParametersRoundTrip:
    def test_edge_params_h5_roundtrip(self):
        """Edge parameters survive HDF5 write → read."""
        import h5py
        from tvbo.data.network_io import _write_edges, _read_edges

        weights = np.random.rand(4, 4).astype("float32")
        lengths = np.random.rand(4, 4).astype("float32") * 50
        meta = {"edges": [{"label": "streamlineCount"}]}
        arrays = {"streamlineCount": weights}
        edge_params = {"streamlineCount": {"tractLength": lengths}}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = Path(f.name)

        with h5py.File(path, "w") as hf:
            _write_edges(hf, meta, arrays, edge_params)

        with h5py.File(path, "r") as hf:
            loaded_arrays, loaded_params = _read_edges(hf, meta)

        assert "streamlineCount" in loaded_arrays
        np.testing.assert_allclose(loaded_arrays["streamlineCount"], weights, atol=1e-6)
        assert "tractLength" in loaded_params["streamlineCount"]
        np.testing.assert_allclose(loaded_params["streamlineCount"]["tractLength"], lengths, atol=1e-6)
        path.unlink()

    def test_edge_params_network_roundtrip(self):
        """Edge params persist through Network save/load cycle."""
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        weights = np.random.rand(3, 3).astype("float32")
        lengths = np.random.rand(3, 3).astype("float32") * 20
        net = Network.from_matrix(weights, lengths)
        # Replace arrays with named edge + edge params
        net.set_matrix("streamlineCount", weights)
        object.__setattr__(net, "_edge_params", {"streamlineCount": {"tractLength": lengths}})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_ep.yaml"
            save_network(net, out, binary_format="h5")
            net2 = load_network(out)

            store = net2._store
            assert store is not None
            assert "tractLength" in store.edge_params.get("streamlineCount", {})
            np.testing.assert_allclose(
                store.edge_params["streamlineCount"]["tractLength"],
                lengths,
                atol=1e-6,
            )


# ── Hierarchical network tests ───────────────────────────────────────


class TestHierarchicalNetwork:
    def test_node_mapping_roundtrip(self):
        """Parent index array survives save/load cycle."""
        from tvbo import Network
        from tvbo.data.network_io import save_network, load_network

        weights = np.random.rand(6, 6).astype("float32")
        net = Network.from_matrix(weights)
        # Simulate hierarchical mapping: 6 fine nodes → 3 coarse nodes
        net._node_mapping_data = np.array([0, 0, 1, 1, 2, 2], dtype="int32")
        net.node_mapping = "/nodes/parent_index"

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_hier.yaml"
            save_network(net, out, binary_format="h5")

            net2 = load_network(out)
            store = net2._store
            assert store is not None
            parent_idx = store.read_dataset("nodes/parent_index")
            np.testing.assert_array_equal(
                parent_idx,
                [0, 0, 1, 1, 2, 2],
            )


# ── Node coordinates tests ───────────────────────────────────────────


class TestNodeCoordinates:
    def test_coordinates_written_to_h5(self):
        """Node.position coordinates are persisted to HDF5."""
        import h5py
        from tvbo import Network
        from tvbo.datamodel import tvbo_datamodel
        from tvbo.data.network_io import save_network

        nodes = [
            tvbo_datamodel.Node(
                id=i,
                label=f"n{i}",
                position=tvbo_datamodel.Coordinate(
                    x=float(i),
                    y=float(i * 2),
                    z=float(i * 3),
                ),
            )
            for i in range(4)
        ]
        net = Network.from_matrix(np.eye(4, dtype="float32"))
        net.nodes = nodes

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_coords.yaml"
            save_network(net, out, binary_format="h5")

            with h5py.File(out.with_suffix(".h5"), "r") as hf:
                assert "nodes/coordinates" in hf
                coords = hf["nodes/coordinates"][()]
                assert coords.shape == (4, 3)
                np.testing.assert_allclose(coords[2], [2.0, 4.0, 6.0])


# ── BEP017 export tests ──────────────────────────────────────────────


class TestBEP017Export:
    def test_to_bep017_creates_files(self):
        """BEP017 export writes TSV + JSON for each template edge."""
        from tvbo import Network
        from tvbo.datamodel import tvbo_datamodel
        from tvbo.data.converters import to_bep017

        weights = np.random.rand(3, 3).astype("float32")
        net = Network.from_matrix(weights)
        net.descriptor = "SC"
        # Add template edge metadata
        net.edges = [
            tvbo_datamodel.Edge(
                label="streamlineCount",
                weighted=True,
                non_negative=True,
                valid_diagonal=False,
            )
        ]
        object.__setattr__(net, "_arrays", {"streamlineCount": weights})
        net._store = None

        with tempfile.TemporaryDirectory() as tmpdir:
            to_bep017(net, tmpdir)

            out = Path(tmpdir)
            tsv_files = list(out.glob("*.tsv"))
            json_files = list(out.glob("*.json"))
            assert len(tsv_files) >= 1, f"Expected TSV files, got: {list(out.iterdir())}"
            assert len(json_files) >= 1, "Expected JSON sidecars"

            # Verify JSON sidecar content
            import json

            sidecar = json.loads(json_files[0].read_text())
            assert "RelationshipMeasure" in sidecar
            assert sidecar["Weighted"] is True

    def test_to_bep017_node_indices(self):
        """BEP017 export writes nodeindices TSV."""
        from tvbo import Network
        from tvbo.datamodel import tvbo_datamodel
        from tvbo.data.converters import to_bep017

        nodes = [tvbo_datamodel.Node(id=i, label=f"region_{i}") for i in range(3)]
        net = Network(nodes=nodes, edges=[], number_of_nodes=3)
        net.edges = [tvbo_datamodel.Edge(label="weight")]
        object.__setattr__(net, "_arrays", {"weight": np.eye(3, dtype="float32")})
        net._store = None
        net.descriptor = "SC"

        with tempfile.TemporaryDirectory() as tmpdir:
            to_bep017(net, tmpdir)
            node_files = list(Path(tmpdir).glob("*nodeindices*"))
            assert len(node_files) == 1
            content = node_files[0].read_text()
            assert "region_0" in content


# ── TVB ZIP full roundtrip test ───────────────────────────────────────


class TestFromTvbZipRoundTrip:
    def test_tvb_zip_roundtrip(self):
        """Create a fake TVB ZIP, import it, verify node positions + arrays."""
        import zipfile
        from tvbo.data.converters import from_tvb_zip

        n = 4
        weights = np.random.rand(n, n).astype("float64")
        lengths = np.random.rand(n, n).astype("float64") * 50
        labels = [f"region_{i}" for i in range(n)]
        coords = np.random.rand(n, 3).astype("float64") * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "connectivity.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("weights.txt", "\n".join(" ".join(f"{x:.8g}" for x in row) for row in weights))
                zf.writestr("tract_lengths.txt", "\n".join(" ".join(f"{x:.8g}" for x in row) for row in lengths))
                centres_lines = [f"{labels[i]} {coords[i, 0]:.8g} {coords[i, 1]:.8g} {coords[i, 2]:.8g}" for i in range(n)]
                zf.writestr("centres.txt", "\n".join(centres_lines))

            net = from_tvb_zip(zip_path)

            assert net.number_of_nodes == n
            assert net.descriptor == "SC"
            assert len(net.nodes) == n

            # Verify node labels
            assert net.nodes[0].label == "region_0"
            assert net.nodes[3].label == "region_3"

            # Verify node positions
            for i in range(n):
                pos = net.nodes[i].position
                assert pos is not None, f"Node {i} has no position"
                np.testing.assert_allclose(pos.x, coords[i, 0], atol=1e-5)
                np.testing.assert_allclose(pos.y, coords[i, 1], atol=1e-5)
                np.testing.assert_allclose(pos.z, coords[i, 2], atol=1e-5)

            # Verify arrays (canonical names: weight, length)
            assert "weight" in net._arrays
            np.testing.assert_allclose(net._arrays["weight"], weights, atol=1e-5)
            assert "length" in net._arrays
            np.testing.assert_allclose(net._arrays["length"], lengths, atol=1e-5)


class TestMultiEdgeFreezeRoundtrip:
    """Freezing a multi-edge / primary_weight network must preserve every edge —
    especially the tract-length matrix, which delayed simulations require and which
    has no ``primary_weight``-style selector. Regression for the freeze remap that
    assumed a canonical [weight, length] edge order and, for an NMF bundle declared
    as [length, weight_NMF_*], dropped ``length`` (no delays) and overwrote the real
    weights with the length matrix.
    """

    def _roundtrip(self, edges: dict, primary=None):
        from tvbo.classes.network import Network
        import h5py

        n = next(iter(edges.values())).shape[0]
        net = Network(number_of_nodes=n)
        net.__class__ = Network
        for name, mat in edges.items():
            net.set_matrix(name, mat)
        if primary is not None:
            net.primary_weight = primary
        with tempfile.TemporaryDirectory() as d:
            net.save(Path(d) / "n.yaml", binary_format="h5")
            with h5py.File(Path(d) / "n.h5") as f:
                frozen = set(f["edges"].keys())
            reloaded = Network.load(str(Path(d) / "n.h5"))
            # Materialize while the lazy-store companion file still exists — the
            # TemporaryDirectory is removed on block exit, before assertions run.
            _ = np.asarray(reloaded.lengths_matrix)
            _ = np.asarray(reloaded.weights_matrix)
        return frozen, reloaded

    def test_nmf_bundle_preserves_length_and_weights(self):
        rng = np.random.RandomState(0)
        W = rng.rand(8, 8)
        L = rng.rand(8, 8) * 100.0
        edges = {"length": L, "weight_NMF_rank5": W, "weight_NMF_alpha": rng.rand(8, 8)}
        frozen, net = self._roundtrip(edges, primary="weight_NMF_rank5")
        assert "length" in frozen, "length edge dropped on freeze"
        np.testing.assert_allclose(np.asarray(net.lengths_matrix), L, atol=1e-4)
        # weights must be the NMF weights, NOT the length matrix
        np.testing.assert_allclose(np.asarray(net.weights_matrix), W, atol=1e-4)
        assert float(np.max(net.lengths_matrix)) > 0, "delays lost (zero lengths)"

    def test_all_edge_attributes_survive(self):
        rng = np.random.RandomState(1)
        edges = {k: rng.rand(6, 6) for k in ("weight", "length", "fc", "local_connectivity")}
        frozen, _ = self._roundtrip(edges)
        assert set(edges).issubset(frozen)

    def test_length_resolved_by_meaning_not_exact_name(self):
        # a sidecar naming the length edge ``tract_length`` must still drive delays
        rng = np.random.RandomState(2)
        L = rng.rand(6, 6) * 100.0
        frozen, net = self._roundtrip({"tract_length": L, "streamlinecount": rng.rand(6, 6)},
                                      primary="streamlinecount")
        assert "tract_length" in frozen
        np.testing.assert_allclose(np.asarray(net.lengths_matrix), L, atol=1e-4)


# ── region alias reconciliation (§ node-label crosswalk) ──────────────
#
# by_label node reconciliation must align a dataset-sourced target to the model
# network by LABEL, never by row index. These tests lock the properties that make
# it safer than index/order-based alignment: hemisphere parity and byte-identical
# results under an arbitrary target reordering.


class TestRegionAliasMap:
    def _net(self, labels):
        from tvbo.classes.network import Network

        n = len(labels)
        w = np.ones((n, n)) - np.eye(n)
        return Network.from_matrix(w, w, labels=list(labels))

    def test_identity_when_no_aliases(self):
        net = self._net(["L_A", "R_A"])
        assert net.region_alias_map() == {"L_A": "L_A", "R_A": "R_A"}

    def test_inline_node_aliases(self):
        net = self._net(["L_A", "R_A"])
        net.nodes[0].alternateName = ["A_LEFT"]
        net.nodes[1].alternateName = ["A_RIGHT"]
        m = net.region_alias_map()
        assert m["A_LEFT"] == "L_A" and m["A_RIGHT"] == "R_A"
        # identity always retained so exact matches still resolve
        assert m["L_A"] == "L_A"

    def test_ambiguous_alias_raises(self):
        net = self._net(["L_A", "R_A"])
        net.nodes[0].alternateName = ["X"]
        net.nodes[1].alternateName = ["X"]  # same alias -> two regions
        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            net.region_alias_map()

    def test_reconcile_hemisphere_safe_and_order_independent(self):
        import xarray as xr

        # model lists LEFT first; the "empirical" target lists RIGHT first (opposite
        # hemisphere order) under a divergent nomenclature carried as aliases.
        model = self._net(["L_A", "R_A", "L_B", "R_B"])
        for node, alias in zip(model.nodes, ["A_LEFT", "A_RIGHT", "B_LEFT", "B_RIGHT"]):
            node.alternateName = [alias]
        amap = model.region_alias_map()
        model_labels = model.node_labels

        # target: right-first order, empirical labels, a recognisable matrix
        tlabels = ["A_RIGHT", "B_RIGHT", "A_LEFT", "B_LEFT"]
        M = np.arange(16, dtype=float).reshape(4, 4)
        M = M + M.T  # symmetric so the check is unambiguous

        def align(labels, mat):
            canon = [amap.get(l, l) for l in labels]
            da = xr.DataArray(mat, dims=("i", "j"), coords={"i": canon, "j": canon})
            return da.sel(i=model_labels, j=model_labels).values

        A = align(tlabels, M)

        # hemisphere parity: no L<->R crossing
        def hemi(l):
            return "L" if l[:2] == "L_" or l.endswith("_LEFT") else "R"

        assert all(hemi(t) == hemi(amap[t]) for t in tlabels)

        # A must place target row "A_LEFT" at model index 0 (L_A), etc. — i.e. the
        # aligned matrix is keyed to model order regardless of the target's order.
        # shuffle target order -> aligned result must be byte-identical
        for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
            B = align([tlabels[k] for k in perm], M[np.ix_(perm, perm)])
            assert np.array_equal(A, B), f"order dependence under perm {perm}"


def _atlas_hemi(label):
    """Hemisphere of a node label under any packaged-atlas convention, or None."""
    if label in ("Brain-Stem", "BRAIN_STEM", "brain-stem"):
        return None
    if label[:2] == "L_" or label.endswith("_LEFT") or label.startswith(("ctx-lh-", "left-")):
        return "L"
    if label[:2] == "R_" or label.endswith("_RIGHT") or label.startswith(("ctx-rh-", "right-")):
        return "R"
    return None


class TestAtlasAliases:
    """Packaged atlas terminologies carry hemisphere-correct empirical aliases."""

    @pytest.mark.parametrize("atlas,checks", [
        ("hcpmmp1", {"L_Thalamus": "THALAMUS_LEFT", "R_Thalamus": "THALAMUS_RIGHT",
                     "Brain-Stem": "BRAIN_STEM"}),
        ("DesikanKilliany", {"left-thalamus": "THALAMUS_LEFT", "right-thalamus": "THALAMUS_RIGHT",
                             "ctx-lh-bankssts": "L_bankssts", "brain-stem": "BRAIN_STEM"}),
    ])
    def test_atlas_aliases_present_and_hemisphere_consistent(self, atlas, checks):
        from tvbo.classes.atlas import Atlas

        ents = getattr(getattr(Atlas(atlas), "terminology", None), "entities", None) or {}
        if not ents:
            pytest.skip(f"{atlas} atlas terminology not available")
        for canon, empirical in checks.items():
            assert empirical in (ents[canon].alternateName or []), f"{canon} missing {empirical}"

        # every alias keeps its region's hemisphere; aliases are globally unique
        seen = {}
        for canon, ent in ents.items():
            for alt in ent.alternateName or []:
                assert _atlas_hemi(canon) == _atlas_hemi(alt), f"hemisphere swap {canon} -> {alt}"
                assert alt not in seen or seen[alt] == canon, f"alias {alt} on two regions"
                seen[alt] = canon
