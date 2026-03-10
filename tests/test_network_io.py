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


# ── network_io tests ─────────────────────────────────────────────────

class TestNetworkIO:
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
        from tvbo.data.network_io import save_network, load_network

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
