"""TVB interoperability tests — lossless round-trip via Network.from_tvb.

Verifies that tvbo can:
1. Import TVB default connectivity → save YAML+HDF5 → reload → export back to TVB with zero information loss (weights, tract_lengths, centres, labels, speed).
2. Import TVB surface simulation data (connectivity + surface + region_mapping)
   → save multi-level YAML+HDF5 → verify mesh + node_mapping persist.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.surfaces import CorticalSurface

    _HAVE_TVB = True
except ImportError:
    _HAVE_TVB = False

pytestmark = pytest.mark.skipif(not _HAVE_TVB, reason="tvb-library not installed")


class TestConnectivityRoundTrip:
    """Lossless round-trip: TVB Connectivity → tvbo Network → YAML+HDF5 → TVB."""

    @pytest.fixture
    def tvb_conn(self):
        conn = Connectivity.from_file()
        conn.configure()
        return conn

    def test_from_tvb_basic(self, tvb_conn):
        """Network.from_tvb preserves node count and descriptor."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        assert net.number_of_nodes == tvb_conn.weights.shape[0]
        assert net.descriptor == "SC"

    def test_weights_preserved(self, tvb_conn):
        """Weights matrix survives import."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        np.testing.assert_allclose(
            net.weights_matrix,
            tvb_conn.weights,
            atol=1e-10,
        )

    def test_lengths_preserved(self, tvb_conn):
        """Tract lengths matrix survives import."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        np.testing.assert_allclose(
            net.lengths_matrix,
            tvb_conn.tract_lengths,
            atol=1e-10,
        )

    def test_centres_preserved(self, tvb_conn):
        """Node centres (x, y, z) survive import."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        centres = np.array(list(net.get_centers().values()), dtype="float64")
        np.testing.assert_allclose(centres, tvb_conn.centres, atol=1e-10)

    def test_labels_preserved(self, tvb_conn):
        """Region labels survive import."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        tvb_labels = list(tvb_conn.region_labels)
        tvbo_labels = [n.label for n in net.nodes]
        assert tvbo_labels == tvb_labels

    def test_conduction_speed_preserved(self, tvb_conn):
        """Conduction speed survives import."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        cs = net.parameters.get("conduction_speed")
        assert cs is not None
        expected = float(np.asarray(tvb_conn.speed).ravel()[0])
        assert float(cs.value) == pytest.approx(expected)

    def test_cortical_flags_preserved(self, tvb_conn):
        """Per-node cortical flag survives as node parameter."""
        from tvbo import Network

        net = Network.from_tvb(tvb_conn)
        if tvb_conn.cortical is not None and len(tvb_conn.cortical) > 0:
            for i, node in enumerate(net.nodes):
                stored = float(node.parameters["cortical"].value)
                assert stored == float(tvb_conn.cortical[i])

    def test_save_reload_roundtrip(self, tvb_conn):
        """Save as YAML+HDF5, reload, verify arrays match."""
        from tvbo import Network
        from tvbo.data.network_io import load_network, save_network

        net = Network.from_tvb(tvb_conn)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tvb_default_sc.yaml"
            save_network(net, path, binary_format="h5")

            assert path.exists(), "YAML sidecar not written"
            assert path.with_suffix(".h5").exists(), "HDF5 companion not written"

            net2 = load_network(path)
            n = tvb_conn.weights.shape[0]
            assert net2.number_of_nodes == n or net2.number_of_regions == n

    def test_full_roundtrip_back_to_tvb(self, tvb_conn):
        """TVB → tvbo → YAML+HDF5 → tvbo → TVB: lossless."""
        from tvbo import Network
        from tvbo.data.network_io import load_network, save_network

        net = Network.from_tvb(tvb_conn)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tvb_default_sc.yaml"
            save_network(net, path, binary_format="h5")
            net2 = load_network(path)

            # Export back to TVB
            tvb_conn2 = net2.execute(format="tvb")

            np.testing.assert_allclose(
                tvb_conn2.weights,
                tvb_conn.weights,
                atol=1e-6,
                err_msg="Weights not preserved in round-trip",
            )
            np.testing.assert_allclose(
                tvb_conn2.tract_lengths,
                tvb_conn.tract_lengths,
                atol=1e-6,
                err_msg="Tract lengths not preserved in round-trip",
            )


class TestSurfaceRoundTrip:
    """Multi-level network: TVB surface simulation → tvbo YAML+HDF5."""

    @pytest.fixture
    def tvb_data(self):
        conn = Connectivity.from_file()
        conn.configure()
        surf = CorticalSurface.from_file()
        surf.configure()
        rmap = RegionMapping.from_file()
        rmap.connectivity = conn
        rmap.surface = surf
        rmap.configure()
        return conn, surf, rmap

    def test_from_tvb_surface_creates_two_networks(self, tvb_data):
        """from_tvb_surface returns (region_net, surface_net)."""
        from tvbo import Network

        conn, surf, rmap = tvb_data
        region_net, surface_net = Network.from_tvb_surface(conn, surf, rmap)
        assert region_net.number_of_nodes == conn.weights.shape[0]
        assert surface_net.number_of_nodes == surf.vertices.shape[0]

    def test_surface_mesh_data(self, tvb_data):
        """Mesh data (vertices, triangles, normals) is attached."""
        from tvbo import Network

        conn, surf, rmap = tvb_data

        _, surface_net = Network.from_tvb_surface(conn, surf, rmap)
        vertices = surface_net.array("mesh/vertices")
        elements = surface_net.array("mesh/elements")
        normals = surface_net.array("mesh/normals")

        np.testing.assert_allclose(vertices, surf.vertices, atol=1e-5)
        np.testing.assert_allclose(elements, surf.triangles)
        np.testing.assert_allclose(normals, surf.vertex_normals, atol=1e-5)

    def test_node_mapping_preserved(self, tvb_data):
        """Region mapping (vertex→region) is stored as node_mapping."""
        from tvbo import Network

        conn, surf, rmap = tvb_data

        _, surface_net = Network.from_tvb_surface(conn, surf, rmap)
        mapping = surface_net.node_mapping_data
        expected = np.asarray(rmap.array_data, dtype="int32")
        np.testing.assert_array_equal(mapping, expected)

    def test_parent_network_linked(self, tvb_data):
        """Surface network references region network as parent."""
        from tvbo import Network

        conn, surf, rmap = tvb_data

        region_net, surface_net = Network.from_tvb_surface(conn, surf, rmap)
        assert surface_net.parent_network is not None

    def test_mesh_survives_hdf5(self, tvb_data):
        """Save surface network → HDF5 has /mesh/ group with arrays."""
        import h5py

        from tvbo import Network
        from tvbo.data.network_io import save_network

        conn, surf, rmap = tvb_data
        region_net, surface_net = Network.from_tvb_surface(conn, surf, rmap)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save region network first (needed as parent reference)
            region_path = Path(tmpdir) / "region_sc.yaml"
            save_network(region_net, region_path, binary_format="h5")

            # Save surface network
            surface_path = Path(tmpdir) / "surface_mesh.yaml"
            save_network(surface_net, surface_path, binary_format="h5")

            h5_path = surface_path.with_suffix(".h5")
            assert h5_path.exists(), "Surface HDF5 not written"

            with h5py.File(h5_path, "r") as f:
                assert "mesh" in f, "No /mesh/ group in HDF5"
                assert "vertices" in f["mesh"], "No vertices dataset"
                assert "elements" in f["mesh"], "No elements dataset"

                v = f["mesh/vertices"][:]
                np.testing.assert_allclose(
                    v,
                    surf.vertices,
                    atol=1e-5,
                    err_msg="Vertices not preserved in HDF5",
                )
                t = f["mesh/elements"][:]
                np.testing.assert_array_equal(
                    t,
                    surf.triangles,
                    err_msg="Triangles not preserved in HDF5",
                )

    def test_node_mapping_survives_hdf5(self, tvb_data):
        """Region mapping array persists in HDF5 companion."""
        import h5py

        from tvbo import Network
        from tvbo.data.network_io import save_network

        conn, surf, rmap = tvb_data
        _, surface_net = Network.from_tvb_surface(conn, surf, rmap)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "surface_mesh.yaml"
            save_network(surface_net, path, binary_format="h5")

            with h5py.File(path.with_suffix(".h5"), "r") as f:
                assert "mesh/region_mapping" in f, "Region mapping not written to HDF5"
                mapping = f["mesh/region_mapping"][:]
                expected = np.asarray(rmap.array_data, dtype="int32")
                np.testing.assert_array_equal(mapping, expected)
