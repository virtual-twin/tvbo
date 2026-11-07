"""One directed edge ``a -> b``, sent through every array format, both `edge_matrix_files` declarations, every importer, the companion round trip and every backend's coupling.

Every matrix a Network holds is receiver-row: ``W[1, 0]`` is the weight of the edge ``0 -> 1``, the convention TVB, tvboptim and each tvbo backend reduce with (`tvbo.data.array_formats.CANONICAL_EDGE_DIMS`). A reader that knows its file is written the other way round converts by name; nothing in tvbo transposes a connectivity, which the last test greps for.
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tvbo import Network, SimulationExperiment
from tvbo.data.array_formats import CANONICAL_EDGE_DIMS, read_array
from tvbo.data.network_io import load_network, save_network

from .array_files import write_arrays

A_TO_B = np.array([[0.0, 0.0], [1.0, 0.0]])
"""Receiver-row: node ``b`` (row 1) receives from node ``a`` (column 0)."""
ROOT = Path(__file__).resolve().parents[1]


def _is_a_to_b(matrix) -> bool:
    return np.array_equal(np.asarray(matrix), A_TO_B)


@pytest.mark.parametrize("kind", ["h5", "zarr", "npy", "npz", "mat", "csv", "tsv", "txt"])
def test_every_format_reads_the_edge_where_it_was_written(tmp_path, kind):
    path, key = write_arrays(tmp_path, kind, weight=A_TO_B)
    assert _is_a_to_b(read_array(path, key, dims=CANONICAL_EDGE_DIMS).values)
    flipped, key = write_arrays(tmp_path / "flipped", kind, weight=A_TO_B.T)
    assert _is_a_to_b(read_array(flipped, key, dims=("source", "target")).values), "a sender-row file is transposed by name"


def test_an_edge_matrix_file_declares_its_orientation(tmp_path):
    np.savetxt(tmp_path / "receiver.csv", A_TO_B, delimiter=",")
    np.savetxt(tmp_path / "sender.csv", A_TO_B.T, delimiter=",")
    receiver = Network(edge_matrix_files=[{"path": str(tmp_path / "receiver.csv"), "dims": ["target", "source"]}])
    sender = Network(edge_matrix_files=[{"path": str(tmp_path / "sender.csv"), "dims": ["source", "target"]}])
    assert _is_a_to_b(receiver.matrix("weight", format="dense"))
    assert _is_a_to_b(sender.matrix("weight", format="dense"))
    with pytest.raises(ValueError, match="receiver.csv.*no dims are declared"):
        Network(edge_matrix_files=[str(tmp_path / "receiver.csv")])


def _tvb_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("weights.txt", "0 0\n1 0\n")
        zf.writestr("tract_lengths.txt", "0 0\n10 0\n")
        zf.writestr("centres.txt", "a 0 0 0\nb 1 0 0\n")
    return path


def _tvb_h5(path):
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("weights", data=A_TO_B)
        f.create_dataset("tract_lengths", data=10 * A_TO_B)
        f.create_dataset("centres", data=np.array([[0.0, 0, 0], [1.0, 0, 0]]))
        f.create_dataset("region_labels", data=np.array([b"a", b"b"]))
        f.create_dataset("cortical", data=np.array([1, 1]))
    return path


def test_the_tvb_importers_keep_tvbs_receiver_row(tmp_path):
    """TVB's ``weights[i, j]`` is the connection from ``j`` into ``i``, the orientation its coupling takes on the left of the dot product, so a TVB file is read as it is."""
    from tvbo import export
    from tvbo.adapters.tvb import from_tvb_h5, from_tvb_zip

    for net in (from_tvb_zip(_tvb_zip(tmp_path / "conn.zip")), from_tvb_h5(_tvb_h5(tmp_path / "conn.h5"))):
        assert _is_a_to_b(net.matrix("weight", format="dense"))
        assert np.array_equal(net.matrix("length", format="dense"), 10 * A_TO_B)
        assert [n.label for n in net.nodes] == ["a", "b"]
    assert _is_a_to_b(export.load("tvb-zip", tmp_path / "conn.zip").matrix("weight", format="dense"))
    assert _is_a_to_b(export.resolve_by_extension(".zip").importer(tmp_path / "conn.zip").matrix("weight", format="dense"))
    assert export.load("tvb-h5", tmp_path / "conn.h5").nodes[0].parameters["cortical"].value == 1.0
    with pytest.raises(ValueError, match="import-only"):
        export.render(
            SimulationExperiment(dynamics={"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}), "tvb-zip"
        )


def _bids(tmp_path, matrix):
    where = tmp_path / "bids"
    where.mkdir(parents=True)
    np.savetxt(where / "atlas-Probe_meas-streamlineCount_relmat.dense.tsv", matrix, delimiter="\t")
    (where / "atlas-Probe_nodeindices.tsv").write_text("index\tlabel\n0\ta\n1\tb\n")
    return where


def test_a_bids_relmat_states_no_orientation_so_it_must_be_symmetric(tmp_path):
    from tvbo import export

    with pytest.raises(ValueError, match="streamlineCount_relmat.dense.tsv: no dims are declared .* not symmetric"):
        Network.from_bids(_bids(tmp_path, A_TO_B), primary_weight="streamlineCount")
    both = Network.from_bids(_bids(tmp_path / "sym", A_TO_B + A_TO_B.T), primary_weight="streamlineCount")
    assert np.array_equal(both.matrix("weight", format="dense"), A_TO_B + A_TO_B.T)
    assert [n.label for n in both.nodes] == ["a", "b"]
    assert export.load("bids-bep017", tmp_path / "sym" / "bids", primary_weight="streamlineCount").number_of_nodes == 2
    loaded = Network(primary_weight="streamlineCount").load_from_bids(
        tmp_path / "sym" / "bids", observational_measures=["streamlineCount"]
    )
    assert loaded.observational_measures == ["streamlineCount"] and loaded.parcellation.label == "Probe"


@pytest.mark.parametrize("binary_format", ["h5", "zarr"])
def test_the_companion_round_trip_keeps_the_edge(tmp_path, binary_format):
    if binary_format == "zarr":
        pytest.importorskip("zarr")
    net = Network.from_matrix(A_TO_B, labels=["a", "b"])
    save_network(net, tmp_path / "net.yaml", binary_format=binary_format)
    assert _is_a_to_b(load_network(tmp_path / "net.yaml").matrix("weight", format="dense"))


G2D = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}
LINEAR = {
    "name": "Linear",
    "iri": "tvbo:Linear",
    "delayed": False,
    "parameters": {"a": {"name": "a", "value": 0.5}, "b": {"name": "b", "value": 0.0}},
}
EDGE = [{"source": 0, "target": 1, "directed": True, "parameters": {"weight": {"name": "weight", "value": 1.0}}}]


def _experiment(edges):
    network = {
        "number_of_nodes": 2,
        "nodes": [{"id": 0, "label": "a"}, {"id": 1, "label": "b"}],
        "edges": edges,
        "coupling": {"Linear": LINEAR},
    }
    return SimulationExperiment(
        dynamics=G2D, network=network, integration={"method": "Heun", "step_size": 0.05, "duration": 2.0}
    )


def _per_node(result):
    data = next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )
    return data.sel(variable="V").values.reshape(data.sizes["time"], -1)


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
        pytest.param("jax", marks=pytest.mark.backend_jax),
        pytest.param("tvb", marks=pytest.mark.backend_tvb),
    ],
)
def test_each_backend_drives_the_target_not_the_source(backend):
    """With the one edge ``a -> b`` and no coupling offset, ``a`` runs as it would unconnected and ``b`` does not."""
    coupled = _per_node(_experiment(EDGE).run(format=backend))
    alone = _per_node(_experiment([]).run(format=backend))
    assert np.allclose(coupled[:, 0], alone[:, 0]), f"{backend}: node a has no afferent and must run as if unconnected"
    assert not np.allclose(coupled[:, 1], alone[:, 1]), (
        f"{backend}: node b receives a and must differ from its unconnected run"
    )


def test_nothing_in_tvbo_transposes_a_connectivity():
    """A reader or importer converts by name (`declare`, `sender_row`); ``.T`` on a connectivity is a flip waiting to happen."""
    transposes = re.compile(r"\b(weights?|W|lengths?|connectivity|adjacency|sc|SC)\.T\b")
    sources = [
        path
        for path in [*(ROOT / "tvbo").rglob("*.py"), *(ROOT / "tvbo").rglob("*.mako")]
        if "datamodel" not in path.parts and path.name != "array_formats.py" and not path.name.endswith(".mako.py")
    ]
    hits = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in sources
        if transposes.search(text := path.read_text())
        for n, line in enumerate(text.splitlines(), 1)
        if transposes.search(line)
    ]
    assert hits == []
