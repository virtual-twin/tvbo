"""A layer that runs node by its ``dimension_labels``, stored sparse or dense, saves and reads back in every format with its edge parameter, and a save that fails while writing it leaves the folder as it was.

HDF5 labels a dense layer's axes with dimension scales; a sparse layer's ``data`` holds its stored entries and has no axis to label, and a Zarr array has no dimension scales.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy import sparse

from tvbo import Network
from tvbo.data import network_io

FORMATS = ["h5", "zarr", "csv"]
ROUTE = np.array([[0.0, 1.5], [0.0, 0.0], [2.5, 0.0]])


def _network(layout):
    """A network on A, B, C with a weight layer and a 3 x 2 layer ``route`` over the columns s1 and s2, stored as *layout*, with an edge parameter."""
    net = Network(
        nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")],
        edges=[{"label": "weight"}, {"label": "route", "dimension_labels": ["s1", "s2"]}],
    )
    held = sparse.csr_matrix(ROUTE) if layout == "csr" else ROUTE
    net.set_matrix("weight", np.ones((3, 3)))
    net.set_matrix("route", held)
    next(edge for edge in net.edges if edge.label == "route").format = layout  # set_matrix picks the format by size
    net.set_array("edges/route/edge_parameters/gain", 2 * held)
    return net


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("layout", ["csr", "dense"])
@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_layer_running_node_by_its_dimension_labels_saves_and_reads_back(tmp_path, binary_format, layout):
    _network(layout).save(tmp_path / "n.yaml", binary_format=binary_format)
    back = Network.from_file(str(tmp_path / "n.yaml"))
    route = next(edge for edge in back.edges if edge.label == "route")
    assert (list(route.dimension_labels), back.stored_format("route")) == (["s1", "s2"], layout)
    np.testing.assert_array_equal(_dense(back.matrix("route")), ROUTE)
    np.testing.assert_array_equal(_dense(back.edge_parameter_arrays()["route"]["gain"]), 2 * ROUTE)
    if binary_format == "h5" and layout == "dense":
        with h5py.File(tmp_path / "n.h5") as stored:
            assert [label.decode() for label in stored["edges/route/data"].dims[1][0][()]] == ["s1", "s2"]


@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_save_that_fails_while_writing_a_sparse_layer_leaves_the_folder_as_it_was(tmp_path, monkeypatch, binary_format):
    """A save that fails once the sparse layer is written, at its edge parameter, leaves the files of the save before it as they were and nothing beside them."""
    _network("csr").save(tmp_path / "n.yaml", binary_format=binary_format)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    write_matrix = network_io.write_matrix

    def failing(grp, matrix, **kwargs):
        if "edge_parameters" in grp.name:
            raise RuntimeError("the edge parameter could not be written")
        return write_matrix(grp, matrix, **kwargs)

    monkeypatch.setattr(network_io, "write_matrix", failing)
    with pytest.raises(RuntimeError, match="the edge parameter could not be written"):
        _network("csr").save(tmp_path / "n.yaml", binary_format=binary_format)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()} == before
