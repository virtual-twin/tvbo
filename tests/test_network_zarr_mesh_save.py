"""A Zarr save writes a network's mesh as an HDF5 save does, and one that fails while writing it leaves nothing behind.

Every network is synthetic: four vertices and two triangles, held in memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo import Network
from tvbo.data import network_io

VERTICES = np.arange(12, dtype="float32").reshape(4, 3) / 7
ELEMENTS = np.array([[0, 1, 2], [1, 2, 3]], dtype="int32")
NORMALS = np.ones((4, 3), dtype="float32")
MESH = {"element_type": "triangle", "number_of_vertices": 4, "number_of_elements": 2}


def _surface(**kwargs):
    """A four-node network holding a mesh's vertices, elements and normals in memory."""
    net = Network(number_of_nodes=4, **kwargs)
    for name, value in (("vertices", VERTICES), ("elements", ELEMENTS), ("normals", NORMALS)):
        net.set_array(f"mesh/{name}", value)
    return net


@pytest.mark.parametrize("record", [False, True], ids=["arrays", "mesh-record"])
@pytest.mark.parametrize("binary_format", ["h5", "zarr"])
def test_a_mesh_saves_and_reads_back_in_every_format(tmp_path, binary_format, record):
    """A network holding a mesh, with and without the ``mesh`` record that names its element type and counts, saves as Zarr as it saves as HDF5, and its vertices, elements and normals read back equal."""
    _surface(**({"mesh": MESH} if record else {})).save(tmp_path / "n.yaml", binary_format=binary_format)
    store = Network.from_file(str(tmp_path / "n.yaml"))._store
    for name, value in (("vertices", VERTICES), ("elements", ELEMENTS), ("normals", NORMALS)):
        np.testing.assert_array_equal(store.read_dataset(f"mesh/{name}"), value)


def test_a_zarr_save_that_fails_writing_the_mesh_leaves_nothing(tmp_path, monkeypatch):
    """A first Zarr save into an empty folder that fails while writing the mesh leaves the folder empty: no ``n.zarr`` that no sidecar names."""

    def fail(*_, **__):
        raise RuntimeError("the mesh could not be written")

    monkeypatch.setattr(network_io, "_write_mesh", fail)
    with pytest.raises(RuntimeError, match="the mesh could not be written"):
        _surface().save(tmp_path / "n.yaml", binary_format="zarr")
    assert not any(tmp_path.iterdir())
