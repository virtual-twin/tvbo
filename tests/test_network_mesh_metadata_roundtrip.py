"""A surface network's mesh record, its element type and its vertex and element counts, survives a load and a save in every format.

A curated surface network carries the record only as the attributes of its companion's ``/mesh/`` group, with no ``mesh`` record in its sidecar; the synthetic networks here are saved that way, and the slow case reads the smallest curated one.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from tvbo import Network

MESH = {"element_type": "triangle", "number_of_vertices": 4, "number_of_elements": 2}
CURATED = "tpl-fsLR_cohort-HCPYA_rec-dTOR_atlas-Lobar_den-32k_desc-surf_relmat"


def _record(net) -> dict:
    """The network's mesh record as plain values, or ``None`` when it has none."""
    mesh = getattr(net, "mesh", None)
    if mesh is None:
        return None
    return {key: None if getattr(mesh, key) is None else type(value)(getattr(mesh, key)) for key, value in MESH.items()}


def _saved_as_curated(folder, binary_format):
    """A four-node surface network saved in *binary_format* with its mesh record in its companion's attributes alone, as each curated surface network is, and read back."""
    net = Network(number_of_nodes=4, mesh=MESH)
    net.set_array("mesh/vertices", np.arange(12, dtype="float32").reshape(4, 3) / 7)
    net.set_array("mesh/elements", np.array([[0, 1, 2], [1, 2, 3]], dtype="int32"))
    net.save(folder / "n.yaml", binary_format=binary_format)
    sidecar = yaml.safe_load((folder / "n.yaml").read_text())
    sidecar.pop("mesh")
    (folder / "n.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return Network.from_file(str(folder / "n.yaml"))


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
@pytest.mark.parametrize("source_format", ["h5", "zarr"])
def test_a_mesh_record_held_by_the_companion_survives_a_save(tmp_path, source_format, binary_format):
    """A mesh record the companion's ``/mesh/`` attributes hold, an HDF5 or a Zarr companion's, is the network's ``mesh`` once read, and a save in any format keeps it."""
    (tmp_path / "source").mkdir()
    (tmp_path / "saved").mkdir()
    net = _saved_as_curated(tmp_path / "source", source_format)
    assert _record(net) == MESH
    net.save(tmp_path / "saved" / "n.yaml", binary_format=binary_format)
    assert _record(Network.from_file(str(tmp_path / "saved" / "n.yaml"))) == MESH


@pytest.mark.slow
@pytest.mark.parametrize("binary_format", ["h5", "csv"])
def test_a_curated_surface_network_keeps_its_mesh_record_through_a_save(tmp_path, binary_format):
    """The curated Lobar surface network, named by its iri, reads its mesh record from its companion, a triangle mesh of 64,984 vertices and 129,960 elements, and a save keeps it."""
    want = {"element_type": "triangle", "number_of_vertices": 64984, "number_of_elements": 129960}
    net = Network(iri=f"tvbo:networks/{CURATED}")
    assert _record(net) == want
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    assert _record(Network.from_file(str(tmp_path / "n.yaml"))) == want
