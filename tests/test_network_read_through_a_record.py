"""A network read through a record, a curated one named by its ``iri`` or any one named by ``data_file``, holds every slot the record states, and a save in every format keeps them.

The network states nothing of its own beside the reference, so its record is the curated sidecar or the saved one. The slow cases read the smallest curated surface network, 64,984 vertices.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from tvbo import Network
from tvbo.classes.network import NETWORK_DIR

VOLUME = "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat"
SURFACE = "tpl-fsLR_cohort-HCPYA_rec-dTOR_atlas-Lobar_den-32k_desc-surf_relmat"
W = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])


def _stated(record: dict) -> dict:
    """The slots a record states beside its connectivity and its reference."""
    return {
        key: value
        for key, value in record.items()
        if key not in ("tvbo_class", "schema_version", "data_file", "nodes", "edges")
    }


def _assert_keeps_the_curated_record(name, folder, binary_format):
    """The curated network *name*, named by its iri and saved into *folder* in *binary_format*, states each slot its curated sidecar states, as it states it, and reads back with the curated node mapping."""
    curated = yaml.safe_load((NETWORK_DIR / f"{name}.yaml").read_text())
    net = Network(iri=f"tvbo:networks/{name}")
    net.save(folder / "n.yaml", binary_format=binary_format)
    assert _stated(curated).items() <= _stated(yaml.safe_load((folder / "n.yaml").read_text())).items()
    back = Network.from_file(str(folder / "n.yaml"))
    for key in _stated(curated):
        assert getattr(back, key) == getattr(net, key), key
    if "node_mapping" in curated:
        np.testing.assert_array_equal(
            back.node_mapping_data, Network.from_file(str(NETWORK_DIR / f"{name}.yaml")).node_mapping_data
        )


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
def test_a_curated_network_named_by_its_iri_keeps_its_record_through_a_save(tmp_path, binary_format):
    """The curated Desikan-Killiany connectome keeps its ``label``, ``bids``, ``parcellation``, ``tractogram`` and ``time_unit``."""
    _assert_keeps_the_curated_record(VOLUME, tmp_path, binary_format)


@pytest.mark.slow
@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
def test_a_curated_surface_network_named_by_its_iri_keeps_its_record_through_a_save(tmp_path, binary_format):
    """The curated Lobar surface keeps its ``parent_network``, its ``node_mapping`` and the vertex-to-region map it names."""
    _assert_keeps_the_curated_record(SURFACE, tmp_path, binary_format)


def test_a_network_named_by_data_file_holds_what_its_record_states(tmp_path):
    """The label, a conduction speed other than the default, a further parameter and an applied node selection of a saved record are the reading network's; a slot the reading network states is its own."""
    net = Network(
        label="mine",
        nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")],
        edges=[{"label": "weight"}],
        parameters={"conduction_speed": {"value": 5.0, "unit": "mm_per_ms"}, "G": {"value": 0.3}},
    )
    net.set_matrix("weight", W)
    net.save(tmp_path / "n.yaml")
    Network(data_file=str(tmp_path / "n.yaml"), node_selection={"drop": ["C"]}).save(tmp_path / "picked.yaml")
    read = Network(data_file=str(tmp_path / "picked.yaml"))
    assert (read.label, read.parameters["conduction_speed"].value, read.parameters["G"].value) == ("mine", 5.0, 0.3)
    assert (read.node_labels, list(read.node_selection.drop), read.node_selection.applied) == (["A", "B"], ["C"], True)
    np.testing.assert_array_equal(read.matrix("weight", format="dense"), W[:2, :2])
    own = Network(data_file=str(tmp_path / "n.yaml"), label="theirs", parameters={"conduction_speed": {"value": 7.0}})
    assert (own.label, own.parameters["conduction_speed"].value, own.parameters["G"].value) == ("theirs", 7.0, 0.3)
