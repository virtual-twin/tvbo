"""``Network.labels`` is each node's label with its id, read from the network's own record: it is the same once every atlas file is gone, and for a network whose record numbers its nodes by its atlas's lookup labels it is what the atlas's terminology gives. A graph drawn from the network's matrix labels each node by its own label."""

from __future__ import annotations

import pytest

from tvbo import Network
from tvbo.classes.atlas import Atlas

NUMBERED_BY_LOOKUP_LABEL = {
    "DesikanKilliany": "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat",
    "Destrieux": "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Destrieux_desc-SC_relmat",
    "Yeo17": "tpl-MNI152NLin6Asym_cohort-HCPYA_rec-dTOR_atlas-Yeo17_desc-SC_relmat",
}
SCHAEFER = "tpl-MNI152NLin6Asym_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-17Networks_scale-100_desc-SC_relmat"


@pytest.mark.parametrize("atlas", sorted(NUMBERED_BY_LOOKUP_LABEL))
def test_a_network_numbered_by_its_atlas_s_lookup_labels_gives_them(atlas):
    net = Network(iri=f"tvbo:networks/{NUMBERED_BY_LOOKUP_LABEL[atlas]}")
    entities = net.get_atlas().terminology.entities.values()
    assert net.labels == {entity.name: entity.lookupLabel for entity in entities}


@pytest.mark.parametrize("atlas", sorted(NUMBERED_BY_LOOKUP_LABEL))
def test_hidden_atlases_hides_every_atlas_and_curated_network(request, atlas):
    """Under the ``hidden_atlases`` fixture the atlas has no metadata file to read and its curated network is not found, so the tests that use the fixture run with neither."""
    request.getfixturevalue("hidden_atlases")
    assert Atlas.of(atlas).metadata_file is None
    with pytest.raises(FileNotFoundError):
        Network(iri=f"tvbo:networks/{NUMBERED_BY_LOOKUP_LABEL[atlas]}")


@pytest.mark.parametrize("binary_format", ["h5", "csv"])
@pytest.mark.parametrize("name", [*sorted(NUMBERED_BY_LOOKUP_LABEL.values()), SCHAEFER])
def test_labels_are_the_same_with_every_atlas_file_gone(tmp_path, request, name, binary_format):
    """A curated network saved and read back with no atlas file and no curated network to be found gives the labels it gave with them: its nodes' labels and ids, one for each node."""
    net = Network(iri=f"tvbo:networks/{name}")
    want = net.labels
    assert want == {node.label: node.id for node in net.nodes} and len(want) == net.number_of_nodes
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    request.getfixturevalue("hidden_atlases")
    assert Network.from_file(str(tmp_path / "n.yaml")).labels == want


def test_a_graph_drawn_from_the_matrix_labels_each_node_by_its_label(request):
    """The Desikan-Killiany network declares its layers and no edge between two nodes, so its graph is drawn from the weight matrix, each node labelled by its own label, the atlas present or gone."""
    net = Network(iri=f"tvbo:networks/{NUMBERED_BY_LOOKUP_LABEL['DesikanKilliany']}")
    drawn = [data["label"] for _, data in sorted(net.create_graph().nodes(data=True))]
    request.getfixturevalue("hidden_atlases")
    assert drawn == [data["label"] for _, data in sorted(net.create_graph().nodes(data=True))] == net.node_labels
