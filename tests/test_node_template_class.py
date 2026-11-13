"""A network's ``node_template`` is a ``NodeTemplate``: the slots every node shares, and none that says which node it is. ``Node`` has the same slots and adds the per-node slots ``id``, ``label``, ``position`` and ``region``, without being a ``NodeTemplate``, so it keeps its field order and no validator takes a node for a template.

A template that sets a per-node slot is refused, naming it, however the network is built: from a mapping, from a file, from a built node, or as either generated form's record. A curated template carries no ``id``, and the ``id: -1`` an older tvbo wrote on every template it saved, and only that value, is dropped when a template is read, so the networks it saved still read.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from tvbo import Network
from tvbo.classes.network import NETWORK_DIR
from tvbo.datamodel import pydantic as pm
from tvbo.datamodel import schema as dm

SCHAEFER = "tpl-FSLMNI152_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-17Networks_scale-100_desc-SC_relmat"
SAVED_BEFORE = Path(__file__).parent / "reference_data" / "saved_networks" / "network_saved_with_a_node_template_id.yaml"
PER_NODE = {"id": 0, "label": "x", "position": {"x": 0.0, "y": 0.0, "z": 0.0}, "region": "r"}
NODE_FIELDS = [
    "label",
    "alternateName",
    "description",
    "parameters",
    "record",
    "id",
    "size",
    "dynamics",
    "position",
    "region",
    "hemisphere",
    "homologue",
    "state",
    "events",
    "subnetwork",
]


def test_a_node_has_the_template_slots_and_the_per_node_slots_without_being_a_template():
    shared = [field.name for field in dataclasses.fields(dm.NodeTemplate)]
    assert shared == list(pm.NodeTemplate.model_fields) == [name for name in NODE_FIELDS if name not in PER_NODE]
    assert not issubclass(dm.Node, dm.NodeTemplate) and not issubclass(pm.Node, pm.NodeTemplate)
    assert list(pm.Node.model_fields) == NODE_FIELDS
    required_first = ["id", *(name for name in NODE_FIELDS if name != "id")]  # a dataclass puts its required field first
    assert [field.name for field in dataclasses.fields(dm.Node)] == required_first


@pytest.mark.parametrize("slot", sorted(PER_NODE))
@pytest.mark.parametrize(
    "build",
    [
        lambda template: Network(number_of_nodes=2, node_template=template),
        lambda template: dm.Network(node_template=template),
        lambda template: pm.Network(node_template=template),
    ],
    ids=["tvbo", "dataclass", "pydantic"],
)
def test_a_template_that_sets_a_per_node_slot_is_refused_naming_it(build, slot):
    with pytest.raises(ValueError, match=rf"`node_template` sets `{slot}`; the network's data gives each node its own"):
        build({slot: PER_NODE[slot], "parameters": {"g": {"value": 0.5}}})


def test_a_template_read_from_a_file_that_sets_a_per_node_slot_is_refused(tmp_path):
    (tmp_path / "n.yaml").write_text(yaml.safe_dump({"number_of_nodes": 2, "node_template": {"id": 0, "size": 3}}))
    with pytest.raises(ValueError, match=r"`node_template` sets `id`"):
        Network.from_file(str(tmp_path / "n.yaml"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Network(number_of_nodes=2, node_template=dm.Node(id=-1, parameters={"g": {"value": 0.5}})),
        lambda: Network(node_template=dm.Node(id=-1, parameters={"g": {"value": 0.5}})),
        lambda: dm.Network(node_template=dm.Node(id=-1)),
        lambda: pm.Network(node_template=pm.Node(id=-1)),
    ],
    ids=["tvbo", "tvbo-no-nodes", "dataclass", "pydantic"],
)
def test_a_node_given_as_the_template_is_refused(build):
    with pytest.raises(ValueError, match=r"`node_template` sets `id`"):
        build()


def test_a_template_reaches_every_node_without_an_id():
    net = Network(number_of_nodes=2, node_template={"hemisphere": "left", "parameters": {"g": {"value": 0.5}}})
    assert [(str(node.hemisphere), sorted(node.parameters)) for node in net.nodes] == [("left", ["g"]), ("left", ["g"])]
    assert type(net.node_template) is dm.NodeTemplate


@pytest.mark.parametrize("binary_format", ["h5", "csv"])
def test_a_curated_template_carries_no_id_and_a_save_writes_none(tmp_path, binary_format):
    assert "id" not in yaml.safe_load((NETWORK_DIR / f"{SCHAEFER}.yaml").read_text())["node_template"]
    net = Network(iri=f"tvbo:networks/{SCHAEFER}")
    assert type(net.node_template) is dm.NodeTemplate
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    template = yaml.safe_load((tmp_path / "n.yaml").read_text())["node_template"]
    assert "id" not in template and "functional_network" in template["parameters"]
    assert list(Network.from_file(str(tmp_path / "n.yaml")).node_template.parameters) == ["functional_network"]


def test_a_network_an_older_tvbo_saved_reads_without_the_id_it_wrote_on_its_template():
    """The file is what ``save`` wrote before templates had their own class: its template carries ``id: -1`` beside what every node shares."""
    assert yaml.safe_load(SAVED_BEFORE.read_text())["node_template"]["id"] == -1
    net = Network.from_file(str(SAVED_BEFORE))
    assert type(net.node_template) is dm.NodeTemplate and net.node_template.parameters
    assert all(sorted(node.parameters) == sorted(net.node_template.parameters) for node in net.nodes)


def test_a_curated_sidecar_as_an_older_tvbo_wrote_it_reads_without_the_template_id(tmp_path):
    """A copy of a curated sidecar from before, such as a kit holds, carries ``id: -1`` on its template and reads as the curated network does."""
    record = yaml.safe_load((NETWORK_DIR / f"{SCHAEFER}.yaml").read_text())
    record["node_template"] = {"id": -1, **record["node_template"]}
    record["data_file"] = str(NETWORK_DIR / record["data_file"])
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    net = Network.from_file(str(tmp_path / "n.yaml"))
    assert type(net.node_template) is dm.NodeTemplate and list(net.node_template.parameters) == ["functional_network"]
    assert net.number_of_nodes == Network(iri=f"tvbo:networks/{SCHAEFER}").number_of_nodes


@pytest.mark.parametrize("value", [0, 2, -2])
def test_a_template_id_other_than_the_one_an_older_tvbo_wrote_is_refused(tmp_path, value):
    record = yaml.safe_load(SAVED_BEFORE.read_text())
    record["node_template"]["id"] = value
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    with pytest.raises(ValueError, match=r"`node_template` sets `id`"):
        Network.from_file(str(tmp_path / "n.yaml"))
