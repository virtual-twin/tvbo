"""Per-region values declared on a network by label land on the regions they name.

A network loaded from data (a BIDS directory, a companion file, a curated IRI, a generator) numbers its own nodes. Declaring a per-region value by index pins it to that numbering, and a network that loaded its nodes from data used to discard declared nodes outright, so the value was silently lost. A node declared by ``label`` alone names no index: it is applied to the materialised node carrying that label, whatever position the data put it at, and a label the network does not carry is an error rather than a dropped value.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from tvbo.classes.network import Network
from tvbo.data.param_io import resolve_network_node
from tvbo.datamodel.schema import Parameter
from tvbo.templates.tvboptim.utils import get_node_param_overrides

LABELS = ["L.A1", "R.A1", "L.V1"]


@pytest.fixture
def bids_dir(tmp_path):
    """A three-region BEP017 directory whose node index names each region."""
    rows = ["matrix_index\tnode_file\tnode_index\tlabel"] + [f"{i}\tatlas-Probe\t{i}\t{lab}" for i, lab in enumerate(LABELS)]
    (tmp_path / "atlas-Probe_nodeindices.tsv").write_text("\n".join(rows) + "\n")
    for measure, matrix in (("streamlineCount", np.ones((3, 3)) - np.eye(3)), ("tractLength", np.full((3, 3), 10.0))):
        stem = tmp_path / f"atlas-Probe_meas-{measure}_relmat"
        np.savetxt(f"{stem}.dense.tsv", matrix, delimiter="\t")
        (tmp_path / f"{stem.name}.json").write_text(json.dumps({"RelationshipMeasure": measure}))
    return tmp_path


def _load(tmp_path, bids_dir, nodes, node_template=None):
    spec = {
        "tvbo_class": "tvbo:Network",
        "label": "probe",
        "bids_dir": str(bids_dir),
        "structural_measures": ["streamlineCount", "tractLength"],
        "nodes": nodes,
        **({"node_template": node_template} if node_template else {}),
    }
    path = tmp_path / "net.yaml"
    path.write_text(yaml.safe_dump(spec))
    return Network.from_file(str(path))


def test_a_value_declared_by_label_lands_on_that_region(tmp_path, bids_dir):
    """Declared out of the data's order, each value still reaches the node its label names."""
    net = _load(
        tmp_path,
        bids_dir,
        [{"label": "L.V1", "parameters": {"a": {"value": -0.3}}}, {"label": "L.A1", "parameters": {"a": {"value": -0.1}}}],
    )
    by_label = {n.label: n for n in net.nodes}
    assert [n.label for n in net.nodes] == LABELS
    assert float(by_label["L.A1"].parameters["a"].value) == -0.1
    assert float(by_label["L.V1"].parameters["a"].value) == -0.3
    assert not by_label["R.A1"].parameters


def test_a_node_declaring_its_id_unassigned_is_still_matched_by_label(tmp_path, bids_dir):
    """``id: null`` says the same as no ``id``, so the data's id stands and the declared value lands."""
    net = _load(tmp_path, bids_dir, [{"id": None, "label": "R.A1", "parameters": {"a": {"value": 0.4}}}])
    by_label = {n.label: n for n in net.nodes}
    assert by_label["R.A1"].id == LABELS.index("R.A1")
    assert float(by_label["R.A1"].parameters["a"].value) == 0.4


def test_the_backend_receives_the_values_in_the_datas_node_order(tmp_path, bids_dir):
    """What codegen builds from the nodes is indexed by the data's order, not the declaration's."""
    net = _load(
        tmp_path,
        bids_dir,
        [{"label": lab, "parameters": {"a": {"value": v}}} for lab, v in (("L.V1", -0.3), ("R.A1", -0.2), ("L.A1", -0.1))],
    )
    assert get_node_param_overrides(net, 3, {"a": 0.0}) == {"a": [-0.1, -0.2, -0.3]}


def test_declared_values_merge_with_what_the_data_attached(tmp_path, bids_dir):
    """A region size the BIDS directory carries survives beside a declared excitability, and both read by name."""
    (bids_dir / "atlas-Probe_desc-regionSize.tsv").write_text("label\troi_size\nL.A1\t12.0\nR.A1\t9.0\nL.V1\t30.0\n")
    net = _load(tmp_path, bids_dir, [{"label": "L.A1", "parameters": {"a": {"value": -0.1}}}])
    node = {n.label: n for n in net.nodes}["L.A1"]
    assert float(node.parameters["a"].value) == -0.1
    assert float(node.parameters["roi_size"].value) == 12.0


def test_a_label_the_network_does_not_carry_raises(tmp_path, bids_dir):
    with pytest.raises(ValueError, match="R.V1"):
        _load(tmp_path, bids_dir, [{"label": "R.V1", "parameters": {"a": {"value": 0.0}}}])


def test_label_and_id_keys_cannot_be_mixed(tmp_path, bids_dir):
    with pytest.raises(ValueError, match="mixes"):
        _load(tmp_path, bids_dir, [{"label": "L.A1"}, {"id": 1, "label": "R.A1"}])


def test_a_label_declared_twice_raises(tmp_path, bids_dir):
    with pytest.raises(ValueError, match="more than once"):
        _load(tmp_path, bids_dir, [{"label": "L.A1"}, {"label": "L.A1"}])


def test_an_id_keyed_list_is_still_the_graph(tmp_path):
    """Nodes that carry ids are an explicit graph, as before."""
    net = Network(nodes=[{"id": 0, "label": "x"}, {"id": 1, "label": "y"}])
    assert [n.label for n in net.nodes] == ["x", "y"]


def test_the_datamodel_constructs_a_node_named_by_label_alone():
    """The dialect gives it the unassigned sentinel, so a study's datamodel pass does not reject it."""
    from tvbo.datamodel import tvbo_datamodel as dm
    from tvbo.datamodel.dialect import UNASSIGNED_NODE_ID

    assert dm.Node(label="L.A1").id == UNASSIGNED_NODE_ID


def test_label_keyed_nodes_without_data_to_match_raise():
    """With no data materialising nodes there is nothing to attach to, and a guessed index is exactly what the key avoids."""
    with pytest.raises(ValueError, match="materialised no node"):
        Network(nodes=[{"label": "L.A1", "parameters": {"a": {"value": 0.0}}}])


def test_a_study_carries_label_keyed_nodes_to_its_experiment(tmp_path, bids_dir):
    """The study's datamodel pass and the runnable experiment both accept the declaration, and the values land by label."""
    from tvbo import SimulationStudy

    network = {
        "bids_dir": str(bids_dir),
        "structural_measures": ["streamlineCount", "tractLength"],
        "nodes": [{"label": "R.A1", "parameters": {"a": {"value": -0.2}}}],
    }
    recipe = {
        "tvbo_class": "tvbo:SimulationStudy",
        "citekey": "Probe",
        "experiments": [{"id": 1, "dynamics": {"name": "Generic2dOscillator"}, "network": network}],
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(recipe))
    exp = SimulationStudy.from_file(str(path)).get_experiment(1)
    by_label = {n.label: n for n in exp.network.nodes}
    assert int(by_label["R.A1"].id) == 1
    assert float(by_label["R.A1"].parameters["a"].value) == -0.2


IN_SET = {"parameters": {"in_set": {"value": 0.0}}}


def test_a_template_default_reaches_nodes_that_declare_other_keys(tmp_path, bids_dir):
    """A template key is merged into a node's parameters, beside the keys that node already declares, rather than skipped because the node declares any."""
    net = _load(tmp_path, bids_dir, [{"label": "L.A1", "parameters": {"a": {"value": -0.1}}}], IN_SET)
    node = {n.label: n for n in net.nodes}["L.A1"]
    assert float(node.parameters["a"].value) == -0.1
    assert [float(n.parameters["in_set"].value) for n in net.nodes] == [0.0, 0.0, 0.0]


def test_template_values_are_typed(tmp_path, bids_dir):
    """Every node receives the datamodel's Parameter, its own instance, never the template's raw mapping."""
    net = _load(tmp_path, bids_dir, [{"label": "L.A1", "parameters": {"a": {"value": -0.1}}}], IN_SET)
    received = [n.parameters["in_set"] for n in net.nodes]
    assert all(type(p) is Parameter for p in received)
    assert len({id(p) for p in received}) == len(received)


def test_an_explicit_key_wins_over_the_template(tmp_path, bids_dir):
    """A key a node declares keeps its value; the template fills only the nodes that leave it out."""
    net = _load(tmp_path, bids_dir, [{"label": "R.A1", "parameters": {"in_set": {"value": 1.0}}}], IN_SET)
    assert [float(n.parameters["in_set"].value) for n in net.nodes] == [0.0, 1.0, 0.0]


def test_a_node_set_is_a_template_default_plus_labelled_members(tmp_path, bids_dir):
    """A 0/1 membership declared once as a default and once per member reads as one per-node vector in the data's order, whichever of the two is applied last."""
    members = [{"label": lab, "parameters": {"in_set": {"value": 1.0}}} for lab in ("L.V1", "L.A1")]
    net = _load(tmp_path, bids_dir, members, IN_SET)
    np.testing.assert_array_equal(resolve_network_node(net, "in_set"), [1.0, 0.0, 1.0])
    for order in (
        (net._expand_node_template, net._apply_declared_nodes),
        (net._apply_declared_nodes, net._expand_node_template),
    ):
        for apply in order:
            apply()
        np.testing.assert_array_equal(resolve_network_node(net, "in_set"), [1.0, 0.0, 1.0])
