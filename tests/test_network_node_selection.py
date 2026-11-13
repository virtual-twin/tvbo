"""A network declared as a node subset of another network.

A selection names the nodes to keep or to drop by label. The network it sits on is then the selected nodes in the order the source stores them, with every layer and every per-node array restricted alike, and its transforms see the restricted layers. Expected values are computed in numpy from the same matrices.

Two tests hold neighbouring behaviour to account without the slot: a stored network whose node list is not as long as its matrices, and a label a reconciled reference cannot find. The rest follows the selection through each way a network gets its nodes and arrays, through a save, a kit and a rebuild, and through what it refuses.
"""

from __future__ import annotations

import re

import h5py
import numpy as np
import pydantic
import pytest
import yaml

pytest.importorskip("jax")

from tvbo import Network, SimulationExperiment  # noqa: E402
from tvbo.data import matrix_io  # noqa: E402
from tvbo.data.matrix_io import write_matrix  # noqa: E402
from tvbo.datamodel import pydantic as pydantic_model  # noqa: E402
from tvbo.datamodel.schema import Dataset  # noqa: E402
from tvbo.run.study import rebuild_invalidated  # noqa: E402

LABELS = ["A", "B", "C", "D"]
W = np.array([[0.0, 0.2, 0.5, 0.1], [0.2, 0.0, 0.9, 0.4], [0.5, 0.9, 0.0, 0.3], [0.1, 0.4, 0.3, 0.0]])
LENGTH = np.array([[0.0, 30.0, 60.0, 15.0], [30.0, 0.0, 45.0, 75.0], [60.0, 45.0, 0.0, 90.0], [15.0, 75.0, 90.0, 0.0]])
FC = np.array([[0.0, 0.7, 0.1, 0.3], [0.7, 0.0, 0.2, 0.6], [0.1, 0.2, 0.0, 0.5], [0.3, 0.6, 0.5, 0.0]])
GAIN = np.array([1.0, 2.0, 3.0, 4.0])
QUERY = {"atlas": "Toy", "desc": "SC", "suffix": "relmat"}


def _ix(labels):
    return [LABELS.index(k) for k in labels]


def _sub(matrix, labels):
    return matrix[np.ix_(_ix(labels), _ix(labels))]


@pytest.fixture
def stored(tmp_path):
    """A four-node network on disk with weights, lengths, an observational layer and a per-node array."""
    path = tmp_path / "stored.yaml"
    Network.from_matrix(weights=W, lengths=LENGTH, labels=LABELS).save(path)
    with h5py.File(path.with_suffix(".h5"), "a") as h:
        write_matrix(h["edges"].create_group("BoldCorrelation"), FC, fmt="dense")
        h.require_group("nodes").create_dataset("gain", data=GAIN)
    doc = yaml.safe_load(path.read_text())
    doc["observational_measures"] = ["BoldCorrelation"]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _selecting(stored, selection=None, extra=""):
    """A network file that includes the stored one whole and adds a selection."""
    path = stored.parent / "selecting.yaml"
    lines = ["tvbo_class: tvbo:Network", "<<: !include stored.yaml", "data_file: stored.h5"]
    if selection is not None:
        lines.append(f"node_selection: {yaml.safe_dump(selection, default_flow_style=True).strip()}")
    path.write_text("\n".join(lines) + "\n" + extra)
    return path


def _dense(net, layer):
    return np.asarray(net.matrix(layer, format="dense"))


def test_without_a_selection_the_including_network_is_the_stored_one(stored):
    net = Network.from_file(str(_selecting(stored)))
    assert [str(n.label) for n in net.nodes] == LABELS
    np.testing.assert_array_equal(_dense(net, "weight"), W)


def test_dropping_a_node_restricts_every_layer_and_every_per_node_array(stored):
    net = Network.from_file(str(_selecting(stored, {"drop": ["B"]})))
    kept = ["A", "C", "D"]
    assert [str(n.label) for n in net.nodes] == kept
    assert [int(n.id) for n in net.nodes] == [0, 1, 2]
    assert net.number_of_nodes == 3
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, kept))
    np.testing.assert_array_equal(_dense(net, "length"), _sub(LENGTH, kept))
    np.testing.assert_array_equal(net.observations["BoldCorrelation"], _sub(FC, kept))
    np.testing.assert_array_equal(np.asarray(net.array("nodes/gain")), GAIN[_ix(kept)])


def test_keeping_is_dropping_the_rest_and_the_stored_order_stands(stored):
    net = Network.from_file(str(_selecting(stored, {"keep": ["D", "A"]})))
    assert [str(n.label) for n in net.nodes] == ["A", "D"]
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, ["A", "D"]))


SCALE_BY_LARGEST = "transforms:\n  - name: weight\n    equation: {rhs: 'weight / max(weight)'}\n"


def test_without_a_selection_a_transform_scales_by_the_largest_stored_weight(stored):
    net = Network.from_file(str(_selecting(stored, extra=SCALE_BY_LARGEST)))
    np.testing.assert_allclose(_dense(net, "weight"), W / W.max())


def test_a_transform_sees_the_restricted_layer(stored):
    """Scaling by the largest weight after dropping the node that carries it scales by the largest weight left."""
    net = Network.from_file(str(_selecting(stored, {"drop": ["B"]}, extra=SCALE_BY_LARGEST)))
    kept = ["A", "C", "D"]
    np.testing.assert_allclose(_dense(net, "weight"), _sub(W, kept) / _sub(W, kept).max())


def _trajectory(network_spec, tmp_path, name):
    """A short delayed relaxation on one network, as node columns."""
    spec = {
        "id": 1,
        "label": name,
        "dynamics": {
            "name": "Relaxation",
            "system_type": "continuous",
            "output": ["x"],
            "coupling_inputs": {"c": {}},
            "state_variables": {"x": {"equation": {"rhs": "1 - x + c"}, "initial_value": 0.1, "coupling_variable": True}},
        },
        "network": {
            **network_spec,
            "coupling": {
                "c": {
                    "delayed": True,
                    "incoming_states": ["x"],
                    "pre_expression": {"rhs": "x"},
                    "post_expression": {"rhs": "gx"},
                }
            },
            "parameters": {"conduction_speed": {"value": 3.0, "unit": "mm/ms"}},
        },
        "integration": {"method": "euler", "step_size": 1.0, "duration": 60.0, "transient_time": 0.0, "unit": "ms"},
        "execution": {"precision": "float64"},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    exp = SimulationExperiment.from_file(str(path))
    data = np.asarray(exp.run("tvboptim").integration.data)
    return data.reshape(-1, len(exp.network.nodes)), [str(n.label) for n in exp.network.nodes]


def _reading(path, labels):
    """A network on its own node list whose weights and lengths are read from a stored network by label."""
    return {
        "nodes": [{"id": i, "label": k} for i, k in enumerate(labels)],
        "edges": [
            {"label": "weight", "used": {"iri": str(path), "output": "weight", "reconcile": "by_label"}},
            {"label": "length", "unit": "mm", "used": {"iri": str(path), "output": "length", "reconcile": "by_label"}},
        ],
    }


def test_a_run_on_the_selection_is_a_run_on_the_same_nodes_written_out(stored, tmp_path):
    kept = ["A", "C", "D"]
    by_hand = tmp_path / "by_hand.yaml"
    Network.from_matrix(weights=_sub(W, kept), lengths=_sub(LENGTH, kept), labels=kept).save(by_hand)
    expected, expected_labels = _trajectory(_reading(by_hand, kept), tmp_path, "written_out")
    selected, labels = _trajectory({**_reading(stored, LABELS), "node_selection": {"drop": ["B"]}}, tmp_path, "selected")
    assert labels == expected_labels == kept
    np.testing.assert_array_equal(selected, expected)


def _bound_to_its_subject(tmp_path, **extra):
    """A network whose weights are read from the run's own subject, bound to the one member of a cohort."""
    folder = tmp_path / "sc" / "sub-01" / "dwi"
    folder.mkdir(parents=True)
    Network.from_matrix(weights=W, lengths=LENGTH, labels=LABELS).save(folder / "sub-01_atlas-Toy_desc-SC_relmat.yaml")
    dataset = Dataset(dataset_id="toy_sc", bids_root=str(tmp_path / "sc"), cohorts=[{"cohort_id": "one", "members": ["01"]}])
    spec = {
        "nodes": [{"id": i, "label": k} for i, k in enumerate(LABELS)],
        "edges": [{"label": "weight", "used": {"cohort": "one", "query": QUERY, "output": "weight", "reconcile": "by_label"}}],
        **extra,
    }
    return Network(**spec).bind_references(subject="01", datasets=[dataset], results_root=tmp_path)


def test_without_a_selection_a_per_subject_layer_is_the_subjects_own(tmp_path):
    np.testing.assert_array_equal(_dense(_bound_to_its_subject(tmp_path), "weight"), W)


def test_a_per_subject_layer_follows_the_selection(tmp_path):
    """A layer read from the run's own subject is restricted like a stored one."""
    net = _bound_to_its_subject(tmp_path, node_selection={"drop": ["B"]})
    assert [str(n.label) for n in net.nodes] == ["A", "C", "D"]
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, ["A", "C", "D"]))


def test_a_label_the_network_does_not_carry_is_named(stored):
    with pytest.raises(ValueError, match=r"Z") as raised:
        Network.from_file(str(_selecting(stored, {"drop": ["Z"]})))
    assert all(label in str(raised.value) for label in LABELS)


def test_keep_and_drop_together_are_refused(stored):
    with pytest.raises(ValueError, match=r"keep.*drop|drop.*keep"):
        Network.from_file(str(_selecting(stored, {"keep": ["A", "C"], "drop": ["B"]})))


def test_a_selection_that_leaves_no_node_is_refused(stored):
    with pytest.raises(ValueError, match=r"no node"):
        Network.from_file(str(_selecting(stored, {"drop": LABELS})))


def test_a_node_list_shorter_than_the_stored_matrices_is_refused(stored, tmp_path):
    """A node list of two beside four-by-four stored layers is refused at load with the count, not run over the four nodes the layers index."""
    doc = yaml.safe_load(stored.read_text())
    doc["data_file"] = str(stored.with_suffix(".h5"))
    doc["nodes"] = [n for n in doc["nodes"] if str(n["label"]) in ("A", "C")]
    doc["number_of_nodes"] = 2
    short = tmp_path / "short.yaml"
    short.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match=r"2 node"):
        net = Network.from_file(str(short))
        _dense(net, "weight")


def test_a_label_a_reconciled_reference_cannot_find_is_named(stored, tmp_path):
    """A ``by_label`` reference whose network lacks one of the model's labels raises a ``ValueError`` naming that label."""
    spec = {
        "tvbo_class": "tvbo:Network",
        "nodes": [{"id": 0, "label": "A"}, {"id": 1, "label": "Z"}],
        "edges": [{"label": "weight", "used": {"iri": str(stored), "output": "weight", "reconcile": "by_label"}}],
    }
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    with pytest.raises(ValueError, match=r"Z"):
        _dense(Network.from_file(str(path)), "weight")


def test_a_node_list_longer_than_the_stored_matrices_is_refused_and_a_projection_is_not_compared(stored, tmp_path):
    """The check is on the layers stored for these nodes: a matrix that is not square, as a projection onto another network's nodes is, says nothing about their number."""
    doc = yaml.safe_load(stored.read_text())
    doc["data_file"] = str(stored.with_suffix(".h5"))
    doc["nodes"] = [*doc["nodes"], {"id": 4, "label": "E"}]
    doc["number_of_nodes"] = 5
    longer = tmp_path / "longer.yaml"
    longer.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match=r"lists 5 node\(s\).*stored\.h5 stores the layer 'weight' as 4 x 4.*node_selection"):
        Network.from_file(str(longer))
    with h5py.File(stored.with_suffix(".h5"), "a") as h:
        write_matrix(h["edges"].create_group("gain"), np.ones((2, 4)), fmt="dense")
    assert Network.from_file(str(stored)).number_of_nodes == 4


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"drop": ["B", "B"]}, r"`node_selection\.drop` lists \['B'\] more than once"),
        ({"keep": ["A", "A", "C"]}, r"`node_selection\.keep` lists \['A'\] more than once"),
        ({}, r"states neither `keep` nor `drop`"),
        ({"keep": ["A"], "drop": ["B"]}, r"states both `keep` and `drop`"),
        (
            {"keep": ["Z", "A"]},
            r"`node_selection\.keep` names \['Z'\], which this network does not carry\. It carries 4 node\(s\): A, B, C, D\.",
        ),
        ({"drop": ["A", "B", "C", "D"]}, r"leaves no node of the 4"),
    ],
)
def test_a_selection_that_selects_nothing_definite_is_refused(stored, selection, message):
    with pytest.raises(ValueError, match=message):
        Network.from_file(str(_selecting(stored, selection)))


def test_an_unknown_label_on_a_large_network_is_answered_with_the_nearest(tmp_path):
    labels = [f"region_{side}_{i:02d}" for side in ("left", "right") for i in range(15)]
    with pytest.raises(
        ValueError, match=r"names \['region_left_7'\].*carries 30 node\(s\): the nearest are .*region_left_07"
    ) as raised:
        Network(nodes=[{"id": i, "label": k} for i, k in enumerate(labels)], node_selection={"drop": ["region_left_7"]})
    assert "region_right_14" not in str(raised.value)


def test_a_label_two_nodes_share_selects_neither(tmp_path):
    nodes = [{"id": i, "label": k} for i, k in enumerate(["A", "B", "B"])]
    with pytest.raises(ValueError, match=r"names \['B'\], a label several nodes of this network carry"):
        Network(nodes=nodes, node_selection={"drop": ["B"]})


def test_a_label_is_matched_through_a_nodes_other_names():
    """A node's alternate name selects it, as it aligns it in a reconciliation by label."""
    nodes = [
        {"id": 0, "label": "A", "alternateName": ["Alpha"]},
        {"id": 1, "label": "B"},
        {"id": 2, "label": "C", "alternateName": ["Gamma"]},
    ]
    assert Network(nodes=nodes, node_selection={"keep": ["Gamma", "Alpha"]}).node_labels == ["A", "C"]
    with pytest.raises(ValueError, match=r"lists \['A', 'Alpha'\] more than once"):
        Network(nodes=nodes, node_selection={"drop": ["A", "Alpha"]})


def test_listed_edges_go_with_their_nodes_and_are_renumbered():
    """A graph listed edge by edge under identifiers that are not positions: the edges of the dropped node go, the others address the kept nodes by their new numbers, and weights, lengths and delays are those of the smaller graph."""
    spec = {
        "nodes": [{"id": 10, "label": "A"}, {"id": 20, "label": "B"}, {"id": 30, "label": "C"}],
        "edges": [
            {"source": 10, "target": 20, "parameters": {"weight": {"value": 0.5}, "length": {"value": 30.0}}},
            {"source": 20, "target": 30, "parameters": {"weight": {"value": 0.7}, "length": {"value": 45.0}}},
            {"source": 10, "target": 30, "parameters": {"weight": {"value": 0.9}, "length": {"value": 60.0}}},
        ],
    }
    whole, net = Network(**spec), Network(**spec, node_selection={"drop": ["B"]})
    assert [(int(n.id), str(n.label)) for n in net.nodes] == [(0, "A"), (1, "C")]
    assert [(e.source, e.target) for e in net.edges] == [(0, 1)]
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(_dense(net, layer), _dense(whole, layer)[np.ix_([0, 2], [0, 2])])
    np.testing.assert_array_equal(
        np.asarray(net.calculate_delays()), np.asarray(whole.calculate_delays())[np.ix_([0, 2], [0, 2])]
    )
    assert net.number_of_nodes == 2 and whole.number_of_nodes == 3


def test_per_node_values_stay_with_their_nodes(stored):
    """A template value and a value declared for a kept node by its label land on the kept nodes; a declaration for a dropped node is refused."""
    recipe = {
        "data_file": str(stored.with_suffix(".h5")),
        "node_selection": {"drop": ["B"]},
        "node_template": {"parameters": {"bias": {"value": 0.25}}},
    }
    net = Network(
        **recipe,
        nodes={
            "A": {"parameters": {"gain": {"value": 1.5}}},
            "C": {"parameters": {"gain": {"value": 3.5}}},
            "D": {"parameters": {"gain": {"value": 4.5}}},
        },
    )
    np.testing.assert_array_equal(net.node_parameter_vectors["gain"], [1.5, 3.5, 4.5])
    np.testing.assert_array_equal(net.node_parameter_vectors["bias"], [0.25, 0.25, 0.25])
    with pytest.raises(ValueError, match=r"Nodes \['B'\] are declared by label and dropped by `node_selection`"):
        Network(**recipe, nodes={"B": {"parameters": {"gain": {"value": 2.5}}}})


@pytest.mark.parametrize("fmt", ["csr", "coo"])
def test_a_sparse_layer_and_an_edge_parameter_are_restricted_in_their_format(stored, fmt):
    from scipy import sparse

    local = sparse.random(4, 4, density=0.6, random_state=3, format="csr")
    with h5py.File(stored.with_suffix(".h5"), "a") as h:
        write_matrix(h["edges"].create_group("local"), local, fmt=fmt)
        write_matrix(h["edges/weight"].require_group("edge_parameters").create_group("speed"), LENGTH / 10.0, fmt="dense")
    net = Network.from_file(str(_selecting(stored, {"keep": ["A", "C", "D"]})))
    got = net.array("edges/local")
    assert sparse.issparse(got) and got.format == fmt
    np.testing.assert_array_equal(got.toarray(), local.toarray()[np.ix_([0, 2, 3], [0, 2, 3])])
    np.testing.assert_array_equal(net.edge_parameter_arrays()["weight"]["speed"], _sub(LENGTH, ["A", "C", "D"]) / 10.0)
    assert net.matrix_info("local").shape == (3, 3) and net.matrix_info("nodes/gain").shape == (3,)


def test_a_recipe_selects_from_a_stored_network_it_names(stored):
    """The form a recipe writes: the network names its companion and the nodes to leave out."""
    net = Network(data_file=str(stored.with_suffix(".h5")), node_selection={"drop": ["C"]})
    assert net.node_labels == ["A", "B", "D"] and isinstance(net._store, matrix_io.NodeSelectedStore)
    np.testing.assert_array_equal(_dense(net, "length"), _sub(LENGTH, ["A", "B", "D"]))


def test_a_selection_of_a_selection_selects_from_what_is_left(stored):
    inner = _selecting(stored, {"drop": ["B"]})
    net = Network(data_file=str(inner), node_selection={"keep": ["D", "C"]})
    assert net.node_labels == ["C", "D"] and [int(n.id) for n in net.nodes] == [0, 1]
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, ["C", "D"]))
    np.testing.assert_array_equal(np.asarray(net.array("nodes/gain")), GAIN[_ix(["C", "D"])])
    with pytest.raises(
        ValueError, match=r"names \['B'\], which this network does not carry\. It carries 3 node\(s\): A, C, D"
    ):
        Network(data_file=str(inner), node_selection={"drop": ["B"]})


CURATED = "tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat"


def test_a_curated_network_is_selected_like_a_stored_one():
    """Seventeen lobar regions without the two cerebellar ones: layers and node coordinates are those of the fifteen."""
    whole = Network(iri=CURATED)
    drop = [k for k in whole.node_labels if k.endswith("Cerebellum")]
    kept = [i for i, k in enumerate(whole.node_labels) if k not in drop]
    net = Network(iri=CURATED, node_selection={"drop": drop})
    assert len(drop) == 2 and net.node_labels == [whole.node_labels[i] for i in kept]
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(_dense(net, layer), _dense(whole, layer)[np.ix_(kept, kept)])
    np.testing.assert_array_equal(
        np.asarray(net.array("nodes/coordinates")), np.asarray(whole.array("nodes/coordinates"))[kept]
    )
    np.testing.assert_array_equal(net.node_positions(), whole.node_positions()[kept])


def test_a_generated_graph_and_an_atlas_connectome_are_selected_after_they_are_built():
    ring = {"graph_generator": {"name": "ring", "type": "Cycle"}, "number_of_nodes": 5}
    whole, net = Network(**ring), Network(**ring, node_selection={"drop": ["node_1"]})
    assert net.node_labels == ["node_0", "node_2", "node_3", "node_4"]
    np.testing.assert_array_equal(_dense(net, "weight"), _dense(whole, "weight")[np.ix_([0, 2, 3, 4], [0, 2, 3, 4])])

    atlas = {"parcellation": {"atlas": {"name": "DesikanKilliany"}}}
    whole = Network(**atlas)
    cortical = [i for i, k in enumerate(whole.node_labels) if str(k).startswith("ctx-")]
    net = Network(**atlas, node_selection={"keep": [whole.node_labels[i] for i in cortical]})
    assert 0 < len(cortical) < whole.number_of_nodes == 87 and net.number_of_nodes == len(cortical)
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(_dense(net, layer), _dense(whole, layer)[np.ix_(cortical, cortical)])


def test_a_reference_is_read_for_the_selected_nodes(stored, tmp_path):
    """A layer and a transform argument aligned by label follow the selection; an array in the unselected node count that is read by position is refused, with both counts."""
    by_label = {"iri": str(stored), "output": "weight", "reconcile": "by_label"}
    spec = {
        "nodes": [{"id": i, "label": k} for i, k in enumerate(LABELS)],
        "node_selection": {"drop": ["B"]},
        "edges": [{"label": "weight", "used": by_label}],
        "transforms": [
            {"name": "weight", "equation": {"rhs": "weight / max(reference)"}, "arguments": {"reference": {"used": by_label}}}
        ],
    }
    kept = _sub(W, ["A", "C", "D"])
    np.testing.assert_allclose(_dense(Network(**spec), "weight"), kept / kept.max())
    positional = {**spec, "edges": [{"label": "weight", "used": {"iri": str(stored), "output": "weight"}}], "transforms": []}
    with pytest.raises(
        ValueError,
        match=r"shape \(4, 4\) for a network of 3 nodes \(the 3 its `node_selection` keeps of 4\).*reconcile: by_label",
    ):
        _dense(Network(**positional), "weight")


def test_another_network_reads_a_selecting_network_as_it_delivers(stored):
    """A reference that names the selecting network gets its layer on the selected nodes with the transform applied there, which is how the other order of selection and transform is written."""
    selecting = _selecting(stored, {"drop": ["B"]}, extra=SCALE_BY_LARGEST)
    reader = Network(
        nodes=[{"id": 0, "label": "D"}, {"id": 1, "label": "A"}],
        edges=[{"label": "weight", "used": {"iri": str(selecting), "output": "weight", "reconcile": "by_label"}}],
    )
    kept = _sub(W, ["A", "C", "D"])
    np.testing.assert_allclose(_dense(reader, "weight"), (_sub(W, ["D", "A"])) / kept.max())


def test_a_per_node_measure_read_from_another_network_follows_the_selection(stored, tmp_path):
    """A network parameter gathered from the nodes of a referenced network comes in that network's node order and is cut like a stored per-node array."""
    source = Network(
        nodes=[{"id": i, "label": k, "parameters": {"alpha": {"value": float(i + 1)}}} for i, k in enumerate(LABELS)]
    )
    source.save(tmp_path / "measures.yaml")
    extra = f"parameters:\n  gradient: {{source: {tmp_path / 'measures.yaml'}, measure: alpha}}\n"
    net = Network.from_file(str(_selecting(stored, {"drop": ["B"]}, extra=extra)))
    np.testing.assert_array_equal(np.asarray(net.parameters["gradient"].value, dtype=float), [1.0, 3.0, 4.0])
    whole = Network.from_file(str(_selecting(stored, extra=extra)))
    np.testing.assert_array_equal(np.asarray(whole.parameters["gradient"].value, dtype=float), [1.0, 2.0, 3.0, 4.0])


def _measured(tmp_path, source_labels, own_labels):
    """The values a network on *own_labels* reads for `alpha` from a source on *source_labels* holding 1, 2, 3, ... in its own order."""
    nodes = [{"id": i, "label": k, "parameters": {"alpha": {"value": float(i + 1)}}} for i, k in enumerate(source_labels)]
    Network(nodes=nodes).save(tmp_path / "measures.yaml")
    reader = Network(
        nodes=[{"id": i, "label": k} for i, k in enumerate(own_labels)],
        parameters={"gradient": {"source": str(tmp_path / "measures.yaml"), "measure": "alpha"}},
    )
    return np.asarray(reader.parameters["gradient"].value, dtype=float).tolist()


def test_a_per_node_measure_follows_the_labels_when_the_source_orders_the_same_nodes_otherwise(tmp_path):
    """A source that carries every label of the network gives each node its own value, also with as many nodes; a source that names the same number of nodes otherwise is read in its order."""
    assert _measured(tmp_path, ["C", "B", "A"], ["A", "B", "C"]) == [3.0, 2.0, 1.0]
    assert _measured(tmp_path, ["X", "Y", "Z"], ["A", "B", "C"]) == [1.0, 2.0, 3.0]


def test_a_per_node_measure_from_a_source_that_shares_only_some_labels_is_refused(tmp_path):
    """A source that shares some of the network's labels and lacks others is neither the same nodes nor another nomenclature for them, and one that shares none is another nomenclature only with as many nodes; each other source is refused, naming the labels it lacks and both conditions for a read in order."""
    rule = re.escape(
        "A measure is read in the source's order only from a network with as many nodes that shares no label with this one"
    )
    for source, lacking in ((["B", "A", "z"], r"\['C'\]"), (["A", "A", "B"], r"\['C'\]"), (["x", "y"], r"\['A', 'B', 'C'\]")):
        with pytest.raises(ValueError, match=rf"has no node labelled {lacking}\. {rule}"):
            _measured(tmp_path, source, ["A", "B", "C"])


def test_a_per_node_measure_read_into_a_saved_selection_is_aligned_by_label(stored, tmp_path):
    """A network loaded with its selection already applied reads the per-node measure of its source by node label; a source that lacks one of its labels is refused with both counts."""
    source = Network(
        nodes=[{"id": i, "label": k, "parameters": {"alpha": {"value": float(i + 1)}}} for i, k in enumerate(LABELS)]
    )
    source.save(tmp_path / "measures.yaml")
    Network.from_file(str(_selecting(stored, {"drop": ["B"]}))).save(tmp_path / "saved.yaml")
    doc = yaml.safe_load((tmp_path / "saved.yaml").read_text())
    doc["parameters"]["gradient"] = {"source": str(tmp_path / "measures.yaml"), "measure": "alpha"}
    (tmp_path / "saved.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    back = Network.from_file(str(tmp_path / "saved.yaml"))
    np.testing.assert_array_equal(np.asarray(back.parameters["gradient"].value, dtype=float), [1.0, 3.0, 4.0])

    Network(nodes=[{"id": i, "label": k, "parameters": {"alpha": {"value": 1.0}}} for i, k in enumerate(["A", "B"])]).save(
        tmp_path / "short.yaml"
    )
    doc["parameters"]["gradient"]["source"] = str(tmp_path / "short.yaml")
    (tmp_path / "saved.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match=r"'gradient'.*2 values of 'alpha'.*3 nodes.*\['C', 'D'\]"):
        Network.from_file(str(tmp_path / "saved.yaml"))


_PRODUCERS = "import numpy as np\n\ndef square(n):\n    return np.ones((int(n), int(n)))\n"


def test_a_produced_layer_has_to_come_in_the_selected_count(stored, tmp_path, monkeypatch):
    """A layer a `producer:` computes is served when it has a row for each kept node and refused, with both counts, when it is built for another number of nodes."""
    from tvbo.data import param_io

    (tmp_path / "selection_producers.py").write_text(_PRODUCERS)
    monkeypatch.syspath_prepend(str(tmp_path))
    param_io.clear_cache()

    def produced(n):
        call = {"callable": {"name": "square", "module": "selection_producers"}, "arguments": {"n": {"name": "n", "value": n}}}
        net = Network(
            data_file=str(stored.with_suffix(".h5")),
            node_selection={"drop": ["B"]},
            edges=[{"label": "gain", "producer": call}],
        )
        return _dense(net, "gain")

    np.testing.assert_array_equal(produced(3), np.ones((3, 3)))
    with pytest.raises(ValueError, match=r"'gain'.*4 x 4.*3 nodes \(the 3 its `node_selection` keeps of 4\)"):
        produced(4)


def test_the_selection_leaves_the_callers_nodes_and_edges_as_they_were():
    """A network built from another's node and edge objects selects on copies, so the other keeps its numbering."""
    first = Network(
        nodes=[{"id": 10, "label": "A"}, {"id": 20, "label": "B"}, {"id": 30, "label": "C"}],
        edges=[{"source": 10, "target": 20}, {"source": 10, "target": 30}],
    )
    second = Network(nodes=first.nodes, edges=first.edges, node_selection={"drop": ["B"]})
    assert [(int(n.id), str(n.label)) for n in second.nodes] == [(0, "A"), (1, "C")]
    assert [(e.source, e.target) for e in second.edges] == [(0, 1)]
    assert [(int(n.id), str(n.label)) for n in first.nodes] == [(10, "A"), (20, "B"), (30, "C")]
    assert [(e.source, e.target) for e in first.edges] == [(10, 20), (10, 30)]


def test_a_label_a_reconciled_reference_cannot_find_on_a_long_axis_is_answered_with_the_nearest(tmp_path):
    labels = [f"region_{i:02d}" for i in range(30)]
    Network.from_matrix(weights=np.ones((30, 30)), labels=labels).save(tmp_path / "long.yaml")
    reader = Network(
        nodes=[{"id": 0, "label": "region_01"}, {"id": 1, "label": "regoin_02"}],
        edges=[{"label": "weight", "used": {"iri": str(tmp_path / "long.yaml"), "output": "weight", "reconcile": "by_label"}}],
    )
    with pytest.raises(ValueError, match=r"\['regoin_02'\].*carries 30 label\(s\): the nearest are region_02"):
        _dense(reader, "weight")


def test_a_sidecar_that_lists_no_node_beside_a_companion_says_so(stored, tmp_path):
    (tmp_path / "bare.yaml").write_text(f"tvbo_class: tvbo:Network\ndata_file: {stored.with_suffix('.h5')}\n")
    with pytest.raises(
        ValueError, match=r"lists no node of its own \(1 placeholder node\(s\)\).*stores the layer '\w+' as 4 x 4"
    ):
        Network.from_file(str(tmp_path / "bare.yaml"))


def test_a_mesh_and_a_route_to_another_network_are_refused(stored):
    with pytest.raises(ValueError, match=r"cannot restrict a network with a route to another network \('gain'\)"):
        Network.from_file(
            str(_selecting(stored, {"drop": ["B"]}, extra="edges:\n  - {label: gain, target_network: sensors.yaml}\n"))
        )
    with h5py.File(stored.with_suffix(".h5"), "a") as h:
        h.require_group("mesh").create_dataset("vertices", data=np.zeros((4, 3), dtype="float32"))
    with pytest.raises(ValueError, match=r"cannot restrict a network that carries a mesh"):
        Network.from_file(str(_selecting(stored, {"drop": ["B"]})))


def test_a_saved_selection_is_the_selected_network_and_says_how_it_was_derived(stored, tmp_path):
    """The companion holds the kept nodes' arrays and the sidecar keeps the selection, marked applied, so a reload is the same network and selects nothing again."""
    net = Network.from_file(str(_selecting(stored, {"drop": ["B"]})))
    net.save(tmp_path / "saved.yaml")
    kept = ["A", "C", "D"]
    doc = yaml.safe_load((tmp_path / "saved.yaml").read_text())
    assert doc["node_selection"] == {"drop": ["B"], "applied": True}
    assert [n["label"] for n in doc["nodes"]] == kept and doc["number_of_nodes"] == 3
    with h5py.File(tmp_path / "saved.h5") as h:
        assert h["edges/weight"].attrs["shape"].tolist() == [3, 3] and h["nodes/gain"].shape == (3,)
    back = Network.from_file(str(tmp_path / "saved.yaml"))
    assert back.node_labels == kept and not isinstance(back._store, matrix_io.NodeSelectedStore)
    for layer, full in (("weight", W), ("length", LENGTH)):
        np.testing.assert_array_equal(_dense(back, layer), _sub(full, kept))
    np.testing.assert_array_equal(np.asarray(back.array("nodes/gain")), GAIN[_ix(kept)])
    back.save(tmp_path / "again.yaml")
    assert yaml.safe_load((tmp_path / "again.yaml").read_text())["nodes"] == doc["nodes"]


def test_a_dumped_selection_is_read_back_without_selecting_twice(stored, tmp_path):
    """A resolved network written out as text states its kept nodes and the applied selection. Read back, one whose layers come by label is the same network, and one that still points at the unselected companion is selected from it again by label."""
    by_label = Network(**_reading(stored, LABELS), node_selection={"drop": ["B"]})
    again = Network.from_string(by_label.to_yaml())
    assert again.node_labels == ["A", "C", "D"] and again.node_selection.applied
    np.testing.assert_array_equal(_dense(again, "weight"), _sub(W, ["A", "C", "D"]))

    dumped = tmp_path / "dumped.yaml"
    dumped.write_text(Network.from_file(str(_selecting(stored, {"drop": ["B"]}))).to_yaml())
    back = Network.from_file(str(dumped))
    assert back.node_labels == ["A", "C", "D"] and back.node_selection.applied
    for layer, full in (("weight", W), ("length", LENGTH)):
        np.testing.assert_array_equal(_dense(back, layer), _sub(full, ["A", "C", "D"]))
    np.testing.assert_array_equal(np.asarray(back.array("nodes/gain")), GAIN[_ix(["A", "C", "D"])])


def test_a_dumped_experiment_reads_back_as_what_ran(stored, tmp_path):
    """The text dump of an experiment over a selected stored network and over a selected curated one names the unselected source and the kept nodes; read back, each is the network that ran."""
    curated = Network(iri=CURATED)
    for network in (
        {"data_file": str(stored.with_suffix(".h5")), "node_selection": {"drop": ["B"]}},
        {"iri": CURATED, "node_selection": {"drop": [k for k in curated.node_labels if k.endswith("Cerebellum")]}},
    ):
        path = tmp_path / "selected.yaml"
        path.write_text(yaml.safe_dump({"id": 1, "label": "selected", "network": network}, sort_keys=False))
        ran = SimulationExperiment.from_file(str(path)).network
        dumped = tmp_path / "dumped.yaml"
        dumped.write_text(SimulationExperiment.from_file(str(path)).to_yaml())
        back = SimulationExperiment.from_file(str(dumped)).network
        assert "applied: true" in dumped.read_text() and back.node_labels == ran.node_labels
        for layer in ("weight", "length"):
            np.testing.assert_array_equal(_dense(back, layer), _dense(ran, layer))


def test_a_dumped_selection_whose_source_has_changed_is_refused(stored, tmp_path):
    """Selected again from a companion that now carries a fifth node, the selection keeps a node the record does not list, which is named."""
    dumped = tmp_path / "dumped.yaml"
    dumped.write_text(Network.from_file(str(_selecting(stored, {"drop": ["B"]}))).to_yaml())
    five = np.pad(W, ((0, 1), (0, 1)), constant_values=0.1)
    Network.from_matrix(weights=five, lengths=five * 100, labels=[*LABELS, "E"]).save(stored)
    with pytest.raises(
        ValueError,
        match=r"beside the 3 node\(s\) this record lists.*keeps 4: 0 listed and no longer given \[\], 1 given and not listed \['E'\]",
    ):
        Network.from_file(str(dumped))


def test_a_dumped_selection_of_a_generated_graph_is_refused_and_a_saved_one_reads_back(tmp_path):
    """A generated graph is generated again on load at the count its record states, which after a selection is the kept count, so a dump would read back as a smaller graph under the kept labels; a saved one carries its arrays."""
    ring = Network(graph_generator={"name": "ring", "type": "Cycle"}, number_of_nodes=5, node_selection={"drop": ["node_1"]})
    dumped = tmp_path / "dumped.yaml"
    dumped.write_text(ring.to_yaml())
    with pytest.raises(ValueError, match=r"applied on a generated graph.*`number_of_nodes` \(4\).*`Network\.save`"):
        Network.from_file(str(dumped))
    ring.save(tmp_path / "saved.yaml")
    saved = Network.from_file(str(tmp_path / "saved.yaml"))
    assert saved.node_labels == ring.node_labels == ["node_0", "node_2", "node_3", "node_4"]
    np.testing.assert_array_equal(_dense(saved, "weight"), _dense(ring, "weight"))


def test_a_changed_selection_on_a_saved_selected_network_is_refused_as_saved(stored, tmp_path):
    """A network read from a saved selection holds only the kept nodes, so a selection changed on it cannot be applied; the refusal says it was saved, not set by hand."""
    Network.from_file(str(_selecting(stored, {"drop": ["B"]}))).save(tmp_path / "saved.yaml")
    net = Network.from_file(str(tmp_path / "saved.yaml"))
    net.node_selection.drop = ["A"]
    net.invalidate_resolution()
    with pytest.raises(
        ValueError, match=r"lists \['A'\], which the selection leaves out.*a saved selected network.*its source network"
    ):
        rebuild_invalidated([net])


def test_a_selection_marked_applied_by_hand_on_the_uncut_nodes_is_refused(stored):
    """`applied` says the listed nodes are the selected ones; a recipe that sets it beside the uncut node list is named for the nodes the selection leaves out."""
    with pytest.raises(ValueError, match=r"marked applied, and this record lists \['B'\], which the selection leaves out"):
        Network.from_file(str(_selecting(stored, {"drop": ["B"], "applied": True})))
    with pytest.raises(
        ValueError, match=r"marked applied, and this record lists \['B', 'C'\], which the selection leaves out"
    ):
        Network.from_file(str(_selecting(stored, {"keep": ["A", "D"], "applied": True})))


def test_a_rebuild_selects_again_and_follows_a_changed_selection(stored):
    """Invalidating the resolution puts the unselected nodes back, so the rebuild applies whatever the selection then says."""
    net = Network(data_file=str(stored.with_suffix(".h5")), node_selection={"drop": ["B"]})
    assert net.node_labels == ["A", "C", "D"]
    net.invalidate_resolution()
    net._resolve()
    assert net.node_labels == ["A", "C", "D"] and [int(n.id) for n in net.nodes] == [0, 1, 2]
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, ["A", "C", "D"]))
    net.node_selection.drop = ["A", "D"]
    net.invalidate_resolution()
    net._resolve()
    assert net.node_labels == ["B", "C"]
    np.testing.assert_array_equal(_dense(net, "length"), _sub(LENGTH, ["B", "C"]))

    listed = Network(
        nodes=[{"id": 10, "label": "A"}, {"id": 20, "label": "B"}, {"id": 30, "label": "C"}],
        edges=[{"source": 10, "target": 20}, {"source": 10, "target": 30}],
        node_selection={"drop": ["B"]},
    )
    listed.invalidate_resolution()
    assert [(int(n.id), str(n.label)) for n in listed.nodes] == [(10, "A"), (20, "B"), (30, "C")]
    assert [(e.source, e.target) for e in listed.edges] == [(10, 20), (10, 30)] and not listed.node_selection.applied


def test_a_rebuild_reads_a_referenced_layer_again_for_the_new_selection(stored):
    """A layer read through `used:` is read again for the nodes a changed selection keeps, as a stored one is."""
    net = Network(**_reading(stored, LABELS), node_selection={"drop": ["B"]})
    np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, ["A", "C", "D"]))
    for drop, kept in ((["C"], ["A", "B", "D"]), (["A", "D"], ["B", "C"])):
        net.node_selection.drop = drop
        net.invalidate_resolution()
        net._resolve()
        assert net.node_labels == kept
        np.testing.assert_array_equal(_dense(net, "weight"), _sub(W, kept))
        np.testing.assert_array_equal(_dense(net, "length"), _sub(LENGTH, kept))


def _override_reaches_the_run(tmp_path, monkeypatch, network, overrides):
    """The network a run receives after `--set` *overrides* on an experiment over *network*: its node labels and the side of its weights."""
    from tvbo.run import study as study_run

    path = tmp_path / "overridden.yaml"
    path.write_text(yaml.safe_dump({"id": 1, "label": "overridden", "network": network}, sort_keys=False))
    experiment = SimulationExperiment.from_file(str(path))
    seen = {}

    def run(experiment, *args, **kwargs):
        net = experiment.network
        seen["labels"], seen["side"] = list(net.node_labels), _dense(net, "weight").shape

    monkeypatch.setattr(study_run, "_run_one", run)
    study_run.run_experiment(experiment, str(path), tmp_path / "results", options=study_run.RunOptions(overrides=overrides))
    return seen


def test_an_override_of_the_whole_selection_runs_it(stored, tmp_path, monkeypatch):
    """`--set network.node_selection` with a whole record, onto a network that selects and onto one that does not, runs the record set."""
    stored_file = {"data_file": str(stored.with_suffix(".h5"))}
    for network in ({**stored_file, "node_selection": {"drop": ["B"]}}, stored_file):
        seen = _override_reaches_the_run(tmp_path, monkeypatch, network, [("network.node_selection", {"keep": ["A", "C"]})])
        assert seen == {"labels": ["A", "C"], "side": (2, 2)}


def test_an_override_of_the_selection_runs_the_new_selection(stored, tmp_path, monkeypatch):
    network = {"data_file": str(stored.with_suffix(".h5")), "node_selection": {"drop": ["B"]}}
    seen = _override_reaches_the_run(tmp_path, monkeypatch, network, [("network.node_selection.drop", ["A", "D"])])
    assert seen == {"labels": ["B", "C"], "side": (2, 2)}


def test_an_override_of_the_size_of_a_selected_graph_runs_the_new_size_selected(tmp_path, monkeypatch):
    network = {
        "graph_generator": {"name": "ring", "type": "Cycle"},
        "number_of_nodes": 5,
        "node_selection": {"drop": ["node_1"]},
    }
    seen = _override_reaches_the_run(tmp_path, monkeypatch, network, [("network.number_of_nodes", 7)])
    assert seen == {"labels": ["node_0", "node_2", "node_3", "node_4", "node_5", "node_6"], "side": (6, 6)}


def test_a_kit_carries_the_selected_network(stored, tmp_path):
    """The frozen network file holds the kept nodes and their arrays, and the experiment read back from the frozen spec runs on the same network."""
    from tvbo.cli import workflow as kit

    spec = {
        "id": 1,
        "dynamics": {
            "name": "Relaxation",
            "system_type": "continuous",
            "output": ["x"],
            "state_variables": {"x": {"equation": {"rhs": "1 - x"}, "initial_value": 0.1}},
        },
        "network": {"data_file": str(stored.with_suffix(".h5")), "node_selection": {"drop": ["B"]}},
        "integration": {"method": "euler", "step_size": 1.0, "duration": 5.0, "unit": "ms"},
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(spec, sort_keys=False))
    experiment = SimulationExperiment.from_file(str(recipe))
    spec_dir = tmp_path / "kit" / "spec"
    spec_dir.mkdir(parents=True)
    (spec_dir / "experiment.yaml").write_text(kit._freeze_spec_yaml(experiment, spec_dir))
    frozen = yaml.safe_load((spec_dir / "network.yaml").read_text())
    assert [n["label"] for n in frozen["nodes"]] == ["A", "C", "D"] and frozen["node_selection"] == {
        "drop": ["B"],
        "applied": True,
    }
    assert "node_selection" not in yaml.safe_load((spec_dir / "experiment.yaml").read_text())["network"]
    rerun = SimulationExperiment.from_file(str(spec_dir / "experiment.yaml"))
    assert rerun.network.node_labels == ["A", "C", "D"]
    for layer, full in (("weight", W), ("length", LENGTH)):
        np.testing.assert_array_equal(_dense(rerun.network, layer), _sub(full, ["A", "C", "D"]))
    assert experiment.network.node_labels == ["A", "C", "D"], "the experiment's own network is as it was after the freeze"


def test_both_generated_forms_hold_the_slot():
    record = pydantic_model.Network(node_selection={"keep": ["A"]})
    assert record.node_selection.keep == ["A"] and record.node_selection.applied is None
    with pytest.raises(pydantic.ValidationError, match="keeps"):
        pydantic_model.Network(node_selection={"keeps": ["A"]})
