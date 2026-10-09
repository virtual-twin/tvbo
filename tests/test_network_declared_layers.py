"""A layer a network declares beats its source's layer of that name, and the source fills the rest.

A network names one source of connectivity (a ``data_file`` companion, a curated network by ``iri``, a ``bids_dir``, a graph generator, a ``parcellation``) and may declare single layers itself, computed by a ``producer:`` or read through ``used:``. One rule holds for every source. Template edges are declarations and hold no connectivity, so the source still loads. A declared layer replaces the source's layer of that name under any spelling (``weights`` is ``weight``), in every lookup and in what a run materialises. The nodes and every other layer are the source's.

What a network lists beside its source is read by the kind of edge. A layer entry that only names its layer keeps every slot it states and takes the rest, with the matrix, from the source. A route to or from another network is kept as written and never stands in for the source. An edge between two of the network's own nodes is a listed graph, which a source cannot sit beside: the network is refused, naming both.

The ``producer:`` side is pinned here against the sources; the ``used:`` side sits beside the cohort fixtures in ``test_cohort_references.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tvbo.classes.network import Network
from tvbo.data import param_io

LABELS = ["L.A1", "R.A1", "L.V1", "R.V1"]
SOURCE_WEIGHTS = np.ones((4, 4)) - np.eye(4)
SOURCE_LENGTHS = np.full((4, 4), 10.0)

_MODULE = '''
import numpy as np

def ring(n, scale):
    """Each node linked to the next, a matrix no source here holds."""
    return np.roll(float(scale) * np.eye(int(n)), 1, axis=1)

def motif():
    return {"weights": np.ones((4, 4)) - np.eye(4), "lengths": np.full((4, 4), 10.0), "node_labels": ["L.A1", "R.A1", "L.V1", "R.V1"]}

def motif_network():
    from tvbo.classes.network import Network

    return Network.from_matrix(weights=np.ones((4, 4)) - np.eye(4), lengths=np.full((4, 4), 10.0), labels=["L.A1", "R.A1", "L.V1", "R.V1"])
'''


@pytest.fixture
def study_dir(tmp_path, monkeypatch):
    """A directory holding the producer and builder module, importable as a study's own code is."""
    (tmp_path / "layer_sources.py").write_text(_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("layer_sources", None)
    param_io.clear_cache()
    yield tmp_path
    sys.modules.pop("layer_sources", None)
    param_io.clear_cache()


def _companion(where):
    Network.from_matrix(weights=SOURCE_WEIGHTS, lengths=SOURCE_LENGTHS, labels=LABELS).save(where / "group.yaml")
    return {"data_file": str(where / "group.h5")}


def _bids_directory(where):
    rows = ["matrix_index\tnode_file\tnode_index\tlabel"] + [f"{i}\tatlas-Probe\t{i}\t{lab}" for i, lab in enumerate(LABELS)]
    (where / "atlas-Probe_nodeindices.tsv").write_text("\n".join(rows) + "\n")
    for measure, matrix in (("streamlineCount", SOURCE_WEIGHTS), ("tractLength", SOURCE_LENGTHS)):
        np.savetxt(where / f"atlas-Probe_meas-{measure}_relmat.dense.tsv", matrix, delimiter="\t")
    return {"bids_dir": str(where), "structural_measures": ["streamlineCount", "tractLength"]}


def _generator(where):
    builder = {"name": "motif", "module": "layer_sources"}
    return {"number_of_nodes": 4, "graph_generator": {"name": "Motif", "type": "Motif", "builder": builder}}


def _parcellation(where):
    return {"parcellation": {"atlas": {"name": "DesikanKilliany"}}}


def _generated_network(where):
    builder = {"name": "motif_network", "module": "layer_sources"}
    return {"number_of_nodes": 4, "graph_generator": {"name": "Motif", "type": "Motif", "builder": builder}}


def _curated(where):
    return {"iri": CURATED}


def _matrix_file(where):
    np.savetxt(where / "weights.csv", SOURCE_WEIGHTS, delimiter=",")
    return {"edge_matrix_files": [str(where / "weights.csv")]}


CURATED = "tvbo:tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat"
EVERY_SOURCE = [_companion, _curated, _bids_directory, _generator, _generated_network, _parcellation]
LISTED = {"source": 0, "target": 1, "parameters": {"weight": {"value": 0.5}}}


def _produced(label, n):
    call = {"callable": {"name": "ring", "module": "layer_sources"}, "arguments": {"n": {"value": n}, "scale": {"value": 2.0}}}
    return {"label": label, "producer": call}


@pytest.mark.parametrize("source", [_companion, _bids_directory, _generator, _parcellation])
def test_a_produced_layer_replaces_the_sources_under_any_spelling_and_the_source_fills_the_rest(study_dir, source):
    """Declared as ``weights``, the produced matrix is what ``matrix("weight")`` returns and what a run materialises, whatever the source holds as ``weight``."""
    spec = source(study_dir)
    plain = Network(**spec)
    n = plain.number_of_nodes
    ring = np.roll(2.0 * np.eye(n), 1, axis=1)
    assert not np.array_equal(plain.matrix("weight", format="dense"), ring)

    net = Network(**spec, edges=[_produced("weights", n)])
    np.testing.assert_array_equal(net.matrix("weight", format="dense"), ring)
    np.testing.assert_array_equal(net.materialize("weight").arrays["edges/weight"], ring)
    np.testing.assert_array_equal(net.matrix("length", format="dense"), plain.matrix("length", format="dense"))
    assert net.node_labels == plain.node_labels


@pytest.mark.parametrize("source", [_bids_directory, _parcellation])
def test_a_layer_that_is_only_named_does_not_stand_in_for_the_source(study_dir, source):
    """A template edge stating no origin is a name for a layer, so the source's connectome still loads beneath it and the edge stays declared."""
    spec = source(study_dir)
    plain, net = Network(**spec), Network(**spec, edges=[{"label": "weight"}])
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(net.matrix(layer, format="dense"), plain.matrix(layer, format="dense"))
    assert net.node_labels == plain.node_labels
    assert [str(edge.label) for edge in net.edges] == ["weight"]


def _layer(net, name):
    return next(edge for edge in net.edges if str(edge.label) == name)


@pytest.mark.parametrize("source", EVERY_SOURCE)
def test_a_named_layer_keeps_what_it_declares_and_the_source_supplies_the_matrix(study_dir, source):
    """A layer entry stating a unit and a description beside a source keeps both, under the spelling it chose, on the matrix the source holds."""
    spec = source(study_dir)
    plain = Network(**spec)
    net = Network(**spec, edges=[{"label": "weights", "unit": "count", "description": "streamline counts"}])
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(net.matrix(layer, format="dense"), plain.matrix(layer, format="dense"))
    assert net.node_labels == plain.node_labels
    mine = _layer(net, "weights")
    assert (str(mine.unit), str(mine.description)) == ("count", "streamline counts")
    assert [str(edge.label) for edge in net.edges if str(edge.label) in ("weight", "weights")] == ["weights"]
    np.testing.assert_array_equal(
        net.materialize("weight").arrays["edges/weight"], plain.materialize("weight").arrays["edges/weight"]
    )


def test_a_named_layer_takes_from_the_source_every_slot_it_does_not_state(study_dir):
    """The companion's layer is dense and unsigned; an entry that states only ``non_negative: false`` is dense still and signed, and the companion's other layer is untouched."""
    spec = _companion(study_dir)
    net = Network(**spec, edges=[{"label": "weight", "non_negative": False}])
    mine, theirs = _layer(net, "weight"), _layer(Network(**spec), "weight")
    assert str(mine.format) == str(theirs.format) == "dense"
    assert (bool(mine.non_negative), bool(theirs.non_negative)) == (False, True)
    assert [str(edge.label) for edge in net.edges] == ["weight", "length"]


def test_a_named_layer_that_restates_a_default_overrides_the_source(study_dir):
    """What an entry states wins even where it states the slot's default: ``directed: false`` beside a companion whose layer is directed."""
    spec = _companion(study_dir)
    sidecar = study_dir / "group.yaml"
    sidecar.write_text(sidecar.read_text().replace("directed: false", "directed: true", 1))
    assert bool(_layer(Network(**spec), "weight").directed)
    assert not bool(_layer(Network(**spec, edges=[{"label": "weight", "directed": False}]), "weight").directed)


@pytest.mark.parametrize("source", EVERY_SOURCE)
@pytest.mark.parametrize("ends", [{}, {"source": 0, "target": 2}], ids=["whole", "between-two-nodes"])
def test_a_route_stays_beside_a_source_and_never_stands_in_for_it(study_dir, source, ends):
    """An edge to another network is no connectivity of this one: the source loads beneath it and the route is kept as written, with or without endpoints."""
    spec = source(study_dir)
    plain = Network(**spec)
    route = {"label": "drive", "target_network": "..", "source_var": "phi", **ends}
    net = Network(**spec, edges=[route])
    for layer in ("weight", "length"):
        np.testing.assert_array_equal(net.matrix(layer, format="dense"), plain.matrix(layer, format="dense"))
    assert net.node_labels == plain.node_labels
    kept = _layer(net, "drive")
    assert (str(kept.target_network), str(kept.source_var), kept.source, kept.target) == (
        "..",
        "phi",
        ends.get("source"),
        ends.get("target"),
    )


@pytest.mark.parametrize("source", [_companion, _curated, _bids_directory, _generator, _generated_network, _matrix_file])
def test_a_listed_edge_beside_a_source_is_refused_naming_both(study_dir, source):
    """A graph is read or listed. One edge between two of the network's own nodes beside a source raises, naming the source and the edge."""
    spec = source(study_dir)
    named = next(key for key in ("data_file", "iri", "bids_dir", "graph_generator", "edge_matrix_files") if key in spec)
    with pytest.raises(ValueError, match=rf"(?s){named}.*0 -> 1.*read or listed"):
        Network(**spec, edges=[LISTED])


def test_a_matrix_file_is_read_beside_a_layer_entry_and_a_route(study_dir):
    """Layer entries and routes hold no connectivity, so an ``edge_matrix_files`` matrix still loads beside them."""
    spec = _matrix_file(study_dir)
    net = Network(**spec, edges=[{"label": "weight", "unit": "count"}, {"label": "drive", "target_network": ".."}])
    np.testing.assert_array_equal(net.matrix("weight", format="dense"), SOURCE_WEIGHTS)
    assert str(_layer(net, "weight").unit) == "count"


def test_a_produced_weight_layer_replaces_the_matrix_files(study_dir):
    """The file holds the one ``weight`` matrix, so a ``weight`` layer that states its own origin is the network's and the file gives only the node count."""
    net = Network(**_matrix_file(study_dir), edges=[_produced("weights", 4)])
    np.testing.assert_array_equal(net.matrix("weight", format="dense"), np.roll(2.0 * np.eye(4), 1, axis=1))
    assert net.number_of_nodes == 4 and [str(edge.label) for edge in net.edges] == ["weights"]


def test_a_curated_network_named_by_iri_loads_beside_declared_edges_and_nodes(study_dir):
    """``iri`` is a source like the others: a produced layer replaces the curated one, a node declared by label lands on the curated node, and the rest is the curated network's."""
    plain = Network(iri=CURATED)
    n, region = plain.number_of_nodes, plain.node_labels[2]
    assert n > 1
    ring = np.roll(2.0 * np.eye(n), 1, axis=1)
    net = Network(iri=CURATED, edges=[_produced("weights", n)], nodes={region: {"parameters": {"gain": {"value": 3.0}}}})
    np.testing.assert_array_equal(net.matrix("weight", format="dense"), ring)
    np.testing.assert_array_equal(net.matrix("length", format="dense"), plain.matrix("length", format="dense"))
    assert net.node_labels == plain.node_labels
    assert float(net.nodes[2].parameters["gain"].value) == 3.0


def test_a_parcellation_named_as_text_beside_a_listed_graph_names_its_atlas():
    """A listed graph is the network; a ``parcellation`` beside it says which atlas its nodes are from, written as text or as a record, and loads nothing."""
    nodes = [{"id": 0, "label": "a"}, {"id": 1, "label": "b"}]
    net = Network(parcellation="DesikanKilliany", nodes=nodes, edges=[LISTED])
    assert net.number_of_nodes == 2 and net.node_labels == ["a", "b"]
    assert str(net.parcellation.atlas.name) == "DesikanKilliany"
    assert float(net.matrix("weight", format="dense")[1, 0]) == 0.5


def test_a_node_template_applies_through_the_datamodel_route_too():
    """A template's node parameters reach every node whether the network is built from a mapping or from its datamodel record."""
    from tvbo.datamodel import schema

    spec = {"number_of_nodes": 2, "node_template": {"id": 0, "parameters": {"g": {"value": 0.5}}}}
    for net in (Network(**spec), Network.from_datamodel(schema.Network(**spec))):
        nodes = net.nodes.values() if isinstance(net.nodes, dict) else net.nodes
        assert [sorted((getattr(n, "parameters", None) or {}).keys()) for n in nodes] == [["g"], ["g"]]
