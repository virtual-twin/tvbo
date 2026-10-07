"""A layer a network declares beats its source's layer of that name, and the source fills the rest.

A network names one source of connectivity (a ``data_file`` companion, a ``bids_dir``, a graph generator, a ``parcellation``) and may declare single layers itself, computed by a ``producer:`` or read through ``used:``. One rule holds for every source. Template edges are declarations and hold no connectivity, so the source still loads. A declared layer replaces the source's layer of that name under any spelling (``weights`` is ``weight``), in every lookup and in what a run materialises. The nodes and every other layer are the source's.

The ``producer:`` side is pinned here against all four sources; the ``used:`` side sits beside the cohort fixtures in ``test_cohort_references.py``.
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
