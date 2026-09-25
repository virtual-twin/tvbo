"""``network.nodes.<attr>`` — a named per-node array, the node-side twin of ``network.edges.<label>``.

A network can carry measured per-region vectors: a region size, a per-node current range fitted elsewhere. Before this, only two hardcoded measures (``positions``, ``instrength``) were referenceable, so a study with such a vector had no way to reach it from a spec and pasted the array into code instead — which is how a 379-element table ends up living in a Python file.

Resolution order is what the tests pin: node ``parameters`` win over the companion file (a spec can override what the file carries), an explicit ``nodes.<attr>`` is never shadowed by an edge label of the same name, and an unknown attribute raises instead of resolving to something else.
"""

import numpy as np
import pytest

from tvbo.data import param_io  # noqa: E402
from tvbo.datamodel import schema as tvbo_datamodel  # noqa: E402


class _FakeStore:
    """Minimal companion store: the ``read_dataset`` contract Network exposes as ``_store``."""

    def __init__(self, datasets):
        self._datasets = datasets

    def read_dataset(self, path):
        return self._datasets[path]  # KeyError for an absent dataset, as h5py does


class _FakeNet:
    """A network reduced to what the node resolver reads: nodes, a store, and a matrix."""

    def __init__(self, node_params=None, datasets=None, weight=None):
        self.nodes = [
            tvbo_datamodel.Node(id=i, label=f"n{i}", parameters=(node_params[i] if node_params else None)) for i in range(3)
        ]
        self._store = _FakeStore(datasets) if datasets is not None else None
        self._weight = weight

    def matrix(self, label, format=None):
        return self._weight if label == "weight" else None


def _params(value):
    return {"roi_size": tvbo_datamodel.Parameter(name="roi_size", value=value)}


def test_attribute_reads_from_the_companion_store():
    net = _FakeNet(datasets={"nodes/roi_size": [1.5, 2.5, 3.5]})
    assert np.array_equal(param_io.resolve_network_node(net, "roi_size"), [1.5, 2.5, 3.5])


def test_node_parameters_win_over_the_companion_store():
    """So a spec can override a value the network file carries, rather than being stuck with it."""
    net = _FakeNet(node_params=[_params(10.0), _params(20.0), _params(30.0)], datasets={"nodes/roi_size": [1.0, 1.0, 1.0]})
    assert np.array_equal(param_io.resolve_network_node(net, "roi_size"), [10.0, 20.0, 30.0])


def test_a_partially_set_parameter_is_not_a_vector():
    """Set on some nodes only, it falls through to the file rather than being zero-filled."""
    net = _FakeNet(node_params=[_params(10.0), None, _params(30.0)], datasets={"nodes/roi_size": [1.0, 2.0, 3.0]})
    assert np.array_equal(param_io.resolve_network_node(net, "roi_size"), [1.0, 2.0, 3.0])


def test_unknown_attribute_resolves_to_nothing():
    assert param_io.resolve_network_node(_FakeNet(datasets={}), "not_a_thing") is None


def test_explicit_nodes_form_is_not_shadowed_by_an_edge_label():
    """``nodes.weight`` must give the node vector even though ``weight`` is an edge-matrix alias."""
    net = _FakeNet(datasets={"nodes/weight": [7.0, 8.0, 9.0]}, weight=np.ones((3, 3)))
    got = param_io._resolve_ref("network.nodes.weight", context=net, where="test")
    assert np.array_equal(got, [7.0, 8.0, 9.0])
    edges = param_io._resolve_ref("network.edges.weight", context=net, where="test")
    assert edges.shape == (3, 3)


def test_unknown_nodes_attribute_raises_naming_both_sources():
    net = _FakeNet(datasets={})
    with pytest.raises(ValueError, match="nodes/missing"):
        param_io._resolve_ref("network.nodes.missing", context=net, where="test")
