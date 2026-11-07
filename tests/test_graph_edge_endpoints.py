"""``edge.source`` is a ``Node.id``, and the graph builders must read it as one.

``Network.node_index_map`` is the authority: ``Node.id`` is a ``dcterms:identifier``, not a position, so a network may declare ``[{id: 1}, {id: 2}]`` and its edges then address nodes by those ids while the matrices stay indexed by declaration order. ``_edge_matrix`` resolves endpoints that way; ``graph`` and ``create_graph`` key their nodes by ``node.id`` and so must use the endpoint directly. Treating it as a position instead is invisible whenever the ids happen to equal the positions, which is every network in the database — hence these tests declare ids that do not.
"""

import numpy as np
import pytest

from tvbo.classes.network import Network


def _two_node_network(directed):
    """Ids 1 and 2 — deliberately not positions 0 and 1 — with one edge between them."""
    return Network(
        label="endpoint test",
        number_of_nodes=2,
        nodes=[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}],
        edges=[{"source": 1, "target": 2, "directed": directed, "parameters": {"weight": {"value": 0.75}}}],
    )


@pytest.mark.parametrize("directed", [True, False])
def test_the_graph_has_no_node_the_network_did_not_declare(directed):
    """Reading an id as a position invents index-keyed nodes beside the id-keyed ones."""
    G = _two_node_network(directed).graph
    assert set(G.nodes) == {1, 2}


@pytest.mark.parametrize("directed", [True, False])
def test_the_edge_connects_the_nodes_it_names(directed):
    G = _two_node_network(directed).graph
    assert G.has_edge(1, 2)
    assert G.has_edge(2, 1) is (not directed)


def test_create_graph_agrees_with_the_graph_property():
    """Two builders over one spec: they must describe the same connectome."""
    net = _two_node_network(directed=True)
    assert set(net.create_graph().nodes) == set(net.graph.nodes)
    assert set(net.create_graph().edges()) == set(net.graph.edges())


def test_the_weights_matrix_places_the_edge_where_the_graph_does():
    """``_edge_matrix`` is target-by-source, so a 1 -> 2 edge lands at ``[1, 0]``."""
    W = _two_node_network(directed=True).weights_matrix
    assert W.shape == (2, 2)
    assert W[1, 0] == pytest.approx(0.75)
    assert np.count_nonzero(W) == 1
