"""An inline edge coupling loads at every depth of a network: on its own edges and inside the subnetworks of its node template and of its listed nodes.

``Edge.coupling`` is a reference by name in the datamodel, and the datamodel builds a node template's or a listed node's subnetwork, edges included, before the tvbo ``Network`` of that level sees it, so a coupling defined inline there would land in the name slot. The network takes every inline coupling off before that build and puts it back as a ``Coupling`` on the edge it came from, so each resolved subnetwork edge carries its coupling function and parameters.
"""

import copy

from tvbo.classes.network import Network
from tvbo.datamodel.schema import Coupling

PROJECTION = {"name": "projection", "coupling_function": {"rhs": "w * x"}, "parameters": {"w": {"value": 0.5}}}


def _subnetwork():
    return {"number_of_nodes": 3, "edges": [{"source": 0, "target": 1, "coupling": copy.deepcopy(PROJECTION)}]}


def _assert_projection(edge):
    assert type(edge.coupling) is Coupling
    assert float(edge.coupling.parameters["w"].value) == 0.5
    assert str(edge.coupling.coupling_function.rhs) == "w * x"


def test_a_template_subnetworks_coupling_reaches_every_node():
    net = Network(number_of_nodes=2, node_template={"subnetwork": _subnetwork()})

    _assert_projection(net.node_template.subnetwork.edges[0])
    for node in net.nodes:
        assert isinstance(node.subnetwork, Network)
        _assert_projection(node.subnetwork.edges[0])


def test_a_listed_nodes_subnetwork_keeps_its_coupling():
    net = Network(nodes=[{"id": 0, "label": "a", "subnetwork": _subnetwork()}])

    (node,) = net.nodes
    assert isinstance(node.subnetwork, Network)
    _assert_projection(node.subnetwork.edges[0])


def test_a_coupling_two_templates_deep_loads():
    net = Network(
        number_of_nodes=2, node_template={"subnetwork": {"number_of_nodes": 2, "node_template": {"subnetwork": _subnetwork()}}}
    )

    for node in net.nodes:
        for unit in node.subnetwork.nodes:
            _assert_projection(unit.subnetwork.edges[0])


def test_a_top_level_coupling_is_unchanged():
    net = Network(number_of_nodes=3, edges=[{"source": 0, "target": 1, "coupling": copy.deepcopy(PROJECTION)}])

    _assert_projection(net.edges[0])


def test_the_callers_spec_is_left_as_written():
    spec = {
        "number_of_nodes": 2,
        "node_template": {"subnetwork": _subnetwork()},
        "edges": [{"source": 0, "target": 1, "coupling": copy.deepcopy(PROJECTION)}],
    }
    written = copy.deepcopy(spec)

    Network(**spec)

    assert spec == written
