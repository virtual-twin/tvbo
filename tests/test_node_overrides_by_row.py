"""Per-node values declared on a network's nodes land on each node's row, its position in ``network.nodes``.

A node's ``id`` is an identifier, not an index (`Network.node_index_map`): a network may number its nodes 10 and 20, and its edges then address them by those ids while the arrays are indexed by position. Indexing by id raised `IndexError` for initial states and silently dropped per-node parameters.
"""

from tvbo.classes.network import Network
from tvbo.templates.tvboptim.utils import get_node_param_overrides, get_node_state_overrides


def _network():
    return Network(
        nodes=[
            {"id": 10, "label": "a", "parameters": {"I": {"value": -0.1}}, "state": {"x": {"value": 0.8}}},
            {"id": 20, "label": "b", "parameters": {"I": {"value": -0.2}}},
        ],
        edges=[{"source": 10, "target": 20, "weight": 1.0}],
    )


def test_an_initial_state_lands_on_the_row_of_its_node():
    assert get_node_state_overrides(_network(), 2, ["x"], [0.0]) == {"x": [0.8, 0.0]}


def test_a_parameter_lands_on_the_row_of_its_node():
    assert get_node_param_overrides(_network(), 2, {"I": 0.0}) == {"I": [-0.1, -0.2]}
