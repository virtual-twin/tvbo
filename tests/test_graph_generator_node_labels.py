"""A graph-generator builder may NAME the nodes it generates.

A motif whose nodes ARE particular regions — a two-node PPC/PFC decision circuit, a
cortex/thalamus pair — says which is which through the builder's own labels rather than taking
positional ones (``node_0``, ``node_1``). Everything keyed downstream inherits them: an
observation's node coordinate, a figure's ``sel: {node: PFC}``, a report's per-node table.
Unlabelled, the only way to reach one module is by index, which is exactly the positional
binding the container format exists to remove.

These pin the contract: a builder returning a dict may supply ``node_labels``, one per node;
the labels reach ``Network.node_labels``; a wrong-length list is an error rather than a
silent truncation or pad; and a builder that supplies none still gets the positional
default.
"""

from pathlib import Path

import pytest
import yaml

from tvbo.classes.network import Network

_BUILDER_MODULE = '''
import numpy as np

W = np.array([[0.0, 1.0], [1.0, 0.0]])
L = np.zeros((2, 2))


def named():
    return {"weights": W, "lengths": L, "node_labels": ["PPC", "PFC"]}


def unnamed():
    return {"weights": W, "lengths": L}


def too_few_labels():
    return {"weights": W, "lengths": L, "node_labels": ["PPC"]}


def named_as_array():
    return {"weights": W, "lengths": L, "node_labels": np.array(["PPC", "PFC"])}


def named_with_params():
    return {"weights": W, "lengths": L, "node_labels": ["PPC", "PFC"],
            "node_params": {"I_e": [0.0118, 0.0]}}
'''


@pytest.fixture
def two_node_network(tmp_path):
    """A Network spec whose builder lives beside it, as a study's does."""
    (tmp_path / "motif_builders.py").write_text(_BUILDER_MODULE)

    def _load(func_name):
        spec = {
            "tvbo_class": "tvbo:Network",
            "label": "two-node motif",
            "number_of_nodes": 2,
            "graph_generator": {
                "name": "Motif",
                "type": "Motif",
                "builder": {"name": func_name, "module": "motif_builders"},
            },
        }
        path = tmp_path / f"{func_name}.yaml"
        path.write_text(yaml.safe_dump(spec))
        return Network.from_file(str(path))

    return _load


def test_a_builder_can_name_its_nodes(two_node_network):
    net = two_node_network("named")
    assert net.node_labels == ["PPC", "PFC"]
    assert [n.label for n in net.nodes] == ["PPC", "PFC"]


def test_labels_and_per_node_parameters_land_on_the_same_nodes(two_node_network):
    """Naming must not disturb the per-node parameter materialisation it sits beside."""
    net = two_node_network("named_with_params")
    assert net.node_labels == ["PPC", "PFC"]
    assert [float(n.parameters["I_e"].value) for n in net.nodes] == [0.0118, 0.0]


def test_a_builder_that_names_nothing_keeps_the_positional_default(two_node_network):
    net = two_node_network("unnamed")
    assert net.node_labels == ["node_0", "node_1"]


def test_a_wrong_length_label_list_raises(two_node_network):
    """Silently padding or truncating would mis-key every node after the short point."""
    with pytest.raises(ValueError, match="node_labels"):
        two_node_network("too_few_labels")


def test_labels_may_arrive_as_an_array(two_node_network):
    """A builder reading its labels off an atlas hands back a numpy array, not a list.

    Testing an array for emptiness by truthiness asks it for a scalar truth value, which
    raises for anything longer than one element — so the whole network fails to materialise
    on the most natural way to produce the labels.
    """
    net = two_node_network("named_as_array")
    assert net.node_labels == ["PPC", "PFC"]
