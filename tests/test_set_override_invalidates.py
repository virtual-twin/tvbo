"""A ``--set`` override that a materialised object has already consumed must take effect.

Resolution latches: a Network builds its connectivity once and caches it, which is what stops it rebuilding on every access. ``--set`` mutates the loaded object AFTER that, so an override of anything the network resolved from — a graph generator's parameters, a parcellation — was reported on the console and then ignored. The run completed, the FC came out bit-identical to the un-overridden one, and nothing anywhere said so.

That is the worst shape a defect can take: a flag that says it did something, a result that looks right, and no way to tell from the output that the run was not the one asked for. So the contract pinned here is that an override invalidates what it feeds.
"""

from __future__ import annotations

import numpy as np

from tvbo.classes.network import Network
from tvbo.cli.run import _apply_metadata_overrides


class _Experiment:
    """The minimum the override walker traverses."""

    def __init__(self, network):
        self.network = network
        self.name = "exp"


def _network(**kw):
    return Network(number_of_nodes=3, nodes=[], **kw)


def test_a_resolved_network_is_invalidated_by_an_override_beneath_it():
    net = _network()
    net._resolved = True
    net.set_array("edges/weight", np.ones((3, 3)))
    _apply_metadata_overrides(_Experiment(net), ["network.number_of_nodes=4"])
    assert net._resolved is False and "edges/weight" not in net.arrays


def test_the_override_value_itself_still_lands():
    net = _network()
    net._resolved = True
    _apply_metadata_overrides(_Experiment(net), ["network.number_of_nodes=4"])
    assert net.number_of_nodes == 4


def test_produced_matrices_are_dropped_too():
    """`_producers_resolved` latches separately and would survive a bare `_resolved` reset."""
    net = _network()
    net._resolved = True
    object.__setattr__(net, "_producers_resolved", True)
    object.__setattr__(net, "_arrays", {"weight": np.eye(3)})
    _apply_metadata_overrides(_Experiment(net), ["network.number_of_nodes=4"])
    assert net._get_arrays() == {}


def test_an_override_nowhere_near_a_network_invalidates_nothing():
    class _Exec:
        def __init__(self):
            self.random_seed = 0

    class _Exp:
        def __init__(self, net):
            self.network, self.execution = net, _Exec()

    net = _network()
    net._resolved = True
    exp = _Exp(net)
    _apply_metadata_overrides(exp, ["execution.random_seed=7"])
    assert exp.execution.random_seed == 7
    assert net._resolved is True


def test_the_innermost_materialised_object_wins():
    """Two networks side by side: overriding one must not rebuild the other."""

    class _Exp:
        def __init__(self, a, b):
            self.a, self.b = a, b

    a, b = _network(), _network()
    a._resolved = b._resolved = True
    _apply_metadata_overrides(_Exp(a, b), ["a.number_of_nodes=4"])
    assert a._resolved is False and b._resolved is True


def test_invalidate_resolution_is_idempotent():
    net = _network()
    net.invalidate_resolution()
    net.invalidate_resolution()
    assert net._resolved is False and net._get_arrays() == {}


def test_a_graph_generator_parameter_is_reachable_and_invalidates():
    """The path that motivated this: swapping the connectome a builder reads."""
    from tvbo.datamodel.schema import GraphGenerator

    net = _network()
    net.graph_generator = GraphGenerator(  # attached after construction: no builder runs
        name="G",
        type="G",
        builder={"name": "build", "module": "mod"},
        parameters={"connectome": {"name": "connectome", "value": "a.mat"}},
    )
    net._resolved = True
    _apply_metadata_overrides(_Experiment(net), ["network.graph_generator.parameters.connectome.value=b.mat"])
    assert net.graph_generator.parameters["connectome"].value == "b.mat"
    assert net._resolved is False
