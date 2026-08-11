"""An `Edge` may declare a `producer:`, so a matrix is computed rather than pre-built.

`producer:` already said HOW a derived value is made, but only a `Parameter` carried it — so a network whose weight matrix is a deterministic function of the spec's own inputs (a
mesh operator, a rule-generated connectome) had no way to say so. Anything too large to inline had to be written out by a prep script and referenced by path, which means the spec
could not execute until someone ran that script, and the file could drift from the recipe that described it without anything noticing.

The trap this pins is the one that actually bit: the accessors do NOT agree on one entry point. `matrix()` walks a resolution order; `weights` reads `_arrays` directly. Resolving
the producer in only one of them leaves the other silently falling through to whatever it does when a connectome is missing — a full run that completes and is wrong.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tvbo.classes.network import Network
from tvbo.data import param_io

_MODULE = '''
import numpy as np

CALLS = []

def ring(n, scale=1.0):
    """A tiny deterministic connectome: each node linked to the next."""
    CALLS.append((n, scale))
    W = np.zeros((int(n), int(n)))
    for i in range(int(n)):
        W[i, (i + 1) % int(n)] = float(scale)
    return W

def pair():
    return {"weight": np.eye(3), "length": np.full((3, 3), 2.0)}

def nothing():
    return None
'''


@pytest.fixture
def producer_module(tmp_path, monkeypatch):
    (tmp_path / "edge_producers.py").write_text(_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("edge_producers", None)
    param_io.clear_cache()
    yield tmp_path
    sys.modules.pop("edge_producers", None)
    param_io.clear_cache()


def _network(producer, label="weight", n=4):
    from tvbo.datamodel.schema import Edge

    return Network(number_of_nodes=n, nodes=[], edges=[Edge(label=label, weighted=True, producer=producer)])


def _call(name, **arguments):
    from tvbo.datamodel.schema import Argument, FunctionCall

    return FunctionCall(
        callable={"name": name, "module": "edge_producers"},
        arguments={k: Argument(name=k, value=v) for k, v in arguments.items()},
    )


def test_a_produced_matrix_is_returned_by_matrix(producer_module):
    net = _network(_call("ring", n=4, scale=2.0))
    got = np.asarray(net.matrix("weight"))
    assert got.shape == (4, 4) and got[0, 1] == 2.0 and got[1, 0] == 0.0


def test_the_weights_accessor_sees_it_too(producer_module):
    """The bug this exists for: two accessors, one of them silently missing the producer."""
    net = _network(_call("ring", n=4, scale=2.0))
    np.testing.assert_array_equal(np.asarray(net.weights), np.asarray(net.matrix("weight")))


def test_it_is_built_once_however_many_accessors_ask(producer_module):
    import edge_producers

    net = _network(_call("ring", n=4))
    net.matrix("weight")
    net.weights
    net.matrix("weight")
    assert len(edge_producers.CALLS) == 1


def test_a_stored_matrix_wins_over_the_producer(producer_module):
    """A produced matrix is the LAST resort, so it cannot shadow one that was set."""
    net = _network(_call("ring", n=4, scale=2.0))
    net.set_matrix("weight", np.ones((4, 4)))
    np.testing.assert_array_equal(np.asarray(net.matrix("weight")), np.ones((4, 4)))


def test_output_picks_one_array_out_of_a_producer_returning_several(producer_module):
    from tvbo.datamodel.schema import Edge

    call = _call("pair")
    call.output = "length"
    net = Network(number_of_nodes=3, nodes=[], edges=[Edge(label="length", weighted=True, producer=call)])
    np.testing.assert_array_equal(np.asarray(net.matrix("length")), np.full((3, 3), 2.0))


def test_the_edge_label_names_the_matrix_it_produces(producer_module):
    """A producer on `length` must not answer a request for `weight`."""
    net = _network(_call("ring", n=4), label="length")
    assert net.matrix("weight") is None
    assert net.matrix("length") is not None


def test_a_producer_returning_nothing_is_an_error_not_an_empty_network(producer_module):
    net = _network(_call("nothing"))
    with pytest.raises(ValueError, match="returned nothing"):
        net.matrix("weight")


def test_a_failed_producer_is_retried_rather_than_latched(producer_module):
    """A caught failure must not leave the network looking permanently connectome-less."""
    net = _network(_call("nothing"))
    with pytest.raises(ValueError):
        net.matrix("weight")
    with pytest.raises(ValueError, match="returned nothing"):
        net.weights


def test_an_edge_without_a_producer_costs_nothing(producer_module):
    from tvbo.datamodel.schema import Edge

    net = Network(number_of_nodes=2, nodes=[], edges=[Edge(label="weight", weighted=True)])
    assert net.matrix("weight") is None


def test_producer_and_a_literal_are_mutually_exclusive(producer_module):
    """The three provenance claims disagree by construction; saying two is a spec error."""
    from tvbo.datamodel.schema import Edge

    edge = Edge(label="weight", weighted=True, producer=_call("ring", n=4))
    edge.value = np.zeros((4, 4))
    with pytest.raises(ValueError, match="mutually exclusive"):
        param_io.resolve(edge)


def test_the_error_names_the_edge_by_its_label(producer_module):
    """An Edge is identified by `label`, not `name`; '<unnamed>' would be useless."""
    from tvbo.datamodel.schema import Edge

    edge = Edge(label="local_connectivity", weighted=True, producer=_call("ring", n=4))
    edge.value = np.zeros((4, 4))
    with pytest.raises(ValueError, match="local_connectivity"):
        param_io.resolve(edge)
