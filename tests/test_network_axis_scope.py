"""`network.`-scoped exploration axes resolve to the graph leaf they sweep.

The scope is declarative: an axis names a network attribute (`network.edges.weight`,
`network.conduction_speed`) and codegen resolves it to the live backend-graph leaf, so every cell sees its own value without a Network or graph rebuild.

The canonical form is `network.edges.<label>`; `network.weight(s)`/`network.length(s)` are shortcuts. Both go through `edge_label`, the same resolver observations use, so an
axis and an observation can never disagree about which matrix a reference means.
"""

import pytest

from tvbo.templates.tvboptim.utils import network_axis_leaf


@pytest.mark.parametrize(
    "ref, leaf",
    [
        # Canonical `network.edges.<label>` form.
        ("network.edges.weight", "weights"),
        ("network.edges.length", "lengths"),
        # Shortcuts resolve to the same leaf as the canonical form — singular and plural alike, so a recipe cannot half-work on spelling.
        ("network.weight", "weights"),
        ("network.weights", "weights"),
        ("network.length", "lengths"),
        ("network.lengths", "lengths"),
        # The network's own scalars.
        ("network.conduction_speed", "speed"),
    ],
)
def test_network_axis_resolves_to_graph_leaf(ref, leaf):
    assert network_axis_leaf(ref) == leaf


@pytest.mark.parametrize(
    "ref",
    [
        "MurrayWangDM.mu",  # dynamics-scoped
        "EIBLinearCoupling.wLRE",  # coupling-scoped
        "execution.random_seed",  # execution-scoped
        "conduction_speed",  # bare name is not a scoped reference
        "",
        None,
        3.0,
    ],
)
def test_non_network_refs_return_none(ref):
    """Out-of-scope references route through the dynamics/coupling path untouched."""
    assert network_axis_leaf(ref) is None


def test_edges_weight_is_not_misparsed_as_dynamics():
    """`network.edges.weight` must not rsplit into prefix 'network.edges'.

    Splitting a scoped path on the LAST dot leaves prefix='network.edges', which matches neither the network scope nor a coupling key, so the axis silently
    falls through and is emitted as a dynamics parameter — a wrong-scope write that sweeps nothing and yields identical cells rather than an error.
    """
    assert network_axis_leaf("network.edges.weight") == "weights"


def test_unsweepable_edge_attribute_raises():
    """An edge attribute with no graph leaf fails at codegen, not silently.

    A matrix like `fc` is legitimately referenceable by an observation but has no live leaf to sweep, so naming it as an axis is a recipe error.
    """
    with pytest.raises(ValueError, match="no graph leaf to sweep"):
        network_axis_leaf("network.edges.fc")


def test_unknown_network_attribute_raises():
    with pytest.raises(ValueError, match="unknown network attribute"):
        network_axis_leaf("network.bogus")


def test_error_names_the_sweepable_set():
    """The error tells the author what they may sweep instead."""
    with pytest.raises(ValueError) as e:
        network_axis_leaf("network.bogus")
    msg = str(e.value)
    assert "network.conduction_speed" in msg
    assert "network.edges.weight" in msg
