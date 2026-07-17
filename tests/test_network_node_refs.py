"""Node-vector network references (``network.positions`` / ``network.instrength``).

The node-level analogue of the connectome-matrix refs: an observation source, a
pipeline-callable argument, or an observer (``dynamics``) parameter may reference a
per-node vector derived from the network, which is embedded once as a module
constant. The subtle correctness point is that ``parse_reference`` splits
``network.positions`` into ``('network', 'positions')`` — so ``ref_to_code`` resolves
the BARE key ``'positions'``, while ``collect_network_node_arrays`` scans the FULL
``'network.positions'`` string. ``node_label`` must accept both forms or the emitted
constant name and the resolved reference silently disagree (the callable then gets a
``kwargs.get('positions')`` -> ``None`` instead of the embedded vector).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tvbo.templates.tvboptim.utils import (
    node_label, node_const, collect_network_node_arrays,
)


@pytest.mark.parametrize("ref", ["network.positions", "positions",
                                 "network.instrength", "instrength"])
def test_node_label_accepts_both_qualified_and_bare(ref):
    """Both the full `network.X` form (collect) and the bare `X` key that
    parse_reference hands ref_to_code resolve to the same measure."""
    assert node_label(ref) == ref.split(".")[-1]


@pytest.mark.parametrize("ref", ["network.weight", "weight", "network.edges.length",
                                 "network.observations.Bold", "theta", "positions.x", 42, None])
def test_node_label_rejects_non_node_refs(ref):
    """Edge matrices, state variables, network.observations, sub-refs, non-strings -> None."""
    assert node_label(ref) is None


def test_node_const_names():
    assert node_const("positions") == "_network_node_positions"
    assert node_const("instrength") == "_network_node_instrength"


def test_emitted_constant_name_matches_resolved_reference():
    """The name collect/emit uses (node_const on the full-form measure) is identical to
    the name ref_to_code resolves to (node_const on the parse_reference-stripped key).
    This is the exact invariant the prefix bug broke."""
    for full in ("network.positions", "network.instrength"):
        emitted = node_const(node_label(full))              # collect / emit side (full form)
        stripped_key = full.split(".", 1)[1]                # parse_reference('network.X') -> ('network','X')
        resolved = node_const(node_label(stripped_key))     # ref_to_code side (bare key)
        assert node_label(stripped_key) is not None, "bare key must resolve (the prefix bug)"
        assert emitted == resolved


class _Net:
    def node_positions(self):
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)

    def matrix(self, lab):
        return np.array([[0, 1, 2, 0], [1, 0, 0, 3],
                         [0, 0, 0, 1], [2, 0, 1, 0]], float) if lab == "weight" else None


def test_collect_from_source_pipeline_and_observer_params():
    """A node ref is embedded whether it appears in a source, a pipeline argument, or an
    observer dynamics parameter; instrength is the weighted in-degree (row sum)."""
    obs_pipeline = NS(source=["theta"],
                      pipeline=[NS(arguments={"positions": NS(value="network.positions"),
                                              "k": NS(value=2)})],
                      dynamics=None)
    obs_observer = NS(source=["theta"], pipeline=[],
                      dynamics=NS(parameters={"instr": NS(source="network.instrength")}))
    exp = NS(network=_Net(), observations={"a": obs_pipeline, "b": obs_observer})

    arrays = collect_network_node_arrays(exp)
    assert set(arrays) == {"positions", "instrength"}
    assert np.allclose(arrays["positions"], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert np.allclose(arrays["instrength"], [3, 4, 1, 3])   # weight.sum(axis=1) = incoming


def test_collect_raises_when_vector_unbuildable():
    """A referenced node vector that cannot be built from the network is a hard error,
    not a silent empty constant."""
    class _NoWeight(_Net):
        def matrix(self, lab):
            return None
    exp = NS(network=_NoWeight(),
             observations={"a": NS(source=["network.instrength"], pipeline=[], dynamics=None)})
    with pytest.raises(ValueError):
        collect_network_node_arrays(exp)
