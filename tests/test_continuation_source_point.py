"""A branch declares where it starts, and a backend that cannot honour that says so.

`Continuation.branches` accepts `source_point: "<kind>:<n>"`. The BifurcationKit codim-1 path emits a periodic-orbit continuation, which starts from a Hopf point; it has no equilibrium branch switching. Before, a `bp:` or `fold:` source was parsed for its index and its KIND was dropped, so the emitted Julia looked for Hopf points, found none where the declaration meant a pitchfork, and wrote a result with the branch simply missing — a spec that validates, a run that succeeds, and an output that is quietly short one branch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tvbo.adapters.bifurcationkit import BifurcationKitAdapter


def _branch(source_point, name="po"):
    """A minimal periodic-orbit BranchSwitch stand-in: only what `_prepare_branch` reads."""
    return SimpleNamespace(
        name=name, source_point=source_point, continuation=None, discretization=None, delta_p=None, bothside=None
    )


@pytest.mark.parametrize("source", ["hopf:all", "hopf:1", "hopf:-1", None])
def test_a_hopf_source_is_accepted(source):
    """The kinds this path actually emits, including the unset default."""
    assert BifurcationKitAdapter._prepare_branch(_branch(source))


@pytest.mark.parametrize("source", ["bp:1", "fold:2", "bp:all"])
def test_a_source_the_backend_cannot_start_from_is_refused(source):
    """Silently emitting a Hopf switch for a declared branch point is the failure this guards."""
    with pytest.raises(ValueError, match="source_point"):
        BifurcationKitAdapter._prepare_branch(_branch(source))


def test_the_refusal_names_the_branch_and_what_to_declare_instead():
    """A spec author has to be able to act on it without reading the adapter."""
    with pytest.raises(ValueError) as excinfo:
        BifurcationKitAdapter._prepare_branch(_branch("bp:1", name="nontrivial"))
    message = str(excinfo.value)
    assert "nontrivial" in message
    assert "hopf" in message
    assert "initial_state" in message
