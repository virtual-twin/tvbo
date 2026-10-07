"""`observations.<name>` means the same thing wherever a recipe writes it.

A loss, an exploration builder and a study analysis all name an observation that way. A pipeline argument that spelled it the same used to be emitted as a quoted string, so the declared function received the literal text ``"observations.recorded_ts"`` and failed on the first attribute it read — a codegen success that only fails at run time.
"""

from __future__ import annotations

from tvbo.codegen.templater import canonical_observation_ref

NAMES = ("recorded_ts", "psd")


def test_the_spec_spelling_resolves_to_the_observation():
    assert canonical_observation_ref("observations.recorded_ts", NAMES) == "recorded_ts"


def test_a_named_output_survives_the_rewrite():
    """`observations.psd.frequencies` still has to reach the named output, not just the observation."""
    assert canonical_observation_ref("observations.psd.frequencies", NAMES) == "psd.frequencies"


def test_a_network_measure_is_left_alone():
    """`network.observations.<measure>` is a different namespace and must not be rewritten into it."""
    assert canonical_observation_ref("network.observations.BoldCorrelation", NAMES) == "network.observations.BoldCorrelation"


def test_a_name_no_observation_carries_is_left_alone():
    """Rewriting it would turn a broken reference into a differently broken one, with the original spelling lost from the error."""
    assert canonical_observation_ref("observations.nope", NAMES) == "observations.nope"


def test_a_non_string_passes_through():
    assert canonical_observation_ref(0.1, NAMES) == 0.1
