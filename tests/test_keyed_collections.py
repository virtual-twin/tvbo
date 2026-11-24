"""A keyed collection means the same thing in both spellings.

``parameters: {TR: {value: 720.0}}`` says a Parameter *called* ``TR``, so writing the name a second time inside the member is a redundancy this project's records are written without; the generated models alone would reject the member as incomplete, and the list spelling of the same collection as not a mapping.

Both are the dialect's business, and what these pin is that the spelling does not change what a record means.
"""

from __future__ import annotations

import pytest

from tvbo import datamodel as dm


@pytest.mark.backend_core
def test_a_member_is_named_by_its_key():
    """The mapping spelling, with no redundant inner name."""
    coupling = dm.Coupling(name="C", parameters={"a": {"value": 1.5}})

    assert coupling.parameters["a"].name == "a"
    assert coupling.parameters["a"].value == 1.5


@pytest.mark.backend_core
def test_the_list_spelling_means_the_same():
    """A list of bare identifiers, and a list of whole members, both arrive keyed."""
    bare = dm.Function(name="f", arguments=["v", "w"])
    assert [str(argument) for argument in bare.arguments] == ["v", "w"]

    whole = dm.Coupling(name="C", parameters=[{"name": "a", "value": 1.5}])
    assert whole.parameters["a"].value == 1.5


@pytest.mark.backend_core
def test_a_member_may_still_state_its_own_name():
    """Fifty curated files spell it; the key and the name agree, and the name is kept."""
    coupling = dm.Coupling(name="C", parameters={"a": {"name": "a", "value": 1.5}})
    assert coupling.parameters["a"].name == "a"


@pytest.mark.backend_core
def test_a_curated_record_loads():
    """An ``iri`` reference to a record written without redundant names: ``BOLD_TVB`` writes ``TR: {value: 720.0}``."""
    observation = dm.Observation(iri="tvbo:BOLD_TVB")

    assert observation.parameters["TR"].value == 720.0
    assert observation.parameters["TR"].name == "TR"
    assert observation.pipeline


@pytest.mark.backend_core
def test_an_unkeyable_list_is_left_to_fail():
    """Members stating no identifier cannot be keyed, and guessing is worse than raising."""
    from tvbo.datamodel.dialect import key_members

    data = {"parameters": [{"value": 1.5}]}
    key_members("Coupling", data)

    assert data["parameters"] == [{"value": 1.5}]


@pytest.mark.backend_core
def test_a_list_valued_collection_is_not_keyed():
    """``inlined_as_list`` is a collection whose spelling IS a list; it has no keys."""
    from tvbo.datamodel.dialect import KEYED_COLLECTIONS

    assert "constructor_args" not in KEYED_COLLECTIONS.get("ClassReference", {})
