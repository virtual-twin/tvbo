"""Naming an entity by ``iri`` is a spelling the dialect expands, before validation.

A recipe may point at a curated entity instead of spelling it out. That expansion has to
happen while the *authored* keys are still distinguishable from schema defaults, because
after construction every slot carrying a default reads as though it had been written — and
"the recipe did not say this" is exactly the question the merge has to answer.

Two properties follow, and both are pinned here: the recipe always wins over the entry, and
a record that states its own ``name`` is a definition rather than a reference, so its
``iri`` is grounding and expands nothing.
"""

from __future__ import annotations

import pytest

from tvbo.datamodel import pydantic as pyd
from tvbo.datamodel import schema

GENERATED_FORMS = pytest.mark.parametrize("model", (schema, pyd), ids=("dataclass", "pydantic"))


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_reference_adopts_the_curated_record(model):
    """An entity named solely by ``iri`` arrives carrying what that entry declares."""
    coupling = model.Coupling(iri="tvbo:FastLinearCoupling")

    assert coupling.name == "FastLinearCoupling"
    assert coupling.pre_expression is not None
    assert "G" in coupling.parameters


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_curated_delayed_is_applied(model):
    """The reason this belongs before validation, and not in a post-init hook.

    ``delayed`` carries a schema default of ``True``. After construction, an explicit
    ``true`` and an absent key are the same value, so the guard that would have applied the
    entry could never fire and no curated ``delayed:`` was ever honoured.
    ``FastLinearCoupling`` describes itself as instantaneous and declares ``delayed: false``;
    TVBO ran it delayed anyway.
    """
    assert model.Coupling(iri="tvbo:FastLinearCoupling").delayed is False
    assert model.Coupling(iri="tvbo:Linear").delayed is True


@pytest.mark.backend_core
@GENERATED_FORMS
def test_the_recipe_wins_over_the_entry(model):
    """The entry is the base; anything the recipe states supervenes, leaf by leaf."""
    coupling = model.Coupling(iri="tvbo:FastLinearCoupling", delayed=True)
    assert coupling.delayed is True

    refined = model.Coupling(iri="tvbo:Linear", parameters={"a": {"value": 99.0}})
    assert refined.parameters["a"].value == 99.0
    assert len(refined.parameters) > 1, "siblings from the entry were dropped"


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_definition_is_not_expanded(model):
    """A record stating its own ``name`` is a definition; its ``iri`` only grounds it.

    Fifty curated files are written that way. Expanding them would re-derive a definition
    from a name lookup — and ``ReducedWongWangFunc.yaml`` grounds on ``tvbo:ReducedWongWang``
    while stating its own name, so it would be overwritten by the different, canonical
    ``ReducedWongWang.yaml``.
    """
    coupling = model.Coupling(iri="tvbo:FastLinearCoupling", name="MyOwnCoupling")

    assert coupling.name == "MyOwnCoupling"
    assert coupling.delayed is True, "a definition adopted the entry it merely grounds on"


@pytest.mark.backend_core
@GENERATED_FORMS
def test_an_iri_naming_nothing_is_left_alone(model):
    """It may name an entity that exists only in the ontology, which this pass cannot reach.

    Distinguishing that from a typo is `tvbo validate`'s job, and `enrich()`'s — which
    raises, because there the caller asked.
    """
    coupling = model.Coupling(iri="tvbo:NoSuchCouplingAnywhere")
    assert coupling.iri == "tvbo:NoSuchCouplingAnywhere"


@pytest.mark.backend_core
def test_the_curated_record_still_loads_as_itself():
    """The regression the definition guard exists for, on the file that exposed it."""
    from tvbo.classes.dynamics import Dynamics

    dynamics = Dynamics.from_db("ReducedWongWangFunc")

    assert "H" in dynamics.functions
    assert list(dynamics.functions["H"].arguments) == ["x"]
