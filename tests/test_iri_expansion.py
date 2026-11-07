"""Naming an entity by ``iri`` is a spelling the dialect expands, before validation.

A recipe may point at a curated entity instead of spelling it out. That expansion has to happen while the *authored* keys are still distinguishable from schema defaults, because after construction every slot carrying a default reads as though it had been written — and "the recipe did not say this" is exactly the question the merge has to answer.

Two properties follow, and both are pinned here: an ``iri`` always points at the curated entry, and whatever else the record states, its ``name`` included, overrides that entry leaf by leaf.
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

    ``delayed`` carries a schema default of ``True``. After construction, an explicit ``true`` and an absent key are the same value, so the guard that would have applied the entry could never fire and no curated ``delayed:`` was ever honoured.
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
def test_a_stated_name_overrides_and_still_expands(model):
    """A ``name`` beside the ``iri`` is one more override, never a reason to skip the entry."""
    coupling = model.Coupling(iri="tvbo:FastLinearCoupling", name="MyOwnCoupling")

    assert coupling.name == "MyOwnCoupling"
    assert coupling.delayed is False, "the entry the iri points at was not applied"
    assert "G" in coupling.parameters


@pytest.mark.backend_core
@GENERATED_FORMS
def test_an_iri_naming_nothing_is_left_alone(model):
    """It may name an entity that exists only in the ontology, which this pass cannot reach.

    Distinguishing that from a typo is `tvbo validate`'s job, and `enrich()`'s — which raises, because there the caller asked.
    """
    coupling = model.Coupling(iri="tvbo:NoSuchCouplingAnywhere")
    assert coupling.iri == "tvbo:NoSuchCouplingAnywhere"
    assert coupling.name == "NoSuchCouplingAnywhere", "the record takes the iri's local name"


@pytest.mark.backend_core
def test_the_curated_record_still_loads_as_itself():
    """A curated record loads as itself, its functions intact."""
    from tvbo.classes.dynamics import Dynamics

    dynamics = Dynamics.from_db("ReducedWongWangFunc")

    assert "H" in dynamics.functions
    assert list(dynamics.functions["H"].arguments) == ["x"]


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_scoped_iri_expands_and_names_the_record_it_addresses(model):
    """``tvbo:observation_models/bold_tvb`` addresses ``bold_tvb`` among the observations, not a record called ``observation_models/bold_tvb``."""
    scoped, unscoped = (
        model.Observation(iri=iri, source=["S_e"]) for iri in ("tvbo:observation_models/bold_tvb", "tvbo:BOLD_TVB")
    )

    assert scoped.name == unscoped.name == "BOLD_TVB"
    assert len(scoped.pipeline) == len(unscoped.pipeline) > 0


@pytest.mark.backend_core
def test_an_iri_addresses_a_record_by_its_folder():
    from tvbo.data.registry import curated_key

    assert curated_key("tvbo:observation_models/bold_tvb", "Observation") == "bold_tvb"
    assert curated_key("tvbo:BOLD_TVB", "Observation") == "BOLD_TVB"
    assert curated_key("ReducedWongWang", "Dynamics") == "ReducedWongWang"
    assert curated_key("tvbo:coupling_functions/Linear", "Coupling") == "Linear"
    assert curated_key("tvbo:models/Linear", "Coupling") is None, "a record in another folder is not this class's"
    assert curated_key("tvbo:coupling_functions/Linear", "Observation") is None, (
        "a scope of another class addresses no observation"
    )
    assert curated_key("tvbo:result/exp-1", "Observation") is None, "a run's output is not a curated record"


@pytest.mark.backend_core
def test_a_record_iri_ends_in_its_file_stem():
    """A record answers to its ``name:`` and to its stem in any case, but its IRI is the stem it is filed under, in its folder however deep."""
    from tvbo.data.registry import iri_for

    assert iri_for("Dynamics", "ModelJansen1995") == "tvbo:models/Jansen1995"
    assert iri_for("Observation", "BOLD_TVB") == "tvbo:observation_models/bold_tvb"
    assert iri_for("Dynamics", "HodgkinHuxley") == "tvbo:models/HodgkinHuxley", "a model filed in models/julia/"
    with pytest.raises(FileNotFoundError):
        iri_for("Coupling", "Kuramoto")


@pytest.mark.backend_core
def test_a_singular_scope_still_resolves_and_names_its_folder_form():
    from tvbo.data.registry import curated_key, local_name

    with pytest.warns(FutureWarning, match="tvbo:observation_models/bold_tvb"):
        assert curated_key("tvbo:observation/bold_tvb", "Observation") == "bold_tvb"
    with pytest.warns(FutureWarning, match="tvbo:themes/default"):
        assert local_name("tvbo:theme/default") == "default"


@pytest.mark.backend_core
def test_a_study_schema_term_written_in_full_is_not_a_retired_scope():
    from tvbo.data.registry import iri_target

    assert iri_target("https://w3id.org/tvbo/study/StudyLayout") is None, "the study module's namespace is no folder"


@pytest.mark.backend_core
def test_a_network_iri_scoped_to_another_folder_resolves_no_network():
    from tvbo.classes.network import Network

    assert Network._resolve_network_iri("tvbo:networks/none") is None
    assert Network._resolve_network_iri("tvbo:atlases/DesikanKilliany") is None, "an atlas is not a network"
    assert Network._resolve_network_iri("tvbo:DesikanKilliany") is not None
