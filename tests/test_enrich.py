"""``enrich()`` — filling a record's gaps from the entity it names.

Construction resolves what is deterministic, local and cheap: the dialect expands an
``iri`` from one curated file. Everything past that is an act the caller asks for, and
``enrich()`` is the verb. What these pin is that it is one verb with one contract, reached
by a rule rather than a list — the schema says which classes carry it, and the class says
which sources answer.
"""

from __future__ import annotations

import pytest

from tvbo.datamodel import pydantic as pyd
from tvbo.datamodel import schema

GENERATED_FORMS = pytest.mark.parametrize("model", (schema, pyd), ids=("dataclass", "pydantic"))


@pytest.mark.backend_core
def test_the_schema_says_which_classes_are_enrichable():
    """A class that may name an entity elsewhere is one that can be filled from it.

    Nothing lists them: declaring the slot is what makes a class's records enrichable, and
    a subclass inherits the base its parent was given.
    """
    from tvbo.behaviour._enrich import IriEnrichable

    for form in (schema, pyd):
        for cls_name in ("Coupling", "Dynamics", "Observation", "Function", "LossFunction"):
            assert issubclass(getattr(form, cls_name), IriEnrichable), f"{form.__name__}.{cls_name}"

    assert "iri" not in schema.Integrator.__dataclass_fields__
    assert not issubclass(schema.Integrator, IriEnrichable)


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_record_is_filled_from_what_it_names(model):
    """By ``name`` when it has one; a self-contained record simply gains nothing."""
    coupling = model.Coupling(name="KuramotoCoupling")
    assert coupling.pre_expression is None

    coupling.enrich()
    assert coupling.pre_expression is not None
    assert "a" in coupling.parameters


@pytest.mark.backend_core
@GENERATED_FORMS
def test_filling_never_overwrites(model):
    """Gap-filling is the whole contract: what the record carries, it keeps."""
    coupling = model.Coupling(
        name="KuramotoCoupling",
        pre_expression={"rhs": "sin(x_j - x_i)"},
        parameters={"a": {"value": 99.0}},
    )

    coupling.enrich()

    assert str(coupling.pre_expression.rhs) == "sin(x_j - x_i)"
    assert coupling.parameters["a"].value == 99.0
    assert len(coupling.parameters) > 1, "siblings from the entry were dropped"


@pytest.mark.backend_core
@GENERATED_FORMS
def test_the_first_source_that_resolves_answers_alone(monkeypatch, model):
    """Ranked, not combined — otherwise one record enriches to two different things.

    Topping a curated record up from the ontology adds whatever the curators left out,
    and only where the ontology resolves the same names: ``Sigmoidal`` filled from both
    came to 5 parameters locally and 10 in CI, from the same YAML.

    Asserted on the mechanism rather than a count, so it holds wherever it runs.
    """
    reached = []
    monkeypatch.setattr(
        type(model.Coupling(name="probe")),
        "_from_ontology",
        lambda self, key: reached.append(key) or True,
    )

    coupling = model.Coupling(name="Mine", iri="tvbo:Sigmoidal").enrich()

    assert coupling.parameters, "the database source did not answer"
    assert reached == [], "the ontology was consulted after the database had answered"


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_later_source_answers_when_the_first_does_not(monkeypatch, model):
    """Ranked means fallback, not exclusivity: the ontology still covers what is uncurated."""
    reached = []
    monkeypatch.setattr(
        type(model.Coupling(name="probe")),
        "_from_ontology",
        lambda self, key: reached.append(key) or True,
    )

    model.Coupling(name="Mine", iri="tvbo:NotCuratedAnywhere").enrich()

    assert reached == ["NotCuratedAnywhere"]


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_pointer_that_resolves_nowhere_raises(model):
    """An ``iri`` is a pointer, and one pointing at nothing is a typo worth hearing about."""
    coupling = model.Coupling(iri="tvbo:NoSuchCouplingAnywhere")

    with pytest.raises(LookupError, match="NoSuchCouplingAnywhere"):
        coupling.enrich()


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_bare_name_that_resolves_nowhere_does_not(model):
    """A self-contained record that happens to be called something is the normal case."""
    coupling = model.Coupling(name="MyOwnPrivateCoupling")
    assert coupling.enrich() is coupling


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_key_redirects_the_lookup(model):
    """A record named for its role in a network still declares what it is an instance of."""
    coupling = model.Coupling(name="ExcitatoryCoupling")
    coupling.enrich(key="KuramotoCoupling")

    assert coupling.name == "ExcitatoryCoupling"
    assert coupling.pre_expression is not None


@pytest.mark.backend_core
@GENERATED_FORMS
def test_a_source_the_class_does_not_have_raises(model):
    """Sources dispatch on the reader the class defines, so asking for a missing one is an error."""
    with pytest.raises(ValueError, match="ontology"):
        model.Function(name="f").enrich(source="ontology")


@pytest.mark.backend_core
@GENERATED_FORMS
def test_filling_leaves_the_class_its_own_containers(model):
    """Assigning a plain container into a LinkML slot is what makes a ``JsonObj``.

    A keyed collection is therefore mutated, never assigned — otherwise the setter wraps
    what it is handed and the typed members inside stop reading as themselves.
    """
    from tvbo.utils import keyed_items

    observation = model.Observation(name="mine", iri="tvbo:BOLD_TVB")
    observation.enrich(source="database")

    names = [name for name, _ in keyed_items(observation.parameters, "parameters")]
    assert "TR" in names
    for _, parameter in keyed_items(observation.parameters, "parameters"):
        assert parameter.value is not None, "a member was flattened by the setter"


@pytest.mark.backend_core
def test_a_model_is_left_where_construction_would_have():
    """Filling a model can add derived variables, which the emitters read in dependency order."""
    from tvbo.classes.dynamics import Dynamics

    model = Dynamics(name="Generic2dOscillator")
    model.enrich()

    assert model.state_variables
    assert model.parameters
