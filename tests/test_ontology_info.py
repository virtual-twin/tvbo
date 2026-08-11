"""`get_info` states as much of a class as that class carries, and refuses to guess.

Most of the ontology is not fully annotated: of its ~1500 classes, roughly half carry no
``definition`` and a dozen carry no ``label``, and every class reached through an import
carries no TVB-O annotation property at all. Reading any of them unconditionally is why
this used to raise ``AttributeError`` on more classes than it printed.

Resolving a name is held to the opposite standard: exact or nothing. The ontology's
``label_search`` matches loosely, which is right for a search box and wrong here — asked
for ``Annotation`` it answers with ``annotation criteria application``, a different class,
giving no sign that it guessed.
"""

from __future__ import annotations

import pytest

from tvbo.ontology.owl import get_info, onto, resolve_class


def _first(predicate):
    """The first ontology class satisfying *predicate*, or a skip when there is none."""
    for cls in onto.classes():
        if predicate(cls):
            return cls
    pytest.skip("no ontology class in this shape")


def test_every_ontology_class_renders():
    """A class the ontology defines is a class `get_info` can print — all of them."""
    classes = list(onto.classes())
    assert classes, "ontology loaded no classes — this test would assert nothing"

    failed = []
    for cls in classes:
        try:
            get_info(cls)
        except Exception as exc:
            failed.append(f"{cls.name}: {type(exc).__name__}: {exc}")
    assert not failed, "\n".join(failed[:10])


def test_a_class_without_a_definition_still_states_what_it_has():
    """The definition section is left out, rather than the whole summary being lost."""
    cls = _first(lambda c: c.label.first() and not (getattr(c, "definition", None) or [None])[0])

    info = get_info(cls)

    assert info.splitlines()[0] == cls.label.first()
    assert info.splitlines()[1] == "=" * len(cls.label.first())


def test_a_class_without_a_label_is_headed_by_its_iri_fragment():
    """Something has to name it, and the fragment is the one thing every class has."""
    cls = _first(lambda c: not c.label.first())

    assert get_info(cls).splitlines()[0] == cls.name


def test_a_name_the_ontology_does_not_define_raises():
    """It named nothing; answering with `None` only moves the failure somewhere worse."""
    with pytest.raises(KeyError, match="NoSuchClassXYZ"):
        get_info("NoSuchClassXYZ")


def test_an_imported_class_resolves_by_its_fragment():
    """`onto[...]` resolves only within TVB-O's own IRI, so imports were unreachable."""
    assert onto["Annotation"] is None, "the premise: the direct lookup does not find it"

    assert resolve_class("Annotation").name == "Annotation"


def test_resolution_by_name_is_exact_not_a_search():
    """A near match is a different class, and returning it silently is the failure."""
    assert resolve_class("Annotation").label.first() == "parcellation annotation"


def test_a_class_resolves_by_its_label_too():
    """The label is what a reader sees, so it has to be a way of asking for the class."""
    cls = _first(lambda c: c.label.first() and c.label.first() != c.name)

    assert resolve_class(cls.label.first()) is cls
