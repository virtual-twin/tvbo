"""APA citation rendering, including the author lists that used to crash it.

BibTeX truncates an author list by ending it with `and others`. pybtex parses that as a
person whose only name is `others` and who has no first name, so formatting code that
takes a first initial from every author raises `IndexError` on it. Twelve of the 115
entries in the shipped bibliography use the idiom, which made ten curated models unable to
produce a report at all.
"""

from __future__ import annotations

import pytest
from pybtex.database import Person

from tvbo.data import db
from tvbo.utils.report import _format_authors, get_citation


def _people(*names: str) -> list[Person]:
    return [Person(name) for name in names]


def test_a_single_author_keeps_its_initial():
    """Given names only, matching the existing house style — pybtex files `H.` as a
    middle name and reports have never shown middle initials."""
    assert _format_authors(_people("Jansen, Ben H.")) == "Jansen, B."


def test_two_authors_are_joined_with_an_ampersand():
    assert _format_authors(_people("Jansen, Ben H.", "Rit, Vincent G.")) == "Jansen, B. & Rit, V."


def test_three_authors_take_a_serial_comma():
    formatted = _format_authors(_people("Deco, Gustavo", "Jirsa, Viktor", "McIntosh, Anthony"))

    assert formatted == "Deco, G., Jirsa, V., & McIntosh, A."


@pytest.mark.parametrize("marker", ["others", "{others}", "et al.", "{et al.}", "al."])
def test_the_bibtex_truncation_idiom_becomes_et_al(marker: str):
    """`and others` is an abbreviation, not a person named Others."""
    assert _format_authors(_people("Proix, Timothee", marker)) == "Proix, T. et al."


def test_an_author_with_no_first_name_keeps_the_surname():
    """A single-name author — an organisation, say — is not dropped and does not raise."""
    assert _format_authors(_people("Anonymous")) == "Anonymous"


def test_no_authors_renders_empty_rather_than_raising():
    assert _format_authors([]) == ""


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("van der Pol, Balthasar", "van der Pol, B."),
        ("di Volo, Matteo", "di Volo, M."),
        ("van Wyk, Michael A", "van Wyk, M."),
    ],
)
def test_a_particled_surname_keeps_its_particles(written: str, expected: str):
    """`van der Pol` is not `Pol`.

    pybtex splits a particled surname across `prelast_names` and `last_names`, so reading
    only the latter cites a different person's name entirely. Four shipped entries carry
    six such authors, and every report citing them was wrong.
    """
    assert _format_authors([Person(written)]) == expected


def test_every_shipped_bibliography_entry_renders():
    """The whole bibliography formats, so no entry can silently break a study's report."""
    entries = db.load_bibliography().entries

    assert entries, "bibliography is empty — this test would assert nothing"
    for key in entries:
        rendered = get_citation(key)
        assert rendered and "not found" not in rendered, f"{key} did not render"
