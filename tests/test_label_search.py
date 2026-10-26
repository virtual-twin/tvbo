"""``label_search`` passes the search term to SPARQL as data, so a term with quotes or backslashes, such as a study title, is matched like any other text instead of breaking the query."""

from __future__ import annotations

import types

import owlready2
import pytest

from tvbo.ontology import query

TITLES = [
    "The Genesis of Chua's Circuit",
    'A hyperchaotic attractor coined from chaotic L"u system',
    "Remark on ``Some Simple Chaotic Flows''",
    "synthesis, \\& biological applications",
]


@pytest.fixture(scope="module")
def titles_onto():
    """A throwaway ontology with one class per title, labelled with that title."""
    onto = owlready2.World().get_ontology("https://example.org/titles.owl")
    with onto:
        for number, title in enumerate(TITLES):
            types.new_class(f"Title{number}", (owlready2.Thing,)).label = [title]
    return onto


@pytest.mark.parametrize("number", range(len(TITLES)))
def test_a_label_with_quotes_or_backslashes_is_found(titles_onto, number):
    found = query.label_search(TITLES[number], exact_match="all", onto=titles_onto)

    assert [entity.name for entity in found] == [f"Title{number}"]
