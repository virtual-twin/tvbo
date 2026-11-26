"""Every TVB-O concept a curated record is grounded to is declared in the distributed ``tvbo.owl``, and a grounding the A-box cannot expand is refused.

A record's ``grounding`` becomes ``tvbo:surrogate_of`` in the A-box. An external target (GO, CL) is declared by its own ontology, which ``tvbo.owl`` does not import, so only the ``tvbo:`` targets are checked here. One that the ontology does not declare dangles: OWLAPI types it as a bare ``owl:NamedIndividual``, with no label.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import tvbo

pytestmark = pytest.mark.backend_core

REPO = pathlib.Path(__file__).resolve().parents[1]
MERGED_PATH = pathlib.Path(tvbo.__file__).resolve().parent / "data" / "ontology" / "tvbo.owl"
DECLARING_TYPES = {OWL.Class, SKOS.Concept, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty}
"""The types that declare a term; ``owl:NamedIndividual`` alone is what OWLAPI gives an IRI it only sees used."""

sys.path.insert(0, str(REPO / "scripts" / "ontology"))

import gen_abox  # noqa: E402


def test_every_tvbo_grounding_target_is_declared():
    targets = {
        target
        for target in gen_abox.build_graph().objects(None, gen_abox.TVBO.surrogate_of)
        if target.startswith(gen_abox.TVBO)
    }
    merged = Graph().parse(MERGED_PATH)
    undeclared = sorted(
        str(target)
        for target in targets
        if not (set(merged.objects(target, RDF.type)) & DECLARING_TYPES and merged.value(target, RDFS.label))
    )
    assert targets, "no record is grounded to a TVB-O concept"
    assert not undeclared, f"grounding targets tvbo.owl does not declare: {undeclared}"


def test_a_grounding_with_an_unknown_prefix_is_refused():
    """A mistyped prefix raises in the A-box build instead of dropping the grounding before the test above can see it."""
    iri = gen_abox.TVBO["models/Example/parameters/a"]
    with pytest.raises(ValueError, match=r"models/Example/parameters/a: grounding 'GOO:0060080'"):
        gen_abox._add_groundings(Graph(), iri, {"grounding": ["GOO:0060080"]})


def test_a_clinical_grounding_expands_by_the_schema_prefix():
    """A LOINC or SNOMED CT grounding reaches the A-box under the namespace the schema declares, as GO does."""
    iri = gen_abox.TVBO["models/Example/parameters/a"]
    g = Graph()
    gen_abox._add_groundings(g, iri, {"grounding": ["loinc:30525-0", "sct:49049000", "GO:0060080"]})
    assert {str(o) for o in g.objects(iri, gen_abox.TVBO.surrogate_of)} == {
        "http://loinc.org/rdf/30525-0",
        "http://snomed.info/id/49049000",
        "http://purl.obolibrary.org/obo/GO_0060080",
    }
