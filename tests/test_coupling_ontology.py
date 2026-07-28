"""The coupling-evaluation scheme enrichment merged into ``tvbo.owl``.

Covers ``ontology/tvb-o-coupling.ttl`` — the mergeable module that attaches
backend-independent labels and per-backend parameter mappings to the
``CouplingStageEvaluation`` enum values (``per_step`` / ``per_stage``) that
LinkML ``gen-owl`` emits into ``tvb-o-struct.owl``.

Two levels of assertion: the authored module carries the intended triples, and
those triples survive the ROBOT merge onto the *same* struct-minted identity
(not a parallel node) in the distributed ``tvbo/data/ontology/tvbo.owl``. The
second guards against a future regeneration silently dropping the enrichment.
"""

from __future__ import annotations

import pathlib

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import tvbo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ontology" / "tvb-o-coupling.ttl"
MERGED_PATH = pathlib.Path(tvbo.__file__).resolve().parent / "data" / "ontology" / "tvbo.owl"

TVBO = Namespace("https://w3id.org/tvbo/")
PER_STAGE = URIRef("https://w3id.org/tvbo/CouplingStageEvaluation#per_stage")
PER_STEP = URIRef("https://w3id.org/tvbo/CouplingStageEvaluation#per_step")
BACKEND_PARAM = TVBO.backendParameter
COUPLING_ENUM = TVBO.CouplingStageEvaluation

EXPECTED_BACKEND_PARAM = {
    PER_STAGE: "tvboptim: recompute_coupling_per_stage=True",
    PER_STEP: "tvboptim: recompute_coupling_per_stage=False",
}


@pytest.fixture(scope="module")
def module() -> Graph:
    assert MODULE_PATH.exists(), f"missing module {MODULE_PATH}"
    g = Graph()
    g.parse(str(MODULE_PATH), format="turtle")
    return g


@pytest.fixture(scope="module")
def merged() -> Graph:
    assert MERGED_PATH.exists(), f"missing merged ontology {MERGED_PATH} (run `make gen-merged`)"
    g = Graph()
    g.parse(str(MERGED_PATH))
    return g


class TestAuthoredModule:
    """The hand-authored Turtle module."""

    def test_declares_backend_parameter_annotation_property(self, module):
        assert (BACKEND_PARAM, RDF.type, OWL.AnnotationProperty) in module

    @pytest.mark.parametrize("value", [PER_STAGE, PER_STEP])
    def test_value_has_backend_parameter(self, module, value):
        params = {str(o) for o in module.objects(value, BACKEND_PARAM)}
        assert params == {EXPECTED_BACKEND_PARAM[value]}

    @pytest.mark.parametrize("value", [PER_STAGE, PER_STEP])
    def test_value_has_pref_label_and_scope_note(self, module, value):
        assert list(module.objects(value, SKOS.prefLabel)), "missing skos:prefLabel"
        assert list(module.objects(value, SKOS.scopeNote)), "missing skos:scopeNote"

    def test_per_stage_alt_label_names_the_backend_flag(self, module):
        alts = {str(o) for o in module.objects(PER_STAGE, SKOS.altLabel)}
        assert "recompute_coupling_per_stage" in alts


class TestSurvivesMerge:
    """The enrichment as it reaches the distributed knowledge graph."""

    @pytest.mark.parametrize("value", [PER_STAGE, PER_STEP])
    def test_backend_parameter_present_after_merge(self, merged, value):
        params = {str(o) for o in merged.objects(value, BACKEND_PARAM)}
        assert EXPECTED_BACKEND_PARAM[value] in params

    @pytest.mark.parametrize("value", [PER_STAGE, PER_STEP])
    def test_enrichment_lands_on_the_struct_identity(self, merged, value):
        """Enrichment and the LinkML-minted enum value are the *same* node.

        The value must keep its struct-provided identity (subClassOf the enum,
        a skos:definition) AND carry the merged-in backendParameter — proof the
        module enriched the existing IRI instead of minting a duplicate.
        """
        assert (value, RDFS.subClassOf, COUPLING_ENUM) in merged
        assert list(merged.objects(value, SKOS.definition)), "lost struct skos:definition"
        assert (value, BACKEND_PARAM, Literal(EXPECTED_BACKEND_PARAM[value])) in merged
