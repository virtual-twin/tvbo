#!/usr/bin/env python3
"""Post-process the LinkML-generated `tvb-o-struct.owl` (T-box).

LinkML's `gen-owl` emits a usable OWL file but stops short of the publication-readiness items in §2.3 of the ontology restructuring plan:

* canonical ontology IRI (`https://w3id.org/tvbo/struct`) instead of
  the synthetic `https://w3id.org/tvbo.owl.ttl`;
* `owl:versionIRI` pinned to the schema `version`;
* `owl:versionInfo` mirroring `pav:version`;
* `dcterms:created` / `dcterms:modified` / `dcterms:creator` /
  `dcterms:contributor` / `dcterms:description`;
* `rdfs:isDefinedBy <ontology IRI>` on every TVB-O class.

We do this as a post-process rather than fighting LinkML's emitter so that future LinkML upgrades remain drop-in.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import yaml
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPO / "schema" / "tvbo_datamodel.yaml"
DEFAULT_OWL = REPO / "ontology" / "tvb-o-struct.owl"

CANONICAL_IRI = URIRef("https://w3id.org/tvbo/struct")
TVBO = Namespace("https://w3id.org/tvbo/")
PAV = Namespace("http://purl.org/pav/")


def _retype_float_facets(g: Graph) -> None:
    """Retype numeric facet literals on ``xsd:float`` datatype restrictions.

    LinkML's ``gen-owl`` emits a float slot's ``minimum_value`` /
    ``maximum_value`` bounds as bare ``0e+00`` / ``1e+00`` literals, which Turtle types as ``xsd:double``. HermiT rejects an ``xsd:double`` facet value on an
    ``owl:onDatatype xsd:float`` restriction (``the maxInclusive facet takes only floats as values when used on an xsd:float datatype``); ELK does not check
    facets, so only HermiT surfaces it. Re-type each facet literal to
    ``xsd:float`` so the restriction is internally consistent for both reasoners.
    """
    facet_predicates = (XSD.minInclusive, XSD.maxInclusive, XSD.minExclusive, XSD.maxExclusive)
    for datatype_node in set(g.subjects(OWL.onDatatype, XSD.float)):
        for restriction_list in g.objects(datatype_node, OWL.withRestrictions):
            for facet_node in g.items(restriction_list):
                for facet in facet_predicates:
                    for value in list(g.objects(facet_node, facet)):
                        retyped = Literal(value.toPython(), datatype=XSD.float)
                        if retyped != value:
                            g.remove((facet_node, facet, value))
                            g.add((facet_node, facet, retyped))


def _augment(g: Graph, schema: dict) -> None:
    # Drop the synthetic ontology declaration LinkML emits; it points at
    # `https://w3id.org/tvbo.owl.ttl` which is not a real artefact.
    legacy = URIRef("https://w3id.org/tvbo.owl.ttl")
    legacy_triples = list(g.triples((legacy, None, None)))
    for t in legacy_triples:
        g.remove(t)

    onto = CANONICAL_IRI
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal(schema.get("name", "tvb-o-struct"))))
    g.add((onto, DCTERMS.title, Literal(schema.get("title", "The Virtual Brain Ontology — Structural T-box"))))
    if schema.get("description"):
        g.add((onto, DCTERMS.description, Literal(schema["description"])))
    if schema.get("license"):
        g.add((onto, DCTERMS.license, URIRef(schema["license"])))
    version = str(schema.get("version", "0.0.0"))
    g.add((onto, OWL.versionIRI, URIRef(f"https://w3id.org/tvbo/struct/{version}")))
    g.add((onto, OWL.versionInfo, Literal(version)))
    g.add((onto, PAV.version, Literal(version)))

    today = datetime.date.today().isoformat()
    if schema.get("created_on"):
        g.add((onto, DCTERMS.created, Literal(str(schema["created_on"]), datatype=XSD.date)))
    g.add((onto, DCTERMS.modified, Literal(today, datatype=XSD.date)))

    for slot, prop in (("created_by", DCTERMS.creator), ("contributors", DCTERMS.contributor)):
        val = schema.get(slot)
        if not val:
            continue
        if isinstance(val, str):
            val = [val]
        for v in val:
            v = str(v)
            if v.startswith(("http://", "https://", "mailto:")):
                g.add((onto, prop, URIRef(v)))
            else:
                g.add((onto, prop, Literal(v)))

    # Hard-coded fallback contributor: the project itself, since the schema does not carry per-author ORCIDs (LinkML's gen-owl rejects bare-string contributor entries, so we inject them here instead).
    g.add((onto, DCTERMS.contributor, Literal("The Virtual Brain Ontology contributors")))
    g.add((onto, DCTERMS.publisher, URIRef("https://www.thevirtualbrain.org/")))

    g.add((onto, RDFS.seeAlso, URIRef("https://w3id.org/tvbo/axioms")))
    g.add((onto, RDFS.seeAlso, URIRef("https://w3id.org/tvbo/data")))

    # `rdfs:isDefinedBy` on every TVB-O class declared in this file.
    for cls in set(g.subjects(RDF.type, OWL.Class)):
        if isinstance(cls, URIRef) and str(cls).startswith(str(TVBO)):
            g.add((cls, RDFS.isDefinedBy, onto))

    # Scope LinkML enum permissible-value labels to their parent enum so that
    # ROBOT report stops flagging cross-enum collisions like
    # `DimensionType#time` vs `EventType#time` vs `SamplingAxis#time` (each carried `rdfs:label "time"`). The enum local-name comes from the IRI fragment before `#`. Demote the bare value to `skos:notation` and synthesize `"<value> (<EnumName>)"` as the human label.
    # Same treatment for slot/property IRIs nested under sub-namespaces (`tvb-o/dbs/scale`, `tvb-o/software/scale`) which collide on the bare local name `scale`. The unscoped sibling (`tvb-o/scale`) is treated as canonical and keeps the bare label.
    PROPERTY_TYPES = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)
    seen_subjects: set[URIRef] = set()
    for ptype in PROPERTY_TYPES:
        for s in g.subjects(RDF.type, ptype):
            if isinstance(s, URIRef):
                seen_subjects.add(s)

    for subj in seen_subjects:
        s = str(subj)
        if not s.startswith(str(TVBO)):
            continue
        local = s[len(str(TVBO)) :]
        scope: str | None = None
        value: str | None = None
        if "#" in local:
            scope, value = local.split("#", 1)
        elif "/" in local:
            head, _, tail = local.rpartition("/")
            if head and tail:
                scope, value = head.replace("/", "."), tail
        if scope is None or value is None:
            continue
        bare_label = Literal(value)
        if (subj, RDFS.label, bare_label) not in g:
            continue
        g.remove((subj, RDFS.label, bare_label))
        g.add((subj, SKOS.notation, bare_label))
        g.add((subj, RDFS.label, Literal(f"{value} ({scope})")))

    # Top-level TVBO ObjectProperty / DatatypeProperty / AnnotationProperty labels (`tvbo:x`, `tvbo:license`, ...) collide with InterLex / SANDS imported class IRIs that share the same readable label. Scope the
    # TVBO-side label as "<name> (slot)" while leaving the imported class label untouched. Bare value moves to `skos:notation`.
    SLOT_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)
    slot_subjects: set[URIRef] = set()
    for st in SLOT_TYPES:
        for s in g.subjects(RDF.type, st):
            if isinstance(s, URIRef):
                slot_subjects.add(s)
    for subj in slot_subjects:
        s = str(subj)
        if not s.startswith(str(TVBO)):
            continue
        local = s[len(str(TVBO)) :]
        if "/" in local or "#" in local:
            continue  # already handled above
        bare_label = Literal(local)
        if (subj, RDFS.label, bare_label) not in g:
            continue
        g.remove((subj, RDFS.label, bare_label))
        g.add((subj, SKOS.notation, bare_label))
        g.add((subj, RDFS.label, Literal(f"{local} (slot)")))

    # ROBOT report flags `missing_label` on imported externals that LinkML references but does not relabel. Inject minimal `rdfs:label`s based on the IRI local name so downstream tooling has something to display.
    SCHEMA = Namespace("http://schema.org/")
    SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")
    XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
    EXTERNAL_LABELS: list[tuple[URIRef, str]] = [
        (SCHEMA.Book, "Book"),
        (SCHEMA.CreativeWork, "Creative work"),
        (SCHEMA.Report, "Report"),
        (SCHEMA.ScholarlyArticle, "Scholarly article"),
        (SCHEMA.Chapter, "Chapter"),
        (SCHEMA.Thesis, "Thesis"),
        (SKOS_NS.Concept, "SKOS Concept"),
        (XSD_NS.string, "string"),
    ]
    for subj, lbl in EXTERNAL_LABELS:
        if not list(g.objects(subj, RDFS.label)):
            g.add((subj, RDFS.label, Literal(lbl)))

    # `tvbo:SoftwarePackage` is declared by us but lacks a label (LinkML emitted only `skos:definition`). Add the human label.
    sp = TVBO.SoftwarePackage
    if not list(g.objects(sp, RDFS.label)):
        g.add((sp, RDFS.label, Literal("Software package")))

    # Dedupe `rdfs:label` per subject. ROBOT flags `multiple_labels` even when two labels carry the same lexical form but differ only in language tag (e.g. `"CouplingInput"` vs `"CouplingInput"@en`). For external IRIs we are not the source of truth: keep at most one label, deduplicating deterministically. For TVBO IRIs, prefer the language-tagged variant when it has identical lexical form.
    for subj in set(g.subjects(RDFS.label, None)):
        if not isinstance(subj, URIRef):
            continue
        labels = list(g.objects(subj, RDFS.label))
        if len(labels) <= 1:
            continue
        # Group labels by lexical value; if all share the same value, collapse to a single label preferring one with a language tag.
        unique_lex = {str(lbl) for lbl in labels}
        if len(unique_lex) == 1:
            preferred = next((lbl for lbl in labels if isinstance(lbl, Literal) and lbl.language), labels[0])
            for lbl in labels:
                if lbl != preferred:
                    g.remove((subj, RDFS.label, lbl))
            continue
        if str(subj).startswith(str(TVBO)):
            continue  # genuine multiple labels on a TVBO IRI: leave alone
        sorted_labels = sorted({str(lbl) for lbl in labels})
        for lbl in labels:
            g.remove((subj, RDFS.label, lbl))
        g.add((subj, RDFS.label, Literal(sorted_labels[0])))
        for lbl in sorted_labels[1:]:
            g.add((subj, SKOS.altLabel, Literal(lbl)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", type=pathlib.Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--owl", type=pathlib.Path, default=DEFAULT_OWL)
    args = ap.parse_args()

    schema = yaml.safe_load(args.schema.read_text())
    g = Graph()
    g.parse(args.owl, format="turtle")
    g.bind("dcterms", DCTERMS)
    g.bind("pav", PAV)
    g.bind("owl", OWL)
    _retype_float_facets(g)
    _augment(g, schema)
    g.serialize(destination=args.owl, format="turtle")
    print(f"\u2713 Augmented ontology header in {args.owl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
