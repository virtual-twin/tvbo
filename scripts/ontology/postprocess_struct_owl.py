#!/usr/bin/env python3
"""Post-process the LinkML-generated `tvb-o-struct.owl` (T-box).

LinkML's `gen-owl` emits a usable OWL file but stops short of the publication-readiness items in §2.3 of the ontology restructuring plan:

* canonical ontology IRI (`https://w3id.org/tvbo/struct`) instead of the synthetic `https://w3id.org/tvbo.owl.ttl`;
* `owl:versionIRI` pinned to the schema `version`;
* `owl:versionInfo` mirroring `pav:version`;
* `dcterms:created` / `dcterms:modified` / `dcterms:creator` / `dcterms:contributor` / `dcterms:description`;
* an `rdfs:label` on every TVB-O class and property, its name qualified as `_qualify_labels` describes;
* no triple for a LinkML annotation tag;
* a slot's `equals_string` as the one-value data range it means.

We do this as a post-process rather than fighting LinkML's emitter so that future LinkML upgrades remain drop-in.

The result is a pure function of the schema: every header date comes from the schema rather than the clock, and the graph is written through `canonical_ttl.write_canonical`, so regenerating from an unchanged schema reproduces the committed file byte for byte.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml
from canonical_ttl import write_canonical
from linkml_runtime.utils.schemaview import SchemaView
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPO / "schema" / "tvbo_datamodel.yaml"
DEFAULT_OWL = REPO / "ontology" / "tvb-o-struct.owl"

CANONICAL_IRI = URIRef("https://w3id.org/tvbo/struct")
TVBO = Namespace("https://w3id.org/tvbo/")
PAV = Namespace("http://purl.org/pav/")
SLOT_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)


def _retype_float_facets(g: Graph) -> None:
    """Retype numeric facet literals on ``xsd:float`` datatype restrictions.

    LinkML's ``gen-owl`` emits a float slot's ``minimum_value`` / ``maximum_value`` bounds as bare ``0e+00`` / ``1e+00`` literals, which Turtle types as ``xsd:double``. HermiT rejects an ``xsd:double`` facet value on an ``owl:onDatatype xsd:float`` restriction (``the maxInclusive facet takes only floats as values when used on an xsd:float datatype``); ELK does not check facets, so only HermiT surfaces it. Re-typing each facet literal to ``xsd:float`` makes the restriction internally consistent for both reasoners.
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


def _equals_string_as_one_of(g: Graph, view: SchemaView) -> None:
    """Write each slot's `equals_string` as the one-value data range it means.

    `gen-owl` writes `equals_string: "tvbo:Network"` as an `xsd:pattern` facet, which reads the value as a regular expression, not as the literal string. ROBOT's `query` round trip drops that anonymous restriction's `rdfs:Datatype` type, so OWLAPI reads the filler as a class it cannot parse and mints `owlapi/error#ErrorN` for it. `owl:oneOf ("tvbo:Network")` states the value exactly and survives the round trip.

    Args:
        g: The parsed T-box, mutated in place.
        view: The LinkML schema with its imports, read for every class's `slot_usage`.
    """
    values = {
        (URIRef(view.get_uri(slot, expand=True)), usage.equals_string)
        for cls in view.all_classes().values()
        for slot, usage in (cls.slot_usage or {}).items()
        if usage.equals_string
    }
    for restriction, prop in list(g.subject_objects(OWL.onProperty)):
        filler = g.value(restriction, OWL.allValuesFrom)
        facets = g.value(filler, OWL.withRestrictions) if filler is not None else None
        patterns = [str(o) for item in g.items(facets) for o in g.objects(item, XSD.pattern)] if facets is not None else []
        if len(patterns) != 1 or (prop, patterns[0]) not in values:
            continue
        for item in list(g.items(facets)):
            g.remove((item, None, None))
        Collection(g, facets).clear()
        g.remove((filler, OWL.withRestrictions, None))
        g.remove((filler, OWL.onDatatype, None))
        one_of = BNode()
        Collection(g, one_of, [Literal(patterns[0])])
        g.add((filler, OWL.oneOf, one_of))


def _split_scope(local: str) -> tuple[str | None, str]:
    """Split a TVB-O local name into the scope it is nested in and its bare name.

    The scope is the enum before `#` (`DimensionType#time`) or the module path before the last `/` (`study/role`), its own `/` written as `.`; a top-level term (`license`) has none.

    Args:
        local: The IRI after `https://w3id.org/tvbo/`.

    Returns:
        The scope, or `None` for a top-level term, and the bare name.
    """
    if "#" in local:
        scope, name = local.split("#", 1)
        return scope, name
    head, _, tail = local.rpartition("/")
    return (head.replace("/", "."), tail) if head and tail else (None, local)


def _drop_annotation_tags(g: Graph) -> None:
    """Remove the triples `gen-owl` writes for a LinkML annotation tag such as `collection_key` or `simple_dict_value`.

    Their predicate is a TVB-O IRI that no property declares, and they direct the YAML dialect, which `hatch_build.py` reads from the schema, not the ontology.

    Args:
        g: The parsed T-box, mutated in place.
    """
    declared = {s for t in SLOT_TYPES for s in g.subjects(RDF.type, t)}
    for predicate in {p for p in g.predicates() if p.startswith(TVBO) and p not in declared}:
        g.remove((None, predicate, None))


def _qualify_labels(g: Graph) -> None:
    """Label every TVB-O class and property by its name as written, qualified where the bare name could repeat another term's label, and keep the bare name as its `skos:notation`.

    The qualifier answers two facts the code cannot show:

    * **Local-name collisions.** A LinkML enum's permissible value, and a slot nested under a sub-namespace, both carry a bare local name that repeats across parents — `DimensionType#time` vs `EventType#time` vs `SamplingAxis#time`, or `tvb-o/dbs/scale` vs `tvb-o/software/scale`. The parent is the IRI fragment before `#`, or the path before the final `/`, and the label becomes `"<value> (<parent>)"`; an unscoped sibling such as `tvb-o/scale` is the canonical one and keeps its bare label.
    * **Slot labels against imported classes.** A top-level TVBO property (`tvbo:x`, `tvbo:license`) shares its readable label with an InterLex or SANDS class of the same name. Only the TVBO side is qualified, to `"<name> (slot)"`; the imported class label is left untouched.

    This applies both to a term whose label is its bare name and to one `gen-owl` leaves unlabeled, as it does every attribute that more than one class declares; a top-level class keeps, or gets, its bare name.

    Args:
        g: The parsed T-box, mutated in place.
    """
    properties = {s for t in SLOT_TYPES for s in g.subjects(RDF.type, t)}
    for term in properties | set(g.subjects(RDF.type, OWL.Class)):
        if not (isinstance(term, URIRef) and term.startswith(TVBO)):
            continue
        scope, name = _split_scope(term[len(TVBO) :])
        qualifier = scope or ("slot" if term in properties else None)
        bare = Literal(name)
        labels = set(g.objects(term, RDFS.label))
        if labels and (qualifier is None or bare not in labels):
            continue
        g.remove((term, RDFS.label, bare))
        if qualifier:
            g.add((term, SKOS.notation, bare))
        g.add((term, RDFS.label, Literal(f"{name} ({qualifier})" if qualifier else name)))


def _augment(g: Graph, schema: dict) -> None:
    """Write the §2.3 ontology header into *g* and repair the labels ROBOT report flags on imported terms.

    Args:
        g: The parsed T-box, mutated in place.
        schema: The LinkML schema, read for title, version, licence and provenance.
    """
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

    if schema.get("created_on"):
        g.add((onto, DCTERMS.created, Literal(str(schema["created_on"]), datatype=XSD.date)))
    modified = schema.get("last_updated_on") or schema.get("created_on")
    if not modified:
        raise SystemExit(
            "Schema declares neither `last_updated_on` nor `created_on`, so the "
            "ontology's dcterms:modified would be the build date and the artefact "
            "would differ on every run. Declare one of them."
        )
    g.add((onto, DCTERMS.modified, Literal(str(modified), datatype=XSD.date)))

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
    _equals_string_as_one_of(g, SchemaView(str(args.schema)))
    _drop_annotation_tags(g)
    _qualify_labels(g)
    _augment(g, schema)
    write_canonical(g, args.owl)
    print(f"\u2713 Augmented ontology header in {args.owl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
