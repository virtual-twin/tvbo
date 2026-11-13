"""How far the biological grounding reaches: links from TVB-O entities to Gene Ontology terms, counted from the merged ontology the package ships.

A grounding link is any statement whose object is a GO term: `tvbo:surrogate_of` and its refinements, the legacy `oboInOwl:hasDbXref` annotation the same pairs still carry, and `skos:broadMatch` for a concept-level alignment. A pair of entity and term is counted once however many predicates state it, and every count is read from the graph rather than maintained beside it, so the numbers grow with curation. `go_grounding` is shaped as a study analysis callable: `callable: {name: go_grounding, module: tvbo.ontology.grounding}` persists labelled arrays a figure's bar panel can name by dimension.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr

from tvbo.ontology.owl import ONTO_DIR
from tvbo.ontology.query import sparql_query

GO_PREFIX = "http://purl.obolibrary.org/obo/GO_"
CONCEPT_PREDICATE = "http://www.w3.org/2004/02/skos/core#broadMatch"
DOMAINS = ("Parameter", "StateVariable", "DerivedVariable", "Dynamics", "Concept")
"""The kinds of entity a grounding can hang on, in the order a figure lists them."""


def merged_ontology_path() -> Path:
    """The GO-grounded ontology artifact shipped with the package."""
    return Path(ONTO_DIR) / "tvbo.owl"


@cache
def load_world(path: Path | None = None):
    """The ontology at *path* (the shipped merged artifact by default) in an owlready2 world of its own, so the runtime ontology is left untouched.

    The world is parsed once per *path* and shared by every caller, so a caller only reads it.
    """
    import owlready2

    world = owlready2.World()
    world.get_ontology(str(path or merged_ontology_path())).load()
    return world


def go_links(world) -> list[tuple[str, str, str]]:
    """Every distinct (entity, GO term, kind) link in *world*, kind being `grounding` or `concept`.

    A pair stated both by `skos:broadMatch` and by a grounding predicate is a `grounding` link, whatever order the rows arrive in.
    """
    rows = sparql_query("SELECT DISTINCT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(STRSTARTS(STR(?o), '" + GO_PREFIX + "')) }", flatten_result=False, world=world)
    links: dict[tuple[str, str], str] = {}
    for subject, predicate, term in rows:
        key = (subject.iri, str(getattr(term, "iri", term)))
        if getattr(predicate, "iri", "") == CONCEPT_PREDICATE:
            links.setdefault(key, "concept")
        else:
            links[key] = "grounding"
    return [(s, o, kind) for (s, o), kind in links.items()]


def _domain(world, iri: str) -> str:
    """The entity's kind among `DOMAINS`, read from its `rdf:type`, or `Other`."""
    entity = world[iri]
    names = {getattr(t, "name", str(t)) for t in getattr(entity, "is_a", [])}
    return next((d for d in DOMAINS if d in names), "Other")


def go_grounding(path: Path | None = None) -> dict:
    """The grounding counted: links and grounded entities per domain as labelled arrays over `domain`, and the five totals the manuscript's `gogrounding` analysis reports under the same names."""
    world = load_world(path)
    links = go_links(world)
    domain_of = {s: _domain(world, s) for s, _, _ in links}
    domains = list(DOMAINS)
    by_domain = np.array([sum(1 for s, _, _ in links if domain_of[s] == d) for d in domains])
    entities_by_domain = np.array([len({s for s, _, _ in links if domain_of[s] == d}) for d in domains])
    n_concept = sum(1 for _, _, kind in links if kind == "concept")
    return {
        "links_by_domain": xr.DataArray(by_domain, dims=["domain"], coords={"domain": domains}),
        "entities_by_domain": xr.DataArray(entities_by_domain, dims=["domain"], coords={"domain": domains}),
        "go_n_links": len(links),
        "go_n_tvb_classes": len(domain_of),
        "go_n_go_terms": len({o for _, o, _ in links}),
        "go_n_grounding_links": len(links) - n_concept,
        "go_n_concept_links": n_concept,
    }
