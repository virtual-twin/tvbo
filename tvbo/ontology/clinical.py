"""Where brain network models have reached patients: clinical application studies by clinical domain and by the node dynamical model each used, read from the clinical layer of the shipped merged ontology.

The clinical layer (`ontology/tvb-o-clinical.ttl`, merged into the shipped `tvbo.owl`) files every published application of a whole-brain or neural-mass model to patients as a `tvbc:ClinicalApplicationStudy` with the `tvbc:ClinicalDomain` it applies to (an ICD-11 code under its ICD-11 chapter, `skos:broader`) and the `tvbc:NodeDynamicsModel` it uses. `clinical_landscape` counts that graph with SPARQL and is shaped as a study analysis callable, `callable: {name: clinical_landscape, module: tvbo.ontology.clinical}`, so a figure binds the counts by dimension and every number grows with curation instead of being maintained beside the ontology.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from tvbo.data.registry import NAMESPACE
from tvbo.ontology.grounding import load_world
from tvbo.ontology.query import sparql_query

CLINICAL = NAMESPACE + "clinical/"
SKOS = "http://www.w3.org/2004/02/skos/core#"


def _by_entity(world, query: str) -> dict:
    """The rows of *query* keyed by the IRI of their first column, the remaining columns as strings (an entity as its IRI); an entity matched by two rows (two labels, codes or chapters) is refused rather than one of them kept by row order."""
    found: dict = {}
    for entity, *values in sparql_query(query, flatten_result=False, world=world):
        row = tuple(str(getattr(v, "iri", v)) for v in values)
        if entity.iri in found:
            raise ValueError(f"{entity.iri} is stated twice where the clinical layer states it once: {found[entity.iri]} and {row}")
        found[entity.iri] = row
    return found


def clinical_landscape(path: Path | None = None) -> dict:
    """The clinical layer counted: distinct studies per clinical domain and node dynamical model as a `(domain, model)` array, studies per domain with each domain's ICD-11 code and chapter as coordinates, and the three totals.

    Domains are ordered by ICD-11 chapter and, within a chapter, by their number of studies; models by their number of studies. A study that used two models counts once for each model and once for its domain. Each domain, chapter and model carries exactly one `skos:prefLabel` (and a domain one ICD-11 code and chapter); a second one raises.
    """
    world = load_world(path)
    links = sparql_query(f"SELECT ?s ?d ?m WHERE {{ ?s <{CLINICAL}appliesToClinicalDomain> ?d . ?s <{CLINICAL}usesNodeDynamics> ?m . }}", flatten_result=False, world=world)
    domains = _by_entity(world, f"SELECT ?d ?l ?code ?c WHERE {{ ?d a <{CLINICAL}ClinicalDomain> . ?d <{SKOS}prefLabel> ?l . ?d <{CLINICAL}icd11Code> ?code . ?d <{SKOS}broader> ?c . }}")
    chapters = _by_entity(world, f"SELECT DISTINCT ?c ?n ?l WHERE {{ ?d a <{CLINICAL}ClinicalDomain> . ?d <{SKOS}broader> ?c . ?c <{SKOS}notation> ?n . ?c <{SKOS}prefLabel> ?l . }}")
    models = {m: label for m, (label,) in _by_entity(world, f"SELECT ?m ?l WHERE {{ ?m a <{CLINICAL}NodeDynamicsModel> . ?m <{SKOS}prefLabel> ?l . }}").items()}

    studies: defaultdict = defaultdict(set)
    for s, d, m in links:
        for key in (d.iri, m.iri, (d.iri, m.iri)):
            studies[key].add(s.iri)
    chapter_rank = {c: i for i, c in enumerate(sorted(chapters, key=lambda c: chapters[c][0]))}
    domain_order = sorted(domains, key=lambda d: (chapter_rank[domains[d][2]], -len(studies[d]), domains[d][0]))
    model_order = sorted(models, key=lambda m: (-len(studies[m]), models[m]))

    counts = np.array([[len(studies[d, m]) for m in model_order] for d in domain_order])
    domain_coords = {
        "domain": [domains[d][0] for d in domain_order],
        "icd11": ("domain", [domains[d][1] for d in domain_order]),
        "chapter": ("domain", [chapters[domains[d][2]][0] for d in domain_order]),
        "chapter_label": ("domain", [chapters[domains[d][2]][1] for d in domain_order]),
    }
    return {
        "applications": xr.DataArray(counts, dims=["domain", "model"], coords={**domain_coords, "model": [models[m] for m in model_order]}),
        "studies_by_domain": xr.DataArray(np.array([len(studies[d]) for d in domain_order]), dims=["domain"], coords=domain_coords),
        "n_studies": len({s.iri for s, _, _ in links}),
        "n_domains": len(domain_order),
        "n_models": len(model_order),
    }
