"""A-box generator for TVB-O.

Reads YAML records under `tvbo/database/` and emits Turtle with one
`owl:NamedIndividual` per database entry, typed by the corresponding
LinkML class IRI from the T-box (`ontology/tvb-o-struct.owl`).

This replaces the legacy `tvbo/data/ontology/tvb-o.owl` A-box, which had
~1300 entities wrongly typed as `owl:Class`. See PR-N audit and PR-Q
(class->individual migration superseded by automated A-box generation).

Output: `ontology/tvb-o-data.ttl`.

Scope (initial pilot): models. Each model emits:
  - the Dynamics individual itself
  - one Parameter individual per `parameters[*]`
  - one StateVariable individual per `state_variables[*]`
  - groundings as `oboInOwl:hasDbXref` on parameter/state-variable

Subsequent extensions (separate PRs) will add coupling_functions,
integrators, observation_models, networks, atlases, software, studies.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB = ROOT / "tvbo" / "database"

TVBO = Namespace("http://www.thevirtualbrain.org/tvb-o/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
SCHEMA = Namespace("http://schema.org/")
# SANDS classes are imported from openMINDS via schema/SANDS.yaml. The struct
# OWL exposes their class IRIs under the InterLex `atom` namespace (see
# default_prefix in schema/SANDS.yaml). Type atlas/coordinate-space individuals
# directly against those existing T-box classes; do NOT mint new tvbo:* classes.
ATOM = Namespace("http://uri.interlex.org/tgbugs/uris/readable/")

PREFIX_MAP = {
    "GO": "http://purl.obolibrary.org/obo/GO_",
    "CL": "http://purl.obolibrary.org/obo/CL_",
    "UBERON": "http://purl.obolibrary.org/obo/UBERON_",
    "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_",
    "MESH": "http://purl.bioontology.org/ontology/MESH/",
    "tvbo": str(TVBO),
    "atom": str(ATOM),
}


def expand_curie(curie: str) -> URIRef | None:
    if ":" not in curie:
        return None
    pfx, local = curie.split(":", 1)
    base = PREFIX_MAP.get(pfx)
    if base is None:
        return None
    return URIRef(base + local)


def safe_local(name: str) -> str:
    return name.replace(" ", "_")


def model_iri(model_name: str, slot: str | None = None) -> URIRef:
    base = TVBO[safe_local(model_name)]
    if slot is None:
        return base
    return URIRef(f"{base}/{safe_local(slot)}")


def _study_iri(citekey: str) -> URIRef:
    return URIRef(f"{TVBO}studies/{safe_local(citekey)}")


# Mirror of bib_to_studies._sanitize_citekey: lookup-side normalisation so
# `references:` strings in yaml files match the sanitised filenames in
# tvbo/database/studies/.
import re as _re
_CITEKEY_SAFE_RE = _re.compile(r"[^A-Za-z0-9_.-]")


def _norm_citekey(raw: str) -> str:
    s = raw.replace("{", "").replace("}", "")
    s = _re.sub(r"\\['`^\"~=.]?\{?([A-Za-z])\}?", r"\1", s)
    s = _CITEKEY_SAFE_RE.sub("_", s)
    return s.strip("_")


# Populated lazily by `build_graph` after scanning `studies/`. Used by
# emit_model and emit_generic_record to back-link `references:` strings.
_STUDY_CITEKEYS: set[str] = set()
# Case-folded -> canonical map so e.g. `Fitzhugh1961` resolves to the
# study individual built from `FitzHugh1961.yaml`.
_STUDY_CITEKEYS_CI: dict[str, str] = {}


def _resolve_study(citekey: str) -> str | None:
    """Return the canonical study citekey for `citekey`, or None."""
    if citekey in _STUDY_CITEKEYS:
        return citekey
    return _STUDY_CITEKEYS_CI.get(citekey.lower())


# --- Reusable emitters parametrised by parent IRI. --------------------------
# Used uniformly by emit_model and emit_generic_record so that every record
# (model, coupling, integrator, observation, ...) receives the same metadata
# coverage including groundings.

def _add_groundings(g: Graph, iri: URIRef, data: dict) -> None:
    for cur in data.get("grounding") or []:
        u = expand_curie(str(cur))
        if u is not None:
            g.add((iri, OBOINOWL.hasDbXref, u))


def _add_equation(g: Graph, iri: URIRef, data: dict) -> None:
    eq = data.get("equation")
    if not isinstance(eq, dict):
        return
    if eq.get("rhs"):
        g.add((iri, TVBO.rhs, Literal(eq["rhs"])))
    if eq.get("lhs"):
        g.add((iri, TVBO.lhs, Literal(eq["lhs"])))


def _scoped_label(parent_iri: URIRef, symbol: str, category: str) -> str:
    """Return `"<symbol> (<parent_local> <category>)"` to keep `rdfs:label`
    unique across models. Avoids ROBOT `duplicate_label` ERRORs caused by
    bare per-model symbols (e.g. `y1`, `x_i`) colliding across dynamics."""
    parent_local = str(parent_iri).rsplit("/", 1)[-1]
    return f"{symbol} ({parent_local} {category})"


def emit_parameter(g: Graph, parent_iri: URIRef, key: str, p: dict) -> None:
    iri = URIRef(f"{parent_iri}/parameters/{safe_local(key)}")
    symbol = p.get("name", key)
    g.add((iri, RDF.type, OWL.NamedIndividual))
    g.add((iri, RDF.type, TVBO.Parameter))
    g.add((iri, SKOS.notation, Literal(symbol)))
    g.add((iri, RDFS.label, Literal(_scoped_label(parent_iri, symbol, "parameter"))))
    if p.get("description"):
        g.add((iri, DCTERMS.description, Literal(p["description"])))
    if p.get("definition"):
        g.add((iri, SKOS.definition, Literal(p["definition"])))
    if "value" in p and p["value"] is not None:
        g.add((iri, SCHEMA.defaultValue, Literal(p["value"])))
    if p.get("unit"):
        g.add((iri, TVBO.unit, Literal(p["unit"])))
    _add_groundings(g, iri, p)
    g.add((parent_iri, TVBO.hasParameter, iri))


def emit_state_variable(g: Graph, parent_iri: URIRef, key: str,
                        sv: dict) -> None:
    iri = URIRef(f"{parent_iri}/state_variables/{safe_local(key)}")
    symbol = sv.get("name", key)
    g.add((iri, RDF.type, OWL.NamedIndividual))
    g.add((iri, RDF.type, TVBO.StateVariable))
    g.add((iri, SKOS.notation, Literal(symbol)))
    g.add((iri, RDFS.label, Literal(_scoped_label(parent_iri, symbol, "state variable"))))
    if sv.get("description"):
        g.add((iri, DCTERMS.description, Literal(sv["description"])))
    if sv.get("definition"):
        g.add((iri, SKOS.definition, Literal(sv["definition"])))
    _add_equation(g, iri, sv)
    _add_groundings(g, iri, sv)
    g.add((parent_iri, TVBO.hasStateVariable, iri))


def emit_derived_variable(g: Graph, parent_iri: URIRef, key: str,
                          dv: dict) -> None:
    iri = URIRef(f"{parent_iri}/derived_variables/{safe_local(key)}")
    symbol = dv.get("name", key)
    g.add((iri, RDF.type, OWL.NamedIndividual))
    g.add((iri, RDF.type, TVBO.DerivedVariable))
    g.add((iri, SKOS.notation, Literal(symbol)))
    g.add((iri, RDFS.label, Literal(_scoped_label(parent_iri, symbol, "derived variable"))))
    if dv.get("description"):
        g.add((iri, DCTERMS.description, Literal(dv["description"])))
    _add_equation(g, iri, dv)
    _add_groundings(g, iri, dv)
    g.add((parent_iri, TVBO.hasDerivedVariable, iri))


def emit_model(g: Graph, path: pathlib.Path) -> None:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "name" not in data:
        return
    name = data["name"]
    iri = model_iri(name)
    g.add((iri, RDF.type, OWL.NamedIndividual))
    g.add((iri, RDF.type, TVBO.Dynamics))
    g.add((iri, RDFS.label, Literal(name)))
    if data.get("description"):
        g.add((iri, DCTERMS.description, Literal(data["description"])))
    if data.get("model_type"):
        g.add((iri, TVBO.model_type, Literal(data["model_type"])))
    for key, p in (data.get("parameters") or {}).items():
        if isinstance(p, dict):
            emit_parameter(g, iri, key, p)
    for key, sv in (data.get("state_variables") or {}).items():
        if isinstance(sv, dict):
            emit_state_variable(g, iri, key, sv)
    for key, dv in (data.get("derived_variables") or {}).items():
        if isinstance(dv, dict):
            emit_derived_variable(g, iri, key, dv)
    # Cross-link references to study individuals when citekeys match.
    for ref in data.get("references") or []:
        if isinstance(ref, str):
            nk = _resolve_study(_norm_citekey(ref))
            if nk:
                g.add((iri, DCTERMS.references, _study_iri(nk)))


# --- Generic per-folder emitter for non-model database categories. -----------
# Mapping: folder name -> class IRI. SANDS classes (BrainAtlas,
# CommonCoordinateSpace) reuse their existing openMINDS/InterLex IRIs from
# schema/SANDS.yaml; do not mint tvbo:Atlas / tvbo:CoordinateSpace.
GENERIC_FOLDER_TYPES: dict[str, URIRef] = {
    "coupling_functions": TVBO.Coupling,
    "integrators": TVBO.Integrator,
    "networks": TVBO.Network,
    "software": TVBO.SoftwarePackage,
    "experiments": TVBO.SimulationExperiment,
    "continuations": TVBO.Continuation,
    "observation_models": TVBO.Observation,
    "atlases": ATOM.BrainAtlas,
    "coordinate_spaces": ATOM.CommonCoordinateSpace,
    # studies/: one yaml-per-study, generated by
    # `dev/OntologicalRestructuring/tools/bib_to_studies.py` from the
    # bibtex bibliographies. Typed against schema.org rather than minting
    # a new tvbo class. Specific subtypes (Book / Thesis / etc.) are
    # refined per-record in `_refine_study_type`.
    "studies": SCHEMA.ScholarlyArticle,
}


# bibtex/study `type:` -> schema.org subclass for individual typing.
STUDY_TYPE_MAP: dict[str, URIRef] = {
    "article": SCHEMA.ScholarlyArticle,
    "conference_paper": SCHEMA.ScholarlyArticle,
    "book": SCHEMA.Book,
    "book_chapter": SCHEMA.Chapter,
    "thesis": SCHEMA.Thesis,
    "technical_report": SCHEMA.Report,
    "preprint": SCHEMA.ScholarlyArticle,
    "misc": SCHEMA.CreativeWork,
}


def _record_label(data: dict, fallback: str) -> str:
    return str(data.get("name") or data.get("label") or fallback)


# Folders whose YAML `name:` field is a generic family identifier
# (e.g. all Schaefer2018 atlases share `name: Schaefer2018`). For these
# we use the BIDS-style filename stem as the unique `rdfs:label` and keep
# `name:` as a `skos:altLabel`. Avoids ROBOT `duplicate_label` ERRORs.
_FILENAME_LABEL_FOLDERS = {"atlases", "networks", "coordinate_spaces"}


def emit_generic_record(g: Graph, folder: str, cls_iri: URIRef,
                        path: pathlib.Path) -> URIRef | None:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return None
    label = _record_label(data, path.stem)
    iri = URIRef(f"{TVBO}{folder}/{safe_local(path.stem)}")
    g.add((iri, RDF.type, OWL.NamedIndividual))
    # Studies refine their schema.org subtype based on the bibtex entry type.
    if folder == "studies":
        cls_iri = STUDY_TYPE_MAP.get(str(data.get("type") or ""), cls_iri)
    g.add((iri, RDF.type, cls_iri))
    if folder in _FILENAME_LABEL_FOLDERS:
        g.add((iri, RDFS.label, Literal(path.stem)))
        if label and label != path.stem:
            g.add((iri, SKOS.altLabel, Literal(label)))
    else:
        g.add((iri, RDFS.label, Literal(label)))
    if data.get("description"):
        g.add((iri, DCTERMS.description, Literal(str(data["description"]))))
    # Common scalar metadata on software / experiments / coord-spaces / studies.
    for slot, prop in (
        ("homepage", SCHEMA.url),
        ("url", SCHEMA.url),
        ("repository", SCHEMA.codeRepository),
        ("doi", SCHEMA.identifier),
        ("license", DCTERMS.license),
        ("abbreviation", SKOS.altLabel),
        ("nativeUnit", TVBO.unit),
        ("anatomicalAxesOrientation", TVBO.anatomicalAxesOrientation),
        ("axesOrigin", TVBO.axesOrigin),
        ("title", DCTERMS.title),
        ("year", DCTERMS.issued),
        ("journal", SCHEMA.isPartOf),
        ("booktitle", SCHEMA.isPartOf),
        ("publisher", SCHEMA.publisher),
        ("volume", SCHEMA.volumeNumber),
        ("number", SCHEMA.issueNumber),
        ("pages", SCHEMA.pagination),
    ):
        if data.get(slot):
            g.add((iri, prop, Literal(str(data[slot]))))
    for alt in data.get("alternateName") or []:
        g.add((iri, SKOS.altLabel, Literal(str(alt))))
    for author in data.get("authors") or []:
        g.add((iri, SCHEMA.author, Literal(str(author))))
    _add_groundings(g, iri, data)
    # Cross-link: any string in `references:` that matches a study citekey
    # becomes a `dcterms:references` link to the corresponding study IRI.
    for ref in data.get("references") or []:
        if isinstance(ref, str):
            nk = _resolve_study(_norm_citekey(ref))
            if nk:
                g.add((iri, DCTERMS.references, _study_iri(nk)))
    rp = data.get("reference_publication")
    if isinstance(rp, str):
        nk = _resolve_study(_norm_citekey(rp))
        if nk:
            g.add((iri, DCTERMS.references, _study_iri(nk)))
    # Reuse the model emitters for nested parameters / state_variables /
    # derived_variables so coverage is identical across categories.
    for key, p in (data.get("parameters") or {}).items():
        if isinstance(p, dict):
            emit_parameter(g, iri, key, p)
    for key, sv in (data.get("state_variables") or {}).items():
        if isinstance(sv, dict):
            emit_state_variable(g, iri, key, sv)
    for key, dv in (data.get("derived_variables") or {}).items():
        if isinstance(dv, dict):
            emit_derived_variable(g, iri, key, dv)
    return iri


def build_graph() -> Graph:
    g = Graph()
    g.bind("tvbo", TVBO)
    g.bind("owl", OWL)
    g.bind("dcterms", DCTERMS)
    g.bind("oboInOwl", OBOINOWL)
    g.bind("schema", SCHEMA)
    g.bind("skos", SKOS)
    for pfx, base in PREFIX_MAP.items():
        g.bind(pfx, base)

    onto = URIRef("https://w3id.org/tvbo/data")
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, DCTERMS.title, Literal("TVB-O A-box (database individuals)")))
    g.add((onto, DCTERMS.description, Literal(
        "Generated A-box: one owl:NamedIndividual per YAML database entry. "
        "Companion to ontology/tvb-o-struct.owl (T-box) and "
        "ontology/tvb-o-axioms.ttl (axioms). Replaces the legacy "
        "tvbo/data/ontology/tvb-o.owl A-box."
    )))
    g.add((onto, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    g.add((onto, RDFS.seeAlso, URIRef("https://w3id.org/tvbo/struct")))
    g.add((onto, RDFS.seeAlso, URIRef("https://w3id.org/tvbo/axioms")))

    # Pre-scan studies/ so emitters can resolve `references:` strings to
    # study individual IRIs before emission. Order-independent.
    _STUDY_CITEKEYS.clear()
    _STUDY_CITEKEYS_CI.clear()
    for sp in (DB / "studies").glob("*.yaml"):
        _STUDY_CITEKEYS.add(sp.stem)
        _STUDY_CITEKEYS_CI[sp.stem.lower()] = sp.stem

    for path in sorted((DB / "models").glob("*.yaml")):
        emit_model(g, path)
    for folder, cls_iri in GENERIC_FOLDER_TYPES.items():
        for path in sorted((DB / folder).glob("*.yaml")):
            emit_generic_record(g, folder, cls_iri, path)
    return g
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=str(ROOT / "ontology" / "tvb-o-data.ttl"))
    args = ap.parse_args()

    g = build_graph()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    n_inds = sum(1 for _ in g.subjects(RDF.type, OWL.NamedIndividual))
    n_models = sum(1 for _ in g.subjects(RDF.type, TVBO.Dynamics))
    n_params = sum(1 for _ in g.subjects(RDF.type, TVBO.Parameter))
    n_svs = sum(1 for _ in g.subjects(RDF.type, TVBO.StateVariable))
    print(f"Wrote {out}")
    print(f"  triples:           {len(g)}")
    print(f"  NamedIndividuals:  {n_inds}")
    print(f"  Dynamics models:   {n_models}")
    print(f"  Parameters:        {n_params}")
    print(f"  StateVariables:    {n_svs}")
    n_dv = sum(1 for _ in g.subjects(RDF.type, TVBO.DerivedVariable))
    print(f"  DerivedVariables:  {n_dv}")
    for folder, cls_iri in GENERIC_FOLDER_TYPES.items():
        n = sum(1 for _ in g.subjects(RDF.type, cls_iri))
        local = str(cls_iri).rsplit("/", 1)[-1]
        print(f"  {local + ' (' + folder + ')':<36}{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
