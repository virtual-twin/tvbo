"""NeuroML-core ingestion generator for TVB-O.

Walks the NeuroML2 core ``ComponentType`` definitions (the LEMS type system bundled inside the jNeuroML jar) and emits two artifacts from one pass:

- ``ontology/tvb-o-neuroml.ttl`` — the semantic module merged into ``tvbo.owl``.
  One ``owl:Class`` per ``ComponentType`` under the tvbo-scoped namespace
  ``https://w3id.org/tvbo/neuroml/``; ``extends`` becomes ``rdfs:subClassOf``;
  each class carries ``skos:exactMatch`` to its direct NeuroML IRI (``http://www.neuroml.org/schema/neuroml2#<name>``) so both identifiers denote
  the same type; local exposures / requirements / event ports / parameters are recorded as annotations. This is the "tvbo.owl references NeuroML-core" link,
  alongside the GO bridge.

- ``tvbo/data/ontology/neuroml_contracts.json`` — the compiled, ``extends``-
  accumulated contract index the NeuroML adapter loads at emit time (stdlib
  ``json``, no owlready2/rdflib/pylems on the hot path). It is a projection of the same ingested data: for each type, the full inherited set of exposures,
  requirements, parameters, event ports, attachments, children, and on-start assignments. This is what grounds the adapter's base-type contracts.

Both outputs are checked in and CI-guarded against drift, exactly like the other generated ontology artifacts. Regenerate with ``make gen-neuroml`` (or directly)
whenever the bundled jNeuroML version changes.

The RDF/Dublin-Core/XML metadata plumbing types that the core files pull in for
``<notes>``/``<annotation>`` handling (``rdf_*``, ``dc_*``, ``notes``, ...) are not neuroscience component types and are filtered out; the count of skipped
types is logged so the filtering is never silent.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import tempfile
import zipfile

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

ROOT = pathlib.Path(__file__).resolve().parents[2]

TVBO = Namespace("https://w3id.org/tvbo/")
NML = Namespace("https://w3id.org/tvbo/neuroml/")
NEUROML2 = Namespace("http://www.neuroml.org/schema/neuroml2#")
NML_ONT = URIRef("https://w3id.org/tvbo/neuroml")

DEFAULT_TTL = ROOT / "ontology" / "tvb-o-neuroml.ttl"
DEFAULT_CONTRACTS = ROOT / "tvbo" / "data" / "ontology" / "neuroml_contracts.json"

# RDF / Dublin-Core / XML metadata infrastructure the core files include for
# <notes>/<annotation> handling. These are not NeuroML neuroscience types.
_PLUMBING_RE = re.compile(r"^(rdf|dc|dcterms|bibtex|sbml|xsd|lems)_")
_PLUMBING_NAMES = {"notes", "annotation", "property", "baseAnnotation_without_ns"}


def _is_plumbing(name: str) -> bool:
    """True for RDF/Dublin-Core/XML metadata types (skipped from the ontology)."""
    return bool(_PLUMBING_RE.match(name)) or name in _PLUMBING_NAMES


def locate_core_types(dest: str) -> str:
    """Extract ``NeuroML2CoreTypes/*.xml`` from the bundled jNeuroML jar.

    Locates the jar inside the installed ``pyneuroml`` package so the version is never hard-coded, and unpacks the core type XML into *dest*.

    Args:
        dest: Directory to extract into.

    Returns:
        Path to the extracted ``NeuroML2CoreTypes`` directory.
    """
    import pyneuroml

    libdir = os.path.join(os.path.dirname(pyneuroml.__file__), "lib")
    jars = glob.glob(os.path.join(libdir, "jNeuroML-*-jar-with-dependencies.jar"))
    if not jars:
        raise FileNotFoundError(
            f"No jNeuroML-*-jar-with-dependencies.jar under {libdir}. Install the neuroml extra: pip install tvbo[neuroml]."
        )

    def _version(path):
        """Numeric version tuple, so 0.14.0 sorts above 0.9.0 (lexically it does not)."""
        m = re.search(r"jNeuroML-([0-9]+(?:\.[0-9]+)*)", os.path.basename(path))
        return tuple(int(p) for p in m.group(1).split(".")) if m else ()

    jar = max(jars, key=_version)
    with zipfile.ZipFile(jar) as z:
        members = [n for n in z.namelist() if n.startswith("NeuroML2CoreTypes/") and n.endswith(".xml")]
        z.extractall(dest, members=members)
    core = os.path.join(dest, "NeuroML2CoreTypes")
    print(f"  extracted {len(members)} core XML files from {os.path.basename(jar)}", file=sys.stderr)
    return core


def load_model(core_dir: str):
    """Load every core-type XML into one pylems ``Model`` (includes resolved)."""
    from lems.model.model import Model

    model = Model(include_includes=True)
    for fname in sorted(os.listdir(core_dir)):
        if fname.endswith(".xml"):
            model.import_from_file(os.path.join(core_dir, fname))
    return model


def _chain(component_types, name):
    """The ``extends`` chain child→…→root as a list of ComponentTypes."""
    seq = []
    seen = set()
    while name and name in component_types and name not in seen:
        ct = component_types[name]
        seq.append(ct)
        seen.add(name)
        name = ct.extends
    return seq


def _accumulate(component_types, name):
    """Accumulate every inherited slot for *name* up its ``extends`` chain.

    Walks root→child so a nearer type overrides a farther one, and drops the metadata-plumbing children (``notes``/``annotation``/``property``).

    Returns:
        A contract dict: ``extends``, ``chain``, and the accumulated
        ``exposures`` / ``requirements`` / ``parameters`` (name→dimension),
        ``event_ports`` (name→direction), ``attachments`` (name→type),
        ``children`` (name→{type, multiple}), ``component_references``
        (name→type), and ``on_start`` (var→value).
    """
    exposures, requirements, parameters = {}, {}, {}
    event_ports, attachments, children, on_start = {}, {}, {}, {}
    component_references = {}
    for ct in reversed(_chain(component_types, name)):
        for e in ct.exposures:
            exposures[e.name] = e.dimension
        for r in ct.requirements:
            requirements[r.name] = r.dimension
        for p in ct.parameters:
            parameters[p.name] = p.dimension
        for ev in ct.event_ports:
            event_ports[ev.name] = ev.direction
        for a in ct.attachments:
            attachments[a.name] = getattr(a, "type", None)
        # A ComponentReference names another component by id; a Child nests one.
        for cr in (getattr(ct, "component_references", None) or {}).values():
            component_references[cr.name] = getattr(cr, "type", None)
        for ch in ct.children:
            if _is_plumbing(ch.name):
                continue
            children[ch.name] = {
                "type": getattr(ch, "type", None),
                "multiple": bool(getattr(ch, "multiple", False)),
            }
        dyn = getattr(ct, "dynamics", None)
        for os_block in getattr(dyn, "on_starts", []) or []:
            for sa in getattr(os_block, "state_assignments", []) or []:
                on_start[sa.variable] = sa.value
    self_ct = component_types[name]
    return {
        "extends": self_ct.extends,
        "chain": [c.name for c in _chain(component_types, name)],
        "exposures": exposures,
        "requirements": requirements,
        "parameters": parameters,
        "event_ports": event_ports,
        "attachments": attachments,
        "children": children,
        "component_references": component_references,
        "on_start": on_start,
    }


# Annotation properties recording each type's locally-declared LEMS slots.
_ANNOT = {
    "exposes": (NML.exposes, "exposes", "A quantity this NeuroML ComponentType exposes."),
    "requires": (NML.requires, "requires", "A quantity this NeuroML ComponentType requires from its context."),
    "eventPort": (NML.eventPort, "event port", "An event port declared by this NeuroML ComponentType."),
    "hasParameter": (NML.hasParameter, "has parameter", "A parameter declared by this NeuroML ComponentType."),
}


def build_ttl(component_types, domain_names) -> Graph:
    """Build the ``tvb-o-neuroml.ttl`` graph (classes + subClassOf + cross-ref).

    Records local (not accumulated) exposures/requirements/event-ports/parameters as annotations; inheritance is carried by ``rdfs:subClassOf`` for a reasoner
    to accumulate. The compiled JSON contract carries the accumulated form.
    """
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("tvbo", TVBO)
    g.bind("nml", NML)
    g.bind("neuroml2", NEUROML2)

    g.add((NML_ONT, RDF.type, OWL.Ontology))
    g.add((NML_ONT, DCTERMS.title, Literal("TVB-O NeuroML-core reference", lang="en")))
    g.add(
        (
            NML_ONT,
            DCTERMS.description,
            Literal(
                "OWL rendering of the NeuroML2 core LEMS ComponentType hierarchy: one class "
                "per ComponentType, extends as rdfs:subClassOf, cross-referenced to the "
                "canonical NeuroML type via skos:exactMatch.",
                lang="en",
            ),
        )
    )
    g.add((NML_ONT, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    g.add((NML_ONT, RDFS.seeAlso, URIRef("http://www.neuroml.org/schema/neuroml2")))

    for prop, label, comment in _ANNOT.values():
        g.add((prop, RDF.type, OWL.AnnotationProperty))
        g.add((prop, RDFS.label, Literal(label, lang="en")))
        g.add((prop, RDFS.comment, Literal(comment, lang="en")))

    for name in sorted(domain_names):
        ct = component_types[name]
        cls = NML[name]
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(name, lang="en")))
        g.add((cls, RDFS.isDefinedBy, NML_ONT))
        g.add((cls, SKOS.exactMatch, NEUROML2[name]))
        if ct.extends and ct.extends in domain_names:
            g.add((cls, RDFS.subClassOf, NML[ct.extends]))
        if getattr(ct, "description", None):
            g.add((cls, DCTERMS.description, Literal(str(ct.description), lang="en")))
        for e in ct.exposures:
            g.add((cls, NML.exposes, Literal(e.name)))
        for r in ct.requirements:
            g.add((cls, NML.requires, Literal(r.name)))
        for ev in ct.event_ports:
            g.add((cls, NML.eventPort, Literal(ev.name)))
        for p in ct.parameters:
            g.add((cls, NML.hasParameter, Literal(p.name)))
    return g


def build_contracts(component_types, domain_names) -> dict:
    """Compile the accumulated contract index for every domain type."""
    return {name: _accumulate(component_types, name) for name in sorted(domain_names)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the TVB-O NeuroML-core ontology module + contract index.")
    ap.add_argument("-o", "--output", default=str(DEFAULT_TTL), help="Turtle module output path.")
    ap.add_argument("--contracts", default=str(DEFAULT_CONTRACTS), help="JSON contract index output path.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        core = locate_core_types(tmp)
        model = load_model(core)

    all_types = dict(model.component_types)

    def _is_domain(name: str) -> bool:
        """Domain type unless it or any ``extends`` ancestor is metadata plumbing.

        The BioModels-qualifier and RDF annotation types (``bqbiol_*``,
        ``bqmodel_*``, ``rdfs_seeAlso``, ...) descend from ``baseAnnotation_*``, so the check must walk the chain, not just the type's own name.
        """
        return not any(_is_plumbing(ct.name) for ct in _chain(all_types, name))

    domain_names = {n for n in all_types if _is_domain(n)}
    skipped = sorted(n for n in all_types if not _is_domain(n))
    print(
        f"  {len(all_types)} ComponentTypes total; {len(domain_names)} domain, "
        f"{len(skipped)} plumbing skipped: {', '.join(skipped[:8])}"
        f"{' …' if len(skipped) > 8 else ''}",
        file=sys.stderr,
    )

    g = build_ttl(all_types, domain_names)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    print(f"✓ {len(domain_names)} classes → {out}", file=sys.stderr)

    contracts = build_contracts(all_types, domain_names)
    cout = pathlib.Path(args.contracts)
    cout.parent.mkdir(parents=True, exist_ok=True)
    cout.write_text(json.dumps(contracts, indent=2, sort_keys=True) + "\n")
    print(f"✓ {len(contracts)} contracts → {cout}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
