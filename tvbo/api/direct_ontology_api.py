# -*- coding: utf-8 -*-
"""
Direct RDF/SPARQL-based ontology API.

Efficient access to the TVB ontology using direct owlready2 SPARQL queries.
Enables schema ↔ ontology interlinking via class URIs.

No fallbacks - fails fast on unexpected data.
"""

import os
from typing import Any, Dict, List, Optional

import owlready2 as owl
from sympy import latex, symbols

from tvbo.ontology import owl as ontology, query

# The platform serves the generated, individual-based ontology
# (tvbo/data/ontology/tvbo.owl). It is loaded into a DEDICATED owlready2 World,
# isolated from owl.py's global class-based `onto` that the tvbo core depends on
# (see dev/runtime_ontology_migration.md). owl.py's class-based high-level API
# stays on the deprecated ontology until the Phase B rewrite.
_GENERATED_ONTO_PATH = os.path.join(ontology.ONTO_DIR, "tvbo.owl")


def _load_generated_ontology():
    """Load the generated ontology into its own World (no reasoning; ROBOT
    already materialised the ELK inferences at build time)."""
    world = owl.World()
    return world.get_ontology("file://" + _GENERATED_ONTO_PATH).load()


# Type mapping: database types → generated-ontology class names (the LinkML
# schema classes; the generated ontology types domain entities against these).
TYPE_MAPPING = {
    "dynamics": "Dynamics",
    "network": "Network",
    "integrator": "Integrator",
    "coupling": "Coupling",
    "experiment": "SimulationExperiment",
    "study": "SimulationStudy",
    "observation": "Observation",
    "atlas": "Parcellation",
    "software": "SoftwarePackage",
}

CURIE_PREFIXES = {
    "tvbo:": "https://w3id.org/tvbo/",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "owl:": "http://www.w3.org/2002/07/owl#",
}


def _get_label(entity) -> str:
    """Get entity label."""
    if hasattr(entity, "label") and entity.label:
        return entity.label.first() or entity.name
    return entity.name


def _get_symbol(entity) -> str:
    """Get LaTeX symbol representation."""
    if hasattr(entity, "symbol") and entity.symbol and entity.symbol.first():
        return rf"${latex(symbols(entity.symbol.first()))}$"
    return _get_label(entity)


def _get_type(entity) -> str:
    """Primary type/class label. Entity-local so it is world-agnostic.

    A domain individual's asserted class and a class's superclass both live in
    ``is_a`` as a ThingClass, so the first non-``Thing`` entry is the natural
    type for both. Returns ``"Thing"`` when none is present.
    """
    for parent in getattr(entity, "is_a", []) or []:
        if isinstance(parent, owl.ThingClass) and parent.name != "Thing":
            return (parent.label.first() if parent.label else None) or parent.name
    return "Thing"


def _serialize_entity(entity) -> Dict[str, Any]:
    """Serialize an ontology entity to dict."""
    # Safely get optional attributes
    definition = ""
    if hasattr(entity, "definition") and entity.definition:
        definition = entity.definition.first() or ""

    description = ""
    if hasattr(entity, "description") and entity.description:
        description = entity.description.first() or ""
    elif hasattr(entity, "title") and entity.title:
        description = entity.title.first() or ""
    elif definition:
        description = definition.split(".")[0]

    is_a = [p.name for p in entity.is_a if isinstance(p, owl.ThingClass) and p.name != "Thing"]
    requires = [r.storid for r in entity.requires] if hasattr(entity, "requires") and entity.requires else []

    return {
        "id": f"onto_{entity.storid}",
        "storid": entity.storid,
        "iri": entity.iri,
        "type": "ontology",
        "ontology_type": _get_type(entity),
        "name": _get_label(entity),
        "label": _get_label(entity),
        "symbol": _get_symbol(entity),
        "title": _get_label(entity),
        "description": description,
        "definition": definition,
        "is_a": is_a,
        "requires": requires,
        "tags": [_get_type(entity)] + is_a[:2],
    }


class DirectOntologyAPI:
    """Direct RDF/SPARQL-based ontology API."""

    def __init__(self):
        self.onto = _load_generated_ontology()
        self.world = self.onto.world

    def search(self, term: str, limit: int = 100, exact_match: bool = False) -> List[Dict[str, Any]]:
        """Search ontology via SPARQL."""
        if not term:
            return []

        results = query.label_search(
            term,
            include=["synonym", "acronym", "symbol", "tvbSourceVariable"],
            exact_match=["symbol", "acronym"] if not exact_match else "all",
            onto=self.onto,
        )
        serialized = [_serialize_entity(r) for r in results]

        # label_search returns an unordered set; rank so the queried entity
        # surfaces first (exact label > prefix > substring). Necessary now that
        # the individual-based ontology adds many sub-entities (a model's
        # parameters/state variables) that also match a model-name search.
        t = term.strip().lower()

        def _rank(item):
            label = (item.get("label") or "").lower()
            if label == t:
                return 0
            if label.startswith(t):
                return 1
            if t in label:
                return 2
            return 3

        serialized.sort(key=_rank)
        return serialized[:limit]

    def get_by_storid(self, storid: int) -> Dict[str, Any]:
        """Get entity by storage ID."""
        entity = self.world._get_by_storid(storid)
        return _serialize_entity(entity)

    def get_by_iri(self, iri: str) -> Optional[Dict[str, Any]]:
        """Get entity by IRI. Returns None if not found."""
        entity = self.onto.search_one(iri=iri)
        return _serialize_entity(entity) if entity else None

    def get_by_curie(self, curie: str) -> Optional[Dict[str, Any]]:
        """Get entity by CURIE (e.g., 'tvbo:Equation')."""
        for prefix, base in CURIE_PREFIXES.items():
            if curie.startswith(prefix):
                return self.get_by_iri(base + curie[len(prefix) :])
        return self.get_by_iri(curie)

    def get_children(self, storid: int) -> Dict[str, Any]:
        """Get child classes/instances via direct RDF access."""
        entity = self.world._get_by_storid(storid)
        nodes, links = [], []
        seen = set()

        for predicate, child in query.get_children(entity, onto=self.onto):
            if child and child.storid not in seen:
                seen.add(child.storid)
                nodes.append(_serialize_entity(child))
                links.append({"source": child.storid, "target": storid, "type": predicate.replace("rdfs:subClassOf", "is_a")})

        if hasattr(entity, "requires"):
            for req in entity.requires:
                if req.storid not in seen:
                    seen.add(req.storid)
                    nodes.append(_serialize_entity(req))
                    links.append({"source": storid, "target": req.storid, "type": "requires"})

        return {"nodes": nodes, "links": links}

    def get_parents(self, storid: int) -> Dict[str, Any]:
        """Get parent classes via direct RDF access."""
        entity = self.world._get_by_storid(storid)
        nodes, links = [], []
        seen = set()

        for predicate, parent in query.get_parents(entity, onto=self.onto):
            if parent and parent.storid not in seen:
                seen.add(parent.storid)
                nodes.append(_serialize_entity(parent))
                links.append({"source": storid, "target": parent.storid, "type": predicate.replace("rdfs:subClassOf", "is_a")})

        for parent in entity.is_a:
            if isinstance(parent, owl.ThingClass) and parent.name != "Thing" and parent.storid not in seen:
                seen.add(parent.storid)
                nodes.append(_serialize_entity(parent))
                links.append({"source": storid, "target": parent.storid, "type": "is_a"})

        return {"nodes": nodes, "links": links}

    def get_relationships(self, storid: int) -> Dict[str, Any]:
        """Get all relationships (both directions)."""
        entity = self.world._get_by_storid(storid)
        center = _serialize_entity(entity)
        nodes, links = [center], []
        seen = {storid}

        for data in [self.get_children(storid), self.get_parents(storid)]:
            for node in data["nodes"]:
                if node["storid"] not in seen:
                    seen.add(node["storid"])
                    nodes.append(node)
            links.extend(data["links"])

        return {"nodes": nodes, "links": links, "center": center}

    def get_class_hierarchy(self) -> Dict[str, Any]:
        """Get the full ontology graph for visualization.

        Includes both classes (the system scaffold: Dynamics, Coupling,
        Parameter, …) and individuals (the domain entities: JansenRit, its
        parameters, …), since the generated ontology models domain entities as
        individuals. ``is_a`` links a node to its superclass (for a class) or
        its asserted type (for an individual) — both are ThingClasses. Deeper
        object-property edges (hasParameter, …) are surfaced on demand via
        ``get_children`` / ``get_parents`` rather than inlined here.
        """
        nodes, links = [], []
        seen = set()

        for entity in list(self.onto.classes()) + list(self.onto.individuals()):
            if entity.storid in seen:
                continue
            seen.add(entity.storid)
            nodes.append(_serialize_entity(entity))

            # is_a links: superclasses (classes) or asserted types (individuals)
            for parent in getattr(entity, "is_a", []) or []:
                if isinstance(parent, owl.ThingClass) and parent.name != "Thing":
                    links.append(
                        {
                            "source": entity.storid,
                            "target": parent.storid,
                            "type": "is_a",
                            "label": "is_a",
                        }
                    )

        return {"nodes": nodes, "links": links}

    def get_schema_ontology_link(self, schema_class_name: str) -> Optional[Dict[str, Any]]:
        """Get ontology concept linked to a schema class."""
        return self.get_by_iri(f"https://w3id.org/tvbo/{schema_class_name}")

    def enrich_database_item(self, item: Dict[str, Any], type_name: str) -> Dict[str, Any]:
        """Enrich database item with ontology information."""
        onto_class_name = TYPE_MAPPING.get(type_name.lower(), type_name)
        onto_info = self.get_schema_ontology_link(onto_class_name)

        if onto_info:
            item["ontology_class"] = onto_info
            item["ontology_iri"] = onto_info["iri"]
            item["ontology_definition"] = onto_info["definition"]

        name = item.get("name") or item.get("label")
        if name:
            specific = self.search(name, limit=1, exact_match=True)
            if specific:
                item["ontology_instance"] = specific[0]
                item["ontology_parents"] = specific[0]["is_a"]

        return item

    def sparql(self, query_string: str, flatten: bool = True) -> List[Any]:
        """Execute SPARQL query against the generated ontology world."""
        return query.sparql_query(query_string, flatten_result=flatten, world=self.world)


# Singleton
_api: Optional[DirectOntologyAPI] = None


def get_direct_ontology_api() -> DirectOntologyAPI:
    """Get or create singleton."""
    global _api
    if _api is None:
        _api = DirectOntologyAPI()
    return _api
