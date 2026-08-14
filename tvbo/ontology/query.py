#
# Module: query.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""SPARQL-based query helpers for the TVBO ontology.

This module provides thin wrappers around owlready2's SPARQL engine and the low-level triple store to look up ontology classes and individuals by label,
synonym, acronym or symbol, traverse relationships (parents and children) and normalise IRIs to their prefixed form.
"""

from typing import Any, List, Tuple, Union
from tvbo.ontology import owl as ontology
import owlready2

prefixes = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/2002/07/owl#": "owl:",
    "https://w3id.org/tvbo/": "tvbo:",
}


def iri2prefix(iri: str) -> str:
    """Abbreviate a full IRI to its prefixed (CURIE-like) form.

    Each known namespace base (`rdf`, `rdfs`, `owl`, `tvbo`) is replaced by its short prefix; an IRI with no matching namespace is returned unchanged.

    Args:
        iri: The absolute IRI to abbreviate.

    Returns:
        The IRI with any known namespace base replaced by its prefix.
    """
    for base, prefix in prefixes.items():
        iri = iri.replace(base, prefix)
    return iri


def convert_greek_to_latin(text: str) -> str:
    """
    Converts Greek letters and the micro sign (µ) in the input text to their corresponding Latin names.

    Args:
    text (str): The input text that may contain Greek letters or the micro sign.

    Returns:
    str: The text with all Greek letters and the micro sign replaced by their Latin names.
    """
    greek_to_latin = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "µ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "ο": "omicron",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "υ": "upsilon",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
        "Α": "Alpha",
        "Β": "Beta",
        "Γ": "Gamma",
        "Δ": "Delta",
        "Ε": "Epsilon",
        "Ζ": "Zeta",
        "Η": "Eta",
        "Θ": "Theta",
        "Ι": "Iota",
        "Κ": "Kappa",
        "Λ": "Lambda",
        "Μ": "Mu",
        "Ν": "Nu",
        "Ξ": "Xi",
        "Ο": "Omicron",
        "Π": "Pi",
        "Ρ": "Rho",
        "Σ": "Sigma",
        "Τ": "Tau",
        "Υ": "Upsilon",
        "Φ": "Phi",
        "Χ": "Chi",
        "Ψ": "Psi",
        "Ω": "Omega",
    }

    return "".join(greek_to_latin.get(char, char) for char in text)


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Recursively flattens a list of lists.

    Args:
        nested_list (list): A list that may contain nested lists.

    Returns:
        list: A flattened list with all elements from nested lists.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def sparql_query(query_string: str, flatten_result: bool = True, world: Any = None) -> List[Any]:
    """Run a SPARQL query against an ontology world and collect the results.

    Undefined entities are tolerated (`error_on_undefined_entities=False`) so that optional clauses referencing annotation properties absent from the
    generated ontology match nothing instead of raising.

    Args:
        query_string: The SPARQL query to execute.
        flatten_result: If `True`, recursively flatten the result rows into a
            single flat list; otherwise return the raw rows.
        world: An owlready2 `World` to query. Defaults to the global runtime
            ontology's world when `None`.

    Returns:
        The query results, flattened into a single list when `flatten_result`
        is `True`, otherwise the raw list of result rows.
    """
    # ``world`` lets callers query an ontology other than the global default (e.g. the platform's generated individual-based ontology loaded in its own owlready2 World); defaults to the class-based runtime ontology.
    # error_on_undefined_entities=False: optional clauses may reference annotation properties (e.g. tvbo:synonym) that are absent from the generated ontology; treat those as matching nothing rather than raising.
    world = world if world is not None else ontology.onto.world
    res: List[Any] = list(world.sparql(query_string, error_on_undefined_entities=False))
    return flatten_list(res) if flatten_result else res


def _search_by_label(label: str) -> List[Any]:
    """
    Search for a term in the ontology
    Args:
        label: string with term to be searched
    Returns:
        list: list of all nodes containing the search term in their label/definition
    """
    sparql_string = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX tvbo: <https://w3id.org/tvbo/>

    SELECT ?class
    WHERE {{
        ?class a owl:Class ;
            rdfs:label ?label .
        FILTER (str(?label) = "{label}")
    }}
    """
    print(sparql_string)
    return sparql_query(sparql_string)


def get_class_relationships(class_iri: Union[str, Any]) -> List[Tuple[Any, Any]]:
    """Return all direct predicate-object pairs for an ontology class.

    Args:
        class_iri: Either the class IRI as a string, or an owlready2 entity
            whose `iri` attribute is used.

    Returns:
        A list of `(predicate, object)` rows for every triple whose subject is
        the given class.
    """
    if not isinstance(class_iri, str):
        class_iri = class_iri.iri

    return sparql_query(
        f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX tvbo: <https://w3id.org/tvbo/>

        SELECT ?p ?object
        WHERE {{
            <{class_iri}> ?p ?object .
        }}
        """,
        flatten_result=False,
    )


def instance_class_relationship(subject_iri: str, predicate: str = "prov:used") -> List[Tuple[Any, Any]]:
    """Return classes linked to a subject through an OWL restriction.

    Follows `owl:Restriction` nodes attached to the subject and returns the classes referenced by their `owl:someValuesFrom`, optionally constrained to
    restrictions on a given `owl:onProperty`.

    Args:
        subject_iri: IRI of the subject class or individual to inspect.
        predicate: Prefixed property (e.g. `prov:used`) that the restriction
            must be `owl:onProperty` of; an empty string removes this
            constraint.

    Returns:
        A list of `(predicate, object)` rows describing the restriction
        property and the class it points to.
    """
    predicate_restriction = f"?restriction owl:onProperty {predicate} ." if predicate else ""

    query_string = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX tvbo: <https://w3id.org/tvbo/>

    SELECT ?p ?object
    WHERE {{
    <{subject_iri}> ?p ?restriction .
    ?restriction a owl:Restriction .
    {predicate_restriction}
    ?restriction owl:someValuesFrom ?object .
    }}
    """
    return sparql_query(
        query_string,
        flatten_result=False,
    )


def _label_search(label: str) -> List[Any]:
    label.replace("$", "")
    sparql_string = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX tvbo: <https://w3id.org/tvbo/>

    SELECT ?subject
    WHERE {{
    ?subject a owl:Class .
    OPTIONAL {{ ?subject rdfs:label ?label . }}
    OPTIONAL {{ ?subject tvbo:synonym ?synonym . }}
    OPTIONAL {{ ?subject tvbo:acronym ?acronym . }}
    OPTIONAL {{ ?subject tvbo:symbol ?symbol . }}

    FILTER (
        (BOUND(?label) && CONTAINS(LCASE(?label), "{label.lower()}")) ||
        (BOUND(?synonym) && LCASE(?synonym) = "{label.lower()}") ||
        (BOUND(?acronym) && CONTAINS(LCASE(?acronym), "{label.lower()}")) ||
        (BOUND(?symbol) && LCASE(?symbol) = "{label.lower().replace("$", "")}")
    )
    }}
    """
    return sparql_query(sparql_string)


def build_filter(label: str, field: str, exact: bool, case_sensitive: bool) -> str:
    """Build a single SPARQL `FILTER` clause matching a variable against a label.

    The clause guards the variable with `BOUND` and compares it to `label` using either equality (exact) or `CONTAINS` (substring), optionally
    lowercasing both sides for case-insensitive matching.

    Args:
        label: The search term to match against.
        field: Name of the SPARQL variable (without the leading `?`) to test.
        exact: If `True`, require an exact match; otherwise match a substring.
        case_sensitive: If `False`, compare using `LCASE` on both operands.

    Returns:
        A SPARQL boolean expression suitable for use inside a `FILTER`.
    """
    if case_sensitive:
        if exact:
            return f'(BOUND(?{field}) && ?{field} = "{label}")'
        else:
            return f'(BOUND(?{field}) && CONTAINS(?{field}, "{label}"))'
    else:
        if exact:
            return f'(BOUND(?{field}) && LCASE(?{field}) = "{label.lower()}")'
        else:
            return f'(BOUND(?{field}) && CONTAINS(LCASE(?{field}), "{label.lower()}"))'


def label_search(
    label: str,
    include: List[str] = ["synonym", "acronym", "symbol", "tvbSourceVariable"],
    exact_match: Union[str, List[str]] = [
        "symbol",
        "acronym",
        "synonym",
        "tvbSourceVariable",
    ],
    case_sensitive: bool = False,
    root_class: Any = None,
    greek_to_latin: bool = True,
    ignore_underscore: bool = False,
    types: List[str] = ["owl:Class", "owl:NamedIndividual"],
    onto: Any = None,
) -> List[owlready2.ThingClass]:
    """Search the ontology for entities matching a label across several fields.

    Builds a SPARQL query that tests `rdfs:label`, `skos:altLabel` and each included annotation property (e.g. `synonym`, `acronym`, `symbol`) against
    the search term, then returns the matching classes and/or individuals.
    Optionally restricts the results to descendants of a given root class.

    Args:
        label: The term to search for.
        include: Extra annotation fields to match on. Bare names are prefixed
            with `tvbo:`; values containing `:` are used as-is.
        exact_match: Field name, list of field names, or `"all"` for which
            matching must be exact rather than substring-based.
        case_sensitive: Whether comparisons are case sensitive.
        root_class: If given, keep only results that are descendants of this
            class (an entity or a label string resolved via `search_one`).
        greek_to_latin: If `True`, transliterate Greek letters in `label` to
            their Latin names before searching.
        ignore_underscore: If `True`, strip underscores from `label`.
        types: The RDF types to restrict subjects to (e.g. `owl:Class`,
            `owl:NamedIndividual`).
        onto: The ontology to query. Defaults to the global runtime ontology.

    Returns:
        A de-duplicated list of matching ontology entities, optionally filtered
        to descendants of `root_class`.
    """
    if greek_to_latin:
        label = convert_greek_to_latin(label)
    if ignore_underscore:
        label = label.replace("_", "")

    label = label.replace("$", "")
    exact_match = [exact_match] if isinstance(exact_match, str) else exact_match

    optional_clauses = []
    filters = []

    def add_clause_and_filter(field, field_name):
        """Append an OPTIONAL clause and its matching filter for one field."""
        optional_clauses.append(f"OPTIONAL {{ ?subject {field} ?{field_name} . }}")
        filters.append(
            build_filter(
                label,
                field_name,
                "all" in exact_match or field_name in exact_match,
                case_sensitive,
            )
        )

    # Adding label-related clauses and filters
    add_clause_and_filter("rdfs:label", "label")
    add_clause_and_filter("skos:altLabel", "altLabel")

    # Optional include handling
    for inc in include:
        add_clause_and_filter(
            f"tvbo:{inc}" if ":" not in inc else inc,
            inc if ":" not in inc else inc.split(":")[1],
        )

    optional_clauses_str = "\n    ".join(optional_clauses)
    filters_str = " ||\n        ".join(filters)

    sparql_string = rf"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX tvbo: <https://w3id.org/tvbo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?subject
WHERE {{
    ?subject a ?type  .
    {optional_clauses_str}
    FILTER (?type IN ({", ".join(types)}))
    FILTER (
        {filters_str}
    )
}}
    """
    onto = onto if onto is not None else ontology.onto
    results = list(set(sparql_query(sparql_string, world=onto.world)))
    if root_class:
        if isinstance(root_class, str):
            root_class = onto.search_one(label=root_class)
        results = ontology.intersection(results, root_class.descendants(include_self=False))
    return results


def get_children(cl: Any, onto: Any = None) -> List[Tuple[str, Any]]:
    """Return the incoming edges of a class, i.e. entities that point to it.

    Scans the triple store for triples whose object is `cl` and returns each predicate together with the subject entity, giving the class's immediate
    children in the relationship graph.

    Args:
        cl: The target class as an owlready2 entity, a label string, or an
            integer identifier (zero-padded to six digits and resolved by
            `identifier`).
        onto: The ontology to query. Defaults to the global runtime ontology.

    Returns:
        A list of `(predicate, entity)` tuples, with predicates abbreviated to
        their prefixed form.
    """
    onto = onto if onto is not None else ontology.onto
    world = onto.world
    if isinstance(cl, str):
        cl = onto.search_one(label=cl)
    if isinstance(cl, int):
        cl = onto.search_one(identifier=str(cl).zfill(6))

    storid = cl.storid

    predicates = world._get_obj_triples_o_p(storid)
    edges = []
    for p in predicates:
        if p < 0:
            continue
        for o in world._get_obj_triples_po_s(p=p, o=storid):
            if o < 0:
                continue
            edges.append(
                (
                    iri2prefix(world._unabbreviate(p)),
                    onto.search_one(iri=world._unabbreviate(o)),
                )
            )
    return edges


def get_parents(cl: Any, onto: Any = None) -> List[Tuple[str, Any]]:
    """Return the outgoing edges of a class, i.e. entities it points to.

    Scans the triple store for triples whose subject is `cl` and returns each predicate together with the resolvable object entity, giving the class's
    immediate parents in the relationship graph.

    Args:
        cl: The source class as an owlready2 entity, a label string, or an
            integer identifier (zero-padded to six digits and resolved by
            `identifier`).
        onto: The ontology to query. Defaults to the global runtime ontology.

    Returns:
        A list of `(predicate, entity)` tuples, with predicates abbreviated to
        their prefixed form; objects that do not resolve to an entity are
        skipped.
    """
    onto = onto if onto is not None else ontology.onto
    world = onto.world
    if isinstance(cl, str):
        cl = onto.search_one(label=cl)
    if isinstance(cl, int):
        cl = onto.search_one(identifier=str(cl).zfill(6))

    storid = cl.storid

    edges = []
    for p, o in world._get_obj_triples_s_po(s=storid):
        if o < 0 or p < 0:
            continue
        onto_class = onto.search_one(iri=world._unabbreviate(o))
        if onto_class:
            edges.append(
                (
                    iri2prefix(world._unabbreviate(p)),
                    onto_class,
                )
            )
    return edges
