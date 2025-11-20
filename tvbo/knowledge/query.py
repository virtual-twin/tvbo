#
# Module: query.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
from typing import Any, List, Tuple, Union
from tvbo.knowledge import ontology
import owlready2

prefixes = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/2002/07/owl#": "owl:",
    "http://www.thevirtualbrain.org/tvb-o/": "tvbo:",
}


def iri2prefix(iri: str) -> str:
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


def sparql_query(query_string: str, flatten_result: bool = True) -> List[Any]:
    res: List[Any] = list(ontology.onto.world.sparql(query_string))
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
    PREFIX tvbo: <http://www.thevirtualbrain.org/tvb-o/>

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
    if not isinstance(class_iri, str):
        class_iri = class_iri.iri

    return sparql_query(
        f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX tvbo: <http://www.thevirtualbrain.org/tvb-o/>

        SELECT ?p ?object
        WHERE {{
            <{class_iri}> ?p ?object .
        }}
        """,
        flatten_result=False,
    )


def instance_class_relationship(
    subject_iri: str, predicate: str = "prov:used"
) -> List[Tuple[Any, Any]]:
    predicate_restriction = (
        f"?restriction owl:onProperty {predicate} ." if predicate else ""
    )

    query_string = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX tvbo: <http://www.thevirtualbrain.org/tvb-o/>

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
    PREFIX tvbo: <http://www.thevirtualbrain.org/tvb-o/>

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
        (BOUND(?symbol) && LCASE(?symbol) = "{label.lower().replace('$', '')}")
    )
    }}
    """
    return sparql_query(sparql_string)


def build_filter(label: str, field: str, exact: bool, case_sensitive: bool) -> str:
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
) -> List[owlready2.ThingClass]:
    if greek_to_latin:
        label = convert_greek_to_latin(label)
    if ignore_underscore:
        label = label.replace("_", "")

    label = label.replace("$", "")
    exact_match = [exact_match] if isinstance(exact_match, str) else exact_match

    optional_clauses = []
    filters = []

    def add_clause_and_filter(field, field_name):
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
            f"tvbo:{inc}" if not ":" in inc else inc,
            inc if not ":" in inc else inc.split(":")[1],
        )

    optional_clauses_str = "\n    ".join(optional_clauses)
    filters_str = " ||\n        ".join(filters)

    sparql_string = rf"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX tvbo: <http://www.thevirtualbrain.org/tvb-o/>
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
    results = list(set(sparql_query(sparql_string)))
    if root_class:
        if isinstance(root_class, str):
            root_class = ontology.onto.search_one(label=root_class)
        results = ontology.intersection(
            results, root_class.descendants(include_self=False)
        )
    return results


def get_children(cl: Any) -> List[Tuple[str, Any]]:
    if isinstance(cl, str):
        cl = ontology.onto.search_one(label=cl)
    if isinstance(cl, int):
        cl = ontology.onto.search_one(identifier=str(cl).zfill(6))

    storid = cl.storid

    predicates = ontology.onto.world._get_obj_triples_o_p(storid)
    predicates_unabr = [
        ontology.onto.world._unabbreviate(p)
        for p in ontology.onto.world._get_obj_triples_o_p(storid)
    ]
    edges = []
    for p in predicates:
        if p < 0:
            continue
        for o in ontology.onto.world._get_obj_triples_po_s(p=p, o=storid):
            if o < 0:
                continue
            edges.append(
                (
                    iri2prefix(ontology.onto.world._unabbreviate(p)),
                    ontology.onto.search_one(iri=ontology.onto.world._unabbreviate(o)),
                )
            )
    return edges


def get_parents(cl: Any) -> List[Tuple[str, Any]]:
    if isinstance(cl, str):
        cl = ontology.onto.search_one(label=cl)
    if isinstance(cl, int):
        cl = ontology.onto.search_one(identifier=str(cl).zfill(6))

    storid = cl.storid

    edges = []
    for p, o in ontology.onto.world._get_obj_triples_s_po(s=storid):
        if o < 0 or p < 0:
            continue
        onto_class = ontology.onto.search_one(iri=ontology.onto.world._unabbreviate(o))
        if onto_class:
            edges.append(
                (
                    iri2prefix(ontology.onto.world._unabbreviate(p)),
                    onto_class,
                )
            )
    return edges
