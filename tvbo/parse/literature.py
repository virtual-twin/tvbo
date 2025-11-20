#
# Module: literature.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
# Handling studies
"""
from typing import Any, List
from tvbo.knowledge import query
from tvbo.knowledge.ontology import intersection, get_onto


def format_authors(authors_list: List[str]) -> List[str]:
    """
    Format the authors list for citations.

    Parameters:
        authors_list (list): The list of authors.

    Returns:
        list: A list of formatted author names.
    """
    formatted_authors = []

    for author_str in authors_list:
        # Split the string into individual authors
        authors = author_str.split(" and ")
        for author in authors:
            # Split each author's name into parts
            parts = author.split()
            if len(parts) >= 2:
                # Format as "Lastname, F. M."
                formatted_name = (
                    f"{parts[-1]}, {' '.join([p[0] + '.' for p in parts[:-1]])}"
                )
            else:
                # If there's only one part, use it as is
                formatted_name = author

            formatted_authors.append(formatted_name)

    return formatted_authors


def render_citation(citation: Any, style: str = "apa") -> str:
    """
    Render a citation as either a BibTeX entry or an APA citation.

    Parameters:
        citation (owlready2 instance): An instance with citation details.
        style (str): The citation style ('bibtex' or 'apa').

    Returns:
        str: The formatted citation.
    """
    # Extract fields
    authors = format_authors(citation.author)
    year = citation.year[0] if citation.year else "Unknown Year"
    title = citation.title[0] if citation.title else "Unknown Title"
    journal = citation.journal[0] if citation.journal else "Unknown Journal"
    volume = citation.volume[0] if citation.volume else "Unknown Volume"
    pages = citation.pages[0] if citation.pages else "Unknown Pages"
    label = citation.label[0] if citation.label else "UnknownLabel"

    # Format authors for both styles

    if style.lower() == "bibtex":
        return (
            f"@article{{{label},\n    author = {{{' and '.join(authors)}}},\n    title = {{{title}}},\n    "
            f"journal = {{{journal}}},\n    year = {{{year}}},\n    volume = {{{volume}}},\n    "
            f"pages = {{{pages}}}\n}}"
        )

    elif style.lower() == "apa":
        return (
            f"{', '.join(authors)} ({year}). {title}. *{journal}*, {volume}, {pages}."
        )

    else:
        return "Unsupported citation style."


def get_used_model(article_instance: Any) -> List[Any]:
    onto = get_onto()
    res = query.instance_class_relationship(article_instance.iri, 'prov:used')
    res = [r[1] for r in res]
    return intersection(res, onto.NeuralMassModel.descendants())
