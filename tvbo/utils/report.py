#  report.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#

"""
Report Module
=============

This module provides utilities for generating reports related to model parameters and configurations.

.. moduleauthor:: Leon K. Martin

Functions:
----------
"""

import operator
from typing import Any

import pandas as pd
from tvbo.data import db


def parameter_report(param_setting, decimals=3, format="latex", **kwargs):
    """
    Generate a report of parameter settings.

    Parameters
    ----------
    param_setting : object
        Parameter setting object.
    decimals : int, optional
        Number of decimal places for formatting. Default is 3.
    format : str, optional
        Format for the report: 'latex', 'pandas', or 'markdown'. Default is 'latex'.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    pandas.DataFrame or str
        Report table if format is 'pandas', LaTeX string if format is 'latex', or markdown string if format is 'markdown'.

    Raises
    ------
    ValueError
        If the provided format is not recognized.
    """

    short_caption = "Parameter values for the {} model*.".format(param_setting.model.label.first().replace("_", "-"))

    long_caption = short_caption + " " + "UID is the unique identifier of the parameter in the ontology."

    report_table = pd.DataFrame()
    report_table.index.name = "Parameter"
    # for k, v in param_settingconfig.items():
    for k in sorted(param_setting.config, key=operator.attrgetter("name")):
        v = param_setting.config[k]

        parameter = "$" + k.symbol.first() + "$"
        report_table.at[parameter, "UID"] = "TVBO:" + str(k.identifier.first())
        report_table.at[parameter, "value"] = v
        unit = k.unit.first()
        if unit is None:
            unit = ""
        report_table.at[parameter, "unit"] = "$1" + unit.replace("^-1", "^{-1}") + "$"

    if format == "pandas":
        return report_table
    elif format.lower() == "latex":
        latex = (
            report_table.style.format(decimal=".", thousands=",", precision=decimals)
            .to_latex(
                position="h!",
                hrules=True,
                # float_format="%.2f",
                caption=(long_caption, short_caption),
                label="tab_{}_setting".format(param_setting.model.label.first(), **kwargs),
            )
            .replace("\\$", "$")
        )
        latex = latex.replace(
            r"\end{table}",
            r"""\begin{tablenotes}
\small
\item[*] \footnotesize{This table was automatically generated with TVB-O.}
\end{tablenotes}
\end{table}""",
        )
        return latex
    elif format.lower() == "markdown":
        md = report_table.style.format(decimal=".", thousands=",", precision=decimals).to_markdown()
        return md
    else:
        raise ValueError("Unknown format: {}".format(format))


def model_report():
    """
    Generate a report for the model.

    Returns
    -------
    None
    """
    pass


def save_latex(conf, fpath):
    """
    Save a LaTeX report to a file.

    Parameters
    ----------
    conf : object
        Configuration object.
    fpath : str
        File path to save the LaTeX report.

    Returns
    -------
    None
    """
    with open(fpath, "w") as texfile:
        texfile.write(conf.get_report(format="latex"))


##############
# References #
##############


def render_citation(citation: Any, style: str = "apa") -> str:
    """Render an ontology citation instance as formatted text.

    Args:
        citation: An owlready2 instance with author, year, title, journal, volume, pages, label.
        style: 'bibtex' or 'apa'.

    Returns:
        str: The formatted citation.
    """
    authors_list = citation.author
    formatted_authors = []
    for author_str in authors_list:
        for author in author_str.split(" and "):
            parts = author.split()
            if len(parts) >= 2:
                formatted_authors.append(f"{parts[-1]}, {' '.join([p[0] + '.' for p in parts[:-1]])}")
            else:
                formatted_authors.append(author)

    year = citation.year[0] if citation.year else "Unknown Year"
    title = citation.title[0] if citation.title else "Unknown Title"
    journal = citation.journal[0] if citation.journal else "Unknown Journal"
    volume = citation.volume[0] if citation.volume else "Unknown Volume"
    pages = citation.pages[0] if citation.pages else "Unknown Pages"
    label = citation.label[0] if citation.label else "UnknownLabel"

    if style.lower() == "bibtex":
        return (
            f"@article{{{label},\n    author = {{{' and '.join(formatted_authors)}}},\n    title = {{{title}}},\n    "
            f"journal = {{{journal}}},\n    year = {{{year}}},\n    volume = {{{volume}}},\n    "
            f"pages = {{{pages}}}\n}}"
        )
    elif style.lower() == "apa":
        return f"{', '.join(formatted_authors)} ({year}). {title}. *{journal}*, {volume}, {pages}."
    else:
        return "Unsupported citation style."


def get_citation(citation_key) -> str:
    """Retrieve a BibTeX entry by its citation key and render it as an APA-style plain text citation.

    Args:
        citation_key (str): The citation key to retrieve.

    Returns:
        str: The citation formatted in APA style, or an error message if not found.
    """
    bib_data = db.load_bibliography()
    if citation_key in bib_data.entries:
        entry = bib_data.entries[citation_key]
        # Format authors
        authors = entry.persons.get("author", [])
        author_str = ""
        if len(authors) == 1:
            author_str = f"{authors[0].last_names[0]}, {authors[0].first_names[0][0]}."
        elif len(authors) == 2:
            author_str = f"{authors[0].last_names[0]}, {authors[0].first_names[0][0]}. & {authors[1].last_names[0]}, {authors[1].first_names[0][0]}."
        elif len(authors) > 2:
            author_str = (
                ", ".join([f"{a.last_names[0]}, {a.first_names[0][0]}." for a in authors[:-1]])
                + f", & {authors[-1].last_names[0]}, {authors[-1].first_names[0][0]}."
            )

        # Format title
        title = entry.fields.get("title", "").capitalize()

        # Format year
        year = entry.fields.get("year", "n.d.")

        # Format journal or book title
        source = entry.fields.get("journal", entry.fields.get("booktitle", ""))

        # Format volume, issue, and pages
        volume = entry.fields.get("volume", "")
        number = entry.fields.get("number", "")
        pages = entry.fields.get("pages", "")

        # Assemble APA-style citation
        citation = f"{author_str} ({year}). {title}. *{source}*"
        if volume:
            citation += f", {volume}"
            if number:
                citation += f"({number})"
        if pages:
            citation += f", {pages.replace('--', '-')}"
        citation += "."
        return citation
    else:
        return f"Citation key '{citation_key}' not found."


def to_pdf(render, outputfile):
    """Convert Markdown text to a PDF file via pandoc.

    Uses `pypandoc` with the `xelatex` PDF engine and a 3.5 cm page margin to
    render the given Markdown source and write the result to disk.

    Args:
        render: Markdown-formatted source text to convert.
        outputfile: Path where the generated PDF is written.
    """
    import pypandoc

    pypandoc.convert_text(
        render,
        "pdf",
        format="md",
        outputfile=outputfile,
        extra_args=["--pdf-engine=xelatex", "-V", "geometry:margin=3.5cm"],
    )
