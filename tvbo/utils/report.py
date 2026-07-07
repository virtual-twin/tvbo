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
from typing import Any, Sequence

import pandas as pd
from tvbo.data import db


_EMPTY_MARKERS = {"", "—", "-", "None", "nan"}


def md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    aligns: Sequence[str] | None = None,
    empty: str = "—",
) -> str:
    """Render a GitHub-markdown table, omitting columns with no data.

    A column is dropped when every one of its data cells is empty (blank,
    ``None``, or one of the placeholder markers). Kept columns render their
    empty cells as ``empty``. This keeps auto-generated report tables narrow:
    a parameter set with no ``default``/``domain``/``flags`` values shows only
    the columns that carry information.

    Args:
        headers: Column titles.
        rows: One sequence of cell values per row.
        aligns: Per-column alignment, ``'l'``/``'r'``/``'c'``; defaults to left.
        empty: Placeholder rendered for an empty cell in a kept column.

    Returns:
        The markdown table as a string (header, rule, and body rows).
    """
    n = len(headers)
    norm = [[("" if c is None else str(c)).strip() for c in row] for row in rows]

    def _blank(cell: str) -> bool:
        return cell in _EMPTY_MARKERS

    keep = [j for j in range(n) if any(not _blank(r[j]) for r in norm)] if norm else list(range(n))
    aligns = list(aligns) if aligns else ["l"] * n
    rule = {"l": ":---", "r": "---:", "c": ":--:"}

    head = "| " + " | ".join(headers[j] for j in keep) + " |"
    sep = "|" + "|".join(rule.get(aligns[j], ":---") for j in keep) + "|"
    body = "\n".join(
        "| " + " | ".join((r[j] if not _blank(r[j]) else empty) for j in keep) + " |"
        for r in norm
    )
    return "\n".join([head, sep] + ([body] if body else []))


# ── Report cell formatters ──────────────────────────────────────────────────
# Shared by the model / experiment / coupling report templates so the table
# building lives here (the adapter) rather than being duplicated in each Mako.


def slot(obj, name, default=None):
    """Safe attribute access on a report object (``getattr`` with a default)."""
    return getattr(obj, name, default) if obj is not None else default


def present(value):
    """True when a value carries information (not ``None`` / empty / ``''``)."""
    return value not in (None, "", [], {})


def name_items(collection):
    """Yield ``(name, obj)`` pairs from a name-keyed dict, list, or ``None``."""
    if not collection:
        return []
    if hasattr(collection, "items"):
        return list(collection.items())
    values = collection.values() if hasattr(collection, "values") else collection
    return [(slot(v, "name", f"item_{i}"), v) for i, v in enumerate(values)]


def unit_text(unit):
    """Render a unit as inline LaTeX, or the empty marker when absent."""
    from tvbo.utils.units import unit_to_latex

    unit_ltx = unit_to_latex(unit) if unit else ""
    return "$" + unit_ltx + "$" if unit_ltx else "—"


def range_text(range_obj):
    """One-line summary of an explored range / domain (values, ``[lo, hi]``, step, n)."""
    if not range_obj:
        return ""
    values = slot(range_obj, "explored_values", None)
    if values:
        values = [str(v) for v in values]
        return "{" + ", ".join(values[:8]) + ("..." if len(values) > 8 else "") + "}"
    lo, hi = slot(range_obj, "lo"), slot(range_obj, "hi")
    step, n_points = slot(range_obj, "step"), slot(range_obj, "n")
    parts = []
    if lo is not None or hi is not None:
        parts.append(f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]")
    if step is not None:
        parts.append(f"step={step}")
    if n_points is not None:
        parts.append(f"n={n_points}")
    if slot(range_obj, "log_scale", False):
        parts.append("log")
    return ", ".join(parts)


def distribution_text(distribution):
    """One-line summary of a sampling distribution (name, domain, axis, seed)."""
    if not distribution:
        return ""
    parts = [str(slot(distribution, "name", "Distribution"))]
    domain = range_text(slot(distribution, "domain"))
    if domain:
        parts.append(domain)
    axis, seed = slot(distribution, "axis"), slot(distribution, "seed")
    if axis:
        parts.append(f"axis={axis}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return " ".join(parts)


def metadata_text(obj):
    """Domain / Sampling cell: bounds + enforcement + distribution."""
    from tvbo.utils import domain_enforcement

    bits = []
    dom = slot(obj, "domain")
    if present(dom):
        bits.append(range_text(dom))
        enf = domain_enforcement(dom)  # none / clamp / wrap (boundaries folded into domain)
        if enf != "none":
            bits.append(f"enforce={enf}")
    if present(slot(obj, "distribution")):
        bits.append(distribution_text(slot(obj, "distribution")))
    return "; ".join(bit for bit in bits if bit) or "—"


def flag_text(obj, flags=None):
    """Flags cell: boolean flags + shape / dataset / reported optimum.

    ``flags`` is a list of ``(attr, label)`` pairs; it defaults to the standard
    parameter flags (``free``, ``heterogeneous``).
    """
    flags = _PARAM_FLAGS if flags is None else flags
    labels = [label for name, label in flags if slot(obj, name, False)]
    for attr, key in (("shape", "shape"), ("dataset_path", "data"), ("reported_optimum", "optimum")):
        val = slot(obj, attr)
        if val is not None and val != "":
            labels.append(f"{key}={val}")
    return ", ".join(labels) or "—"


_STATE_VAR_FLAGS = [("coupling_variable", "coupling"), ("stimulation_variable", "stimulation"), ("record", "recorded")]
_PARAM_FLAGS = [("free", "free"), ("heterogeneous", "heterogeneous")]


def state_variable_table(svars):
    """Markdown State-Variables table (empty columns dropped) from a name->obj map."""
    from sympy import Symbol, latex

    rows = [
        [f"${latex(Symbol(name))}$", slot(sv, "initial_value", ""), unit_text(slot(sv, "unit")),
         f"{slot(sv, 'equation_type', 'differential')} (order {slot(sv, 'equation_order', 1)})",
         metadata_text(sv), flag_text(sv, _STATE_VAR_FLAGS),
         slot(sv, "description", "") or slot(sv, "definition", "") or ""]
        for name, sv in name_items(svars)
    ]
    return md_table(["Variable", "Initial Value", "Unit", "Equation", "Domain / Sampling", "Flags", "Description"], rows)


def parameter_table(params):
    """Markdown Parameters table (empty columns dropped) from a name->obj map."""
    from sympy import Symbol, latex

    rows = [
        [f"${latex(Symbol(name))}$", slot(p, "value", ""), slot(p, "default", ""), unit_text(slot(p, "unit")),
         metadata_text(p), flag_text(p, _PARAM_FLAGS),
         slot(p, "description", "") or slot(p, "definition", "") or ""]
        for name, p in name_items(params)
    ]
    return md_table(["Parameter", "Value", "Default", "Unit", "Domain / Sampling", "Flags", "Description"],
                    rows, aligns=["l", "r", "l", "l", "l", "l", "l"])


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
