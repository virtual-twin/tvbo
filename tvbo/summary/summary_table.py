#
# Module: summary_table.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import pandas as pd
from pathlib import Path
import glob
import yaml

from tvbo.knowledge import ontology

df_info = ontology.ontology_info(return_df=True, print_info=False)

df_info.index.name = "TVB-O component"


def ontostats2latex():
    short_caption = "Ontology metrics."

    long_caption = (
        "\\textbf{{{}}} TVB-O contains classes, instances and properties to formulate axioms of BNM "
        "knowledge."
    ).format(short_caption)

    latex = df_info.style.to_latex(
        position="h!",
        hrules=True,
        # float_format="%.2f",
        caption=(long_caption, short_caption),
        label="tab_results_tvbo_stats",
    ).replace("_", " ")

    return latex


def ontostats2markdown():
    # TODO: are short_caption and long_caption needed? if so, maybe they could be constants, since they are used in
    # ontostats2latex() and ontostats2markdown()
    short_caption = "Ontology metrics."

    # long_caption = "\\textbf{{{}}} TVB-O contains classes, instances and properties to formulate axioms of BNM knowledge.".format(
    #     short_caption
    # )

    markdown = df_info.to_markdown(
        # position="h!",
        # hrules=True,
        # float_format="%.2f",
        # caption=(long_caption, short_caption),
        # label="tab_results_tvbo_stats",
    ).replace("_", " ")

    return markdown


def summarize_models():
    short_caption = "Neural mass models in TVB-O."

    long_caption = "\\textbf{{{}}} List of currently annotated TVB models.".format(
        short_caption
    )

    df_models_summary = pd.DataFrame()
    for i, (k, v) in enumerate(ontology.get_models().items()):
        if k in ["Epileptor", "Wong-Wang Deco", "Stefanescu-Jirsa", "Infinite Theta"]:
            continue

        params = ontology.get_model_parameters(v)
        # funcs = ontology.get_model_functions(v)  # not used
        svs = ontology.get_model_statevariables(v)
        # references = v.references.first()  # not used
        df_models_summary.at[i, "NMM"] = k
        df_models_summary.at[i, "UID"] = "TVBO:" + v.identifier.first()
        df_models_summary.at[i, "parameters"] = int(len(params))
        df_models_summary.at[i, "dimensions"] = int(len(svs))
        df_models_summary.at[i, "reference"] = r"\citep" + str(v.has_reference).replace(
            "[", "{"
        ).replace("]", "}")
    # Sort and render after population
    df_models_summary.sort_values("NMM", inplace=True)
    df_models_summary.reset_index(inplace=True, drop=True)
    latex = (
        df_models_summary.style.format(decimal=".", thousands=",", precision=0)
        .format_index(axis=1, formatter="${}$".format)
        .hide(axis=0)
        .to_latex(
            position="h!",
            hrules=True,
            convert_css=True,
            caption=(long_caption, short_caption),
            label="NMM_overview",
            environment="table*",
        )
        .replace("_", " ")
    )
    latex = latex.replace(
        r"\end{table*}",
        """\\begin{tablenotes}
\\small
\\centering
\\item[*] \\footnotesize{This table was automatically generated with TVB-O.}
\\end{tablenotes}
\\end{table*}""",
    )

    return latex


# TODO: review this code
# for i in df_info.index:
#     df_info.index.rename(i, i.capitalize().replace("_", " "))


# def latex_table():

# def markdown_table():


# def word_table(dout):
#     tbl_header = OxmlElement(
#         "w:tblHeader"
#     )  # create new oxml element flag which indicates that row is header row

#     word_document = Document()
#     document_name = "tvbo-table1"

#     table = word_document.add_table(rows=1, cols=2)
#     table.allow_autofit = True

#     first_row_props = table.rows[
#         0
#     ]._element.get_or_add_trPr()  # get if exists or create new table row properties el
#     first_row_props.append(tbl_header)  # now first row is the header row

#     # table.style = "Plain Table 5"

#     for k, v in info.items():
#         row = table.add_row()
#         row.cells[0].text = k
#         row.cells[1].text = str(len(v))

#     word_document.save(join(dout, document_name + ".docx"))


def summarize_properties() -> pd.DataFrame:
    """
    Summarize the properties of the TVB-O class.

    Returns:
        pd.DataFrame: A DataFrame containing the summarized properties.
    """
    annot_props = pd.DataFrame()
    i = 0
    for anoprop in ontology.onto.annotation_properties():
        if ontology.onto.bibtex in anoprop.is_a:
            continue
        annot_props.at[i, "type"] = "Annotation property"
        annot_props.at[i, "label"] = anoprop.label.first()
        annot_props.at[i, "definition"] = anoprop.definition.first()
        i += 1
    annot_props.sort_values(
        "label",
        ascending=True,
        key=lambda col: col.str.lower(),
        inplace=True,
        ignore_index=True,
    )

    object_props = pd.DataFrame()
    i = 0
    for obprop in ontology.onto.object_properties():
        # if ontology.onto.bibtex in anoprop.is_a:
        #     continue
        object_props.at[i, "type"] = "Object property"
        object_props.at[i, "label"] = obprop.label.first()
        object_props.at[i, "definition"] = obprop.definition.first()
        i += 1

        object_props.sort_values(
            "label",
            ascending=True,
            key=lambda col: col.str.lower(),
            inplace=True,
            ignore_index=True,
        )

    # Merge
    property_table = (
        pd.concat(
            [annot_props, object_props],
            keys=[annot_props.type[0], object_props.type[0]],
            names=["Property type"],
        )
        .drop("type", axis=1)
        .dropna(how="all")
    )

    return property_table


def properties2latex():
    """
    Generate a LaTeX table from the summarized properties of the TVB-O class.
    Returns:
        str: The LaTeX code representing the table.
    Raises:
        None
    """

    property_table = summarize_properties()
    short_caption = "\\textbf{TVB-O class properties.}"

    long_caption = (
        short_caption
        + " "
        + "Annotation properties describe class properties, whereas object properties describe the predicate of class "
        "relationships"
    )

    latex = (
        property_table.style.to_latex(
            position="h!",
            hrules=True,
            clines="skip-last;index",
            convert_css=True,
            # float_format="%.2f",
            caption=(long_caption, short_caption),
            label="TVB-O class properties",
            multirow_align="t",
            environment="longtable",
        )
    ).replace("_", r"\_")

    return latex


def summarize_models_quarto():
    """
    Generate a Quarto-compatible Markdown table summarizing TVB-O neural mass models.
    """

    short_caption = "Neural mass models in TVB-O."
    long_caption = f"**{short_caption}** List of currently annotated TVB models."

    df_models_summary = pd.DataFrame()

    # Build summary table
    for i, (k, v) in enumerate(ontology.get_models().items()):
        if k in ["Epileptor", "Wong-Wang Deco", "Stefanescu-Jirsa", "Infinite Theta"]:
            continue

        params = ontology.get_model_parameters(v)
        svs = ontology.get_model_statevariables(v)

        df_models_summary.at[i, "NMM"] = k
        df_models_summary.at[i, "UID"] = "TVBO:" + v.identifier.first()
        df_models_summary.at[i, "Parameters"] = len(params)
        df_models_summary.at[i, "Dimensions"] = len(svs)
        df_models_summary.at[i, "Reference"] = ";".join(
            f"@{ref}" for ref in v.has_reference
        )

    # Sort and reset index
    df_models_summary.sort_values("NMM", inplace=True)
    df_models_summary.reset_index(inplace=True, drop=True)

    # Build Quarto Markdown table
    header = "| NMM | UID | Parameters | Dimensions | Reference |\n"
    header += "|------|------|------------|------------|-----------|\n"
    rows = "\n".join(
        "| {} | {} | {} | {} | {} |".format(
            row["NMM"],
            row["UID"],
            int(row["Parameters"]),
            int(row["Dimensions"]),
            f"[{row['Reference']}]",
        )
        for _, row in df_models_summary.iterrows()
    )

    # Add Quarto-style table caption at the end
    md_table = f"{header}{rows}"

    return md_table


def summarize_studies_quarto():
    """
    Generate a Quarto-compatible Markdown table summarizing the literature database entries.

    Columns: Study (YAML ID) | Year | Citation
    """

    short_caption = "Literature database of simulation studies."
    long_caption = f"**{short_caption}** Curated entries in TVB-O."

    # Locate the tvbo/data/db directory relative to this file
    # __file__ = .../tvbo/summary/summary_table.py -> parents[1] == .../tvbo
    root = Path(__file__).resolve().parents[1]
    db_dir = root / "data" / "db"

    rows = []
    for path in sorted(glob.glob(str(db_dir / "*.yaml"))):
        p = Path(path)
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        key = data.get("key", p.stem)
        year = data.get("year", "—")
        try:
            year = int(year)
        except Exception:
            year = "—"
        rows.append({
            "Study (YAML ID)": key,
            "Year": year,
            "Citation": f"[@{key}]",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Study (YAML ID)", inplace=True)
        df.reset_index(drop=True, inplace=True)

    header = "| Study (YAML ID) | Year | Citation |\n"
    header += "|:----------------|:----:|:---------|\n"
    body = "\n".join(
        "| {} | {} | {} |".format(r["Study (YAML ID)"], r["Year"], r["Citation"]) for _, r in df.iterrows()
    )
    md_table = f"{header}{body}"

    return md_table


def summarize_unified_quarto():
    """
    Generate a single Quarto-compatible Markdown table combining:
      - Local Dynamics (Model) entries from tvbo/data/db/Model/*.yaml
      - SimulationStudy entries from tvbo/data/db/*.yaml

    Columns: Entry | Type | Year | Reference
    """

    short_caption = "Models and simulation studies in TVB-O."
    long_caption = f"**{short_caption}** Unified view with a Type column distinguishing Local Dynamics (Model) and SimulationStudy entries."

    root = Path(__file__).resolve().parents[1]
    db_dir = root / "data" / "db"
    model_dir = db_dir / "Model"

    rows = []

    # Models (Local Dynamics)
    # Prefer ontology-derived list (complete); fall back to YAML models if ontology is unavailable
    added_models = False
    try:
        models = ontology.get_models()
        for i, (k, v) in enumerate(models.items()):
            if k in ["Epileptor", "Wong-Wang Deco", "Stefanescu-Jirsa", "Infinite Theta"]:
                continue
            params = ontology.get_model_parameters(v)
            svs = ontology.get_model_statevariables(v)
            refs = list(getattr(v, "has_reference", []) or [])
            ref_str = "—"
            if refs:
                ref_items = [f"@{str(r)}" for r in refs]
                ref_str = f"[{';'.join(ref_items)}]"

            rows.append({
                "Entry": k + f" (params: {len(params)}, states: {len(svs)})",
                "Type": "Local Dynamics (Model)",
                "Year": "—",
                "Reference": ref_str,
            })
        added_models = True
    except Exception:
        # Lightweight fallback: scan YAML model specs
        for path in sorted(glob.glob(str(model_dir / "*.yaml"))):
            p = Path(path)
            with open(p, "r") as f:
                data = yaml.safe_load(f) or {}

            name = data.get("name") or data.get("label") or p.stem
            refs = data.get("has_reference", []) or []
            if isinstance(refs, str):
                refs = [refs]
            ref_str = f"[@{'; @'.join(refs)}]" if refs else "—"

            params = data.get("parameters", {}) or {}
            svs = data.get("state_variables", {}) or {}

            rows.append({
                "Entry": name + f" (params: {len(params)}, states: {len(svs)})",
                "Type": "Local Dynamics (Model)",
                "Year": "—",
                "Reference": ref_str,
            })

    # Simulation studies
    for path in sorted(glob.glob(str(db_dir / "*.yaml"))):
        p = Path(path)
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}

        key = data.get("key", p.stem)
        year = data.get("year", "—")
        try:
            year = int(year)
        except Exception:
            year = "—"

        rows.append({
            "Entry": key,
            "Type": "SimulationStudy",
            "Year": year,
            "Reference": f"[@{key}]",
        })

    # Build DataFrame and sort
    df = pd.DataFrame(rows)
    if not df.empty:
        # Drop duplicates robustly (e.g., duplicate study YAMLs)
        df.drop_duplicates(subset=["Type", "Entry"], inplace=True)
        df.sort_values(["Type", "Entry"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Markdown table
    header = "| Entry | Type | Year | Reference |\n"
    header += "|:------|:-----|:----:|:----------|\n"
    body = "\n".join(
        "| {} | {} | {} | {} |".format(r["Entry"], r["Type"], r["Year"], r["Reference"]) for _, r in df.iterrows()
    )

    md_table = f"{header}{body}"

    return md_table
