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
import re
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import pandas as pd
from tvbo.data import db


_EMPTY_MARKERS = {"", "—", "-", "None", "nan"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")

_MATHRM_RE = re.compile(r"\\mathrm\{([^}]*)\}")
_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_MARKUP_RE = re.compile(r"[{}\\_^$]")


def _visual_width(cell: Any) -> int:
    """Approximate the *rendered* character width of a markdown/LaTeX cell.

    A cell like ``$\\mathrm{s}$`` renders as a single ``s``, so its raw source
    length badly over-states how wide it is on the page. This strips the math
    delimiters, ``\\mathrm`` wrappers and control sequences so column sizing
    tracks what the reader sees, not the LaTeX source length.
    """
    s = _MATHRM_RE.sub(r"\1", str(cell))
    s = _CMD_RE.sub("x", s)
    s = _MARKUP_RE.sub("", s)
    return max(len(s.strip()), 1)


def _as_prose(headers: Sequence[str], rows: Sequence[Sequence[str]], keep: Sequence[int]) -> str:
    """A grid too small to earn a float, written as a sentence.

    One surviving column is a list of its own values (the caller's heading already
    says what they are). Otherwise each row reads ``Header value``, fields separated
    by commas and rows by semicolons, so the sentence carries its own column names
    and needs no caption to be understood.
    """
    if not keep:
        return ""
    if len(keep) == 1:
        return ", ".join(r[keep[0]] for r in rows if r[keep[0]] not in _EMPTY_MARKERS)

    def _clause(row):
        """``Exp 50 (Duration: 2000 ms)``: the first column names the subject."""
        subject = f"{headers[keep[0]]} {row[keep[0]]}".strip()
        rest = ", ".join(f"{headers[j]}: {row[j]}" for j in keep[1:] if row[j] not in _EMPTY_MARKERS)
        return f"{subject} ({rest})" if rest else subject

    return "; ".join(_clause(r) for r in rows) + "."


def md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    aligns: Sequence[str] | None = None,
    empty: str = "",
    col_cap: int = 44,
    col_floor: int = 9,
) -> str:
    """Render a GitHub-markdown table, omitting columns with no data.

    A column is dropped when every one of its data cells is empty (blank,
    ``None``, or one of the placeholder markers). Kept columns render their
    empty cells as ``empty`` (blank by default — no ``—`` placeholder). This
    keeps auto-generated report tables narrow: a parameter set with no
    ``default``/``domain``/``flags`` values shows only the columns that carry
    information.

    Always returns a table. A caller that would rather write a small grid as a
    sentence calls [`table_or_prose`](#tvbo.utils.report.table_or_prose) instead;
    the decision needs the caller's subject and keying to read correctly, and
    [`read_md_tables`](#tvbo.utils.report.read_md_tables) is the documented inverse
    of this function only while it renders a table.

    Args:
        headers: Column titles.
        rows: One sequence of cell values per row.
        aligns: Per-column alignment, ``'l'``/``'r'``/``'c'``; defaults to left.
        empty: Placeholder rendered for an empty cell in a kept column.
        col_cap: Width above which a column stops earning more of the page.
        col_floor: Width below which a column stops giving it up, so a short
            column keeps enough room to typeset its own cells.

    Returns:
        The markdown table as a string: header, rule, and body rows.
    """
    n = len(headers)
    norm = [[("" if c is None else str(c)).strip() for c in row] for row in rows]

    def _blank(cell: str) -> bool:
        return cell in _EMPTY_MARKERS

    keep = [j for j in range(n) if any(not _blank(r[j]) for r in norm)] if norm else list(range(n))

    aligns = list(aligns) if aligns else ["l"] * n

    # Size each kept column's separator to its rendered content width, clamped to
    # [col_floor, col_cap], so pandoc/LaTeX allocates PDF column widths *proportional to
    # content* instead of equally: a lone name column no longer hogs space while a long
    # description is squished. The floor matters as much as the cap -- a 3-character `ID`
    # column beside two 44-character prose columns gets ~3 % of the text width, which is
    # narrower than the word it holds, so its cells collide with the next column.
    def _sep(j):
        widths = [_visual_width(headers[j])] + [_visual_width(r[j]) for r in norm]
        width = max(col_floor, min(max(widths), col_cap))
        a = aligns[j] if j < len(aligns) else "l"
        if a == "r":
            return "-" * (width - 1) + ":"
        if a == "c":
            return ":" + "-" * (width - 2) + ":"
        return ":" + "-" * (width - 1)

    head = "| " + " | ".join(headers[j] for j in keep) + " |"
    sep = "|" + "|".join(_sep(j) for j in keep) + "|"
    body = "\n".join(
        "| " + " | ".join((r[j] if not _blank(r[j]) else empty) for j in keep) + " |"
        for r in norm
    )
    return "\n".join([head, sep] + ([body] if body else []))


def table_or_prose(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    aligns: Sequence[str] | None = None,
    min_cells: int = 3,
    **kwargs,
) -> str:
    """Render a grid as a table, or as a sentence when it is too small to earn a float.

    A numbered, captioned table announces to the reader that something has to be looked
    up, and journals cap how many a paper may carry; a table holding two numbers spends
    that budget on nothing. The threshold is `min_cells` values outside the key column,
    and at least two rows — so a lone `| Term |` column collapses to its values, a single
    declared event stops being a one-row float, and two experiments differing only in
    duration become a clause. Anything larger stays a table.

    Opt-in, because the sentence reads correctly only where the first column names a
    subject. A parameter block, a state-variable list or a scorecard has no such subject
    and stays a table however small it is — call `md_table` for those.

    Args:
        headers: Column titles; the first names the subject of each clause.
        rows: One sequence of cell values per row.
        aligns: Per-column alignment; ignored on the prose path.
        min_cells: Values outside the first column below which the grid is prose.
        **kwargs: Forwarded to `md_table` when the grid stays a table.

    Returns:
        A markdown table, or a sentence when the grid falls under the threshold.
    """
    n = len(headers)
    norm = [[("" if c is None else str(c)).strip() for c in row] for row in rows]
    keep = [j for j in range(n) if any(r[j] not in _EMPTY_MARKERS for r in norm)] if norm else list(range(n))
    if norm and (len(norm) < 2 or len(norm) * (len(keep) - 1) < min_cells):
        return _as_prose(headers, norm, keep)
    return md_table(headers, rows, aligns=aligns, **kwargs)


class MarkdownTable(NamedTuple):
    """One parsed markdown table, tagged with the heading it appeared under."""

    section: str
    headers: list[str]
    rows: list[dict]


def _cells(line: str) -> list[str]:
    """A table row's cells, honouring ``\\|`` escapes inside a cell."""
    parts = _CELL_SPLIT_RE.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip().replace("\\|", "|") for p in parts]


def read_md_tables(source) -> list[MarkdownTable]:
    """Read the GitHub-markdown tables out of a document — the inverse of `md_table`.

    Lets a report *compute* from a hand-maintained analysis file (a replication's
    `targets.md`, a divergence register) instead of restating its contents in prose,
    so the two can never disagree.

    Args:
        source: A path to a markdown file, or the markdown text itself.

    Returns:
        One `MarkdownTable` per table found, each row a `{header: cell}` dict and
        each table tagged with the nearest preceding heading.
    """
    text = source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in (".md", ".markdown", ".qmd") and path.exists():
            text = path.read_text(encoding="utf-8")

    lines = str(text).splitlines()
    tables: list[MarkdownTable] = []
    section = ""
    i = 0
    while i < len(lines):
        heading = _HEADING_RE.match(lines[i])
        if heading:
            section = heading.group(2).strip()
            i += 1
            continue
        is_header = lines[i].lstrip().startswith("|") and i + 1 < len(lines) and _RULE_RE.match(lines[i + 1].strip())
        if not is_header:
            i += 1
            continue
        headers = _cells(lines[i])
        rows = []
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            cells = _cells(lines[i])
            rows.append({h: (cells[j] if j < len(cells) else "") for j, h in enumerate(headers)})
            i += 1
        tables.append(MarkdownTable(section, headers, rows))
    return tables


# ── The replication-report toolkit ──────────────────────────────────────────
# Every replication report does the same handful of things: format a number that may not have
# been computed, open a result or analysis container, read a value off the recipe, embed a
# figure with the published original beside it, caption it from the recipe's own metadata, and
# score the run against the targets written before it. Those live here, once, so a report holds
# only what is specific to its study -- its metrics -- and ten reports cannot drift apart on the
# parts they share.


_FIG_LABEL_RE = re.compile(r"(EDF|Fig)(\d+)")

_MISSING = "—"


def fmt(x, digits: int = 2, missing: str = _MISSING) -> str:
    """A computed number for prose, or *missing* when it could not be computed.

    A report reads containers that may not exist yet, and a half-run study must render rather
    than crash: an absent number shows as a dash, which is visibly not a result.
    """
    import math

    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return missing
    return f"{x:.{digits}f}" if digits else f"{int(round(x))}"


def sci(x, digits: int = 2, missing: str = _MISSING) -> str:
    """A computed number in scientific notation, or *missing*."""
    import math

    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return missing
    return f"{x:.{digits}e}"


def value_of(obj):
    """The `.value` of a recipe object, or the object itself when it is already a scalar."""
    return getattr(obj, "value", obj)


def recipe_param(experiment, name, group: str = "dynamics"):
    """A declared parameter's value, read from the recipe rather than typed into prose.

    *group* selects where to look: ``"dynamics"`` for the model's parameters, or the name of a
    single event/coupling whose parameters to read. Returns None when the name is not declared,
    so a renamed parameter shows as a dash instead of silently reporting a stale literal.
    """
    holder = getattr(experiment, group, None) if group != "dynamics" else experiment.dynamics
    params = getattr(holder, "parameters", None)
    if params is None:
        return None
    items = params.items() if hasattr(params, "items") else [
        (getattr(p, "name", None), p) for p in params]
    return next((value_of(p) for n, p in items if n == name), None)


def open_result(out_dir, experiment: str | None = None):
    """The result container of an experiment, or None when it has not been run.

    Network sidecars share the directory and are excluded by name — opening one instead of the
    result is the failure this exists to prevent.
    """
    import xarray as xr

    out_dir = Path(out_dir)
    root = out_dir / "nc" / experiment if experiment else out_dir
    files = [f for f in sorted(root.rglob("*.h5")) if "network" not in f.name]
    return xr.open_dataset(files[0], engine="h5netcdf") if files else None


def result_sidecar(out_dir, experiment: str) -> dict:
    """The YAML sidecar `tvbo run` wrote beside a result, or an empty dict."""
    import yaml

    root = Path(out_dir) / "nc" / experiment
    files = [f for f in sorted(root.glob("*.yaml")) if "network" not in f.name] if root.is_dir() else []
    return yaml.safe_load(files[0].read_text()) if files else {}


def sidecar_value(meta: dict, *path):
    """A value dug out of a sidecar by key path, unwrapping a ``{value: ...}`` leaf."""
    node = meta
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node.get("value") if isinstance(node, dict) else node


def analysis_dataset(out_dir, name):
    """A declared analysis's own container, or None when the analysis has not been run.

    The report reads the same container the figures do, so a number in the prose and the number
    in the panel are the same number — never two computations of one quantity.
    """
    import xarray as xr

    from tvbo.data.dataref import analysis_container_path

    path = analysis_container_path(Path(out_dir), name)
    return xr.open_dataset(path, engine="h5netcdf") if path.exists() else None


def analysis_output(out_dir, name, variable):
    """One named output of a declared analysis, matched by tvbo's own output resolution."""
    from tvbo.data.dataref import match_output

    ds = analysis_dataset(out_dir, name)
    if ds is None:
        return None
    try:
        return ds[match_output(ds.data_vars, variable)]
    except KeyError:
        return None


def analysis_scalar(out_dir, name, variable):
    """One scalar out of a declared analysis, or None when it has not been run."""
    da = analysis_output(out_dir, name, variable)
    return None if da is None else float(da.values)


def crossref_div(identifier: str, content: str, caption: str) -> str:
    """Wrap *content* as a Quarto cross-referenceable float with a COMPUTED caption.

    Quarto's `tbl-cap`/`fig-cap` cell options take a literal string, so a caption holding a
    computed value has to use the cross-reference div instead: the div's last paragraph is the
    caption, and it is ordinary markdown. This is what gives a printed table a real "Table N"
    number and a `@tbl-…` target rather than leaving it captionless in the flow.

    Args:
        identifier: Reference id, e.g. ``"tbl-scorecard"``. Must carry a float prefix
            (``tbl-``, ``fig-``, ``lst-``) or Quarto will not number it.
        content: The table or figure markdown.
        caption: One sentence saying what the reader is looking at.
    """
    return f"::: {{#{identifier}}}\n\n{content.strip()}\n\n{caption.strip()}\n\n:::\n"


def is_internal() -> bool:
    """True in the INTERNAL build — the one allowed to open the paper's © figures.

    Quarto exposes the *input filename* as ``QUARTO_DOCUMENT_FILE``, which is why the
    public/internal split is two entry files rather than two formats in one file.
    """
    import os

    return os.environ.get("QUARTO_DOCUMENT_FILE", "").startswith("report_internal")


def may_show_original(cleared: bool = False) -> bool:
    """Whether this build is permitted to embed the paper's published figure.

    Two grounds, and only two. The **INTERNAL build** is local and git-ignored, so the original
    never leaves the machine. **Documented copyright clearance** from the publisher and the
    authors permits it anywhere, including the shareable PDF — that is a real case, not a
    hypothetical, and a study that has obtained clearance says so by passing ``cleared=True``.

    No study in this repository currently has clearance, so in practice the internal build is
    the only route. Default to ``False``: clearance is something a study proves it has, never
    something the code assumes.
    """
    return bool(cleared) or is_internal()


def figure_label(figure) -> tuple[str, int]:
    """The paper's own label for a figure, parsed from the name the recipe declares.

    Returns ``("Fig", 4)`` or ``("EDF", 10)``, and ``("New", 0)`` for a figure the paper has
    no counterpart for. Sorting on it puts the main-text figures in order, the extended data
    after them and our own last, so a report never hardcodes a figure list.
    """
    match = _FIG_LABEL_RE.search(slot(figure, "name", "") or "")
    return (match.group(1), int(match.group(2))) if match else ("New", 0)


def figure_title(figure) -> str:
    """A figure's heading: the paper's number, or its own name where it has none.

    A replication answers questions the paper left open, and those answers are figures with
    no published counterpart. Titling one with a number would present it as the paper's, so
    an unnumbered figure is headed by its declared ``label:``, or failing that by its own
    name — visibly ours, which is the whole point of the distinction.
    """
    kind, number = figure_label(figure)
    if kind == "EDF":
        return f"Extended Data Fig. {number}"
    if kind == "Fig":
        return f"Figure {number}"
    declared = slot(figure, "label", "") or ""
    if declared:
        return str(declared)
    name = str(slot(figure, "name", "") or "figure").split("_", 1)[-1]
    return name.replace("_", " ").strip().capitalize()


def find_figure(name: str, *studies):
    """The declared figure of that name, across one or more loaded studies."""
    for study in studies:
        for figure in (getattr(study, "figures", None) or []):
            if slot(figure, "name") == name:
                return figure
    return None


def figure_caption(figure, *studies) -> str:
    """A figure's public-facing caption — its own ``description:`` in the recipe.

    Single source of truth: the caption cannot drift from the figure it describes, and it is
    never the paper's caption (that would be plagiarism) nor the internal A/B framing. Accepts
    a figure or its name (with the studies to look it up in — a study may span more than one
    spec). Returns "" for an unknown name, so a caption is missing rather than a crash.
    """
    if isinstance(figure, str):
        figure = find_figure(figure, *studies)
    return " ".join(str(slot(figure, "description", "") or "").split())


_FIGURE_ORDER = {"Fig": 0, "EDF": 1, "New": 2}


def figures_in_paper_order(figures) -> list:
    """The study's figures, ordered as the paper prints them, with ours after."""
    return sorted(figures, key=lambda f: (_FIGURE_ORDER[figure_label(f)[0]],
                                          figure_label(f)[1], slot(f, "name", "")))


def figure_targets(figure, rows: Sequence[dict], column: str = "Fig(s)") -> list[dict]:
    """The declared targets a figure carries, joined on the targets table's own figure column.

    Lets a per-figure status callout be *derived* from the scorecard rather than asserted
    beside it, so the two cannot disagree.
    """
    kind, number = figure_label(figure)
    if kind == "New":
        return []       # a figure the paper never printed carries none of its targets
    want = f"EDF{number}" if kind == "EDF" else str(number)
    hits = []
    for row in rows:
        tokens = (re.match(r"(EDF\d+|\d+)", t.strip()) for t in re.split(r"[,;]", row.get(column, "")))
        if any(m and m.group(1) == want for m in tokens):
            hits.append(row)
    return hits


DIVERGENCE_CLASSES = {
    "A": "Value drift — same symbol, different number",
    "B": "Algorithm substitution — code computes a different operation",
    "C": "Undocumented configuration — never stated at all",
    "D": "Underdetermined prose — several readings, one correct",
    "E": "Convention traps — same name, different meaning",
    "F": "Unreleased — no code to compare against",
}


def divergence_register(source) -> dict:
    """Parse a study's ``methods-vs-code.md`` into per-class counts and rows.

    The register is a skill-mandated artifact of any replication whose study ships code,
    and its counts are quoted in the report's prose. Parsing it here means the report can
    never disagree with the register it cites — the drift the register itself documents.

    Rows are recognised by a leading ``| <class><n> |`` cell. ``material`` counts only rows
    whose final cell opens with a bold "Yes", which is the convention of the classes that
    carry a materiality column; classes without one report ``material`` as ``None`` rather
    than zero, so a caption can say what it actually counted.
    """
    text = Path(source).read_text() if Path(source).exists() else str(source)
    classes: dict[str, dict] = {}
    scores = False
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if cells and cells[0] in ("#", "ID"):
            scores = any(c.lower().startswith("material") for c in cells)
            continue
        m = re.match(r"^\|\s*([A-Z])(\d+)\s*\|", line)
        if not m:
            continue
        entry = classes.setdefault(m.group(1), {"ids": [], "material": None, "rows": []})
        entry["ids"].append(f"{m.group(1)}{m.group(2)}")
        entry["rows"].append(cells)
        if scores:
            entry["material"] = (entry["material"] or 0) + bool(re.match(r"\*\*Yes", cells[-1]))
    for key, entry in classes.items():
        entry["count"] = len(entry["ids"])
        entry["title"] = DIVERGENCE_CLASSES.get(key, "")
    scored = [e for e in classes.values() if e["material"] is not None]
    return {
        "classes": dict(sorted(classes.items())),
        "total": sum(e["count"] for e in classes.values()),
        "material": sum(e["material"] for e in scored),
        "scored": sum(e["count"] for e in scored),
    }


def report_figure(ours, theirs=None, stage=Path("_figures"), credit: str = "the authors",
                  label: str = "", missing: str = "", width: float = 6.7,
                  dpi: int = 300, cleared: bool = False) -> Path | None:
    """The image a report embeds for one figure, staged inside the render project.

    This is the A/B helper every replication report used to carry its own copy of. Pass
    ``theirs=None`` — what the PUBLIC build does — and the copyrighted original is never
    opened, let alone embedded. Pass it in the INTERNAL build and the two are composed
    left-right at a common height. Staging keeps the render reading only from its own project
    directory, and makes the composite a gitignored artifact rather than a committed file.

    Args:
        ours: Our rendered figure. A missing file returns None rather than a blank slot.
        theirs: The published original — one path, or several stacked vertically when the
            paper splits one quantity across scans. None embeds ours alone.
        stage: Directory beside the report to stage into (created if absent).
        credit: Attribution over the original, e.g. ``"Pang et al. 2023 (c)"``.
        label: Qualifier after "TVBO replication", e.g. the parcellation or backend.
        missing: Drawn in the original's pane when it cannot be found, so the A/B still shows
            which side is absent instead of silently rendering as a single panel.
        width: Composite width in inches — the report's text-block width.
        dpi: Raster resolution of the composite.
        cleared: True only when the study holds documented copyright clearance for the
            published figure. Without it, composing an original outside the INTERNAL build
            raises rather than shipping it.

    Returns:
        The staged path to embed, or None when our figure has not been rendered.
    """
    import shutil

    ours, stage = Path(ours), Path(stage)
    if theirs is not None and not may_show_original(cleared):
        raise RuntimeError(
            "refusing to compose a published original into the PUBLIC build. `report.pdf` is "
            "the shareable artifact, so embedding the paper's figure needs one of two grounds: "
            "the INTERNAL build (resolve the reference image behind `if is_internal()`), or "
            "documented copyright clearance from the publisher and authors (`cleared=True`). "
            "See the A/B section of the writing-reports skill.")
    if not ours.is_file():
        return None
    stage.mkdir(parents=True, exist_ok=True)
    if theirs is None:
        return Path(shutil.copyfile(ours, stage / ours.name))

    from tvbo.utils.figure_compare import Pane, side_by_side

    return side_by_side(
        [Pane(theirs, f"Original — {credit}", missing or "original not available"),
         Pane(ours, f"TVBO replication{f' ({label})' if label else ''}")],
        stage / f"{ours.stem}_ab.png", width=width, dpi=dpi)


VERDICTS = {
    "met": "met",
    "short": "short of criterion",
    "out": "not attempted",
    "blocked": "input unobtainable",
}
"""The four outcomes a replication target can have.

`short` is the only one that is a failure of the replication: it was run and missed. `out` was
declared before running and covers two cases its row must separate — the target tests nothing
another target does not, or it is in scope and simply not done yet. `blocked` is an input that
cannot be obtained. Collapsing them lets a scope decision read as a failure, and a failure hide
inside a scope decision.

`out` is labelled *not attempted* rather than *out of scope* precisely because of that second
case: a target still owed is not a target judged unnecessary, and a column header saying
"out of scope" would quietly assert the stronger claim for both.
"""

TIERS = ("core", "extended")
"""How central a target is to the paper's claims — independent of whether it was met.

A tier is not an outcome. Reusing an outcome word (`out`) as a tier makes the tally cross itself:
the same targets appear in a scope row and a status column, and the reader cannot tell whether
the table is counting two things or one.
"""


class Scorecard:
    """A replication's targets, read from the `targets.md` written before anything ran.

    Owns the vocabulary, the tally, the reason register and the figure join, so a report can
    state a verdict only where the targets file supports one — the tally, the per-figure callout
    and the shortfall prose all come from this single reading of that file.

    The file's shape is fixed by the replicating-studies skill: one or more tables carrying a
    `Status` column, plus a register table carrying a `Why it falls short` column keyed by `ID`.
    """

    def __init__(self, source, verdicts: dict | None = None, tiers: Sequence[str] = TIERS):
        tables = read_md_tables(source)
        self.verdicts = dict(verdicts or VERDICTS)
        self.tiers = list(tiers)
        self.rows = [r for t in tables if "Status" in t.headers for r in t.rows]
        self.reasons = {r["ID"]: r[self.WHY]
                        for t in tables if self.WHY in t.headers for r in t.rows}

    WHY = "Why it falls short"

    def _key(self, row) -> int:
        digits = "".join(c for c in row["ID"] if c.isdigit())
        return int(digits) if digits else 0

    def of(self, *verdicts) -> list[dict]:
        """Every target with one of *verdicts*, in target-number order."""
        return sorted((r for r in self.rows if r["Status"].strip() in verdicts), key=self._key)

    def count(self, *verdicts) -> int:
        return len(self.of(*verdicts))

    def verdict(self, row) -> str:
        """A row's outcome, spelled the way the reader sees it."""
        status = row["Status"].strip()
        return self.verdicts.get(status, status)

    def headline(self, row) -> str:
        """The target's headline, without its parenthetical and trailing qualifiers."""
        return row["Target"].split(",")[0].split("(")[0].strip()

    def reason(self, row) -> str:
        return self.reasons.get(
            row["ID"], "No reason is recorded in `targets.md` — that is a gap.")

    def tally_table(self, tier_column: str = "Scope") -> str:
        """Targets counted by tier against outcome — each target in exactly one cell."""
        counts = {tier: {v: 0 for v in self.verdicts} for tier in self.tiers}
        for row in self.rows:
            tier, status = row[tier_column].strip(), row["Status"].strip()
            if tier in counts and status in counts[tier]:
                counts[tier][status] += 1
        body = [[tier, *(counts[tier][v] for v in self.verdicts), sum(counts[tier].values())]
                for tier in self.tiers if sum(counts[tier].values())]
        body.append(["**all**",
                     *(sum(counts[t][v] for t in self.tiers) for v in self.verdicts),
                     len(self.rows)])
        return md_table(["Tier", *self.verdicts.values(), "Total"], body,
                        aligns=["l"] + ["r"] * (len(self.verdicts) + 1))

    def target_table(self, columns: Sequence[str] = ("ID", "Target", "Fig(s)", "Scope",
                                                     "Fidelity", "Status")) -> str:
        """One row per target, with its outcome spelled out."""
        headers = ["Tier" if c == "Scope" else c for c in columns]
        cell = {"Target": self.headline, "Status": self.verdict}
        return md_table(headers,
                        [[cell[c](r) if c in cell else r[c] for c in columns] for r in self.rows],
                        aligns=["l"] * len(headers))

    def for_figure(self, figure, column: str = "Fig(s)") -> list[dict]:
        """Every target a figure carries, joined on the targets table's own figure column."""
        return figure_targets(figure, self.rows, column)

    def figure_callout(self, figure, scored_in: str = "@sec-scorecard") -> str:
        """A figure's verdict, assembled from the outcome of every target it carries.

        Red is reserved for a target that was attempted and missed. A declared scope decision is
        not a failure of the figure, and an unobtainable input is a gap in the data — both are
        yellow, and a figure whose targets all met is green.
        """
        def names(rows):
            ids = sorted((r["ID"] for r in rows), key=lambda s: int("".join(
                c for c in s if c.isdigit()) or 0))
            return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + f" and {ids[-1]}"

        by = {}
        for row in self.for_figure(figure):
            by.setdefault(row["Status"].strip(), []).append(row)
        if not by:
            return ""
        met, short = by.get("met", []), by.get("short", [])
        out, blocked = by.get("out", []), by.get("blocked", [])
        kind = "important" if short else "warning" if (out or blocked) and not met else "note"
        said = []
        if met:
            said.append(f"{names(met)} met")
        if short:
            said.append(f"{names(short)} attempted and short of its criterion")
        if out:
            said.append(f"{names(out)} declared unattempted")
        if blocked:
            said.append(f"{names(blocked)} blocked on an unobtainable input")
        tail = f" Each is scored in {scored_in}." if (short or out or blocked) else ""
        return f"::: {{.callout-{kind}}}\n{'; '.join(said)}.{tail}\n:::\n"

    def shortfall_prose(self) -> str:
        """The shortfall, as one paragraph per outcome — never one undifferentiated list.

        Separate paragraphs are what stop a scope decision reading as a failure. The default
        wording states what each outcome means before naming its targets; reword it in the
        report if a study needs to, but keep the three groups apart.
        """
        def sentences(rows):
            return " ".join(f"**{r['ID']}**, {self.headline(r)}. {self.reason(r)}" for r in rows)

        blocks, groups = [], [
            ("short", "Attempted and {}", "These were run and did not meet the "
             "criterion written for them, so they are the replication's own shortfall."),
            ("out", "Declared {}", "Nothing was attempted here and nothing failed. Each "
             "row says which of two things it is: a target judged, before anything was run, to "
             "add no test of the paper's claims that another target does not already make, or "
             "one that is in scope and simply not done yet. The first is a closed decision, the "
             "second an open commitment, and the row must not blur them."),
            ("blocked", "Blocked — {}", "These would be in scope, and the "
             "method for them is the one already used elsewhere in this replication. What is "
             "missing is data we cannot get."),
        ]
        groups = [(s, t.format(self.verdicts.get(s, s)), lead) for s, t, lead in groups]
        for status, title, lead in groups:
            rows = self.of(status)
            if rows:
                blocks.append(f"**{title} ({len(rows)}).** {lead} {sentences(rows)}\n")
        return "\n".join(blocks)


def show_report_figure(ours, theirs=None, **kwargs) -> None:
    """`report_figure`, displayed in the current cell.

    For reports that emit figures from a plain python cell. Prefer embedding the path
    `report_figure` returns as markdown — that gets a figure number, a caption and a
    cross-reference target; this exists so a report with many inline call sites can share the
    one implementation without restructuring every cell.
    """
    from IPython.display import Image, display

    staged = report_figure(ours, theirs, **kwargs)
    if staged is None:
        print(f"{Path(ours).name} is not rendered")
        return
    display(Image(str(staged)))


# ── Report cell formatters ──────────────────────────────────────────────────
# Shared by the model / experiment / coupling report templates so the table
# building lives here (the adapter) rather than being duplicated in each Mako.


def slot(obj, name, default=None):
    """Safe attribute access on a report object (``getattr`` with a default)."""
    return getattr(obj, name, default) if obj is not None else default


def present(value):
    """True when a value carries information (not ``None`` / empty / ``''``)."""
    return value not in (None, "", [], {})


_SYMBOL_LATEX_FNS = None


def _symbol_latex(text):
    """Render ``text`` as an inline-LaTeX symbol via sympy, imported lazily once.

    sympy is a heavy import deliberately kept out of this module's import path (as
    are the other local imports here), so the ``(Symbol, latex)`` pair is cached on
    first use rather than re-imported per table row.

    sympy renders a symbol *name* verbatim, so a LaTeX-active character in the
    source notation (``% # & $``) survives unescaped and would corrupt the
    enclosing ``$...$`` cell — ``%`` silently comments out the rest of the line,
    ``$`` closes math mode. sympy never emits these for a symbol, so they are
    escaped after rendering: a no-op for ordinary notation (Greek, sub/superscripts,
    ``\\`` commands), whose ``\\ { } _ ^`` sympy emits legitimately and must keep.
    """
    global _SYMBOL_LATEX_FNS
    if _SYMBOL_LATEX_FNS is None:
        from sympy import Symbol, latex

        _SYMBOL_LATEX_FNS = (Symbol, latex)
    Symbol, latex = _SYMBOL_LATEX_FNS
    return re.sub(r"(?<!\\)([%#&$])", r"\\\1", latex(Symbol(text)))


def display_symbol(obj, name):
    """Inline-LaTeX symbol for a report row, preferring an explicit ``symbol`` override.

    When a parameter / variable carries a ``symbol`` slot (e.g. ``w_+`` for an
    identifier ``w_plus``, or ``S^{(E)}`` for ``S_e``), render *that* symbol so the
    report matches the source's own notation; otherwise fall back to the element's
    name. Fully sympy-native — the override string is itself rendered via
    ``sympy.latex(Symbol(...))``, so it inherits Greek/subscript/superscript handling.
    """
    sym = slot(obj, "symbol", None)
    return _symbol_latex(sym) if sym else _symbol_latex(name)


def format_number(value, decimals=4):
    """APA-style numeric formatting for report-table cells.

    Rounds to at most ``decimals`` decimal places and strips trailing zeros, so raw
    floats render publication-clean — ``0.8333333333`` → ``0.8333``,
    ``314.1592653589793`` → ``314.1593``, ``40000.0`` → ``40000``, ``0.0`` → ``0`` —
    while very large or very small magnitudes fall back to scientific notation
    (``1e-06``). Non-numeric values (strings, symbolic expressions, arrays) and
    booleans pass through unchanged.
    """
    if isinstance(value, bool):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    if x != x or x in (float("inf"), float("-inf")):  # nan / inf
        return value
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 10.0 ** (-decimals):
        mant, _, exp = f"{x:.{decimals}e}".partition("e")
        return f"{mant.rstrip('0').rstrip('.')}e{int(exp):+d}"
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


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
    return "$" + unit_ltx + "$" if unit_ltx else ""


def range_text(range_obj):
    """One-line summary of an explored range / domain (values, ``[lo, hi]``, step, n)."""
    if not range_obj:
        return ""
    values = slot(range_obj, "explored_values", None)
    if values:
        values = [str(format_number(v)) for v in values]
        return "{" + ", ".join(values[:8]) + ("..." if len(values) > 8 else "") + "}"
    lo, hi = slot(range_obj, "lo"), slot(range_obj, "hi")
    step, n_points = slot(range_obj, "step"), slot(range_obj, "n")
    parts = []
    if lo is not None or hi is not None:
        parts.append(f"[{format_number(lo) if lo is not None else '-inf'}, {format_number(hi) if hi is not None else 'inf'}]")
    if step is not None:
        parts.append(f"step={format_number(step)}")
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
    return "; ".join(bit for bit in bits if bit) or ""


def flag_text(obj, flags=None):
    """Flags cell: boolean flags + shape / dataset / reported optimum.

    ``flags`` is a list of ``(attr, label)`` pairs; it defaults to the standard
    parameter flags (``free``, ``heterogeneous``). A purely symbolic shape such
    as ``(n_nodes,)`` is skipped: it names the broadcast dimension rather than a
    concrete size, so it carries no information for a reader and would otherwise
    keep an empty Flags column alive. A concrete shape like ``(84, 84)`` is kept.
    """
    flags = _PARAM_FLAGS if flags is None else flags
    labels = [label for name, label in flags if slot(obj, name, False)]
    for attr, key in (("shape", "shape"), ("source", "data"), ("reported_optimum", "optimum")):
        val = slot(obj, attr)
        if val is None or val == "":
            continue
        if attr == "shape" and not any(ch.isdigit() for ch in str(val)):
            continue
        if attr == "source" and not isinstance(val, str):
            # `source` may be a structured object (iri / path / producer); show a
            # concise pointer, never its raw repr.
            val = (getattr(val, "iri", None) or getattr(val, "path", None)
                   or getattr(val, "name", None) or type(val).__name__)
        labels.append(f"{key}={val}")
    return ", ".join(labels) or ""


_STATE_VAR_FLAGS = [("coupling_variable", "coupling"), ("stimulation_variable", "stimulation"), ("record", "recorded")]
_PARAM_FLAGS = [("free", "free"), ("heterogeneous", "heterogeneous")]


def equation_latex(eq, derivative_notation="dot", symbol_names=None, mul_symbol=None):
    """One SymPy equation as LaTeX, with the derivative written the report's way.

    Takes an already-parsed ``Eq`` — never a source string. Re-parsing an authored
    right-hand side needs a symbol vocabulary assembled by hand, and every symbol the
    assembler forgets (an event's name, a coupling term) turns into a silent fall-back to
    raw Python in the middle of the Methods section. ``Dynamics.get_equations()`` has
    already done that resolution against the model's own scope, so this only prints.

    Args:
        eq: A SymPy ``Eq``; a derivative left-hand side gets dot notation.
        derivative_notation: ``"dot"`` for ``\\dot{x}``, anything else for ``dx/dt``.
        symbol_names: ``{Symbol: latex}`` display overrides (``Dynamics.symbol_map()``).
        mul_symbol: Passed through to ``sympy.latex``.
    """
    from sympy import Derivative, Eq, latex

    symbol_names = symbol_names or {}
    if derivative_notation == "dot" and isinstance(eq, Eq) and isinstance(eq.lhs, Derivative):
        deriv = eq.lhs
        base = latex(deriv.expr, mul_symbol=mul_symbol, symbol_names=symbol_names)
        lhs = derivative_latex(base, time_order(deriv))
        return f"{lhs} = {latex(eq.rhs, mul_symbol=mul_symbol, symbol_names=symbol_names)}"
    return latex(eq, mul_symbol=mul_symbol, symbol_names=symbol_names)


def time_order(derivative):
    """How many times *derivative* differentiates with respect to time.

    By name: ``Symbol("t")`` and ``Symbol("t", real=True)`` are different objects
    that print identically, so an identity test reads order 0 for a first
    derivative taken in a scope that carries assumptions — and prints
    ``\\frac{d^0}{d t^0}``.
    """
    return sum(1 for variable in derivative.variables if str(variable) == "t")


def derivative_latex(base, order):
    """``base`` under ``order`` time derivatives, dotted where dots exist."""
    dots = {1: "dot", 2: "ddot", 3: "dddot"}.get(order)
    return f"\\{dots}{{{base}}}" if dots else f"\\frac{{d^{order}}}{{d t^{order}}} {base}"


EQUATION_GROUPS = {
    "state": "state-equations",
    "derived": "derived-variables",
    "derived_parameters": "derived-parameters",
    "functions": "functions",
    "output": "output-transformations",
}
"""Display group → the key :meth:`Dynamics.get_equations` files it under."""


def equation_name(eq):
    """The name an equation defines: ``x`` for ``Eq(Derivative(x, t), …)``, ``Sigm`` for
    ``Eq(Sigm(v), …)``."""
    from sympy import Derivative, Symbol

    lhs = eq.lhs.expr if isinstance(eq.lhs, Derivative) else eq.lhs
    return lhs.name if isinstance(lhs, Symbol) else type(lhs).__name__


def model_equation_groups(model, delta=None):
    """Every equation a model states, grouped for display and already parsed.

    One call to :meth:`Dynamics.get_equations`, which resolves each equation against the
    model's own scope and folds conditional branches into a ``Piecewise``. A template that
    rebuilds any of these groups from ``equation.rhs`` instead re-parses without that scope
    — the mistake :func:`equation_latex` exists to prevent — and drops outright every
    equation written purely as branches, whose ``rhs`` is ``None``.

    Args:
        model: The `Dynamics` to read.
        delta: Optional `model_delta` result; state and derived variables are narrowed to
            the ones it reports as changed, so a derived model shows only its own additions.
    """
    groups = model.get_equations(format="dict")
    equations = {name: list(groups.get(key, [])) for name, key in EQUATION_GROUPS.items()}
    if delta is not None:
        equations["state"] = [eq for eq in equations["state"] if equation_name(eq) in delta.eq_svars]
        equations["derived"] = [eq for eq in equations["derived"] if equation_name(eq) in delta.dvars]
    return equations


def model_equations(model, kind="state", derivative_notation="dot", mul_symbol=None):
    """``(name, latex)`` for a model's equations of one kind, from its symbolic form.

    ``kind`` selects ``state`` (state variables) or ``derived`` (derived variables). The
    equations come from :meth:`Dynamics.get_equations`, so the report shows the same
    expressions the backend integrates. The *name* comes back with the LaTeX because a
    numbered report has to anchor each equation on the variable it defines, and because a
    variant prints only the subset its delta names.
    """
    collection = "state_variables" if kind == "state" else "derived_variables"
    members = getattr(model, collection, None) or {}
    symbol_names = model.symbol_map() if hasattr(model, "symbol_map") else {}
    return [(name, equation_latex(eq, derivative_notation, symbol_names, mul_symbol))
            for name, eq in _equations_of(model).items() if name in members]


def model_equations_latex(model, kind="state", derivative_notation="dot", mul_symbol=None):
    """A model's equations of one kind, each as LaTeX (names dropped)."""
    return [ltx for _, ltx in model_equations(model, kind, derivative_notation, mul_symbol)]


def _model_vocabulary(model):
    """The symbol and function names a model's authored expressions may reference."""
    names = []
    for collection in ("parameters", "derived_parameters", "state_variables",
                       "derived_variables", "coupling_terms", "coupling_inputs"):
        names += [n for n, _ in name_items(slot(model, collection, None))]
    return names, [n for n, _ in name_items(slot(model, "functions", None))]


def model_functions(model, derivative_notation="dot", mul_symbol=None):
    """``(name, latex)`` for a model's named functions, written ``f(args) = rhs``.

    Unlike the state equations these are not in ``get_equations()``, so the authored
    right-hand side is parsed against the model's own vocabulary — assembled from the
    model rather than by hand, so a symbol the assembler would have forgotten cannot fall
    back to raw Python in the middle of the Methods.
    """
    from tvbo.parse.expression import parse_eq

    symbols, functions = _model_vocabulary(model)
    out = []
    for name, func in name_items(slot(model, "functions", None)):
        rhs = slot(slot(func, "equation"), "rhs", "")
        if rhs in (None, ""):
            continue
        args = slot(func, "arguments", None)
        args = list(args.values()) if hasattr(args, "values") else list(args or [])
        args = [str(slot(a, "name", a)) for a in args]
        try:
            body = equation_latex(parse_eq(str(rhs), parameters=symbols + args, functions=functions),
                                  derivative_notation, None, mul_symbol)
        except Exception:
            body = str(rhs)
        out.append((name, f"\\operatorname{{{name}}}({', '.join(args)}) = {body}"))
    return out


def study_sweeps(experiments):
    """Every parameter any of these experiments sweeps, mapped to its range.

    A parameter one experiment sweeps is not well described by the single value another
    happens to hold it at, so the symbol table shows the range instead.
    """
    merged = {}
    for exp in experiments:
        merged.update({k: v for k, v in sweep_axes(exp).items() if v})
    return merged


def _plural(n, noun):
    """``1 state variable`` / ``2 state variables`` — a count with its noun agreed."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def variant_sentence(variant, equations, baseline):
    """The lead-in to a variant's delta: who uses it, and what it changes.

    Replaces reprinting a near-identical system. Where the equations it redefines were
    themselves numbered, they are named by cross-reference, so the reader is pointed at
    the equation above rather than asked to diff two blocks by eye.
    """
    ids = _id_text(slot(e, "id", "") for e in variant.experiments)
    label = slot(variant.model, "label", None) or slot(variant.model, "name", None) or ""
    redefined = sorted(variant.delta.eq_svars - variant.delta.new_svars)
    clauses = []
    if redefined:
        refs = [equations.ref(baseline, key) or f"the equation for ${_symbol_latex(key)}$"
                for key in redefined]
        clauses.append("redefines " + ", ".join(refs))
    if variant.delta.new_svars:
        clauses.append("adds " + _plural(len(variant.delta.new_svars), "state variable"))
    if variant.delta.coupling_inputs:
        clauses.append("adds " + _plural(len(variant.delta.coupling_inputs), "coupling input"))
    if variant.delta.params:
        clauses.append("re-tunes " + _plural(len(variant.delta.params), "parameter"))
    many = len(variant.experiments) != 1
    named = label and label != str(variant.delta.base_label)
    subject = f"**{label}**, which" if named else "a variant of it, which"
    if clauses:
        changes = ", ".join(clauses[:-1]) + (" and " if len(clauses) > 1 else "") + clauses[-1]
        against = f" relative to {variant.delta.base_label}" if named else ""
    else:
        changes, against = "is identical", (f" to {variant.delta.base_label}" if named else "")
    return f"Experiment{'s' if many else ''} {ids} use{'' if many else 's'} {subject} {changes}{against}."


def coupling_of(experiments):
    """The distinct couplings these experiments use, in declared order."""
    seen, out = set(), []
    for exp in experiments:
        candidates = slot(exp, "coupling", None) or slot(slot(exp, "network"), "coupling", None)
        if candidates is None:
            continue
        if hasattr(candidates, "values"):
            candidates = list(candidates.values())
        elif not isinstance(candidates, (list, tuple)):
            candidates = [candidates]
        for cpl in candidates:
            key = str(slot(cpl, "name", None) or slot(cpl, "label", None) or id(cpl))
            if key not in seen:
                seen.add(key)
                out.append(cpl)
    return out


def coupling_prose(experiments, equations=None):
    """Each distinct coupling a family uses, rendered by the coupling's own report.

    ``Coupling.report`` already writes this block — equation, pre/post decomposition
    and incoming states — so the study report calls it rather than carrying a second
    rendering of the same object that could drift from it. Its parameter table is
    suppressed: :func:`symbol_table` lists those symbols with the model's, where they
    are captioned and numbered, and most of them are the model's own.

    Pass the report's *equations* so the coupling equation is numbered into the same
    sequence as the state equations. Without it the coupling is the one display
    equation on the page a reader cannot cite — eleven of them across ten studies,
    and in every case the equation that joins the nodes into a network.
    """
    blocks = []
    for cpl in coupling_of(experiments):
        try:
            blocks.append(cpl.report("markdown", parameters=False, equations=equations).strip())
        except Exception:
            label = slot(cpl, "label", None) or slot(cpl, "name", "coupling")
            blocks.append(f"**Coupling: {label}**")
    return "\n\n".join(b for b in blocks if b)


def event_table(events, derivative_notation="dot"):
    """Markdown table of a model's events (spike conditions, stimuli, resets).

    An event is part of the model's definition — a stimulus protocol is not decoration —
    so it belongs in the report beside the state equations. Its condition and effect are
    rendered symbolically like every other equation.
    """
    from sympy import sympify

    def _expr(obj, *names):
        for n in names:
            e = slot(obj, n)
            if e is None:
                continue
            rhs = slot(e, "rhs", e)
            if rhs in (None, ""):
                continue
            try:
                return f"${equation_latex(sympify(str(rhs)), derivative_notation)}$"
            except Exception:
                return f"`{rhs}`"
        return ""

    rows = []
    for name, ev in name_items(events):
        if str(name).startswith("_"):
            continue
        params = slot(ev, "parameters", None)
        rows.append([
            f"`{name}`",
            str(slot(ev, "event_type", "") or ""),
            _expr(ev, "condition"),
            _expr(ev, "equation", "effect"),
            ", ".join(f"{p} = {format_number(slot(v, 'value', ''))}"
                      for p, v in name_items(params)) if params else "",
            slot(ev, "description", "") or slot(ev, "label", "") or "",
        ])
    return table_or_prose(["Event", "Type", "Condition", "Effect", "Parameters", "Description"], rows)


def state_variable_table(svars):
    """Markdown State-Variables table (empty columns dropped) from a name->obj map."""
    rows = [
        [f"${display_symbol(sv, name)}$", format_number(slot(sv, "initial_value", "")), unit_text(slot(sv, "unit")),
         f"{slot(sv, 'equation_type', 'differential')} (order {slot(sv, 'equation_order', 1)})",
         metadata_text(sv), flag_text(sv, _STATE_VAR_FLAGS),
         slot(sv, "description", "") or slot(sv, "definition", "") or ""]
        for name, sv in name_items(svars)
    ]
    return md_table(["Variable", "Initial Value", "Unit", "Equation", "Domain / Sampling", "Flags", "Description"], rows)


def param_table(collection, name_header="Parameter", symbolic=True, flags=None, derived=None):
    """Markdown table for any parameter-like collection, empty columns dropped.

    Renders the full column set (name, value, default, unit, domain/sampling,
    flags, description) and lets :func:`md_table` drop every column that is empty
    across all rows, so each collection shows only the columns that carry data.
    One builder serves model parameters, coupling terms, and the stimulation,
    integration, noise, and hyperparameter tables, instead of a hand-written
    table per section.

    Args:
        collection: A name->obj map or a list of parameter-like objects.
        name_header: Title of the first (name) column, e.g. ``Term``.
        symbolic: Render the name as inline-LaTeX ``$symbol$`` when true, else plain.
        flags: ``(attr, label)`` pairs for :func:`flag_text`; defaults to the
            standard parameter flags.
        derived: ``name -> unit`` the dimensional check *forced* rather than read,
            from :func:`derived_units`. Such a unit renders parenthesised, so a
            reader can tell what the model states from what its equations imply.
    """
    def _name(name, p):
        return f"${display_symbol(p, name)}$" if symbolic else str(name)

    rows = [
        [_name(name, p), format_number(slot(p, "value", "")), format_number(slot(p, "default", "")),
         unit_text(slot(p, "unit")) or derived_unit_text(derived, name),
         metadata_text(p), flag_text(p, flags),
         slot(p, "description", "") or slot(p, "definition", "") or ""]
        for name, p in name_items(collection)
    ]
    return md_table([name_header, "Value", "Default", "Unit", "Domain / Sampling", "Flags", "Description"],
                    rows, aligns=["l", "r", "l", "l", "l", "l", "l"])


def derived_unit_text(derived, name):
    """A propagation-derived unit in parentheses, or nothing.

    Parenthesised because it is not a claim the model makes: additive homogeneity
    forces it from the quantities beside it. Printing it bare would put a unit
    nobody wrote into the published record.
    """
    latex = (derived or {}).get(str(name))
    return f"(${latex}$)" if latex else ""


def unit_latex(unit):
    """A propagated unit expression as inline LaTeX, named where it has a name.

    Propagation yields `kilogram*meter**2/(1000*ampere*second**3)`; that is `mV`,
    and reads far better said that way. Only where no curated unit matches does
    the base-unit product itself get typeset. Upright roman either way — a unit
    is not a variable, and italic `mV` reads as `m` times `V`.
    """
    from sympy import latex

    from tvbo.utils.units import unit_named, unit_to_latex

    if unit is None:
        return ""
    named = unit_named(unit)
    return unit_to_latex(named) if named else latex(unit)


def derived_units(verdicts):
    """Units the dimensional check *forced*, keyed by quantity name, as LaTeX.

    A quantity beside declared ones in a sum is not free: additive homogeneity
    fixes it exactly. That is worth showing — it is the difference between a
    model whose units are unstated and one whose units are unstatable — but it
    has to be shown as *derived*, never merged into what the model declares.
    """
    from tvbo.analysis.units import quantity_name

    return {
        quantity_name(symbol): unit_latex(unit)
        for verdict in verdicts
        for symbol, unit in verdict.inferred.items()
        if unit is not None
    }


def unit_verdict_table(verdicts):
    """Markdown table of each equation's dimensional standing.

    Three-valued, because two of the three answers are not failures.
    `underdetermined` says the model does not declare enough to check, which is
    the honest answer for the 24 of 39 curated models that declare no units at
    all; reporting those as inconsistent would pressure invented declarations
    into the published record.
    """
    from tvbo.analysis.units import CONSISTENT, INCONSISTENT

    labels = {CONSISTENT: "consistent", INCONSISTENT: "**inconsistent**"}
    rows = [
        [_inline(_verdict_symbol(verdict)), _inline(unit_latex(verdict.unit)),
         labels.get(verdict.status, verdict.status), verdict.detail]
        for verdict in verdicts
    ]
    return md_table(["Equation", "Unit", "Dimensional check", "Detail"], rows)


def _verdict_symbol(verdict):
    """The equation's left-hand side in the report's own notation.

    Built from the quantity's name rather than the SymPy left-hand side: the
    analysis view holds state variables as functions of `t`, and `\\dot{y_0(t)}`
    says the same thing as `\\dot{y_0}` twice.
    """
    from sympy import Derivative

    lhs = getattr(verdict.equation, "lhs", None)
    base = display_symbol(None, verdict.name)
    return derivative_latex(base, time_order(lhs)) if isinstance(lhs, Derivative) else base


def _inline(latex):
    """Wrap already-rendered LaTeX for inline display, or nothing."""
    return f"${latex}$" if latex else ""


def unit_verdicts(model, strictness="dimensional"):
    """Every equation's dimensional verdict, for the tables above.

    Resolved once per report: the check inlines every model function each time
    it runs, and both the verdict table and the derived-unit marking read it.
    """
    from tvbo.analysis.units import check_units

    return check_units(model, strictness=strictness)


def parameter_table(params, derived=None):
    """Markdown model Parameters table (empty columns dropped) from a name->obj map."""
    return param_table(params, name_header="Parameter", derived=derived)


def _scalar(value):
    """A value keyed as a number where it is one, so ``6`` and ``6.0`` are one setting.

    Compared as text they are two, and the report then invents a difference that does
    not exist: Jansen1995's coupling writes ``v0: 6.0`` where its model writes ``6``,
    and the glossary listed $v_0$ twice — once as the model's, once as a symbol the
    coupling supposedly introduces. The conversion is exact, never rounded, so a
    genuine difference in the last decimal still reads as one.
    """
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return str(value)


def _param_signature(p):
    """Everything that makes a parameter's *setting* distinct, for delta and merge tests.

    Value and defining expression alone are not enough: a re-tuned ``distribution``,
    ``domain`` or per-node ``heterogeneous`` flag changes what runs while leaving both
    unchanged. Koller2024's per-node inherent-frequency dispersion differs from its
    baseline in nothing else, and a value-only comparison reports the two models as
    identical — so the variant vanishes from the report rather than being described.
    """
    return (_scalar(slot(p, "value", "")), str(slot(slot(p, "equation"), "rhs", "")),
            str(slot(p, "distribution", "")), str(slot(p, "domain", "")),
            bool(slot(p, "heterogeneous", False)))


def model_delta(model, baseline):
    """Names of what *model* adds or changes relative to *baseline*.

    Compares two related models (e.g. a controlled variant against its
    uncontrolled base) and returns the subsets that are new or redefined, so a
    report can render only the **delta** instead of repeating every shared state
    variable, parameter, derived variable and coupling input.

    Returns a :class:`~types.SimpleNamespace` with:

    - ``eq_svars`` — state variables whose *equation* is new or changed (shown
      in *State Equations*).
    - ``new_svars`` — state variables absent from the baseline (shown in the
      *State Variables* table; a merely re-tuned equation is not a new variable).
    - ``dvars`` — derived variables that are new or redefined.
    - ``params`` — parameters that are new or whose value/equation changed.
    - ``coupling_inputs`` — coupling inputs absent from the baseline.
    - ``base_label`` — a human label for the baseline, for the "relative to" note.
    """
    from types import SimpleNamespace

    def _keyed(coll):
        if not coll:
            return {}
        if hasattr(coll, "items"):
            return dict(coll.items())
        return {slot(v, "name", i): v for i, v in enumerate(coll)}

    b_eqs, m_eqs = _equations_of(baseline), _equations_of(model)

    def _rhs(eqs, key):
        eq = eqs.get(key)
        return str(getattr(eq, "rhs", eq))

    _psig = _param_signature

    b_sv, m_sv = _keyed(slot(baseline, "state_variables", {})), _keyed(slot(model, "state_variables", {}))
    b_dv, m_dv = _keyed(slot(baseline, "derived_variables", {})), _keyed(slot(model, "derived_variables", {}))
    b_p, m_p = _keyed(slot(baseline, "parameters", {})), _keyed(slot(model, "parameters", {}))
    b_ci = set(_keyed(slot(baseline, "coupling_inputs", {})))

    return SimpleNamespace(
        eq_svars={k for k in m_sv if k not in b_sv or _rhs(m_eqs, k) != _rhs(b_eqs, k)},
        new_svars={k for k in m_sv if k not in b_sv},
        dvars={k for k in m_dv if k not in b_dv or _rhs(m_eqs, k) != _rhs(b_eqs, k)},
        params={k for k, p in m_p.items() if k not in b_p or _psig(p) != _psig(b_p[k])},
        coupling_inputs={k for k in _keyed(slot(model, "coupling_inputs", {})) if k not in b_ci},
        base_label=slot(baseline, "label", None) or slot(baseline, "name", "the base model"),
    )


def experiment_models(experiment):
    """Every distinct model an experiment integrates — one, or one per node.

    ``SimulationExperiment.dynamics`` carries the single model of a homogeneous network.
    A heterogeneous one leaves it unset and declares a model per node (Deco2014's spiking
    populations, Mongillo2008's pre/post pair), either inline or by name into the
    network's own catalogue, so a report reading only the former describes nothing at all
    for those studies.
    """
    single = slot(experiment, "dynamics")
    if single is not None:
        return [single]
    net = slot(experiment, "network") or slot(experiment, "connectivity")
    catalogue = dict(name_items(slot(net, "dynamics", None)))
    out, seen = [], set()
    for node in (slot(net, "nodes", None) or []):
        model = slot(node, "dynamics", None)
        if isinstance(model, str):
            model = catalogue.get(model)
        key = str(slot(model, "name", None) or slot(model, "label", None) or id(model))
        if model is not None and key not in seen:
            seen.add(key)
            out.append(model)
    return out or list(catalogue.values())


def _equations_of(model):
    """A model's symbolic equations, however it was built.

    Every ``Dynamics`` answers ``get_equations`` — the generated forms carry it from
    ``DynamicsBehaviour`` — so a per-node model on a heterogeneous network is read where
    it stands, against the same parse the backend integrates. Treating it as "no
    equations" printed a Methods section with **no mathematics at all** for every
    heterogeneous study: Mongillo2008's twenty-nine experiments got symbol tables and
    nothing else, and Deco2014 lost both of its spiking families.

    The promotion below is for something that is not a ``Dynamics`` at all: the authored
    right-hand sides must not be re-parsed here against a hand-assembled vocabulary (see
    :func:`equation_latex`), so ``Dynamics.from_datamodel`` gives back the same resolution
    instead. A model that still cannot resolve is a legitimate miss, not a crash.
    """
    if hasattr(model, "get_equations"):
        return model.get_equations() or {}
    try:
        from tvbo.classes.dynamics import Dynamics

        return Dynamics.from_datamodel(model).get_equations() or {}
    except Exception:
        return {}


def _model_signature(model):
    """What makes two models the same model, for collapsing repeats in a report."""
    eqs = _equations_of(model)
    return (str(slot(model, "label", "") or slot(model, "name", "")),
            tuple(sorted((k, str(getattr(v, "rhs", v))) for k, v in (eqs or {}).items())),
            tuple(sorted((k, _param_signature(p))
                         for k, p in name_items(slot(model, "parameters", {})))))


def model_families(experiments):
    """The experiments' models, grouped into families, each printed once.

    A **family** is one system: its first model is written out in full, and every later
    model in it contributes only its :func:`model_delta`, the way Jansen1995 §3.3 adds a
    flash stimulus by printing Eq. 18 alone instead of reprinting the six-equation column.
    Two models share a family when they span the same state variables, which also settles
    the case a delta cannot: a model that introduces the *entire* state is not a variant
    but a second system — Pang2023's mass model against its wave field, Koller2024's
    Jansen–Rit against its Kuramoto — and starts a family of its own, printed in full.

    Membership is tested by **subset-or-superset** of the family's first model, not by
    equality and not by mere overlap. Equality splits Jansen1995's delayed column off from
    the column it extends (it only adds ``z0``/``z1``); bare overlap goes wrong the other
    way and merges genuinely unrelated systems that happen to share auxiliary state —
    Pang2023's wave field and its BEI mass model both carry the four Balloon–Windkessel
    haemodynamic variables, and overlap presented the mass model, which the paper never
    even deposited, as a *variant* of the wave field.

    Returns one namespace per family, in the order the experiments declare them, with
    ``label``, ``base`` and ``variants`` (each a namespace of ``model``, ``experiments``
    and, for a variant, its ``delta`` against the base), plus the family's own
    ``experiments`` and ``shared_parameters`` — the parameter names every member defines,
    which are the only ones an experiment table can compare without leaving holes.
    """
    from types import SimpleNamespace

    def _same_system(base, state):
        """True when *state* extends or restricts *base* rather than replacing it."""
        if not base and not state:
            return True
        return bool(base) and bool(state) and (state <= base or base <= state)

    families = []
    for exp, model in ((e, m) for e in experiments for m in experiment_models(e)):
        state = frozenset(dict(name_items(slot(model, "state_variables", {}) or {})))
        family = next((f for f in families if _same_system(f.base_state, state)), None)
        if family is None:
            family = SimpleNamespace(base_state=state, models=[], experiments=[])
            families.append(family)
        # Once per experiment, not once per model: a heterogeneous experiment contributes
        # several models to the same family — Deco2014's spiking columns declare a dozen —
        # and appending per model gave its comparison table six rows for three experiments.
        if exp not in family.experiments:
            family.experiments.append(exp)
        signature = _model_signature(model)
        for entry in family.models:
            if entry.signature == signature:
                if exp not in entry.experiments:
                    entry.experiments.append(exp)
                break
        else:
            family.models.append(SimpleNamespace(model=model, signature=signature,
                                                 experiments=[exp], delta=None))

    out = []
    for family in families:
        family.base = family.models[0]
        family.variants = family.models[1:]
        for entry in family.variants:
            entry.delta = model_delta(entry.model, family.base.model)
        family.label = (slot(family.base.model, "label", None)
                        or slot(family.base.model, "name", None) or "Model")
        shared = None
        for entry in family.models:
            names = set(dict(name_items(slot(entry.model, "parameters", {}) or {})))
            shared = names if shared is None else (shared & names)
        family.shared_parameters = shared or set()
        out.append(family)
    return out


def _initial_text(sv):
    """A state variable's starting value: its distribution, else the value it starts at.

    An undeclared ``initial_value`` still starts somewhere — ``tvbo.utils.initial_value``
    supplies the fallback the run actually uses — so the report prints what is integrated
    rather than leaving the cell blank. Today's report drops the column entirely whenever
    no model declares one, which hides the starting state from every reader.
    """
    from tvbo.utils import initial_value

    if present(slot(sv, "distribution")):
        return distribution_text(slot(sv, "distribution"))
    return str(format_number(initial_value(sv)))


def _value_text(p, swept=None):
    """A parameter's setting: its swept range where a sweep replaces it, else its value."""
    if swept:
        return swept
    value = format_number(slot(p, "value", ""))
    dist = distribution_text(slot(p, "distribution")) if present(slot(p, "distribution")) else ""
    if dist and value in ("", None):
        return dist
    return f"{value} ({dist})" if dist else str(value)


def _meaning(obj, name=""):
    """The glossary cell for a symbol: what the recipe says it means, or nothing.

    Deliberately does *not* fall back to the symbol's own name — printing "y0" as the
    meaning of $y_0$ fills the cell without informing anyone, and hides from the author
    that the description is missing.
    """
    return str(slot(obj, "description", "") or slot(obj, "definition", "")
               or slot(obj, "label", "") or "")


def symbol_table(model, swept=None, couplings=()):
    """One dense glossary of every symbol in a model: state, parameters, derived, coupling.

    ``Symbol | Kind | Meaning | Value | Unit``, where every row fills every cell — a state
    variable contributes the value it starts at, a parameter its value (or the range a
    sweep gives it), a derived parameter its defining expression. Replaces the three
    separate tables (state variables, parameters, derived parameters), whose column sets
    do not overlap and which therefore cannot be merged without leaving most of the grid
    empty. Derived *variables* are deliberately absent: they are equations and appear as
    equations, so listing them here would print each one twice.

    A coupling's parameters land here too, rather than in a table of their own after the
    coupling block: they are symbols of the same system, and a coupling usually restates
    the model's. Jansen1995's sigmoid coupling declares $e_0$, $r$ and $v_0$ at exactly
    the model's values — a separate table repeated three rows the reader had already
    read. A coupling parameter that genuinely differs, or that the model does not
    declare, keeps its row and is marked as the coupling's.

    Args:
        model: The ``Dynamics`` to describe.
        swept: Optional ``{parameter name: range text}``, so a parameter an experiment
            sweeps shows the range it takes rather than a single value it never holds.
        couplings: Couplings whose parameters belong to the same system.
    """
    swept = swept or {}
    rows = []
    for name, sv in name_items(slot(model, "state_variables", {})):
        flags = flag_text(sv, _STATE_VAR_FLAGS)
        rows.append([f"${display_symbol(sv, name)}$", f"state ({flags})" if flags else "state",
                     _meaning(sv, name), _initial_text(sv), unit_text(slot(sv, "unit"))])
    for name, p in name_items(slot(model, "parameters", {})):
        flags = flag_text(p, _PARAM_FLAGS)
        kind = "parameter (swept)" if name in swept else (f"parameter ({flags})" if flags else "parameter")
        rows.append([f"${display_symbol(p, name)}$", kind, _meaning(p, name),
                     _value_text(p, swept.get(name)), unit_text(slot(p, "unit"))])
    for name, dp in name_items(slot(model, "derived_parameters", {})):
        rhs = slot(slot(dp, "equation"), "rhs", "")
        rows.append([f"${display_symbol(dp, name)}$", "derived", _meaning(dp, name),
                     f"${_safe_latex(rhs)}$" if rhs != "" else "", unit_text(slot(dp, "unit"))])
    declared = dict(name_items(slot(model, "parameters", {})))
    seen = set()
    for cpl in couplings:
        for name, p in name_items(slot(cpl, "parameters", {})):
            if name in seen or (name in declared and _param_signature(p) == _param_signature(declared[name])):
                continue
            seen.add(name)
            rows.append([f"${display_symbol(p, name)}$", "coupling", _meaning(p, name),
                         _value_text(p, swept.get(name)), unit_text(slot(p, "unit"))])
    if not rows:
        return ""
    return table_or_prose(["Symbol", "Kind", "Meaning", "Value", "Unit"], rows,
                    aligns=["l", "l", "l", "r", "l"])


def _safe_latex(expression):
    """A derived parameter's right-hand side as LaTeX, falling back to its source text."""
    from sympy import sympify

    try:
        return equation_latex(sympify(str(expression)))
    except Exception:
        return str(expression)


def _axis_range(axis):
    """An exploration axis's extent, whether it lists values or bounds a domain."""
    if present(slot(axis, "explored_values")):
        return range_text(axis)
    return range_text(slot(axis, "domain")) or ""


def sweep_axes(experiment):
    """``{axis name: range text}`` for every parameter an experiment explores.

    Axis names are scoped (``network.G``, ``execution.random_seed``); a bare name is a
    model parameter, which is what lets a swept parameter show its *range* in the symbol
    and experiment tables instead of a single value it never actually holds.
    """
    return {slot(axis, "name", None) or str(name): _axis_range(axis)
            for exploration in (slot(experiment, "explorations", None) or [])
            for name, axis in name_items(slot(exploration, "parameters", None))}


def time_text(value, unit=None, decimals=4):
    """A time with the integrator's own unit, never an assumed one.

    The integration block used to print a hardcoded ``ms``, so Jansen1995 — whose rate
    constants are per second and whose recipe declares 2.0 — reported a 2 ms run of a
    model that integrates for 2 s, with a 0.5 ms transient in place of 0.5 s.
    """
    if value in (None, ""):
        return ""
    return f"{format_number(value, decimals)} {unit or 's'}".strip()


def _speed_text(conduction_speed):
    """A conduction speed with its own unit, or "none" where the network has no delays.

    A speed is not a time: routing it through :func:`time_text` prints "3 mm_per_ms" or,
    where the unit is undeclared, the seconds default applied to a velocity.
    """
    value = slot(conduction_speed, "value", None)
    if not value:
        return "none"
    unit = str(slot(conduction_speed, "unit", None) or "mm/ms").replace("_per_", "/")
    return f"{format_number(value)} {unit}"


def _reference_text(experiment):
    """The paper figure(s) an experiment targets, from its declared references."""
    refs = slot(experiment, "references", None) or []
    refs = refs if isinstance(refs, (list, tuple)) else [refs]
    out = [str(slot(r, "label", None) or slot(r, "name", None) or r).strip() for r in refs]
    return ", ".join(r for r in out if r)


def experiment_facts(experiment, shared_parameters=()):
    """Ordered ``{column: cell}`` of everything an experiment can differ from its siblings in.

    Spans the model parameters the family shares, the network, the integrator and the
    sweep — because the differences that matter are rarely all of one kind: Jansen1995's
    seven experiments differ in node count, delay and duration and in **not one** model
    parameter, so a parameters-only comparison would come out blank.

    Only parameters *every* member of the family defines are included. A parameter a
    variant introduces has no counterpart in the base and would leave a hole in the
    column, which is the one thing a merged table must not do; the variant's own delta
    describes it instead.
    """
    net, integ = (slot(experiment, "network") or slot(experiment, "connectivity")), slot(experiment, "integration")
    unit = slot(integ, "unit", None)
    swept = sweep_axes(experiment)
    facts = {"Exp": str(slot(experiment, "id", "")), "Fig": _reference_text(experiment)}

    nodes = slot(net, "number_of_nodes", None) or slot(net, "number_of_regions", None)
    if nodes is None and present(slot(net, "nodes", None)):
        nodes = len(slot(net, "nodes"))
    facts["Nodes"] = "" if nodes is None else str(nodes)
    facts["Delays"] = _speed_text(slot(net, "conduction_speed"))
    facts["Method"] = str(slot(integ, "method", "") or "")
    facts["$\\Delta t$"] = time_text(slot(integ, "step_size", None), unit)
    facts["Duration"] = time_text(slot(integ, "duration", None), unit)
    facts["Transient"] = time_text(slot(integ, "transient_time", None), unit)

    params = dict(name_items(slot(slot(experiment, "dynamics"), "parameters", {})))
    for name in sorted(shared_parameters):
        facts[f"${_symbol_latex(name)}$"] = _value_text(params[name], swept.get(name)) if name in params else ""
    for axis, extent in swept.items():
        if axis in params:
            continue
        facts[f"${_symbol_latex(axis.split('.')[-1])}$"] = extent
    facts["Part"] = str(slot(experiment, "part", "") or "main")
    return facts


def experiment_table(experiments, shared_parameters=(), orient="auto", caption_only_varying=True):
    """One table comparing a family's experiments, carrying only what actually varies.

    Every quantity constant across the experiments is dropped: it is stated once in the
    symbol table or the settings sentence, and repeating it down a column says nothing.

    Orientation is chosen to keep the table narrow, because columns are the axis that
    cannot be paged: whichever of the two axes is shorter becomes the columns, ties going
    to experiments-as-rows (the conventional study-design layout). Pass ``orient`` as
    ``"rows"`` or ``"columns"`` — meaning where the *experiments* go — to pin it, so a
    study's Methods keeps its shape when a seventh experiment lands.
    """
    facts = [experiment_facts(exp, shared_parameters) for exp in experiments]
    if not facts:
        return ""
    keys = list(facts[0])
    if caption_only_varying:
        keys = [k for k in keys
                if k in ("Exp",) or len({f.get(k, "") for f in facts}) > 1]
    if orient == "auto":
        orient = "rows" if len(facts) >= len(keys) - 1 else "columns"
    if orient == "rows":
        return table_or_prose(keys, [[f.get(k, "") for k in keys] for f in facts])
    head = [keys[0]] + [f.get(keys[0], "") for f in facts]
    return table_or_prose(head, [[k] + [f.get(k, "") for f in facts] for k in keys[1:]])


def experiment_title(experiment):
    """An experiment's heading text, without the id the heading already carries.

    Recipes commonly open a label with the experiment's own number, so the heading came
    out as "Experiment 30: Exp 30 — FIC+EIB tuning". Six of Schirner2023's ten read that
    way. Stripping the prefix also drops the dash the recipe used to attach it.
    """
    label = str(slot(experiment, "label", "") or "").strip()
    ident = re.escape(str(slot(experiment, "id", "")))
    stripped = re.sub(rf"^(?:exp(?:eriment)?\.?\s*){ident}\b\s*[-–—:.]*\s*", "", label,
                      flags=re.IGNORECASE)
    return stripped or label or "simulation"


def settings_sentence(experiment):
    """The factual half of an experiment's paragraph, composed from what it declares.

    Solver, step, duration, transient, network size and swept range are stated here so a
    recipe's authored ``description:`` never has to restate them — the numbers a
    description repeats are the numbers that go stale when the recipe changes. What the
    description says about *why* an experiment exists is left untouched.
    """
    net, integ = (slot(experiment, "network") or slot(experiment, "connectivity")), slot(experiment, "integration")
    unit = slot(integ, "unit", None)
    nodes = slot(net, "number_of_nodes", None) or slot(net, "number_of_regions", None)
    if nodes is None and present(slot(net, "nodes", None)):
        nodes = len(slot(net, "nodes"))

    clauses = []
    if nodes:
        clauses.append(f"{nodes} node" + ("s" if int(nodes) != 1 else ""))
    if slot(integ, "method", None):
        step = time_text(slot(integ, "step_size", None), unit)
        clauses.append(f"{slot(integ, 'method')} at $\\Delta t$ = {step}" if step else str(slot(integ, "method")))
    if slot(integ, "duration", None) is not None:
        clauses.append(f"run for {time_text(slot(integ, 'duration'), unit)}")
    if slot(integ, "transient_time", None):
        clauses.append(f"the first {time_text(slot(integ, 'transient_time'), unit)} discarded")
    swept = sweep_axes(experiment)
    if swept:
        clauses.append("sweeping " + ", ".join(f"${_symbol_latex(a.split('.')[-1])}$ over {r}"
                                               for a, r in swept.items()))
    if not clauses:
        return ""
    sentence = ", ".join(clauses) + "."
    return sentence[:1].upper() + sentence[1:]


_SAMPLING_SLOTS = (("period", "period"), ("downsample_period", "downsample"),
                   ("aggregation", "aggregation"), ("imaging_modality", "modality"),
                   ("time_unit", "scale"), ("voi", "VOI"), ("skip_t", "skip"),
                   ("tail_samples", "tail"), ("window_size", "window"))


def pipeline_text(pipeline):
    """A pipeline as arrow-separated step names.

    A step is named by what the recipe *calls* it, not by the library function it
    happens to dispatch to. Reading ``callable`` first printed Deco2014's five-step
    BOLD pipeline as ``? → ? → fftconvolve → ? → ?`` — every step declares a ``name``
    and only the convolution also names an implementation, so the one step that
    resolved showed a scipy entry point where the reader wanted "convolve".
    """
    steps = pipeline if isinstance(pipeline, (list, tuple)) else ([pipeline] if pipeline else [])
    names = [slot(s, "name", None) or slot(s, "label", None) or slot(s, "function", None)
             or slot(slot(s, "callable", None), "name", None)
             or slot(slot(s, "class_call", None), "name", None) for s in steps]
    return " → ".join(str(n) if n else "?" for n in names)


def _id_text(ids):
    """Experiment ids in numeric order, comma-separated.

    Ordering them as text gives ``1, 2, 20, 21, 3, 30`` — Deco2014 records the same
    three observables across ten experiments and listed them in exactly that order.
    """
    def key(i):
        s = str(i)
        return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)

    return ", ".join(sorted((str(i) for i in ids), key=key))


def _observation_row(name, obs, observed_names):
    """One observation's cells, primary or derived — the two differ only in *Source*.

    Sampling arrives as ``(label, value)`` pairs rather than joined text so the table
    can drop the ones every observation shares.
    """
    sources = slot(obs, "source", None)
    sources = sources if isinstance(sources, (list, tuple)) else ([sources] if sources else [])
    names = [str(slot(s, "name", None) or s) for s in sources]
    derived = any(n in observed_names for n in names)
    sampling = tuple((label, str(slot(obs, attr))) for attr, label in _SAMPLING_SLOTS
                     if slot(obs, attr, None) is not None)
    return (str(slot(obs, "label", None) or name),
            ", ".join(names) if derived else _source_text(names),
            sampling, pipeline_text(slot(obs, "pipeline", [])),
            str(slot(obs, "description", "") or ""))


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _source_text(names):
    """An observation's sources: symbols as math, anything else as literal code.

    A source is usually a state variable, but it can equally be a pointer into external
    data — ``phenotype:Schirner2023_HCPYA_phenotype#PMAT24_A_RTCR``, or a dotted path like
    ``network.observations.BoldCorrelation``. Typesetting those as math asks the engine to
    parse ``#`` and ``:`` as operators; pandoc rejected the first outright ("unexpected
    '#'") and passed it through as raw TeX, and the second rendered as a product of
    variables named after its path segments.
    """
    if not names:
        return ""
    return ", ".join(f"${_symbol_latex(n)}$" if _IDENTIFIER.fullmatch(n) else f"`{n}`"
                     for n in names)


class Observations(NamedTuple):
    """What a study records: the grid, the settings it shares, and its long-form notes."""

    table: str
    shared: str
    notes: str


def observation_table(experiments):
    """Everything the study records, as one table plus the prose the table cannot hold.

    Primary and derived observations share a column set — a derived one names source
    observations where a primary one names a state expression — so they merge into one
    table rather than two half-empty ones. Identical observations collapse onto a single
    row listing the experiments that declare them, which is where the real duplication
    sits: a ten-experiment study usually records the same two things ten times.

    Three things keep the grid dense, measured across the studies that have the widest
    ones (Deco2014's 29 observations, Schirner2023's 34):

    - **Shared settings are lifted out.** A sampling setting every observation agrees on
      is stated once instead of per row. The one that matters is ``time_unit``: a study
      declares its clock once, so the same value repeated on every one of those 63 rows
      said nothing that the line above the table could not.
    - **Sampling and pipeline are one column.** Each was under half full and they are
      complementary: both answer *how the raw state becomes the reported quantity*.
      Apart they left a 34 %-empty grid; merged, ``Reduction`` fills 66–91 %.
    - **Descriptions become prose.** They are paragraphs — the Balloon–Windkessel note
      runs to four lines — in a column filled by 12 % of Schirner2023's rows. As a cell
      they widen the table for everyone; below it they read as text.
    """
    rows = {}
    for exp in experiments:
        observations = slot(exp, "observations", None) or {}
        observed = set(dict(name_items(observations)))
        for name, obs in name_items(observations):
            cells = _observation_row(name, obs, observed)
            rows.setdefault(cells, []).append(str(slot(exp, "id", "")))
    if not rows:
        return Observations("", "", "")

    per_row = [dict(cells[2]) for cells in rows]
    shared = {k: v for k, v in per_row[0].items() if all(r.get(k) == v for r in per_row)}

    def _reduction(cells):
        return "; ".join(part for part in
                         (", ".join(f"{k}={v}" for k, v in cells[2] if k not in shared), cells[3])
                         if part)

    table = table_or_prose(["Observation", "Experiments", "Source", "Reduction"],
                     [[cells[0], _id_text(ids), cells[1], _reduction(cells)]
                      for cells, ids in rows.items()])
    notes = "\n\n".join(f"**{cells[0]}.** {cells[4]}" for cells in rows if cells[4])
    return Observations(table, ", ".join(f"{k} = {v}" for k, v in shared.items()), notes)


def _slug(text):
    """A stable cross-reference slug: lowercase, non-alphanumerics collapsed to hyphens."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(text).lower())).strip("-") or "x"


def section_slug(text):
    """An ASCII anchor for a generated heading, so no renderer has to derive one.

    Left to itself Quarto builds a heading's identifier from the heading *text*, which
    here is recipe-authored and may hold anything: Cortes2013 labels an experiment with
    ``I₀``, and the derived Typst label ``<…-in-i₀-…>`` failed the compile outright with
    "unclosed label". Emitting our own slug keeps the identifier alphanumeric whatever the
    label says, and makes it stable — it no longer changes when someone edits the wording.
    """
    return _slug(text)


class Equations:
    """Numbering and cross-reference labels for one rendered report.

    A report that prints ``$$...$$`` and nothing else cannot be referred to: Jansen1995
    numbers 19 equations and its prose says "Eq. 3" and "Eqs. 15-17", and our render had
    no way to point at any of them. This assigns each equation a display number and,
    where the target format supports one, an anchor.

    ``style`` is the caller's choice, per the ``equations=`` argument:

    - ``semantic`` — anchor from the model and the variable (``#eq-jansenrit-y3``), so a
      reference survives an experiment being added or reordered ahead of it.
    - ``sequential`` — anchor is the number (``#eq-4``); reads naturally in raw markdown
      but silently repoints every reference when an equation is inserted before it.
    - ``none`` — no numbering at all.

    ``format`` picks the syntax: ``qmd`` emits Quarto's ``{#eq-...}``, everything else
    emits ``\\tag{n}`` inside the math, which MathJax and pandoc-to-LaTeX both honour.
    """

    def __init__(self, style="semantic", format="markdown", prefix=""):
        self.style, self.format, self.prefix = style, format, prefix
        self.n = 0
        self.anchors = {}
        self._used = set()

    def unique_anchor(self, anchor):
        """*anchor*, suffixed if this report already used it.

        A variant may share its base model's name — Pang2023's haemodynamic wave model is
        declared under the same name as the wave model it extends — and both redefine the
        same state variable. Emitting the anchor twice makes every reference to it
        ambiguous, and Quarto resolves the duplicate silently. Tables register here too,
        so one report has one namespace: Mongillo2008 declares four distinct ``Ext_E``
        input models and would otherwise emit ``#tbl-delta-ext-e`` four times.
        """
        candidate, n = anchor, 2
        while candidate in self._used:
            candidate, n = f"{anchor}-{n}", n + 1
        self._used.add(candidate)
        return candidate

    def block(self, expression, model=None, key=None):
        """Render one display equation, numbered and anchored per this report's style."""
        if self.style == "none":
            return f"$${expression}$$"
        self.n += 1
        if self.style == "sequential":
            anchor = f"eq-{self.prefix}{self.n}" if self.prefix else f"eq-{self.n}"
        else:
            stem = _slug(slot(model, "name", None) or slot(model, "label", None) or "model")
            anchor = f"eq-{self.prefix}{stem}-{_slug(key or self.n)}"
        anchor = self.unique_anchor(anchor)
        if key is not None and model is not None:
            self.anchors[(id(model), str(key))] = anchor
        if self.format == "qmd":
            return f"$${expression}$$ {{#{anchor}}}"
        return f"$${expression} \\tag{{{self.n}}}$$"

    def ref(self, model, key):
        """A cross-reference to an equation already rendered, or "" if it was not."""
        anchor = self.anchors.get((id(model), str(key)))
        if not anchor:
            return ""
        return f"@{anchor}" if self.format == "qmd" else f"Eq. ({anchor.rsplit('-', 1)[-1]})"


_PYTHON_CELL = re.compile(r"```+\s*\{python\}.*?```+", re.S)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)


def unrendered_equations(source):
    """Display equations written by hand in a report body, as ``(line, equation)``.

    An equation belongs in a report only if the code runs it, and it gets there by being
    rendered from the recipe — never typed. A typed one can drift from what executes, and
    the reader has no way to tell which they are looking at. Pang2023 carried the paper's
    PDE, hand-set, above a section explaining that TVBO does not integrate that PDE.

    Executable cells are stripped first, so equations that :meth:`SimulationStudy.report`
    emits are not flagged: the check is for ``$$…$$`` typed into the prose.

    Assert this is empty in the report's own harness cell, so a hand-written equation
    fails the render rather than reaching a reader::

        bad = report.unrendered_equations("report.qmd")
        assert not bad, f"hand-written equations: {bad}"

    Args:
        source: Path to a ``.qmd``/``.md`` file, or its text.

    Returns:
        ``(line number, equation source)`` for each hand-written display equation, in
        document order. Empty when the report renders all of its mathematics.
    """
    text = Path(source).read_text(encoding="utf-8") if _looks_like_path(source) else str(source)
    prose = _PYTHON_CELL.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return [(prose[:m.start()].count("\n") + 1, " ".join(m.group(1).split()))
            for m in _DISPLAY_MATH.finditer(prose)]


def _looks_like_path(source):
    """Whether *source* names a file rather than being the document text itself."""
    if isinstance(source, Path):
        return True
    text = str(source)
    return "\n" not in text and len(text) < 4096 and Path(text).is_file()


def captioned(table, caption, anchor, format="markdown", anchors=None):
    """A table with its caption and cross-reference anchor attached.

    An uncaptioned table is a wall of numbers, and in LaTeX it still steps the table
    counter — which is why a Methods section that emitted thirty anonymous tables pushed
    the first captioned table in Results out to "Table 34". Captioning them removes the
    need for the counter reset rather than working around it.

    Pass ``anchors`` (the report's :class:`Equations`) so tables share the equations'
    anchor namespace and a repeated model name cannot mint the same ``#tbl-`` twice.

    Input that is not a table — what `table_or_prose` returns for a grid too small to
    earn a float — passes through uncaptioned, because captioning prose would announce a
    table the reader cannot see and, in LaTeX, number one that was never typeset.
    """
    text = str(table).strip()
    if not text:
        return ""
    if not text.startswith("|"):
        return f"{table}\n"
    if format != "qmd":
        return f"{table}\n\n**Table.** {caption}\n"
    label = f"tbl-{_slug(anchor)}"
    label = anchors.unique_anchor(label) if anchors is not None else label
    return f"{table}\n\n: {caption} {{#{label}}}\n"


def variant_parameter_table(family):
    """One table of every parameter the family's variants change, not one table each.

    A study that varies a model across many experiments produces many two-row deltas —
    Mongillo2008 emitted twenty-one of them — and a page of two-row tables is not a
    readable Methods section. Collapsing them keeps the same information in one grid, and
    the *Variant* column carries what the separate captions used to.

    With only one variant that column has one value, repeated down the page to say what
    the sentence introducing the variant said a line earlier. It is left blank so
    :func:`md_table` drops it, which is also what stops a long model label from taking a
    third of the table's width.
    """
    contributors = [e for e in family.variants if e.delta.params]
    rows = []
    for entry in contributors:
        label = slot(entry.model, "label", None) or slot(entry.model, "name", None) or "variant"
        ids = _id_text(slot(e, "id", "") for e in entry.experiments)
        cell = "" if len(contributors) == 1 else f"{label} (exp {ids})"
        params = dict(name_items(slot(entry.model, "parameters", {})))
        for name in sorted(entry.delta.params):
            p = params.get(name)
            rows.append([cell, f"${display_symbol(p, name)}$",
                         _value_text(p), unit_text(slot(p, "unit")), _meaning(p, name)])
    if not rows:
        return ""
    return table_or_prose(["Variant", "Parameter", "Value", "Unit", "Description"], rows,
                    aligns=["l", "l", "r", "l", "l"])


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


_ET_AL_MARKERS = frozenset({"others", "et al.", "al."})


def _format_person(person) -> str:
    """One author as `Last, F.`, using only the name parts the entry actually carries.

    BibTeX truncates an author list by ending it with `and others`, which pybtex parses as
    a person whose sole name is `others` and who has no first name; the same idiom appears
    in the wild as `et al.`. Taking a first initial unconditionally raised `IndexError` on
    every entry written that way.

    A surname particle is a name part in its own right: pybtex files `van der` under
    `prelast_names` and only `Pol` under `last_names`, so reading the latter alone cites
    a different person entirely.
    """
    last = " ".join([*person.prelast_names, *person.last_names]).strip()
    if last.strip("{}").lower() in _ET_AL_MARKERS:
        return "et al."
    initials = " ".join(f"{name[0]}." for name in person.first_names if name)
    return f"{last}, {initials}" if initials else last


def _format_authors(persons) -> str:
    """An APA author list: `A`, `A & B`, `A, B, & C` — with a trailing `et al.` absorbed."""
    names = [_format_person(p) for p in persons]
    if not names:
        return ""
    if names[-1] == "et al.":
        others = names[:-1]
        return f"{', '.join(others)} et al." if others else "et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return f"{', '.join(names[:-1])}, & {names[-1]}"


def _parse_authors(author_fields: Sequence[str]) -> list:
    """BibTeX author strings — `Wilson, Hugh R. and Cowan, Jack D.` — as pybtex `Person`s.

    A BibTeX name is written either `First Last` or `Last, First`, and the ontology's
    citations carry both. Splitting on whitespace and calling the last word the surname
    therefore garbled every comma-ordered entry — `Wilson, Hugh R. and Cowan, Jack D.`
    rendered as `R., W. H., D., C. J.` — besides dropping particles and treating the
    `and others` truncation idiom as a person. Parsing is pybtex's job.
    """
    if db.Person is None:
        raise ImportError(db.PYBTEX_MISSING)
    return [db.Person(name) for field in author_fields for name in field.split(" and ")]


def render_citation(citation: Any, style: str = "apa") -> str:
    """Render an ontology citation instance as formatted text.

    Args:
        citation: An owlready2 instance with author, year, title, journal, volume, pages, label.
        style: 'bibtex' or 'apa'.

    Returns:
        str: The formatted citation.
    """
    persons = _parse_authors(citation.author)

    year = citation.year[0] if citation.year else "Unknown Year"
    title = citation.title[0] if citation.title else "Unknown Title"
    journal = citation.journal[0] if citation.journal else "Unknown Journal"
    volume = citation.volume[0] if citation.volume else "Unknown Volume"
    pages = citation.pages[0] if citation.pages else "Unknown Pages"
    label = citation.label[0] if citation.label else "UnknownLabel"

    if style.lower() == "bibtex":
        authors = " and ".join(_format_person(person) for person in persons)
        return (
            f"@article{{{label},\n    author = {{{authors}}},\n    title = {{{title}}},\n    "
            f"journal = {{{journal}}},\n    year = {{{year}}},\n    volume = {{{volume}}},\n    "
            f"pages = {{{pages}}}\n}}"
        )
    elif style.lower() == "apa":
        return f"{_format_authors(persons)} ({year}). {title}. *{journal}*, {volume}, {pages}."
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
        author_str = _format_authors(entry.persons.get("author", []))

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
