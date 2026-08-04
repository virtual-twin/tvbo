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

    When only a single column survives the drop, the result is **not** a table
    but a plain-text comma-separated list of that column's values — a lone
    ``| Term |`` / ``| c_grid |`` table reads as clutter, so it collapses to
    ``c_grid``. The caller's section header supplies the context.

    Args:
        headers: Column titles.
        rows: One sequence of cell values per row.
        aligns: Per-column alignment, ``'l'``/``'r'``/``'c'``; defaults to left.
        empty: Placeholder rendered for an empty cell in a kept column.
        col_cap: Width above which a column stops earning more of the page.
        col_floor: Width below which a column stops giving it up, so a short
            column keeps enough room to typeset its own cells.

    Returns:
        The markdown table as a string (header, rule, and body rows), or a
        plain-text list when a single column remains.
    """
    n = len(headers)
    norm = [[("" if c is None else str(c)).strip() for c in row] for row in rows]

    def _blank(cell: str) -> bool:
        return cell in _EMPTY_MARKERS

    keep = [j for j in range(n) if any(not _blank(r[j]) for r in norm)] if norm else list(range(n))

    # A single surviving column is a list, not a table: emit plain text.
    if norm and len(keep) == 1:
        j = keep[0]
        return ", ".join(r[j] for r in norm if not _blank(r[j]))

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
    from sympy import Derivative, Eq, Symbol, latex

    symbol_names = symbol_names or {}
    if derivative_notation == "dot" and isinstance(eq, Eq) and isinstance(eq.lhs, Derivative):
        deriv = eq.lhs
        order = sum(1 for v in deriv.variables if v == Symbol("t"))
        base = latex(deriv.expr, mul_symbol=mul_symbol, symbol_names=symbol_names)
        dots = {1: "dot", 2: "ddot", 3: "dddot"}.get(order)
        lhs = (f"\\{dots}{{{base}}}" if dots
               else f"\\frac{{d^{order}}}{{d t^{order}}} {base}")
        return f"{lhs} = {latex(eq.rhs, mul_symbol=mul_symbol, symbol_names=symbol_names)}"
    return latex(eq, mul_symbol=mul_symbol, symbol_names=symbol_names)


def model_equations_latex(model, kind="state", derivative_notation="dot", mul_symbol=None):
    """A model's equations of one kind, each as LaTeX, straight from its symbolic form.

    ``kind`` selects ``state`` (state variables) or ``derived`` (derived variables). The
    equations come from :meth:`Dynamics.get_equations`, so the report shows the same
    expressions the backend integrates.
    """
    collection = "state_variables" if kind == "state" else "derived_variables"
    members = getattr(model, collection, None) or {}
    symbol_names = model.symbol_map() if hasattr(model, "symbol_map") else {}
    return [equation_latex(eq, derivative_notation, symbol_names, mul_symbol)
            for name, eq in model.get_equations().items() if name in members]


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
    return md_table(["Event", "Type", "Condition", "Effect", "Parameters", "Description"], rows)


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


def param_table(collection, name_header="Parameter", symbolic=True, flags=None):
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
    """
    def _name(name, p):
        return f"${display_symbol(p, name)}$" if symbolic else str(name)

    rows = [
        [_name(name, p), format_number(slot(p, "value", "")), format_number(slot(p, "default", "")), unit_text(slot(p, "unit")),
         metadata_text(p), flag_text(p, flags),
         slot(p, "description", "") or slot(p, "definition", "") or ""]
        for name, p in name_items(collection)
    ]
    return md_table([name_header, "Value", "Default", "Unit", "Domain / Sampling", "Flags", "Description"],
                    rows, aligns=["l", "r", "l", "l", "l", "l", "l"])


def parameter_table(params):
    """Markdown model Parameters table (empty columns dropped) from a name->obj map."""
    return param_table(params, name_header="Parameter")


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

    b_eqs, m_eqs = baseline.get_equations(), model.get_equations()

    def _rhs(eqs, key):
        eq = eqs.get(key)
        return str(getattr(eq, "rhs", eq))

    def _psig(p):
        return (str(slot(p, "value", "")), str(slot(slot(p, "equation"), "rhs", "")))

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


_ET_AL_MARKERS = frozenset({"others", "et al.", "al."})


def _format_person(person) -> str:
    """One author as `Last, F.`, using only the name parts the entry actually carries.

    BibTeX truncates an author list by ending it with `and others`, which pybtex parses as
    a person whose sole name is `others` and who has no first name; the same idiom appears
    in the wild as `et al.`. Taking a first initial unconditionally raised `IndexError` on
    every entry written that way.
    """
    last = " ".join(person.last_names).strip()
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
