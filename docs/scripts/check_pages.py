#!/usr/bin/env python
"""Style checks for documentation PAGES (.qmd / .md), beyond the hard-wrap rule slopfmt enforces.

Distinct from the repo-root ``scripts/check_prose.py``, which checks comment and docstring prose inside SOURCE files. This one reads the rendered-page surface instead.

Reports four defects that make a page read as machine-written:

* **em-dash** — density above the budget. Ordinary technical prose sits near 1 per 1000 words; a page far above that is punctuating by reflex, usually where a colon, a comma or a full stop belongs.
* **alt** — placeholder alternative text on an image. An empty markdown alt is fine when the attribute block supplies ``fig-alt``, which is how a figure carries accessible text without also printing a caption.
* **tabset** — a tabset whose tabs are sequential steps rather than alternatives. Tabs are for choosing between equivalents; steps belong in order on the page.
* **skeleton** — the templated Learn / Do / Understand / Look up / Related index shape, repeated across pages until every landing reads the same.

Usage:

- ``python docs/scripts/check_pages.py <paths>``
- ``python docs/scripts/check_pages.py --budget 6 <paths>`` sets a different em-dash budget
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

SKELETON = {"Learn", "Do", "Understand", "Look up", "Related"}
PLACEHOLDER_ALT = re.compile(r"!\[\s*(alt text|image|figure|placeholder|todo|tbd)?\s*\]", re.IGNORECASE)
STEPWISE = re.compile(r"^(step\s*\d|\d+[.)]\s|install|installation|setup|then\b|next\b|finally\b)", re.IGNORECASE)

# `**Label** — the gloss` and `- [Link](x) — the gloss`: a short lead-in followed by its definition. That dash is the idiom a glossary is written in, not a sentence reaching for a connector, so the density budget drops that one dash and keeps the words, staying pointed at mid-sentence use without shrinking the denominator.
GLOSS = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?=.{0,70}—)[^—]*—")


def prose_of(text: str) -> str:
    """The document reduced to running prose.

    Fenced blocks go first, so code is never mistaken for writing. Table rows and headings go with them, because the dash in an empty table cell or a heading's subtitle is punctuation the page needs rather than a sentence punctuated by reflex, and counting them buries the real findings. What remains is paragraphs, list items and callout text, which is where the density budget means something.
    """
    out, fenced = [], False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or stripped.startswith(("|", "#", "#|")):
            continue
        out.append(GLOSS.sub(lambda m: m.group(0).replace("—", ""), re.sub(r"`[^`]*`", "", line), count=1))
    return "\n".join(out)


def check(path: pathlib.Path, budget: float) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    found: list[str] = []

    body = prose_of(text)
    count = body.count("—")
    n_words = len(body.split())
    if n_words >= 200:
        density = 1000 * count / n_words
        if density > budget:
            found.append(f"{path}: em-dash: {density:.1f} per 1000 words ({count} in {n_words}), budget {budget:g}")

    for i, line in enumerate(lines, 1):
        if PLACEHOLDER_ALT.search(line) and "](" in line and "fig-alt=" not in line:
            found.append(f"{path}:{i}: alt: placeholder alternative text")

    headings = [h.strip() for h in re.findall(r"^##\s+(.+)$", text, re.M)]
    if len(SKELETON & set(headings)) >= 3:
        found.append(f"{path}: skeleton: templated {' / '.join(h for h in headings if h in SKELETON)} index shape")

    for block in re.findall(r":::+\s*\{[^}]*panel-tabset[^}]*\}(.*?)\n:::", text, re.S):
        tabs = [t.strip() for t in re.findall(r"^###\s+(.+)$", block, re.M)]
        stepwise = [t for t in tabs if STEPWISE.match(t)]
        if len(tabs) >= 3 and len(stepwise) >= 2:
            found.append(f"{path}: tabset: tabs look like sequential steps, not alternatives ({', '.join(stepwise)})")

    return found


def relative_to_docs(path: pathlib.Path) -> str:
    """The path as the sidebar names it: relative to ``docs/``, however the caller spelled it.

    CI passes repo-relative paths (``docs/Models/index.qmd``) and a local run passes docs-relative ones (``Models/index.qmd``); the phase map is keyed the second way.
    """
    parts = path.as_posix().lstrip("./").split("/")
    if parts and parts[0] == "docs":
        parts = parts[1:]
    return "/".join(parts)


def spine_pages() -> set[str]:
    """Pages sitting under a NUMBERED phase, read from the generated map.

    The unnumbered chapters carry a documentation backlog the plan deliberately defers, so enforcing the page budget on them would fail every build for work nobody has scheduled. The spine is what a reader walks, and the spine is what must stay clean.
    """
    lua = pathlib.Path(__file__).parent.parent / "_static" / "phase_map.lua"
    if not lua.is_file():
        sys.exit(
            f"--spine needs {lua}, which scripts/build_phase_map.py writes; run a render (or that script) first. Filtering to an empty spine would report a clean run over no pages at all."
        )
    return set(re.findall(r'\["([^"]+)"\]\s*=\s*\{number=\d+,', lua.read_text(encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--budget", type=float, default=8.0, help="em-dashes per 1000 words (default 8)")
    # Findings always print, `--quiet` drops only the trailing summary: this runs as a CI gate, and a gate that fails without naming the page it failed on is a gate nobody can act on. Same meaning the sibling check_cli_examples.py gives the flag, since CI passes it to both.
    ap.add_argument("--quiet", action="store_true", help="report findings only, without the trailing summary")
    ap.add_argument(
        "--spine", action="store_true", help="check only the numbered phases of the spine, read from _static/phase_map.lua"
    )
    args = ap.parse_args()

    paths = [p for p in args.paths if p.is_file()]
    if args.spine:
        spine = spine_pages()
        paths = [p for p in paths if relative_to_docs(p) in spine]
        print(f"[check-pages] spine only: {len(paths)} page(s)")

    findings = [f for path in paths for f in check(path, args.budget)]
    for line in findings:
        print(line)
    if not args.quiet:
        kinds: dict[str, int] = {}
        for line in findings:
            kind = line.split(": ")[1] if ": " in line else "?"
            kinds[kind] = kinds.get(kind, 0) + 1
        print(
            f"\n{len(findings)} finding(s): " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
            if findings
            else "\nclean"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
