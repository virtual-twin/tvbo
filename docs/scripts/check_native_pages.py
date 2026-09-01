#!/usr/bin/env python
"""Check that a page declaring `native: true` draws nothing by hand.

A native page states what it wants in a recipe and a figure spec, runs it, and shows what came back. The moment it reaches for matplotlib it has two sources of truth: the spec says one thing and the plotting code draws another, and only the code is on screen. That is the failure this repository keeps hitting — a declaration accepted, emitted, and silently not honoured.

The flag ratchets. A page without it is not checked, so the migration proceeds one page at a time; a page that has it can never quietly regress, because the check runs in CI. Removing the flag to make this pass is the one move that defeats the purpose.

Usage: ``python scripts/check_native_pages.py`` from ``docs/``, exiting 1 on any violation.
"""

from __future__ import annotations

import pathlib
import re
import sys

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
NATIVE = re.compile(r"^native:\s*true\s*$", re.MULTILINE)
FENCE = re.compile(r"^\s*```+\s*\{?(python|\{python\})", re.IGNORECASE)
BANNED = {
    "matplotlib": re.compile(r"\b(?:import\s+matplotlib|from\s+matplotlib\b)"),
    "pyplot call": re.compile(r"\bplt\s*\."),
    "seaborn": re.compile(r"\b(?:import\s+seaborn|from\s+seaborn\b)"),
    "axes call": re.compile(r"\bax(?:es)?\s*\.\s*(?:plot|scatter|hist|imshow|bar|errorbar|contour|axvline|axhline|fill_between|pcolormesh)\b"),
}


def is_native(page: pathlib.Path) -> bool:
    """Whether the page opted in, read off its YAML frontmatter."""
    head = FRONTMATTER.match(page.read_text(encoding="utf-8"))
    return bool(head and NATIVE.search(head.group(1)))


def offences(page: pathlib.Path) -> list[str]:
    """Hand-drawing in this page's python cells, as ``line: what``."""
    found, in_cell = [], False
    for n, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        if not in_cell:
            in_cell = bool(FENCE.match(line))
            continue
        if line.strip().startswith("```"):
            in_cell = False
            continue
        for what, pattern in BANNED.items():
            if pattern.search(line):
                found.append(f"{n}: {what} — {line.strip()[:70]}")
    return found


def main() -> int:
    docs = pathlib.Path(__file__).resolve().parent.parent
    skip = {"_site", "_build", "_freeze", "_output", "_archive", ".quarto"}
    pages = [p for p in sorted(docs.rglob("*.qmd")) if not skip & set(p.relative_to(docs).parts)]
    native = [p for p in pages if is_native(p)]
    report = [f"  {p.relative_to(docs)}:{o}" for p in native for o in offences(p)]
    if report:
        print("pages declaring `native: true` that draw by hand:", file=sys.stderr)
        print("\n".join(report), file=sys.stderr)
        print("\nDeclare the figure in the study's figure spec, or drop the `native:` flag.", file=sys.stderr)
        return 1
    print(f"{len(native)} of {len(pages)} pages declare `native: true`; none draw by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
