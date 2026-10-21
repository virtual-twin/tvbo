#!/usr/bin/env python
"""Check that every citation key a page uses resolves in `references.bib`.

Quarto does not fail a render on an unresolved key: citeproc emits **[@Key2020]** into the page and moves on, so the break is visible only to a reader on the published site. That is how `@Mahjoory2020` survived in `Fitting/JR_tvboptim.qmd` against a bibliography of over 1500 entries.

The reverse direction is deliberately NOT checked. A bibliography is a library, not a manifest of what the current docs happen to cite, so an unused entry is not a defect.

Usage: ``python scripts/check_citations.py`` from ``docs/``, exiting 1 on any unresolved key.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ENTRY = re.compile(r"^@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
CITATION = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_:.#$%&+?<>~/-]*[A-Za-z0-9_])")
FENCE = re.compile(r"^\s*(```+|~~~+)", re.MULTILINE)
CODE_SPAN = re.compile(r"`+[^`\n]*`+")
CROSSREF = re.compile(r"^(fig|tbl|sec|eq|lst|thm|lem|cor|prp|def|exm|exr|nte|tip|wrn|imp|cau)-")


def defined_keys(bib: pathlib.Path) -> set[str]:
    """Citation keys the bibliography defines, `@Comment` blocks excluded."""
    return {k for k in ENTRY.findall(bib.read_text(encoding="utf-8")) if k.lower() != "comment"}


def prose_only(text: str) -> str:
    """The page with code blanked out, so a Julia macro, a Python decorator or a JSON-LD keyword is not read as a citation.

    Both fenced blocks and inline spans go: `@component` and `@dataclass` appear in running prose as code, and citeproc leaves code alone.
    """
    out, fenced, marker = [], False, ""
    for line in text.splitlines():
        opened = FENCE.match(line)
        if opened and not fenced:
            fenced, marker = True, opened.group(1)[0]
        elif fenced and line.strip().startswith(marker * 3):
            fenced = False
            continue
        out.append("" if fenced else CODE_SPAN.sub("", line))
    return "\n".join(out)


def cited_keys(page: pathlib.Path) -> set[str]:
    """Keys the page cites, from `[@key]`, `[@a; @b]` and bare `@key` alike, minus Quarto's own cross-references."""
    found = CITATION.findall(prose_only(page.read_text(encoding="utf-8")))
    return {k for k in found if not CROSSREF.match(k)}


def authored(pages: list[pathlib.Path], root: pathlib.Path) -> list[pathlib.Path]:
    """The pages the repository actually carries. A page git ignores is one the pre-render writes, so its citations belong to quartodoc, not to an author. Asking git keeps the one list of generated paths in `.gitignore`."""
    if not pages:
        return []
    done = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(p) for p in pages),
        capture_output=True,
        text=True,
        cwd=root,
    )
    ignored = {line.strip() for line in done.stdout.splitlines() if line.strip()}
    return [p for p in pages if str(p) not in ignored]


def main() -> int:
    docs = pathlib.Path(__file__).resolve().parent.parent
    bib = docs / "references.bib"
    if not bib.exists():
        print(f"no bibliography at {bib}", file=sys.stderr)
        return 1
    known = defined_keys(bib)
    skip = {"_site", "_build", "_freeze", "_output", "_archive", ".quarto"}
    candidates = [p for p in sorted(docs.rglob("*.qmd")) if not skip & set(p.relative_to(docs).parts)]
    offenders = []
    for page in authored(candidates, docs.parent):
        for key in sorted(cited_keys(page) - known):
            offenders.append(f"  {page.relative_to(docs)}: @{key}")
    if offenders:
        print(f"citation keys with no entry in references.bib ({len(known)} entries):", file=sys.stderr)
        print("\n".join(offenders), file=sys.stderr)
        return 1
    print(f"all citations resolve ({len(known)} entries in references.bib)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
