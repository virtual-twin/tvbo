#!/usr/bin/env python
"""Auto-generate the 'Replication Studies' section of _toc.yml.

Scans Replication/*/ for .qmd files and reads their YAML frontmatter.

Publication gate
----------------
Replication results are embargoed until the corresponding paper is published, so this generator is **default-deny**: a study page is listed only if it explicitly declares ``publish: true`` in its frontmatter.

Every study page must make the decision explicit:

* ``publish: true``   -> listed in the sidebar and the index listing.
* ``publish: false``  -> withheld. Allowed for local drafting, but the page still *renders* into ``_site`` (it matches the ``Replication/**/*.qmd`` render glob in ``_quarto.yml``), so it remains reachable by direct URL. The script warns loudly and prints that URL.
* **missing**         -> hard error. Omission must never be a silent publish.

Run with ``--strict`` (recommended in CI and any deploy job) to also fail on ``publish: false``, guaranteeing that a withheld study never reaches a built site.

Rewrites the block between # BEGIN:replication-autogen … # END:replication-autogen markers in _toc.yml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent
TOC_FILE = DOCS_DIR / "_toc.yml"
REPL_DIR = DOCS_DIR / "Replication"

BEGIN_MARKER = "# BEGIN:replication-autogen"
END_MARKER = "# END:replication-autogen"

L1 = "              "  # 14 sp
L2 = "                  "  # 18 sp

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def frontmatter(qmd_path: Path) -> str | None:
    """Return the raw YAML frontmatter block of a .qmd file, if present."""
    text = qmd_path.read_text(encoding="utf-8")
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    return m.group(1) if m else None


def _field(fm: str, name: str) -> str | None:
    """Read a scalar top-level frontmatter field (quotes stripped)."""
    for line in fm.splitlines():
        m = re.match(rf'^{name}:\s*["\']?(.*?)["\']?\s*$', line)
        if m:
            return m.group(1)
    return None


def extract_title(qmd_path: Path) -> str | None:
    fm = frontmatter(qmd_path)
    return _field(fm, "title") if fm else None


def publish_state(qmd_path: Path) -> bool | None:
    """True / False as declared, or None when the page does not declare it."""
    fm = frontmatter(qmd_path)
    if fm is None:
        return None
    raw = _field(fm, "publish")
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def study_pages() -> list[Path]:
    """Every .qmd a study offers as a page: at its root, or under the layout's docs role.

    A study is a BIDS study dataset, which keeps its report in ``docs/`` (see :mod:`tvbo.utils.study_layout`). A study embedded in this site may instead keep its page at its root, because the site owns the Quarto project and a nested one would split the build.
    Both are listed, so where a study puts its report is the study's choice rather than a rule the sidebar imposes.
    """
    if not REPL_DIR.is_dir():
        return []
    from tvbo.utils.study_layout import relpath

    docs_role = relpath("docs")
    pages: list[Path] = []
    for study_dir in sorted(REPL_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        pages.extend(sorted(study_dir.glob("*.qmd")))
        pages.extend(sorted((study_dir / docs_role).glob("*.qmd")))
    return pages


def enforce_gate(strict: bool) -> list[Path]:
    """Apply the publication gate. Returns the pages cleared for publication."""
    undeclared: list[Path] = []
    withheld: list[Path] = []
    published: list[Path] = []

    for qmd in study_pages():
        state = publish_state(qmd)
        if state is None:
            undeclared.append(qmd)
        elif state:
            published.append(qmd)
        else:
            withheld.append(qmd)

    if withheld:
        print(
            "\n[replication gate] WITHHELD (publish: false) — these still render into _site and are reachable by direct URL:",
            file=sys.stderr,
        )
        for qmd in withheld:
            url = qmd.relative_to(DOCS_DIR).with_suffix(".html")
            print(f"    {qmd.relative_to(DOCS_DIR)}  ->  /{url}", file=sys.stderr)
        print(
            "    Do not deploy this build. Remove the page from docs/ or run with --strict in CI.",
            file=sys.stderr,
        )

    if undeclared:
        print(
            "\n[replication gate] ERROR — replication study pages must declare "
            "`publish: true` or `publish: false` in their frontmatter.\n"
            "Results are embargoed until published, so omission is never treated "
            "as consent to publish.",
            file=sys.stderr,
        )
        for qmd in undeclared:
            print(f"    missing `publish:`  {qmd.relative_to(DOCS_DIR)}", file=sys.stderr)
        raise SystemExit(1)

    if strict and withheld:
        print(
            "\n[replication gate] ERROR — --strict: withheld studies must not be present in a published build.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return published


# Section-level pages that are not studies. They live directly in Replication/ and would otherwise be orphaned: rendered by the glob but absent from the sidebar. `custom-panels.qmd` is deliberately absent: it teaches the figure-spec escape hatch, so the curated region carries it under SPECIFY beside the figure grammar.
SECTION_PAGES = [
    ("The replication pipeline", "Replication/pipeline.qmd"),
]


def build_block(published: list[Path]) -> str:
    lines: list[str] = [BEGIN_MARKER]
    lines.append(f'{L1}- section: "Replication Studies"')
    lines.append(f"{L1}  href: Replication/index.qmd")
    lines.append(f"{L1}  contents:")

    for text, href in SECTION_PAGES:
        if (DOCS_DIR / href).is_file():
            lines.append(f'{L2}- text: "{text}"')
            lines.append(f"{L2}  href: {href}")

    for qmd in published:
        title = extract_title(qmd) or qmd.stem.replace("_", " ").title()
        rel = qmd.relative_to(DOCS_DIR)
        lines.append(f'{L2}- text: "{title}"')
        lines.append(f"{L2}  href: {rel}")

    lines.append(END_MARKER)
    return "\n".join(lines)


def update_toc(strict: bool = False) -> None:
    published = enforce_gate(strict)

    text = TOC_FILE.read_text()
    begin_idx = text.find(BEGIN_MARKER)
    end_idx = text.find(END_MARKER)

    if begin_idx == -1 or end_idx == -1:
        print(f"✗ Markers not found in {TOC_FILE}. Add {BEGIN_MARKER!r} / {END_MARKER!r}.")
        raise SystemExit(1)

    new_block = build_block(published)
    new_text = text[:begin_idx] + new_block + text[end_idx + len(END_MARKER) :]

    print(f"[replication gate] {len(published)} study page(s) cleared for publication.")

    if new_text == text:
        print("Replication Studies TOC section unchanged — skipping write.")
        return

    TOC_FILE.write_text(new_text)
    print(f"✓ Updated Replication Studies section in {TOC_FILE.name}")


if __name__ == "__main__":
    update_toc(strict="--strict" in sys.argv[1:])
