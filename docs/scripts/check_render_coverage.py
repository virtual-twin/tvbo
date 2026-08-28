#!/usr/bin/env python
"""Check that every page the sidebar links to is actually in the project's render set.

A sidebar entry and a `project.render` glob are written in different files and nothing ties them together, so a page can sit in `_toc.yml`, exist on disk, and still be covered by no glob. Quarto reports nothing: it renders the set the globs resolve to and never looks at the sidebar. The link then 404s on the published site, and only on the published site — a local preview renders on demand and hides it.

Both directions are checked: a link with no glob behind it never builds, and a link to a file that no longer exists is already dead.

Usage: ``python scripts/check_render_coverage.py`` from ``docs/``, exiting 1 on either.
"""

from __future__ import annotations

import pathlib
import re
import sys

import yaml

HREF = re.compile(r"href:\s+([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:qmd|md|ipynb))")


def render_set(config: pathlib.Path) -> set[str]:
    """The files `project.render` resolves to, with `!` patterns removed at the end.

    Quarto applies negations after the whole list, not in sequence, so an exclusion holds however early it appears. A pattern ending in ``**`` means every file below that directory, which Python spells ``**/*`` — bare ``**`` matches only the directories themselves.
    """
    patterns = yaml.safe_load(config.read_text(encoding="utf-8"))["project"]["render"]
    included: set[str] = set()
    excluded: set[str] = set()
    for pattern in patterns:
        target = excluded if pattern.startswith("!") else included
        glob = pattern.lstrip("!")
        target.update(str(p) for p in pathlib.Path().glob(f"{glob}/*" if glob.endswith("**") else glob) if p.is_file())
    return included - excluded


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    if pathlib.Path.cwd() != root:
        sys.exit(f"run this from {root}")

    rendered = render_set(root / "_quarto.yml")
    linked = sorted(set(HREF.findall((root / "_toc.yml").read_text(encoding="utf-8"))))

    unrendered = [h for h in linked if h not in rendered and pathlib.Path(h).exists()]
    missing = [h for h in linked if not pathlib.Path(h).exists()]

    for href in unrendered:
        print(f"_toc.yml: {href}: linked but matched by no `project.render` pattern — the page will 404")
    for href in missing:
        print(f"_toc.yml: {href}: linked but not on disk")

    print(f"\n{len(linked)} sidebar link(s), {len(rendered)} file(s) in the render set", end="")
    if unrendered or missing:
        print(f" — {len(unrendered) + len(missing)} broken")
        return 1
    print(" — all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
