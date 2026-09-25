#!/usr/bin/env python
"""Downscale the page figures that stand in as overview-card thumbnails.

`_static/thumbs.yml` names, for each page that has a representative figure, the label of the figure to use. This reads that figure out of the page's frozen output and writes a web-sized copy to `_static/thumbs/<slug>.png`, which a `::: {.cards}` table points at from its Cover column.

A named figure that is not in `_freeze` yet is a page that has not been executed in this working tree, which is a missing thumbnail rather than a failed build: the card falls back to its phase icon and the build says which page to render.
"""

from __future__ import annotations

import pathlib

import yaml
from PIL import Image

DOCS = pathlib.Path(__file__).parent.parent
FREEZE = DOCS / "_freeze"
OUT = DOCS / "_static" / "thumbs"
WIDTH = 800


def main() -> None:
    declared = yaml.safe_load((DOCS / "_static" / "thumbs.yml").read_text()) or {}
    OUT.mkdir(parents=True, exist_ok=True)
    written, missing = 0, []
    for page, label in declared.items():
        source = FREEZE / page / "figure-html" / f"{label}.png"
        if not source.exists():
            missing.append(f"{page} ({label})")
            continue
        image = Image.open(source).convert("RGB")
        height = round(image.height * WIDTH / image.width)
        image.resize((WIDTH, height), Image.LANCZOS).save(OUT / f"{slug(page)}.png", optimize=True)
        written += 1
    print(f"[thumbnails] {written} written -> {OUT.relative_to(DOCS)}")
    if missing:
        print(f"[thumbnails] {len(missing)} page(s) not yet rendered in this tree, so their cards fall back to a phase icon:")
        for page in missing:
            print(f"    {page}")


def slug(page: str) -> str:
    """A page path as a flat file name, which is what keeps every thumbnail in one directory."""
    return page.replace("/", "-")


if __name__ == "__main__":
    main()
