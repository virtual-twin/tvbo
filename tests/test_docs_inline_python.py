"""A docs page that writes an inline ``{python}`` expression turns the Jupyter cache off.

``docs/_quarto.yml`` caches every page's execution (``execute: cache: true``), and Quarto refuses an inline expression under that cache ("Inline expressions cannot be used with Jupyter Cache"), so such a page renders for the docs test, which executes pages without Quarto, and then stops the release's docs publish. The page sets ``execute: cache: false`` in its front matter.
"""

import re
from pathlib import Path

import yaml

DOCS = Path(__file__).parent.parent / "docs"
INLINE = re.compile(r"(?<!`)`\{python\} ")
GENERATED = {"_site", "_freeze", "_build", ".quarto"}


def _front_matter(text: str) -> dict:
    """The page's YAML front matter, empty when it has none."""
    if not text.startswith("---\n"):
        return {}
    return yaml.safe_load(text[4 : text.index("\n---", 4)]) or {}


def test_a_page_with_inline_python_turns_the_jupyter_cache_off():
    cached = (yaml.safe_load((DOCS / "_quarto.yml").read_text()).get("execute") or {}).get("cache", False)
    pages = (path for path in DOCS.rglob("*.qmd") if GENERATED.isdisjoint(path.relative_to(DOCS).parts))
    offenders = [
        str(path.relative_to(DOCS))
        for path in pages
        if INLINE.search(text := path.read_text(encoding="utf-8"))
        and (_front_matter(text).get("execute") or {}).get("cache", cached)
    ]
    assert not offenders, (
        f"{offenders} write inline {{python}} expressions under the project's Jupyter cache; set `execute: cache: false` in their front matter."
    )
