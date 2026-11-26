"""A docs page that writes an inline ``{python}`` expression turns the Jupyter cache off.

``docs/_quarto.yml`` caches every page's execution (``execute: cache: true``), and Quarto refuses an inline expression under that cache ("Inline expressions cannot be used with Jupyter Cache"), so such a page renders for the docs test, which executes pages without Quarto, and then stops the release's docs publish. The page sets ``execute: cache: false`` in its front matter, and no workflow renders with ``--cache`` or ``--cache-refresh``, which force the cache on for every page over that setting.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
INLINE = re.compile(r"(?<!`)`\{python\} ")
FORCED_CACHE = re.compile(r"quarto render\b.*--cache(-refresh)?\b")
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


def test_no_workflow_forces_the_cache_over_a_page_that_turns_it_off():
    forcing = [
        f"{path.name}:{number}"
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORCED_CACHE.search(line)
    ]
    assert not forcing, f"{forcing} render with a cache flag, which overrides a page's `execute: cache: false`."
