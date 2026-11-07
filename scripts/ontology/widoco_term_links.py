"""Point the relative term links of a Widoco page at each term's IRI, and fail if any is left.

Widoco links a namespace term that has no section on the page relative to the ontology IRI and gives the link the full IRI as its title. A link whose title is the namespace plus its href is rewritten to that IRI; an `<a>` that still pairs a relative href with a namespace title, in any attribute order, exits 1.

Usage: python scripts/ontology/widoco_term_links.py docs/_site/ontology/spec/index-en.html
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NAMESPACE = "https://w3id.org/tvbo/"
RELATIVE_TERM_LINK = re.compile(rf'<a href="([^"#:]+)" title="{re.escape(NAMESPACE)}\1"')
ANCHOR = re.compile(r"<a\b([^>]*)>")
RELATIVE_HREF = re.compile(r'\bhref="([^"#:][^":]*)"')
NAMESPACE_TITLE = re.compile(rf'\btitle="{re.escape(NAMESPACE)}')


def absolute_term_links(html: str) -> str:
    """Rewrite each link whose title is the namespace plus its href to that full IRI."""
    return RELATIVE_TERM_LINK.sub(lambda m: f'<a href="{NAMESPACE}{m[1]}" title="{NAMESPACE}{m[1]}"', html)


def relative_term_links(html: str) -> list[str]:
    """The relative href of every `<a>` whose title is a namespace IRI."""
    return [
        href[1] for tag in ANCHOR.finditer(html) if NAMESPACE_TITLE.search(tag[1]) and (href := RELATIVE_HREF.search(tag[1]))
    ]


def main(path: str) -> int:
    """Rewrite the page at *path* in place; return 1 if a relative term link is left."""
    page = Path(path)
    html = absolute_term_links(page.read_text(encoding="utf-8"))
    page.write_text(html, encoding="utf-8")
    left = relative_term_links(html)
    if left:
        print(f"{page}: {len(left)} relative term links left, e.g. {left[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
