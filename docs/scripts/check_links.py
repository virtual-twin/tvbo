"""Resolve every absolute link in the hand-authored pages against the live web.

Run by hand or as ``make docs-links``, never in CI: it needs the network, and a third-party site being briefly down is not a reason to fail a build. Fenced code is skipped, so an f-string that builds a URL is not mistaken for a link.

A ``403`` is reported like any other failure, but it usually means the publisher blocks automated requests rather than that the link is dead. Check such a DOI against ``api.crossref.org/works/<doi>`` before changing it.

Usage:

- ``python docs/scripts/check_links.py`` checks every hand-authored page
- ``python docs/scripts/check_links.py <paths>`` checks only those
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

DOCS = pathlib.Path(__file__).resolve().parent.parent
SKIP = {"_archive", ".quarto", "_site", "_freeze", "api", "datamodel", ".jupyter_cache", "_output", "node_modules"}
LINK = re.compile(r"\((https?://[^)\s<>\"]+)\)")
HREF = re.compile(r'href="(https?://[^"]+)"')


def prose(text: str) -> str:
    out, fenced = [], False
    for line in text.split("\n"):
        if line.strip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def hand_authored() -> list[pathlib.Path]:
    pages = [p for p in DOCS.rglob("*.qmd") if not SKIP & set(p.relative_to(DOCS).parts)]
    pages += [p for p in DOCS.rglob("*.md") if not SKIP & set(p.relative_to(DOCS).parts)]
    return sorted(pages)


def status(url: str) -> tuple[str, str]:
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "20", "-L", "-A", "Mozilla/5.0", url],
        capture_output=True, text=True,
    )
    return url, result.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=pathlib.Path)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    pages = [p for p in args.paths if p.is_file()] or hand_authored()
    where: dict[str, set[str]] = {}
    for page in pages:
        text = prose(page.read_text(encoding="utf-8", errors="ignore"))
        for url in LINK.findall(text) + HREF.findall(text):
            where.setdefault(url.rstrip(".,;"), set()).add(str(page))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(status, sorted(where)))

    broken = [(url, code) for url, code in results if not code.startswith(("2", "3"))]
    for url, code in sorted(broken):
        print(f"{code}  {url}")
        for page in sorted(where[url]):
            print(f"        {page}")
    print(f"\n{len(where)} unique link(s) across {len(pages)} page(s), {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
