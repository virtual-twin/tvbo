#!/usr/bin/env python
"""Join hard-wrapped prose paragraphs back into one line per paragraph.

Reuses the detector in ``~/.claude/tools/slopfmt.py`` rather than reimplementing it, so what this fixes and what CI reports cannot drift apart. Every join is a pair slopfmt flagged as ``hard-wrap``; nothing else is touched, and no word is added, removed or reordered.

Joins run bottom-up so a paragraph wrapped over several lines collapses in one pass, and the loop repeats until the file is clean.

Usage:

- ``python scripts/unwrap_prose.py docs/**/*.qmd`` rewrites in place
- ``python scripts/unwrap_prose.py --check <paths>`` reports only, exiting 1 if any remain
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

SLOPFMT = pathlib.Path.home() / ".claude" / "tools" / "slopfmt.py"


def load_slopfmt():
    spec = importlib.util.spec_from_file_location("slopfmt", SLOPFMT)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load the detector at {SLOPFMT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(text: str) -> str:
    """The document with every run of whitespace collapsed to one space.

    Joining a wrapped pair replaces a newline with a space, and both normalize identically — so this string is invariant under a correct unwrap. Any change to it means a word was altered, dropped or reordered, which is a bug rather than a formatting fix.
    """
    return " ".join(text.split())


BLOCK_START = ("|", "<", ":::", "```", "~~~", "#", ">", "-", "*", "+", "$$", "!", "::", "|")
SENTENCE_END = (".", ":", "!", "?", "**", ")", "*", "`", '"', "'", "—", ";", ",", "|", "\\")


def continues_sentence(prev: str, cur: str) -> bool:
    """True when `cur` continues `prev` even though it starts with a capital.

    The detector in slopfmt only flags a continuation that starts lowercase, because a capital is ambiguous: it may open a deliberate one-sentence-per-line break, which the rule allows. The ambiguity resolves on the *previous* line — a line that does not end a sentence has not finished one, so whatever follows continues it regardless of case. The same reasoning admits a continuation opening with an inline code span or inline math, which slopfmt's prose test skips for the same ambiguity.
    """
    p, c = prev.rstrip(), cur.strip()
    if len(p.strip()) < 45 or not c:
        return False
    if prev.endswith("  ") or p.endswith(SENTENCE_END):
        return False
    if c.startswith(BLOCK_START) or re.match(r"\d+[.)]\s", c):
        return False
    if p.lstrip().startswith(("#", "|", ">", ":::", "$$")) or re.match(r"\d+[.)]\s", p.lstrip()):
        return False
    if "http" in p or c.startswith("http"):
        return False
    return bool(re.match(r"[\[A-Za-z0-9(\"'“‘`$]", c))


def capital_continuations(slopfmt, path: pathlib.Path) -> list[int]:
    """Indices of lines that continue the previous line but open with a capital."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    prose = slopfmt.prose_spans(lines, path.suffix.lower(), source)
    return [i for i in range(1, len(lines)) if i in prose and i - 1 in prose and continues_sentence(lines[i - 1], lines[i])]


def fences_balanced(path: pathlib.Path) -> bool:
    """Whether the file opens and closes every code fence.

    An odd number of markers desynchronizes fence tracking from that point on, so the rest of the file reads as prose and code gets joined like a paragraph. That is a defect in the page rather than something to work around, so the unwrapper refuses the file and says which one.
    """
    markers = sum(
        1 for line in path.read_text(encoding="utf-8", errors="ignore").split("\n") if line.strip().startswith(("```", "~~~"))
    )
    return markers % 2 == 0


def unwrap_once(slopfmt, path: pathlib.Path) -> int:
    """Collapse every flagged pair in one bottom-up pass. Returns the number of joins."""
    wrapped = sorted(
        {f.line - 1 for f in slopfmt.check(path) if f.kind == "hard-wrap"} | set(capital_continuations(slopfmt, path)),
        reverse=True,
    )
    if not wrapped:
        return 0
    before = path.read_text(encoding="utf-8")
    lines = before.splitlines()
    for index in wrapped:
        if index == 0 or index >= len(lines):
            continue
        lines[index - 1] = lines[index - 1].rstrip() + " " + lines[index].strip()
        del lines[index]
    after = "\n".join(lines) + "\n"
    if normalized(after) != normalized(before):
        sys.exit(f"{path}: unwrap changed the text, not just its line breaks — refusing to write")
    path.write_text(after, encoding="utf-8")
    return len(wrapped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--check", action="store_true", help="report without rewriting")
    args = ap.parse_args()

    slopfmt = load_slopfmt()
    total, dirty, skipped = 0, 0, 0
    for path in args.paths:
        if not path.is_file():
            continue
        if not fences_balanced(path):
            print(f"{path}: SKIPPED — unbalanced code fences; fix the stray marker first")
            skipped += 1
            continue
        found = len(
            {f.line - 1 for f in slopfmt.check(path) if f.kind == "hard-wrap"} | set(capital_continuations(slopfmt, path))
        )
        if not found:
            continue
        dirty += 1
        if args.check:
            print(f"{path}: {found} hard-wrapped line(s)")
            total += found
            continue
        joins = 0
        for _ in range(20):
            step = unwrap_once(slopfmt, path)
            joins += step
            if step == 0:
                break
        remaining = len(
            {f.line - 1 for f in slopfmt.check(path) if f.kind == "hard-wrap"} | set(capital_continuations(slopfmt, path))
        )
        print(f"{path}: {joins} join(s)" + (f", {remaining} still flagged" if remaining else ""))
        total += joins

    tail = f", {skipped} skipped for unbalanced fences" if skipped else ""
    print(f"\n{dirty} file(s), {total} " + ("hard-wrapped line(s)" if args.check else "join(s)") + tail)
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
