#!/usr/bin/env python3
"""Fail on comment and docstring prose no formatter will fix.

`ruff format` and `black` leave comment and docstring *prose* byte-identical, so the house rules about it cannot be delegated to the formatter. This checker is where they live. Three rules, each one a defect the tree has accumulated:

- **A standalone `#` run is at most one line.** Anything longer belongs in the docstring of the thing it describes, where quartodoc renders it. Stacked blocks also accrete: the second explanation gets appended and the first is never deleted. A copyright and licence notice at the head of a file is exempt: it is metadata addressed to a licence scanner, not prose addressed to a reader, and it has nowhere else to live.
- **Docstring prose is not hand-wrapped.** A line broken mid-sentence serves the source file's column ruler and nobody else; it reflows badly in the rendered site and makes every later edit a multi-line diff. `E501` is already off, so a paragraph may be one long line.
- **No commented-out code.** Git holds the history.

Run over the paths given, or the default tree. Exit 1 on any violation.

    python scripts/check_prose.py                 # tvbo/ tests/ scripts/ benchmarks/
    python scripts/check_prose.py path/to/file.py # pre-commit passes changed files
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

DEFAULT_ROOTS = ("tvbo", "tests", "scripts", "benchmarks")
SKIP_PARTS = ("__pycache__", ".venv", "_archive", "site-packages")
GENERATED = ("tvbo/datamodel/schema.py", "tvbo/datamodel/pydantic.py", "tvbo/datamodel/dialect_tables.py")

MAX_COMMENT_RUN = 1
WRAP_LIMIT = 100

_SENTENCE_END = re.compile(r"[.!?:;]$")
_DIRECTIVE = re.compile(r"^\s*#\s*(type:|noqa|pragma|ruff:|mypy:|fmt:|isort:|pylint:|!)")
_LICENCE = re.compile(r"copyright|licen[cs]e|SPDX|\(c\)\s*\d{4}|©", re.IGNORECASE)
_CODEISH = re.compile(
    r"^\s*(def |class |return\b|import |from \S+ import|if .+:|for .+ in .+:|while .+:|"
    r"try:|except\b|elif .+:|else:|with .+:|print\(|assert |raise |@\w+|"
    r"\w+\s*=\s*[^=]|\w+\.\w+\()"
)


def _is_code(text: str) -> bool:
    """Whether a comment body reads as commented-out code rather than prose."""
    body = text.strip()
    if len(body) < 4 or not _CODEISH.search(body):
        return False
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        return bool(re.match(r"^\s*(def |class |if |for |while |try:|except|else:|elif |with )", body))


def continues_sentence(a: str, b: str) -> bool:
    """Whether stripped line *b* is the continuation of a sentence begun on stripped line *a*.

    The one place this judgement lives, so `check_prose` and `scripts/unwrap_prose.py` cannot disagree about what counts as hand-wrapped. A continuation is anything that cannot plausibly start a sentence: a lowercase word, an opening bracket, an inline-code backtick, a dash, a quote, or a digit. Restricting it to lowercase missed a paragraph that happened to break before a backtick — two of them survived the first Stage C sweep that way.
    """
    if not a or not b:
        return False
    if a.startswith((">>>", "...", "|", "-", "*", "#", "$$", "```")) or b.startswith((">>>", "...", "|", "-", "*", "```")):
        return False
    if _SENTENCE_END.search(a):
        return False
    return b[0].islower() or b[0] in "([`\u2014\u2013\"'" or b[0].isdigit()


def _comment_runs(lines: list[str]):
    """Yield `(start_line, bodies)` for each run of standalone `#` lines."""
    run: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") and not _DIRECTIVE.match(stripped):
            run.append((i, stripped.lstrip("#").strip()))
        else:
            if run:
                yield run[0][0], run
            run = []
    if run:
        yield run[0][0], run


def _wrapped(text: str) -> list[int]:
    """Offsets of docstring lines broken mid-sentence with a continuation following.

    Fenced and indented blocks are skipped: a doctest, a usage example or a rendered table is code, and its line breaks are load-bearing.
    """
    lines = text.splitlines()
    body = [ln for ln in lines[1:] if ln.strip()]
    base = min((len(ln) - len(ln.lstrip()) for ln in body), default=0)
    out = []
    fenced = False
    for i in range(len(lines) - 1):
        raw_a, raw_b = lines[i], lines[i + 1]
        a, b = raw_a.strip(), raw_b.strip()
        if a.startswith("```"):
            fenced = not fenced
        if fenced or not a or not b:
            continue
        if len(raw_a) - len(raw_a.lstrip()) >= base + 4 or len(raw_b) - len(raw_b.lstrip()) >= base + 4:
            continue
        if continues_sentence(a, b):
            out.append(i)
    return out


def check(path: Path) -> list[str]:
    """Every violation in *path*, formatted as `file:line: message`."""
    rel = path.as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    bad: list[str] = []

    for start, run in _comment_runs(lines):
        if start == 1 and any(_LICENCE.search(body) for _, body in run):
            continue
        for lineno, body in run:
            if _is_code(body):
                bad.append(f"{rel}:{lineno}: commented-out code — delete it, git has the history")
        if len(run) > MAX_COMMENT_RUN:
            bad.append(
                f"{rel}:{start}: {len(run)}-line `#` block — at most {MAX_COMMENT_RUN}; "
                f"move the explanation into the docstring"
            )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return bad + [f"{rel}:{exc.lineno}: does not parse: {exc.msg}"]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc or not node.body:
            continue
        base = getattr(node.body[0], "lineno", 1)
        for off in _wrapped(doc):
            bad.append(f"{rel}:{base + off}: docstring line hand-wrapped mid-sentence — let it run; the renderer wraps it")
    return bad


def iter_files(targets: list[str]):
    """Python files under *targets*, skipping vendored and generated trees."""
    for t in targets:
        p = Path(t)
        candidates = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for f in candidates:
            if f.suffix != ".py" or any(s in f.parts for s in SKIP_PARTS):
                continue
            if f.as_posix() in GENERATED:
                continue
            yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=list(DEFAULT_ROOTS))
    args = ap.parse_args()

    findings = [m for f in iter_files(args.paths or list(DEFAULT_ROOTS)) for m in check(f)]
    for m in findings:
        print(m)
    if findings:
        print(f"\n{len(findings)} prose violation(s). See scripts/check_prose.py for the rules.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
