#!/usr/bin/env python3
"""Fail on comment and docstring prose no formatter will fix.

`ruff format` and `black` leave comment and docstring *prose* byte-identical, so the house rules about it cannot be delegated to the formatter. This checker is where they live. Three rules, each one a defect the tree has accumulated:

- **A standalone `#` run is at most one line.** Anything longer belongs in the docstring of the thing it describes, where quartodoc renders it. Stacked blocks also accrete: the second explanation gets appended and the first is never deleted. A copyright and licence notice at the head of a file is exempt: it is metadata addressed to a licence scanner, not prose addressed to a reader, and it has nowhere else to live.
- **Docstring prose is not hand-wrapped.** A line broken mid-sentence serves the source file's column ruler and nobody else; it reflows badly in the rendered site and makes every later edit a multi-line diff. `E501` is already off, so a paragraph may be one long line.
- **No commented-out code.** Git holds the history.

Two modes, because the two surfaces are at different stages. Python is clear, so it is checked whole-file and all three rules apply. Templates and configuration still carry a backlog of stacked blocks, so there the comment rule is checked against the diff only:
new code cannot add one, and the existing ones do not have to be cleared first. The rule is defined once either way — the convention covers `.mako`, `.yaml` and `.toml` as much as `.py`, so it must not depend on which file it lands in.

Exit 1 on any violation.

    python scripts/check_prose.py                    # whole-file sweep: tvbo/ tests/ scripts/ benchmarks/
    python scripts/check_prose.py path/to/file.py    # pre-commit passes changed files
    python scripts/check_prose.py path/to/x.py.mako  # a named template is read whole-file too
    python scripts/check_prose.py --diff           # added lines only, vs the index
    python scripts/check_prose.py --diff origin/dev  # added lines only, vs a ref (CI)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

DEFAULT_ROOTS = ("tvbo", "tests", "scripts", "benchmarks", "docs/scripts", "schema", "hatch_build.py")
SKIP_PARTS = ("__pycache__", ".venv", "_archive", "site-packages")
GENERATED = ("tvbo/datamodel/schema.py", "tvbo/datamodel/pydantic.py", "tvbo/datamodel/dialect_tables.py")
GENERATED_DIRS = ("tvbo/datamodel/", "docs/datamodel/", "tests/reference_data/", "tvbo/templates/modules/", ".claude/skills/")
"""Trees a generator writes. Their comments come from whatever they were generated from,
so a person cannot fix them here. ``tvbo/templates/modules/`` is mako's compiled cache and
is gitignored — present in a working tree, absent in CI, so counting it makes the two
disagree about a clean run."""

DIFF_SUFFIXES = (".py", ".mako", ".yml", ".yaml", ".toml", ".jl", ".sh")
"""What the comment rule covers in diff mode — source, templates and configuration alike."""

MAX_COMMENT_RUN = 1
WRAP_LIMIT = 100

_SENTENCE_END = re.compile(r"[.!?:;]$")
_SECTION = re.compile(r"^[A-Z][\w /-]{0,24}:(\s|$)")
_ORDERED = re.compile(r"^\d+[.)]\s")
_DIRECTIVE = re.compile(
    r"^\s*#\s*(type:|noqa|pragma|ruff:|mypy:|fmt:|isort:|pylint:|!|:)"
    r"|^\s*#(SBATCH|PBS|BSUB|FLUX|OAR)\b|^\s*#\$\s"
)
"""Lines that open with `#` but declare something rather than say something. Tool pragmas, and the batch directives a scheduler reads out of a job script — a run of `#SBATCH` lines is the job's resource request, so it has no shorter form and nowhere else to live."""
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

    The one place this judgement lives, so `check_prose` and `scripts/unwrap_prose.py` cannot disagree about what counts as hand-wrapped. The test is on *a*: a line that ends without terminal punctuation has not finished its sentence, so whatever follows continues it. Neither side may open a new block — a bullet, a numbered item, a fence, a doctest, a table row or a `Label:` header.

    Judging by *b*'s first character instead is what let two earlier sweeps pass: restricting continuations to a lowercase word missed a break before a backtick, and widening that to brackets and digits still missed every break before a deliberately capitalised word (`ALL its cells`, `TVB-O`, a proper noun).
    """
    if not a or not b:
        return False
    block = (">>>", "...", "|", "-", "*", "=", "~", "^", "+", "```")
    if a.startswith((*block, "#", "$$")) or b.startswith((*block, '"""', "'''")):
        return False
    if _ORDERED.match(a) or _ORDERED.match(b):
        return False
    if _SENTENCE_END.search(a) or _SECTION.match(b):
        return False
    return True


def is_block_row(raw: str, base: int) -> bool:
    """Whether *raw* opens a row of a laid-out block rather than a line of flowing prose.

    Flowing prose starts at exactly *base*; a line indented past it was put there on purpose — an `Args:` entry, a NumPy parameter description, an endpoint listing, a table — and its line breaks carry meaning. Both sides of a candidate pair are tested: an indented first line means a laid-out row, an indented second line means a hanging description.

    The one place this judgement lives, so the checker and `scripts/unwrap_prose.py` cannot disagree. When they did, the codemod flattened an endpoint listing the checker had skipped.
    """
    return bool(raw.strip()) and len(raw) - len(raw.lstrip()) > base


def is_licence_run(start: int, run: list[tuple[int, str]]) -> bool:
    """Whether a `#` run is the file's copyright and licence header.

    Metadata addressed to a licence scanner, not prose addressed to a reader, and it has nowhere else to live — so it is exempt from the one-line rule, and its line breaks must survive the codemod. Shared for the same reason as `is_block_row`: when only the checker knew, the codemod folded `# Author:` into `# Copyright ©` across two dozen files.
    """
    return start == 1 and any(_LICENCE.search(body) for _, body in run)


def _layer(stripped: str, mako: bool) -> str:
    """Which comment layer a `#` line belongs to.

    A Mako template carries two on the same character: `##` opens a comment the renderer strips, so it addresses whoever reads the template, while a single `#` is text the renderer emits, so it addresses whoever reads the generated artifact. Everywhere else there is only one layer, and reporting one keeps adjacent lines in a single run.
    """
    return "##" if mako and stripped.startswith("##") else "#"


def _comment_runs(lines: list[str], path: str = ""):
    """Yield `(start_line, bodies)` for each run of standalone `#` lines.

    A run breaks where the comment layer changes, so a block of template prose and the one-line comment it emits below itself are counted apart rather than as one long block. In a template that emits markdown a `#` line inside a fence is a shell sample in the emitted document, which is content and not a comment at all.
    """
    mako = path.endswith(".mako")
    markdown = path.endswith(".md.mako")
    run: list[tuple[int, str]] = []
    layer, fenced, documented = None, False, False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if markdown and stripped.startswith("```"):
            fenced = not fenced
        if mako and stripped.startswith("<%doc>"):
            documented = "</%doc>" not in stripped
        elif mako and stripped.startswith("</%doc>"):
            documented = False
            continue
        here = _layer(stripped, mako)
        is_comment = (
            stripped.startswith("#") and not _DIRECTIVE.match(stripped) and not (fenced and here == "#") and not documented
        )
        if is_comment and layer in (None, here):
            layer = here
            run.append((i, stripped.lstrip("#").strip()))
            continue
        if run:
            yield run[0][0], run
        run, layer = [], None
        if is_comment:
            layer = here
            run.append((i, stripped.lstrip("#").strip()))
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
        if is_block_row(raw_a, base) or is_block_row(raw_b, base):
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

    for start, run in _comment_runs(lines, rel):
        if is_licence_run(start, run):
            continue
        for lineno, body in run:
            if _is_code(body):
                bad.append(f"{rel}:{lineno}: commented-out code — delete it, git has the history")
        if len(run) > MAX_COMMENT_RUN:
            bad.append(
                f"{rel}:{start}: {len(run)}-line `#` block — at most {MAX_COMMENT_RUN}; "
                f"move the explanation into the docstring"
            )

    if path.suffix != ".py":
        return bad

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


WHOLE_FILE_SUFFIXES = (".py", ".mako")
"""What a whole-file run can read. The docstring rules need an AST and so apply to Python alone; the comment rule is about prose and reads the same in a template."""

SWEEP_SUFFIXES = (".py",)
"""What a directory sweep reads. A template is checked when its path is named, which is how the pre-commit hook passes the files a commit touches."""


def iter_files(targets: list[str]):
    """Checkable files under *targets*, skipping vendored and generated trees."""
    for t in targets:
        p = Path(t)
        explicit = p.is_file()
        candidates = [p] if explicit else sorted(f for suffix in SWEEP_SUFFIXES for f in p.rglob(f"*{suffix}"))
        for f in candidates:
            if not f.name.endswith(WHOLE_FILE_SUFFIXES) or any(s in f.parts for s in SKIP_PARTS):
                continue
            rel = f.as_posix()
            if rel in GENERATED or any(rel.startswith(d) for d in GENERATED_DIRS):
                continue
            yield f


def _added_comment_runs(diff: str):
    """Yield `(path, run)` for each over-long run of comment lines the diff adds.

    A run starting at line 1 is a file header — the copyright and licence block — and is exempt, as it is whole-file. Only added lines are read, so an existing block is untouched until someone edits it.
    """
    path, lineno, run, layer = None, 0, [], None

    def close():
        nonlocal run, layer
        finished = run if len(run) > MAX_COMMENT_RUN and run[0][0] > 1 else []
        run, layer = [], None
        return finished

    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            if block := close():
                yield path, block
            path = raw[6:]
        elif raw.startswith("@@"):
            if block := close():
                yield path, block
            lineno = int(raw.split("+", 1)[1].split(",")[0].split(" ")[0])
        elif raw.startswith("+") and not raw.startswith("+++"):
            body = raw[1:].strip()
            here = _layer(body, bool(path) and path.endswith(".mako"))
            is_comment = body.startswith("#") and not _DIRECTIVE.match(body)
            if not (is_comment and layer in (None, here)) and (block := close()):
                yield path, block
            if is_comment:
                layer = here
                run.append((lineno, body.lstrip("#").strip()))
            lineno += 1
        elif block := close():
            yield path, block
    if block := close():
        yield path, block


def _exempt_lines(path: str) -> set[int]:
    """Line numbers the comment rule does not read, for a template.

    Inside a `<%doc>` block the prose IS the docstring, which is where the rule wants it. Inside a fence of a template that emits markdown, a `#` line is a shell sample the reader of the emitted document is meant to see. Read from the working tree because a unified diff of added lines alone cannot say whether either is open.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return set()
    markdown = path.endswith(".md.mako")
    inside, fenced, documented = set(), False, False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if markdown and stripped.startswith("```"):
            fenced = not fenced
        elif stripped.startswith("<%doc>"):
            documented = "</%doc>" not in stripped
            inside.add(i)
            continue
        elif stripped.startswith("</%doc>"):
            documented = False
            inside.add(i)
            continue
        if fenced or documented:
            inside.add(i)
    return inside


def check_diff(base: str | None) -> tuple[int, list[str]]:
    """Violations among the lines a diff adds, as `(exit_code, messages)`.

    Args:
        base: Ref to compare against, or None to compare against the index — which is
            what pre-commit stages.
    """
    import subprocess

    if base:
        for probe in (["rev-parse", "--verify", "--quiet", base], ["merge-base", base, "HEAD"]):
            done = subprocess.run(["git", *probe], capture_output=True, text=True)
            if done.returncode != 0 or not done.stdout.strip():
                return 2, [f"cannot compare against {base!r}: no such ref or no shared history — the check did not run."]
        cmd = ["git", "diff", "-U0", done.stdout.strip()]
    else:
        cmd = ["git", "diff", "-U0", "--cached"]

    diff = subprocess.run([*cmd, "--", "."], capture_output=True, text=True, check=True).stdout
    bad = []
    for path, run in _added_comment_runs(diff):
        if not path or not path.endswith(DIFF_SUFFIXES):
            continue
        if any(path.startswith(d) for d in GENERATED_DIRS):
            continue
        if path.endswith(".mako") and run[0][0] in _exempt_lines(path):
            continue
        bad.append(
            f"{path}:{run[0][0]}: {len(run)}-line `#` block added — at most {MAX_COMMENT_RUN}; "
            f"move the explanation into the docstring, or cut it to one line"
        )
    return (1 if bad else 0), bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument(
        "--diff",
        nargs="?",
        const="",
        metavar="REF",
        help="check only the comment rule, only on added lines, over every covered suffix; "
        "against REF, or the index when REF is omitted",
    )
    args = ap.parse_args()

    if args.diff is not None:
        code, messages = check_diff(args.diff or None)
        for m in messages:
            print(m)
        if code == 1:
            print(f"\n{len(messages)} stacked comment block(s) added. See scripts/check_prose.py for the rules.")
        return code

    findings = [m for f in iter_files(args.paths or list(DEFAULT_ROOTS)) for m in check(f)]
    for m in findings:
        print(m)
    if findings:
        print(f"\n{len(findings)} prose violation(s). See scripts/check_prose.py for the rules.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
