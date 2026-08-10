#!/usr/bin/env python3
"""Join hand-wrapped comment and docstring prose, and delete commented-out code.

The companion codemod to `scripts/check_prose.py`: that one reports, this one fixes the two mechanical rules. Relocating a comment block into a docstring and reconciling two comments that state one rule are judgement calls and stay manual.

Safe to apply without reading every hunk, because each rewrite is checked against an invariant rather than eyeballed:

- **Unwrapping preserves meaning.** Every docstring's text, with runs of whitespace collapsed, must be byte-identical before and after. A single mismatch aborts the whole file.
- **The file still parses**, and its AST structure is unchanged apart from docstring contents.
- **Deleting a comment cannot change behaviour** — it was already a comment.

    python scripts/unwrap_prose.py --check tvbo/          # report, change nothing
    python scripts/unwrap_prose.py --apply tvbo/data/param_io.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_prose import (  # noqa: E402
    _SENTENCE_END,
    WRAP_LIMIT,
    _comment_runs,
    _is_code,
    iter_files,
)


def _norm(text: str) -> str:
    """Text with every whitespace run collapsed, for comparing before against after."""
    return " ".join(text.split())


def _joinable(a: str, b: str) -> bool:
    """Whether stripped line *a* wraps into stripped line *b*."""
    if not a or not b or _SENTENCE_END.search(a) or len(a) >= WRAP_LIMIT:
        return False
    if a.startswith((">>>", "...", "|", "-", "*", "#", "$$", "```")) or b.startswith((">>>", "...", "|", "-", "*", "```")):
        return False
    return b[0].islower() or b[0] in "(["


def _unwrap_block(lines: list[str], base_indent: int) -> list[str]:
    """Join wrapped prose within one indented block, leaving code blocks alone."""
    out: list[str] = []
    fenced = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        indented = len(raw) - len(raw.lstrip()) >= base_indent + 4 if stripped else False
        prev = out[-1].strip() if out else ""
        if (
            out
            and not fenced
            and not indented
            and _joinable(prev, stripped)
            and len(out[-1]) - len(out[-1].lstrip()) < base_indent + 4
        ):
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(raw)
    return out


def _rewrite_docstrings(src: str) -> str:
    """Unwrap prose inside every docstring, leaving the rest of the file untouched."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits: list[tuple[int, int, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body or not ast.get_docstring(node, clean=False):
            continue
        expr = node.body[0]
        lo, hi = expr.lineno - 1, expr.end_lineno
        block = [ln.rstrip("\n") for ln in lines[lo:hi]]
        body = [ln for ln in block[1:] if ln.strip()]
        indent = min((len(ln) - len(ln.lstrip()) for ln in body), default=0)
        new = _unwrap_block(block, indent)
        if new != block:
            edits.append((lo, hi, [ln + "\n" for ln in new]))
    for lo, hi, new in sorted(edits, reverse=True):
        lines[lo:hi] = new
    return "".join(lines)


def _rewrite_comments(src: str) -> str:
    """Join wrapped `#` prose and drop commented-out code, run by run.

    Deletion is all-or-nothing per run. Deleting only the lines that individually parse as code guts a commented-out example and leaves a mangled remainder, so a run is dropped only when it is overwhelmingly code, and kept whole otherwise.

    Prose joins reuse the original text verbatim. Re-emitting a comment as `f"{indent}# {body}"` would flatten the inner indentation of a snippet somebody laid out deliberately.
    """
    lines = src.splitlines(keepends=True)
    drop: set[int] = set()
    edits: dict[int, str] = {}
    for _, run in _comment_runs([ln.rstrip("\n") for ln in lines]):
        codey = [ln for ln, body in run if _is_code(body)]
        if codey and len(codey) >= max(2, int(0.6 * len(run))):
            drop.update(ln - 1 for ln, _ in run)
            continue
        if codey:
            continue
        keep: list[tuple[int, str]] = [(ln, body) for ln, body in run]
        for idx in range(len(keep) - 1, 0, -1):
            prev_no, prev_body = keep[idx - 1]
            cur_no, cur_body = keep[idx]
            if _joinable(prev_body, cur_body):
                keep[idx - 1] = (prev_no, prev_body + " " + cur_body)
                keep.pop(idx)
                drop.add(cur_no - 1)
        for lineno, body in keep:
            raw = lines[lineno - 1]
            if body == raw.strip().lstrip("#").strip():
                continue
            indent = raw[: len(raw) - len(raw.lstrip())]
            edits[lineno - 1] = f"{indent}# {body}\n"
    out = []
    for i, raw in enumerate(lines):
        if i in drop:
            continue
        out.append(edits.get(i, raw))
    return "".join(out)


def _docstring_map(src: str) -> dict[str, str]:
    """Every docstring in *src*, keyed by node path, whitespace-normalized."""
    tree = ast.parse(src)
    found: dict[str, str] = {}
    counter = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        found[f"{getattr(node, 'name', '<module>')}#{counter}"] = _norm(doc)
        counter += 1
    return found


def process(path: Path, apply: bool) -> tuple[int, str | None]:
    """Rewrite *path*; return `(lines_changed, error)`. Nothing is written on error."""
    original = path.read_text(encoding="utf-8")
    try:
        before = _docstring_map(original)
    except SyntaxError as exc:
        return 0, f"does not parse: {exc.msg}"

    try:
        new = _rewrite_comments(_rewrite_docstrings(original))
    except SyntaxError as exc:
        return 0, f"rewrite produced invalid syntax: {exc.msg}"
    if new == original:
        return 0, None

    try:
        after = _docstring_map(new)
    except SyntaxError as exc:
        return 0, f"rewrite broke the parse: {exc.msg}"
    if after != before:
        changed = [k for k in before if before.get(k) != after.get(k)]
        return 0, f"docstring text changed for {changed[:3]} — refusing to write"

    delta = len(original.splitlines()) - len(new.splitlines())
    if apply:
        path.write_text(new, encoding="utf-8")
    return delta, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=["tvbo"])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report without writing (default)")
    mode.add_argument("--apply", action="store_true", help="write the rewrites")
    args = ap.parse_args()

    touched = removed = failed = 0
    for f in iter_files(args.paths or ["tvbo"]):
        delta, err = process(f, apply=args.apply)
        if err:
            print(f"SKIP {f}: {err}")
            failed += 1
        elif delta:
            touched += 1
            removed += delta
            print(f"{'rewrote' if args.apply else 'would rewrite'} {f}  (-{delta} lines)")
    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {touched} file(s), -{removed} lines; {failed} skipped by the invariant")
    return 1 if failed and args.apply else 0


if __name__ == "__main__":
    sys.exit(main())
