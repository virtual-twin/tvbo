"""Fail on newly added stacked ``#`` comment blocks.

A run of two or more consecutive whole-line ``#`` comments is the shape the code
convention forbids: an explanation that needs more than one line belongs in the
enclosing docstring, where the API documentation tooling can also reach it. One short
line of rationale above a genuinely non-obvious statement is allowed.

Only lines the diff *adds* are checked, so the existing blocks in the tree do not have to
be cleared before the rule can be enforced on new code. With no arguments the comparison
is against the index (what ``pre-commit`` stages); pass a ref to compare against that
instead, e.g. ``python scripts/check_comment_blocks.py origin/dev`` in CI.
"""

from __future__ import annotations

import subprocess
import sys

SUFFIXES = (".py", ".mako", ".yml", ".yaml", ".toml", ".jl", ".sh")
"""Suffixes the convention covers — source, templates and configuration alike."""

GENERATED = ("tests/reference_data/", "tvbo/datamodel/", "docs/datamodel/", ".claude/skills/")
"""Paths written by a generator. Their comments come from whatever they were generated
from, so a person cannot fix them here and the rule has nothing to say about them."""

MAX_RUN = 1


def is_comment(line: str) -> bool:
    """A whole-line ``#`` comment the rule speaks about.

    Two spellings are not prose and so are not part of a run: a ``#!`` shebang, and a
    ``#:`` attribute doc, which is the only way Python offers to document a module-level
    constant — a constant has no docstring slot to move the explanation into.
    """
    stripped = line.strip()
    return stripped.startswith("#") and not stripped.startswith(("#!", "#:"))


def added_blocks(diff: str):
    """Yield ``(path, lines)`` for each run of added comment lines over the limit.

    A run that starts at line 1 is a file header — the copyright and licence block every
    TVBO source file opens with — and is exempt: it documents the file, not a statement.
    """
    path, lineno, run = None, 0, []

    def close():
        nonlocal run
        finished = run if len(run) > MAX_RUN and run[0][0] > 1 else []
        run = []
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
            body = raw[1:]
            if is_comment(body):
                run.append((lineno, body))
            elif block := close():
                yield path, block
            lineno += 1
        elif block := close():
            yield path, block
    if block := close():
        yield path, block


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else None
    if base is None:
        cmd = ["git", "diff", "-U0", "--cached"]
    else:
        resolved = subprocess.run(["git", "rev-parse", "--verify", "--quiet", base], capture_output=True, text=True)
        if resolved.returncode != 0:
            print(f"cannot compare against {base!r}: no such ref — the check did not run.")
            return 2
        merge_base = subprocess.run(
            ["git", "merge-base", base, "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        cmd = ["git", "diff", "-U0", merge_base]
    diff = subprocess.run([*cmd, "--", "."], capture_output=True, text=True, check=True).stdout

    failures = [
        (path, block)
        for path, block in added_blocks(diff)
        if path and path.endswith(SUFFIXES) and not path.startswith(GENERATED)
    ]
    if not failures:
        return 0

    for path, block in failures:
        print(f"{path}:{block[0][0]}: {len(block)} stacked comment lines")
        for lineno, line in block:
            print(f"   {lineno}: {line}")
    print(
        f"\n{len(failures)} stacked comment block(s). Move the explanation into the "
        "enclosing docstring, or cut it to one line."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
