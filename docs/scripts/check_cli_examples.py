"""Check every ``tvbo …`` command in the documentation against the installed CLI.

A command tree drifts faster than the prose describing it, and a copy-paste example that names a verb, a sub-command or an option the CLI dropped is worse than no example. This walks every shell block in the docs, extracts the ``tvbo`` invocations, and resolves each one against the Typer app itself.

What it checks, and nothing more:

* the verb or command group exists
* a group's sub-command exists
* every long option is one the resolved command declares

What it deliberately does not check: argument values, environment, or whether the command would succeed. Those need a run, and a docs lint should not run anything.

Usage:

- ``python docs/scripts/check_cli_examples.py <paths>``
- ``python docs/scripts/check_cli_examples.py --quiet <paths>`` reports findings only
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import sys

FENCE = re.compile(r"^\s*(?:```+|~~~+)\s*\{?(bash|sh|shell|console|zsh)\b", re.IGNORECASE)
CLOSE = re.compile(r"^\s*(?:```+|~~~+)\s*$")
CONTINUES = re.compile(r"\\\s*$")
PLACEHOLDER = re.compile(r"[<>{}$*]|\.\.\.|\[")


def load_cli():
    import click
    import typer

    from tvbo.cli import app

    return click, typer.main.get_command(app)


def commands(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every `tvbo …` invocation in a shell block, with its line number and continuations joined."""
    out, inside, buffer, start = [], False, "", 0
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").split("\n"), 1):
        if not inside:
            if FENCE.match(line):
                inside = True
            continue
        if CLOSE.match(line):
            inside, buffer = False, ""
            continue
        stripped = line.strip().lstrip("$ ").strip()
        if buffer:
            buffer += " " + stripped.rstrip("\\").strip()
        elif stripped.startswith("tvbo "):
            start, buffer = number, stripped.rstrip("\\").strip()
        else:
            continue
        if not CONTINUES.search(line):
            out.append((start, buffer))
            buffer = ""
    return out


def resolve(click, root, words: list[str]):
    """Walk the command tree, returning (command, consumed) or (None, failing word)."""
    node, index = root, 1
    while index < len(words):
        word = words[index]
        if word.startswith("-"):
            break
        if not isinstance(node, click.Group):
            break
        child = node.commands.get(word)
        if child is None:
            return None, word
        node, index = child, index + 1
    return node, None


def long_options(click, command) -> set[str]:
    names = set()
    for param in getattr(command, "params", []):
        names.update(opt for opt in getattr(param, "opts", []) if opt.startswith("--"))
        names.update(opt for opt in getattr(param, "secondary_opts", []) if opt.startswith("--"))
    return names | {"--help"}


def check(click, root, path: pathlib.Path, tally: dict[str, int]) -> list[str]:
    found = []
    for number, line in commands(path):
        tally["seen"] += 1
        if PLACEHOLDER.search(line):
            tally["templated"] += 1
            continue
        tally["checked"] += 1
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if len(words) < 2:
            continue
        command, missing = resolve(click, root, words)
        if command is None:
            found.append(f"{path}:{number}: unknown command `{missing}` in `{line}`")
            continue
        declared = long_options(click, command)
        for word in words:
            if not word.startswith("--"):
                continue
            option = word.split("=", 1)[0]
            if option not in declared:
                found.append(f"{path}:{number}: `{command.name}` has no option `{option}`")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        click, root = load_cli()
    except Exception as error:
        # Loud, not lenient: this is the only thing standing between the docs and a command that no longer exists, and a green run over zero invocations is indistinguishable from a green run over all of them.
        print(f"cannot import the CLI, so nothing was checked: {error}", file=sys.stderr)
        return 1

    findings, scanned = [], 0
    tally = {"seen": 0, "checked": 0, "templated": 0}
    for path in args.paths:
        if not path.is_file():
            continue
        scanned += 1
        findings.extend(check(click, root, path, tally))

    for finding in findings:
        print(finding)
    if not args.quiet:
        verdict = f"{len(findings)} finding(s)" if findings else "clean"
        print(
            f"\n{verdict}: {tally['checked']} of {tally['seen']} invocations resolved against the CLI "
            f"across {scanned} page(s); {tally['templated']} skipped as templated"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
