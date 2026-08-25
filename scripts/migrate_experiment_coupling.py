"""Move an experiment's `coupling:` under its `network:`, where 1.0 declares it.

A coupling acts over a connectivity, so `SimulationExperiment` has no coupling slot: a recipe that still declares one raises `TypeError: SimulationExperiment.__init__() got an unexpected keyword argument 'coupling'` and the study does not load at all. `network.coupling` is keyed by name, so the block moves one level down and gains its name as the key.

    coupling:                          network:
      name: Linear        becomes        ...
      iri: tvbo:Linear                   coupling:
                                           Linear:
                                             name: Linear
                                             iri: tvbo:Linear

The rewrite is line-based, not a YAML round-trip. These recipes are hand-authored -- wrapped flow sequences, aligned comments, anchors, `!include` tags -- and a round-trip reformats all of it, burying the one intended change in a thousand-line diff. So only the coupling block's own lines move; every other byte is left as written.

Reports by default and rewrites only under `--apply`. A site it cannot migrate unambiguously is reported and left alone rather than guessed at.

    python scripts/migrate_experiment_coupling.py <path>...            # report
    python scripts/migrate_experiment_coupling.py --apply <path>...    # rewrite
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"^(?P<indent> *)coupling:\s*(?P<anchor>&\S+)?\s*$")
"""A block-style `coupling:` opening its own mapping, optionally anchored."""

FLOW = re.compile(r"^(?P<indent> *)coupling:\s*(?P<body>\{.*\})\s*$")
"""A one-line flow `coupling: {name: Linear, ...}`."""

REF = re.compile(r"^(?P<indent> *)coupling:\s*\*(?P<ref>\S+)\s*$")
"""A one-line `coupling: *anchor`, which names its coupling only through the anchor."""

NAME = re.compile(r"^(?P<indent> *)name:\s*[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
"""The coupling's own `name:`. Matched at its own indent only -- a nested `parameters: {a: {name: a}}` also carries one, and keying the coupling `a` would rename it."""


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index one past the last line belonging to the block opened at *start*."""
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


ALIAS = re.compile(r"^(?P<indent> *)network:\s*\*(?P<anchor>\S+)\s*$")


def _network_insert(lines: list[str], before: int, indent: int) -> int | None:
    """Where to insert inside the sibling `network:` nearest above *before*.

    A `network: *anchor` alias is expanded in place to a block that merges the anchor, so the experiment keeps the shared network while gaining a coupling of its own.
    """
    opener = re.compile(rf"^ {{{indent}}}network:\s*(&\S+)?\s*$")
    for i in range(before - 1, -1, -1):
        line = lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) < indent:
            return None
        if opener.match(line):
            return _block_end(lines, i, indent)
        alias = ALIAS.match(line)
        if alias and len(alias.group("indent")) == indent:
            lines[i : i + 1] = [f"{alias.group('indent')}network:\n", f"{alias.group('indent')}  <<: *{alias.group('anchor')}\n"]
            return i + 2
    return None


def _already_declared(lines: list[str], insert: int, indent: int) -> bool:
    """Whether the network block ending at *insert* already declares a coupling."""
    opener = re.compile(rf"^ {{{indent + 2}}}coupling:")
    for i in range(insert - 1, -1, -1):
        line = lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            return False
        if opener.match(line):
            return True
    return False


def _inside_network(lines: list[str], start: int, indent: int) -> bool:
    """Whether the block at *start* is already nested inside a `network:` mapping."""
    opener = re.compile(r"^ *network:\s*(&\S+)?\s*$")
    for i in range(start - 1, -1, -1):
        line = lines[i]
        if not line.strip():
            continue
        here = len(line) - len(line.lstrip())
        if here < indent:
            return bool(opener.match(line))
    return False


MERGE = re.compile(r"^ *<<:\s*\*(?P<anchor>\S+)\s*$")


def _anchored_name(lines: list[str], anchor: str) -> str | None:
    """The `name:` declared by the block carrying `&anchor`, wherever it is defined."""
    opener = re.compile(rf"^(?P<indent> *)\S+:\s*&{re.escape(anchor)}\s*$")
    for i, line in enumerate(lines):
        found = opener.match(line)
        if not found:
            continue
        indent = len(found.group("indent"))
        return _own_name(lines[i + 1 : _block_end(lines, i, indent)], indent + 2, lines, False)
    return None


def _own_name(body: list[str], indent: int, lines: list[str], flow: bool) -> str | None:
    """The coupling's own `name:`, or the one it inherits through `<<:`.

    Matched at the coupling's own indent: a nested ``parameters: {a: {name: a}}`` carries a ``name:`` too, and keying the coupling ``a`` would rename it.
    """
    if flow:
        found = re.search(r"\bname:\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)", body[0])
        return found.group(1) if found else None
    for line in body:
        found = NAME.match(line)
        if found and len(found.group("indent")) == indent:
            return found.group("name")
    for line in body:
        merge = MERGE.match(line)
        if merge:
            return _anchored_name(lines, merge.group("anchor"))
    return None


def _one_site(lines: list[str], skip: int) -> tuple | None:
    """The *skip*-th migratable coupling from the bottom, with what is known about it."""
    seen = 0
    for i in range(len(lines) - 1, -1, -1):
        block, flow, ref = BLOCK.match(lines[i]), FLOW.match(lines[i]), REF.match(lines[i])
        match = block or flow or ref
        if not match:
            continue
        indent = len(match.group("indent"))
        if _inside_network(lines, i, indent):
            continue  # already declared where 1.0 wants it
        if seen < skip:
            seen += 1
            continue
        end = _block_end(lines, i, indent) if block else i + 1
        if ref:
            name, body = _anchored_name(lines, ref.group("ref")), []
        else:
            body = lines[i + 1 : end] if block else [match.group("body")]
            name = _own_name(body, indent + 2, lines, flow is not None)
        return i, indent, end, body, name, block, flow, ref, match
    return None


def _migrate_text(text: str) -> tuple[str, list[str]]:
    """Relocate every removed-spelling coupling in *text*, and say what happened.

    One move per pass, then a full rescan: a move shifts every index below it, so walking the list while mutating it processes shifted content and can migrate one site twice. `skip` steps past the sites already found unmigratable, which is what makes the loop terminate.
    """
    lines = text.splitlines(keepends=True)
    notes: list[str] = []
    skip = 0
    while (site := _one_site(lines, skip)) is not None:
        i, indent, end, body, name, block, flow, ref, match = site
        insert = None if name is None else _network_insert(lines, i, indent)

        if name is None:
            notes.append(f"line {i + 1}: SKIPPED - no `name:` to key it by")
            skip += 1
        elif insert is None:
            notes.append(f"line {i + 1}: SKIPPED - no sibling `network:` mapping to move it into")
            skip += 1
        elif _already_declared(lines, insert, indent):
            notes.append(f"line {i + 1}: SKIPPED - `network.coupling` already declared; merge by hand")
            skip += 1
        else:
            pad = " " * (indent + 2)
            # The anchor rides along on the re-keyed line: dropping it leaves every later
            # `*alias` pointing at a definition that no longer exists.
            tag = f" {match.group('anchor')}" if block and match.group("anchor") else ""
            if ref:
                nested = [f"{pad}coupling:\n", f"{pad}  {name}: *{ref.group('ref')}\n"]
            elif block:
                nested = [f"{pad}coupling:\n", f"{pad}  {name}:{tag}\n"] + ["    " + b if b.strip() else b for b in body]
            else:
                nested = [f"{pad}coupling:\n", f"{pad}  {name}: {match.group('body')}\n"]
            lines = lines[:insert] + nested + lines[insert:i] + lines[end:]
            notes.append(f"line {i + 1}: moved into `network.coupling`, keyed as {name!r}")
    return "".join(lines), list(reversed(notes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="YAML files, or directories to search recursively.")
    parser.add_argument("--apply", action="store_true", help="Rewrite the files instead of reporting.")
    args = parser.parse_args(argv)

    files = sorted({f for p in args.paths for f in ([p] if p.is_file() else p.rglob("*.y*ml"))})
    changed = skipped = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "coupling:" not in text:
            continue
        migrated, notes = _migrate_text(text)
        for note in notes:
            print(f"{path}: {note}")
            skipped += "SKIPPED" in note
        if migrated != text:
            changed += 1
            if args.apply:
                path.write_text(migrated, encoding="utf-8")

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {changed} file(s) of {len(files)} scanned; {skipped} need a hand.")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
