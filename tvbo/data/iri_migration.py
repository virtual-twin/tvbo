"""Rewrite the curated-record IRIs a spec names into their canonical ``tvbo:<folder>/<file stem>`` form.

A curated record's IRI is scoped by the database folder it is filed in and ends in its file stem (:func:`tvbo.data.registry.record_iri`). The registry also answers to other spellings, and each is rewritten into that one. A singular scope (``tvbo:observation/bold_tvb``) maps to its folder. A scoped name that is the record's ``name:`` or its stem in another case (``tvbo:observation_models/BOLD_TVB``, ``tvbo:atlases/DesikanKilliany``) becomes its stem. An unscoped name (``tvbo:Linear``) is the record the slot it sits in resolves it to when that slot ranges over one curated class (``network: {iri: tvbo:DesikanKilliany}`` is the structural connectome filed under that atlas), else the record of that name in the one folder that files it.

:func:`migrate_text` rewrites both in a YAML document, or in the fenced YAML blocks of a Markdown or Quarto page, and leaves every other byte as written. A name it cannot place is reported and left alone.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from tvbo.data.registry import (
    _CATEGORIES,
    _IRI_RE,
    _IRI_SCOPES,
    _LEGACY_SCOPES,
    DATABASE_ROOT,
    iri_for,
    local_name,
)

_UNSCOPED = re.compile(r"^tvbo:(?P<name>[A-Za-z_][\w.-]*)$")
_FENCE = re.compile(r"^(?P<fence>`{3,}|~{3,})[ \t]*\{?\.?ya?ml\b[^\n]*\n(?P<body>.*?)^(?P=fence)[ \t\r]*$", re.M | re.S)
_MARKDOWN = {".md", ".qmd"}
_YAML = {".yaml", ".yml"}


@dataclass(frozen=True)
class Change:
    """One curated IRI the migration met: rewritten to *new*, or left as written with the *reason* why."""

    line: int
    old: str
    new: str | None = None
    reason: str = ""


@cache
def _json_schema() -> dict:
    """The generated JSON Schema of the datamodel, the schema's slots and terms as the build wrote them."""
    import tvbo.datamodel

    return json.loads((Path(tvbo.datamodel.__file__).parent / "tvbo_datamodel.schema.json").read_text(encoding="utf-8"))


@cache
def _slot_folders() -> dict[str, frozenset[str]]:
    """Every slot the schema declares, mapped to the database folders of the curated classes it ranges over (empty for any other range)."""

    def refs(node):
        """The class names every ``$ref`` under *node* points at."""
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                yield ref.rsplit("/", 1)[-1].removesuffix("__identifier_optional")
            for value in node.values():
                yield from refs(value)
        elif isinstance(node, list):
            for value in node:
                yield from refs(value)

    folders: dict[str, set[str]] = defaultdict(set)
    for definition in _json_schema()["$defs"].values():
        for slot, prop in (definition.get("properties") or {}).items():
            folders[slot] |= {_CATEGORIES[cls] for cls in refs(prop) if cls in _CATEGORIES}
    return {slot: frozenset(found) for slot, found in folders.items()}


@cache
def _schema_terms() -> frozenset[str]:
    """Every class, enum, slot and permissible-value name the schema declares: an unscoped IRI spelling one of them names that term unless its slot says otherwise."""
    schema = _json_schema()
    terms = set(schema["$defs"])
    for definition in schema["$defs"].values():
        terms |= set(definition.get("properties") or {})
        terms |= {str(value) for value in definition.get("enum") or []}
    return frozenset(terms)


def _context(keys: tuple[str | None, ...], root: frozenset[str]) -> frozenset[str]:
    """The folders the nearest enclosing slot files its records in: none when it ranges over no curated class, the document's own when no enclosing key is a slot."""
    slots = _slot_folders()
    for key in reversed(keys):
        if key != "iri" and key in slots:
            return slots[key]
    return root


@cache
def _record_iri(folder: str, name: str) -> str | None:
    """The canonical IRI of the record the registry resolves *name* to in *folder* (by its ``name:`` or file stem in any case, or a network by its atlas), or ``None`` when it resolves none."""
    try:
        return iri_for(_IRI_SCOPES[folder], name)
    except FileNotFoundError:
        return None


def _canonical(value: str, keys: tuple[str | None, ...], root: frozenset[str]) -> Change | None:
    """The change *value* takes at the slot path *keys*, or ``None`` when it is canonical already or names no curated record."""
    scoped = _IRI_RE.match(value)
    if scoped:
        prefix, scope, name = scoped.group("prefix", "scope", "name")
        folder = _LEGACY_SCOPES.get(scope, scope)
        if folder not in _IRI_SCOPES:
            return None
        new = (_record_iri(folder, name) if prefix == "tvbo" else None) or f"{prefix}:{folder}/{name}"
        return Change(0, value, new) if new != value else None
    unscoped = _UNSCOPED.match(value)
    if not unscoped:
        return None
    name = unscoped.group("name")
    context = _context(keys, root)
    if len(context) == 1 and (resolved := _record_iri(next(iter(context)), name)):
        return Change(0, value, resolved)
    held = frozenset(folder for folder in _IRI_SCOPES if _record_iri(folder, name))
    if not held:
        return None
    candidates = held & context if context else held
    if len(candidates) == 1 and (context or name not in _schema_terms()):
        return Change(0, value, _record_iri(next(iter(candidates)), name))
    if context and not candidates:
        return Change(0, value, reason=f"its slot files {', '.join(sorted(context))}, none of which resolves {name!r}")
    if len(candidates) > 1:
        return Change(0, value, reason=f"{', '.join(sorted(held))} each resolve {name!r} and no slot says which")
    return Change(
        0, value, reason=f"{name!r} is also a schema term and no slot says it names the record in {next(iter(held))}"
    )


def _scalars(node, keys: tuple[str | None, ...] = ()):
    """``(scalar node, enclosing keys)`` for every plain or quoted scalar under *node* that starts with ``tvbo:``."""
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            name = key.value if isinstance(key, yaml.ScalarNode) else None
            if name != "tvbo_class":
                yield from _scalars(value, (*keys, name))
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            yield from _scalars(value, keys)
    elif isinstance(node, yaml.ScalarNode) and node.style not in ("|", ">") and node.value.startswith("tvbo:"):
        yield node, keys


def _document_folders(document, path: Path | None) -> frozenset[str]:
    """The folder a document's own records are filed in: its ``tvbo_class`` envelope's, else the database folder *path* sits in."""
    if isinstance(document, yaml.MappingNode):
        for key, value in document.value:
            if isinstance(key, yaml.ScalarNode) and key.value == "tvbo_class" and isinstance(value, yaml.ScalarNode):
                folder = _CATEGORIES.get(local_name(value.value))
                if folder:
                    return frozenset({folder})
    if path is not None and DATABASE_ROOT is not None:
        try:
            folder = path.resolve().relative_to(DATABASE_ROOT.resolve()).parts[0]
        except ValueError:
            return frozenset()
        if folder in _IRI_SCOPES:
            return frozenset({folder})
    return frozenset()


def _migrate_yaml(text: str, offset: int, path: Path | None, edits: list, changes: list) -> str | None:
    """Collect the edits one YAML text needs, positioned at *offset* in its file; the parse error when it is not YAML."""
    try:
        documents = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as error:
        return str(error).splitlines()[0]
    seen: set[int] = set()
    for document in documents:
        root = _document_folders(document, path)
        for node, keys in _scalars(document):
            if node.start_mark.index in seen:
                continue
            seen.add(node.start_mark.index)
            change = _canonical(node.value, keys, root)
            if change is None:
                continue
            start, end = offset + node.start_mark.index, offset + node.end_mark.index
            if change.new and change.old not in text[node.start_mark.index : node.end_mark.index]:
                change = Change(0, change.old, reason="it is spelled with escapes; rewrite it by hand")
            changes.append((start, change))
            if change.new:
                edits.append((start, end, node.value, change.new))
    return None


def migrate_text(text: str, path: Path | None = None) -> tuple[str, list[Change], list[str]]:
    """``(rewritten text, changes, blocks skipped as not YAML)`` for a YAML document, or for a Markdown/Quarto page when *path* says it is one."""
    edits: list[tuple[int, int, str, str]] = []
    located: list[tuple[int, Change]] = []
    skipped: list[str] = []
    if path is not None and path.suffix.lower() in _MARKDOWN:
        for block in _FENCE.finditer(text):
            error = _migrate_yaml(block.group("body"), block.start("body"), path, edits, located)
            if error:
                skipped.append(f"line {text.count(chr(10), 0, block.start('body')) + 1}: {error}")
    else:
        error = _migrate_yaml(text, 0, path, edits, located)
        if error:
            skipped.append(error)
    changes = [
        Change(text.count("\n", 0, start) + 1, c.old, c.new, c.reason) for start, c in sorted(located, key=lambda x: x[0])
    ]
    for start, end, old, new in sorted(edits, reverse=True):
        text = text[:start] + text[start:end].replace(old, new, 1) + text[end:]
    return text, changes, skipped


def spec_files(paths: list[Path]) -> list[Path]:
    """The YAML, Markdown and Quarto files at or under *paths*, sorted; a hidden directory under a searched one (``.venv``, ``.git``, ``.quarto``) is not entered."""
    suffixes = _YAML | _MARKDOWN
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found |= {
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in suffixes
                and not any(part.startswith(".") for part in p.relative_to(path).parts)
            }
        elif path.suffix.lower() in suffixes:
            found.add(path)
    return sorted(found)
