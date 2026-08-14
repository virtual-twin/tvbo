"""Freeze what TVBO serializes every database file back out as.

The database is TVBO's published record. Schema validation says each file is *permitted*;
it says nothing about what TVBO makes of it. A slot silently dropped on load, a default
materialised where the author wrote nothing, an `inf` turned into `null` by a serializer
setting — each leaves every file still valid and still passing, while changing the record.

This corpus freezes the canonical dump of all 443 files and fails on any difference, so
such a change has to be re-baselined under ``--regenerate-golden`` in its own commit and
reviewed as what it is: an edit to the published record.

Building it found three gaps that had nothing to do with serialization:

* 121 network sidecars open with the ``tvbo_class`` envelope TVBO's own dumper writes, and
  TVBO's LinkML loader rejected it — the key was stripped in three other places and not in
  the one both load paths share;
* all five ``coordinate_spaces`` files carry a ``description`` their class did not declare;
* ``reducers`` had no class at all, so its one recipe was read as a raw dict and validated
  by nothing.

With those fixed the corpus is 443/443 with no expected failures, which is why there are
none here: a gap that has to be written down as an ``xfail`` is a gap that could instead be
closed.
"""

from __future__ import annotations

import difflib

import pytest

from .database_corpus import DB, REPO, collect
from .golden import GoldenCorpus

pytestmark = pytest.mark.backend_core

CASES = collect()


def _case_id(path) -> str:
    """``models/Jansen1995.yaml`` -> ``models__Jansen1995`` — flat, stable, unique."""
    return str(path.relative_to(DB).with_suffix("")).replace("/", "__")


IDS = [_case_id(path) for path, _ in CASES]


def _diff(produced: str, expected: str) -> str | None:
    """A unified diff of the two dumps, or ``None`` when they are identical."""
    if produced == expected:
        return None
    lines = list(difflib.unified_diff(expected.splitlines(), produced.splitlines(), "frozen", "produced", lineterm="", n=2))
    head = lines[:40]
    if len(lines) > 40:
        head.append(f"... and {len(lines) - 40} more diff lines")
    return "\n".join(head)


CORPUS = GoldenCorpus(
    REPO / "tests" / "reference_data" / "database",
    ".yaml",
    write=lambda path, produced: path.write_text(produced, encoding="utf-8"),
    read=lambda path: path.read_text(encoding="utf-8"),
    compare=_diff,
)


def _dump(path, class_name: str) -> str:
    """Load one database file as its class and serialize it back through TVBO's dumper."""
    from tvbo.datamodel import schema
    from tvbo.utils import to_yaml, yaml_loader

    return to_yaml(yaml_loader.load(path, target_class=getattr(schema, class_name)))


@pytest.mark.parametrize(("path", "class_name"), CASES, ids=IDS)
def test_the_dump_of_a_database_file_is_unchanged(path, class_name, regenerate: bool):
    produced = _dump(path, class_name)
    assert "_source_file" not in produced, (
        f"{path.relative_to(REPO)} dumped a machine-local path. `_source_file` is where the "
        "file came from, not part of what it says; it must never reach the record."
    )
    CORPUS.check(_case_id(path), produced, regenerate=regenerate, what="database dump")


@pytest.mark.parametrize(("path", "class_name"), CASES, ids=IDS)
def test_a_dump_reloads_to_itself(path, class_name):
    """``dump(load(dump(x))) == dump(x)`` — the dump is a complete statement of the object.

    Independent of the frozen corpus: that one catches a *change*, this one catches a dump
    the loader cannot read back, which would be equally frozen and equally wrong.
    """
    from tvbo.datamodel import schema
    from tvbo.utils import to_yaml, yaml_loader

    once = _dump(path, class_name)
    twice = to_yaml(yaml_loader.loads(once, target_class=getattr(schema, class_name)))
    assert once == twice, _diff(twice, once)


def test_the_corpus_and_the_database_describe_the_same_files(regenerate: bool):
    CORPUS.reconcile(IDS, regenerate=regenerate, what="database files")
