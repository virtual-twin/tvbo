"""Freeze what TVBO serializes every database file back out as.

The database is TVBO's published record. Schema validation says each file is *permitted*;
it says nothing about what TVBO makes of it. A slot silently dropped on load, a default materialised where the author wrote nothing, an `inf` turned into `null` by a serializer setting — each leaves every file still valid and still passing, while changing the record.

This corpus freezes the canonical dump of all 454 files, in each generated form, and fails on any difference, so such a change has to be re-baselined under ``--regenerate-golden`` in its own commit and reviewed as what it is: an edit to the published record.

Building it found three gaps that had nothing to do with serialization:

* 121 network sidecars open with the ``tvbo_class`` envelope TVBO's own dumper writes, and
  TVBO's LinkML loader rejected it — the key was stripped in three other places and not in
  the one both load paths share;
* all five ``coordinate_spaces`` files carry a ``description`` their class did not declare;
* ``reducers`` had no class at all, so its one recipe was read as a raw dict and validated
  by nothing.

With those fixed the corpus is 443/443 with no expected failures, which is why there are none here: a gap that has to be written down as an ``xfail`` is a gap that could instead be closed.
"""

from __future__ import annotations

import difflib

import pytest

from .database_corpus import DB, REPO, collect
from .golden import GoldenCorpus, text_discriminates

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


FORMS = ("schema", "pydantic")
"""The two generated forms of the datamodel, each serving the same database file to its own callers.

LinkML's loader builds a dataclass from ``schema``; Pydantic validation builds a model from ``pydantic``. A corpus over one of them says nothing about the other, and the runtime is served from both — so each is frozen on the path it is actually served from, and the difference between the two records is a diff in this directory rather than a surprise at the moment one replaces the other.
"""


def _corpus(form: str) -> GoldenCorpus:
    """The frozen dumps of one generated form."""
    return GoldenCorpus(
        REPO / "tests" / "reference_data" / (f"database_{form}" if form != "schema" else "database"),
        ".yaml",
        write=lambda path, produced: path.write_text(produced, encoding="utf-8"),
        read=lambda path: path.read_text(encoding="utf-8"),
        compare=_diff,
        discriminates=text_discriminates,
    )


CORPORA = {form: _corpus(form) for form in FORMS}


def _dump(path, class_name: str, form: str) -> str:
    """Load one database file as its class in *form*, and serialize it back through TVBO's dumper.

    One loader and one dumper for both forms — `yaml_loader.load` dispatches on the target class and `to_yaml` on the object — so a difference between the two records is a difference between the generated classes, never between two hand-written paths.
    """
    import importlib

    from tvbo.utils import to_yaml, yaml_loader

    module = importlib.import_module(f"tvbo.datamodel.{form}")
    return to_yaml(yaml_loader.load(path, getattr(module, class_name)))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("path", "class_name"), CASES, ids=IDS)
def test_the_dump_of_a_database_file_is_unchanged(path, class_name, form, regenerate: bool):
    produced = _dump(path, class_name, form)
    assert "_source_file" not in produced, (
        f"{path.relative_to(REPO)} dumped a machine-local path. `_source_file` is where the "
        "file came from, not part of what it says; it must never reach the record."
    )
    CORPORA[form].check(_case_id(path), produced, regenerate=regenerate, what=f"database dump ({form})")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("path", "class_name"), CASES, ids=IDS)
def test_a_dump_reloads_to_itself(path, class_name, form):
    """``dump(load(dump(x))) == dump(x)`` — the dump is a complete statement of the object.

    Independent of the frozen corpus: that one catches a *change*, this one catches a dump the loader cannot read back, which would be equally frozen and equally wrong.
    """
    import importlib

    from tvbo.utils import to_yaml, yaml_loader

    module = importlib.import_module(f"tvbo.datamodel.{form}")
    once = _dump(path, class_name, form)
    twice = to_yaml(yaml_loader.loads(once, target_class=getattr(module, class_name)))
    assert once == twice, _diff(twice, once)


@pytest.mark.parametrize("form", FORMS)
def test_the_corpus_and_the_database_describe_the_same_files(form: str, regenerate: bool):
    CORPORA[form].reconcile(IDS, regenerate=regenerate, what=f"database files ({form})")
