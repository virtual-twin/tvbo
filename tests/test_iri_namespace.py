"""A curated record's IRI and a schema term's IRI never share a path, and every record IRI written anywhere is a term.

Every curated record is ``https://w3id.org/tvbo/<folder>/<file stem>`` (:func:`tvbo.data.registry.record_iri`), its folder the database folder it is filed in. Every schema term sits at the root or under its module's namespace (``study/``, ``dbs/``, ``sw/``, ``units/``, ...). The two stay apart only while no folder is also a module's namespace. The hand-written ontology stays attached to the records, and a spec's IRI dereferences once w3id redirects it, only while each names a record by the IRI the A-box mints for it.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

import pytest
import yaml

from tvbo.data import registry

pytestmark = pytest.mark.backend_core

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "scripts" / "ontology"))

import gen_abox  # noqa: E402

GENERATED_TTL = {"tvb-o-data.ttl", "tvb-o-biology.ttl", "tvb-o.shacl.ttl", "tvb-o-neuroml.ttl"}
"""The ``ontology/*.ttl`` files a ``make`` target writes; the rest are written by hand."""

HAND_WRITTEN_TTL = sorted(p for p in (REPO / "ontology").glob("*.ttl") if p.name not in GENERATED_TTL)

FOLDERS = sorted(p.name for p in registry.DATABASE_ROOT.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
"""Every database folder, the IRI scope of the records filed in it; the registry resolves most of them, and the A-box mints IRIs for some it does not (``coordinate_spaces``)."""

SPEC_ROOTS = [
    REPO / "tvbo" / "database",
    REPO / "tvbo" / "templates",
    REPO / "tvbo" / "skills",
    REPO / "skills",
    REPO / "docs",
]
"""Where specs, and the pages and skills that teach them, write record IRIs."""

GENERATED_DIRS = {REPO / "docs" / "api", REPO / "docs" / "datamodel"}
"""Pages a docs build writes from docstrings and the schema; with every directory whose name starts with ``_`` or ``.`` (``docs/_site``, ``docs/_freeze``) they are not searched."""

RECORD_IRI = re.compile(r"(?:tvbo:|" + re.escape(registry.NAMESPACE) + r")(" + "|".join(FOLDERS) + r")/([\w-]+(?:\.[\w-]+)*)")
"""A record IRI as a CURIE or in full, ending before any trailing sentence punctuation."""


def _segment(iri: str) -> str | None:
    """The first path segment of *iri* under the TVB-O namespace, or ``None`` when it is not under it or is the namespace itself."""
    if not iri.startswith(registry.NAMESPACE):
        return None
    return iri[len(registry.NAMESPACE) :].split("/", 1)[0].split("#", 1)[0] or None


def _module_namespaces() -> dict[str, str]:
    """Every namespace segment a schema module or an ontology file declares, mapped to the file that declares it."""
    found: dict[str, str] = {}
    for path in sorted((REPO / "schema").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        prefixes = (document.get("prefixes") or {}).values()
        iris = [document.get("id") or ""] + [
            p.get("prefix_reference", "") if isinstance(p, dict) else str(p) for p in prefixes
        ]
        found.update({segment: path.name for segment in map(_segment, iris) if segment})
    for path in sorted((REPO / "ontology").glob("*.ttl")):
        for iri in re.findall(r"@prefix\s+[\w-]*:\s*<([^>]+)>", path.read_text(encoding="utf-8")):
            if segment := _segment(iri):
                found[segment] = path.name
    return found


@functools.cache
def _records(folder: str) -> frozenset[str]:
    """The names and file stems of the records *folder* files, every local name the registry resolves in it."""
    index = registry._build_name_index(registry.DATABASE_ROOT / folder)
    return frozenset(index) | {path.stem for path in index.values()}


@pytest.fixture(scope="module")
def abox_subjects() -> set[str]:
    """Every subject of the A-box ``gen_abox`` builds from the database."""
    return {str(subject) for subject in gen_abox.build_graph().subjects()}


def _spec_files() -> list[Path]:
    """The YAML, Markdown, Quarto and template files under :data:`SPEC_ROOTS`."""
    return sorted(
        path
        for root in SPEC_ROOTS
        for path in root.rglob("*")
        if path.suffix in {".yaml", ".yml", ".md", ".qmd", ".tmpl"}
        and not any(part.startswith(("_", ".")) for part in path.relative_to(root).parent.parts)
        and not any(path.is_relative_to(generated) for generated in GENERATED_DIRS)
    )


@pytest.mark.parametrize("folder", sorted(registry._IRI_SCOPES))
def test_every_filed_record_is_an_individual_of_the_abox(folder: str, abox_subjects: set[str]):
    """Each record the registry resolves, one filed in a subfolder too, is the term its :func:`~tvbo.data.registry.record_iri` names."""
    index = registry._build_name_index(registry.DATABASE_ROOT / folder)
    unminted = sorted(
        name
        for name, path in index.items()
        if str(gen_abox.expand_curie(registry.record_iri(folder, path))) not in abox_subjects
    )
    assert not unminted, f"{folder}/ files records the A-box does not mint: {unminted}"


def test_every_record_iri_a_spec_writes_is_a_term(abox_subjects: set[str]):
    """A spec, page or skill that names a record names the term w3id will serve; a record's ``name:`` or a stem in another case resolves at run time but is no term, and ``tvbo migrate-iris`` rewrites it."""
    dangling = sorted(
        f"{path.relative_to(REPO)}: tvbo:{folder}/{local}"
        for path in _spec_files()
        for folder, local in set(RECORD_IRI.findall(path.read_text(encoding="utf-8")))
        if f"{registry.NAMESPACE}{folder}/{local}" not in abox_subjects
    )
    assert not dangling, "record IRIs that name no term:\n" + "\n".join(dangling)


def test_no_folder_is_a_schema_module_namespace():
    """A folder that were also a module's namespace would let a record and a schema term claim one IRI."""
    modules = _module_namespaces()
    shared = {folder: modules[folder] for folder in FOLDERS if folder in modules}
    assert not shared, f"database folders that are also module namespaces: {shared}"


@pytest.mark.parametrize("path", HAND_WRITTEN_TTL, ids=lambda p: p.name)
def test_the_hand_written_ontology_names_records_that_are_filed(path: Path, abox_subjects: set[str]):
    """Each ``<folder>/<stem>`` IRI a hand-written file states is a record the A-box mints, so its statements land on that individual."""
    text = path.read_text(encoding="utf-8")
    pattern = re.escape(registry.NAMESPACE) + r"(" + "|".join(FOLDERS) + r")/([^/#>\s]+)"
    missing = sorted(
        {
            f"{folder}/{name}"
            for folder, name in re.findall(pattern, text)
            if f"{registry.NAMESPACE}{folder}/{name}" not in abox_subjects
        }
    )
    assert not missing, f"{path.name} names records the A-box does not mint: {missing}"


@pytest.mark.parametrize("path", HAND_WRITTEN_TTL, ids=lambda p: p.name)
def test_the_hand_written_ontology_names_no_model_by_its_root_iri(path: Path):
    """A model is the record ``models/<name>``; a statement about ``tvbo:<name>`` would describe an individual nothing else mentions."""
    text = path.read_text(encoding="utf-8")
    models = _records("models")
    flat = re.findall(r"(?<![\w/])tvbo:([A-Za-z_][\w.-]*)", text) + re.findall(
        re.escape(f"<{registry.NAMESPACE}") + r"([A-Za-z_][\w.-]*)>", text
    )
    stale = sorted({name for name in flat if name in models})
    assert not stale, f"{path.name} names models at the root instead of under models/: {stale}"
