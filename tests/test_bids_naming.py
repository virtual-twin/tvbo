"""The curated atlases and networks are named by BIDS (register I11i): each stem parses, spells its entities in one order, names its template by its TemplateFlow identifier, and agrees with the files it ships and with the record's own ``bids:`` block; a surface spells its parent network's ``cohort-`` and ``rec-``. Each coordinate space a record states is the template its name spells and a curated coordinate-space record, named by its TemplateFlow identifier.

Only the files at the top of ``atlases/`` and ``networks/`` are records; ``networks/bids/dk_average/`` is a BEP017 dataset of its own, read as a directory, and is not named here. A stem with no ``-`` in it (``SmileyNetwork``) names no BIDS entity and is not held to these rules.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from bids.layout.writing import build_path

from tvbo.data.cohort import BIDS_ENTITY_SHORT_KEYS
from tvbo.data.converters import relmat_entities
from tvbo.data.network_io import RELMAT_PATTERNS
from tvbo.data.registry import DATABASE_ROOT

pytestmark = pytest.mark.backend_core

FOLDERS = ("atlases", "networks")

ORDER = ("sub", "tpl", "ses", "cohort", "acq", "rec", "hemi", "space", "atlas", "seg", "scale", "res", "den", "desc")
"""The order a stem spells its entities in (I11i's R1)."""

TEMPLATES = frozenset({"MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNI152NLin2009bAsym", "fsLR", "fsaverage"})
"""The TemplateFlow identifiers the curated records are in; a record in another template adds its identifier here."""

NO_TEMPLATE = frozenset({"acq-EEGBrainstorm65_sensors", "acq-MEGBrainstorm276_sensors", "acq-SEEG588_sensors"})
"""Sensor arrays positioned in Brainstorm's and TVB's own head frames, which no standard template names (I11i's R2 exceptions)."""

COORDINATE_SPACES = TEMPLATES | {"MNI152NLin2009bSym", "MNI152NLin2009cSym"}
"""The TemplateFlow identifiers ``coordinate_spaces/`` holds a record of."""


def _stem(path: Path) -> str:
    """A file's name without its extensions, ``.nii.gz`` whole."""
    return path.name.split(".", 1)[0]


def _parse(stem: str) -> tuple[list[tuple[str, str]], str] | None:
    """The entities a BIDS-style stem spells, in its order, and its suffix; ``None`` when it is not one."""
    *pairs, suffix = stem.split("_")
    if not pairs or not suffix.isalpha() or not all(re.fullmatch(r"[a-z]+-[0-9A-Za-z]+", pair) for pair in pairs):
        return None
    return [tuple(pair.split("-", 1)) for pair in pairs], suffix


FILES = [
    path
    for folder in FOLDERS
    for path in sorted((DATABASE_ROOT / folder).iterdir())
    if path.is_file() and path.name != "dataset_description.json"
]
NAMED = [path for path in FILES if "-" in _stem(path)]
NETWORK_RECORDS = [path for path in NAMED if path.parent.name == "networks" and path.suffix == ".yaml"]
SURFACES = [path for path in NETWORK_RECORDS if "parent_network" in yaml.safe_load(path.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("path", NAMED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_bids_style_stem_parses(path: Path):
    assert _parse(_stem(path)) is not None, f"{path.name} is not a run of key-value entities and a suffix"


@pytest.mark.parametrize("path", NAMED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_entities_are_spelled_in_one_order(path: Path):
    keys = [key for key, _ in _parse(_stem(path))[0]]
    unknown = [key for key in keys if key not in ORDER]
    assert not unknown, f"{path.name} spells entities outside {ORDER}: {unknown}"
    assert keys == sorted(keys, key=ORDER.index) and len(set(keys)) == len(keys), f"{path.name} spells {keys}, out of {ORDER}"


@pytest.mark.parametrize("path", NAMED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_template_is_named_by_its_templateflow_identifier(path: Path):
    entities = dict(_parse(_stem(path))[0])
    if _stem(path) in NO_TEMPLATE:
        assert "tpl" not in entities
        return
    assert "tpl" in entities or "sub" in entities, f"{path.name} names no template"
    assert entities.get("tpl", next(iter(TEMPLATES))) in TEMPLATES, (
        f"{path.name}: tpl-{entities['tpl']} is not one of {sorted(TEMPLATES)}"
    )


@pytest.mark.parametrize("path", [p for p in FILES if p.suffix != ".yaml"], ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_data_file_shares_its_records_stem(path: Path):
    """An image, centres table or companion is found by its record's stem, so it carries that stem: ``<stem>.nii.gz`` and ``<stem>.h5`` beside ``<stem>.yaml``, ``<x>_centers.txt`` beside ``<x>_dseg.yaml``."""
    stem = _stem(path)
    record = path.with_name(f"{stem.removesuffix('_centers')}_dseg.yaml" if stem.endswith("_centers") else f"{stem}.yaml")
    assert record.name in {p.name for p in path.parent.glob("*.yaml")}, f"{path.name} has no record {record.name}"


@pytest.mark.parametrize("folder", FOLDERS)
def test_no_two_spellings_differ_only_by_case(folder: str):
    """A value is spelled one way across a folder, so ``hcpmmp1`` and ``HCPMMP1``, or ``Nlin`` and ``NLin``, never name one thing twice."""
    spellings: dict[str, set[str]] = {}
    for path in NAMED:
        if path.parent.name == folder:
            entities, suffix = _parse(_stem(path))
            for value in [*(value for _, value in entities), suffix]:
                spellings.setdefault(value.lower(), set()).add(value)
    clashes = sorted(sorted(found) for found in spellings.values() if len(found) > 1)
    assert not clashes, f"{folder}/ spells these values in more than one case: {clashes}"


@pytest.mark.parametrize("path", NETWORK_RECORDS, ids=lambda p: p.name)
def test_bids_block_spells_the_file_name(path: Path):
    """A network's ``bids:`` block states the entities and the suffix its file name spells, no more and no fewer."""
    block = yaml.safe_load(path.read_text(encoding="utf-8")).get("bids") or {}
    entities, suffix = _parse(path.stem)
    stated = {BIDS_ENTITY_SHORT_KEYS.get(key, key): str(value) for key, value in block.items() if key != "suffix"}
    assert (stated, block.get("suffix")) == (dict(entities), suffix)


@pytest.mark.parametrize("path", [p for p in NETWORK_RECORDS if p.stem.endswith("_relmat")], ids=lambda p: p.name)
def test_relmat_file_name_is_built_back_from_the_record(path: Path):
    """``Network.bids_filename``'s patterns spell a curated connectome's name from its record as the database spells it."""
    sidecar = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert build_path(relmat_entities(sidecar), RELMAT_PATTERNS) == f"{path.stem}.h5"


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: p.name)
def test_a_surface_spells_its_parent_networks_cohort_and_reconstruction(path: Path):
    """A surface maps its vertices onto the regions of its ``parent_network``, so it spells the parent's ``cohort-`` and ``rec-``, and none the parent does not (I11i-e)."""
    parent = path.with_name(yaml.safe_load(path.read_text(encoding="utf-8"))["parent_network"])
    assert parent.exists(), f"{path.name} names a parent network the database does not hold: {parent.name}"
    mine, theirs = (dict(_parse(p.stem)[0]) for p in (path, parent))
    spelled = {key: (mine.get(key), theirs.get(key)) for key in ("cohort", "rec")}
    assert all(own == inherited for own, inherited in spelled.values()), f"{path.name} spells (own, parent's) {spelled}"


def _coordinate_spaces(node) -> set[str]:
    """Every ``coordinateSpace`` a record states, at any depth, by name."""
    if isinstance(node, list):
        return set().union(*map(_coordinate_spaces, node))
    if not isinstance(node, dict):
        return set()
    stated = node.get("coordinateSpace")
    own = {stated["name"] if isinstance(stated, dict) else str(stated)} if stated else set()
    return own.union(*map(_coordinate_spaces, node.values()))


@pytest.mark.parametrize("path", [p for p in FILES if p.suffix == ".yaml"], ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_coordinate_space_a_record_states_is_a_curated_one(path: Path):
    curated = {p.stem for p in (DATABASE_ROOT / "coordinate_spaces").glob("*.yaml")}
    stated = _coordinate_spaces(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert stated <= curated, (
        f"{path.name} states coordinate spaces coordinate_spaces/ holds no record of: {sorted(stated - curated)}"
    )


@pytest.mark.parametrize("path", [p for p in NAMED if p.suffix == ".yaml"], ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_record_states_the_template_its_name_spells(path: Path):
    """Every ``coordinateSpace`` an atlas or network record states is the template its ``tpl-`` spells, and one whose name spells no template states none."""
    template = dict(_parse(_stem(path))[0]).get("tpl")
    stated = _coordinate_spaces(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert stated <= {template} - {None}, f"{path.name} spells tpl-{template} and states {sorted(stated)}"


@pytest.mark.parametrize("path", sorted((DATABASE_ROOT / "coordinate_spaces").glob("*.yaml")), ids=lambda p: p.stem)
def test_a_coordinate_space_record_is_named_by_its_templateflow_identifier(path: Path):
    assert path.stem in COORDINATE_SPACES, f"{path.stem} is not one of {sorted(COORDINATE_SPACES)}"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == path.stem
