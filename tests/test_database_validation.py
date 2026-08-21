"""Validate every YAML in tvbo/database/ against the shipped JSON Schema.

Each YAML file in a known subdirectory is loaded and validated against the
corresponding target LinkML class. The test parametrizes over every file so
that failures point directly at the offending file.

Validation goes through the *shipped* ``tvbo/datamodel/tvbo_datamodel.schema.json``
(generated from the LinkML source by ``hatch_build.py``) and the lightweight
``jsonschema`` library — exactly the path the ``tvbo validate schema`` CLI takes.
Using the shipped artifact rather than re-running LinkML's runtime validator keeps
one validation source of truth (the test can no longer pass while the CLI fails, or
vice-versa) and avoids importing ``linkml`` here, whose enums are mutated to an
unhashable form once ``tvbo`` is imported elsewhere in a combined test run.
"""
import json

import jsonschema
import pytest
import yaml

from .database_corpus import REPO, TARGETS, collect, uncovered

SCHEMA_JSON = REPO / "tvbo" / "datamodel" / "tvbo_datamodel.schema.json"

CASES = collect()
IDS = [str(p.relative_to(REPO)) for p, _ in CASES]


@pytest.fixture(scope="module")
def validators():
    """One ``jsonschema`` validator per target class, ``$ref``-ing into ``$defs``.

    Mirrors ``tvbo validate schema``: each document is validated as an instance of
    its target class via a ``$ref`` into the schema's ``$defs``.
    """
    if not SCHEMA_JSON.exists():
        pytest.skip(
            f"Generated JSON Schema missing at {SCHEMA_JSON}; run `make gen-linkml`."
        )
    full = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    defs = full.get("$defs", {})
    cache = {}
    for cls in set(TARGETS.values()):
        class_schema = {
            "$schema": full.get("$schema"),
            "$defs": defs,
            "$ref": f"#/$defs/{cls}",
        }
        validator_cls = jsonschema.validators.validator_for(class_schema)
        cache[cls] = validator_cls(class_schema)
    return cache


@pytest.mark.parametrize(("path", "target_class"), CASES, ids=IDS)
def test_database_yaml_validates(validators, path, target_class):
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        pytest.skip(f"{path} is not a top-level mapping")
    messages = [
        f"{e.message} in /{'/'.join(str(p) for p in e.absolute_path)}"
        for e in validators[target_class].iter_errors(data)
    ]
    assert not messages, (
        f"{path.relative_to(REPO)} failed validation as {target_class}:\n  - "
        + "\n  - ".join(messages)
    )


def test_every_directory_of_metadata_has_a_class():
    """No corner of the database may be authored metadata that nothing validates.

    ``coordinate_spaces`` and ``reducers`` each sat outside `TARGETS` for as long as they
    existed, so nothing checked them — one had a `description` its class did not declare,
    the other had no class at all.
    """
    assert not uncovered(), (
        "database directories no entry of TARGETS claims: "
        + ", ".join(uncovered())
        + ". Add the class to tests/database_corpus.py (declaring it in the schema first "
        "if it has none), so validation and the golden dump both cover it."
    )


def test_an_iri_identifies_one_record():
    """Two records claiming one ``iri`` disagree about which of them is that entity.

    ``iri`` is identity, and `enrich()` reads it as the entity to fill from — so a second
    record claiming it is filled from the first, silently. `ReducedWongWangFunc` states the
    sigmoid as a `function` H and once claimed `tvbo:ReducedWongWang`; enriching it added a
    *derived variable* H from the canonical record beside its own function of that name.
    A variant states `derived_from_model:` instead, which relates without asserting.
    """
    claims = {}
    collisions = {}
    for path, _ in CASES:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or not data.get("iri"):
            continue
        held = claims.setdefault(data["iri"], [])
        held.append(f"{data.get('name')} ({path.relative_to(REPO)})")
        if len(held) > 1:
            collisions[data["iri"]] = held

    assert not collisions, "one iri, several records:\n" + "\n".join(
        f"  {iri}: " + ", ".join(held) for iri, held in sorted(collisions.items())
    )
