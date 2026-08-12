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
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCHEMA_JSON = REPO / "tvbo" / "datamodel" / "tvbo_datamodel.schema.json"
DB = REPO / "tvbo" / "database"

# Map subdirectory -> target LinkML class
TARGETS = {
    "models": "Dynamics",
    "coupling_functions": "Coupling",
    "integrators": "Integrator",
    "observation_models": "Observation",
    "experiments": "SimulationExperiment",
    "studies": "SimulationStudy",
    "networks": "Network",
    "atlases": "BrainAtlas",
    "software": "SimulationTool",
    "continuations": "Continuation",
    "graph_generators": "GraphGenerator",
}


def _collect():
    cases = []
    for sub, cls in TARGETS.items():
        for path in sorted((DB / sub).rglob("*.y*ml")):
            cases.append((path, cls))
    return cases


CASES = _collect()
IDS = [str(p.relative_to(REPO)) for p, _ in CASES]


@pytest.fixture(scope="module")
def validators():
    """One ``jsonschema`` validator per target class, ``$ref``-ing into ``$defs``.

    Mirrors ``tvbo validate schema``: each document is validated as an instance of
    its target class via a ``$ref`` into the schema's ``$defs``.
    """
    if not SCHEMA_JSON.exists():
        pytest.skip(f"Generated JSON Schema missing at {SCHEMA_JSON}; run `make gen-linkml`.")
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
        f"{e.message} in /{'/'.join(str(p) for p in e.absolute_path)}" for e in validators[target_class].iter_errors(data)
    ]
    assert not messages, f"{path.relative_to(REPO)} failed validation as {target_class}:\n  - " + "\n  - ".join(messages)
