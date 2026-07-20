"""Validate every YAML in tvbo/database/ against the LinkML schema.

Each YAML file in a known subdirectory is loaded and validated against the
corresponding target LinkML class. The test parametrizes over every file so
that failures point directly at the offending file.
"""
import os
from pathlib import Path

import pytest
import yaml
from linkml.validator import Validator
from linkml.validator.plugins import JsonschemaValidationPlugin

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schema" / "tvbo_datamodel.yaml"
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
def validator():
    # LinkML resolves imports relative to the schema's directory.
    os.chdir(REPO / "schema")
    return Validator(
        str(SCHEMA),
        validation_plugins=[JsonschemaValidationPlugin(closed=False)],
    )


@pytest.mark.parametrize(("path", "target_class"), CASES, ids=IDS)
def test_database_yaml_validates(validator, path, target_class):
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        pytest.skip(f"{path} is not a top-level mapping")
    result = validator.validate(data, target_class=target_class)
    messages = [issue.message for issue in result.results]
    assert not messages, (
        f"{path.relative_to(REPO)} failed validation as {target_class}:\n  - "
        + "\n  - ".join(messages)
    )
