"""Validate the Pydantic loader (``tvbo.utils.pydantic_loader``).

This is the *strict* validation path (``extra="forbid"``) used by the TVBO platform's experiment builder to guarantee every assembled experiment is a valid TVBO object before download/save. It complements ``test_database_validation.py`` (lenient LinkML JSON-schema, ``closed=False``).

The loader normalizes TVBO's human-friendly keyed-dict YAML (where a dict key is the member's ``name``) into the shape the Pydantic models expect, then validates. We assert that:

* every ground-truth file in the experiment-building categories validates,
* a full experiment round-trips (load -> dump -> load),
* genuinely invalid input is rejected (it is a real validator, not a coercer),
* keyed-dict key-injection and file-envelope stripping behave as designed.

A small number of fringe classes whose generated Pydantic models lag the LinkML schema are tracked as ``xfail`` (see ``todo.md`` in the platform repo).
"""

from pathlib import Path

import pytest

from tvbo.utils import pydantic_loader as pl

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "tvbo" / "database"

# Registry subdirectory -> target Pydantic class. These are the categories the experiment builder assembles from; all are expected to validate strictly.
CORE_TARGETS = {
    "models": "Dynamics",
    "networks": "Network",
    "coupling_functions": "Coupling",
    "observation_models": "Observation",
    "integrators": "Integrator",
    "experiments": "SimulationExperiment",
    "atlases": "BrainAtlas",
    "continuations": "Continuation",
}

# The NeuroML import staging area (database/models/neuroml/) holds raw auto-converted NeuroML files that are not yet mapped onto the tvbo schema (e.g. null coupling_inputs, a NeuroML-only `components` slot). They are not curated building blocks and are not used to assemble experiments, so they are out of scope for strict schema validation. See the platform todo.md.
EXCLUDE_DIRS = ("/neuroml/",)

EXPERIMENTS_DIR = DB / "experiments"


def _collect():
    cases = []
    for sub, cls in CORE_TARGETS.items():
        for path in sorted((DB / sub).rglob("*.y*ml")):
            if any(part in str(path) for part in EXCLUDE_DIRS):
                continue
            cases.append((path, cls))
    return cases


CASES = _collect()
IDS = [str(p.relative_to(DB)) for p, _ in CASES]


@pytest.mark.parametrize(("path", "target_class"), CASES, ids=IDS)
def test_core_database_validates_strictly(path, target_class):
    pl.load(path, target_class)  # raises pydantic.ValidationError on failure


def test_all_canonical_experiments_validate():
    files = sorted(p for p in EXPERIMENTS_DIR.glob("*.y*ml"))
    assert files, "no canonical experiment YAMLs found"
    for f in files:
        exp = pl.load(f, "SimulationExperiment")
        assert exp.id is not None


def test_experiment_round_trips():
    f = EXPERIMENTS_DIR / "RWW_BOLD_FC_Optimization.yaml"
    exp = pl.load(f, "SimulationExperiment")
    dumped = pl.dump(exp)
    reloaded = pl.loads(dumped, "SimulationExperiment")
    assert reloaded.id == exp.id
    assert set((reloaded.observations or {})) == set((exp.observations or {}))
    assert set((reloaded.functions or {})) == set((exp.functions or {}))


def test_keyed_dict_keys_are_injected_as_name():
    yaml_text = """
id: 1
label: KeyInjection
dynamics:
  name: Demo
  parameters:
    a: {value: 0.27}
    b: {value: 0.108}
"""
    exp = pl.loads(yaml_text, "SimulationExperiment")
    params = exp.dynamics.parameters
    assert params["a"].name == "a"
    assert params["b"].name == "b"


def test_list_form_collections_are_coerced_to_keyed_dicts():
    # Odoo many2many export and JS builder collectors emit lists; the loader coerces them into the schema's keyed-dict form using each member's name.
    yaml_text = """
id: 5
label: ListForm
dynamics:
  name: Demo
  parameters:
    - {name: a, value: 0.27}
    - {name: b, value: 0.108}
observations:
  - {name: bold}
  - {name: fc}
"""
    exp = pl.loads(yaml_text, "SimulationExperiment")
    assert set(exp.dynamics.parameters) == {"a", "b"}
    assert exp.dynamics.parameters["a"].value == 0.27
    assert set(exp.observations) == {"bold", "fc"}


def test_scalar_list_text_blob_is_split():
    # Odoo stores list[str] slots (e.g. references) as a newline/bulleted Text blob; the loader splits it back into a list.
    yaml_text = """
id: 9
label: Refs
references: |-
  - Wong & Wang (2006) J Neurosci 26:1314
  - Deco et al. (2014) J Neurosci 34:7886
"""
    exp = pl.loads(yaml_text, "SimulationExperiment")
    assert exp.references == [
        "Wong & Wang (2006) J Neurosci 26:1314",
        "Deco et al. (2014) J Neurosci 34:7886",
    ]


def test_file_envelope_keys_are_stripped():
    # tvbo_class / schema_version annotate the serialized class; not datamodel slots.
    yaml_text = """
tvbo_class: tvbo:SimulationExperiment
schema_version: "0.4.0"
id: 7
label: Envelope
"""
    exp = pl.loads(yaml_text, "SimulationExperiment")
    assert exp.id == 7


@pytest.mark.parametrize(
    "payload",
    [
        {"label": "no id"},  # missing required id
        {"id": 1, "totally_bogus_key": 123},  # extra forbidden
        {"id": 1, "dynamics": ["not", "a", "model"]},  # wrong nested type
    ],
)
def test_invalid_input_is_rejected(payload):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        pl.validate(payload, "SimulationExperiment")
