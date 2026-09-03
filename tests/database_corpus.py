"""Which schema class each directory of ``tvbo/database/`` holds.

Shared by the two tests that walk the whole database — schema validation and the golden dump — so a new directory cannot be covered by one and silently invisible to the other.
That is not hypothetical: ``coordinate_spaces`` and ``reducers`` were absent from the validation map, and neither their missing ``description`` slot nor their missing class was caught until something tried to load them.

Deliberately stdlib-only. ``test_database_validation`` must not import ``tvbo`` — LinkML's enums are mutated to an unhashable form once it does, so that test validates through the shipped JSON Schema instead, and importing this module must not change that.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "tvbo" / "database"

#: Database subdirectory -> the LinkML class every file in it is an instance of.
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
    "coordinate_spaces": "CommonCoordinateSpace",
    "reducers": "Reducer",
    "themes": "Theme",
}


def collect() -> list[tuple[Path, str]]:
    """Every database YAML with the class it should load as, in a stable order."""
    return [(path, cls) for sub, cls in TARGETS.items() for path in sorted((DB / sub).rglob("*.y*ml"))]


def uncovered() -> list[str]:
    """Database subdirectories holding YAML that no entry of `TARGETS` claims.

    A directory of authored metadata that nothing knows the class of is validated by nothing and frozen by nothing — the state ``reducers`` was in.
    """
    return sorted(
        directory.name
        for directory in DB.iterdir()
        if directory.is_dir() and directory.name not in TARGETS and any(directory.rglob("*.y*ml"))
    )
