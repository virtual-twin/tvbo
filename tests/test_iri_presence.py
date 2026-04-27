"""Regression test: every YAML in `tvbo/database/models/` carries an `iri`.

The `iri` slot is the cross-surface stable identifier (LinkML ↔ OWL ↔ API ↔
Odoo) defined in main proposal §15.1. Currently only the `models/` subtree is
fully covered. This test locks in that coverage and prevents regressions while
the remaining subtrees (coupling_functions, integrators, networks, atlases,
experiments, observation_models) are backfilled (tracked in plan §11.2).
"""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "tvbo" / "database"

# Subdirectories where every YAML must carry a top-level `iri`.
# Add subdirectories here as their backfill PRs land.
REQUIRED_IRI_SUBDIRS = ["models"]


def _collect():
    cases = []
    for sub in REQUIRED_IRI_SUBDIRS:
        # Only top-level YAMLs in each subdir; per-backend subdirectories
        # (e.g. models/julia/, models/neuroml/) are imported subsets and
        # excluded from this gate (tracked separately in plan §11.2).
        for path in sorted((DB / sub).glob("*.y*ml")):
            cases.append(path)
    return cases


CASES = _collect()
IDS = [str(p.relative_to(REPO)) for p in CASES]


@pytest.mark.parametrize("path", CASES, ids=IDS)
def test_yaml_has_iri(path):
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict), f"{path} is not a top-level mapping"
    assert "iri" in data, (
        f"{path.relative_to(REPO)} is missing the `iri` slot. "
        "The `iri` slot is required on every entity in REQUIRED_IRI_SUBDIRS "
        "for cross-surface identity (see plan §11.2 and main proposal §15.1)."
    )
    iri = data["iri"]
    assert isinstance(iri, str) and iri.strip(), (
        f"{path.relative_to(REPO)} has an empty `iri` value."
    )
