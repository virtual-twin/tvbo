"""A run records what it did in its study — and when it has no study, beside its results.

`tvbo run tvbo/database/experiments/<X>.yaml -o <somewhere>` used to write `prov-exp<N>_*.yaml` into the installed database, because the record followed the spec's own directory whether or not that directory was a study. Four test modules then failed on files no one had added.
"""

from __future__ import annotations

import json
from pathlib import Path

from tvbo.cli.run import _provenance_root


def _study(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "s", "BIDSVersion": "1.8.0"}))
    return root


def test_a_study_keeps_its_own_prov_even_when_the_data_goes_elsewhere(tmp_path):
    """The documented rule: provenance describes what the STUDY did, so `-o` must not scatter it."""
    study = _study(tmp_path / "study")
    (study / "recipe.yaml").write_text("tvbo_class: tvbo:SimulationStudy\n")
    assert _provenance_root(str(study / "recipe.yaml"), tmp_path / "elsewhere") == study


def test_a_spec_below_the_root_still_records_in_the_study(tmp_path):
    """The root is the nearest ancestor declaring itself one, not the directory the file sits in."""
    study = _study(tmp_path / "study")
    nested = study / "recipes"
    nested.mkdir()
    (nested / "recipe.yaml").write_text("tvbo_class: tvbo:SimulationStudy\n")
    assert _provenance_root(str(nested / "recipe.yaml"), None) == study


def test_a_spec_in_no_study_records_beside_its_results(tmp_path):
    """A curated experiment run out of the database has no `prov/` of its own; writing one into the database mutates an installed package."""
    database = tmp_path / "database" / "experiments"
    database.mkdir(parents=True)
    (database / "Curated.yaml").write_text("id: 8\n")
    out = tmp_path / "run"
    assert _provenance_root(str(database / "Curated.yaml"), out) == out.resolve()


def test_with_no_output_directory_it_falls_back_to_the_spec(tmp_path):
    """Nothing to follow: the results land beside the spec too, so the record belongs there with them."""
    database = tmp_path / "database" / "experiments"
    database.mkdir(parents=True)
    (database / "Curated.yaml").write_text("id: 8\n")
    assert _provenance_root(str(database / "Curated.yaml"), None) == database.resolve()
