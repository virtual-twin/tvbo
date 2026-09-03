"""The frame a run names its recorded inputs in: its study, and when it has no study, its results.

A `used:` edge is only an edge if both ends spell the container the same way, so the frame has to be the study rather than the directory the spec happens to sit in. `tvbo run tvbo/database/experiments/<X>.yaml -o <somewhere>` once resolved it to the installed database, which made every recorded input a path no reader of the results could follow.
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
    """The documented rule: a recorded input is named relative to the STUDY, so `-o` must not move the frame."""
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
    """A curated experiment run out of the database is in no study, so the results it wrote are the only frame its inputs can be named in."""
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
