# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""An ``!include``d fragment keeps meaning what it said in the file it was written in."""

import json

import yaml

from tvbo.cli.figures import figure_origins
from tvbo.utils.yaml_loader import IncludedMapping, include_origin, load_as_dict


def _study(tmp_path):
    """A minimal study dataset holding a figure record as a spec fragment."""
    root = tmp_path / "MyStudy"
    (root / "spec").mkdir(parents=True)
    (root / "code").mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "MyStudy", "DatasetType": "study"}))
    (root / "spec" / "fig-one_figure.yaml").write_text("name: fig_one\nlabel: A figure\ncode_modules: [my_panels]\n")
    return root


def test_an_included_mapping_is_a_dict_that_remembers_its_file(tmp_path):
    root = _study(tmp_path)
    (tmp_path / "study.yaml").write_text("figures:\n  - !include MyStudy/spec/fig-one_figure.yaml\n")
    figure = load_as_dict(str(tmp_path / "study.yaml"))["figures"][0]
    assert figure == {"name": "fig_one", "label": "A figure", "code_modules": ["my_panels"]}
    assert include_origin(figure) == root / "spec"
    assert include_origin({"name": "written inline"}) is None


def test_an_included_mapping_still_dumps_as_a_mapping():
    """The load path re-serialises the parsed document for LinkML, and a dict subclass has no representer of its own."""
    assert yaml.safe_dump(IncludedMapping({"a": 1}, "/tmp")) == yaml.safe_dump({"a": 1})


def test_a_figure_included_from_a_study_renders_against_that_study(tmp_path):
    """Not against whoever included it: the fragment's `code_modules` and paths are relative to the study it lives in."""
    root = _study(tmp_path)
    manuscript = tmp_path / "manuscript.yaml"
    manuscript.write_text("figures:\n  - !include MyStudy/spec/fig-one_figure.yaml\n")
    assert figure_origins(manuscript) == {"fig_one": root}


def test_a_figure_included_from_outside_any_study_claims_no_origin(tmp_path):
    """A record kept beside the manuscript that includes it has no study of its own, so it renders against the includer."""
    spec_dir = tmp_path / "figures" / "spec"
    spec_dir.mkdir(parents=True)
    (spec_dir / "fig-two_figure.yaml").write_text("name: fig_two\nlabel: A figure\n")
    manuscript = tmp_path / "manuscript.yaml"
    manuscript.write_text("figures:\n  - !include figures/spec/fig-two_figure.yaml\n")
    assert figure_origins(manuscript) == {}


def test_a_figure_written_inline_claims_no_origin(tmp_path):
    """So it keeps resolving against its own spec, which is what an inline record already says."""
    spec = tmp_path / "study.yaml"
    spec.write_text("figures:\n  - name: fig_inline\n    label: A figure\n")
    assert figure_origins(spec) == {}
