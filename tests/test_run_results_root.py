"""`tvbo run` writes results where its figures read them.

A study run is one command that produces results AND the figures that bind them. The two stages once resolved the container layout independently, and when they disagreed the failure was silent: with the documented `-o output/nc`, analyses landed in `output/nc/results/` while figures looked in `output/results/`, so a run rendered this run's experiments against a PREVIOUS run's analyses. Both now ask the layout record for one directory, and these pin that.
"""

from pathlib import Path

import pytest

from tvbo.cli.run import _results_root, _spec_base
from tvbo.utils.study_layout import study_path


@pytest.fixture
def study(tmp_path):
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: Study\n", encoding="utf-8")
    return spec


def test_a_run_persists_into_the_studys_own_results_directory(study):
    """No `-o` is not "discard": a run always writes where the layout says results go."""
    assert _results_root(str(study), None) == study_path("results", root=study.parent)


def test_out_dir_overrides_the_location_and_nothing_else(study, tmp_path):
    """`-o` relocates the results directory verbatim, with no normalisation of the path given."""
    elsewhere = tmp_path / "run17"
    assert _results_root(str(study), elsewhere) == elsewhere.resolve()
    assert _results_root(str(study), tmp_path / "a" / "b") == (tmp_path / "a" / "b").resolve()


def test_the_writer_and_the_figure_reader_agree(study):
    """The figure stage resolves layers against the study root, whose results directory this is."""
    assert _results_root(str(study), None) == study_path("results", root=_spec_base(str(study)))


def test_resolving_an_already_resolved_root_is_idempotent(study):
    """Callers pass the resolved root down, so a second resolution must not move it."""
    once = _results_root(str(study), None)
    assert _results_root(str(study), once) == once


def test_spec_base_is_the_study_directory(study):
    assert _spec_base(str(study)) == study.parent


def test_a_spec_that_is_not_a_file_falls_back_to_the_cwd():
    assert _spec_base("tvbo:SomeCurie") == Path.cwd()
