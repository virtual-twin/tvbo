"""`tvbo run` writes analyses where its figures read them.

A study run is one command that produces results AND the figures that bind them. The two stages resolve the container layout independently, and when they disagreed the failure was
silent: with the documented `-o output/nc`, analyses landed in `output/nc/results/` while figures looked in `output/results/`, so a run rendered this run's experiments against a
PREVIOUS run's analyses. These pin the two to one mapping.
"""

from pathlib import Path

import pytest

from tvbo.cli.run import _container_root, _spec_base


@pytest.fixture
def study(tmp_path):
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: Study\n", encoding="utf-8")
    (tmp_path / "output" / "nc").mkdir(parents=True)
    return spec


def _figure_root(spec, out_dir):
    """Where the figure stage resolves containers — `<container root's parent>/output`."""
    return _container_root(str(spec), out_dir).parent / "output"


@pytest.mark.parametrize("out", [None, "output", "output/nc"])
def test_every_documented_out_dir_maps_to_one_container_root(study, out):
    out_dir = None if out is None else study.parent / out
    assert _container_root(str(study), out_dir) == study.parent / "output"


@pytest.mark.parametrize("out", [None, "output", "output/nc"])
def test_the_analysis_writer_and_the_figure_reader_agree(study, out):
    out_dir = None if out is None else study.parent / out
    assert _container_root(str(study), out_dir) == _figure_root(study, out_dir)


@pytest.mark.parametrize("has_output_subdir", [True, False])
def test_a_relocated_run_keeps_the_two_stages_together(study, tmp_path, has_output_subdir):
    """Never split them, whatever the out-dir looks like."""
    elsewhere = tmp_path / "run17"
    (elsewhere / "output").mkdir(parents=True) if has_output_subdir else elsewhere.mkdir()
    assert _container_root(str(study), elsewhere) == elsewhere / "output"
    assert _figure_root(study, elsewhere) == elsewhere / "output"


def test_the_container_root_is_always_named_output(study, tmp_path):
    """Figures resolve `<base>/output/results/<name>`, so the root must be that `output`."""
    for od in (None, study.parent / "output", study.parent / "output/nc", tmp_path / "x"):
        assert _container_root(str(study), od).name == "output"


def test_spec_base_is_the_study_directory(study):
    assert _spec_base(str(study)) == study.parent


def test_a_spec_that_is_not_a_file_falls_back_to_the_cwd():
    assert _spec_base("tvbo:SomeCurie") == Path.cwd()
