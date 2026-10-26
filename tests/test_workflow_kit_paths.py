"""A kit lives in the study's build root, and its shards are told apart by name.

Two things were resolved relative to the current working directory rather than to the study: a kit's default output (`output/<study>/<engine>`) and an array task's slice, which was disambiguated by nesting under `$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID`. The first put build artifacts wherever the CLI happened to run; the second made a shard identifiable only by where it sat, so the gather had to walk a tree and a shard moved out of it became anonymous. Both now come from the layout record and from the filename's own `split-` entity.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tvbo.adapters.bids import build_result_path
from tvbo.cli.workflow import _kits_root
from tvbo.utils.study_layout import study_path


@pytest.fixture
def study(tmp_path):
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: Study\n", encoding="utf-8")
    return spec


def _experiment(split=None, subject=None):
    return SimpleNamespace(
        id=3,
        dynamics=SimpleNamespace(name="JansenRit"),
        _active_subject=subject,
        _active_split=split,
    )


def test_a_kit_is_packaged_into_the_studys_own_build_root(study):
    assert _kits_root(str(study)) == study_path("kits", root=study.parent)


def test_the_build_root_is_hidden_and_untracked(study):
    """A kit reproduces from the spec, so it is a build artifact rather than a published one."""
    from tvbo.utils.study_layout import is_tracked, load_layout, relpath

    record = load_layout()
    assert relpath("kits", record).startswith(".tvbo/")
    assert not is_tracked(relpath("kits", record), record, ())


def test_a_spec_that_is_not_a_file_falls_back_to_the_working_directory(tmp_path, monkeypatch):
    """A CURIE or database name has no directory of its own; `Path(spec).parent` would silently be the cwd anyway."""
    monkeypatch.chdir(tmp_path)
    assert _kits_root("study:Deco2014") == study_path("kits", root=Path.cwd())


def test_a_shard_carries_its_index_in_its_name():
    assert build_result_path(_experiment(split=7)) == "exp-3_model-JansenRit_split-0007_result.h5"


def test_an_unsharded_run_has_no_split_entity():
    """`split-` marks a partial result, so a whole run must not claim to be one."""
    assert "split" not in build_result_path(_experiment())


def test_two_shards_of_one_experiment_do_not_collide():
    """The defect the per-job directory nesting was hiding: identically named slices."""
    assert build_result_path(_experiment(split=0)) != build_result_path(_experiment(split=1))


def test_a_shard_index_is_padded_so_the_names_sort():
    """The gather concatenates by parameter value, but a human reading the directory should see them in order."""
    names = [build_result_path(_experiment(split=i)) for i in (2, 10)]
    assert sorted(names) == names


def test_a_per_subject_shard_keeps_both_entities():
    """A cohort fan-out that also shards its sweep must stay unique in both directions."""
    name = build_result_path(_experiment(split=12, subject="01"))
    assert name.startswith("sub-01_exp-3_") and "split-0012" in name
