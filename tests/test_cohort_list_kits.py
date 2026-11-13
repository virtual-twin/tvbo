"""A study that keeps its subject lists out of its recipe, run and frozen into a kit.

The dataset names a file for its subjects, and the cohort a network reads per subject names a file for its members or admits them by a rule. The fan-out is the one the typed lists give, and a kit carries each list as a file beside its frozen spec, which states no list. Every identifier is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")

from tests import test_cohort_references as cohort_case  # noqa: E402
from tests.test_cohort_references import (  # noqa: E402
    LISTED,
    MEMBERS,
    OUTSIDER,
    _analyses,
    _dataset,
    _expected,
    _frozen_kit,
    _study_file,
)
from tvbo import SimulationExperiment, SimulationStudy  # noqa: E402
from tvbo.data import analysis_io  # noqa: E402
from tvbo.utils.study_layout import study_path  # noqa: E402

root = cohort_case.root  # five subjects' connectomes on disk

BY_FILE = {"members_file": "sourcedata/cohorts/four.txt"}
BY_RULE = {"rule": {"measure": "score", "grouping": "equal_count", "n_groups": 2, "group": 1}}


def _experiment(root, tmp_path, cohort):
    """The relaxation experiment of a study at *tmp_path* whose dataset lists its subjects in a table and whose cohort ``four`` is declared by *cohort*."""
    (tmp_path / "dataset_description.json").write_text('{"Name": "ToyCohort", "BIDSVersion": "1.9.0"}')
    lists = tmp_path / "sourcedata" / "cohorts"
    lists.mkdir(parents=True)
    (lists / "four.txt").write_text("".join(f"sub-{m}\n" for m in MEMBERS))
    (lists / "listed.tsv").write_text("participant_id\n" + "".join(f"sub-{s}\n" for s in LISTED))
    everyone = [*MEMBERS, OUTSIDER, "07"]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    (lists / "scores.tsv").write_text(
        "participant_id\tscore\n" + "".join(f"sub-{s}\t{v}\n" for s, v in zip([*everyone, "08", "09"], scores, strict=True))
    )
    path = _study_file(root, tmp_path, dataset={"dataset_id": "toy_fc", "subjects_file": "sourcedata/cohorts/listed.tsv"})
    spec = yaml.safe_load(path.read_text())
    spec["datasets"][0]["cohorts"] = [{"cohort_id": "four", **cohort}]
    spec["datasets"][0]["phenotype_file"] = "sourcedata/cohorts/scores.tsv"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return SimulationStudy.from_file(str(path)).get_experiment(1)


@pytest.mark.parametrize("cohort", [BY_FILE, BY_RULE], ids=["file", "rule"])
def test_the_fan_out_reads_the_lists_from_the_studys_files(root, tmp_path, cohort):
    """The table lists four subjects, two of them among the four members, as the typed lists of the neighbouring module do."""
    experiment = _experiment(root, tmp_path, cohort)
    assert experiment.subject_selection() == {
        "dataset": 4,
        "cohorts": {"four": 4},
        "subjects": ["04", "01"],
        "excluded": [OUTSIDER, "07"],
    }


def test_the_fan_out_follows_the_files(root, tmp_path):
    experiment = _experiment(root, tmp_path, BY_FILE)
    assert experiment.dataset_subject_ids() == ["04", "01"]
    (tmp_path / "sourcedata/cohorts/four.txt").write_text("sub-01\nsub-02\n")
    assert experiment.dataset_subject_ids() == ["01"]
    (tmp_path / "sourcedata/cohorts/listed.tsv").write_text("participant_id\nsub-02\nsub-01\n")
    assert experiment.dataset_subject_ids() == ["02", "01"]


@pytest.mark.parametrize("cohort", [BY_FILE, BY_RULE], ids=["file", "rule"])
def test_a_kit_carries_each_list_as_a_file_and_its_spec_states_none(root, tmp_path, cohort, caplog):
    analysis_io.run_analyses(_analyses(), study_path("results", root=tmp_path), datasets=[_dataset(root)])
    experiment = _experiment(root, tmp_path, cohort)
    declared = yaml.safe_dump(yaml.safe_load((tmp_path / "ToyCohort.yaml").read_text())["datasets"])

    out = tmp_path / "kit"
    with caplog.at_level("INFO", logger="tvbo"):
        bundle, frozen, path = _frozen_kit(experiment, out)
    assert bundle == "../data/toy_fc"
    assert sorted(p.name for p in (out / "data/toy_fc").iterdir()) == ["sub-01", "sub-04"]
    assert frozen["dataset"].get("subjects") in (None, {}, [])
    assert frozen["dataset"]["subjects_file"] == "dataset-toy_fc_subjects.yaml"
    (carried,) = frozen["dataset"]["cohorts"]
    assert carried["cohort_id"] == "four" and carried["members_file"] == "cohort-four_members.txt"
    assert "members" not in carried and "rule" not in carried
    assert "the 2 of its 4 members the experiment reads" in carried["description"]
    stated = "group 1 of the 2 equal_count groups of 'score' in scores.tsv" if "rule" in cohort else "listed in four.txt"
    assert f"Admitted in the study as {stated}." in carried["description"]
    assert yaml.safe_load((path.parent / "dataset-toy_fc_subjects.yaml").read_text()) == [
        {"subject_id": "04"},
        {"subject_id": "01"},
    ]
    assert (path.parent / "cohort-four_members.txt").read_text() == "04\n01\n"
    assert "wrote cohort-four_members.txt beside the frozen spec (2 subjects)" in caplog.text
    assert "sub-0" not in caplog.text and "'04'" not in caplog.text

    assert experiment.dataset.subjects_file == "sourcedata/cohorts/listed.tsv", (
        "the study's own record is as declared after the freeze"
    )
    assert yaml.safe_dump(yaml.safe_load((tmp_path / "ToyCohort.yaml").read_text())["datasets"]) == declared

    rerun = SimulationExperiment.from_file(str(path))
    assert rerun.dataset_subject_ids() == ["04", "01"]
    rerun._active_subject = "01"
    rerun._bind_network_references(results_root=study_path("results", root=tmp_path))
    np.testing.assert_allclose(rerun.network.matrix("weight", format="dense"), _expected("01")[0], rtol=1e-6)


def test_typed_lists_are_frozen_typed_as_before(root, tmp_path):
    """A list the recipe states stays in the frozen spec, and no list file is written."""
    from tests.test_cohort_references import _fan_out_experiment

    (tmp_path / "dataset_description.json").write_text('{"Name": "ToyCohort", "BIDSVersion": "1.9.0"}')
    analysis_io.run_analyses(_analyses(), study_path("results", root=tmp_path), datasets=[_dataset(root)])
    _, frozen, path = _frozen_kit(_fan_out_experiment(root, tmp_path), tmp_path / "kit")
    assert list(frozen["dataset"]["subjects"]) == ["04", "01"] and "subjects_file" not in frozen["dataset"]
    assert [(c["cohort_id"], c["members"]) for c in frozen["dataset"]["cohorts"]] == [("four", ["04", "01"])]
    assert sorted(p.name for p in path.parent.iterdir()) == ["experiment.yaml", "network.h5", "network.yaml"]


def test_a_plan_names_a_subject_file_that_is_missing(tmp_path):
    """An experiment that reads no cohort is planned without a subject axis when its subjects cannot be discovered; a subject file that is not there is named instead."""
    from tvbo.run.workflow import _dataset_subject_axis

    assert _dataset_subject_axis(SimulationExperiment(dataset={"dataset_id": "toy"})) is None
    experiment = SimulationExperiment(dataset={"dataset_id": "toy", "subjects_file": str(tmp_path / "gone.txt")})
    assert not experiment.referenced_cohorts()
    with pytest.raises(FileNotFoundError, match=r"dataset 'toy'.*gone\.txt"):
        _dataset_subject_axis(experiment)


def test_a_cohort_analysis_digest_follows_a_list_in_a_file(root, tmp_path):
    """The staleness digest of an analysis over a cohort is taken from the members the file gives, so a grown list marks its aggregate for a re-run."""
    from tvbo.data.study_manifest import _analysis_fingerprint
    from tvbo.datamodel.schema import Dataset

    listing = tmp_path / "four.txt"
    listing.write_text("".join(f"{m}\n" for m in MEMBERS[:3]))
    dataset = Dataset(dataset_id="toy_sc", bids_root=str(root), cohorts=[{"cohort_id": "four", "members_file": "four.txt"}])
    analysis = _analyses()[0]

    def digest():
        return _analysis_fingerprint(analysis, analysis_io.used_cohorts(analysis, [dataset], tmp_path), {})

    before = digest()
    assert digest() == before
    listing.write_text("".join(f"{m}\n" for m in MEMBERS))
    assert digest() != before
    path = analysis_io.run_analysis(analysis, tmp_path / "results", datasets=[dataset], source_dir=tmp_path)
    record = yaml.safe_load(path.with_suffix(".yaml").read_text())
    assert record["cohorts"] == {"four": {"dataset": "toy_sc", "n_members": 4}}


def test_a_kit_that_bundles_no_data_still_carries_the_subject_list(tmp_path):
    """A frozen dataset that names a subject file names one beside the frozen spec, so the frozen experiment lists its subjects from the kit alone."""
    from tvbo.cli import workflow as kit

    study = tmp_path / "study"
    (study / "sourcedata" / "cohorts").mkdir(parents=True)
    (study / "dataset_description.json").write_text('{"Name": "Toy", "BIDSVersion": "1.9.0"}')
    (study / "sourcedata/cohorts/listed.tsv").write_text("participant_id\n" + "".join(f"sub-{s}\n" for s in LISTED))
    path = study / "experiment.yaml"
    dataset = {"dataset_id": "toy", "subjects_file": "sourcedata/cohorts/listed.tsv"}
    path.write_text(yaml.safe_dump({"id": 1, "label": "listed", "dataset": dataset}, sort_keys=False))
    experiment = SimulationExperiment.from_file(str(path))
    spec_dir = tmp_path / "kit" / "spec"
    spec_dir.mkdir(parents=True)
    (spec_dir / "experiment.yaml").write_text(kit._freeze_spec_yaml(experiment, spec_dir))
    assert (
        yaml.safe_load((spec_dir / "experiment.yaml").read_text())["dataset"]["subjects_file"] == "dataset-toy_subjects.yaml"
    )
    assert SimulationExperiment.from_file(str(spec_dir / "experiment.yaml"))._dataset_subjects() == LISTED
    assert experiment.dataset.subjects_file == "sourcedata/cohorts/listed.tsv"


def test_two_lists_a_kit_would_write_to_one_file_are_refused(tmp_path):
    from tvbo.cli import workflow as kit

    taken = {}
    kit._write_listing(tmp_path, "cohort", "a b", ["01"], taken)
    with pytest.raises(ValueError, match=r"cohorts 'a b' and 'a_b' would both be written to cohort-a_b_members\.txt"):
        kit._write_listing(tmp_path, "cohort", "a_b", ["02"], taken)
