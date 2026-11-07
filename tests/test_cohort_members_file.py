"""A list of subjects kept in a file: a cohort's ``members_file`` and a dataset's ``subjects_file``.

A typed list is a copy of the list its producer keeps: it goes stale while a pipeline still finishes subjects, and it puts participant identifiers into the recipe. Naming the file keeps one source, read each time the list is used. Every identifier here is synthetic.
"""

from __future__ import annotations

import pytest
import yaml

from tvbo.data import cohort
from tvbo.datamodel import Dataset
from tvbo.utils import to_dict, yaml_loader

IDS = ["s03", "s01", "s07"]


def _members(source_dir, **declared):
    """The members of a cohort declared by a recipe in *source_dir*."""
    dataset = Dataset(dataset_id="toy_sc", bids_root="unused", cohorts=[{"cohort_id": "finished", **declared}])
    return cohort.cohort_members(*cohort.find_cohort([dataset], "finished"), source_dir)


def test_typed_members_are_returned_in_declared_order(tmp_path):
    assert _members(tmp_path, members=["s03", "s01"]) == ["s03", "s01"]


def test_a_typed_cohort_dumps_its_keys_in_the_order_it_had_while_members_was_required():
    """``members`` stays ahead of ``label`` and ``description`` in a dumped cohort, so a recipe that types its lists dumps to the bytes it dumped to before the file and rule slots."""
    from tvbo.utils import to_dict

    typed = {"cohort_id": "finished", "label": "Finished", "description": "Every finished subject.", "members": ["s03", "s01"]}
    dataset = Dataset(dataset_id="toy_sc", cohorts=[typed])
    dumped = to_dict(dataset)["cohorts"]["finished"]
    assert list(dumped) == ["cohort_id", "members", "label", "description"]


def test_members_are_read_from_a_plain_list_beside_the_recipe(tmp_path):
    (tmp_path / "subjects_done.txt").write_text("# finished so far\nsub-s03\ns01\n\n")
    assert _members(tmp_path, members_file="subjects_done.txt") == ["s03", "s01"]


def test_members_are_read_from_a_participants_table(tmp_path):
    (tmp_path / "participants.tsv").write_text("verified\tparticipant_id\ntrue\tsub-s03\ntrue\tsub-s01\n")
    assert _members(tmp_path, members_file="participants.tsv") == ["s03", "s01"]


def test_a_comma_separated_table_and_one_with_a_byte_order_mark_are_read_by_participant(tmp_path):
    (tmp_path / "participants.csv").write_text("verified,participant_id\ntrue,sub-s03\ntrue,sub-s01\n")
    assert _members(tmp_path, members_file="participants.csv") == ["s03", "s01"]
    (tmp_path / "marked.tsv").write_text("\ufeffparticipant_id\tverified\nsub-s03\ttrue\n", encoding="utf-8")
    assert _members(tmp_path, members_file="marked.tsv") == ["s03"]


def test_a_table_row_without_a_participant_is_refused_by_its_row(tmp_path):
    (tmp_path / "gap.tsv").write_text("participant_id\tverified\nsub-s03\ttrue\n\ttrue\n")
    with pytest.raises(ValueError, match=r"gap\.tsv, row 3 names no participant"):
        _members(tmp_path, members_file="gap.tsv")


def test_a_list_named_in_an_included_dataset_is_read_from_the_including_recipe(tmp_path):
    """A dataset brought into an experiment by ``!include`` from another directory names its list relative to the experiment's file, not to the fragment's."""
    from tvbo import SimulationExperiment

    (tmp_path / "fragments").mkdir()
    (tmp_path / "fragments" / "toy_dataset.yaml").write_text("dataset_id: toy\nsubjects_file: listed.txt\n")
    (tmp_path / "experiment.yaml").write_text("id: 1\nlabel: included\ndataset: !include fragments/toy_dataset.yaml\n")
    (tmp_path / "fragments" / "listed.txt").write_text("s07\n")
    with pytest.raises(FileNotFoundError, match=r"'listed\.txt'"):
        SimulationExperiment.from_file(str(tmp_path / "experiment.yaml"))._dataset_subjects()
    (tmp_path / "listed.txt").write_text("s03\ns01\n")
    assert SimulationExperiment.from_file(str(tmp_path / "experiment.yaml"))._dataset_subjects() == ["s03", "s01"]


def test_a_table_without_a_participant_column_is_refused_with_its_columns(tmp_path):
    (tmp_path / "participants.tsv").write_text("subject\tverified\ns03\ttrue\n")
    with pytest.raises(ValueError, match=r"participants\.tsv.*participant_id.*verified") as raised:
        _members(tmp_path, members_file="participants.tsv")
    assert "s03" not in str(raised.value)


def test_a_list_that_changes_changes_the_members(tmp_path):
    """The file is read at each use: a longer list and a rewritten one of the same length are both seen."""
    listing = tmp_path / "subjects_done.txt"
    listing.write_text("s03\n")
    assert _members(tmp_path, members_file="subjects_done.txt") == ["s03"]
    listing.write_text("s03\ns01\n")
    assert _members(tmp_path, members_file="subjects_done.txt") == ["s03", "s01"]
    listing.write_text("s03\ns07\n")
    assert _members(tmp_path, members_file="subjects_done.txt") == ["s03", "s07"]


@pytest.mark.parametrize(
    "extra", [{"members": ["s03"]}, {"rule": {"measure": "score", "grouping": "equal_count", "n_groups": 2, "group": 1}}]
)
def test_a_file_beside_another_way_of_stating_the_members_is_refused(tmp_path, extra):
    (tmp_path / "subjects_done.txt").write_text("s03\n")
    with pytest.raises(ValueError, match=r"members_file.*one way|one way.*members_file"):
        _members(tmp_path, members_file="subjects_done.txt", **extra)


def test_a_cohort_that_states_no_members_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"declares no `members`"):
        _members(tmp_path)


def test_a_missing_file_is_named_with_the_places_tried(tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}")
    spec = tmp_path / "spec"
    spec.mkdir()
    with pytest.raises(FileNotFoundError, match=r"subjects_done\.txt") as raised:
        _members(spec, members_file="lists/subjects_done.txt")
    assert str(spec / "lists" / "subjects_done.txt") in str(raised.value)
    assert str(tmp_path / "lists" / "subjects_done.txt") in str(raised.value)


def test_a_relative_path_is_read_beside_the_recipe_and_else_from_the_study_root(tmp_path):
    """A fragment under a study's spec directory names a list by its place in the study; a copy beside the recipe is the one read."""
    (tmp_path / "dataset_description.json").write_text("{}")
    spec, lists = tmp_path / "spec", tmp_path / "sourcedata" / "cohorts"
    spec.mkdir()
    lists.mkdir(parents=True)
    (lists / "done.txt").write_text("s03\ns01\n")
    assert _members(spec, members_file="sourcedata/cohorts/done.txt") == ["s03", "s01"]
    assert _members(tmp_path, members_file="sourcedata/cohorts/done.txt") == ["s03", "s01"]
    beside = spec / "sourcedata" / "cohorts"
    beside.mkdir(parents=True)
    (beside / "done.txt").write_text("s07\n")
    assert _members(spec, members_file="sourcedata/cohorts/done.txt") == ["s07"]


def test_an_absolute_path_is_taken_as_written(tmp_path):
    (tmp_path / "done.txt").write_text("s03\n")
    assert _members(tmp_path / "elsewhere", members_file=str(tmp_path / "done.txt")) == ["s03"]


def test_a_path_without_a_recipe_is_read_from_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "done.txt").write_text("s03\n")
    monkeypatch.chdir(tmp_path)
    assert _members(None, members_file="done.txt") == ["s03"]


def _fragments(tmp_path, listed):
    """A study with one list file under ``sourcedata/`` and two dataset fragments under ``spec/``: one includes the file into the slot, one names it."""
    (tmp_path / "dataset_description.json").write_text("{}")
    (tmp_path / "spec").mkdir()
    (tmp_path / "sourcedata").mkdir()
    (tmp_path / "sourcedata" / "listed.yaml").write_text("# written by the pipeline\n" + yaml.safe_dump(listed))
    return tmp_path / "spec"


def test_a_member_file_loads_what_an_include_of_it_loads(tmp_path):
    spec = _fragments(tmp_path, ["s03", "s01", 7])
    (spec / "included.yaml").write_text(
        "dataset_id: toy\ncohorts:\n  - cohort_id: done\n    members:\n      !include ../sourcedata/listed.yaml\n"
    )
    (spec / "named.yaml").write_text(
        "dataset_id: toy\ncohorts:\n  - cohort_id: done\n    members_file: sourcedata/listed.yaml\n"
    )
    included, named = (yaml_loader.load(str(spec / name), Dataset) for name in ("included.yaml", "named.yaml"))
    got, want = (cohort.cohort_members(*cohort.find_cohort([dataset], "done"), spec) for dataset in (named, included))
    assert got == want == ["s03", "s01", "7"]
    assert "s03" not in yaml.safe_dump(to_dict(named))


def test_a_subject_file_loads_the_records_an_include_of_it_loads(tmp_path):
    """The file holds what the slot holds, so a record's other fields arrive with it."""
    spec = _fragments(tmp_path, [{"subject_id": "s03", "group": "patient"}, {"subject_id": "s01"}])
    (spec / "included.yaml").write_text("dataset_id: toy\nsubjects:\n  !include ../sourcedata/listed.yaml\n")
    (spec / "named.yaml").write_text("dataset_id: toy\nsubjects_file: sourcedata/listed.yaml\n")
    included, named = (yaml_loader.load(str(spec / name), Dataset) for name in ("included.yaml", "named.yaml"))
    got, want = cohort.dataset_subjects(named, spec), cohort.dataset_subjects(included, spec)
    assert list(got) == list(want) == ["s03", "s01"]
    assert {k: to_dict(v) for k, v in got.items()} == {k: to_dict(v) for k, v in want.items()}
    assert got["s03"].group == "patient"
    assert "s03" not in yaml.safe_dump(to_dict(named))


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("subjects.txt", "sub-s03\ns01\n"),
        ("participants.tsv", "participant_id\tage\nsub-s03\t31\nsub-s01\t27\n"),
        ("subjects.yaml", "s03: {}\ns01: {}\n"),
        ("subjects.json", '["s03", "s01"]'),
    ],
)
def test_a_dataset_reads_its_subjects_from_a_file_of_any_listed_kind(tmp_path, name, text):
    (tmp_path / name).write_text(text)
    assert list(cohort.dataset_subjects(Dataset(dataset_id="toy", subjects_file=name), tmp_path)) == ["s03", "s01"]


def test_a_dataset_lists_its_subjects_in_one_of_the_two_slots(tmp_path):
    (tmp_path / "subjects.txt").write_text("s03\n")
    both = Dataset(dataset_id="toy", subjects=["s01"], subjects_file="subjects.txt")
    with pytest.raises(ValueError, match=r"`subjects` and `subjects_file`"):
        cohort.dataset_subjects(both, tmp_path)
    assert cohort.dataset_subjects(Dataset(dataset_id="toy"), tmp_path) == {}


def test_a_file_that_lists_nobody_is_refused(tmp_path):
    (tmp_path / "empty.txt").write_text("# nothing yet\n\n")
    with pytest.raises(ValueError, match=r"lists no member"):
        _members(tmp_path, members_file="empty.txt")
    with pytest.raises(ValueError, match=r"lists no subject"):
        cohort.dataset_subjects(Dataset(dataset_id="toy", subjects_file="empty.txt"), tmp_path)


def test_a_message_about_a_list_from_a_file_gives_counts_and_no_identifier(tmp_path):
    """A repeated entry and a member outside the dataset are counted; the same faults in a typed list are named, since the recipe already states them."""
    (tmp_path / "twice.txt").write_text("s03\ns01\ns03\n")
    with pytest.raises(ValueError, match=r"twice\.txt lists 1 identifier\(s\) more than once") as raised:
        _members(tmp_path, members_file="twice.txt")
    assert "s03" not in str(raised.value)

    (tmp_path / "subjects.txt").write_text("s03\ns01\n")
    (tmp_path / "members.txt").write_text("s03\ns07\ns09\n")

    def strangers(**members):
        dataset = Dataset(dataset_id="toy", subjects_file="subjects.txt", cohorts=[{"cohort_id": "done", **members}])
        with pytest.raises(ValueError, match=r"names 2 member\(s\) the dataset 'toy' does not list") as raised:
            cohort.cohort_members(*cohort.find_cohort([dataset], "done"), tmp_path)
        return str(raised.value)

    assert "s07" not in strangers(members_file="members.txt") and "s09" not in strangers(members_file="members.txt")
    assert "s07" in strangers(members=["s03", "s07", "s09"])


def test_one_list_serves_two_datasets(tmp_path):
    """Two datasets hold the same subjects' files under different roots; each declares the cohort by naming the one list, and both follow it."""
    listing = tmp_path / "done.txt"
    listing.write_text("s03\ns01\n")
    structural = Dataset(dataset_id="sc", cohorts=[{"cohort_id": "done_sc", "members_file": "done.txt"}])
    functional = Dataset(dataset_id="fc", cohorts=[{"cohort_id": "done_fc", "members_file": "done.txt"}])
    datasets = [structural, functional]

    def members():
        return [cohort.cohort_members(*cohort.find_cohort(datasets, c), tmp_path) for c in ("done_sc", "done_fc")]

    assert members() == [["s03", "s01"], ["s03", "s01"]]
    listing.write_text("s03\ns01\ns07\n")
    assert members() == [IDS[:2] + ["s07"]] * 2
