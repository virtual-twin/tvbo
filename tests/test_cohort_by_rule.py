"""A cohort admitted by a rule on a phenotype measure instead of by a list of members.

The rule names a measure of the dataset's phenotype table, how the candidates are cut into groups along it (``equal_count``: groups of equal size by rank; ``equal_width``: bins of equal width over the range) and which group the cohort holds. The members are computed from the table when the cohort is used. Expected groups are computed here in numpy from the same scores, and every identifier is synthetic.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from tvbo.analysis.statistics import GROUPINGS
from tvbo.data.cohort import cohort_members, find_cohort
from tvbo.datamodel import CohortGrouping, Dataset

SUBJECTS = [f"s{i:02d}" for i in range(12)]
SCORE = np.array([5.0, 1.0, 9.0, 2.0, 30.0, 3.0, 8.0, 4.0, 7.0, 6.0, 10.0, 60.0])


def _table(tmp_path, scores=SCORE, subjects=SUBJECTS, name="pheno"):
    """A ``Phenotype`` sidecar with one measure and its companion; returns the sidecar's name."""
    with h5py.File(tmp_path / f"{name}.h5", "w") as f:
        f.create_dataset("measures/score", data=np.asarray(scores, dtype=float))
    record = {
        "tvbo_class": "tvbo:Phenotype",
        "dataset_id": "toy",
        "subjects": list(subjects),
        "measures": ["score"],
        "data_file": f"{name}.h5",
    }
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(record))
    return f"{name}.yaml"


def _rule(group, n_groups=3, grouping="equal_count", **extra):
    return {"measure": "score", "grouping": grouping, "n_groups": n_groups, "group": group, **extra}


def _split(tmp_path, cohorts, **dataset):
    """Each declared cohort's members, for a dataset that lists the twelve subjects and names the table, unless *dataset* says otherwise."""
    dataset = {"subjects": SUBJECTS, "phenotype_file": "pheno.yaml", **dataset}
    ds = Dataset(dataset_id="toy", cohorts=cohorts, **{k: v for k, v in dataset.items() if v is not None})
    return {c["cohort_id"]: cohort_members(*find_cohort([ds], c["cohort_id"]), tmp_path) for c in cohorts}


def _one(tmp_path, rule, **dataset):
    return _split(tmp_path, [{"cohort_id": "g", "rule": rule}], **dataset)["g"]


def _by_rank(scores, subjects, n_groups):
    return [[subjects[i] for i in sorted(run)] for run in np.array_split(np.argsort(scores, kind="stable"), n_groups)]


def test_the_schema_offers_the_groupings_the_statistics_implement():
    """One vocabulary: a grouping a rule may name is one ``group_index`` computes."""
    assert sorted(CohortGrouping.__members__) == sorted(GROUPINGS)
    with pytest.raises(ValueError, match="by_rank"):
        CohortGrouping("by_rank")


def test_equal_count_gives_groups_of_equal_size_in_the_candidates_order(tmp_path):
    """Three rank groups of twelve hold four each, every subject in one group, each group listed in the dataset's order and not by score."""
    _table(tmp_path)
    got = _split(tmp_path, [{"cohort_id": f"g{k}", "rule": _rule(k)} for k in (1, 2, 3)])
    assert [got[f"g{k}"] for k in (1, 2, 3)] == _by_rank(SCORE, SUBJECTS, 3)
    assert [len(m) for m in got.values()] == [4, 4, 4]
    assert sorted(sum(got.values(), [])) == SUBJECTS
    assert got["g3"] == ["s02", "s04", "s10", "s11"]


def test_equal_count_gives_the_longer_groups_first_when_the_size_does_not_divide(tmp_path):
    _table(tmp_path)
    got = _split(tmp_path, [{"cohort_id": f"g{k}", "rule": _rule(k, n_groups=5)} for k in range(1, 6)])
    assert [len(m) for m in got.values()] == [3, 3, 2, 2, 2]


def test_equal_width_gives_bins_of_equal_width_over_the_range(tmp_path):
    """Three bins over 1 to 60 are cut at 20.67 and 40.33: ten subjects, one and one."""
    _table(tmp_path)
    got = _split(tmp_path, [{"cohort_id": f"w{k}", "rule": _rule(k, grouping="equal_width")} for k in (1, 2, 3)])
    edges = np.linspace(SCORE.min(), SCORE.max(), 4)
    index = np.clip(np.digitize(SCORE, edges[1:-1]), 0, 2)
    assert [got[f"w{k + 1}"] for k in range(3)] == [
        [s for s, i in zip(SUBJECTS, index, strict=True) if i == k] for k in range(3)
    ]
    assert [len(m) for m in got.values()] == [10, 1, 1]


def test_a_bin_is_closed_below_and_the_last_one_on_both_sides(tmp_path):
    """Over 0 to 4 in two bins the cut is 2: a value of 2 belongs to the upper bin, and the largest value to the last."""
    subjects, scores = ["a", "b", "c", "d"], [0.0, 2.0, 3.0, 4.0]
    _table(tmp_path, scores, subjects)
    got = _split(tmp_path, [{"cohort_id": f"w{k}", "rule": _rule(k, 2, "equal_width")} for k in (1, 2)], subjects=subjects)
    assert got == {"w1": ["a"], "w2": ["b", "c", "d"]}


def test_equal_values_stay_in_one_bin_and_are_cut_in_the_candidates_order_by_rank(tmp_path):
    """Four equal scores between a lower and a higher one: one bin holds all four, while two rank groups of three cut them after the second in the dataset's order."""
    subjects, scores = ["a", "b", "c", "d", "e", "f"], [5.0, 5.0, 1.0, 5.0, 9.0, 5.0]
    _table(tmp_path, scores, subjects)
    width = _split(
        tmp_path, [{"cohort_id": f"w{k}", "rule": _rule(k, 3, "equal_width")} for k in (1, 2, 3)], subjects=subjects
    )
    assert width == {"w1": ["c"], "w2": ["a", "b", "d", "f"], "w3": ["e"]}
    rank = _split(tmp_path, [{"cohort_id": f"g{k}", "rule": _rule(k, 2)} for k in (1, 2)], subjects=subjects)
    assert rank == {"g1": ["a", "b", "c"], "g2": ["d", "e", "f"]}


def test_a_rule_inside_another_cohort_splits_that_cohorts_members_only(tmp_path):
    """Ranks and the range are taken over the parent's members: the parent's own largest score closes the last bin."""
    _table(tmp_path)
    parent = SUBJECTS[:6]
    cohorts = [
        {"cohort_id": "first", "members": parent},
        {"cohort_id": "first_low", "rule": _rule(1, 2, of="first")},
        {"cohort_id": "first_wide", "rule": _rule(2, 2, "equal_width", of="first")},
    ]
    got = _split(tmp_path, cohorts)
    assert got["first_low"] == _by_rank(SCORE[:6], parent, 2)[0] == ["s01", "s03", "s05"]
    assert got["first_wide"] == ["s04"]


def test_the_parent_may_be_read_from_a_file_or_admitted_by_a_rule(tmp_path):
    _table(tmp_path)
    (tmp_path / "half.txt").write_text("".join(f"{s}\n" for s in SUBJECTS[6:]))
    cohorts = [
        {"cohort_id": "listed", "members_file": "half.txt"},
        {"cohort_id": "listed_high", "rule": _rule(2, 2, of="listed")},
        {"cohort_id": "top_of_high", "rule": _rule(3, 3, of="listed_high")},
    ]
    got = _split(tmp_path, cohorts)
    assert got["listed_high"] == _by_rank(SCORE[6:], SUBJECTS[6:], 2)[1] == ["s06", "s10", "s11"]
    assert got["top_of_high"] == ["s11"]


def test_a_rule_that_leads_back_to_itself_and_a_parent_nobody_declares_are_refused(tmp_path):
    _table(tmp_path)
    loop = [{"cohort_id": "a", "rule": _rule(1, 2, of="b")}, {"cohort_id": "b", "rule": _rule(1, 2, of="a")}]
    with pytest.raises(ValueError, match=r"leads back to it \(a -> b -> a\)"):
        _split(tmp_path, loop)
    with pytest.raises(
        LookupError, match=r"splits the cohort 'nobody', which the dataset 'toy' does not declare \(declared: \['g'\]\)"
    ):
        _one(tmp_path, _rule(1, 2, of="nobody"))


def test_the_candidates_are_the_datasets_subjects_and_else_the_tables(tmp_path):
    """A dataset that lists four of the twelve splits those four; one that lists none splits everyone the table holds; a listed file counts as listed."""
    _table(tmp_path)
    four = ["s04", "s00", "s11", "s01"]
    assert _one(tmp_path, _rule(2, 2), subjects=four) == ["s04", "s11"]
    assert _one(tmp_path, _rule(3), subjects=None) == _by_rank(SCORE, SUBJECTS, 3)[2]
    (tmp_path / "four.txt").write_text("".join(f"{s}\n" for s in four))
    assert _one(tmp_path, _rule(2, 2), subjects=None, subjects_file="four.txt") == ["s04", "s11"]


def test_a_candidate_without_a_finite_value_is_counted_and_refused(tmp_path):
    """One NaN, one infinite score and one listed subject the table does not hold: three candidates without a value, named by count alone."""
    scores = SCORE.copy()
    scores[[2, 4]] = [np.nan, np.inf]
    _table(tmp_path, scores)
    with pytest.raises(
        ValueError, match=r"3 of its 13 candidates have no finite value of 'score' in pheno\.yaml.*missing: exclude"
    ) as raised:
        _one(tmp_path, _rule(1), subjects=[*SUBJECTS, "s99"])
    assert not any(s in str(raised.value) for s in ("s02", "s04", "s99"))


def test_excluded_candidates_belong_to_no_group_and_do_not_move_the_ranks_or_the_range(tmp_path):
    """Without the two largest scores the range ends at 10, so two bins are cut at 5.5, and two rank groups hold five each."""
    scores = SCORE.copy()
    scores[[4, 11]] = np.nan
    _table(tmp_path, scores)
    kept = [s for s in SUBJECTS if s not in ("s04", "s11")]
    low, high = (_one(tmp_path, _rule(k, 2, "equal_width", missing="exclude")) for k in (1, 2))
    assert low == [s for s in kept if SCORE[SUBJECTS.index(s)] < 5.5] and high == [
        s for s in kept if SCORE[SUBJECTS.index(s)] >= 5.5
    ]
    ranks = [_one(tmp_path, _rule(k, 2, missing="exclude")) for k in (1, 2)]
    assert [len(m) for m in ranks] == [5, 5] and sorted(sum(ranks, [])) == kept
    with pytest.raises(ValueError, match=r"none of its 2 candidates has a finite value"):
        _one(tmp_path, _rule(1, 2, missing="exclude"), subjects=["s04", "s11"])


def test_a_group_nobody_falls_in_is_refused_with_counts(tmp_path):
    """Six bins over 1 to 60 leave the fourth empty, and five rank groups of three candidates leave the last two."""
    _table(tmp_path)
    with pytest.raises(ValueError, match=r"group 4 of the 6 equal_width groups of 'score' holds none of the 12 candidates"):
        _one(tmp_path, _rule(4, 6, "equal_width"))
    with pytest.raises(ValueError, match=r"group 5 of the 5 equal_count groups of 'score' holds none of the 3 candidates"):
        _one(tmp_path, _rule(5, 5), subjects=SUBJECTS[:3])


def test_equal_scores_have_no_range_to_cut(tmp_path):
    _table(tmp_path, np.full(12, 7.0))
    with pytest.raises(
        ValueError, match=r"12 values of 'score' are all equal, so their range has no width to cut into 3 bins"
    ):
        _one(tmp_path, _rule(1, 3, "equal_width"))
    assert _one(tmp_path, _rule(1, 1, "equal_width")) == SUBJECTS
    assert [len(_one(tmp_path, _rule(k, 3))) for k in (1, 2, 3)] == [4, 4, 4]


@pytest.mark.parametrize(
    ("group", "refusal"),
    [(0, r"rule\.group\n  Input should be greater than or equal to 1"), (4, "asks for group 4 of 3; groups are counted from 1")],
    ids=["below", "above"],
)
def test_a_group_outside_the_split_is_refused(tmp_path, group, refusal):
    """Group 0 is below the schema's minimum, so the record is refused as it is read; a group past the split is refused when the members are drawn."""
    _table(tmp_path)
    with pytest.raises(ValueError, match=refusal):
        _one(tmp_path, _rule(group))


def test_a_rule_beside_typed_members_is_refused(tmp_path):
    _table(tmp_path)
    with pytest.raises(ValueError, match=r"`members` and `rule`.*one way"):
        _split(tmp_path, [{"cohort_id": "both", "members": SUBJECTS[:2], "rule": _rule(1, 2)}])


def test_the_table_is_the_datasets_unless_the_rule_names_another(tmp_path):
    """The rule's own table wins, a table nobody names is asked for, a missing one is named with the places tried and an unknown measure with those on offer."""
    _table(tmp_path)
    _table(tmp_path, SCORE[::-1], name="reversed")
    assert _one(tmp_path, _rule(3, phenotype_file="reversed.yaml")) == _by_rank(SCORE[::-1], SUBJECTS, 3)[2]
    with pytest.raises(ValueError, match=r"neither the rule nor the dataset 'toy' names a `phenotype_file`"):
        _one(tmp_path, _rule(1), phenotype_file=None)
    with pytest.raises(
        FileNotFoundError, match=r"cohort 'g': `phenotype_file` names the file 'absent\.yaml', which is not at .*absent\.yaml"
    ):
        _one(tmp_path, _rule(1), phenotype_file="absent.yaml")
    with pytest.raises(ValueError, match=r"cohort 'g': pheno\.yaml declares no measure 'speed' \(measures: \['score'\]\)"):
        _one(tmp_path, {**_rule(1), "measure": "speed"})


def test_the_members_follow_the_table(tmp_path):
    """The table is read at each use: new values in the companion move the groups without a change to the sidecar."""
    _table(tmp_path)
    assert _one(tmp_path, _rule(3)) == ["s02", "s04", "s10", "s11"]
    with h5py.File(tmp_path / "pheno.h5", "w") as f:
        f.create_dataset("measures/score", data=SCORE[::-1].copy())
    assert _one(tmp_path, _rule(3)) == _by_rank(SCORE[::-1], SUBJECTS, 3)[2] == ["s00", "s01", "s07", "s09"]


def test_a_tabular_phenotype_file_is_read_by_participant_and_column(tmp_path):
    """A BIDS table: ``n/a`` is a missing value, an entry that is no number is refused by row without repeating it, and so is a participant listed twice."""
    rows = "".join(f"sub-{s}\t{'n/a' if s == 's04' else v}\t1\n" for s, v in zip(SUBJECTS, SCORE, strict=True))
    (tmp_path / "pheno.tsv").write_text("participant_id\tscore\tother\n" + rows)
    kept = ["s00", "s01", "s03", "s05", "s07", "s09"]
    assert _one(tmp_path, _rule(1, 2, missing="exclude"), phenotype_file="pheno.tsv") == kept
    with pytest.raises(ValueError, match=r"1 of its 12 candidates have no finite value"):
        _one(tmp_path, _rule(1, 2), phenotype_file="pheno.tsv")
    with pytest.raises(ValueError, match=r"pheno\.tsv has no column 'speed' \(columns: \['score', 'other'\]\)"):
        _one(tmp_path, {**_rule(1), "measure": "speed"}, phenotype_file="pheno.tsv")
    (tmp_path / "bad.tsv").write_text("participant_id\tscore\nsub-s00\tfast\n")
    with pytest.raises(ValueError, match=r"bad\.tsv, row 2: column 'score' holds a value that is not a number") as raised:
        _one(tmp_path, _rule(1, 1), phenotype_file="bad.tsv")
    assert "fast" not in str(raised.value) and "s00" not in str(raised.value)
    (tmp_path / "twice.tsv").write_text("participant_id\tscore\nsub-s00\t1\nsub-s00\t2\n")
    with pytest.raises(ValueError, match=r"twice\.tsv lists 1 subject\(s\) more than once") as raised:
        _one(tmp_path, _rule(1, 1), phenotype_file="twice.tsv")
    assert "s00" not in str(raised.value)


def test_a_comma_separated_phenotype_table_is_read_and_another_kind_of_file_is_refused_by_name(tmp_path):
    rows = "".join(f"sub-{s},{v}\n" for s, v in zip(SUBJECTS, SCORE, strict=True))
    (tmp_path / "pheno.csv").write_text("participant_id,score\n" + rows)
    assert _one(tmp_path, _rule(3), phenotype_file="pheno.csv") == ["s02", "s04", "s10", "s11"]
    (tmp_path / "pheno.xlsx").write_bytes(b"not a table")
    with pytest.raises(ValueError, match=r"pheno\.xlsx.*\.xlsx.*\.tsv.*\.csv"):
        _one(tmp_path, _rule(3), phenotype_file="pheno.xlsx")


def test_a_phenotype_row_without_a_participant_is_refused_by_its_row(tmp_path):
    rows = "".join(f"sub-{s}\t{v}\n" for s, v in zip(SUBJECTS, SCORE, strict=True))
    (tmp_path / "gap.tsv").write_text("participant_id\tscore\n" + rows + "\t5.0\n")
    with pytest.raises(ValueError, match=r"gap\.tsv, row 14 names no participant"):
        _one(tmp_path, _rule(1, 1), phenotype_file="gap.tsv", subjects=None)


def test_a_rule_leaves_no_identifier_in_the_record(tmp_path):
    """The dataset as a recipe or a report states it holds the rule and the paths, and nobody's identifier."""
    from tvbo.utils import to_dict

    ds = Dataset(
        dataset_id="toy",
        subjects_file="subjects.txt",
        phenotype_file="pheno.yaml",
        cohorts=[{"cohort_id": "g", "rule": _rule(1)}],
    )
    text = yaml.safe_dump(to_dict(ds))
    assert "equal_count" in text and not any(s in text for s in SUBJECTS)
