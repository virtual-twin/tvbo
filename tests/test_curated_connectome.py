"""Consistency-based thresholding of a cohort's connectomes, reached through its curated record as a study reaches it, against a hand-worked case and an independent transcription of a published implementation.

The reference is cell 12 of Koller et al.'s released `20_results.ipynb` (doi:10.1038/s41467-024-47860-x), written out here as the authors ran it on their 776 subjects: mean and coefficient of variation per region pair, the upper triangle's lowest-CV pairs kept to the requested density, mirrored.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo import Function
from tvbo.datamodel.schema import Analysis


def _threshold():
    """The callable ``tvbo:functions/ConsistencyThreshold`` records."""
    return Function.from_string("iri: tvbo:functions/ConsistencyThreshold").resolved_callable


def _stack(values, labels=None):
    """*values* (subject, node, node) as a cohort read stacks them."""
    k, n, _ = values.shape
    labels = labels or [f"r{i}" for i in range(n)]
    return xr.DataArray(
        values,
        dims=("subject", "node_i", "node_j"),
        coords={"subject": [f"s{i}" for i in range(k)], "node_i": labels, "node_j": labels},
    )


def _symmetric_cohort(rng, k=25, n=30, zero_share=0.4):
    """Lognormal symmetric connectomes with a zero diagonal and pairs absent in some subjects."""
    w = rng.lognormal(1.0, 1.0, (k, n, n)) * (rng.random((k, n, n)) > zero_share)
    w = np.triu(w, 1)
    return w + w.transpose(0, 2, 1)


def _koller_cell_12(ws, ls, p):
    """Koller et al.'s cell 12, with their variable names: *ws* and *ls* are lists of subject matrices."""
    number_of_regions = ws[0].shape[0]
    number_of_possible_connections = number_of_regions**2
    number_connections_preserved = int(np.round((number_of_possible_connections) * p / 2))
    Ws = np.array(ws).transpose(2, 1, 0)
    Wmean = np.mean(Ws, axis=2)
    Wcv = np.full_like(Wmean, np.inf)
    np.true_divide(np.std(Ws, axis=2, ddof=1), Wmean, out=Wcv, where=Wmean != 0)
    Wmean = np.triu(Wmean)
    ind = np.argwhere(Wmean)
    Wcv_sorted = np.sort(Wcv[ind[:, 0], ind[:, 1]])
    Wmean[Wcv > Wcv_sorted[:number_connections_preserved][-1]] = 0
    Ls = np.array(ls).transpose(2, 1, 0)
    lengths_mean = np.triu(np.mean(Ls, axis=2))
    lengths_mean[Wcv > Wcv_sorted[:number_connections_preserved][-1]] = 0
    return Wmean + Wmean.T, lengths_mean + lengths_mean.T


def test_the_record_names_the_function_and_an_analysis_states_only_its_arguments():
    analysis = Analysis(name="group_sc", iri="tvbo:functions/ConsistencyThreshold", arguments={"density": 0.15})
    assert (analysis.callable.module, analysis.callable.name) == ("tvbo.analysis.connectome", "consistency_threshold")
    assert analysis.arguments["density"].value == 0.15
    assert analysis.arguments["weight"].description and analysis.arguments["length"].description


def test_a_hand_worked_cohort_keeps_the_two_most_consistent_pairs():
    """Three subjects, four regions: pairs (0,1) and (2,3) vary least, (0,2) is absent everywhere, the rest vary more."""
    pairs = {
        (0, 1): [10, 10, 11],
        (2, 3): [4, 5, 4],
        (0, 3): [1, 9, 2],
        (1, 2): [0, 6, 3],
        (1, 3): [2, 2, 20],
        (0, 2): [0, 0, 0],
    }
    w = np.zeros((3, 4, 4))
    for (i, j), v in pairs.items():
        w[:, i, j] = w[:, j, i] = v
    out = _threshold()(_stack(w), density=0.25)
    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = np.mean(pairs[(0, 1)])
    expected[2, 3] = expected[3, 2] = np.mean(pairs[(2, 3)])
    np.testing.assert_allclose(out["weight"].values, expected)
    assert list(out) == ["weight"]
    assert list(out["weight"]["node_i"].values) == ["r0", "r1", "r2", "r3"]


def test_it_equals_the_published_implementation_on_a_random_cohort():
    rng = np.random.default_rng(3)
    w = _symmetric_cohort(rng)
    lengths = np.where(w > 0, rng.uniform(10, 150, w.shape), 0.0)
    lengths = np.triu(lengths, 1) + np.triu(lengths, 1).transpose(0, 2, 1)
    out = _threshold()(_stack(w), density=0.15, length=_stack(lengths))
    ref_w, ref_l = _koller_cell_12(list(w), list(lengths), 0.15)
    np.testing.assert_allclose(out["weight"].values, ref_w, rtol=1e-12, atol=0)
    np.testing.assert_allclose(out["length"].values, ref_l, rtol=1e-12, atol=0)
    assert np.count_nonzero(out["weight"].values) == round(30 * 30 * 0.15 / 2) * 2


def test_lengths_and_subjects_are_paired_by_label_not_by_position():
    rng = np.random.default_rng(5)
    w = _symmetric_cohort(rng, k=12, n=10)
    lengths = np.where(w > 0, rng.uniform(10, 150, w.shape), 0.0)
    lengths = np.triu(lengths, 1) + np.triu(lengths, 1).transpose(0, 2, 1)
    weight, length = _stack(w), _stack(lengths)
    shuffled = length.isel(subject=rng.permutation(12), node_i=rng.permutation(10), node_j=rng.permutation(10))
    a = _threshold()(weight, density=0.3, length=length)
    b = _threshold()(weight, density=0.3, length=shuffled)
    np.testing.assert_array_equal(a["length"].values, b["length"].values)


def test_a_pair_absent_from_every_subject_is_never_kept_even_when_the_density_asks_for_more():
    w = np.zeros((4, 3, 3))
    w[:, 0, 1] = w[:, 1, 0] = [1, 2, 3, 4]
    out = _threshold()(_stack(w), density=1.0)
    assert np.count_nonzero(out["weight"].values) == 2


def test_a_density_too_small_for_one_pair_keeps_none():
    w = _stack(_symmetric_cohort(np.random.default_rng(1), k=5, n=6))
    assert not _threshold()(w, density=0.01)["weight"].values.any()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda w: w.isel(subject=0), "`subject` axis"),
        (lambda w: w.isel(subject=[0]), "at least two subjects"),
    ],
)
def test_a_weight_without_a_cohort_to_vary_over_is_refused(make, message):
    w = _stack(_symmetric_cohort(np.random.default_rng(0), k=3, n=5))
    with pytest.raises(ValueError, match=message):
        _threshold()(make(w), density=0.2)


@pytest.mark.parametrize("density", [0.0, -0.1, 1.5])
def test_a_density_outside_zero_to_one_is_refused(density):
    w = _stack(_symmetric_cohort(np.random.default_rng(0), k=3, n=5))
    with pytest.raises(ValueError, match="density"):
        _threshold()(w, density=density)
