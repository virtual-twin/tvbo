"""Ward hierarchical clustering of correlation profiles, reached through the curated record ``HierarchicalClustering`` as a study reaches it.

``HierarchicalClustering`` correlates the rows of a feature matrix over its features, builds Ward's tree on the rows of that correlation matrix and cuts it below a declared multiple of its highest merge, as Koller et al. (2024, doi:10.1038/s41467-024-47860-x) cluster the components of their NMF fits for Fig. 9f (released ``scripts/results/48_results.ipynb``: ``np.corrcoef`` in cell 40, ``dendrogram(linkage(rhom, method='ward'))`` in cell 47, the clusters as the leaves scipy colours ``C1`` and ``C2`` in cell 49). Each of those points is pinned here on a case worked out by hand: the correlation of the rows and Ward's heights on it (``FOUR``), the leaf order scipy's dendrogram draws, and the clusters its colour threshold gives, strictly below the cut, numbered from the left, 0 for a unit joined to none. Rows are taken in label order, so a tie (``TIED``) is broken by the labels and not by the order the rows come in.
"""

import itertools

import numpy as np
import pytest
import xarray as xr

from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

from .test_curated_statistics import _curated, _seed

U = np.array([1.0, -1.0, 1.0, -1.0])
V = np.array([1.0, 1.0, -1.0, -1.0])
FOUR = {"a": U, "b": 4 * U + 3 * V, "c": V, "d": 10.0 - U}
"""Four rows over four features whose correlations and Ward heights are worked out by hand.

``U`` and ``V`` are orthogonal, each sums to zero and both have the same norm, so two rows ``p U + q V``, plus any constant, correlate as the cosine of the angle between ``(p, q)`` and the other row's pair: r(a, b) = 4/5, r(a, c) = 0, r(a, d) = -1, r(b, c) = 3/5, r(b, d) = -4/5, r(c, d) = 0, the 10 added to d changing none of them. The rows of the correlation matrix are a (1, 4/5, 0, -1), b (4/5, 1, 3/5, -4/5), c (0, 3/5, 1, 0) and d (-1, -4/5, 0, 1), and their squared distances, in 25ths, are 12 (a, b), 76 (a, c), 264 (a, d), 40 (b, c), 252 (b, d) and 124 (c, d). Ward first joins a and b at sqrt(12/25) = 2 sqrt(3)/5. The Lance-Williams update gives the pair's squared distance to c as (2/3)(76/25) + (2/3)(40/25) - (1/3)(12/25) = 44/15 and to d as (2/3)(264/25) + (2/3)(252/25) - (1/3)(12/25) = 68/5, and 44/15 is below 124/25, the squared distance from c to d, so c joins next at sqrt(44/15) = 2 sqrt(165)/15. d joins last at sqrt((3/4)(68/5) + (2/4)(124/25) - (1/4)(44/15)) = sqrt(896/75) = 8 sqrt(42)/15, which is also sqrt(3/2) times the distance from d's row to the mean of the other three, (3/5, 4/5, 8/15, -3/5). scipy writes the lower id of a merge first and draws it on the left: a (0) and b (1) form 4, c (2) and 4 form 5, d (3) and 5 form 6, so from the left the leaves are d, c, a, b."""
FOUR_LINKAGE = [
    [0.0, 1.0, 2 * np.sqrt(3) / 5, 2.0],
    [2.0, 4.0, 2 * np.sqrt(165) / 15, 3.0],
    [3.0, 5.0, 8 * np.sqrt(42) / 15, 4.0],
]
FOUR_CORRELATION = [[1.0, 0.0, -1.0, -0.8], [0.0, 1.0, 0.0, 0.6], [-1.0, 0.0, 1.0, 0.8], [-0.8, 0.6, 0.8, 1.0]]
"""The correlation of ``FOUR`` in leaf order d, c, a, b."""

HADAMARD = np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]])
TIED = {"a": HADAMARD[0], "b": HADAMARD[1], "c": HADAMARD[2], "d": HADAMARD.sum(axis=0)}
"""Four rows whose first merge is a tie between three.

a, b and c are the three rows of a Hadamard matrix that sum to zero, orthogonal to each other, so they do not correlate with each other, and d = a + b + c = (3, -1, -1, -1) correlates 1/sqrt(3) with each. Among the rows of the correlation matrix a, b and c then lie sqrt(10/3 - 4/sqrt(3)), about 1.01, from d and sqrt(2) from each other, so Ward's first merge joins d to one of the three at the same height, and which one is decided by the order the rows are taken in."""


@pytest.fixture
def rng():
    """A seeded generator, the same in every test."""
    return np.random.default_rng(0)


def _rows(table, order=None):
    """The rows of *table*, a mapping of label to row, as a labelled ``(component, pair)`` matrix, in *order* or the mapping's own."""
    order = list(table) if order is None else list(order)
    return xr.DataArray(np.stack([table[k] for k in order]), dims=("component", "pair"), coords={"component": order})


def _cluster(features, cut=0.7):
    """``HierarchicalClustering`` of *features* along ``component``, cut at *cut*."""
    return _curated("HierarchicalClustering")(features, dim="component", cut=cut)


def test_two_separated_groups_give_two_clusters_each_kept_together_in_the_leaf_order(rng):
    """Three noisy copies of one random pattern and three of another, given interleaved, correlate about 0.99 within a group and about 0 across, so the cut at 0.7 of the highest merge leaves two clusters, one per group, each a run of the leaf order and numbered 1 for the run that comes first."""
    patterns = {"p": rng.standard_normal(200), "q": rng.standard_normal(200)}
    names = ["q1", "p0", "q2", "p2", "p1", "q0"]
    got = _cluster(_rows({name: patterns[name[0]] + 0.1 * rng.standard_normal(200) for name in names}, names))
    groups = [name[0] for name in got["leaf_order"]["component"].values.tolist()]
    assert groups[:3] == [groups[0]] * 3 and groups[3:] == [groups[3]] * 3 and groups[0] != groups[3]
    assert got["leaf_order"].values.tolist() == list(range(6))
    assert got["cluster"].values.tolist() == [1, 1, 1, 2, 2, 2]


def test_the_result_is_the_same_whatever_order_the_rows_come_in():
    """``TIED`` in each of its 24 row orders gives the same four outputs, to the bit and with the same labels, though its first merge joins d to one of a, b and c at one height: the rows are put in label order before they are correlated, so the labels break the tie."""
    first, *others = [_cluster(_rows(TIED, order)) for order in itertools.permutations(TIED)]
    for got in others:
        for key, value in first.items():
            xr.testing.assert_identical(got[key], value)


def test_ward_heights_of_four_rows_are_the_hand_computed_merges_of_their_correlation_rows():
    """``FOUR`` gives the correlations, the merges and their heights worked out in its docstring, the leaves in the order d, c, a, b, and that order on both axes of the correlation matrix."""
    got = _cluster(_rows(FOUR))
    assert got["leaf_order"]["component"].values.tolist() == ["d", "c", "a", "b"]
    assert got["correlation"].dims == ("component", "component_j")
    assert got["correlation"]["component"].values.tolist() == got["correlation"]["component_j"].values.tolist()
    assert got["correlation"]["component"].values.tolist() == ["d", "c", "a", "b"]
    np.testing.assert_allclose(got["correlation"].values, FOUR_CORRELATION, rtol=0, atol=1e-14)
    assert got["linkage"].dims == ("merge", "field")
    assert got["linkage"]["field"].values.tolist() == ["first", "second", "height", "size"]
    np.testing.assert_allclose(got["linkage"].values, FOUR_LINKAGE, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("cut", "clusters"),
    [(0.7, [0, 1, 1, 1]), (1.0, [0, 1, 1, 1]), (0.3, [0, 0, 1, 1]), (1.5, [1, 1, 1, 1])],
    ids=["default", "at_the_highest_merge", "below_the_second", "above_every_merge"],
)
def test_the_cut_keeps_the_merges_strictly_below_its_multiple_of_the_highest(cut, clusters):
    """In the leaf order d, c, a, b of ``FOUR`` the merges lie at 0.693, 1.713 and 3.456. At scipy's default 0.7 the cut is at 2.419: a, b and c are cluster 1, and d, joined to them only above it, is 0. At 1 the cut is the highest merge itself, which is not below it, so d stays 0. At 0.3 (1.037) only a and b join, and they are cluster 1 although two units joined to none come before them; at 1.5 (5.185) every merge is below and all four are one cluster."""
    assert _cluster(_rows(FOUR), cut=cut)["cluster"].values.tolist() == clusters


def test_rows_without_labels_are_named_by_their_position():
    """A matrix whose ``component`` axis carries no labels, as a concatenation of several fits along it may, has its rows named 0, 1, ... in the order given, the numbering Koller et al.'s notebook clusters by, and gives the result of the same rows so labelled."""
    values = np.stack(list(FOUR.values()))
    bare = _cluster(xr.DataArray(values, dims=("component", "pair")))
    labelled = _cluster(xr.DataArray(values, dims=("component", "pair"), coords={"component": [0, 1, 2, 3]}))
    assert bare["leaf_order"]["component"].values.tolist() == [3, 2, 0, 1]
    for key, value in labelled.items():
        xr.testing.assert_identical(bare[key], value)


@pytest.mark.parametrize("row", [np.full(4, 2.0), np.array([1.0, np.nan, 0.0, 2.0])], ids=["constant", "nan"])
def test_a_row_whose_correlation_is_undefined_is_refused_by_its_label(row):
    """A constant row and a row holding NaN correlate with nothing, and the refusal names the unit."""
    with pytest.raises(ValueError, match="`features` has no correlation for 'e' along 'component'"):
        _cluster(_rows({**FOUR, "e": row}))


def test_features_that_are_not_one_row_per_unit_against_one_feature_axis_are_refused(rng):
    """A third dimension, which would need a tree per slice and so a leaf order per slice, is refused, as are a matrix without ``component``, a bare matrix, a label given twice, fewer than two units and fewer than three features."""
    with pytest.raises(ValueError, match=r"two dimensions; its dimensions are \['component', 'pair', 'trial'\]"):
        _cluster(xr.DataArray(rng.random((4, 5, 2)), dims=("component", "pair", "trial")))
    with pytest.raises(ValueError, match=r"two dimensions; its dimensions are \['component'\]"):
        _cluster(rng.random(4))
    with pytest.raises(ValueError, match="does not run along 'component'"):
        _cluster(xr.DataArray(rng.random((4, 5)), dims=("unit", "pair")))
    with pytest.raises(ValueError, match="bare array of rank 2"):
        _cluster(rng.random((4, 5)))
    with pytest.raises(ValueError, match="carries 'a' more than once along 'component'"):
        _cluster(xr.DataArray(rng.random((3, 5)), dims=("component", "pair"), coords={"component": ["a", "b", "a"]}))
    for shape in ((1, 5), (4, 2)):
        with pytest.raises(ValueError, match="needs at least two units along 'component' and 3 features"):
            _cluster(xr.DataArray(rng.random(shape), dims=("component", "pair")))


@pytest.mark.parametrize("cut", [None, 0, -0.5, np.nan, np.inf, True, "0.7", [0.7]])
def test_a_cut_that_is_not_one_finite_number_above_0_is_refused(cut):
    """The cut is declared, so a missing one is refused with zero, a negative, NaN, infinity, a boolean, text and a list."""
    with pytest.raises(ValueError, match="`cut` is the height the tree is cut at as a multiple of its highest merge"):
        _cluster(_rows(FOUR), cut=cut)


def test_it_runs_as_a_study_analysis_and_stores_its_four_outputs_with_their_labels(tmp_path):
    """Named by ``iri`` and run on a seeded container laid out as an NMF fit's ``w``, pairs by components, it stores the four outputs with the component labels in leaf order, read back from the file."""
    features = _rows(FOUR).transpose("pair", "component")
    _seed(tmp_path, "factors", features)
    analysis = Analysis(
        name="component_clusters",
        iri="tvbo:functions/HierarchicalClustering",
        arguments={"features": {"used": {"analysis": "factors", "output": "factors"}}, "dim": "component", "cut": 0.7},
    )
    analysis_io.run_analysis(analysis, tmp_path)

    want = _cluster(features)
    with xr.open_dataset(analysis_io.container_path("component_clusters", tmp_path), engine="h5netcdf") as stored:
        assert set(stored.data_vars) == {f"observation__{key}" for key in want}
        assert stored["observation__correlation"].dims == ("component", "component_j")
        assert stored["observation__cluster"].dims == stored["observation__leaf_order"].dims == ("component",)
        assert stored["component"].values.tolist() == ["d", "c", "a", "b"]
        assert stored["field"].values.tolist() == ["first", "second", "height", "size"]
        for key, value in want.items():
            np.testing.assert_array_equal(stored[f"observation__{key}"].values, value.values)
