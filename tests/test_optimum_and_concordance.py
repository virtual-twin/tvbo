"""Lin's concordance correlation coefficient and the value at the optimum of a score, reached through their curated records as a study reaches them.

``ConcordanceCorrelation`` is ``2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))**2)`` with population moments over the units of one named dimension, every other dimension kept and labelled inputs paired by label. It is 1 for identical values, ``1 - c**2 / (2 var(x) + c**2)`` for ``y = x + c``, -1 for ``y = -x`` about a zero mean, NaN where a unit is NaN, and equal to the line Koller et al. compute it with (doi:10.1038/s41467-024-47860-x; released ``scripts/analysis/40_analysis_ef_ccc.py`` l. 43, transcribed in :func:`_koller_concordance_correlation`).

``AtOptimum`` reads a value array at the cell where a score over named dimensions is largest or smallest and returns that value, the score and each dimension's coordinate there. NaN scores are ignored, a score that is NaN everywhere is refused by name, and a tie goes to the cell with the smallest coordinates compared in the sorted order of the dimensions' names, so transposing the dimensions or reversing the labels along them leaves the answer where it is, while a positional argmax moves.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

from .test_curated_statistics import _curated, _seed

K_DIM, V_DIM = "KuramotoCoupling.a", "network.conduction_speed"
K = np.logspace(-8, -2, 4)
V = np.array([1.0, 2.0, 5.0])
NODES = ["r0", "r1", "r2", "r3", "r4"]
SCORES = np.array([[0.1, 0.4, 0.2], [0.3, 0.9, 0.0], [0.5, -0.2, 0.6], [0.7, 0.8, 0.35]])
"""A fit score on K x v whose maximum, 0.9, lies at (K[1], V[1]) and whose minimum, -0.2, at (K[2], V[1])."""


@pytest.fixture
def rng():
    """A seeded generator, the same in every test."""
    return np.random.default_rng(0)


def _koller_concordance_correlation(x, y):
    """``concordance_correlation`` of Koller et al.'s released ``40_analysis_ef_ccc.py``, l. 41-43, as written there."""
    return (2 * np.corrcoef(x, y)[0][1] * np.std(x) * np.std(y)) / (np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)


def _grid(values):
    """*values* on the K x v grid, labelled as a sweep's container labels them."""
    return xr.DataArray(values, dims=(K_DIM, V_DIM), coords={K_DIM: K, V_DIM: V}, name="fit")


def _per_node(rng):
    """One random map of the regions per grid cell, on K x v x node."""
    return xr.DataArray(
        rng.normal(9.0, 0.5, (K.size, V.size, len(NODES))),
        dims=(K_DIM, V_DIM, "node"),
        coords={K_DIM: K, V_DIM: V, "node": NODES},
    )


# ------------------------------------------------------------------ the concordance correlation


def test_the_concordance_of_an_array_with_itself_is_one(rng):
    """Identical values agree perfectly."""
    x = rng.normal(9.0, 0.5, 200)
    assert abs(float(_curated("ConcordanceCorrelation")(x, x.copy(), dim="node")) - 1.0) < 1e-12


@pytest.mark.parametrize("shift", [0.05, 1.0, -3.0])
def test_a_shift_lowers_the_concordance_by_the_analytic_amount(shift, rng):
    """For y = x + c the covariance is var(x) and the denominator 2 var(x) + c**2, so the coefficient is 1 - c**2 / (2 var(x) + c**2), below 1 where Pearson's r is 1."""
    x = rng.normal(9.0, 0.5, 200)
    got = float(_curated("ConcordanceCorrelation")(x, x + shift, dim="node"))
    assert abs(got - (1 - shift**2 / (2 * np.var(x) + shift**2))) < 1e-12
    assert got < 1.0


def test_the_concordance_of_a_centred_array_with_its_negative_is_minus_one(rng):
    """Values reflected about their common mean of zero disagree perfectly."""
    x = rng.normal(0.0, 1.0, 200)
    x -= x.mean()
    assert abs(float(_curated("ConcordanceCorrelation")(x, -x, dim="node")) + 1.0) < 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_the_concordance_equals_kollers_line_on_random_maps(seed):
    """An empirical and a partly related simulated map of 1,000 regions give Koller et al.'s coefficient to 1e-12, called as he calls it, the empirical map first."""
    rng = np.random.default_rng(seed)
    emp = rng.normal(9.3, 0.3, 1000)
    sim = 0.4 * emp + rng.normal(5.6, 0.4, 1000)
    got = float(_curated("ConcordanceCorrelation")(emp, sim, dim="node"))
    assert abs(got - _koller_concordance_correlation(emp, sim)) < 1e-12


def test_the_concordance_is_one_per_grid_cell_with_regions_paired_by_label(rng):
    """A simulated map per K x v cell against one empirical map, its regions in another order, gives one coefficient per cell, Koller et al.'s on that cell's map."""
    sim = _per_node(rng)
    emp = xr.DataArray(rng.normal(9.3, 0.3, len(NODES)), dims=["node"], coords={"node": NODES})
    got = _curated("ConcordanceCorrelation")(sim, emp.isel(node=[3, 0, 4, 1, 2]), dim="node")
    assert got.dims == (K_DIM, V_DIM)
    np.testing.assert_array_equal(got[K_DIM].values, K)
    for i, j in np.ndindex(K.size, V.size):
        assert abs(float(got[i, j]) - _koller_concordance_correlation(emp.values, sim.values[i, j])) < 1e-12


def test_a_nan_region_makes_the_concordance_nan_as_in_kollers_line(rng):
    """A region without a value is not dropped: the coefficient is NaN, as numpy's moments make Koller et al.'s."""
    x, y = rng.random(20), rng.random(20)
    y[7] = np.nan
    assert np.isnan(float(_curated("ConcordanceCorrelation")(x, y, dim="node")))
    assert np.isnan(_koller_concordance_correlation(x, y))


def test_a_region_only_one_map_carries_is_refused(rng):
    """Two maps over different regions are refused, naming the regions, rather than compared over the shared ones."""
    x = xr.DataArray(rng.random(4), dims=["node"], coords={"node": ["a", "b", "c", "d"]})
    y = xr.DataArray(rng.random(4), dims=["node"], coords={"node": ["a", "b", "c", "e"]})
    with pytest.raises(
        ValueError, match="`x` and `y` do not carry the same labels along 'node': 'd', 'e' are in only one of them"
    ):
        _curated("ConcordanceCorrelation")(x, y, dim="node")


# ---------------------------------------------------------------------- the value at an optimum


@pytest.mark.parametrize(("direction", "cell"), [("max", (1, 1)), ("min", (2, 1))])
def test_the_optimum_of_a_two_dimension_score_carries_the_value_the_score_and_both_coordinates(direction, cell, rng):
    """The largest or the smallest cell of a K x v score is found, and the value per region there comes back with the score and the cell's K and v."""
    value = _per_node(rng)
    got = _curated("AtOptimum")(_grid(SCORES), value, direction)
    i, j = cell
    assert float(got["score"]) == SCORES[i, j]
    assert (got[K_DIM], got[V_DIM]) == (K[i], V[j])
    assert got["value"].dims == ("node",) and got["value"]["node"].values.tolist() == NODES
    np.testing.assert_array_equal(got["value"].values, value.values[i, j])


def test_a_nan_score_is_passed_over(rng):
    """NaN cells hold no score, so the optimum is that of the other cells, where a positional argmax stops at the first NaN."""
    scores = SCORES.copy()
    scores[0, 0] = scores[3, 2] = np.nan
    assert np.unravel_index(np.argmax(scores), scores.shape) == (0, 0)
    for direction, (i, j) in (("max", (1, 1)), ("min", (2, 1))):
        got = _curated("AtOptimum")(_grid(scores), _per_node(rng), direction)
        assert (got[K_DIM], got[V_DIM], float(got["score"])) == (K[i], V[j], SCORES[i, j])


def test_a_score_that_is_nan_everywhere_is_refused_by_name(rng):
    """A score with no number in any cell has no optimum, and the refusal names the score."""
    with pytest.raises(ValueError, match=r"`score` \(fit\) has no cell that is not NaN"):
        _curated("AtOptimum")(_grid(np.full(SCORES.shape, np.nan)), _per_node(rng), "max")


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_tie_goes_to_the_smallest_coordinates_whatever_the_order_of_dimensions_and_labels(transpose, reverse, rng):
    """Two cells share the maximum and the one with the smaller K wins, K's name sorting before v's, in every layout of the same score; a positional argmax takes the other cell where the labels are reversed or the dimensions transposed, but not both."""
    scores = SCORES.copy()
    scores[1, 2] = scores[3, 0] = 2.0
    score, value = _grid(scores), _per_node(rng)
    want = value.values[1, 2]
    if reverse:
        flipped = {K_DIM: slice(None, None, -1), V_DIM: slice(None, None, -1)}
        score, value = score.isel(flipped), value.isel(flipped)
    if transpose:
        score, value = score.transpose(V_DIM, K_DIM), value.transpose("node", V_DIM, K_DIM)
    first = np.unravel_index(np.argmax(score.values), score.shape)
    positional = {d: score[d].values[k] for d, k in zip(score.dims, first, strict=True)}
    assert (positional[K_DIM] == K[1]) == (reverse == transpose)

    got = _curated("AtOptimum")(score, value, "max")
    assert (got[K_DIM], got[V_DIM]) == (K[1], V[2])
    np.testing.assert_array_equal(got["value"].values, want)


def test_the_value_is_read_at_the_labels_of_the_optimum_not_its_position(rng):
    """A value whose K labels run the other way is read at the optimum's K and v, not at the optimum's position in the score."""
    value = _per_node(rng)
    got = _curated("AtOptimum")(_grid(SCORES), value.isel({K_DIM: slice(None, None, -1)}), "max")
    np.testing.assert_array_equal(got["value"].values, value.values[1, 1])


def test_a_dimension_without_labels_is_labelled_by_position(rng):
    """A score over unlabelled components, as a factorization's selection rule scores them, picks a component by position and returns that position."""
    rho = xr.DataArray([0.2, -0.7, 0.5, 0.1], dims=["component"])
    subnetwork = xr.DataArray(rng.random((4, 3, 3)), dims=("component", "node_i", "node_j"))
    got = _curated("AtOptimum")(rho, subnetwork, "min")
    assert int(got["component"]) == 1 and float(got["score"]) == -0.7
    np.testing.assert_array_equal(got["value"].values, subnetwork.values[1])


def test_a_value_that_does_not_share_the_cells_of_the_score_is_refused(rng):
    """A value without one of the score's dimensions, or over other labels along it, is refused by name, as are a bare score and a direction that is neither max nor min."""
    score, value = _grid(SCORES), _per_node(rng)
    at_optimum = _curated("AtOptimum")
    with pytest.raises(ValueError, match="`value` does not run along 'network.conduction_speed'"):
        at_optimum(score, value.isel({V_DIM: 0}), "max")
    with pytest.raises(ValueError, match="`score` and `value` do not carry the same labels along 'network.conduction_speed'"):
        at_optimum(score, value.isel({V_DIM: [0, 1]}), "max")
    with pytest.raises(ValueError, match="`score` is a bare array"):
        at_optimum(SCORES, value, "max")
    with pytest.raises(ValueError, match="direction must be one of"):
        at_optimum(score, value, None)


# ------------------------------------------------------------------------- as study analyses


def test_both_run_as_study_analyses_and_each_coordinate_is_an_output(tmp_path, rng):
    """The coefficient per cell is stored under its analysis's name, and the value at its maximum, the score and both coordinates as outputs a figure binds by name."""
    sim = _per_node(rng)
    emp = xr.DataArray(rng.normal(9.3, 0.3, len(NODES)), dims=["node"], coords={"node": NODES})
    _seed(tmp_path, "simulated", sim)
    _seed(tmp_path, "empirical", emp)
    concordance = Analysis(
        name="concordance",
        iri="tvbo:functions/ConcordanceCorrelation",
        arguments={
            "x": {"used": {"analysis": "simulated", "output": "simulated"}},
            "y": {"used": {"analysis": "empirical", "output": "empirical"}},
            "dim": "node",
        },
    )
    at_best_fit = Analysis(
        name="at_best_fit",
        iri="tvbo:functions/AtOptimum",
        arguments={
            "score": {"used": {"analysis": "concordance", "output": "concordance"}},
            "value": {"used": {"analysis": "simulated", "output": "simulated"}},
            "direction": "max",
        },
    )
    analysis_io.run_analysis(concordance, tmp_path)
    analysis_io.run_analysis(at_best_fit, tmp_path)

    want = _curated("ConcordanceCorrelation")(sim, emp, dim="node")
    i, j = np.unravel_index(np.argmax(want.values), want.shape)
    with xr.open_dataset(analysis_io.container_path("concordance", tmp_path), engine="h5netcdf") as stored:
        assert stored["observation__concordance"].dims == (K_DIM, V_DIM)
        np.testing.assert_array_equal(stored["observation__concordance"].values, want.values)
    with xr.open_dataset(analysis_io.container_path("at_best_fit", tmp_path), engine="h5netcdf") as stored:
        assert set(stored.data_vars) == {f"observation__{key}" for key in ("value", "score", K_DIM, V_DIM)}
        np.testing.assert_array_equal(stored["observation__value"].values, sim.values[i, j])
        assert float(stored["observation__score"]) == want.values[i, j]
        assert (float(stored[f"observation__{K_DIM}"]), float(stored[f"observation__{V_DIM}"])) == (K[i], V[j])
