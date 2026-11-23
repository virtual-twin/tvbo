"""Pearson's and Spearman's correlation along one named dimension, reached through the curated record ``Correlation`` as a study reaches it.

``Correlation`` correlates ``x`` with ``y`` over the units of ``dim`` and keeps every other dimension with its labels. ``spearman`` is Pearson's r over the ranks, tied values given the mean of the ranks they span, so a monotone curve gives 1 where Pearson's r stays below it, and the tied pair ``TIED_X``, ``TIED_Y`` gives the value worked out by hand in ``TIED_RHO``. A unit enters where both sides are finite, so a NaN drops its own pair only. Labelled inputs are paired by label, and a label only one of them carries is refused by name. On random data with ties the coefficient equals the scipy calls Koller et al. compute theirs with (doi:10.1038/s41467-024-47860-x): ``spearmanr`` at its defaults for the Figs. 3b, 7a and Suppl. Fig. 2a correlations (released ``scripts/analysis/03_analysis_aggregate_results.py`` l. 180, transcribed in :func:`_koller_spearman`) and ``pearsonr`` at its defaults for the FC fits (``scripts/analysis/40_analysis_fc_corr.py`` l. 152, transcribed in :func:`_koller_pearson`).
"""

import warnings

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

from .test_curated_statistics import _curated, _seed

METHODS = ["pearson", "spearman"]
SCALING = "omega_gradient_scaling"
TIED_X = [10.0, 20.0, 20.0, 30.0, 40.0]
TIED_Y = [3.0, 1.0, 4.0, 4.0, 5.0]
TIED_RHO = 29 / 38
"""Spearman's r of ``TIED_X`` and ``TIED_Y`` from the rank definition. The average ranks are 1, 2.5, 2.5, 4, 5 and 2, 1, 3.5, 3.5, 5, both with mean 3, so the deviations are -2, -0.5, -0.5, 1, 2 and -1, -2, 0.5, 0.5, 2. Their products sum to 2 + 1 - 0.25 + 0.5 + 4 = 7.25, and the squares of each side to 4 + 0.25 + 0.25 + 1 + 4 = 9.5 and 1 + 4 + 0.25 + 0.25 + 4 = 9.5, so r = 7.25 / sqrt(9.5 * 9.5) = 29 / 38. Ranks 1 to 5 without averaging would give 0.9, and the formula without ties, 1 - 6 sum(d**2) / (n (n**2 - 1)), on the average ranks 1 - 6 * 4.5 / 120 = 0.775."""


@pytest.fixture
def rng():
    """A seeded generator, the same in every test."""
    return np.random.default_rng(0)


def _koller_spearman(x, y):
    """``spearmanr(gradient, effective_frequency)[0]`` of Koller et al.'s released ``03_analysis_aggregate_results.py`` l. 180, scipy at its defaults as in every Spearman call of the release."""
    return scipy.stats.spearmanr(x, y)[0]


def _koller_pearson(x, y):
    """``pearsonr(FC_plv_avg[out].flatten(), FC_plv_emp[out].flatten())[0]`` of Koller et al.'s released ``40_analysis_fc_corr.py`` l. 152, scipy at its defaults."""
    return scipy.stats.pearsonr(x, y)[0]


KOLLER = {"pearson": _koller_pearson, "spearman": _koller_spearman}


def test_a_monotone_curve_gives_spearman_one_and_pearson_below_one():
    """An exponential of x keeps the order of x exactly, so the ranks agree and Spearman's r is 1, while the values lie on no line and Pearson's r stays below 1."""
    x = np.linspace(0.0, 3.0, 40)
    y = np.exp(2.0 * x)
    fn = _curated("Correlation")
    assert abs(float(fn(x, y, dim="node", method="spearman")["r"]) - 1.0) < 1e-12
    assert float(fn(x, y, dim="node", method="pearson")["r"]) < 0.99


def test_ties_take_the_mean_of_the_ranks_they_span():
    """Two tied values in x and two in y give the average-rank coefficient worked out in ``TIED_RHO``."""
    got = _curated("Correlation")(np.array(TIED_X), np.array(TIED_Y), dim="node", method="spearman")
    assert abs(float(got["r"]) - TIED_RHO) < 1e-15
    assert int(got["n"]) == 5


@pytest.mark.parametrize("method", METHODS)
def test_a_pair_with_a_nan_on_either_side_is_dropped_and_the_rest_ranked_among_themselves(method, rng):
    """NaN in x at some units and in y at others removes those pairs only: r and n are those of the 56 pairs left, ranked among themselves, which is scipy's coefficient of those pairs, where Koller et al.'s ``spearmanr`` call at its defaults returns NaN for the whole slice."""
    x, y = rng.normal(size=60), rng.normal(size=60)
    x[[3, 17]] = np.nan
    y[[17, 40, 52]] = np.nan
    kept = np.isfinite(x) & np.isfinite(y)
    fn = _curated("Correlation")
    got = fn(x, y, dim="node", method=method)
    assert int(got["n"]) == int(kept.sum()) == 56
    assert float(got["r"]) == float(fn(x[kept], y[kept], dim="node", method=method)["r"])
    assert abs(float(got["r"]) - KOLLER[method](x[kept], y[kept])) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(_koller_spearman(x, y))


def test_fewer_than_three_pairs_or_a_constant_side_give_nan_beside_their_count():
    """Two finite pairs and a constant side state no correlation, as every correlation of the module, and still count their pairs."""
    fn = _curated("Correlation")
    two = fn(np.array([0.0, 1.0, np.nan, 3.0]), np.array([1.0, 2.0, 5.0, np.nan]), dim="node", method="spearman")
    assert np.isnan(float(two["r"])) and int(two["n"]) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        flat = fn(np.arange(6.0), np.full(6, 0.1), dim="node", method="pearson")
    assert np.isnan(float(flat["r"])) and int(flat["n"]) == 6


def test_every_other_dimension_is_kept_with_its_labels(rng):
    """A template over nodes against a map per scaling and trial gives one r and n per scaling and trial, labelled and without the template's units, each that of its own slice, whichever axis the nodes run along."""
    nodes = [f"n{i}" for i in range(30)]
    template = xr.DataArray(rng.normal(size=30), dims=["node"], coords={"node": nodes}, attrs={"units": "Hz"})
    maps = xr.DataArray(
        rng.normal(size=(3, 4, 30)) + template.values,
        dims=(SCALING, "trial", "node"),
        coords={SCALING: [0.0, 0.75, 1.5], "trial": [0, 1, 2, 3], "node": nodes},
    )
    maps[1, 2, :5] = np.nan
    fn = _curated("Correlation")
    got = fn(template, maps, dim="node", method="spearman")
    assert got["r"].dims == got["n"].dims == (SCALING, "trial")
    assert not got["r"].attrs
    assert got["r"][SCALING].values.tolist() == [0.0, 0.75, 1.5] and got["r"]["trial"].values.tolist() == [0, 1, 2, 3]
    for i, j in np.ndindex(3, 4):
        one = fn(template, maps[i, j], dim="node", method="spearman")
        assert (float(got["r"][i, j]), int(got["n"][i, j])) == (float(one["r"]), int(one["n"]))
    assert int(got["n"][1, 2]) == 25
    moved = fn(template, maps.transpose("node", SCALING, "trial"), dim="node", method="spearman")
    np.testing.assert_array_equal(moved["r"].values, got["r"].values)


def test_labelled_inputs_pair_on_their_labels_not_their_positions(rng):
    """The values of y in another node order give the coefficient of the pairs their labels name."""
    nodes = np.array([f"n{i}" for i in range(40)])
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.5, size=40)
    order = rng.permutation(40)
    labelled_x = xr.DataArray(x, dims=["node"], coords={"node": nodes})
    shuffled_y = xr.DataArray(y[order], dims=["node"], coords={"node": nodes[order]})
    fn = _curated("Correlation")
    for method in METHODS:
        got = fn(labelled_x, shuffled_y, dim="node", method=method)
        assert float(got["r"]) == float(fn(x, y, dim="node", method=method)["r"])


def test_a_label_only_one_side_carries_is_refused_by_name(rng):
    """Two maps over different nodes are refused, naming the nodes, rather than correlated over the shared ones."""
    x = xr.DataArray(rng.random(4), dims=["node"], coords={"node": ["a", "b", "c", "d"]})
    y = xr.DataArray(rng.random(4), dims=["node"], coords={"node": ["a", "b", "c", "e"]})
    with pytest.raises(
        ValueError, match="`x` and `y` do not carry the same labels along 'node': 'd', 'e' are in only one of them"
    ):
        _curated("Correlation")(x, y, dim="node", method="spearman")


def test_a_method_or_an_input_that_does_not_fit_is_refused(rng):
    """A method other than pearson or spearman, none at all, an input without ``dim`` and a bare 2-D input are refused."""
    fn = _curated("Correlation")
    for method in ("kendall", None):
        with pytest.raises(ValueError, match="method must be one of"):
            fn(rng.random(5), rng.random(5), dim="node", method=method)
    with pytest.raises(ValueError, match="does not run along 'node'"):
        fn(xr.DataArray(rng.random(5), dims=["time"]), rng.random(5), dim="node", method="pearson")
    with pytest.raises(ValueError, match="bare array of rank 2"):
        fn(rng.random(5), rng.random((5, 2)), dim="node", method="pearson")


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", METHODS)
def test_r_equals_kollers_scipy_call_on_random_maps(method, seed):
    """A template over the 900 oscillators of the sheet and a related map, rounded to 0.01 so that most of its values are tied, give Koller et al.'s coefficient to 1e-12 over all 900 pairs."""
    rng = np.random.default_rng(seed)
    template = rng.uniform(-1.0, 1.0, 900)
    frequency = np.round(9.5 + 0.2 * template + rng.normal(0.0, 0.1, 900), 2)
    got = _curated("Correlation")(template, frequency, dim="node", method=method)
    assert int(got["n"]) == 900
    assert abs(float(got["r"]) - KOLLER[method](template, frequency)) < 1e-12


def test_it_runs_as_a_study_analysis_and_stores_r_and_n_with_their_labels(tmp_path, rng):
    """Named by ``iri`` and run on seeded containers, it stores ``observation__r`` and ``observation__n`` on the trial axis, read back from the file."""
    nodes = [f"n{i}" for i in range(30)]
    template = xr.DataArray(rng.normal(size=30), dims=["node"], coords={"node": nodes})
    maps = xr.DataArray(
        rng.normal(size=(2, 30)) + template.values, dims=("trial", "node"), coords={"trial": [0, 1], "node": nodes}
    )
    _seed(tmp_path, "template", template)
    _seed(tmp_path, "maps", maps)
    analysis = Analysis(
        name="template_map_rho",
        iri="tvbo:functions/Correlation",
        arguments={
            "x": {"used": {"analysis": "template", "output": "template"}},
            "y": {"used": {"analysis": "maps", "output": "maps"}},
            "dim": "node",
            "method": "spearman",
        },
    )
    analysis_io.run_analysis(analysis, tmp_path)

    want = _curated("Correlation")(template, maps, dim="node", method="spearman")
    with xr.open_dataset(analysis_io.container_path("template_map_rho", tmp_path), engine="h5netcdf") as stored:
        assert set(stored.data_vars) == {"observation__r", "observation__n"}
        assert stored["observation__r"].dims == ("trial",) and stored["trial"].values.tolist() == [0, 1]
        np.testing.assert_array_equal(stored["observation__r"].values, want["r"].values)
        assert stored["observation__n"].values.tolist() == [30, 30]
