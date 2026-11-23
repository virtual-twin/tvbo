"""The centre of values over one named dimension and a band around it, reached through the curated record ``SpreadBand`` as a study reaches it.

``SpreadBand`` reduces ``values`` along ``dim`` to ``center`` and to ``band``, whose last dimension ``bound`` holds the two edges in the order ``lower``, ``upper`` that the ``band`` and ``errorbar`` marks read by position. ``mean_sd`` is the mean with one standard deviation either side, the squared deviations over ``n - ddof``; ``median_iqr`` is the median with the 25th and 75th percentiles, each at rank ``p (n - 1)`` of the sorted values and interpolated linearly. NaN entries are left out, a slice of nothing but NaN gives NaN without an error, and every other dimension is kept with its labels. The results equal the lines Koller et al. draw their bands with (doi:10.1038/s41467-024-47860-x): Fig. 5b's ``nanmean`` and ``nanstd`` (released ``modules/helpers.py`` l. 554-555, transcribed in :func:`_koller_mean_sd`), Fig. 3b's median and percentiles (``scripts/results/03_results.ipynb`` cell 10, transcribed in :func:`_koller_median_iqr`), and Fig. 9d's mean and sample standard deviation of a pandas table (``48_results.ipynb`` cells 9-10).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

from .test_curated_statistics import _curated, _seed

KNOWN = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
"""Values with mean 5 whose squared deviations sum to 32."""

EVEN = [16.0, 0.0, 9.0, 1.0, 25.0, 4.0]
ODD = [36.0, 9.0, 0.0, 25.0, 1.0, 16.0, 4.0]
GROUPS = ["left", "right"]
TIMES = np.arange(0.0, 200.0, 5.0)
SCALING = "omega_gradient_scaling"
FORMS = [("mean_sd", 0), ("median_iqr", None)]
"""Each statistic with the ``ddof`` it takes."""


@pytest.fixture
def rng():
    """A seeded generator, the same in every test."""
    return np.random.default_rng(0)


def _koller_mean_sd(corr):
    """Fig. 5b's centre and edges as Koller et al. compute them over the simulations of axis 0: ``aggregate_results`` in ``modules/helpers.py`` l. 554-555, filled from mean minus to mean plus SD in ``30_results.ipynb`` (file l. 463)."""
    corr_mean = np.nanmean(corr, axis=0)
    corr_sd = np.nanstd(corr, axis=0)
    return corr_mean, corr_mean - corr_sd, corr_mean + corr_sd


def _koller_median_iqr(corr):
    """Fig. 3b's centre and edges as Koller et al. compute them over the simulations of axis 0: ``03_results.ipynb`` cell 10 (file l. 408-409), filled between the two percentiles in cell 11 (file l. 449)."""
    median = np.median(corr, axis=0)
    iqr = np.percentile(corr, [25, 75], axis=0)
    return median, iqr[0], iqr[1]


def _trials(rng):
    """A correlation per trial, hemisphere and 5 ms frame with a few NaN entries and a trial of zeros, as ``aggregate_results`` keeps a trial whose hemisphere had no wave."""
    rows = rng.normal(-0.4, 0.2, (100, len(GROUPS), TIMES.size))
    rows[3] = 0.0
    rows[rng.random(rows.shape) < 0.02] = np.nan
    return xr.DataArray(
        rows, dims=("trial", "group", "time"), coords={"trial": np.arange(100), "group": GROUPS, "time": TIMES}
    )


def _scores(rng):
    """A score per trial and gradient scaling, as Fig. 3b plots them."""
    return xr.DataArray(
        rng.normal(-0.3, 0.25, (100, 31)), dims=("trial", SCALING), coords={SCALING: np.linspace(0.0, 1.5, 31)}
    )


def _assert_band(got, center, lower, upper):
    """*got*'s centre and both edges equal *center*, *lower* and *upper* to 1e-12, NaN where they are NaN."""
    band = got["band"]
    for have, want in ((got["center"], center), (band.sel(bound="lower"), lower), (band.sel(bound="upper"), upper)):
        np.testing.assert_allclose(have.values, want, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------- the two statistics


@pytest.mark.parametrize(("ddof", "sd"), [(0, 2.0), (1, np.sqrt(32 / 7))], ids=["ddof0", "ddof1"])
def test_mean_sd_gives_the_mean_and_one_standard_deviation_either_side_exactly(ddof, sd):
    """2, 4, 4, 4, 5, 5, 7, 9 has mean 5 and squared deviations summing to 32, so its standard deviation is 2 over n and the square root of 32 / 7 over n - 1."""
    got = _curated("SpreadBand")(np.array(KNOWN), dim="trial", statistic="mean_sd", ddof=ddof)
    assert float(got["center"]) == 5.0
    assert got["band"].values.tolist() == [5.0 - sd, 5.0 + sd]


@pytest.mark.parametrize(("values", "quartiles"), [(EVEN, (1.75, 6.5, 14.25)), (ODD, (2.5, 9.0, 20.5))], ids=["even", "odd"])
def test_median_iqr_reads_each_percentile_at_rank_p_n_minus_1_interpolated_linearly(values, quartiles):
    """Sorted, 0, 1, 4, 9, 16, 25 puts the 25th, 50th and 75th percentiles at ranks 1.25, 2.5 and 3.75, and 0, 1, 4, 9, 16, 25, 36 at 1.5, 3 and 4.5, each between its two neighbours in proportion."""
    got = _curated("SpreadBand")(np.array(values), dim="trial", statistic="median_iqr")
    lower, center, upper = quartiles
    assert float(got["center"]) == center
    assert got["band"].values.tolist() == [lower, upper]


# ------------------------------------------------------------------------------------ NaN entries


@pytest.mark.parametrize(
    ("statistic", "ddof", "values"), [("mean_sd", 1, KNOWN), ("median_iqr", None, EVEN)], ids=["mean_sd", "median_iqr"]
)
def test_nan_entries_are_left_out(statistic, ddof, values):
    """NaN entries among the values change nothing: the centre and the band are those of the other values."""
    spread_band = _curated("SpreadBand")
    got = spread_band(np.insert(np.array(values), [0, 3, 3], np.nan), dim="trial", statistic=statistic, ddof=ddof)
    clean = spread_band(np.array(values), dim="trial", statistic=statistic, ddof=ddof)
    assert float(got["center"]) == float(clean["center"])
    assert got["band"].values.tolist() == clean["band"].values.tolist()


@pytest.mark.parametrize(("statistic", "ddof"), FORMS, ids=["mean_sd", "median_iqr"])
def test_a_slice_of_nothing_but_nan_gives_nan_and_raises_nothing(statistic, ddof):
    """A frame no trial has a value for is NaN in the centre and in both edges, as Koller et al.'s ``nanmean`` returns it, and the frames beside it keep their numbers; numpy's warning about the empty slice does not escape either."""
    rows = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 5.0], [4.0, np.nan, 4.0], [3.0, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = _curated("SpreadBand")(xr.DataArray(rows, dims=("trial", "time")), dim="trial", statistic=statistic, ddof=ddof)
    assert np.isnan(got["center"].values[1]) and np.isnan(got["band"].values[1]).all()
    assert np.isfinite(got["center"].values[[0, 2]]).all() and np.isfinite(got["band"].values[[0, 2]]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(np.nanmean(rows, axis=0)[1])


# --------------------------------------------------------------------------- dimensions and labels


def test_every_other_dimension_is_kept_with_its_labels_in_its_order(rng):
    """Over the trials of a correlation per hemisphere and frame, the centre runs over hemisphere and frame with their labels, each entry that of its own slice, wherever the trial axis lies."""
    rows = _trials(rng)
    spread_band = _curated("SpreadBand")
    got = spread_band(rows, dim="trial", statistic="mean_sd", ddof=0)
    assert got["center"].dims == ("group", "time") and got["band"].dims == ("group", "time", "bound")
    assert got["center"]["group"].values.tolist() == GROUPS
    np.testing.assert_array_equal(got["band"]["time"].values, TIMES)
    one = rows.sel(group="right", time=TIMES[7]).values
    np.testing.assert_allclose(float(got["center"].sel(group="right", time=TIMES[7])), np.nanmean(one), rtol=0, atol=1e-12)
    moved = spread_band(rows.transpose("group", "trial", "time"), dim="trial", statistic="mean_sd", ddof=0)
    xr.testing.assert_allclose(moved["band"], got["band"], rtol=0, atol=1e-12)


@pytest.mark.parametrize(("statistic", "ddof"), FORMS, ids=["mean_sd", "median_iqr"])
def test_the_band_ends_on_a_bound_axis_lower_then_upper(statistic, ddof, rng):
    """Over one swept axis the band is an (n, 2) array whose last axis is ``bound``, ``lower`` first and ``upper`` second, the positions the ``band`` and ``errorbar`` marks read its two edges from."""
    scores = _scores(rng)
    band = _curated("SpreadBand")(scores, dim="trial", statistic=statistic, ddof=ddof)["band"]
    assert band.dims == (SCALING, "bound") and band.shape == (scores.sizes[SCALING], 2)
    assert band["bound"].values.tolist() == ["lower", "upper"]
    np.testing.assert_array_equal(band.values[:, 0], band.sel(bound="lower").values)
    assert (band.values[:, 0] < band.values[:, 1]).all()


# --------------------------------------------------------------------------- Koller et al.'s lines


@pytest.mark.parametrize("seed", range(3))
def test_mean_sd_equals_kollers_fig5b_lines_on_random_trials(seed):
    """A correlation per trial and frame with NaN entries and a trial of zeros gives Koller et al.'s Fig. 5b mean and band at ddof 0 to 1e-12, per hemisphere."""
    rows = _trials(np.random.default_rng(seed))
    got = _curated("SpreadBand")(rows, dim="trial", statistic="mean_sd", ddof=0)
    for group in GROUPS:
        hemisphere = {key: value.sel(group=group) for key, value in got.items()}
        _assert_band(hemisphere, *_koller_mean_sd(rows.sel(group=group).values))


@pytest.mark.parametrize("seed", range(3))
def test_median_iqr_equals_kollers_fig3b_lines_on_random_trials(seed):
    """A score per trial and gradient scaling gives Koller et al.'s Fig. 3b median and quartiles to 1e-12; with NaN trials it gives the ``nanmedian`` and ``nanpercentile`` his ``aggregate_results`` takes of the wave proportions (``modules/helpers.py`` l. 545-549)."""
    rng = np.random.default_rng(seed)
    scores = _scores(rng)
    spread_band = _curated("SpreadBand")
    _assert_band(spread_band(scores, dim="trial", statistic="median_iqr"), *_koller_median_iqr(scores.values))
    holed = scores.where(rng.random(scores.shape) > 0.05)
    _assert_band(
        spread_band(holed, dim="trial", statistic="median_iqr"),
        np.nanmedian(holed.values, axis=0),
        *np.nanpercentile(holed.values, [25, 75], axis=0),
    )


def test_mean_sd_at_ddof_1_equals_the_mean_and_sample_sd_of_kollers_fig9d_table(rng):
    """The cross-validation RMSE per repetition and rank, a pandas table as ``48_results.ipynb`` cell 9 builds it, gives per rank the table's mean and the mean minus and plus its ``std`` (ddof 1), the spread seaborn's ``errorbar='sd'`` draws in cell 10, to 1e-12."""
    table = pd.DataFrame(rng.normal(100.5, 0.4, (100, 9)), columns=np.arange(2, 11))
    rmse = xr.DataArray(table.to_numpy(), dims=("repetition", "rank"), coords={"rank": table.columns.to_numpy()})
    got = _curated("SpreadBand")(rmse, dim="repetition", statistic="mean_sd", ddof=1)
    mean, sd = table.mean(), table.std()
    _assert_band(got, mean.to_numpy(), (mean - sd).to_numpy(), (mean + sd).to_numpy())


# ------------------------------------------------------------------------- refusals and study use


def test_a_statistic_or_ddof_that_does_not_fit_is_refused():
    """A statistic outside the two, a ``mean_sd`` without ``ddof``, a ``median_iqr`` with one and values that do not run along ``dim`` are refused, rather than one form of the deviation picked or a ``ddof`` ignored."""
    spread_band = _curated("SpreadBand")
    values = np.arange(5.0)
    with pytest.raises(ValueError, match="statistic must be one of"):
        spread_band(values, dim="trial", statistic="mean_ci")
    with pytest.raises(ValueError, match="`mean_sd` needs `ddof`"):
        spread_band(values, dim="trial", statistic="mean_sd")
    with pytest.raises(ValueError, match="`median_iqr` takes no `ddof`, got 1"):
        spread_band(values, dim="trial", statistic="median_iqr", ddof=1)
    with pytest.raises(ValueError, match="`values` does not run along 'trial'"):
        spread_band(xr.DataArray(values, dims=["time"]), dim="trial", statistic="median_iqr")


def test_it_runs_as_a_study_analysis_and_stores_center_and_band_with_their_labels(tmp_path, rng):
    """Named by ``iri`` with its arguments and no ``ddof``, it stores ``center`` and ``band`` as outputs a figure binds by name, the ``bound`` labels and those of the other dimensions intact."""
    rows = _trials(rng)
    _seed(tmp_path, "rows", rows)
    over_trials = Analysis(
        name="over_trials",
        iri="tvbo:functions/SpreadBand",
        arguments={"values": {"used": {"analysis": "rows", "output": "rows"}}, "dim": "trial", "statistic": "median_iqr"},
    )
    analysis_io.run_analysis(over_trials, tmp_path)

    want = _curated("SpreadBand")(rows, dim="trial", statistic="median_iqr")
    with xr.open_dataset(analysis_io.container_path("over_trials", tmp_path), engine="h5netcdf") as stored:
        assert set(stored.data_vars) == {"observation__center", "observation__band"}
        band = stored["observation__band"]
        assert band.dims == ("group", "time", "bound") and band["bound"].values.tolist() == ["lower", "upper"]
        assert band["group"].values.tolist() == GROUPS
        np.testing.assert_array_equal(band.values, want["band"].values)
        np.testing.assert_array_equal(stored["observation__center"].values, want["center"].values)
