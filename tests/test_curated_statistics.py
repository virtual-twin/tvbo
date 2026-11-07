"""The software-based curated functions against scipy and numpy, and an analysis that names a curated function by ``iri``.

``GroupedCorrelation``, ``FalseDiscoveryControl`` and ``Histogram`` are reached the way a study reaches them, through their records in ``tvbo/database/functions/``, and compared with the library a reader would reach for, on groups and bins formed explicitly here. A second reference forms the groups with :func:`numpy.digitize` and regresses with :func:`scipy.stats.linregress`, the way a study wrote this statistic by hand, and must agree to the last bit, so a study that moves to the curated function keeps its numbers.
"""

import re
import warnings

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from tvbo import Function
from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

GROUPINGS = ["equal_width", "equal_count"]


def _curated(name):
    """The callable ``tvbo:functions/<name>`` records."""
    return Function.from_string(f"iri: tvbo:functions/{name}").resolved_callable


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _sample(rng, n=160):
    """A correlated pair with one far unit, so that equal-width bins leave some groups empty."""
    x = np.append(rng.standard_normal(n - 1), 9.0)
    return x, 0.6 * x + rng.standard_normal(n)


def _explicit_groups(by, n_groups, grouping):
    """Each group's members, formed without the function under test: by interval membership for equal width, by :func:`numpy.array_split` of the stable order for equal count."""
    if grouping == "equal_count":
        return list(np.array_split(np.argsort(by, kind="stable"), n_groups))
    edges = np.linspace(by.min(), by.max(), n_groups + 1)
    return [np.flatnonzero((by >= edges[k]) & ((by < edges[k + 1]) | (k == n_groups - 1))) for k in range(n_groups)]


def _filled(result, key):
    """One per-group output over the groups that have members."""
    return result[key].values[result["size"].values > 0]


# ---------------------------------------------------------------- the grouped correlation


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_grouped_correlation_is_pearson_over_explicit_group_means(grouping, rng):
    x, y = _sample(rng)
    got = _curated("GroupedCorrelation")(x, y, n_groups=12, dim="unit", grouping=grouping)

    members = _explicit_groups(x, 12, grouping)
    assert [m.size for m in members] == got["size"].values.tolist()
    filled = [m for m in members if m.size]
    assert (grouping == "equal_width") == (len(filled) < 12)
    gx, gy = np.array([x[m].mean() for m in filled]), np.array([y[m].mean() for m in filled])
    np.testing.assert_allclose(_filled(got, "x_mean"), gx, rtol=1e-13)
    np.testing.assert_allclose(_filled(got, "y_mean"), gy, rtol=1e-13)
    sd = np.array([np.std(y[m], ddof=1) if m.size > 1 else np.nan for m in filled])
    np.testing.assert_allclose(_filled(got, "y_sd"), sd, rtol=1e-12)
    np.testing.assert_allclose(_filled(got, "y_se"), sd / np.sqrt([m.size for m in filled]), rtol=1e-12)
    assert np.isnan(got["y_mean"].values[got["size"].values == 0]).all()

    over_groups, over_units = scipy.stats.pearsonr(gx, gy), scipy.stats.pearsonr(x, y)
    np.testing.assert_allclose([got["r"], got["p"]], [over_groups.statistic, over_groups.pvalue], rtol=1e-9)
    np.testing.assert_allclose([got["r_unit"], got["p_unit"]], [over_units.statistic, over_units.pvalue], rtol=1e-12)
    slope, intercept = np.polyfit(gx, gy, 1)
    np.testing.assert_allclose([got["slope"], got["intercept"]], [slope, intercept], rtol=1e-9)
    np.testing.assert_allclose(got["fit_x"].values, [gx.min(), gx.max()], rtol=1e-13)
    np.testing.assert_allclose(got["fit_y"].values, intercept + slope * np.array([gx.min(), gx.max()]), rtol=1e-9)
    assert (int(got["n_nonempty_groups"]), int(got["n_units"]), got["n_units_total"]) == (len(filled), x.size, x.size)


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_groups_are_formed_over_another_variable_when_one_is_named(grouping, rng):
    x, y = _sample(rng)
    by = rng.standard_normal(x.size)
    got = _curated("GroupedCorrelation")(x, y, n_groups=8, dim="unit", by=by, grouping=grouping)

    filled = [m for m in _explicit_groups(by, 8, grouping) if m.size]
    gx, gy = np.array([x[m].mean() for m in filled]), np.array([y[m].mean() for m in filled])
    np.testing.assert_allclose(_filled(got, "x_mean"), gx, rtol=1e-13)
    np.testing.assert_allclose([got["r"], got["p"]], list(scipy.stats.pearsonr(gx, gy)), rtol=1e-9)


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_a_unit_missing_in_any_input_is_dropped(grouping, rng):
    x, y = _sample(rng)
    by = rng.standard_normal(x.size)
    holed = [a.copy() for a in (x, y, by)]
    for a, at in zip(holed, ([3, 40], [40, 77, 78], [5]), strict=True):
        a[at] = np.nan
    keep = np.setdiff1d(np.arange(x.size), [3, 5, 40, 77, 78])

    fn = _curated("GroupedCorrelation")
    got = fn(*holed[:2], n_groups=8, dim="unit", by=holed[2], grouping=grouping)
    want = fn(x[keep], y[keep], n_groups=8, dim="unit", by=by[keep], grouping=grouping)
    for key in want:
        np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(want[key]), err_msg=key)
    assert int(got["n_units"]) == keep.size


def test_a_one_member_group_has_a_mean_and_no_spread():
    x = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 2.9, 3.0])
    y = np.array([1.0, 2.0, 1.5, 4.0, 5.0, 7.0, 9.0])
    got = _curated("GroupedCorrelation")(x, y, n_groups=3, dim="unit")

    assert got["size"].values.tolist() == [3, 2, 2]
    solo = _curated("GroupedCorrelation")(np.append(x, 1.6), np.append(y, 3.0), n_groups=6, dim="unit")
    one = int(np.flatnonzero(solo["size"].values == 1)[0])
    assert solo["x_mean"].values[one] == 1.6 and solo["y_mean"].values[one] == 3.0
    assert np.isnan(solo["y_sd"].values[one]) and np.isnan(solo["y_se"].values[one])
    assert np.isfinite(float(solo["r"]))


def test_fewer_than_three_points_state_no_correlation():
    fn = _curated("GroupedCorrelation")
    two = fn(np.array([0.0, 1.0, np.nan, 3.0]), np.array([1.0, 2.0, 5.0, np.nan]), n_groups=2, dim="unit")
    assert int(two["n_units"]) == 2 and two["y_mean"].values.tolist() == [1.0, 2.0]
    assert all(np.isnan(float(two[key])) for key in ("r", "p", "r_unit", "p_unit", "slope", "intercept"))
    assert np.isnan(two["fit_y"].values).all()

    two_groups = fn(np.array([0.0, 0.1, 0.2, 1.0, 0.9]), np.array([1.0, 2.0, 1.5, 4.0, 5.0]), n_groups=2, dim="unit")
    assert int(two_groups["n_nonempty_groups"]) == 2
    assert np.isnan(float(two_groups["r"])) and np.isnan(float(two_groups["p"])) and np.isfinite(float(two_groups["r_unit"]))


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_a_constant_side_states_no_correlation_as_scipy_does(grouping, rng):
    x, _ = _sample(rng, 40)
    flat = np.full(40, 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(scipy.stats.pearsonr(x, flat).statistic)
    fn = _curated("GroupedCorrelation")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for got in (
            fn(x, flat, n_groups=5, dim="unit", grouping=grouping),
            fn(flat, x, n_groups=5, dim="unit", grouping=grouping),
        ):
            assert all(np.isnan(float(got[key])) for key in ("r", "p", "r_unit", "p_unit"))
            assert int(got["n_units"]) == 40


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_every_other_dimension_is_kept_with_one_result_per_element(grouping, rng):
    subjects = [f"s{i:03d}" for i in range(90)]
    score = xr.DataArray(rng.standard_normal(90), dims=["subject"], coords={"subject": subjects})
    values = rng.standard_normal((90, 5)) + 0.5 * score.values[:, None]
    values[rng.random((90, 5)) < 0.2] = np.nan
    values[:, 4] = np.nan
    values[:2, 4] = 1.0
    values = xr.DataArray(values, dims=["subject", "item"], coords={"subject": subjects, "item": [1, 2, 3, 4, 5]})

    fn = _curated("GroupedCorrelation")
    got = fn(score, values, n_groups=6, dim="subject", grouping=grouping)
    assert got["r"].dims == ("item",) and got["y_mean"].dims == ("item", "group") and got["fit_y"].dims == ("item", "end")
    assert got["r"]["item"].values.tolist() == [1, 2, 3, 4, 5]
    for j in range(5):
        one = fn(score, values.isel(item=j), n_groups=6, dim="subject", grouping=grouping)
        for key in ("x_mean", "y_mean", "y_sd", "size", "fit_y", "r", "p", "r_unit", "p_unit", "n_nonempty_groups", "n_units"):
            np.testing.assert_array_equal(got[key].isel(item=j).values, one[key].values, err_msg=f"{key}[{j}]")
    assert np.isnan(got["r"].values[4]) and int(got["n_units"].values[4]) == 2
    assert got["n_units_total"] == int(np.isfinite(values.values).any(axis=1).sum())

    own = fn(score, values, n_groups=6, dim="subject", by=values, grouping=grouping)
    one = fn(score, values.isel(item=2), n_groups=6, dim="subject", by=values.isel(item=2), grouping=grouping)
    np.testing.assert_array_equal(own["r"].values[2], one["r"].values)
    assert not np.array_equal(own["r"].values[:4], got["r"].values[:4])


def test_labelled_inputs_pair_on_their_labels_not_their_positions(rng):
    subjects = np.array([f"s{i:03d}" for i in range(60)])
    x, y = _sample(rng, 60)
    order = rng.permutation(60)[:-4]
    labelled_x = xr.DataArray(x, dims=["subject"], coords={"subject": subjects})
    shuffled_y = xr.DataArray(y[order], dims=["subject"], coords={"subject": subjects[order]})

    fn = _curated("GroupedCorrelation")
    got = fn(labelled_x, shuffled_y, n_groups=5, dim="subject")
    kept = np.sort(order)
    want = fn(x[kept], y[kept], n_groups=5, dim="subject")
    np.testing.assert_allclose([got["r"], got["r_unit"]], [want["r"], want["r_unit"]], rtol=1e-12)
    assert int(got["n_units"]) == 56


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_more_groups_than_units_leaves_groups_empty(grouping, rng):
    x, y = rng.standard_normal(4), rng.standard_normal(4)
    got = _curated("GroupedCorrelation")(x, y, n_groups=10, dim="unit", grouping=grouping)
    assert got["size"].values.sum() == 4 and int(got["n_nonempty_groups"]) == (got["size"].values > 0).sum() <= 4
    assert np.isnan(got["y_mean"].values[got["size"].values == 0]).all()
    if grouping == "equal_count":
        assert got["size"].values.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        np.testing.assert_allclose([got["r"], got["p"]], scipy.stats.pearsonr(x, y), rtol=1e-12)


def test_a_label_that_names_two_units_is_refused(rng):
    x, y = _sample(rng, 6)
    twice = {"subject": ["a", "b", "b", "c", "d", "d"]}
    fn = _curated("GroupedCorrelation")
    with pytest.raises(ValueError, match="`y` carries 'b', 'd' more than once along 'subject'"):
        fn(x, xr.DataArray(y, dims=["subject"], coords=twice), n_groups=2, dim="subject")
    with pytest.raises(ValueError, match="`x` carries 'b', 'd' more than once along 'subject'"):
        fn(
            xr.DataArray(x, dims=["subject"], coords=twice),
            xr.DataArray(y, dims=["subject"], coords=twice),
            n_groups=2,
            dim="subject",
        )


def test_an_input_that_cannot_say_where_its_units_run_is_refused(rng):
    fn = _curated("GroupedCorrelation")
    with pytest.raises(ValueError, match="does not run along 'subject'"):
        fn(xr.DataArray(np.ones(4), dims=["node"]), np.ones(4), n_groups=2, dim="subject")
    with pytest.raises(ValueError, match="bare array of rank 2"):
        fn(np.ones(4), np.ones((4, 3)), n_groups=2, dim="subject")
    with pytest.raises(ValueError, match="grouping must be one of"):
        fn(np.ones(4), np.ones(4), n_groups=2, dim="subject", grouping="quantile")


# ------------------------------------------------- a hand-written form of the same statistic


def _digitize_regression(x, y, n_groups, by=None):
    """Equal-width group means with their line and both correlations, as a study wrote them by hand: :func:`numpy.digitize` on the inner edges, empty bins dropped, :func:`scipy.stats.linregress` over the group means."""
    bs = x if by is None else by
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(bs)
    xs, ys, bs = x[ok], y[ok], bs[ok]
    edges = np.linspace(bs.min(), bs.max(), n_groups + 1)
    idx = np.clip(np.digitize(bs, edges[1:-1], right=False), 0, n_groups - 1)
    members = [m for m in (np.flatnonzero(idx == k) for k in range(n_groups)) if m.size]
    gx, gy = np.array([xs[m].mean() for m in members]), np.array([ys[m].mean() for m in members])
    size = np.array([m.size for m in members])
    sd = np.array([ys[m].std(ddof=1) if m.size > 1 else np.nan for m in members])
    slope, intercept, r, p, _ = scipy.stats.linregress(gx, gy)
    r_unit, p_unit = scipy.stats.pearsonr(xs, ys)
    ends = np.array([gx.min(), gx.max()])
    return {
        "x_mean": gx,
        "y_mean": gy,
        "size": size,
        "y_sd": sd,
        "y_se": sd / np.sqrt(size),
        "fit_x": ends,
        "fit_y": intercept + slope * ends,
        "r": r,
        "p": p,
        "r_unit": r_unit,
        "p_unit": p_unit,
        "n_units": xs.size,
    }


def _step_up(p, q):
    """The Benjamini-Hochberg selection written out: every finite p up to the largest one at or below ``q`` times its rank over the number of tests."""
    keep = np.zeros(p.shape, dtype=bool)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    below = np.flatnonzero(p[order] <= q * np.arange(1, order.size + 1) / order.size)
    if below.size:
        keep[order[: below.max() + 1]] = True
    return keep


def _cohort(rng, columns, n=650, missing=0.15):
    """A score per unit and a ``(unit, column)`` table that leans on it and has holes, the shape of a per-item or per-region cohort table."""
    score = rng.integers(5, 25, n).astype(float)
    score[rng.random(n) < 0.02] = np.nan
    values = 0.05 * score[:, None] * rng.standard_normal(columns) + rng.standard_normal((n, columns))
    values[rng.random((n, columns)) < missing] = np.nan
    return score, values


@pytest.mark.parametrize("n_groups", [20, 30])
def test_one_panel_agrees_bit_for_bit_with_the_hand_written_form(n_groups, rng):
    score, values = _cohort(rng, 1)
    got = _curated("GroupedCorrelation")(score, values[:, 0], n_groups=n_groups, dim="unit")
    want = _digitize_regression(score, values[:, 0], n_groups)
    for key in ("x_mean", "y_mean", "size", "y_sd", "y_se"):
        np.testing.assert_array_equal(_filled(got, key), want[key], err_msg=key)
    for key in ("fit_x", "fit_y", "r", "p", "r_unit", "p_unit", "n_units"):
        np.testing.assert_array_equal(np.asarray(got[key]), want[key], err_msg=key)
    assert int(got["n_nonempty_groups"]) == want["size"].size


@pytest.mark.parametrize(
    ("columns", "n_groups", "by"), [(24, 30, None), (24, 30, "score2"), (24, 20, "own"), (379, 20, None), (90, 20, None)]
)
def test_a_table_agrees_bit_for_bit_with_the_hand_written_form_per_column(columns, n_groups, by, rng):
    score, values = _cohort(rng, columns)
    second = rng.standard_normal(score.size)
    table = xr.DataArray(values, dims=["unit", "column"])
    grouping_variable = {None: None, "score2": second, "own": table}[by]
    got = _curated("GroupedCorrelation")(score, table, n_groups=n_groups, dim="unit", by=grouping_variable)

    want = [
        _digitize_regression(score, values[:, j], n_groups, {None: None, "score2": second, "own": values[:, j]}[by])
        for j in range(columns)
    ]
    for key in ("r", "p", "r_unit", "p_unit", "n_units"):
        np.testing.assert_array_equal(got[key].values, [w[key] for w in want], err_msg=key)

    fdr = _curated("FalseDiscoveryControl")(got["p"], q=0.1)
    np.testing.assert_array_equal(fdr["selected"].values, _step_up(got["p"].values, 0.1))
    counted = _curated("Histogram")(got["r"], n_bins=40, lower=-1.0, upper=1.0, where=fdr["selected"])
    np.testing.assert_array_equal(
        counted["count"].values, np.histogram(got["r"].values[fdr["selected"].values], bins=np.linspace(-1.0, 1.0, 41))[0]
    )


# ------------------------------------------------------------- false discovery control


@pytest.mark.parametrize("method", ["bh", "by"])
def test_adjusted_p_values_are_scipys(method, rng):
    p = np.concatenate([rng.uniform(0, 0.01, 12), rng.uniform(0, 1, 60), [0.2, 0.2, 1.0, 0.0]])
    got = _curated("FalseDiscoveryControl")(p, method=method)
    np.testing.assert_allclose(
        got["p_adjusted"].values, scipy.stats.false_discovery_control(p, method=method), rtol=1e-13, atol=0
    )
    assert int(got["n_tests"]) == p.size and "selected" not in got


def test_a_missing_p_value_is_no_test(rng):
    p = rng.uniform(0, 0.2, 30)
    p[[4, 17]] = np.nan
    got = _curated("FalseDiscoveryControl")(p, q=0.1)
    finite = np.isfinite(p)
    np.testing.assert_allclose(got["p_adjusted"].values[finite], scipy.stats.false_discovery_control(p[finite]), rtol=1e-13)
    assert np.isnan(got["p_adjusted"].values[~finite]).all() and not got["selected"].values[~finite].any()
    assert int(got["n_tests"]) == 28

    none = _curated("FalseDiscoveryControl")(np.full(3, np.nan), q=0.1)
    assert int(none["n_tests"]) == 0 and int(none["n_selected"]) == 0


@pytest.mark.parametrize("bad", [1.5, -0.1, np.inf, -np.inf])
def test_a_value_that_is_no_probability_is_refused_where_it_stands(bad, rng):
    fn = _curated("FalseDiscoveryControl")
    p = rng.uniform(0, 1, 12)
    p[[3, 7]] = bad, np.nan
    with pytest.raises(ValueError, match=rf"1 value\(s\) outside 0 to 1, the first {bad!r} at dim_0\[3\]"):
        fn(p, q=0.1)
    with pytest.raises(ValueError):
        scipy.stats.false_discovery_control(p[np.isfinite(p) | np.isinf(p)])

    table = xr.DataArray(p.reshape(3, 4), dims=["condition", "item"], coords={"item": [10, 11, 12, 13]})
    table[1, 2], table[2, 0] = np.nan, bad
    with pytest.raises(ValueError, match=rf"2 value\(s\) outside 0 to 1, the first {bad!r} at condition\[0\], item=13"):
        fn(table, dim="item")


@pytest.mark.parametrize("q", [1.5, -0.1, 0.0, 1.0, 1, np.inf, np.nan])
def test_a_rate_at_or_beyond_0_or_1_is_refused(q, rng):
    said = f"`q` is the rate false discoveries are held to, one number above 0 and below 1, got {float(q)!r}."
    for method in ("bh", "by"):
        with pytest.raises(ValueError, match=re.escape(said)):
            _curated("FalseDiscoveryControl")(rng.uniform(0, 1, 12), q=q, method=method)


@pytest.mark.parametrize(
    ("q", "given"),
    [([0.05, 0.1], "an array of shape (2,)"), (np.array([0.1]), "an array of shape (1,)"), (True, "True"), ("0.1", "'0.1'")],
)
def test_a_rate_that_is_not_one_number_is_refused(q, given, rng):
    with pytest.raises(ValueError, match=re.escape(f"one number above 0 and below 1, got {given}.")):
        _curated("FalseDiscoveryControl")(rng.uniform(0, 1, 12), q=q)


def test_a_rate_is_one_number_however_it_is_held_and_both_ends_are_open(rng):
    fn = _curated("FalseDiscoveryControl")
    p = rng.uniform(1e-6, 0.2, 30)
    want = fn(p, q=0.1)["selected"].values
    for q in (np.float64(0.1), np.float32(0.1), np.array(0.1), xr.DataArray(0.1)):
        np.testing.assert_array_equal(fn(p, q=q)["selected"].values, want)
    assert "selected" not in fn(p) and "selected" not in fn(p, q=None)
    assert int(fn(p, q=1 - 1e-12)["n_selected"]) == 30 and int(fn(p, q=1e-12)["n_selected"]) == 0


@pytest.mark.parametrize("method", ["bh", "by"])
@pytest.mark.parametrize("q", [0.05, 0.1])
def test_the_selection_is_the_step_up_rule_and_the_adjusted_p_at_or_below_the_rate(method, q, rng):
    fn = _curated("FalseDiscoveryControl")
    for _ in range(50):
        p = np.concatenate([rng.uniform(0, 0.02, rng.integers(0, 15)), rng.uniform(0, 1, rng.integers(5, 80))])
        got = fn(p, q=q, method=method)
        np.testing.assert_array_equal(got["selected"].values, got["p_adjusted"].values <= q)
        if method == "bh":
            np.testing.assert_array_equal(got["selected"].values, _step_up(p, q))
        assert int(got["n_selected"]) == int(got["selected"].values.sum())


def test_a_family_runs_along_the_named_dimension(rng):
    p = xr.DataArray(rng.uniform(0, 0.3, (4, 25)), dims=["condition", "item"], coords={"condition": list("abcd")})
    fn = _curated("FalseDiscoveryControl")
    got = fn(p, q=0.1, dim="item")
    assert got["p_adjusted"].dims == ("condition", "item") and got["n_selected"].dims == ("condition",)
    for i in range(4):
        one = fn(p.isel(condition=i), q=0.1)
        np.testing.assert_array_equal(got["p_adjusted"].values[i], one["p_adjusted"].values)
        assert int(got["n_selected"].values[i]) == int(one["n_selected"])
    whole = fn(p, q=0.1)
    np.testing.assert_allclose(
        whole["p_adjusted"].values.ravel(), scipy.stats.false_discovery_control(p.values.ravel()), rtol=1e-13
    )


# ------------------------------------------------------------------------ the histogram


def test_histogram_is_numpys_over_equal_width_bins(rng):
    values = np.concatenate([rng.uniform(-1.3, 1.3, 400), [np.nan, 1.0, -1.0]])
    got = _curated("Histogram")(values.reshape(13, 31), n_bins=40, lower=-1.0, upper=1.0)
    count, edges = np.histogram(values[np.isfinite(values)], bins=40, range=(-1.0, 1.0))
    np.testing.assert_array_equal(got["count"].values, count)
    np.testing.assert_allclose(got["edge"].values, edges, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got["center"].values, 0.5 * (edges[1:] + edges[:-1]), rtol=0, atol=1e-15)
    assert got["count"].dims == got["center"].dims == ("bin",) and got["n"] == count.sum() and abs(got["width"] - 0.05) < 1e-15


def test_a_mask_restricts_the_count(rng):
    values = xr.DataArray(rng.uniform(-1, 1, 50), dims=["item"], coords={"item": np.arange(50)})
    mask = xr.DataArray(rng.random(50) < 0.4, dims=["item"], coords={"item": np.arange(50)})
    fn = _curated("Histogram")
    want = np.histogram(values.values[mask.values], bins=10, range=(-1.0, 1.0))[0]
    np.testing.assert_array_equal(fn(values, n_bins=10, lower=-1.0, upper=1.0, where=mask)["count"].values, want)
    np.testing.assert_array_equal(
        fn(values.values, n_bins=10, lower=-1.0, upper=1.0, where=mask.values.astype(float))["count"].values, want
    )


def test_a_labelled_mask_is_paired_by_label_in_any_order(rng):
    values = xr.DataArray(rng.uniform(-1, 1, 50), dims=["item"], coords={"item": np.arange(50)})
    mask = xr.DataArray(rng.random(50) < 0.4, dims=["item"], coords={"item": np.arange(50)})
    fn = _curated("Histogram")
    want = np.histogram(values.values[mask.values], bins=10, range=(-1.0, 1.0))[0]
    shuffled = mask.isel(item=rng.permutation(50))
    assert not np.array_equal(shuffled.values, mask.values)
    np.testing.assert_array_equal(fn(values, n_bins=10, lower=-1.0, upper=1.0, where=shuffled)["count"].values, want)
    np.testing.assert_array_equal(
        fn(values.isel(item=rng.permutation(50)), n_bins=10, lower=-1.0, upper=1.0, where=mask)["count"].values, want
    )


def test_a_label_only_the_values_or_only_the_mask_has_is_refused(rng):
    values = xr.DataArray(rng.uniform(-1, 1, 50), dims=["item"], coords={"item": np.arange(50)})
    mask = xr.DataArray(rng.random(50) < 0.4, dims=["item"], coords={"item": np.arange(50)})
    fn = _curated("Histogram")
    with pytest.raises(ValueError, match="same labels along 'item': 0 is in only one of them"):
        fn(values, n_bins=10, lower=-1.0, upper=1.0, where=mask.isel(item=slice(1, None)))
    with pytest.raises(ValueError, match="same labels along 'item': 48, 49 are in only one of them"):
        fn(values.isel(item=slice(None, -2)), n_bins=10, lower=-1.0, upper=1.0, where=mask)


def test_a_mask_marks_every_entry_along_a_dimension_it_lacks_and_has_none_of_its_own(rng):
    table = xr.DataArray(rng.uniform(-1, 1, (6, 50)), dims=["row", "item"], coords={"item": np.arange(50)})
    mask = xr.DataArray(rng.random(50) < 0.4, dims=["item"], coords={"item": np.arange(50)})
    fn = _curated("Histogram")
    want = np.histogram(table.values[:, mask.values], bins=10, range=(-1.0, 1.0))[0]
    np.testing.assert_array_equal(fn(table, n_bins=10, lower=-1.0, upper=1.0, where=mask)["count"].values, want)
    with pytest.raises(ValueError, match=r"`where` runs along \['row'\], which `values` does not"):
        fn(table.isel(row=0), n_bins=10, lower=-1.0, upper=1.0, where=table > 0)


# ------------------------------------------------ naming a curated function from an analysis


def _seed(root, name, array):
    """Persist *array* as analysis *name*'s container, so a ``used:`` reference resolves it."""
    path = analysis_io.container_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({f"observation__{name}": array}).to_netcdf(path, engine="h5netcdf")


def test_an_analysis_written_by_iri_takes_the_function_and_states_only_its_arguments():
    analysis = Analysis(
        name="by_score",
        iri="tvbo:functions/GroupedCorrelation",
        arguments={"n_groups": 20, "dim": "subject", "grouping": "equal_count"},
    )
    assert (analysis.name, analysis.callable.module, analysis.callable.name) == (
        "by_score",
        "tvbo.analysis.statistics",
        "grouped_correlation",
    )
    assert analysis.iri == "tvbo:functions/GroupedCorrelation" and analysis.label == "Correlation over group means"
    assert (analysis.arguments["n_groups"].value, analysis.arguments["grouping"].value) == (20, "equal_count")
    assert analysis.arguments["x"].description and analysis.arguments["n_groups"].description
    assert Analysis(name="plain", iri="tvbo:functions/GroupedCorrelation").arguments["grouping"].value == "equal_width"


def test_every_invocation_expands_a_function_iri():
    from tvbo.datamodel import FunctionCall

    assert FunctionCall(iri="tvbo:functions/SetMean").equation.rhs == "sum(values * membership) / sum(membership)"
    validated = Analysis.model_validate({"name": "counted", "iri": "tvbo:functions/Histogram", "arguments": {"n_bins": 40}})
    assert (validated.name, validated.callable.module, validated.callable.name) == (
        "counted",
        "tvbo.analysis.statistics",
        "histogram",
    )
    assert validated.arguments["n_bins"].value == 40 and validated.arguments["values"].description


def test_an_unscoped_iri_names_the_function_by_its_local_name():
    from tvbo.datamodel import FunctionCall

    scoped = Analysis(name="by_score", iri="tvbo:functions/GroupedCorrelation", arguments={"n_groups": 20})
    for iri in ("tvbo:GroupedCorrelation", "GroupedCorrelation"):
        unscoped = Analysis(name="by_score", iri=iri, arguments={"n_groups": 20})
        assert (unscoped.name, unscoped.iri, unscoped.label) == ("by_score", iri, scoped.label)
        assert (unscoped.callable.module, unscoped.callable.name) == ("tvbo.analysis.statistics", "grouped_correlation")
        assert unscoped.arguments["grouping"].value == "equal_width" and unscoped.arguments["n_groups"].value == 20
    assert FunctionCall(iri="tvbo:SetMean").equation.rhs == FunctionCall(iri="tvbo:functions/SetMean").equation.rhs
    validated = Analysis.model_validate({"name": "counted", "iri": "tvbo:Histogram", "arguments": {"n_bins": 40}})
    assert (validated.callable.name, validated.arguments["n_bins"].value) == ("histogram", 40)


def _with_future_warnings(build):
    """What *build* returns, and the text of every ``FutureWarning`` it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built = build()
    return built, [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


def test_every_other_spelling_the_registry_reads_names_the_same_function_and_a_retired_scope_warns_once():
    from tvbo.datamodel import FunctionCall

    canonical = Analysis(name="by_score", iri="tvbo:functions/GroupedCorrelation", arguments={"n_groups": 20})
    singular, said = _with_future_warnings(
        lambda: Analysis(name="by_score", iri="tvbo:function/GroupedCorrelation", arguments={"n_groups": 20})
    )
    assert len(said) == 1 and "write `tvbo:functions/GroupedCorrelation`" in said[0], said
    assert (singular.callable, singular.arguments) == (canonical.callable, canonical.arguments)
    for iri in ("https://w3id.org/tvbo/functions/GroupedCorrelation", "tvbo:functions/groupedcorrelation"):
        spelled = Analysis(name="by_score", iri=iri, arguments={"n_groups": 20})
        assert (spelled.iri, spelled.callable, spelled.arguments) == (iri, canonical.callable, canonical.arguments)
    step, said = _with_future_warnings(
        lambda: FunctionCall(name="step", iri="tvbo:observation/temporal_average_tvb", function="window_mean")
    )
    assert len(said) == 1, said
    assert (step.iri, step.function, step.equation) == ("tvbo:observation/temporal_average_tvb", "window_mean", None)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="holds no Function called 'NoSuchStatistic'"):
        Analysis(name="typo", iri="tvbo:function/NoSuchStatistic")


def test_migrate_iris_writes_an_analysis_that_names_a_function_in_the_folder_form():
    from tvbo.data.iri_migration import migrate_text

    recipe = (
        "analyses:\n  - {name: a, iri: tvbo:GroupedCorrelation}\n  - {name: b, iri: tvbo:function/Histogram}\n"
        "  - {name: c, iri: tvbo:functions/falsediscoverycontrol}\n"
    )
    migrated, changes, skipped = migrate_text(recipe)
    assert migrated == (
        "analyses:\n  - {name: a, iri: tvbo:functions/GroupedCorrelation}\n  - {name: b, iri: tvbo:functions/Histogram}\n"
        "  - {name: c, iri: tvbo:functions/FalseDiscoveryControl}\n"
    )
    assert [change.line for change in changes] == [2, 3, 4] and not skipped


def test_an_iri_naming_no_curated_function_is_refused_where_nothing_else_would_run():
    from tvbo.datamodel import FunctionCall

    with pytest.raises(ValueError, match="holds no Function called 'NoSuchStatistic'"):
        Analysis(name="typo", iri="tvbo:functions/NoSuchStatistic")
    for invocation in (Analysis, FunctionCall):
        for iri in ("tvbo:NoSuchStatistic", "NoSuchStatistic"):
            said = f"{invocation.__name__} names {iri!r}, which matched no curated function, and states no `equation`, `callable`, `class_call` or `function` of its own, so it has nothing to run."
            with pytest.raises(ValueError, match=re.escape(said)):
                invocation(name="typo", iri=iri, arguments={"n_groups": 20})
    with pytest.raises(ValueError, match="names 'tvbo:NoSuchStatistic', which matched no curated function"):
        Analysis.model_validate({"name": "typo", "iri": "tvbo:NoSuchStatistic"})
    solve = Analysis(iri="tvbo:LyapunovAnalysis", type="lyapunov")
    assert (solve.name, solve.iri, solve.type, solve.callable, solve.equation) == (
        "LyapunovAnalysis",
        "tvbo:LyapunovAnalysis",
        "lyapunov",
        None,
        None,
    )


def test_an_unscoped_iri_naming_no_function_beside_a_form_of_ones_own_merges_nothing_and_that_form_runs(tmp_path, rng):
    from tvbo.datamodel import FunctionCall

    values = xr.DataArray(rng.uniform(-1, 1, 60), dims=["item"], coords={"item": np.arange(60)})
    _seed(tmp_path, "values", values)
    own = Analysis(
        name="counted",
        iri="tvbo:NoSuchStatistic",
        callable={"name": "histogram", "module": "tvbo.analysis.statistics"},
        arguments={"values": {"used": {"analysis": "values", "output": "values"}}, "n_bins": 4, "lower": -1.0, "upper": 1.0},
    )
    assert (own.name, own.iri, own.label, own.callable.name) == ("counted", "tvbo:NoSuchStatistic", None, "histogram")
    assert own.arguments["n_bins"].value == 4 and own.arguments["n_bins"].description is None
    analysis_io.run_analysis(own, tmp_path)
    with xr.open_dataset(analysis_io.container_path("counted", tmp_path), engine="h5netcdf") as stored:
        np.testing.assert_array_equal(
            stored["observation__count"].values, np.histogram(values.values, bins=4, range=(-1.0, 1.0))[0]
        )

    assert Analysis(name="summed", iri="tvbo:NoSuchStatistic", equation="x + y").equation.rhs == "x + y"
    assert FunctionCall(name="kernel", iri="NoSuchStatistic", function="hrf_kernel").function == "hrf_kernel"
    reference = FunctionCall(iri="tvbo:NoSuchStatistic", class_call={"name": "Bold", "module": "tvboptim.observations"})
    assert reference.class_call.name == "Bold" and reference.callable is None


@pytest.mark.parametrize("iri", ["tvbo:functions/GroupedCorrelation", "tvbo:GroupedCorrelation"])
def test_naming_a_callable_function_beside_an_equation_of_ones_own_is_refused(iri):
    with pytest.raises(ValueError, match="records a `callable`, and states an `equation` of its own"):
        Analysis(name="both", iri=iri, equation={"rhs": "x + y"})
    with pytest.raises(ValueError, match="records a `callable`, and states an `equation` of its own"):
        Analysis(name="both", iri=iri, equation="x + y")


def test_naming_an_equation_function_beside_a_callable_of_ones_own_is_refused():
    from tvbo.datamodel import FunctionCall

    own = {"name": "pearsonr", "module": "scipy.stats"}
    for invocation in (Analysis, FunctionCall):
        with pytest.raises(ValueError, match="records an `equation`, and states a `callable` of its own"):
            invocation(name="both", iri="tvbo:functions/SetMean", callable=own)


def test_a_software_based_function_runs_as_a_study_analysis(tmp_path, rng):
    subjects = [f"s{i:03d}" for i in range(80)]
    score = xr.DataArray(rng.standard_normal(80), dims=["subject"], coords={"subject": subjects})
    table = xr.DataArray(
        rng.standard_normal((80, 6)) + score.values[:, None],
        dims=["subject", "item"],
        coords={"subject": subjects, "item": np.arange(1, 7)},
    )
    _seed(tmp_path, "score", score)
    _seed(tmp_path, "table", table)

    arguments = {
        "x": {"used": {"analysis": "score", "output": "score"}},
        "y": {"used": {"analysis": "table", "output": "table"}},
        "n_groups": 8,
        "dim": "subject",
    }
    analysis_io.run_analysis(Analysis(name="grouped", iri="tvbo:functions/GroupedCorrelation", arguments=arguments), tmp_path)
    analysis_io.run_analysis(
        Analysis(
            name="kept",
            iri="tvbo:functions/FalseDiscoveryControl",
            arguments={"p": {"used": {"analysis": "grouped", "output": "p"}}, "q": 0.1},
        ),
        tmp_path,
    )
    analysis_io.run_analysis(
        Analysis(
            name="counted",
            iri="tvbo:functions/Histogram",
            arguments={
                "values": {"used": {"analysis": "grouped", "output": "r"}},
                "where": {"used": {"analysis": "kept", "output": "selected"}},
                "n_bins": 4,
                "lower": -1.0,
                "upper": 1.0,
            },
        ),
        tmp_path,
    )

    want = _curated("GroupedCorrelation")(score, table, n_groups=8, dim="subject")
    with xr.open_dataset(analysis_io.container_path("grouped", tmp_path), engine="h5netcdf") as stored:
        assert stored["observation__r"].dims == ("item",) and stored["observation__y_mean"].dims == ("item", "group")
        np.testing.assert_array_equal(stored["observation__r"].values, want["r"].values)
        np.testing.assert_array_equal(stored["observation__x_mean"].values, want["x_mean"].values)
        assert int(stored["observation__n_units_total"]) == 80
    with xr.open_dataset(analysis_io.container_path("counted", tmp_path), engine="h5netcdf") as stored:
        selected = _step_up(want["p"].values, 0.1)
        np.testing.assert_array_equal(
            stored["observation__count"].values, np.histogram(want["r"].values[selected], bins=4, range=(-1.0, 1.0))[0]
        )


def test_an_equation_based_function_runs_as_a_study_analysis(tmp_path, rng):
    a, b = rng.standard_normal((200, 9)), rng.standard_normal((200, 9))
    b[:, :4] += a[:, :4]
    fc1, fc2 = np.corrcoef(a.T), np.corrcoef(b.T)
    for name, fc in (("simulated", fc1), ("empirical", fc2)):
        _seed(tmp_path, name, xr.DataArray(fc, dims=["node", "node_"]))

    arguments = {
        key: {"used": {"analysis": name, "output": name}} for key, name in (("fc1", "simulated"), ("fc2", "empirical"))
    }
    analysis = Analysis(
        name="fit",
        iri="tvbo:functions/FCCorrelation",
        execution={"backend": "tvboptim", "precision": "float64"},
        arguments=arguments,
    )
    analysis_io.run_analysis(analysis, tmp_path)

    upper = np.triu_indices(9, 1)
    with xr.open_dataset(analysis_io.container_path("fit", tmp_path), engine="h5netcdf") as stored:
        assert abs(float(stored["observation__fit"]) - scipy.stats.pearsonr(fc1[upper], fc2[upper]).statistic) < 1e-12
