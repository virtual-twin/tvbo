"""Nonnegative matrix factorization of a cohort's connectomes, reached through its curated records as a study reaches them, against planted factors and an independent transcription of a published notebook.

The reference is cells 7, 14 and 15 of Koller et al.'s released `48_results.ipynb` (doi:10.1038/s41467-024-47860-x), written out here with their variable names: each subject's weights on the lower-triangle pairs of the group connectome as one row of the data matrix, the error and explained variance of the reconstruction ``W d H``, and each component's mean over subjects put back on the group connectome's pairs and mirrored.
"""

import numpy as np
import pytest
import scipy.sparse
import xarray as xr

from tvbo import Function
from tvbo.analysis.connectome import _als, _start
from tvbo.datamodel.schema import Analysis

NODES = [f"r{i:02d}" for i in range(10)]
"""Region labels whose sorted order is their order here, so the factorization's pairs run as the upper triangle does."""


def _curated(name):
    """The callable ``tvbo:functions/<name>`` records."""
    return Function.from_string(f"iri: tvbo:functions/{name}").resolved_callable


def _factorize(weight, mask, **kwargs):
    return _curated("NonnegativeFactorization")(weight, mask, **kwargs)


def _cross_validate(weight, mask, **kwargs):
    return _curated("NonnegativeFactorizationCrossValidation")(weight, mask, **kwargs)


def _stack(values):
    """*values* (subject, node, node) as a cohort read stacks them."""
    return xr.DataArray(
        values,
        dims=("subject", "node_i", "node_j"),
        coords={"subject": [f"sub-{i:03d}" for i in range(values.shape[0])], "node_i": NODES, "node_j": NODES},
    )


def _planted(rng, rank=3, n_subjects=30, noise=0.0):
    """A cohort whose pairs in the returned mask carry ``w_true @ h_true`` plus uniform noise, and whose other pairs carry unrelated weights the factorization must not read."""
    upper = np.triu_indices(len(NODES), 1)
    kept = rng.random(upper[0].size) < 0.8
    w_true = rng.uniform(0.5, 1.5, (kept.sum(), rank)) * (rng.random((kept.sum(), rank)) < 0.5)
    h_true = rng.uniform(0.5, 1.5, (rank, n_subjects)) * (rng.random((rank, n_subjects)) < 0.6)
    w = np.zeros((n_subjects, len(NODES), len(NODES)))
    w[:, upper[0][kept], upper[1][kept]] = (w_true @ h_true + noise * rng.random((kept.sum(), n_subjects))).T
    w[:, upper[0][~kept], upper[1][~kept]] = rng.uniform(0, 5, (n_subjects, (~kept).sum()))
    mask = np.zeros((len(NODES), len(NODES)))
    mask[upper[0][kept], upper[1][kept]] = 1
    mask = xr.DataArray(mask + mask.T, dims=("node_i", "node_j"), coords={"node_i": NODES, "node_j": NODES})
    return _stack(w + w.transpose(0, 2, 1)), mask, w_true, h_true


def _rows(fit, pairs):
    """The rows of the fit's ``w`` for *pairs*, each a pair of labels in either order."""
    index = {
        frozenset(pair): row
        for row, pair in enumerate(zip(fit["w"]["pair_node_i"].values, fit["w"]["pair_node_j"].values, strict=True))
    }
    return fit["w"].values[[index[frozenset(pair)] for pair in pairs]]


def _koller_cells_7_14_15(weights_nsc, avg_weights, W, H, d):
    """Cells 7, 14 and 15 of Koller et al.'s notebook, with their variable names: *W* holds one row per lower-triangle pair of *avg_weights*, in numpy's boolean-mask order."""
    weights_mask = avg_weights != 0
    weights_mask[np.triu_indices_from(avg_weights, k=1)] = 0
    weights_triu = np.array([weights_nsc[i][weights_mask] for i in range(len(weights_nsc))])
    number_of_subjects = weights_triu.shape[0]
    A = W.dot(np.tile(d, (number_of_subjects, 1)).T * H)
    rmse = np.sqrt(np.mean((weights_triu - A.T) ** 2))
    explained_variance = 1 - np.sum((weights_triu - A.T) ** 2) / np.sum((weights_triu - np.mean(weights_triu)) ** 2)
    components = []
    for i in range(len(d)):
        A_comp = np.outer(W[:, i], d[i] * H[i])
        recon = np.zeros_like(avg_weights)
        recon[weights_mask] = np.mean(A_comp, axis=1)
        components.append(recon + recon.T)
    return rmse, explained_variance, np.array(components)


def _rcppml_c_nnls(a, b, h, sample):
    """``c_nnls`` of RcppML 0.5.6 (``inst/include/RcppML/nnls.hpp`` l. 17-36), line for line: column *sample* of *h* solves ``a x = b`` by coordinate descent from its current value."""
    tol, it = 1.0, 0
    while it < 100 and tol / b.size > 1e-8:
        tol = 0.0
        for i in range(h.shape[0]):
            diff = b[i] / a[i, i]
            if -diff > h[i, sample]:
                if h[i, sample] != 0:
                    b -= a[:, i] * -h[i, sample]
                    tol = 1.0
                    h[i, sample] = 0
            elif diff != 0:
                h[i, sample] += diff
                b -= a[:, i] * diff
                tol += abs(diff / (h[i, sample] + 1e-15))
        it += 1


def _rcppml_predict(A, missing, w, h):
    """``predict`` of RcppML 0.5.6 for a dense *A* with the entries *missing* marks left out (``predict.hpp`` l. 244-287), one column of *h* at a time from zero; with nothing missing it is the unmasked branch (l. 181-205) started from zero too."""
    h[:] = 0
    a = w @ w.T
    for i in range(h.shape[1]):
        rows = np.flatnonzero(missing[:, i])
        a_i = a - w[:, rows] @ w[:, rows].T
        a_i[np.diag_indices_from(a_i)] += 1e-15
        _rcppml_c_nnls(a_i, w @ A[:, i] - w[:, rows] @ A[rows, i], h, i)


def _rcppml_fit(A, missing, w, tol, maxit):
    """``nmf::fit`` of RcppML 0.5.6 (``nmf.hpp`` l. 170-190) with its ``scaleH``, ``scaleW``, ``cor`` and ``sortByDiagonal``; returns ``w``, ``d``, ``h`` and RcppML's ``iter``, which a converged fit does not count up for its last update."""
    h, it = np.zeros((w.shape[0], A.shape[1])), 0
    while it < maxit:
        w_it = w.copy()
        _rcppml_predict(A, missing, w, h)
        d = h.sum(axis=1) + 1e-15
        h /= d[:, None]
        _rcppml_predict(A.T, missing.T, h, w)
        d = w.sum(axis=1) + 1e-15
        w /= d[:, None]
        n, x, y = w.size, w.ravel(), w_it.ravel()
        cor = (n * (x @ y) - x.sum() * y.sum()) / np.sqrt((n * (x @ x) - x.sum() ** 2) * (n * (y @ y) - y.sum() ** 2))
        if 1 - cor < tol:
            break
        it += 1
    order = np.argsort(-d, kind="stable")
    return w[order], d[order], h[order], it


@pytest.fixture(scope="module")
def planted():
    return _planted(np.random.default_rng(7), noise=0.05)


@pytest.fixture(scope="module")
def fit(planted):
    weight, mask, _, _ = planted
    return _factorize(weight, mask, rank=3, seed=11, n_starts=4)


def test_the_records_name_the_functions_and_an_analysis_states_only_its_arguments():
    nmf = Analysis(name="subnetworks", iri="tvbo:functions/NonnegativeFactorization", arguments={"rank": 5, "seed": 3})
    cv = Analysis(name="rank_cv", iri="tvbo:functions/NonnegativeFactorizationCrossValidation", arguments={"ranks": [2, 3]})
    assert (nmf.callable.module, nmf.callable.name) == ("tvbo.analysis.connectome", "nonnegative_factorization")
    assert (cv.callable.module, cv.callable.name) == ("tvbo.analysis.connectome", "nonnegative_factorization_cross_validation")
    assert nmf.arguments["rank"].value == 5 and nmf.arguments["n_starts"].value == 1 and nmf.arguments["tol"].value == 1e-4
    assert cv.arguments["held_out"].value == 0.05 and cv.arguments["max_iter"].value == 100
    assert all(a.description for a in [*nmf.arguments.values(), *cv.arguments.values()])


def test_planted_nonnegative_factors_are_recovered():
    weight, mask, w_true, h_true = _planted(np.random.default_rng(3))
    fit = _factorize(weight, mask, rank=3, seed=5, n_starts=3, tol=1e-12, max_iter=3000)
    upper = np.argwhere(np.triu(mask.values, 1))
    w = _rows(fit, [(NODES[i], NODES[j]) for i, j in upper])
    match = np.corrcoef(w.T, w_true.T)[:3, 3:]
    assert sorted(match.argmax(axis=0)) == [0, 1, 2]
    assert match.max(axis=0).min() > 0.9999
    h = fit["h"].sel(subject=weight["subject"]).values
    assert np.corrcoef(h, h_true)[:3, 3:][match.argmax(axis=0), [0, 1, 2]].min() > 0.9999
    assert fit["rmse"] < 1e-5 * np.mean(w_true @ h_true)


def test_the_factors_sum_to_one_and_the_scale_lives_in_d(planted, fit):
    weight, mask, _, _ = planted
    np.testing.assert_allclose(fit["w"].sum("pair"), 1, rtol=1e-12)
    np.testing.assert_allclose(fit["h"].sum("subject"), 1, rtol=1e-12)
    assert (np.diff(fit["d"].values) <= 0).all()
    scaled = _factorize(weight * 7.0, mask, rank=3, seed=11, n_starts=4)
    np.testing.assert_allclose(scaled["d"], 7.0 * fit["d"], rtol=1e-8)
    for key in ("w", "h"):
        np.testing.assert_allclose(scaled[key], fit[key], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(scaled["explained_variance"], fit["explained_variance"], rtol=1e-8)


def test_subjects_and_regions_are_paired_by_label_not_by_position(planted, fit):
    weight, mask, _, _ = planted
    rng = np.random.default_rng(2)
    nodes, subjects = rng.permutation(len(NODES)), rng.permutation(weight.sizes["subject"])
    shuffled = weight.isel(subject=subjects, node_i=nodes, node_j=rng.permutation(len(NODES)))
    again = _factorize(shuffled, mask.isel(node_i=rng.permutation(len(NODES))), rank=3, seed=11, n_starts=4)
    for key in ("w", "d"):
        xr.testing.assert_identical(again[key], fit[key])
    xr.testing.assert_identical(again["h"].sel(subject=fit["h"]["subject"]), fit["h"])
    xr.testing.assert_identical(again["subnetwork"].sel(node_i=NODES, node_j=NODES), fit["subnetwork"])
    assert (again["rmse"], again["explained_variance"]) == (fit["rmse"], fit["explained_variance"])


def test_the_subnetworks_and_errors_equal_kollers_notebook(planted, fit):
    weight, mask, _, _ = planted
    lower = np.argwhere(np.tril(mask.values, -1))
    W = _rows(fit, [(NODES[i], NODES[j]) for i, j in lower])
    H = fit["h"].sel(subject=weight["subject"]).values
    rmse, explained_variance, components = _koller_cells_7_14_15(list(weight.values), mask.values, W, H, fit["d"].values)
    np.testing.assert_allclose(fit["subnetwork"].values, components, rtol=1e-12, atol=0)
    np.testing.assert_allclose([fit["rmse"], fit["explained_variance"]], [rmse, explained_variance], rtol=1e-10)
    assert fit["subnetwork"].dims == ("component", "node_i", "node_j")


@pytest.mark.parametrize("missing_share", [0.0, 0.15])
def test_one_start_equals_a_column_by_column_transcription_of_rcppml(missing_share):
    rng = np.random.default_rng(0)
    A = rng.random((25, 18)) ** 2
    missing = rng.random(A.shape) < missing_share
    start = _start(3, 0, 3, A.shape[0])
    w_ref, d_ref, h_ref, it_ref = _rcppml_fit(A, missing, start.copy(), 1e-4, 100)
    x = np.where(missing, 0.0, A).T.copy()
    w, d, h, iterations = _als(x, start, 1e-4, 100, scipy.sparse.csr_array(missing.T.astype(float)) if missing.any() else None)
    for got, want in ((w, w_ref), (d, d_ref), (h, h_ref)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
    assert iterations == it_ref + 1


def test_the_best_of_several_starts_fits_at_least_as_well_as_the_first(planted, fit):
    weight, mask, _, _ = planted
    assert fit["rmse"] <= _factorize(weight, mask, rank=3, seed=11, n_starts=1)["rmse"]
    assert 1 <= fit["iterations"] <= 100


def test_cross_validation_gives_one_error_per_repetition_and_rank_whatever_the_order_of_ranks(planted):
    weight, mask, _, _ = planted
    cv = _cross_validate(weight, mask, ranks=[3, 1], seed=4, reps=2)
    assert cv["rmse"].dims == ("repetition", "rank")
    assert list(cv["rmse"]["rank"].values) == [1, 3] and list(cv["rmse"]["repetition"].values) == [1, 2]
    assert np.isfinite(cv["rmse"]).all() and (cv["rmse"] > 0).all()
    xr.testing.assert_identical(_cross_validate(weight, mask, ranks=[1, 3], seed=4, reps=2)["rmse"], cv["rmse"])


def test_the_held_out_error_rises_when_the_rank_is_far_too_low():
    weight, mask, _, _ = _planted(np.random.default_rng(9), rank=4, n_subjects=40, noise=0.01)
    rmse = _cross_validate(weight, mask, ranks=[1, 4], seed=8, reps=3, held_out=0.1)["rmse"]
    assert (rmse.sel(rank=1) > 3 * rmse.sel(rank=4)).all()


def _negative(weight, mask):
    w = weight.copy()
    i, j = np.argwhere(np.triu(mask.values, 1))[0]
    w[2, i, j] = w[2, j, i] = -1.0
    return w, mask


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (_negative, "negative weight"),
        (lambda w, m: (w, m * 0), "marks no region pair"),
        (lambda w, m: (w, m + np.eye(len(NODES))), "own connection"),
        (lambda w, m: (w, m.where(m.node_i < m.node_j, 0)), "not symmetric"),
        (lambda w, m: (w + np.triu(np.ones(len(NODES)), 1), m), "not symmetric"),
        (lambda w, m: (w, m.values), "labelled"),
        (lambda w, m: (w.drop_vars("subject"), m), "labelled"),
    ],
    ids=["negative", "empty-mask", "diagonal", "one-sided-mask", "directed-weight", "bare-mask", "unlabelled-subjects"],
)
def test_input_the_factorization_cannot_read_is_refused(planted, make, message):
    weight, mask = make(*planted[:2])
    with pytest.raises(ValueError, match=message):
        _factorize(weight, mask, rank=2, seed=0)
    with pytest.raises(ValueError, match=message):
        _cross_validate(weight, mask, ranks=[2], seed=0, reps=1)


@pytest.mark.parametrize(
    "arguments", [{"rank": 0}, {"rank": None}, {"n_starts": 0}, {"seed": None}, {"seed": -1}, {"max_iter": 2.5}]
)
def test_a_rank_start_count_seed_or_iteration_cap_that_is_not_a_whole_number_is_refused(planted, arguments):
    with pytest.raises(ValueError, match=next(iter(arguments))):
        _factorize(*planted[:2], **{"rank": 2, "seed": 0, **arguments})
