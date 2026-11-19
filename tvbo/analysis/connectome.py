"""Reductions of a cohort's structural connectomes: one group connectome by consistency-based thresholding, and the subnetworks its subjects share by nonnegative matrix factorization."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse
import xarray as xr

from tvbo.data.dataref import SUBJECT_DIM


def consistency_threshold(weight: xr.DataArray, density: float, length: xr.DataArray | None = None) -> dict[str, xr.DataArray]:
    """Group connectome by consistency-based thresholding (Roberts et al. 2017, doi:10.1016/j.neuroimage.2016.09.053): each region pair's mean weight over the subjects, kept for the share *density* of all region pairs whose weights vary least across subjects.

    *weight* holds one symmetric connectome per subject on a ``subject`` axis, as a cohort read stacks them. A pair's variation is its coefficient of variation, the sample standard deviation (ddof 1) over the mean, and only pairs with a nonzero mean compete. The ``round(n * n * density / 2)`` least varying pairs of the upper triangle are kept, with every pair tied with the last of them, and mirrored, so the kept entries are *density* of all ``n * n``. Means and variations count a subject without the connection as zero. *length*, stacked like *weight* and paired with it by subject and node labels, is averaged over all subjects the same way and kept on the same pairs.

    Returns ``{"weight"}``, and ``"length"`` when *length* is given, on *weight*'s two node axes.

    Raises:
        ValueError: If *weight* has no ``subject`` axis beside two node axes of one length, holds fewer than two subjects, or *density* is not in (0, 1].
    """
    nodes = [d for d in weight.dims if d != SUBJECT_DIM]
    if SUBJECT_DIM not in weight.dims or len(nodes) != 2 or weight.sizes[nodes[0]] != weight.sizes[nodes[1]]:
        raise ValueError(
            f"consistency_threshold needs one connectome per subject on a `{SUBJECT_DIM}` axis beside two node axes of one length; got dims {weight.dims}."
        )
    if weight.sizes[SUBJECT_DIM] < 2:
        raise ValueError(
            "consistency_threshold needs at least two subjects, since a single connectome has no variation across subjects."
        )
    if not 0 < density <= 1:
        raise ValueError(f"density is the share of region pairs kept and must lie in (0, 1]; got {density}.")
    weight = weight.transpose(SUBJECT_DIM, *nodes)
    mean, spread = _mean_and_sd(weight)
    n = mean.shape[0]
    upper = np.triu_indices(n, 1)
    competing = mean[upper] != 0
    cv = spread[upper][competing] / mean[upper][competing]
    kept = np.zeros((n, n), dtype=bool)
    n_kept = min(int(np.round(n * n * density / 2)), cv.size)
    if n_kept:
        kept[upper[0][competing], upper[1][competing]] = cv <= np.sort(cv)[n_kept - 1]
    kept |= kept.T
    coords = {d: weight[d].values for d in nodes if d in weight.coords}
    out = {"weight": xr.DataArray(np.where(kept, mean, 0.0), dims=nodes, coords=coords)}
    if length is not None:
        aligned = length.sel(
            {d: weight[d] for d in (SUBJECT_DIM, *nodes) if d in weight.coords and d in length.coords}
        ).transpose(SUBJECT_DIM, *nodes)
        out["length"] = xr.DataArray(np.where(kept, _mean_and_sd(aligned, spread=False)[0], 0.0), dims=nodes, coords=coords)
    return out


def _mean_and_sd(stack: xr.DataArray, spread: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean over the leading axis of *stack* and, when *spread*, its sample standard deviation, in float64 one subject at a time, so a large cohort is never copied whole."""
    k = stack.shape[0]
    total = np.zeros(stack.shape[1:])
    for s in range(k):
        total += stack[s].values
    mean = total / k
    if not spread:
        return mean, None
    squares = np.zeros_like(mean)
    for s in range(k):
        squares += (stack[s].values - mean) ** 2
    return mean, np.sqrt(squares / (k - 1))


TINY = 1e-15
"""RcppML's ``TINY_NUM`` (``inst/include/RcppMLCommon.hpp`` l. 21): added to a Gram matrix's diagonal and to a factor's sums before they divide it."""

CD_TOL = 1e-8
"""A column's coordinate descent stops once the summed relative change of its entries over one sweep, divided by the rank, is at most this (``RcppMLCommon.hpp`` l. 28)."""

CD_MAX_SWEEPS = 100
"""The most sweeps a column's coordinate descent runs (``RcppMLCommon.hpp`` l. 29)."""

START_RANGES = ((0.0, 1.0), (0.0, 2.0), (1.0, 2.0), (1.0, 10.0))
"""The ranges a uniform random start draws ``w`` from, one picked at random per start (``R/nmf.R`` l. 162)."""

_STARTS, _HELD_OUT = 0, 1
"""Keys that give a seed's random starts and its held-out entries a random stream each."""

_CHUNK = 64
"""Subjects per block when a squared error is summed, so the reconstruction is never held for the whole cohort."""


class _Cohort(NamedTuple):
    """The data matrix ``x`` of subjects by kept pairs, both in the order of their sorted labels; ``order`` places each row's subject on *weight*'s ``subject`` axis, and ``first`` and ``second`` give each pair's two regions as positions in the sorted region ``labels`` of the two ``nodes`` axes."""

    x: np.ndarray
    order: np.ndarray
    nodes: tuple[str, str]
    labels: np.ndarray
    first: np.ndarray
    second: np.ndarray


def nonnegative_factorization(
    weight: xr.DataArray, mask: xr.DataArray, rank: int, seed: int, n_starts: int = 1, tol: float = 1e-4, max_iter: int = 100
) -> dict:
    """The subnetworks a cohort's connectomes share, by nonnegative matrix factorization as RcppML's ``nmf`` computes it (DeBruine et al. 2021, doi:10.1101/2021.09.01.458620), and as Koller et al. (2024, doi:10.1038/s41467-024-47860-x) factorized 776 subjects at rank 5 from 100 starts for their Fig. 9.

    Each subject's weights on the region pairs *mask* keeps form one column of the data matrix ``A``, pairs by subjects, approximated as ``A ≈ w diag(d) h`` with nonnegative ``w``, pairs by *rank* components, and ``h``, components by subjects, each component's column of ``w`` and row of ``h`` summing to one and its scale held in ``d``. The fit follows RcppML 0.5.6, the GitHub version Koller et al.'s script installs, read at commit 5449a5b, whose NMF code is unchanged from December 2022 through 2024: alternating least squares, ``h`` from ``w`` and then ``w`` from ``h`` (``inst/include/RcppML/nmf.hpp`` l. 170-190), each a nonnegative least-squares problem per column solved by sequential coordinate descent (Franc et al. 2005, doi:10.1007/11556121_50; ``nnls.hpp`` l. 17-36) under :data:`CD_TOL` and :data:`CD_MAX_SWEEPS`; each factor rescaled to unit sums after its update, the sums kept as ``d`` (``nmf.hpp`` l. 129-144); the updates stop once one minus the Pearson correlation of ``w`` with its value an iteration earlier is below *tol*, or after *max_iter* iterations (``nmf.hpp`` l. 174-184, ``bits.hpp`` l. 79-92); the components ordered by decreasing ``d`` (``nmf.hpp`` l. 113-126). Of *n_starts* random starts the one with the least squared error is kept, the first of equals (``nmf.hpp`` l. 193-225), each drawn as RcppML draws it (:func:`_start`). The draws come from numpy's generator, seeded by *seed* and the start's number, not from R's, so a fit agrees with RcppML's in method and not to the digit. RcppML's path for a matrix without missing entries starts each coordinate descent from the factor's previous, unit-sum value without taking that value's share off the right-hand side (``predict.hpp`` l. 181-205), which adds it to the solution: about 1e-9 of each entry on Koller et al.'s cohort, where a unit-sum ``h`` is near 1/776 and a solved one near 1e6, and more only in the first update of ``w``, which starts from the random start itself. Here every descent starts from zero, as on RcppML's path with missing entries (``predict.hpp`` l. 246), so no start depends on the one before it.

    The pairs are those *mask* marks off its diagonal, each once, and both they and the subjects are taken in the order of their sorted labels, so neither input's axis order changes the result; *weight* and *mask* are paired by label. The data matrix is the one copy of the cohort made, read one subject at a time.

    Returns:
        ``subnetwork`` on ``component`` and *weight*'s node axes, each component's mean over subjects, ``w d mean(h)``, on the kept pairs and their mirror, as Koller et al. rebuild their subnetworks (released ``48_results.ipynb``, cell 15); ``w`` on ``pair`` and ``component``, each pair's two region labels as the coordinates ``pair_<node axis>``; ``h`` on ``component`` and ``subject``; ``d`` on ``component``; ``rmse``, the root mean squared error of ``w diag(d) h`` over all kept pairs and subjects, and ``explained_variance``, one minus its squared error over the squared deviations from the data's mean (cell 14); and ``iterations``, the updates the kept start ran.

    Raises:
        ValueError: If *weight* is not a labelled stack of one symmetric connectome per subject with finite, nonnegative weights on the kept pairs, if *mask* is not labelled by the same regions, not symmetric, marks a region's own connection or marks no pair, or if *rank*, *n_starts* or *max_iter* is not a whole number of at least 1 or *seed* one of at least 0.
    """
    _require_whole(1, rank=rank, n_starts=n_starts, max_iter=max_iter)
    _require_whole(0, seed=seed)
    cohort = _cohort_matrix(weight, mask)
    best = None
    for start in range(n_starts):
        fit = _als(cohort.x, _start(seed, start, rank, cohort.x.shape[1]), tol, max_iter)
        error = _fit_error(cohort.x, *fit[:3])
        if best is None or error < best[0]:
            best = (error, *fit)
    error, w, d, h, iterations = best
    mean = cohort.x.mean()
    on_subjects = np.empty_like(h)
    on_subjects[:, cohort.order] = h
    rows, cols = (weight.indexes[dim].get_indexer(cohort.labels) for dim in cohort.nodes)
    mean_over_subjects = w * (d * h.mean(axis=1))[:, None]
    subnetwork = np.zeros((rank, *(weight.sizes[dim] for dim in cohort.nodes)))
    subnetwork[:, rows[cohort.first], cols[cohort.second]] = mean_over_subjects
    subnetwork[:, rows[cohort.second], cols[cohort.first]] = mean_over_subjects
    component = {"component": np.arange(rank)}
    pairs = {
        f"pair_{dim}": ("pair", cohort.labels[at]) for dim, at in zip(cohort.nodes, (cohort.first, cohort.second), strict=True)
    }
    return {
        "subnetwork": xr.DataArray(
            subnetwork,
            dims=("component", *cohort.nodes),
            coords={**component, **{dim: weight[dim].values for dim in cohort.nodes}},
        ),
        "w": xr.DataArray(w.T, dims=("pair", "component"), coords={**component, **pairs}),
        "h": xr.DataArray(
            on_subjects, dims=("component", SUBJECT_DIM), coords={**component, SUBJECT_DIM: weight[SUBJECT_DIM].values}
        ),
        "d": xr.DataArray(d, dims=("component",), coords=component),
        "rmse": float(np.sqrt(error / cohort.x.size)),
        "explained_variance": float(1 - error / _squared_sum(cohort.x, lambda block, rows: rows - mean)),
        "iterations": iterations,
    }


def nonnegative_factorization_cross_validation(
    weight: xr.DataArray,
    mask: xr.DataArray,
    ranks,
    seed: int,
    reps: int = 3,
    held_out: float = 0.05,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> dict:
    """How well nonnegative factorizations of a cohort's connectomes impute entries held out of the fit, rank by rank, as RcppML's ``crossValidate`` computes it and Koller et al. (2024, doi:10.1038/s41467-024-47860-x) chose their rank with it (``scripts/analysis/nmf_sc.R``).

    In each of *reps* repetitions a share *held_out* of the entries of the data matrix, subjects' weights on the pairs *mask* keeps as :func:`nonnegative_factorization` builds it, is drawn uniformly without replacement and left out of every column's normal equations (``predict.hpp`` l. 244-287); each of *ranks* is fitted to the rest from the first start of *seed* (:func:`_start`), the same in every repetition, as RcppML 0.5.6 fits it (``R/crossValidate.R`` l. 37-52), and the error of the fit's prediction of the held-out entries is recorded (``nmf.hpp`` l. 348-369). RcppML 0.5.6's ``crossValidate`` has no ``method`` argument: the ``method = "impute"`` Koller et al. pass reaches ``nmf`` as an unused development parameter, and holding entries out and imputing them is the one method it has. The held-out entries come from numpy's generator, seeded by *seed* and the repetition's number, so they are not the ones R draws.

    Returns:
        ``rmse`` on ``repetition``, 1 to *reps*, and ``rank``, *ranks* in increasing order: the root of RcppML's mean squared error over the held-out entries, as Koller et al. plot it (``48_results.ipynb`` cell 9).

    Raises:
        ValueError: If *weight* or *mask* cannot be read as :func:`nonnegative_factorization` reads them, if *ranks* is empty, names a rank twice or one that is not a whole number of at least 1, if *reps* or *max_iter* is not a whole number of at least 1 or *seed* one of at least 0, or if *held_out* does not hold out at least one entry and leave the rest.
    """
    ranks = list(ranks) if ranks is not None else []
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError(f"ranks must name each rank compared once; got {ranks}.")
    for rank in ranks:
        _require_whole(1, rank=rank)
    _require_whole(1, reps=reps, max_iter=max_iter)
    _require_whole(0, seed=seed)
    cohort = _cohort_matrix(weight, mask)
    x = cohort.x
    count = round(held_out * x.size) if 0 < held_out < 1 else 0
    if not 0 < count < x.size:
        raise ValueError(
            f"held_out is the share of entries held out, in (0, 1), and must hold out at least one of the {x.size} entries and leave the rest; got {held_out!r}."
        )
    ranks = sorted(ranks)
    rmse = np.empty((reps, len(ranks)))
    for rep in range(reps):
        rows, cols = _held_out(_rng(seed, _HELD_OUT, rep), x.shape, count)
        truth = x[rows, cols]
        x[rows, cols] = 0.0
        held = scipy.sparse.csr_array((np.ones(count), (rows, cols)), shape=x.shape)
        for j, rank in enumerate(ranks):
            w, d, h, _ = _als(x, _start(seed, 0, rank, x.shape[1]), tol, max_iter, held)
            predicted = np.concatenate(
                [(d * h[:, s]) @ w[:, held.indices[held.indptr[s] : held.indptr[s + 1]]] for s in range(x.shape[0])]
            )
            rmse[rep, j] = np.sqrt(np.mean((predicted - truth) ** 2))
        x[rows, cols] = truth
    return {
        "rmse": xr.DataArray(rmse, dims=("repetition", "rank"), coords={"repetition": np.arange(1, reps + 1), "rank": ranks})
    }


def _require_whole(minimum: int, **values) -> None:
    """Refuse any of *values* that is not a whole number of at least *minimum*."""
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be a whole number of at least {minimum}; got {value!r}.")


def _cohort_matrix(weight, mask) -> _Cohort:
    """*weight*'s subjects by the region pairs *mask* keeps, read one subject at a time; refuses an input a factorization cannot read, naming what is wrong."""
    nodes = tuple(d for d in getattr(weight, "dims", ()) if d != SUBJECT_DIM)
    if not isinstance(weight, xr.DataArray) or SUBJECT_DIM not in weight.dims or len(nodes) != 2:
        raise ValueError(
            f"weight must be one connectome per subject, labelled and stacked on a `{SUBJECT_DIM}` axis beside two node axes; got dims {getattr(weight, 'dims', type(weight).__name__)}."
        )
    for dim in (SUBJECT_DIM, *nodes):
        if dim not in weight.indexes or not weight.indexes[dim].is_unique:
            raise ValueError(
                f"weight's `{dim}` axis must be labelled, each label once: subjects and regions are paired by label, never by position."
            )
    labels = np.sort(weight[nodes[0]].values)
    if not isinstance(mask, xr.DataArray) or set(mask.dims) != set(nodes):
        raise ValueError(
            f"mask must be a labelled connectome on weight's node axes {nodes}; got {getattr(mask, 'dims', type(mask).__name__)}."
        )
    for name, array, dim in (("weight", weight, nodes[1]), ("mask", mask, nodes[0]), ("mask", mask, nodes[1])):
        if dim not in array.indexes or not np.array_equal(np.sort(array.indexes[dim].values), labels):
            raise ValueError(
                f"The `{dim}` axis of {name} must be labelled with the regions of weight's `{nodes[0]}` axis, each once."
            )
    marked = np.nan_to_num(mask.sel({dim: labels for dim in nodes}).transpose(*nodes).values) != 0
    if marked.diagonal().any():
        raise ValueError(
            f"mask marks a region's own connection ({labels[marked.diagonal()][0]!r}); the factorization reads pairs of two regions."
        )
    one_sided = np.argwhere(marked != marked.T)
    if one_sided.size:
        a, b = labels[one_sided[0]]
        raise ValueError(f"mask is not symmetric: it marks ({a!r}, {b!r}) but not ({b!r}, {a!r}).")
    first, second = np.nonzero(np.triu(marked, 1))
    if not first.size:
        raise ValueError("mask marks no region pair to factorize.")
    order = np.argsort(weight[SUBJECT_DIM].values, kind="stable")
    rows, cols = (weight.indexes[dim].get_indexer(labels) for dim in nodes)
    stack = weight.transpose(SUBJECT_DIM, *nodes)
    x = np.empty((order.size, first.size))
    for row, subject in enumerate(order):
        connectome = np.asarray(stack[subject].values, dtype=float)
        x[row] = connectome[rows[first], cols[second]]
        mirror = connectome[rows[second], cols[first]]
        for problem, bad in (
            ("has a non-finite weight", ~np.isfinite(x[row])),
            ("has a negative weight", x[row] < 0),
            ("is not symmetric", x[row] != mirror),
        ):
            if bad.any():
                at = np.argmax(bad)
                raise ValueError(
                    f"The connectome of subject {weight[SUBJECT_DIM].values[subject]!r} {problem} on the pair ({labels[first[at]]!r}, {labels[second[at]]!r}): {x[row, at]} one way and {mirror[at]} the other. The factorization reads finite, nonnegative weights of symmetric connectomes."
                )
    return _Cohort(x, order, nodes, labels, first, second)


def _rng(seed: int, *key: int) -> np.random.Generator:
    """The random stream *key* of *seed*, independent of every other key's."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def _start(seed: int, start: int, rank: int, size: int) -> np.ndarray:
    """Random start *start* of *seed*: ``w``, *rank* by *size*, uniform over a range picked from :data:`START_RANGES` for the first start and, on a fair coin, for a later one, otherwise normal with mean 2 and standard deviation 1, as RcppML draws its starts (``R/nmf.R`` l. 153-174)."""
    rng = _rng(seed, _STARTS, start)
    if start == 0 or rng.random() < 0.5:
        low, high = START_RANGES[rng.integers(len(START_RANGES))]
        return rng.uniform(low, high, (rank, size))
    return rng.normal(2.0, 1.0, (rank, size))


def _held_out(rng: np.random.Generator, shape: tuple[int, int], count: int) -> tuple[np.ndarray, np.ndarray]:
    """*count* entries of a matrix of *shape*, drawn uniformly without replacement, as row and column indices in row-major order: how many fall in each row is drawn first, from the multivariate hypergeometric distribution a uniform draw gives, and then which, row by row, so no array the size of the matrix is made."""
    per_row = rng.multivariate_hypergeometric(np.full(shape[0], shape[1]), count)
    cols = np.concatenate([np.sort(rng.choice(shape[1], k, replace=False)) for k in per_row])
    return np.repeat(np.arange(shape[0]), per_row), cols


def _als(x: np.ndarray, w: np.ndarray, tol: float, max_iter: int, held=None) -> tuple:
    """One start of RcppML's alternating least squares on the data matrix ``A = x.T`` from the start *w* (``nmf.hpp`` l. 170-190), leaving out the entries of *x* that the sparse *held* marks, which the caller has set to zero: returns ``w``, ``d`` and ``h`` with the components ordered by decreasing ``d``, and the number of updates run."""
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        previous = w
        h, _ = _unit_sums(_project(w, x.T, held))
        w, d = _unit_sums(_project(h, x, None if held is None else held.T))
        if 1 - np.corrcoef(w.ravel(), previous.ravel())[0, 1] < tol:
            break
    order = np.argsort(-d, kind="stable")
    return w[order], d[order], h[order], iterations


def _unit_sums(factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """*factor* with each row divided by its sum plus :data:`TINY`, and those divisors (``nmf.hpp`` l. 129-144)."""
    sums = factor.sum(axis=1) + TINY
    return factor / sums[:, None], sums


def _project(f: np.ndarray, y: np.ndarray, held=None) -> np.ndarray:
    """The nonnegative ``g`` that best fits ``y`` as ``f.T @ g``, one column at a time, as RcppML's ``predict`` solves it (``predict.hpp`` l. 178-287); with *held*, holding one row per column of ``g``, the entries it marks are left out of that column's normal equations, their values in *y* being zero."""
    k = f.shape[0]
    gram = (f @ f.T)[:, :, None]
    if held is not None:
        gram = np.repeat(gram, held.shape[0], axis=2)
        for i, j in zip(*np.triu_indices(k), strict=True):
            gram[i, j] -= held @ (f[i] * f[j])
            gram[j, i] = gram[i, j]
    gram[range(k), range(k)] += TINY
    return _nnls(gram, f @ y)


def _nnls(gram: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The nonnegative ``x`` solving ``gram x = b`` column by column, by RcppML's sequential coordinate descent from zero (``nnls.hpp`` l. 17-36): a coordinate that would turn negative is set to zero, and a column stops after a sweep whose summed relative change, divided by the rank, is at most :data:`CD_TOL` (a zeroed coordinate counts as 1), or after :data:`CD_MAX_SWEEPS`. *gram* is one matrix for every column, shaped ``(k, k, 1)``, or one per column, ``(k, k, n)``; *b* is overwritten."""
    k = b.shape[0]
    x = np.zeros_like(b)
    active = np.arange(b.shape[1])
    for _ in range(CD_MAX_SWEEPS):
        g = gram if gram.shape[2] == 1 else gram[:, :, active]
        xa, residual = x[:, active], b[:, active]
        change = np.zeros(active.size)
        for i in range(k):
            step = residual[i] / g[i, i]
            clipped = -step > xa[i]
            delta = np.where(clipped, -xa[i], step)
            xa[i] += delta
            residual -= g[:, i] * delta
            change = np.where(clipped, np.where(delta != 0, 1.0, change), change + np.abs(delta) / (xa[i] + TINY))
        x[:, active], b[:, active] = xa, residual
        active = active[change / k > CD_TOL]
        if not active.size:
            break
    return x


def _fit_error(x: np.ndarray, w: np.ndarray, d: np.ndarray, h: np.ndarray) -> float:
    """The squared error of ``x`` as ``(d h).T @ w``, summed a block of subjects at a time."""
    return _squared_sum(x, lambda block, rows: rows - (d[:, None] * h[:, block]).T @ w)


def _squared_sum(x: np.ndarray, residual) -> float:
    """The sum of ``residual(block, x[block]) ** 2`` over blocks of :data:`_CHUNK` rows of *x*."""
    return sum(
        float(np.sum(residual(block, x[block]) ** 2)) for block in (slice(i, i + _CHUNK) for i in range(0, len(x), _CHUNK))
    )
