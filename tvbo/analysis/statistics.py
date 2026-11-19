"""Statistics over stored, labelled arrays: the implementations behind the software-based curated functions.

Each function here is named by a record in ``tvbo/database/functions/`` (``SpreadBand``, ``Correlation``, ``GroupedCorrelation``, ``ConcordanceCorrelation``, ``FalseDiscoveryControl``, ``Histogram``, ``PhaseLockingValue``, ``PhaseLagIndex``, ``AtOptimum``, ``HierarchicalClustering``) and returns a labelled array or a mapping of them, which a study-level analysis stores under its own name or one variable per key. The correlations, the counts and the false discovery control run along one named dimension and keep every other one, so a statistic per question, region or pair is one call on the whole array rather than a loop in the study; the phase-synchrony measures pair the nodes of the one other, and the value at an optimum is searched over every dimension of its score.
"""

from __future__ import annotations

import numpy as np

GROUPINGS = ("equal_width", "equal_count")
"""How units are cut into groups: bins of equal width over the grouping variable's range, or runs of equal size along its order."""

CORRELATIONS = ("pearson", "spearman")
"""The correlation coefficients: Pearson's over the values, and Spearman's, which is Pearson's over their ranks."""

MIN_POINTS = 3
"""The fewest points a correlation is stated for; two points always correlate perfectly, so their r and p say nothing."""

STATISTICS = ("mean_sd", "median_iqr")
"""What a spread band draws: the mean with one standard deviation either side, or the median with the 25th and 75th percentiles."""

METHODS = ("bh", "by")
"""The false-discovery procedures: Benjamini-Hochberg, and Benjamini-Yekutieli for tests of arbitrary dependence."""

DIRECTIONS = ("max", "min")
"""Which end of a score its optimum lies at: the largest score or the smallest."""


def group_index(values, n_groups: int, grouping: str = "equal_width") -> np.ndarray:
    """The group, ``0 .. n_groups - 1``, of each entry of the finite 1-D *values*.

    ``equal_width`` cuts the range from the smallest to the largest value into *n_groups* bins of equal width, each closed below and open above and the last closed on both sides; a bin no value falls in stays empty. ``equal_count`` orders the values, ties in order of appearance, and cuts that order into *n_groups* consecutive runs whose sizes differ by at most one, the longer runs first, as :func:`numpy.array_split` does; equal values can then land in neighbouring groups.
    """
    values = np.asarray(values, dtype=float)
    if grouping == "equal_width":
        edges = np.linspace(values.min(), values.max(), n_groups + 1)
        return np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_groups - 1)
    if grouping == "equal_count":
        index = np.empty(values.size, dtype=int)
        for k, members in enumerate(np.array_split(np.argsort(values, kind="stable"), n_groups)):
            index[members] = k
        return index
    raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}.")


def _clusters(tree: np.ndarray, order: np.ndarray, cut: float) -> np.ndarray:
    """The cluster of each leaf of the linkage matrix *tree*, in the dendrogram order *order*, the tree cut at *cut* times its highest merge.

    A cluster is the subtree under a merge that lies below the cut where the merge above it, if there is one, does not, as :func:`scipy.cluster.hierarchy.dendrogram` colours the links below its ``color_threshold``; a merge at the cut is not below it. Clusters are numbered 1, 2, ... in the order their leaves come in the dendrogram from the left, the order in which scipy gives them the colours ``C1``, ``C2``, ..., and a leaf that no merge below the cut joins to another is 0, scipy's ``C0``. The numbers do not wrap round after nine as scipy's colours do.
    """
    n = len(order)
    threshold = cut * tree[:, 2].max()
    top = np.full(2 * n - 1, -1)  # each node's highest ancestor or self below the threshold, -1 for none
    for merge in range(n - 2, -1, -1):
        node = n + merge
        if top[node] < 0 and tree[merge, 2] < threshold:
            top[node] = node
        top[tree[merge, :2].astype(int)] = top[node]
    tops = top[order].tolist()
    number = {node: k for k, node in enumerate(dict.fromkeys(t for t in tops if t >= 0), start=1)}
    return np.array([number.get(t, 0) for t in tops])


def hierarchical_clustering(features, dim: str, cut: float) -> dict:
    """Ward hierarchical clustering of the units of *dim* by their correlation profiles, the tree cut at *cut* times its highest merge.

    *features* holds one row per unit of *dim* and one column per feature along its one other dimension. The rows are correlated over the features (Pearson's r, :func:`numpy.corrcoef`), each unit is described by its row of that correlation matrix, and Ward's linkage on the Euclidean distances between those rows builds the tree: :func:`scipy.cluster.hierarchy.linkage` with ``method="ward"`` on the correlation matrix, as Koller et al. (2024, doi:10.1038/s41467-024-47860-x) cluster the components of their NMF fits (released ``scripts/results/48_results.ipynb``, cells 40 and 47). The rows are put in the order of their labels along *dim* before anything is computed, positions 0, 1, ... standing in for labels where *dim* has none, so the result, a tie between two merges included, depends on the labels and never on the order the rows come in. The clusters are cut from the tree as :func:`_clusters` states; 0.7 is the multiple below which :func:`scipy.cluster.hierarchy.dendrogram` colours by default, and the one Koller et al.'s two clusters come from (cell 49).

    A row that is constant or holds a value that is not finite has no correlation and is refused by its label, as are fewer than two units, fewer than :data:`MIN_POINTS` features, a dimension beyond the two, and a *cut* that is not one finite number above 0.

    Returns:
        ``correlation`` on *dim* and *dim* with ``_j`` appended, both labelled and in dendrogram order, so that a heatmap of it shows the clustered matrix; ``linkage``, scipy's linkage matrix of the label-sorted rows on ``merge`` and ``field`` (``first``, ``second``, ``height``, ``size``), whose leaf ids are positions among the sorted labels and whose merge k forms the id n + k; ``leaf_order``, each unit's position in the dendrogram from the left; and ``cluster``, each unit's cluster (:func:`_clusters`). The last two run along *dim* in dendrogram order.
    """
    import xarray as xr
    from scipy.cluster.hierarchy import leaves_list, linkage

    level = np.asarray(cut)
    if level.ndim or level.dtype.kind not in "fiu" or not 0 < float(level) < np.inf:
        raise ValueError(
            f"`cut` is the height the tree is cut at as a multiple of its highest merge, one finite number above 0, got {cut!r}."
        )
    features = _along(features, dim, "features")
    if features.ndim != 2:
        raise ValueError(
            f"`features` holds one row per unit of {dim!r} and one column per feature, two dimensions; its dimensions are {list(features.dims)}."
        )
    if dim not in features.indexes:
        features = features.assign_coords({dim: np.arange(features.sizes[dim])})
    rows = features.sortby(dim).transpose(dim, ...)
    labels, values = rows.indexes[dim], np.asarray(rows.values, dtype=float)
    if values.shape[0] < 2 or values.shape[1] < MIN_POINTS:
        raise ValueError(
            f"`features` needs at least two units along {dim!r} and {MIN_POINTS} features to correlate them over; it has {values.shape[0]} and {values.shape[1]}."
        )
    defined = np.isfinite(values).all(axis=1) & (values.max(axis=1) > values.min(axis=1))
    if not defined.all():
        raise ValueError(
            f"`features` has no correlation for {_some(labels[~defined])} along {dim!r}: a row that is constant or holds a value that is not finite correlates with nothing."
        )
    correlation = np.corrcoef(values)
    tree = linkage(correlation, method="ward")
    order = leaves_list(tree)
    leaves = labels.values[order]
    return {
        "correlation": xr.DataArray(
            correlation[np.ix_(order, order)], dims=[dim, f"{dim}_j"], coords={dim: leaves, f"{dim}_j": leaves}
        ),
        "linkage": xr.DataArray(tree, dims=["merge", "field"], coords={"field": ["first", "second", "height", "size"]}),
        "leaf_order": xr.DataArray(np.arange(len(order)), dims=[dim], coords={dim: leaves}),
        "cluster": xr.DataArray(_clusters(tree, order, float(level)), dims=[dim], coords={dim: leaves}),
    }


def _some(labels, limit: int = 5) -> str:
    """The first *limit* of *labels* as text, with how many more there are."""
    shown = ", ".join(repr(label) for label in list(labels)[:limit])
    return shown if len(labels) <= limit else f"{shown} and {len(labels) - limit} more"


def _along(value, dim: str, key: str):
    """*value* as a labelled array running along *dim*: a labelled one as it is, a bare 1-D one with *dim* as its only axis.

    The labels of *dim* name the units and inputs are paired on them, so a labelled array that carries one of them twice is refused.
    """
    import xarray as xr

    if isinstance(value, xr.DataArray):
        if dim not in value.dims:
            raise ValueError(f"`{key}` does not run along {dim!r}; its dimensions are {list(value.dims)}.")
        labels = value.indexes.get(dim)
        if labels is not None and not labels.is_unique:
            raise ValueError(
                f"`{key}` carries {_some(labels[labels.duplicated()].unique())} more than once along {dim!r}; its labels name the units, so each occurs once."
            )
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(
            f"`{key}` is a bare array of rank {array.ndim}; only a labelled array can say which of its axes is {dim!r}."
        )
    return xr.DataArray(array, dims=[dim])


def _paired_along(first, second, dim: str, names: tuple[str, str]) -> tuple:
    """*first* and *second*, called *names* in a refusal, as labelled arrays along *dim* (:func:`_along`), entry for entry.

    Where both label *dim*, *second* is put in the order of *first*'s labels, and a label only one of them carries is refused (:func:`_same_labels`); otherwise the two are paired by position.
    """
    first, second = _along(first, dim, names[0]), _along(second, dim, names[1])
    if dim in first.indexes and dim in second.indexes:
        _same_labels(first, second, dim, names)
        second = second.reindex({dim: first.indexes[dim]})
    return first, second


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """The 1-based rank of each of the 1-D *values*, tied values given the mean of the ranks they span, as :func:`scipy.stats.rankdata` ranks by default (``method="average"``)."""
    order = np.argsort(values)
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return ranks


def _coefficient(a: np.ndarray, b: np.ndarray, method: str) -> tuple:
    """``(r, n)`` of the 1-D *a* and *b* over the *n* pairs where both are finite: Pearson's r of the values, or of their average ranks for ``spearman``, NaN where :func:`_correlates` finds no correlation defined."""
    kept = np.isfinite(a) & np.isfinite(b)
    a, b = a[kept], b[kept]
    if not _correlates(a, b):
        return np.nan, a.size
    if method == "spearman":
        a, b = _average_ranks(a), _average_ranks(b)
    da, db = a - a.mean(), b - b.mean()
    return np.sum(da * db) / np.sqrt(np.sum(da**2) * np.sum(db**2)), a.size


def correlation(x, y, dim: str, method: str) -> dict:
    """Pearson's or Spearman's correlation coefficient of *x* and *y* over the units of *dim*.

    ``pearson`` correlates the values and ``spearman`` their ranks, tied values given the mean of the ranks they span, so ``spearman`` is the coefficient :func:`scipy.stats.spearmanr` returns, ties included. A unit enters where both *x* and *y* are finite: a NaN on either side drops that pair only, and the ranks are taken over the pairs that enter, where :func:`scipy.stats.spearmanr` and :func:`scipy.stats.pearsonr` at their defaults return NaN for a slice holding any NaN. Labelled inputs are paired on the labels of *dim*, which occur once each and which both carry (:func:`_paired_along`); bare 1-D inputs are paired by position. Every other dimension an input carries is kept, with one coefficient per element of it, computed from that element's own pairs; one that both carry is aligned by xarray's exact join, which refuses labels that differ in content or order. A coefficient is NaN with fewer than :data:`MIN_POINTS` pairs or a constant side, as for every correlation here (:func:`_correlates`), and carries none of the inputs' attributes, such as their units.

    Returns:
        ``r``, the coefficient, and ``n``, the number of pairs it was computed from, both labelled on the inputs' other dimensions.
    """
    import xarray as xr

    if method not in CORRELATIONS:
        raise ValueError(f"method must be one of {CORRELATIONS}, got {method!r}.")
    x, y = _paired_along(x, y, dim, ("x", "y"))
    r, n = xr.apply_ufunc(
        _coefficient,
        x,
        y,
        kwargs={"method": method},
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, int],
        keep_attrs=False,
    )
    return {"r": r, "n": n}


def _correlates(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether a correlation of the paired *a* and *b* is defined: at least :data:`MIN_POINTS` points and neither side constant."""
    return a.size >= MIN_POINTS and np.ptp(a) > 0 and np.ptp(b) > 0


def spread_band(values, dim: str, statistic: str, ddof=None) -> dict:
    """The centre of *values* along *dim* and a band around it, the pair a ``band`` or ``errorbar`` layer draws.

    ``mean_sd`` is the mean with the band from one standard deviation below it to one above, the squared deviations divided by ``n - ddof`` for the *n* values counted. *ddof* is required with it, since both forms are in use: 0 is the population form :func:`numpy.nanstd` computes by default, 1 the sample form of pandas and seaborn. ``median_iqr`` is the median with the band from the 25th to the 75th percentile, each read at rank ``p (n - 1)`` of the sorted values and interpolated linearly between its two neighbours, numpy's default ``method="linear"``; it refuses a *ddof*. NaN entries are left out, as :func:`numpy.nanmean`, :func:`numpy.nanstd`, :func:`numpy.nanmedian` and :func:`numpy.nanpercentile` leave them out, and a slice with no other entry gives NaN. A bare 1-D *values* runs along *dim*; a labelled one keeps every other dimension with its labels.

    Returns:
        ``center`` on the other dimensions of *values*, and ``band`` on them and a last dimension ``bound`` labelled ``lower`` and ``upper``, the order in which the ``band`` and ``errorbar`` marks read the two edges.
    """
    import warnings

    import xarray as xr

    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}.")
    if statistic == "mean_sd" and ddof is None:
        raise ValueError(
            "`mean_sd` needs `ddof`: 0 divides the squared deviations by n, as numpy does, and 1 by n - 1, as pandas and seaborn do."
        )
    if statistic == "median_iqr" and ddof is not None:
        raise ValueError(f"`median_iqr` takes no `ddof`, got {ddof!r}; a percentile has no degrees of freedom.")
    values = _along(values, dim, "values")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a slice of nothing but NaN gives NaN, not a warning
        if statistic == "mean_sd":
            center = values.reduce(np.nanmean, dim)
            sd = values.reduce(np.nanstd, dim, ddof=ddof)
            edges = [center - sd, center + sd]
        else:
            center = values.reduce(np.nanmedian, dim)
            edges = [values.reduce(np.nanpercentile, dim, q=q) for q in (25, 75)]
    band = xr.concat(edges, dim="bound").assign_coords(bound=["lower", "upper"]).transpose(..., "bound")
    return {"center": center, "band": band}


def grouped_correlation(x, y, n_groups: int, dim: str, by=None, grouping: str = "equal_width") -> dict:
    """The correlation of *x* with *y* over group means, the units of *dim* cut into *n_groups* groups, beside the same correlation over single units.

    The units are grouped by *by*, or by *x* when no *by* is given (:func:`group_index`), and each group contributes the mean of its members' *x* and of their *y*. A unit enters where *x*, *y* and *by* are all finite; labelled inputs are paired on the labels of *dim*, each of which names one unit, and only units every input has are kept; bare 1-D inputs are paired by position. Any other dimension an input carries is kept: the statistic is computed once per element of those dimensions, each from its own finite units, so *y* on ``(subject, question)`` gives one result per question, and a *by* that carries ``question`` groups each question by its own slice.

    A correlation, its p-value and the line are NaN where they are undefined: fewer than :data:`MIN_POINTS` groups with members (or units, for the unit-level pair), or a constant *x* or *y*. More groups than units is no error: ``equal_count`` then gives each unit a group of its own and leaves the last groups empty.

    Returns:
        On ``group``, one entry per group in the order of the grouping variable, NaN for a group without members: ``x_mean``, ``y_mean``, ``y_sd`` (sample standard deviation of the members' *y*, NaN for a single member), ``y_se`` (``y_sd`` over the square root of the group's size) and ``size``. Over the group means: ``slope`` and ``intercept`` of the least-squares line of ``y_mean`` on ``x_mean``, its Pearson ``r`` and two-sided ``p``, and the line at the smallest and the largest ``x_mean`` as ``fit_x`` and ``fit_y`` on ``end``. Over single units: ``r_unit`` and ``p_unit``. Counts: ``n_nonempty_groups``, ``n_units`` (units entering) and ``n_units_total`` (units entering at least one element of the other dimensions, a scalar).
    """
    import xarray as xr
    from scipy import stats

    n_groups = int(n_groups)
    if n_groups < 1:
        raise ValueError(f"n_groups must be at least 1, got {n_groups}.")
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}.")
    given = {"x": x, "y": y} if by is None else {"x": x, "y": y, "by": by}
    arrays = xr.broadcast(*xr.align(*(_along(value, dim, key) for key, value in given.items()), join="inner"))
    others = [d for d in arrays[0].dims if d != dim]
    xs, ys, *rest = (np.asarray(a.transpose(*others, dim).values, dtype=float) for a in arrays)
    bs = rest[0] if rest else xs
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(bs)
    shape = xs.shape[:-1]
    group = {key: np.full((*shape, n_groups), np.nan) for key in ("x_mean", "y_mean", "y_sd", "y_se")}
    size = np.zeros((*shape, n_groups), dtype=int)
    line = {key: np.full((*shape, 2), np.nan) for key in ("fit_x", "fit_y")}
    scalar = {key: np.full(shape, np.nan) for key in ("slope", "intercept", "r", "p", "r_unit", "p_unit")}

    for at in np.ndindex(*shape):
        ok = finite[at]
        if not ok.any():
            continue
        ux, uy = xs[at][ok], ys[at][ok]
        index = group_index(bs[at][ok], n_groups, grouping)
        for k in range(n_groups):
            members = np.flatnonzero(index == k)
            if not members.size:
                continue
            sd = uy[members].std(ddof=1) if members.size > 1 else np.nan
            size[at][k] = members.size
            group["x_mean"][at][k], group["y_mean"][at][k] = ux[members].mean(), uy[members].mean()
            group["y_sd"][at][k], group["y_se"][at][k] = sd, sd / np.sqrt(members.size)
        if not _correlates(ux, uy):
            continue
        scalar["r_unit"][at], scalar["p_unit"][at] = stats.pearsonr(ux, uy)
        gx, gy = group["x_mean"][at][size[at] > 0], group["y_mean"][at][size[at] > 0]
        if _correlates(gx, gy):
            slope, intercept, r, p, _ = stats.linregress(gx, gy)
            ends = np.array([gx.min(), gx.max()])
            line["fit_x"][at], line["fit_y"][at] = ends, intercept + slope * ends
            scalar["slope"][at], scalar["intercept"][at], scalar["r"][at], scalar["p"][at] = slope, intercept, r, p

    coords = {name: c for a in arrays for name, c in a.coords.items() if c.dims and set(c.dims) <= set(others)}

    def labelled(values, *extra):
        return xr.DataArray(values, dims=[*others, *extra], coords=coords)

    return {
        **{key: labelled(values, "group") for key, values in group.items()},
        "size": labelled(size, "group"),
        **{key: labelled(values, "end") for key, values in line.items()},
        **{key: labelled(values) for key, values in scalar.items()},
        "n_nonempty_groups": labelled((size > 0).sum(axis=-1)),
        "n_units": labelled(finite.sum(axis=-1)),
        "n_units_total": int(finite.reshape(-1, finite.shape[-1]).any(axis=0).sum()),
    }


def concordance_correlation(x, y, dim: str):
    """Lin's concordance correlation coefficient of *x* and *y* over the units of *dim* (Lin 1989, doi:10.2307/2532051).

    It is ``2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))**2)`` with population moments, numpy's default (``ddof=0``): the Pearson correlation scaled down by how far the two differ in mean and spread, so it is 1 only where *y* equals *x* unit for unit and -1 where *y* is *x* reflected about their common mean. Labelled inputs are paired on the labels of *dim*, which occur once each and which both carry; bare 1-D inputs are paired by position. Every other dimension an input carries is kept, with one coefficient per element of it. A NaN unit makes the coefficient NaN, as it makes numpy's moments NaN.

    Returns:
        The coefficient, labelled on the inputs' other dimensions.
    """
    x, y = _paired_along(x, y, dim, ("x", "y"))
    mean_x, mean_y = x.mean(dim, skipna=False), y.mean(dim, skipna=False)
    covariance = ((x - mean_x) * (y - mean_y)).mean(dim, skipna=False)
    spread = x.var(dim, skipna=False) + y.var(dim, skipna=False) + (mean_x - mean_y) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * covariance / spread


def at_optimum(score, value, direction: str) -> dict:
    """The entry of *value* at the cell where *score* is largest (``max``) or smallest (``min``), with the score there and the cell's coordinates.

    The optimum is searched over every dimension of *score*. *value* carries each of them with the same labels, in any order, and keeps its other dimensions; a dimension without labels is labelled by position, as a reference's ``sel`` reads it (:func:`tvbo.data.dataref.index_bare_axes`). NaN scores are ignored, and a score that is NaN in every cell is refused, naming it. Of several cells with the optimal score the one with the smallest coordinates is taken, compared dimension by dimension in the sorted order of the dimensions' names, so neither the order of the dimensions nor that of the labels along them changes the cell; on a score without NaN whose dimensions are in that order and whose labels ascend, it is the cell :func:`numpy.argmax` or :func:`numpy.argmin` finds first.

    Returns:
        ``value`` at the optimum, on *value*'s other dimensions; ``score``, the optimal score; and one output per dimension of *score*, named by the dimension, holding its coordinate at the optimum.
    """
    import xarray as xr

    from tvbo.data.dataref import index_bare_axes

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}.")
    for key, array in (("score", score), ("value", value)):
        if not isinstance(array, xr.DataArray):
            raise ValueError(
                f"`{key}` is a bare array; only a labelled array names the dimensions an optimum is searched over."
            )
    taken = [d for d in score.dims if d in ("value", "score")]
    if taken:
        raise ValueError(f"`score` runs along {taken}, the name of another output; rename that dimension.")
    for d in score.dims:
        _along(score, d, "score")
        _along(value, d, "value")
    searched = dict.fromkeys(score.dims)
    score, value = index_bare_axes(score, searched), index_bare_axes(value, searched)
    for d in searched:
        _same_labels(score, value, d, ("score", "value"))
    scores = np.asarray(score.values, dtype=float)
    if np.isnan(scores).all():
        named = f" ({score.name})" if score.name else ""
        raise ValueError(f"`score`{named} has no cell that is not NaN, so it has no optimum.")
    best = np.nanmax(scores) if direction == "max" else np.nanmin(scores)
    cells = ({d: score.indexes[d][i] for d, i in zip(score.dims, cell, strict=True)} for cell in np.argwhere(scores == best))
    order = sorted(searched)
    at = min(cells, key=lambda cell: tuple(cell[d] for d in order))
    return {"value": value.sel(at), "score": score.sel(at), **at}


def _rate(q) -> float | None:
    """The false discovery rate *q* as a number, or ``None`` when none is given.

    A rate is one number above 0 and below 1, both ends excluded. At 1 the bound holds for any selection, so nothing is controlled, and the Benjamini-Hochberg rule then keeps every test, as it would for any larger value; at 0 only a p-value of exactly 0 could be kept. NaN, a boolean, text and an array of any length, one entry included, are refused with the two ends and everything beyond them.
    """
    if q is None:
        return None
    value = np.asarray(q)
    number = value.ndim == 0 and value.dtype.kind in "fiu"
    if number and 0.0 < float(value) < 1.0:
        return float(value)
    given = repr(float(value)) if number else f"an array of shape {value.shape}" if value.ndim else repr(q)
    raise ValueError(f"`q` is the rate false discoveries are held to, one number above 0 and below 1, got {given}.")


def _control_family(p: np.ndarray, q, method: str) -> tuple[np.ndarray, np.ndarray]:
    """``(adjusted, selected)`` for one family of p-values, its finite entries the tests and a NaN neither adjusted nor selected."""
    adjusted, selected = np.full(p.shape, np.nan), np.zeros(p.shape, dtype=bool)
    tests = np.flatnonzero(np.isfinite(p))
    if not tests.size:
        return adjusted, selected
    order = tests[np.argsort(p[tests])]
    rank = np.arange(1, tests.size + 1)
    penalty = np.sum(1 / rank) if method == "by" else 1.0
    adjusted[order] = np.clip(np.minimum.accumulate((p[order] * (tests.size / rank) * penalty)[::-1])[::-1], 0.0, 1.0)
    if q is not None:
        below = np.flatnonzero(p[order] <= q * rank / tests.size / penalty)
        selected[order[: below.max() + 1 if below.size else 0]] = True
    return adjusted, selected


def false_discovery_control(p, q=None, method: str = "bh", dim=None) -> dict:
    """The p-values *p* adjusted to control the false discovery rate, and the tests kept at rate *q*.

    The entries of a family that are not NaN are its tests; a NaN is no test, stays NaN, is never selected and is not counted, which is how a test that could not be made is passed in. Any other entry is a probability, and one outside 0 to 1, an infinite one included, is refused. The rate *q* is one number above 0 and below 1 (:func:`_rate`): at 1 nothing would be controlled and at 0 nothing could be kept, so both ends are refused with every value beyond them. A family is all of *p*, or, with *dim*, each run along that dimension, one family per element of the others. ``bh`` is the Benjamini-Hochberg step-up procedure, valid for independent or positively dependent tests, and ``by`` the Benjamini-Yekutieli one, valid under any dependence.

    Returns:
        ``p_adjusted``, shaped and labelled as *p*: the smallest rate at which each test is selected, as :func:`scipy.stats.false_discovery_control` gives it. ``n_tests``, the size of each family. With *q*, ``selected``, true for every test up to the largest p-value that lies at or below ``q`` times its rank over the number of tests, and ``n_selected``, their count per family.
    """
    import xarray as xr

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    q = _rate(q)
    labelled = p if isinstance(p, xr.DataArray) else xr.DataArray(np.asarray(p, dtype=float))
    if dim is not None and dim not in labelled.dims:
        raise ValueError(f"`p` does not run along {dim!r}; its dimensions are {list(labelled.dims)}.")
    values = np.asarray(labelled.values, dtype=float)
    invalid = ~np.isnan(values) & ~((values >= 0) & (values <= 1))
    if invalid.any():
        first = tuple(int(i) for i in np.argwhere(invalid)[0])
        place = ", ".join(
            f"{d}={labelled.indexes[d][i : i + 1].tolist()[0]!r}" if d in labelled.indexes else f"{d}[{i}]"
            for d, i in zip(labelled.dims, first, strict=True)
        )
        raise ValueError(
            f"`p` holds {int(invalid.sum())} value(s) outside 0 to 1, the first {float(values[first])!r} at {place}; a p-value is a probability, and a test that was not made is NaN."
        )
    last = values.reshape(-1) if dim is None else np.moveaxis(values, labelled.get_axis_num(dim), -1)
    adjusted, selected = np.empty(last.shape), np.empty(last.shape, dtype=bool)
    for at in np.ndindex(*last.shape[:-1]):
        adjusted[at], selected[at] = _control_family(last[at], q, method)

    def back(array):
        """A per-test result as an array shaped and labelled like *p*."""
        return labelled.copy(
            data=array.reshape(values.shape) if dim is None else np.moveaxis(array, -1, labelled.get_axis_num(dim))
        )

    out = {"p_adjusted": back(adjusted), "n_tests": np.isfinite(labelled).sum(dim)}
    if q is not None:
        out |= {"selected": back(selected), "n_selected": back(selected).sum(dim)}
    return out


def _same_labels(first, second, dim: str, names: tuple[str, str]) -> None:
    """Refuse the labelled *first* and *second*, called *names* in the message, unless they carry the same labels along *dim*, in any order."""
    unpaired = first.indexes[dim].symmetric_difference(second.indexes[dim])
    if len(unpaired):
        raise ValueError(
            f"`{names[0]}` and `{names[1]}` do not carry the same labels along {dim!r}: {_some(unpaired)} {'is' if len(unpaired) == 1 else 'are'} in only one of them."
        )


def _paired(values, where) -> tuple:
    """The labelled *values* and the labelled mask *where*, entry for entry.

    Along a dimension both label, the two carry the same labels, in any order, and *where* is put in the order of *values*; a label only one of them has is refused, since a mask says for every entry whether it counts. A dimension only one of them labels is paired by position. *where* may lack a dimension of *values*, and then marks every entry along it; a dimension of its own, which no entry of *values* answers to, is refused.
    """
    import xarray as xr

    own = [d for d in where.dims if d not in values.dims]
    if own:
        raise ValueError(f"`where` runs along {own}, which `values` does not; a mask marks entries of `values`.")
    for d in where.dims:
        if d in values.indexes and d in where.indexes:
            _same_labels(values, where, d, ("values", "where"))
    return xr.broadcast(values, where.reindex_like(values))


def histogram(values, n_bins: int, lower: float, upper: float, where=None) -> dict:
    """How many of *values* fall in each of *n_bins* equal-width bins from *lower* to *upper*.

    Every finite entry of *values* is counted, whatever its dimensions; with *where*, only the entries whose *where* is true or non-zero. Two labelled arrays are paired by label: along a dimension both label they carry the same labels, in any order, and a label only one of them has is refused; *where* may lack a dimension of *values*, and then marks every entry along it. Two bare arrays, or a labelled and a bare one, are paired by position. A bin is closed below and open above and the last is closed on both sides, as :func:`numpy.histogram` cuts them; a value outside the range is not counted.

    Returns:
        ``count`` and ``center`` on ``bin``, the bins' ``edge`` on ``bin_edge``, their common ``width``, and ``n``, the number of values counted.
    """
    import xarray as xr

    if where is not None and isinstance(values, xr.DataArray) and isinstance(where, xr.DataArray):
        values, where = _paired(values, where)
    entries = np.asarray(values, dtype=float)
    keep = np.isfinite(entries)
    if where is not None:
        keep &= np.nan_to_num(np.broadcast_to(np.asarray(where, dtype=float), entries.shape)) != 0
    edges = np.linspace(float(lower), float(upper), int(n_bins) + 1)
    count = np.histogram(entries[keep], bins=edges)[0]
    return {
        "count": xr.DataArray(count, dims=["bin"]),
        "center": xr.DataArray(0.5 * (edges[1:] + edges[:-1]), dims=["bin"]),
        "edge": xr.DataArray(edges, dims=["bin_edge"]),
        "width": float(edges[1] - edges[0]),
        "n": int(count.sum()),
    }


def _phase_series(phase, dim: str):
    """*phase* as a labelled array along *dim* and one node dimension, in that order; a bare 2-D array runs along *dim* first and ``node`` second."""
    import xarray as xr

    if not isinstance(phase, xr.DataArray):
        array = np.asarray(phase, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"`phase` is a bare array of rank {array.ndim}; a bare phase is 2-D, along {dim!r} first and its nodes second."
            )
        phase = xr.DataArray(array, dims=[dim, "node"])
    nodes = [d for d in phase.dims if d != dim]
    if dim not in phase.dims or len(nodes) != 1:
        raise ValueError(f"`phase` runs along {list(phase.dims)}; it must run along {dim!r} and one node dimension.")
    return phase.transpose(dim, nodes[0])


def _node_pairs(value, phase):
    """*value*, one entry per two nodes of the ``(dim, node)`` *phase*, on the node dimension and the same name with ``_j`` appended, both labelled as *phase*'s nodes are."""
    import xarray as xr

    node = phase.dims[1]
    labels = phase.indexes.get(node)
    coords = {} if labels is None else {node: labels.values, f"{node}_j": labels.values}
    return xr.DataArray(value, dims=[node, f"{node}_j"], coords=coords)


def phase_locking_value(phase, dim: str = "time"):
    """The phase locking value of every two nodes: the modulus of the mean over *dim* of ``exp(i (phase_j - phase_k))``.

    It is 1 for two nodes whose phase difference stays constant, near 0 for one spread evenly over the circle, and 1 on the diagonal. *phase* is in radians, along *dim* and one node dimension; a bare 2-D array is read with *dim* first. Folded over a run, the same quantity is the ``phase_locking_value`` reducer recipe.

    Returns:
        The node-by-node matrix, labelled as :func:`_node_pairs` labels it.
    """
    phase = _phase_series(phase, dim)
    unit = np.exp(1j * phase.values)
    return _node_pairs(np.abs(unit.T @ unit.conj()) / unit.shape[0], phase)


def phase_lag_index(phase, dim: str = "time"):
    """The phase lag index of every two nodes: the modulus of the mean over *dim* of ``sign(sin(phase_j - phase_k))``.

    It is 1 for two nodes whose phase difference stays strictly inside (0, pi) or strictly inside (-pi, 0), 0 for a zero difference and on the diagonal, and near 0 for a difference spread evenly over the circle. *phase* is read as :func:`phase_locking_value` reads it. One node's row is formed at a time, so the memory is one array of the phase's size rather than one per pair of nodes. Folded over a run, the same quantity is the ``phase_lag_index`` reducer recipe.

    Returns:
        The node-by-node matrix, labelled as :func:`_node_pairs` labels it.
    """
    phase = _phase_series(phase, dim)
    angles = phase.values
    rows = [np.abs(np.mean(np.sign(np.sin(angles[:, [j]] - angles)), axis=0)) for j in range(angles.shape[1])]
    return _node_pairs(np.stack(rows), phase)
