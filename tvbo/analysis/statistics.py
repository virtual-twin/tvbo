"""Statistics over stored, labelled arrays: the implementations behind the software-based curated functions.

Each function here is named by a record in ``tvbo/database/functions/`` (``GroupedCorrelation``, ``FalseDiscoveryControl``, ``Histogram``) and returns a mapping of labelled arrays, which a study-level analysis stores one variable per key. Each runs along one named dimension and keeps every other one, so a statistic per question, region or pair is one call on the whole array rather than a loop in the study.
"""

from __future__ import annotations

import numpy as np

GROUPINGS = ("equal_width", "equal_count")
"""How units are cut into groups: bins of equal width over the grouping variable's range, or runs of equal size along its order."""

MIN_POINTS = 3
"""The fewest points a correlation is stated for; two points always correlate perfectly, so their r and p say nothing."""

METHODS = ("bh", "by")
"""The false-discovery procedures: Benjamini-Hochberg, and Benjamini-Yekutieli for tests of arbitrary dependence."""


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


def _correlates(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether a correlation of the paired *a* and *b* is defined: at least :data:`MIN_POINTS` points and neither side constant."""
    return a.size >= MIN_POINTS and np.ptp(a) > 0 and np.ptp(b) > 0


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
            unpaired = values.indexes[d].symmetric_difference(where.indexes[d])
            if len(unpaired):
                raise ValueError(
                    f"`values` and `where` do not carry the same labels along {d!r}: {_some(unpaired)} {'is' if len(unpaired) == 1 else 'are'} in only one of them."
                )
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
