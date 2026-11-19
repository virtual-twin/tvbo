"""Reductions of a cohort's structural connectomes to one group connectome."""

from __future__ import annotations

import numpy as np
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
