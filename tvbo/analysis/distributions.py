"""Distributions of a labelled array, for a study analysis that reports how a quantity is spread rather than where it sits.

A figure binds an analysis result by the names of its axes, so a histogram is returned as a labelled ``DataArray``: the counts along one dimension whose coordinate is the centre of each bin. Declared in a study as ``callable: {module: tvbo.analysis.distributions, name: histogram}`` with the array under ``values``.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def histogram(values, bins: int = 40, lo: float | None = None, hi: float | None = None, dim: str = "bin") -> xr.DataArray:
    """Counts of the finite entries of *values* in *bins* equal bins between *lo* and *hi*, labelled by the bin centres on *dim*.

    *lo* and *hi* default to the finite minimum and maximum; an entry outside the range is left out, which is how a connectivity matrix's zeros, the pairs with no tract, stay out of a histogram of its tract lengths or delays.
    """
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    lo = float(flat.min()) if lo is None else float(lo)
    hi = float(flat.max()) if hi is None else float(hi)
    counts, edges = np.histogram(flat, bins=int(bins), range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2.0
    return xr.DataArray(counts.astype(float), dims=(dim,), coords={dim: centres}, name="count")
