"""A surface map paints the atlas regions its layer leaves out in `missing_color`, apart from the mesh no region covers.

A per-region map that omits some regions on purpose, the target a quantity is measured in, would otherwise draw them in the same grey as the medial wall, and a reader cannot tell a region left out from cortex no region covers. The mesh here has two medial-wall vertices and three regions of two vertices each; the layer carries two of the regions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.colors import to_rgba

bsplot = pytest.importorskip("bsplot")

from tvbo.adapters import bsplot as adapter
from tvbo.plot import palette

PARCELS = np.array([-1, -1, 0, 0, 1, 1, 2, 2])
"""The region each vertex belongs to, -1 where none does."""


def _drawn(monkeypatch, opts):
    captured = {}
    monkeypatch.setattr(bsplot, "plot_surf", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(adapter, "_surface_mesh", lambda ctx: (np.zeros((PARCELS.size, 3)), np.array([[0, 1, 2]])))
    monkeypatch.setattr(adapter, "_surface_parcellation", lambda *args: (PARCELS, {"A": 0, "B": 1, "C": 2}))
    layer = xr.DataArray([1.0, 2.0], dims="region", coords={"region": ["A", "B"]})
    monkeypatch.setattr(adapter, "load_layer", lambda _layer: layer)
    fig, ax = plt.subplots()
    adapter.surface_panel(
        fig, ax, {"opts": {"atlas": "tvbo:atlas/Test", "symmetric": False, **opts}, "layers": [{}], "base_dir": "."}
    )
    plt.close(fig)
    return captured


def test_a_region_the_layer_leaves_out_takes_the_declared_colour(monkeypatch):
    drawn = _drawn(monkeypatch, {"missing_color": "palette.0"})
    np.testing.assert_array_equal(drawn["mask"], PARCELS == 2)
    np.testing.assert_allclose(drawn["mask_colour"], to_rgba(palette.palette()[0]))


def test_unset_leaves_every_uncovered_vertex_to_the_base_grey(monkeypatch):
    drawn = _drawn(monkeypatch, {})
    assert drawn["mask"] is None
    assert drawn["mask_colour"] is None
