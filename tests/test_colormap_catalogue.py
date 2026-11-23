"""A colormap a figure names resolves for every mark, not only for the surface renderer.

The Scientific Colour Maps (``batlow``, ``davos``, ``turku``, ``lapaz``) are in bsplot's catalogue, which the surface renderer reads, while a matplotlib mark looks a string up in matplotlib's registry alone. Koller2024's Fig. 4a heatmap in ``turku`` therefore failed with "'turku' is not a valid value for cmap" while the surfaces beside it rendered in ``davos``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tvbo.adapters.bsplot import scale_colormap
from tvbo.adapters.colormaps import resolve


@pytest.mark.parametrize("name", ["turku", "davos", "lapaz_r", "batlow"])
def test_a_scientific_colour_map_draws_a_matplotlib_mark(name):
    fig, ax = plt.subplots()
    try:
        mesh = ax.pcolormesh(np.arange(4.0).reshape(2, 2), cmap=scale_colormap(name))
        assert mesh.get_cmap()(0.0) == resolve(name)(0.0)
    finally:
        plt.close(fig)


def test_a_reversed_map_runs_the_other_way():
    assert np.allclose(resolve("lapaz_r")(0.0), resolve("lapaz")(1.0))


def test_a_name_matplotlib_knows_passes_through_untouched():
    assert resolve("viridis") == "viridis"


def test_a_tabled_map_still_comes_from_its_table():
    assert np.allclose(resolve("parula")(0.0)[:3], (0.2081, 0.1663, 0.5292), atol=1e-3)


def test_a_name_no_catalogue_knows_is_refused_where_it_is_resolved():
    with pytest.raises(ValueError, match="not_a_colormap"):
        resolve("not_a_colormap")
