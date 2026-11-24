"""`render: histogram` and `render: joint_density` draw distributions with no study code.

A histogram's bars have to fill the bins the analysis cut, and a bar width typed into a spec is a number derived from the data that goes stale the moment the range moves. A joint density of two quantities is drawn by several replications (Koller2024 Fig. 4b, c; Schirner2023 Fig. 3b), each with its own copy of the same hexbin-and-margins code. Both are built in, and these tests pin what they draw.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from tvbo.adapters import bsplot_distributions as panels
from tvbo.adapters.bsplot import CUSTOM_PANELS, _panel_opts
from tvbo.adapters.bsplot_distributions import histogram_panel, joint_density_panel
from tvbo.datamodel import Panel


@pytest.fixture
def containers(monkeypatch):
    """Containers as in-memory datasets keyed by path; a layer is `{container, output}` as `build_context` resolves it."""
    store = {}

    def open_ds(path):
        return store[path]

    def load_layer(layer):
        ds = store[layer["container"]]
        da = ds[layer["output"]]
        return da.sel(layer["sel"]) if layer.get("sel") else da

    monkeypatch.setattr(panels, "_open_ds", open_ds)
    monkeypatch.setattr(panels, "load_layer", load_layer)
    return store


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


def _histogram(store, count, edge, n=None):
    store["h"] = xr.Dataset(
        {
            "observation__count": ("bin", np.asarray(count, float)),
            "observation__edge": ("bin_edge", np.asarray(edge, float)),
            "observation__n": ((), float(sum(count) if n is None else n)),
        }
    )
    return {"container": "h", "output": "observation__count", "style": {"color": "C0"}}


def test_both_are_registered_as_builtins_so_no_code_modules_are_needed():
    assert CUSTOM_PANELS["histogram"] is histogram_panel
    assert CUSTOM_PANELS["joint_density"] is joint_density_panel


def test_each_bar_fills_its_own_bin(containers, ax):
    """Uneven bins read from the container: a fixed width would leave gaps or overlaps."""
    edge = [0.0, 1.0, 3.0, 7.0]
    layer = _histogram(containers, [2, 5, 1], edge)
    histogram_panel(*ax, {"layers": [layer], "opts": {}})
    bars = ax[1].patches
    assert [b.get_x() for b in bars] == edge[:-1]
    assert [b.get_width() for b in bars] == list(np.diff(edge))
    assert [b.get_height() for b in bars] == [2.0, 5.0, 1.0]


def test_percent_draws_each_count_as_a_share_of_the_values_counted(containers, ax):
    layer = _histogram(containers, [1, 3], [0.0, 1.0, 2.0], n=8)
    histogram_panel(*ax, {"layers": [layer], "opts": {"percent": True}})
    assert [b.get_height() for b in ax[1].patches] == [12.5, 37.5]


def test_the_layer_style_colours_the_bars(containers, ax):
    layer = _histogram(containers, [1, 1], [0.0, 1.0, 2.0])
    layer["style"] = {"color": "#29528a", "edgecolor": "white", "linewidth": 0.5}
    histogram_panel(*ax, {"layers": [layer], "opts": {}})
    bar = ax[1].patches[0]
    assert matplotlib.colors.to_hex(bar.get_facecolor()) == "#29528a"
    assert matplotlib.colors.to_hex(bar.get_edgecolor()) == "#ffffff"


def test_counts_that_do_not_match_the_edges_are_refused(containers, ax):
    layer = _histogram(containers, [1, 1, 1], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="bin edges"):
        histogram_panel(*ax, {"layers": [layer], "opts": {}})


def _drawn_hexbin(monkeypatch, ax):
    """The points hexbin was handed, captured instead of being binned."""
    calls = []
    original = ax.hexbin

    def capture(x, y, **kw):
        calls.append((np.asarray(x), np.asarray(y)))
        return original(x, y, **kw)

    monkeypatch.setattr(ax, "hexbin", capture)
    return calls


def test_two_layers_are_paired_by_label_not_by_position(containers, monkeypatch, ax):
    """The second quantity stored in another order still meets its own entry."""
    containers["p"] = xr.Dataset(
        {
            "observation__length": ("edge", [10.0, 20.0, 30.0]),
            "observation__distance": ("edge", [3.0, 1.0, 2.0]),
        },
        coords={"edge": ["a", "b", "c"]},
    )
    containers["q"] = containers["p"].isel(edge=[2, 0, 1])
    calls = _drawn_hexbin(monkeypatch, ax[1])
    joint_density_panel(
        *ax,
        {
            "layers": [
                {"container": "p", "output": "observation__length"},
                {"container": "q", "output": "observation__distance"},
            ],
            "opts": {},
        },
    )
    x, y = calls[0]
    assert dict(zip(x, y, strict=True)) == {10.0: 3.0, 20.0: 1.0, 30.0: 2.0}


def test_layers_labelling_different_entries_are_refused(containers, ax):
    containers["p"] = xr.Dataset({"observation__length": ("edge", [10.0, 20.0])}, coords={"edge": ["a", "b"]})
    containers["q"] = xr.Dataset({"observation__distance": ("edge", [3.0, 1.0])}, coords={"edge": ["a", "c"]})
    with pytest.raises(ValueError, match="cannot be paired"):
        joint_density_panel(
            *ax,
            {
                "layers": [
                    {"container": "p", "output": "observation__length"},
                    {"container": "q", "output": "observation__distance"},
                ],
                "opts": {},
            },
        )


def test_one_layer_takes_x_from_its_encoded_coordinate(containers, monkeypatch, ax):
    """Schirner2023 Fig. 3b's form: simulated FC keyed by an `empirical_fc` coordinate."""
    containers["s"] = xr.Dataset({"observation__pairs": ("pair", [0.1, 0.2])}, coords={"empirical_fc": ("pair", [0.3, 0.4])})
    calls = _drawn_hexbin(monkeypatch, ax[1])
    joint_density_panel(*ax, {"layers": [{"container": "s", "output": "observation__pairs", "x": "empirical_fc"}], "opts": {}})
    x, y = calls[0]
    assert list(x) == [0.3, 0.4] and list(y) == [0.1, 0.2]


def test_without_two_layers_or_an_encoded_x_it_is_refused(containers, ax):
    containers["s"] = xr.Dataset({"observation__pairs": ("pair", [0.1, 0.2])})
    with pytest.raises(ValueError, match="encoding.x"):
        joint_density_panel(*ax, {"layers": [{"container": "s", "output": "observation__pairs"}], "opts": {}})


def _pairs(store):
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    store["p"] = xr.Dataset({"observation__x": ("edge", x), "observation__y": ("edge", x + rng.normal(size=400))})
    return [{"container": "p", "output": "observation__x"}, {"container": "p", "output": "observation__y"}]


def test_each_margin_shows_its_own_quantity(containers, ax):
    """The top margin is x's distribution and the right one y's, drawn sideways."""
    layers = _pairs(containers)
    joint_density_panel(*ax, {"layers": layers, "opts": {"marginal": "hist", "marginal_bins": 10}})
    top, right = ax[1].child_axes
    assert sum(p.get_height() for p in top.patches) == 400
    assert sum(p.get_width() for p in right.patches) == 400
    assert not top.axison and not right.axison


def test_kde_margins_draw_a_curve_and_both_draw_bars_beneath_it(containers, ax):
    layers = _pairs(containers)
    joint_density_panel(*ax, {"layers": layers, "opts": {"marginal": "both"}})
    top, right = ax[1].child_axes
    assert top.patches and top.lines and right.patches and right.lines


def test_mincnt_zero_draws_the_empty_cells_too(containers, ax):
    """Seaborn's hex jointplot leaves matplotlib's `mincnt` unset, so every cell of the grid is drawn, the empty ones in the map's lowest colour; `mincnt: 0` declares that look."""
    layers = _pairs(containers)
    joint_density_panel(*ax, {"layers": layers, "opts": {"gridsize": 10, "mincnt": 0}})
    counts = ax[1].collections[0].get_array()
    assert counts.min() == 0 and counts.sum() == 400


def test_by_default_only_cells_holding_a_point_are_drawn(containers, ax):
    layers = _pairs(containers)
    joint_density_panel(*ax, {"layers": layers, "opts": {"gridsize": 10}})
    assert ax[1].collections[0].get_array().min() >= 1


def _declared_opts(**slots):
    """A joint-density panel's opts as the renderer resolves them from the slots a spec declares."""
    return _panel_opts(Panel(panel_key="b", kind="custom", render="joint_density", **slots))


def test_a_declared_colorbar_adds_a_scale_of_counts_under_its_label(containers, ax):
    joint_density_panel(*ax, {"layers": _pairs(containers), "opts": _declared_opts(colorbar={"label": "Counts"})})
    assert "Counts" in [a.get_ylabel() for a in ax[1].child_axes]


def test_a_declared_colorbar_without_a_label_still_adds_the_scale(containers, ax):
    joint_density_panel(*ax, {"layers": _pairs(containers), "opts": _declared_opts(colorbar={"show": True})})
    assert len(ax[1].child_axes) == 3
