"""A layer shows a channel at a factor of its stored values (`encoding.display`), converting what is drawn and never what is stored.

Koller2024 prints its coupling at 1000 times the simulated value, and its Fig. 5b reads a time stored in ms in seconds. Declared on the encoding, the conversion runs in the figure's script after the layer's `sel`, which keeps selecting in the stored values; the axis, its ticks, the theme's tick rule and a colour bar keying converted values are in the displayed values, and the result container is left as it was. A factor states no unit, so an axis read in seconds is named by the label the panel declares. A factor that is not positive, a key that is not a channel and one coordinate converted twice are refused when the script is generated.

Self-contained: a synthetic container in ``tmp_path``, rendered under Agg.
"""

from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

K = np.array([0.001, 0.002, 0.003])
"""The coupling as simulated, shown at 1000 times as 1 to 3."""

SPEED = np.array([1.0, 2.0])

V = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.9]])
"""A share per coupling and speed, between 0 and 1."""

TIME = np.arange(0.0, 180001.0, 30000.0)
"""Seven samples over three minutes, stored in ms."""

TRACE = np.sin(TIME / 60000.0)

SEL = {"K": [0.001, 0.003], "speed": 2.0}
"""Two couplings by their stored values, at one speed."""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


@pytest.fixture
def container(tmp_path):
    """A container over coupling and speed, and over time in ms, written where the layout record says an analysis result lives."""
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    ds = xr.Dataset({"v": (("K", "speed"), V), "trace": ("time", TRACE)}, coords={"K": K, "speed": SPEED, "time": TIME})
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    path = analysis_container_path(results, "sweep")
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def _figure(kind, output, encoding, sel=None, theme=None, xlabel=None):
    """One panel drawing analysis `sweep`'s *output* through *encoding*, selected by *sel*, under the declared *xlabel* if any."""
    selection = {k: P.Argument(name=k, value=v) for k, v in (sel or {}).items()} or None
    used = P.DataRef(analysis="sweep", output=output, sel=selection)
    layer = P.Layer(used=used, encoding=P.Encoding(**encoding))
    panel = P.Panel(panel_key="a", kind=kind, xlabel=xlabel, layers=[layer])
    return P.Figure(name="display", layout="a", panel_numbers=False, theme=theme, panels={"a": panel})


def _render(figure, root):
    """Render *figure* in the study at *root*; return the figure and its panel's axes."""
    fig = bsplot.render(figure, str(root), str(root / "out.png"))
    return fig, fig.axes[0]


def _ticks_in_view(axis):
    """The tick positions of *axis* inside its view."""
    lo, hi = sorted(axis.get_view_interval())
    return [t for t in axis.get_majorticklocs() if lo <= t <= hi]


def _digest(path):
    """The container's bytes, as a sha256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_factor_shows_the_axis_at_that_multiple_while_sel_reads_the_stored_values(tmp_path, container):
    """Koller2024's K axis: the coupling selected by its stored values is drawn at 1000 times them, and the container is left as it was. Selected after the conversion, both stored values would land on the shown 1."""
    before = _digest(container)
    encoding = {"x": "K", "y": "v", "display": {"x": {"factor": 1000.0}}}
    _, ax = _render(_figure("cartesian", "v", encoding, sel=SEL), tmp_path)
    (line,) = ax.lines
    np.testing.assert_allclose(line.get_xdata(), [1.0, 3.0])
    np.testing.assert_allclose(line.get_ydata(), V[[0, 2], 1])
    ticks = _ticks_in_view(ax.xaxis)
    assert ticks and min(ticks) >= 0.5 and max(ticks) <= 3.5, ticks
    assert ax.get_xlabel() == "1000 × K"
    assert _digest(container) == before


def test_a_time_stored_in_ms_is_drawn_in_seconds_under_the_label_the_panel_declares(tmp_path, container):
    """Koller2024 Fig. 5b's time axis: `factor: 0.001` puts the samples at their times in seconds and draws the values as stored. The factor states no unit, so the seconds are named by the declared label, and a panel declaring none prints the factor, never a unit."""
    encoding = {"x": "time", "display": {"x": {"factor": 0.001}}}
    _, ax = _render(_figure("cartesian", "trace", encoding, xlabel="Time (s)"), tmp_path)
    (line,) = ax.lines
    np.testing.assert_allclose(line.get_xdata(), TIME / 1000.0)
    np.testing.assert_allclose(line.get_ydata(), TRACE)
    ticks = _ticks_in_view(ax.xaxis)
    assert ticks and min(ticks) >= 0.0 and max(ticks) <= 190.0, ticks
    assert ax.get_xlabel() == "Time (s)"
    _, undeclared = _render(_figure("cartesian", "trace", encoding), tmp_path)
    assert undeclared.get_xlabel() == "0.001 × time"


def test_the_theme_tick_rule_rounds_the_displayed_values(tmp_path, container):
    """Three minutes stored in ms, shown at 1/60000 of it: `ticks: round` puts the ticks on whole displayed numbers, where rounding the stored values (0, 50000, 100000, 150000) would print 0, 0.83, 1.67 and 2.5."""
    encoding = {"x": "time", "display": {"x": {"factor": 1.0 / 60000.0}}}
    _, ax = _render(_figure("cartesian", "trace", encoding, theme=P.Theme(ticks="round", max_ticks=4)), tmp_path)
    np.testing.assert_allclose(ax.get_xticks(), [0.0, 1.0, 2.0, 3.0])


def test_a_colour_bar_keying_a_converted_field_prints_the_converted_numbers(tmp_path, container):
    """A share stored between 0 and 1 painted at 100 times: the heatmap's values and the bar beside it span 10 to 90."""
    encoding = {"x": "speed", "y": "K", "display": {"color": {"factor": 100.0}}}
    fig, ax = _render(_figure("heatmap", "v", encoding), tmp_path)
    (bar,) = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    np.testing.assert_allclose(sorted(bar.get_ylim()), [100.0 * V.min(), 100.0 * V.max()])
    np.testing.assert_allclose(np.sort(np.asarray(ax.collections[0].get_array()).ravel()), np.sort(100.0 * V.ravel()))


@pytest.mark.parametrize("factor", [0.0, -1000.0])
def test_a_factor_that_is_not_positive_is_refused_by_name(tmp_path, container, factor):
    figure = _figure("cartesian", "v", {"x": "K", "display": {"x": {"factor": factor}}}, sel={"speed": 2.0})
    refusal = r"`display\.x` on the layer reading the reference to analysis 'sweep'.*is not positive"
    with pytest.raises(ValueError, match=refusal):
        bsplot.render_code(figure, tmp_path, "fig.png")


@pytest.mark.parametrize(
    ("encoding", "refusal"),
    [
        ({"x": "K", "display": {"w": {"factor": 2.0}}}, r"`display\.w` .*'w' is not a channel"),
        (
            {"x": "K", "color": "K", "display": {"x": {"factor": 2.0}, "color": {"factor": 2.0}}},
            r"'K' is converted by `display\.(x|color)` already",
        ),
    ],
    ids=["not-a-channel", "one-coordinate-twice"],
)
def test_a_display_on_no_channel_or_twice_on_one_coordinate_is_refused_by_name(tmp_path, container, encoding, refusal):
    figure = _figure("cartesian", "v", encoding, sel={"speed": 2.0})
    with pytest.raises(ValueError, match=refusal):
        bsplot.render_code(figure, tmp_path, "fig.png")


def test_a_panels_load_layer_converts_after_selecting_in_the_stored_values(container):
    """A custom or built-in panel reads the layer as the grammar draws it."""
    display = {"x": {"name": "K", "factor": 1000.0, "values": False}}
    da = bsplot.load_layer({"container": str(container), "output": "v", "sel": SEL, "display": display})
    np.testing.assert_allclose(da["K"].values, [1.0, 3.0])
    np.testing.assert_allclose(da.values, V[[0, 2], 1])
