"""`coordinate_system: data` places an annotation at an `x` and a `y` in the panel's data units rather than in axes fractions.

A paper that prints a statistic beside its points with `ax.text(80, 3, ...)` puts it at a place on the data, which no axes fraction states once the limits move. With `coordinate_system: data` the label, its box and an arrow's head and tail are all placed in the data's own units; unset, every position stays an axes fraction. A named corner (`loc`) is a place on the axes, so it is refused beside data coordinates, and so is a data position missing `x` or `y`, which has no axes-fraction default to fall back to.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_SPEC = """title: Data coordinates test
citekey: datacoordstest
figures:
  - name: fig-1
    layout: a
    panels:
      a:
        panel_key: a
PANEL"""

_LAYER = "{used: {analysis: groups, output: mean}, mark: scatter, encoding: {x: group, y: mean}}"

_CARTESIAN = f"""        kind: cartesian
        layers: [{_LAYER}]
        xlim: [0.0, 10.0]
        ylim: [0.0, 20.0]
        annotations:
          - ANNOTATION
"""

_GRID = f"""        kind: grid
        grid: {{ncols: 2, col_labels: [left, right]}}
        cell: {{kind: cartesian}}
        layers: [{_LAYER}, {_LAYER}]
"""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


def _render(root: Path, annotation: str, monkeypatch, panel: str = _CARTESIAN) -> list:
    """Render the one-panel figure with *annotation* and return what each drawn label of its panel was, as the saved figure holds it."""
    (root / "dataset_description.json").write_text('{"Name": "datacoordstest", "BIDSVersion": "1.9.0"}')
    spec = root / "study.yaml"
    spec.write_text(_SPEC.replace("PANEL", panel.replace("ANNOTATION", annotation)), encoding="utf-8")
    path = analysis_container_path(study_path("results", root=root), "groups")
    path.parent.mkdir(parents=True, exist_ok=True)
    group = xr.DataArray([2.0, 5.0, 8.0], dims="group")
    xr.Dataset({"mean": group * 2.0, "r": xr.DataArray(0.5)}, coords={"group": group}).to_netcdf(path, engine="h5netcdf")
    figure = tvbo.SimulationStudy.from_file(str(spec)).figures[0]
    drawn, savefig = [], matplotlib.figure.Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        ax = self.axes[0]
        drawn[:] = [(text, ax, text.get_transform().transform(text.get_position())) for text in ax.texts]
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    bsplot.render(figure, str(root), str(root / "fig.png"))
    return drawn


def _label(drawn, text: str):
    """The drawn label reading *text*, its axes and where it sits on the canvas."""
    (found,) = [entry for entry in drawn if entry[0].get_text() == text]
    return found


@pytest.mark.parametrize(
    ("annotation", "label"),
    [
        ('{text: "mean", x: 6.0, y: 15.0, coordinate_system: data}', "mean"),
        ('{text: "r={:.2f}", used: {analysis: groups, output: r}, x: 6.0, y: 15.0, coordinate_system: data}', "r=0.50"),
        (
            '{text: "r={:.2f}", used: {analysis: groups, output: r}, x: 6.0, y: 15.0, coordinate_system: data, backdrop: false}',
            "r=0.50",
        ),
    ],
    ids=["literal", "bound-with-backdrop", "bound-bare"],
)
def test_a_label_in_data_coordinates_sits_on_that_point_of_the_data(tmp_path, monkeypatch, annotation, label):
    """At (6, 15) on axes running 0 to 10 and 0 to 20, the label sits where the data point (6, 15) is drawn, not at the axes fraction (6, 15) far outside the panel."""
    _text, ax, at = _label(_render(tmp_path, annotation, monkeypatch), label)
    np.testing.assert_allclose(at, ax.transData.transform((6.0, 15.0)))


def test_an_undeclared_frame_stays_axes_fractions(tmp_path, monkeypatch):
    drawn = _render(tmp_path, '{text: "corner", x: 0.25, y: 0.75}', monkeypatch)
    _text, ax, at = _label(drawn, "corner")
    np.testing.assert_allclose(at, ax.transAxes.transform((0.25, 0.75)))


def test_a_grids_own_column_labels_stay_in_axes_fractions(tmp_path, monkeypatch):
    """A grid writes its column labels through the same emitted code as an annotation, so they carry the frame too: axes fractions of the host panel."""
    text, ax, _at = _label(_render(tmp_path, "", monkeypatch, panel=_GRID), "left")
    assert text.get_transform() is ax.transAxes


def test_an_arrow_in_data_coordinates_runs_between_two_points_of_the_data(tmp_path, monkeypatch):
    """The head at (x, y) and the tail `arrow` away from it are both in the data's units."""
    drawn = _render(tmp_path, '{text: "peak", x: 5.0, y: 10.0, arrow: [-2.0, 4.0], coordinate_system: data}', monkeypatch)
    arrow, _ax, _at = _label(drawn, "peak")
    assert isinstance(arrow, matplotlib.text.Annotation)
    assert (arrow.xycoords, arrow.anncoords) == ("data", "data")
    np.testing.assert_allclose([arrow.xy, arrow.xyann], [(5.0, 10.0), (7.0, 6.0)])


def test_an_arrow_with_a_computed_tail_points_at_a_data_position(tmp_path, monkeypatch):
    drawn = _render(
        tmp_path,
        '{text: "", x: 5.0, y: 10.0, tail_x: 8.0, tail_used: {analysis: groups, output: r}, coordinate_system: data}',
        monkeypatch,
    )
    (arrow,) = [text for text, _ax, _at in drawn if isinstance(text, matplotlib.text.Annotation)]
    assert arrow.xycoords == "data"
    np.testing.assert_allclose([arrow.xy, arrow.xyann], [(5.0, 10.0), (8.0, 0.5)])


@pytest.mark.parametrize(
    ("annotation", "named"),
    [
        ('{text: "mean", loc: upper left, coordinate_system: data}', "`loc`"),
        ('{text: "mean", x: 6.0, coordinate_system: data}', "`y`"),
    ],
    ids=["named-corner", "missing-y"],
)
def test_data_coordinates_without_a_data_position_are_refused(tmp_path, monkeypatch, annotation, named):
    with pytest.raises(ValueError, match=named):
        _render(tmp_path, annotation, monkeypatch)
