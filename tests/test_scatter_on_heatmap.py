"""A scatter layer on a heatmap marks cells at their coordinates.

A heatmap's `y` names a coordinate, its rows, and its values are the colour. A scatter drew its values against `x` whatever `y` named, so on a heatmap a point selected to one cell sat at the cell's value instead of its row, and the whole grid failed with matplotlib's "x and y must be the same size". Where `y` names a coordinate of the layer's array, each element is a point at its `x` and `y` coordinates; a `y` naming the output still plots the values.

Self-contained: a synthetic container in ``tmp_path``, rendered under Agg.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

COUPLING = [0.1, 1.0, 10.0]
SPEED = [1.0, 2.0, 3.0, 4.0]
FIELD = np.linspace(-1.0, 1.0, 12).reshape(3, 4)
"""A correlation over coupling (rows) and conduction speed (columns), none of whose values is a coordinate."""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


@pytest.fixture
def study(tmp_path):
    """A study directory holding the grid where the layout record says an analysis result lives."""
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    ds = xr.Dataset({"corr": (("coupling", "speed"), FIELD)}, coords={"coupling": COUPLING, "speed": SPEED})
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    ds.to_netcdf(analysis_container_path(results, "grid"), engine="h5netcdf")
    return tmp_path


def _points(study, sel=None):
    """Render the grid as a heatmap with a scatter layer over it, selected by *sel*, and return the scatter's points as (speed, coupling)."""
    encoding = P.Encoding(x="speed", y="coupling")
    marker = P.DataRef(iri="tvbo:grid", output="corr", sel={k: P.Argument(name=k, value=v) for k, v in (sel or {}).items()})
    figure = P.Figure(
        name="marked",
        layout="a",
        panel_numbers=False,
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="heatmap",
                colorbar={"show": False},
                layers=[
                    P.Layer(used=P.DataRef(iri="tvbo:grid", output="corr"), encoding=encoding),
                    P.Layer(used=marker, mark="scatter", encoding=encoding),
                ],
            )
        },
    )
    fig = bsplot.render(figure, str(study), str(study / "out.png"))
    (dots,) = [c for c in fig.axes[0].collections if type(c).__name__ == "PathCollection"]
    return np.asarray(dots.get_offsets())


def test_a_point_selected_to_one_cell_sits_on_that_cell(study):
    np.testing.assert_allclose(_points(study, {"coupling": 1.0, "speed": 3.0}), [[3.0, 1.0]])


def test_a_scatter_over_the_whole_grid_marks_every_cell(study):
    points = _points(study)
    assert sorted(map(tuple, points.tolist())) == sorted(itertools.product(SPEED, COUPLING))
