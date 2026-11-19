"""A grid's `placeholder` stands in for the cells whose data is missing, and only for them.

A grid's own axes only frames its cells, so a placeholder drawn there lies under every cell, the ones drawn from data included. Each cell is drawn through the guard a mosaic panel is drawn through instead: a cell whose result container is missing shows the grid's label on a bare frame, and the others draw their data. Without a placeholder a missing container is an error worded the same for every kind, that the container was not on disk when the figure was generated, where a built-in kind opened the empty path as the working directory and failed with ``IsADirectoryError``.

Self-contained: synthetic containers in ``tmp_path``, rendered under Agg with the surface backend's draw stubbed out.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo.datamodel.pydantic as P
from tvbo.adapters import bsplot

LABEL = "pending"
MESH = {"surface": {"mesh": "mesh.npz"}}
"""A surface cell on the four-vertex mesh the study holds."""


@pytest.fixture(autouse=True)
def _close_figures():
    """Each render leaves its figure open for the assertions; close them all after the test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_surface_drawing(monkeypatch):
    """Which cells draw is under test, not what a mesh looks like."""
    import bsplot as backend

    monkeypatch.setattr(backend, "plot_surf", lambda **kw: None)


@pytest.fixture
def study(tmp_path):
    """A study directory holding a four-vertex mesh and two analysis results, `values` per vertex and `trace` over `t`; `absent` has none."""
    from tvbo.data.dataref import analysis_container_path
    from tvbo.utils.study_layout import study_path

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    np.savez(tmp_path / "mesh.npz", vertices=vertices, faces=np.array([[0, 1, 2], [1, 3, 2]]))
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    values = xr.Dataset({"v": ("vertex", [1.0, 2.0, 3.0, 4.0])})
    trace = xr.Dataset({"v": ("t", [0.0, 1.0, 0.5])}, coords={"t": [0.0, 0.5, 1.0]})
    for name, ds in (("values", values), ("trace", trace)):
        ds.to_netcdf(analysis_container_path(results, name), engine="h5netcdf")
    return tmp_path


def _grid(study, kind, analyses, placeholder=None, **cell):
    """Render one grid panel holding a *kind* cell per analysis, and return the grid's own axes and its cells in order."""
    encoding = {"encoding": P.Encoding(x="t", y="v")} if kind == "cartesian" else {}
    figure = P.Figure(
        name="grid",
        layout="a",
        panel_numbers=False,
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="grid",
                grid={"ncols": len(analyses)},
                cell={"kind": kind, **cell},
                layers=[P.Layer(used=P.DataRef(analysis=name, output="v"), **encoding) for name in analyses],
                placeholder=placeholder,
            )
        },
    )
    fig = bsplot.render(figure, str(study), str(study / "out.png"))
    (host,) = [ax for ax in fig.axes if ax.child_axes]
    return host, host.child_axes


def _says(ax) -> str:
    """Every text drawn on *ax*, as one line."""
    return " ".join(t.get_text().replace("\n", " ") for t in ax.texts)


def test_only_the_cell_whose_container_is_missing_shows_the_placeholder(study):
    host, cells = _grid(study, "surface", ["values", "absent"], placeholder=LABEL, **MESH)
    assert [LABEL in _says(ax) for ax in (host, *cells)] == [False, False, True]


def test_a_grid_whose_cells_all_have_data_shows_no_placeholder(study):
    host, cells = _grid(study, "surface", ["values", "values"], placeholder=LABEL, **MESH)
    assert not any(LABEL in _says(ax) for ax in (host, *cells))


def test_a_missing_cell_container_without_a_placeholder_says_it_was_not_found(study):
    with pytest.raises(FileNotFoundError, match="not found when the figure was generated"):
        _grid(study, "surface", ["values", "absent"], **MESH)


def test_a_cell_that_falls_back_stays_bare_of_its_declared_ticks(study):
    _host, (drawn, missing) = _grid(study, "cartesian", ["trace", "absent"], placeholder=LABEL, xticks=[0.0, 1.0])
    np.testing.assert_allclose(drawn.get_xticks(), [0.0, 1.0])
    assert list(missing.get_xticks()) == [] and LABEL in _says(missing)
