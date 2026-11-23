"""A `kind: surface` panel draws a mesh an analysis derived, bound to that analysis's result the way a layer is bound to its data.

A parcel mesh (one vertex per region, the regions' adjacency as faces) or a reconstructed surface is computed, not measured, so no file exists for `surface.mesh` to name until a study writes one out by hand. `surface: {used: {analysis: <name>}}` reads `vertices` and `faces` from the analysis's own result container, and when those vertices carry region labels a layer's values meet them by label: a region's value lands on that region's vertex whatever order either array is stored in.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.adapters import figure_workflow as fw
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_LABELS = ["A", "B", "C", "D"]

_SPEC = """title: Derived mesh test
citekey: derivedmeshtest
figures:
  - name: fig-1
    layout: a
    panels:
      a:
        panel_key: a
        PANEL
"""

_SURFACE = """kind: surface
        surface: {used: {analysis: mesh}, view: lateral}
        layers: [{used: {analysis: values, output: v}}]"""


def _figure(root: Path, panel: str = _SURFACE):
    (root / "dataset_description.json").write_text('{"Name": "derivedmeshtest", "BIDSVersion": "1.9.0"}')
    spec = root / "study.yaml"
    spec.write_text(_SPEC.replace("PANEL", panel), encoding="utf-8")
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def _container(root: Path, name: str) -> Path:
    return analysis_container_path(study_path("results", root=root), name)


def _save(root: Path, name: str, variables: dict) -> str:
    path = _container(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(variables).to_netcdf(path, engine="h5netcdf")
    return str(path)


def _mesh(root: Path) -> str:
    """Four labelled vertices, two triangles: a parcel mesh in miniature."""
    vertices = xr.DataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dims=("node", "axis"),
        coords={"node": _LABELS, "axis": ["x", "y", "z"]},
    )
    faces = xr.DataArray([[0, 1, 2], [1, 3, 2]], dims=("face", "corner"))
    return _save(root, "mesh", {"vertices": vertices, "faces": faces})


def _values(root: Path, labels, values) -> str:
    return _save(root, "values", {"v": xr.DataArray(np.asarray(values, float), dims="node", coords={"node": list(labels)})})


@pytest.fixture
def drawn(monkeypatch):
    """The keyword arguments of every bsplot.plot_surf call, captured instead of rendered."""
    import bsplot as backend

    calls = []
    monkeypatch.setattr(backend, "plot_surf", lambda **kw: calls.append(kw))
    return calls


def _draw(root: Path, labels, values, **opts):
    """Run the surface drawer on the miniature mesh with a layer over *labels*."""
    ctx = {
        "layers": [{"container": _values(root, labels, values), "output": "v"}],
        "opts": {"mesh_result": _mesh(root), **opts},
        "key": "a",
        "base_dir": str(root),
    }
    fig, ax = plt.subplots()
    try:
        bsplot.surface_panel(fig, ax, ctx)
    finally:
        plt.close(fig)


def test_the_binding_resolves_to_the_analysis_result_container(tmp_path):
    figure = _figure(tmp_path)
    mesh = _mesh(tmp_path)
    _values(tmp_path, _LABELS, [1, 2, 3, 4])
    (panel,) = bsplot.build_context(figure, tmp_path, "out.png")["panels"]
    assert panel["ctx"]["opts"]["mesh_result"] == str(Path(mesh).resolve())


def test_the_rendered_panel_is_drawn_on_the_analysis_mesh(tmp_path, drawn):
    figure = _figure(tmp_path)
    _mesh(tmp_path)
    _values(tmp_path, _LABELS, [1, 2, 3, 4])
    bsplot.render(figure, str(tmp_path), str(tmp_path / "fig.png"))
    (call,) = drawn
    assert call["vertices"].shape == (4, 3) and call["faces"].tolist() == [[0, 1, 2], [1, 3, 2]]
    assert call["overlay"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_values_meet_the_vertices_by_label_not_by_position(tmp_path, drawn):
    """The layer is stored C, A, D, B: placed by position, every vertex would take a neighbour's value."""
    _draw(tmp_path, ["C", "A", "D", "B"], [3, 1, 4, 2])
    assert drawn[0]["overlay"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_a_layer_over_fewer_regions_leaves_the_others_missing(tmp_path, drawn):
    """Two of the mesh's regions absent from the layer: each value still lands on its own vertex, and the rest are missing rather than zero."""
    _draw(tmp_path, ["D", "B"], [4, 2], missing_color="#000000")
    overlay, mask = drawn[0]["overlay"], drawn[0]["mask"]
    assert np.isnan(overlay[[0, 2]]).all() and overlay[[1, 3]].tolist() == [2.0, 4.0]
    assert mask.tolist() == [True, False, True, False]


def test_a_value_whose_label_is_no_vertex_is_refused_by_name(tmp_path, drawn):
    with pytest.raises(ValueError, match="'Z'"):
        _draw(tmp_path, ["A", "Z"], [1, 9])


def test_the_binding_names_where_the_mesh_is_and_nothing_else(tmp_path):
    """A mesh is both arrays together, so naming one of them would be accepted and then ignored."""
    figure = _figure(tmp_path, _SURFACE.replace("{analysis: mesh}", "{analysis: mesh, output: vertices}"))
    _mesh(tmp_path)
    with pytest.raises(ValueError, match="output"):
        bsplot.build_context(figure, tmp_path, "out.png")


def test_a_second_place_for_the_mesh_is_refused(tmp_path, drawn):
    with pytest.raises(ValueError, match="template"):
        _draw(tmp_path, _LABELS, [1, 2, 3, 4], template="fsaverage")


def test_the_render_rule_waits_for_the_mesh_container(tmp_path):
    """The mesh's analysis is an input of the figure's rule, as the layer's is."""
    figure = _figure(tmp_path)
    mesh = _mesh(tmp_path)
    _values(tmp_path, _LABELS, [1, 2, 3, 4])
    assert str(Path(mesh).resolve()) in fw.emit_figure_rules([figure], base_dir=tmp_path)


def test_a_grid_template_binds_every_cell_to_the_mesh(tmp_path):
    grid = """kind: grid
        cell: {kind: surface, surface: {used: {analysis: mesh}}}
        layers: [{used: {analysis: values, output: v}}, {used: {analysis: values, output: v}}]"""
    figure = _figure(tmp_path, grid)
    mesh = _mesh(tmp_path)
    _values(tmp_path, _LABELS, [1, 2, 3, 4])
    (panel,) = bsplot.build_context(figure, tmp_path, "out.png")["panels"]
    assert [cell["ctx"]["opts"]["mesh_result"] for cell in panel["insets"]] == [str(Path(mesh).resolve())] * 2
