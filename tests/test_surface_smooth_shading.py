"""`surface: {shading: smooth}` blends a coarse mesh's colours and light across its faces, as a renderer's smooth shading draws it.

bsplot paints each face in one colour, the mean of its three corners' values, and lights it by its own normal, so a parcel mesh of a few hundred vertices prints as visible facets where a figure rendered with smooth shading shows a smooth field. `shading: smooth` splits each face into 16 before drawing: a new vertex on an edge takes the mean of the edge's two values and sits on the curved surface the corners' normals describe, so the colour and the light both vary across what was one face. The mesh's own vertices keep their place and their values, every new face keeps the winding the backend culls by, and a vertex without a value still greys every face it touched.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from scipy.spatial import ConvexHull

import tvbo
from tvbo.adapters import bsplot
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
_SQUARE_FACES = np.array([[0, 1, 2], [1, 3, 2]])
_LINEAR = {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}
"""The field 1 + x + 2y on the square's four corners."""


def _icosahedron(radius: float = 50.0):
    """A closed mesh of 12 vertices on a sphere of *radius*, its 20 faces wound outward."""
    g = (1.0 + 5.0**0.5) / 2.0
    corners = [(0, a, b * g) for a in (-1, 1) for b in (-1, 1)]
    corners += [(a, b * g, 0) for a in (-1, 1) for b in (-1, 1)]
    corners += [(b * g, 0, a) for a in (-1, 1) for b in (-1, 1)]
    vertices = np.asarray(corners, float) * radius / np.sqrt(1.0 + g * g)
    faces = ConvexHull(vertices).simplices
    tri = vertices[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(axis=1)) < 0
    return vertices, np.where(inward[:, None], faces[:, ::-1], faces)


def _save(path: Path, variables: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(variables).to_netcdf(path, engine="h5netcdf")
    return str(path)


def _mesh(path: Path, vertices, faces) -> str:
    labels = [f"r{i}" for i in range(len(vertices))]
    vertices = xr.DataArray(vertices, dims=("node", "axis"), coords={"node": labels, "axis": ["x", "y", "z"]})
    return _save(path, {"vertices": vertices, "faces": xr.DataArray(np.asarray(faces), dims=("face", "corner"))})


def _values(path: Path, values: dict) -> str:
    return _save(path, {"v": xr.DataArray(list(values.values()), dims="node", coords={"node": list(values)})})


@pytest.fixture
def drawn(monkeypatch):
    """The keyword arguments of every bsplot.plot_surf call, captured instead of rendered."""
    import bsplot as backend

    calls = []
    monkeypatch.setattr(backend, "plot_surf", lambda **kw: calls.append(kw))
    yield calls
    plt.close("all")


def _draw(tmp_path: Path, values: dict, vertices=_SQUARE, faces=_SQUARE_FACES, **opts):
    """Run the surface drawer on a labelled mesh with a layer over *values*."""
    ctx = {
        "layers": [{"container": _values(tmp_path / "values.h5", values), "output": "v"}],
        "opts": {"mesh_result": _mesh(tmp_path / "mesh.h5", vertices, faces), **opts},
        "key": "a",
        "base_dir": str(tmp_path),
    }
    fig, ax = plt.subplots()
    bsplot.surface_panel(fig, ax, ctx)


def _area(vertices, faces):
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()


def test_flat_is_the_default(tmp_path, drawn):
    _draw(tmp_path, _LINEAR)
    assert drawn[0]["faces"].tolist() == _SQUARE_FACES.tolist()


def test_each_face_becomes_sixteen_and_the_meshs_own_vertices_keep_their_place_and_value(tmp_path, drawn):
    _draw(tmp_path, _LINEAR, shading="smooth")
    (call,) = drawn
    assert len(call["faces"]) == 16 * len(_SQUARE_FACES)
    np.testing.assert_array_equal(call["vertices"][:4], _SQUARE)
    assert call["overlay"][:4].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_on_a_flat_mesh_every_new_vertex_carries_the_field_between_its_corners(tmp_path, drawn):
    """A flat mesh stays flat, and a field linear across it is reproduced at every new vertex: each takes the mean of its edge's two ends."""
    _draw(tmp_path, _LINEAR, shading="smooth")
    vertices, overlay = drawn[0]["vertices"], drawn[0]["overlay"]
    assert len(vertices) == 25 and np.allclose(vertices[:, 2], 0.0)
    np.testing.assert_allclose(overlay, 1.0 + vertices[:, 0] + 2.0 * vertices[:, 1])


def test_new_vertices_lie_on_the_curved_surface_the_normals_describe(tmp_path, drawn):
    """On an icosahedron the corners' normals describe its sphere: the new vertices stay within 5 % of its radius, where midpoints of the flat faces fall up to 19 % inside it."""
    vertices, faces = _icosahedron()
    _draw(tmp_path, {f"r{i}": x for i, x in enumerate(vertices[:, 0])}, vertices, faces, shading="smooth")
    radius = np.linalg.norm(drawn[0]["vertices"], axis=1)
    assert len(radius) == 162 and np.abs(radius - 50.0).max() < 2.5


def test_every_new_face_keeps_the_outward_winding_the_backend_culls_by(tmp_path, drawn):
    vertices, faces = _icosahedron()
    _draw(tmp_path, {f"r{i}": x for i, x in enumerate(vertices[:, 0])}, vertices, faces, shading="smooth")
    tri = drawn[0]["vertices"][drawn[0]["faces"]]
    outward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(axis=1))
    assert len(tri) == 16 * len(faces) and (outward > 0).all()


def test_a_vertex_without_a_value_greys_exactly_the_faces_it_touched(tmp_path, drawn):
    """r0 touches only the first triangle, half the square: that half is greyed and the other is not, and a new vertex between r0 and a valued corner takes that corner's value rather than none."""
    _draw(tmp_path, {"r1": 2.0, "r2": 3.0, "r3": 4.0}, missing_color="#000000", shading="smooth")
    vertices, faces, overlay, mask = (drawn[0][k] for k in ("vertices", "faces", "overlay", "mask"))
    assert len(faces) == 32
    assert _area(vertices, faces[mask[faces].any(axis=1)]) == pytest.approx(0.5)
    assert np.flatnonzero(np.isnan(overlay)).tolist() == [0]


@pytest.mark.parametrize(("opts", "named"), [({"markers": True}, "markers"), ({"edgecolor": "k"}, "edge_color")])
def test_smooth_shading_is_refused_where_there_is_no_painted_face_to_blend(tmp_path, drawn, opts, named):
    """With `markers` the mesh under the dots is flat and unlit; with `edge_color` the outline would trace triangles the mesh does not have."""
    with pytest.raises(ValueError, match=named):
        _draw(tmp_path, _LINEAR, shading="smooth", **opts)


def test_the_declared_slot_reaches_the_drawer(tmp_path, drawn):
    (tmp_path / "dataset_description.json").write_text('{"Name": "smoothtest", "BIDSVersion": "1.9.0"}')
    results = study_path("results", root=tmp_path)
    _mesh(analysis_container_path(results, "mesh"), _SQUARE, _SQUARE_FACES)
    _values(analysis_container_path(results, "values"), _LINEAR)
    spec = tmp_path / "study.yaml"
    spec.write_text(
        """title: Smooth shading test
citekey: smoothtest
figures:
  - name: fig-1
    layout: a
    panels:
      a:
        panel_key: a
        kind: surface
        surface: {used: {analysis: mesh}, shading: smooth}
        layers: [{used: {analysis: values, output: v}}]
""",
        encoding="utf-8",
    )
    figure = tvbo.SimulationStudy.from_file(str(spec)).figures[0]
    bsplot.render(figure, str(tmp_path), str(tmp_path / "fig.png"))
    assert len(drawn[0]["faces"]) == 16 * len(_SQUARE_FACES)
