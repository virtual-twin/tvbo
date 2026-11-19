"""`surface: {markers: true}` draws a layer as one dot per vertex over the mesh, the way a brain plot of region centres shows a parcellated quantity.

The dots are depth-sorted with the mesh in one pass, so the mesh hides the dots behind it: a lateral view of two hemispheres shows the near hemisphere's regions, not both overplotted. They are drawn unlit, so a dot's colour is the colour on the bar that keys it, and the mesh under them is flat in the page colour unless the spec gives it one.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from bsplot.graph.network import create_sphere_mesh

from tvbo.adapters import bsplot

_SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])


def _mesh(tmp_path: Path, vertices, faces) -> str:
    labels = [f"r{i}" for i in range(len(vertices))]
    path = tmp_path / "mesh.h5"
    xr.Dataset(
        {
            "vertices": xr.DataArray(vertices, dims=("node", "axis"), coords={"node": labels, "axis": ["x", "y", "z"]}),
            "faces": xr.DataArray(faces, dims=("face", "corner")),
        }
    ).to_netcdf(path, engine="h5netcdf")
    return str(path)


def _layer(tmp_path: Path, values: dict) -> dict:
    path = tmp_path / "values.h5"
    xr.Dataset({"v": xr.DataArray(list(values.values()), dims="node", coords={"node": list(values)})}).to_netcdf(
        path, engine="h5netcdf"
    )
    return {"container": str(path), "output": "v"}


def _draw(tmp_path: Path, values: dict, vertices=_SQUARE, faces=((0, 1, 2), (1, 3, 2)), **opts):
    """Run the surface drawer with markers on, returning the figure and its axes."""
    ctx = {
        "layers": [_layer(tmp_path, values)] if values else [],
        "opts": {"mesh_result": _mesh(tmp_path, np.asarray(vertices, float), np.asarray(faces)), "markers": True, **opts},
        "key": "a",
        "base_dir": str(tmp_path),
    }
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    bsplot.surface_panel(fig, ax, ctx)
    return fig, ax


@pytest.fixture
def drawn(monkeypatch):
    """The keyword arguments of every bsplot.plot_surf call, captured instead of rendered."""
    import bsplot as backend

    calls = []
    monkeypatch.setattr(backend, "plot_surf", lambda **kw: calls.append(kw))
    yield calls
    plt.close("all")


def _points(image):
    return np.asarray(image.darrays[0].data), np.asarray(image.darrays[1].data)


def _centres(image, n):
    """The centre of each of the *n* dots in a dot surface."""
    return _points(image)[0].reshape(n, -1, 3).mean(axis=1)


def _lifted(vertices, radius):
    """The flat square's vertices raised by *radius* along its normal, +z."""
    return vertices + [0.0, 0.0, radius]


def test_each_valued_vertex_gets_a_dot_over_a_flat_mesh_in_the_page_colour(tmp_path, drawn):
    fig, _ax = _draw(tmp_path, {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0})
    (call,) = drawn
    assert call.get("overlay") is None and call["shade_intensity"] == 0.0
    assert np.allclose(matplotlib.colors.to_rgba(call["color"]), fig.get_facecolor())
    (dots,) = call["additional_surfaces"]
    per_dot = len(_points(dots)[0]) // 4
    assert np.allclose(_centres(dots, 4), _lifted(_SQUARE, 0.5), atol=1e-6)
    assert call["additional_overlays"][0].tolist() == np.repeat([1.0, 2.0, 3.0, 4.0], per_dot).tolist()


def test_a_dot_sits_on_the_surface_so_the_faces_around_its_vertex_do_not_cover_it(tmp_path, drawn):
    """Centred on its vertex a dot is half inside the mesh, and a depth sort by face then draws the neighbouring faces over its front; raised by its radius along the vertex normal it lies on the surface, as a z-buffered render shows it."""
    _draw(tmp_path, {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}, marker_size=0.25)
    assert np.allclose(_centres(drawn[0]["additional_surfaces"][0], 4), _lifted(_SQUARE, 0.25), atol=1e-6)


def test_a_vertex_the_layer_holds_no_value_for_gets_a_dot_in_the_missing_colour(tmp_path, drawn):
    _draw(tmp_path, {"r3": 4.0, "r1": 2.0}, missing_color="#000000")
    (call,) = drawn
    valued, missing = call["additional_surfaces"]
    assert call["additional_colors"][0] is None and np.allclose(call["additional_colors"][1], (0, 0, 0, 1))
    assert call["additional_overlays"][1] is None
    assert np.allclose(_centres(missing, 2), _lifted(_SQUARE[[0, 2]], 0.5), atol=1e-6)
    assert np.allclose(_centres(valued, 2), _lifted(_SQUARE[[1, 3]], 0.5), atol=1e-6)


def test_the_dots_face_outward_as_the_backend_culls_by_winding(tmp_path, drawn):
    _draw(tmp_path, {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0})
    dots = drawn[0]["additional_surfaces"][0]
    points, triangles = _points(dots)
    tri = points[triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centre = _centres(dots, 4)[triangles[:, 0] // (len(points) // 4)]
    outward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - centre)
    assert (outward[np.linalg.norm(normal, axis=1) > 1e-9] > 0).all()


@pytest.mark.parametrize(("size", "radius"), [(None, 0.5), (0.2, 0.2)])
def test_a_dot_is_half_the_median_edge_unless_its_size_is_declared(tmp_path, drawn, size, radius):
    _draw(tmp_path, {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}, **({"marker_size": size} if size else {}))
    dots = drawn[0]["additional_surfaces"][0]
    points, _ = _points(dots)
    first = points[: len(points) // 4]
    assert np.isclose(np.linalg.norm(first - _centres(dots, 4)[0], axis=1).max(), radius)


def test_the_mesh_takes_a_declared_colour(tmp_path, drawn):
    _draw(tmp_path, {"r0": 1.0, "r1": 2.0, "r2": 3.0, "r3": 4.0}, color="#ff0000")
    assert drawn[0]["color"] == "#ff0000" and drawn[0].get("overlay") is None


def test_markers_without_a_layer_are_refused(tmp_path, drawn):
    with pytest.raises(ValueError, match="markers"):
        _draw(tmp_path, {})


def _colour_counts(fig, ax):
    """Red and blue pixels in the whole panel, and the share of page-coloured pixels in its central ninth."""
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    red = (rgb[..., 0] > 200) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    blue = (rgb[..., 2] > 200) & (rgb[..., 0] < 80) & (rgb[..., 1] < 80)
    x0, y0, x1, y1 = ax.get_window_extent().extents
    height = rgb.shape[0]
    rows = slice(int(height - (2 * y1 + y0) / 3), int(height - (y1 + 2 * y0) / 3))
    centre = rgb[rows, int((2 * x0 + x1) / 3) : int((x0 + 2 * x1) / 3)]
    page = np.asarray(matplotlib.colors.to_rgb(fig.get_facecolor())) * 255
    return int(red.sum()), int(blue.sum()), float((np.abs(centre - page).max(axis=-1) < 8).mean())


@pytest.mark.parametrize(("view", "seen"), [("lateral", "blue"), ("medial", "red")])
def test_the_mesh_hides_the_dots_on_its_far_side(tmp_path, view, seen):
    """A closed mesh, red dots on its +x half and blue on its -x half: seen from one side, only that side's colour shows, as dots with the page-coloured mesh between them. Only dots well inside the outline are coloured, since a dot sitting on the surface near it pokes past the outline from either side, as in any depth-buffered render."""
    vertices, faces = create_sphere_mesh((0.0, 0.0, 0.0), 50.0, resolution=16)
    x = vertices[:, 0]
    values = np.where(x > 30, 1.0, np.where(x < -30, -1.0, 0.0))
    fig, _ax = _draw(
        tmp_path,
        {f"r{i}": v for i, v in enumerate(values)},
        vertices=vertices,
        faces=faces[:, ::-1],
        view=view,
        cmap="bwr",
        marker_size=4.0,
    )
    red, blue, between = _colour_counts(fig, _ax)
    plt.close(fig)
    shown, hidden = (blue, red) if seen == "blue" else (red, blue)
    assert shown > 200 and hidden == 0 and between > 0.3
