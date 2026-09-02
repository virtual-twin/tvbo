"""`kind: surface` paints a layer on a mesh, with no study code at all.

A brain map is the most-drawn panel in a network-neuroscience paper, and until now every study registered its own `cortical_surface` callable to draw one — the same forty lines of mesh load, medial-wall masking and symmetric colour limits, re-derived per study, each free to get the limits subtly wrong. The kind is built in and its geometry comes from the network, which is where a mesh belongs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo.adapters.bsplot import CUSTOM_PANELS, _resolve_drawable, surface_panel


class _Layer:
    def __init__(self):
        self.container = None
        self.mark = self.transform = self.sel = self.style = None
        self.used = self.encoding = self.label = None


class _Panel:
    def __init__(self, kind="surface", opts=None, layers=(), render=None, **declared):
        self.kind = kind
        self.opts = opts or {}
        self.surface = self.volume = self.network = self.grid = self.colorbar = self.legend = None
        for name, value in declared.items():  # the kind objects a panel declares, as the schema holds them
            setattr(self, name, value)
        self.layers = list(layers)
        self.render = render
        self.label = self.placeholder = self.path = self.number = None
        self.number_loc = self.insets = self.annotations = None


@pytest.fixture
def mesh_npz(tmp_path):
    """A two-triangle mesh: four vertices is enough to check every branch."""
    path = tmp_path / "mesh.npz"
    np.savez(
        path,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2], [1, 3, 2]]),
    )
    return path


@pytest.fixture
def drawn(monkeypatch):
    """Capture the bsplot.plot_surf call instead of rendering it."""
    import bsplot

    calls = []
    monkeypatch.setattr(bsplot, "plot_surf", lambda **kw: calls.append(kw))
    return calls


def _ctx(tmp_path, values, **opts):
    layer = {"values": np.asarray(values, dtype=float)}
    return {"layers": [layer], "opts": opts, "key": "a", "base_dir": str(tmp_path)}


@pytest.fixture(autouse=True)
def stub_load_layer(monkeypatch):
    """Layer resolution is tested elsewhere; here the layer is its values."""
    import xarray as xr

    from tvbo.adapters import bsplot as adapter

    monkeypatch.setattr(adapter, "load_layer", lambda layer: xr.DataArray(layer["values"], dims=["vertex"]))


def test_registered_as_a_builtin_so_no_code_modules_are_needed():
    assert CUSTOM_PANELS["surface"] is surface_panel


def test_a_surface_panel_gets_a_ctx_and_the_builtin_render_name():
    """`kind: surface` resolves like `custom` — opts and base_dir reach the callable."""
    entry = _resolve_drawable(_Panel(surface={"view": "medial"}), "a", ".")
    assert entry["render"] == "surface"
    assert entry["ctx"]["opts"]["view"] == "medial"


def test_an_explicit_render_still_wins():
    """A study may keep a bespoke interior on the same kind."""
    entry = _resolve_drawable(_Panel(render="my_surface"), "a", ".")
    assert entry["render"] == "my_surface"


def test_values_are_painted_on_the_npz_mesh(tmp_path, mesh_npz, drawn):
    surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, -2.0, 0.5, 0.0], mesh=str(mesh_npz)))
    kw = drawn[0]
    assert kw["vertices"].shape == (4, 3) and kw["faces"].shape == (2, 3)
    np.testing.assert_allclose(kw["overlay"], [1.0, -2.0, 0.5, 0.0])


def test_colour_limits_are_symmetric_by_default(tmp_path, mesh_npz, drawn):
    """A signed map read on an off-centre scale misleads in a way no label catches."""
    surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, -2.0, 0.5, 0.0], mesh=str(mesh_npz)))
    assert (drawn[0]["vmin"], drawn[0]["vmax"]) == (-2.0, 2.0)


def test_explicit_limits_override_the_symmetric_default(tmp_path, mesh_npz, drawn):
    surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, -2.0, 0.5, 0.0], mesh=str(mesh_npz), vmin=0.0, vmax=1.0))
    assert (drawn[0]["vmin"], drawn[0]["vmax"]) == (0.0, 1.0)


def test_percentile_clips_a_single_outlier(tmp_path, mesh_npz, drawn):
    surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, 1.0, 1.0, 50.0], mesh=str(mesh_npz), percentile=75))
    assert drawn[0]["vmax"] < 50.0


def test_a_mask_greys_the_wall_and_leaves_it_out_of_the_range(tmp_path, mesh_npz, drawn):
    """A medial wall must not be coloured as a zero, nor set the colour range."""
    (tmp_path / "mask.txt").write_text("1\n1\n1\n0\n")
    surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, -0.5, 0.5, 99.0], mesh=str(mesh_npz), mask="mask.txt"))
    kw = drawn[0]
    assert bool(kw["mask"][3]) and not bool(kw["mask"][0])
    assert np.isnan(kw["overlay"][3])
    assert (kw["vmin"], kw["vmax"]) == (-1.0, 1.0)


def test_a_length_mismatch_names_the_two_counts(tmp_path, mesh_npz):
    with pytest.raises(ValueError, match="3 values for a mesh of 4 vertices"):
        surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, 2.0, 3.0], mesh=str(mesh_npz)))


def test_a_mask_of_the_wrong_length_is_refused(tmp_path, mesh_npz):
    """A mask sidecar for the wrong parcellation/hemisphere must fail with the two counts, not a cryptic broadcast error from `np.where`."""
    (tmp_path / "mask3.txt").write_text("1\n1\n0\n")  # 3 rows for a 4-vertex mesh
    with pytest.raises(ValueError, match="per-vertex 0/1 mask needs"):
        surface_panel(None, _Ax(), _ctx(tmp_path, [1.0, 2.0, 3.0, 4.0], mesh=str(mesh_npz), mask="mask3.txt"))


def test_an_out_of_range_vertex_index_names_the_mismatch():
    """A kept-index sidecar that does not match the mesh must be named, not raise a bare IndexError from the scatter."""
    import xarray as xr

    from tvbo.adapters.bsplot import _vertex_values

    da = xr.DataArray([1.0, 2.0], dims=["vertex"], coords={"vertex": [0, 99]})
    with pytest.raises(ValueError, match="reaches vertex 99"):
        _vertex_values(da, 4)


def test_a_mesh_source_must_be_declared(tmp_path):
    with pytest.raises(ValueError, match="declare where the mesh comes from"):
        surface_panel(None, _Ax(), _ctx(tmp_path, [1.0]))


def test_an_npz_without_the_named_arrays_is_refused(tmp_path):
    np.savez(tmp_path / "bad.npz", points=np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"missing \['faces', 'vertices'\]"):
        surface_panel(None, _Ax(), _ctx(tmp_path, [1.0], mesh="bad.npz"))


def test_a_network_without_a_mesh_says_so(tmp_path, monkeypatch):
    """The common miss: a network built from edges alone has no geometry to paint on."""
    import tvbo.classes.network as network_mod

    monkeypatch.setattr(network_mod.Network, "from_file", classmethod(lambda cls, path, **kw: object()))
    (tmp_path / "net.yaml").write_text("name: n\n")
    with pytest.raises(ValueError, match="carries no mesh"):
        surface_panel(None, _Ax(), _ctx(tmp_path, [1.0], network="net.yaml"))


class _Ax:
    """The two frame calls a surface makes; anatomy is not a coordinate system."""

    def __init__(self):
        self.aspect = None
        self.axis_state = None
        self.title = None

    def set_aspect(self, v):
        self.aspect = v

    def axis(self, v):
        self.axis_state = v

    def set_title(self, t):
        self.title = t


def test_the_frame_is_turned_off(tmp_path, mesh_npz, drawn):
    ax = _Ax()
    surface_panel(None, ax, _ctx(tmp_path, [1.0, 1.0, 1.0, 1.0], mesh=str(mesh_npz), title="V1"))
    assert ax.aspect == "equal" and ax.axis_state == "off" and ax.title == "V1"
