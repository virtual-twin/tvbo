"""A surface panel's opacity, the spellings a surface parcel answers to, a network panel's weight-shaded connections, and how a quantity read beside the plotted one is cut to the plotted array's place."""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.colors import Colormap

bsplot = pytest.importorskip("bsplot")

from tests.test_figure_codegen import TAHER_BASE, _cartesian_figure, _three_node_network
from tests.test_surface_missing_color import _drawn
from tvbo.adapters import bsplot as adapter


def test_a_surface_panel_draws_at_its_declared_opacity(monkeypatch):
    """`opacity` reaches bsplot's `plot_surf` as its `transparency`; unset, the mesh is opaque."""
    assert _drawn(monkeypatch, {"opacity": 0.4})["transparency"] == 0.4
    assert _drawn(monkeypatch, {})["transparency"] == 1.0


def _parcellation(monkeypatch, keys, hemi):
    """`_surface_parcellation` over a surface atlas carrying `unknown`, `x` and `y` on each side, joined to atlas entities keyed by *keys*."""
    monkeypatch.setattr(bsplot, "load_atlas_labels", lambda **kw: (np.array([0, 1, 1, 2]), ["unknown", "x", "y"]))
    monkeypatch.setattr(adapter, "_atlas_crosswalk", lambda atlas, base_dir: ({key: key for key in keys}, {}))
    monkeypatch.setattr(adapter, "_surface_atlas", inspect.unwrap(adapter._surface_atlas))
    return adapter._surface_parcellation.__wrapped__("tvbo:atlases/Test", ".", "Desikan2006", "fsaverage", "164k", hemi)


def test_a_parcel_answers_to_each_spelling_of_its_name(monkeypatch):
    """`ctx-lh-x`, `ctx_lh_x` and a case variant of either name one parcel, so the join does not depend on how an atlas spells its regions."""
    labels, parcel_of = _parcellation(monkeypatch, ["ctx-lh-x", "ctx_lh_x", "CTX-LH-X"], "lh")
    np.testing.assert_array_equal(labels, [0, 1, 1, 2])
    assert parcel_of == {"ctx-lh-x": 1, "ctx_lh_x": 1, "CTX-LH-X": 1}


def test_a_bare_name_both_hemispheres_carry_is_refused_when_an_entity_is_keyed_by_it(monkeypatch):
    """Under `hemi: both` the bare `x` is a parcel of each side, so an entity keyed by it raises naming both sides rather than painting one of them; the sided spellings land on their own side, and the `unknown` both sides carry passes while no entity is keyed by it."""
    labels, parcel_of = _parcellation(monkeypatch, ["ctx-lh-x", "ctx-rh-x"], "both")
    np.testing.assert_array_equal(labels, [0, 1, 1, 2, 1000, 1001, 1001, 1002])
    assert parcel_of == {"ctx-lh-x": 1, "ctx-rh-x": 1001}
    with pytest.raises(ValueError, match=r"region 'x'.*sides lh, rh"):
        _parcellation(monkeypatch, ["x"], "both")


def test_edge_color_weight_shades_each_connection_by_its_weight(tmp_path, monkeypatch):
    """`edge_color: weight` with no per-edge layer hands bsplot a colour ramp read from each connection's `weight`, starting at zero, while the spheres keep their flat colour."""
    import bsplot.graph as surface_graph

    drawn = {}
    monkeypatch.setattr(surface_graph, "plot_network_on_surface", lambda G, **kw: drawn.update(G=G, **kw))
    opts = {"network": str(_three_node_network(tmp_path / "net.yaml")), "edge_percentile": 0.0, "edge_color": "weight"}
    fig, ax = plt.subplots()
    adapter.network_panel(fig, ax, {"opts": opts, "layers": [], "base_dir": str(tmp_path)})
    plt.close(fig)
    assert drawn["edge_color"] == "auto" and drawn["edge_data_key"] == "weight"
    assert isinstance(drawn["edge_cmap"], Colormap) and drawn["edge_vmin"] == 0.0
    assert drawn.get("node_data_key") is None, "the spheres are not shaded by the weights"


@pytest.fixture
def cut_to():
    """The emitted `_cut_to`, executed from a rendered figure script."""
    ns: dict = {}
    exec(compile(adapter.render_code(_cartesian_figure(), TAHER_BASE, "out.png"), "<figure>", "exec"), ns)
    return ns["_cut_to"]


def test_a_sibling_on_the_time_axis_is_cut_to_the_frame_within_half_a_step(cut_to):
    """A frame between two samples takes the nearer one, a trail takes the samples it holds, and a frame more than half a step past the recording raises rather than taking its last sample."""
    sibling = xr.DataArray(np.arange(10.0), dims="time", coords={"time": np.arange(10) * 0.1})
    assert cut_to(sibling, xr.DataArray(0.0, coords={"time": 0.32}).coords).item() == 3.0
    trail = xr.DataArray(np.zeros(3), dims="time", coords={"time": [0.0, 0.1, 0.2]})
    np.testing.assert_array_equal(cut_to(sibling, trail.coords).values, [0.0, 1.0, 2.0])
    with pytest.raises(KeyError):
        cut_to(sibling, xr.DataArray(0.0, coords={"time": 5.0}).coords)


def test_a_sibling_recorded_on_a_subset_of_nodes_raises(cut_to):
    """Node 1 is not among the nodes the sibling recorded, so no neighbouring node stands in for it."""
    sibling = xr.DataArray([10.0, 20.0, 30.0], dims="node", coords={"node": [0, 2, 4]})
    assert cut_to(sibling, xr.DataArray(0.0, coords={"node": 2}).coords).item() == 20.0
    with pytest.raises(KeyError):
        cut_to(sibling, xr.DataArray(0.0, coords={"node": 1}).coords)


def test_a_coord_of_the_same_length_with_other_values_is_cut_by_value(cut_to):
    """A sweep axis of the same length in another order is not aligned by position: each point takes the sibling's value at its own coordinate, and the sibling's own values leave it as it is."""
    sibling = xr.DataArray([10.0, 20.0, 30.0], dims="K", coords={"K": [1.0, 2.0, 3.0]})
    plotted = xr.DataArray(np.zeros(3), dims="K", coords={"K": [3.0, 1.0, 2.0]})
    np.testing.assert_array_equal(cut_to(sibling, plotted.coords).values, [30.0, 10.0, 20.0])
    assert cut_to(sibling, sibling.coords) is sibling
