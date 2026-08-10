"""Tests for the ``ExplorationResult`` labelling contract.

Every results payload carries named dims, whatever the producer handed over, so
consumers select by key rather than by position. Labelling does not reshape: the
payload keeps the shape the backend emitted, and ``as_grid()`` is what expands the
flat run axis into one dim per exploration axis. No path returns a bare array — a
payload that cannot be reshaped is still labelled.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo.data.types import ExplorationResult
from tvbo.utils import Bunch


C_VALS = np.array([0.1, 0.2, 0.3])
W_VALS = np.array([1.0, 2.0])


def _axis(name, values):
    return Bunch(name=name, n=len(values), explored_values=np.asarray(values))


def test_single_axis_timeseries_is_labelled_by_parameter():
    """The leading run axis takes the swept parameter's name and values."""
    data = np.zeros((len(C_VALS), 50, 2, 1))
    r = ExplorationResult(
        name="sweep", results=data, axes=[_axis("model.c", C_VALS)],
        dt=0.1, output_names=["x", "y"],
    )
    assert isinstance(r.results, xr.DataArray)
    assert r.results.dims == ("model.c", "time", "variable", "node")
    np.testing.assert_allclose(r.results.coords["model.c"].values, C_VALS)
    assert list(r.results.coords["variable"].values) == ["x", "y"]


def test_labelling_preserves_shape():
    """Labels are added without reshaping, so positional consumers still work."""
    data = np.zeros((len(C_VALS), 50, 2, 1))
    r = ExplorationResult(
        name="sweep", results=data, axes=[_axis("model.c", C_VALS)],
        dt=0.1, output_names=["x", "y"],
    )
    assert r.results.shape == data.shape
    assert np.asarray(r.results).shape == data.shape


def test_multi_axis_results_flat_and_grid_expands():
    """A multi-axis payload stays flat under ``point``; ``as_grid`` expands it."""
    n_grid = len(C_VALS) * len(W_VALS)
    data = np.zeros((n_grid, 50, 2, 1))
    r = ExplorationResult(
        name="sweep2", results=data,
        axes=[_axis("model.c", C_VALS), _axis("model.w", W_VALS)],
        dt=0.1, output_names=["x", "y"],
    )
    assert r.results.dims == ("point", "time", "variable", "node")
    grid = r.as_grid()
    assert grid.dims == ("model.c", "model.w", "time", "variable", "node")
    np.testing.assert_allclose(grid.coords["model.w"].values, W_VALS)


def test_trials_only_results_are_labelled():
    """A trials-only ensemble has no swept axis but is still labelled."""
    data = np.zeros((4, 50, 2, 1))
    r = ExplorationResult(
        name="ICs", results=data, axes=[], dt=0.1,
        output_names=["x", "y"], n_trials=4,
    )
    assert r.results.dims == ("trial", "time", "variable", "node")
    assert r.as_grid().dims == ("trial", "time", "variable", "node")


def test_scalar_results_are_labelled():
    """Scalar-per-point results are labelled too, and keep ``optimal`` tracking."""
    r = ExplorationResult(
        name="loss", results=np.array([3.0, 1.0, 2.0]),
        axes=[_axis("model.c", C_VALS)],
    )
    assert isinstance(r.results, xr.DataArray)
    assert r.results.dims == ("model.c",)
    assert r.optimal.flat_index == 1
    assert r.optimal.value == pytest.approx(1.0)


def test_as_grid_never_returns_a_bare_array():
    """A payload that cannot be reshaped into the grid is still labelled.

    The grid shape here disagrees with the payload's leading dim, so the reshape is
    skipped — previously that fell back to the raw array, handing consumers
    positional data with no indication anything had gone wrong.
    """
    data = np.zeros((7, 50, 2, 1))  # 7 does not match the 3-point axis
    r = ExplorationResult(
        name="mismatch", results=data, axes=[_axis("model.c", C_VALS)],
        dt=0.1, output_names=["x", "y"],
    )
    grid = r.as_grid()
    assert isinstance(grid, xr.DataArray)
    assert grid.dims[0] == "point"


def test_none_results_stay_none():
    r = ExplorationResult(name="empty", results=None, axes=[])
    assert r.results is None
    assert r.as_grid() is None


# ── Declared observation axes ───────────────────────────────────────────────────────────


def _stacked(shape, dims=None, ts=None, cell_coords=None):
    from tvbo.data.types import _stacked_to_dataarray

    return _stacked_to_dataarray(
        np.zeros(shape), [_axis("model.c", C_VALS)], intrinsic_ts=ts,
        name="obs", cell_coords=cell_coords, dims=dims,
    )


def test_declared_dims_name_a_swept_observation():
    """A streamed observation's axes come from what it DECLARED, not from a template.

    `(time, node)` and the positional `(node, mode)` fallback have the same rank, so
    nothing raises when the guess is wrong — a 1,338-frame BOLD time axis simply comes
    back named `node`, and every downstream `.sel` is then keyed on the wrong axis.
    """
    da = _stacked((len(C_VALS), 1338, 200), dims=("time", "node"))
    assert da.dims == ("model.c", "time", "node")


def test_declared_dims_win_over_a_matching_time_vector():
    da = _stacked((len(C_VALS), 4, 4), dims=("node", "node_j"), ts=np.arange(4.0))
    assert da.dims == ("model.c", "node", "node_j")
    assert "time" not in da.coords


def test_undeclared_observations_keep_the_positional_fallback():
    """Payloads that declare nothing (a raw swept trajectory) are unchanged."""
    da = _stacked((len(C_VALS), 50, 2), ts=np.arange(50.0))
    assert da.dims == ("model.c", "time", "variable")


def test_declared_dims_apply_on_the_sharded_point_path():
    da = _stacked((2, 1338, 200), dims=("time", "node"),
                  cell_coords={"model.c": np.array([0.1, 0.3])})
    assert da.dims == ("point", "time", "node")
    assert list(da.coords["model.c"].values) == [0.1, 0.3]


def test_a_declared_shape_that_does_not_fit_falls_back_rather_than_raising():
    """A rank mismatch means the declaration is not about this payload; never mislabel."""
    da = _stacked((len(C_VALS), 50, 2), dims=("time",), ts=np.arange(50.0))
    assert da.dims == ("model.c", "time", "variable")


def test_a_declared_singleton_axis_is_not_squeezed_away():
    """A one-node observation still has a node axis — it declared one."""
    da = _stacked((len(C_VALS), 40, 1), dims=("time", "node"), ts=np.arange(40.0))
    assert da.dims == ("model.c", "time", "node")


def test_an_undeclared_trailing_singleton_is_still_squeezed():
    da = _stacked((len(C_VALS), 40, 2, 1), ts=np.arange(40.0))
    assert da.dims == ("model.c", "time", "variable")


def test_full_grid_is_keyed_by_value_when_space_order_differs_from_declared():
    """A full product whose cells arrive in the Space (pytree-leaf) order — NOT the declared
    axis order — is keyed into the grid BY VALUE, never by a positional reshape.

    When swept axes live on different state sub-objects (dynamics / coupling / graph), Space
    emits cells in pytree-leaf order, which differs from the declared ``axes_info`` order. A
    bare ``reshape(grid_sizes)`` then scrambles the surface (each cell reads another cell's
    value). ``cell_coords`` — the per-cell parameter values in the grid's own order — lets
    the assembler place each cell at the index its values map to.
    """
    from tvbo.data.types import _stacked_to_dataarray

    OM, K, V = [10.0, 20.0], [1e-6, 1e-5], [1.0, 10.0]
    axes = [_axis("Osc.omega", OM), _axis("Cpl.a", K), _axis("network.v", V)]
    # Space order = (K, omega, v): K slowest, v fastest — differs from the declared order.
    space_cells = [(o, k, v) for k in K for o in OM for v in V]

    def enc(o, k, v):
        return o * 1000.0 + (6 + np.log10(k)) * 100.0 + v  # unique per (omega, K, v)

    stacked = np.array([enc(o, k, v) for (o, k, v) in space_cells])
    cell_coords = {
        "Osc.omega": [o for (o, k, v) in space_cells],
        "Cpl.a": [k for (o, k, v) in space_cells],
        "network.v": [v for (o, k, v) in space_cells],
    }
    da = _stacked_to_dataarray(stacked, axes, name="obs", cell_coords=cell_coords)
    assert da.dims == ("Osc.omega", "Cpl.a", "network.v")
    for o in OM:
        for k in K:
            for v in V:
                got = float(da.sel({"Osc.omega": o, "Cpl.a": k, "network.v": v}).values)
                assert got == pytest.approx(enc(o, k, v)), (o, k, v, got)

    # Without cell_coords the same Space-order data is reshaped positionally and scrambles,
    # so at least one label reads the wrong cell — this is exactly the bug cell_coords fixes.
    bare = _stacked_to_dataarray(stacked, axes, name="obs")
    mism = sum(
        float(bare.sel({"Osc.omega": o, "Cpl.a": k, "network.v": v}).values) != enc(o, k, v)
        for o in OM for k in K for v in V
    )
    assert mism > 0
