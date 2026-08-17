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

from tvbo.data.types import ExplorationResult, _is_partial_shard
from tvbo.utils import Bunch


C_VALS = np.array([0.1, 0.2, 0.3])
W_VALS = np.array([1.0, 2.0])


def _axis(name, values):
    return Bunch(name=name, n=len(values), explored_values=np.asarray(values))


def test_single_axis_timeseries_is_labelled_by_parameter():
    """The leading run axis takes the swept parameter's name and values."""
    data = np.zeros((len(C_VALS), 50, 2, 1))
    r = ExplorationResult(
        name="sweep",
        results=data,
        axes=[_axis("model.c", C_VALS)],
        dt=0.1,
        output_names=["x", "y"],
    )
    assert isinstance(r.results, xr.DataArray)
    assert r.results.dims == ("model.c", "time", "variable", "node")
    np.testing.assert_allclose(r.results.coords["model.c"].values, C_VALS)
    assert list(r.results.coords["variable"].values) == ["x", "y"]


def test_labelling_preserves_shape():
    """Labels are added without reshaping, so positional consumers still work."""
    data = np.zeros((len(C_VALS), 50, 2, 1))
    r = ExplorationResult(
        name="sweep",
        results=data,
        axes=[_axis("model.c", C_VALS)],
        dt=0.1,
        output_names=["x", "y"],
    )
    assert r.results.shape == data.shape
    assert np.asarray(r.results).shape == data.shape


def test_multi_axis_results_flat_and_grid_expands():
    """A multi-axis payload stays flat under ``point``; ``as_grid`` expands it."""
    n_grid = len(C_VALS) * len(W_VALS)
    data = np.zeros((n_grid, 50, 2, 1))
    r = ExplorationResult(
        name="sweep2",
        results=data,
        axes=[_axis("model.c", C_VALS), _axis("model.w", W_VALS)],
        dt=0.1,
        output_names=["x", "y"],
    )
    assert r.results.dims == ("point", "time", "variable", "node")
    grid = r.as_grid()
    assert grid.dims == ("model.c", "model.w", "time", "variable", "node")
    np.testing.assert_allclose(grid.coords["model.w"].values, W_VALS)


def test_trials_only_results_are_labelled():
    """A trials-only ensemble has no swept axis but is still labelled."""
    data = np.zeros((4, 50, 2, 1))
    r = ExplorationResult(
        name="ICs",
        results=data,
        axes=[],
        dt=0.1,
        output_names=["x", "y"],
        n_trials=4,
    )
    assert r.results.dims == ("trial", "time", "variable", "node")
    assert r.as_grid().dims == ("trial", "time", "variable", "node")


def test_scalar_results_are_labelled():
    """Scalar-per-point results are labelled too, and keep ``optimal`` tracking."""
    r = ExplorationResult(
        name="loss",
        results=np.array([3.0, 1.0, 2.0]),
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
        name="mismatch",
        results=data,
        axes=[_axis("model.c", C_VALS)],
        dt=0.1,
        output_names=["x", "y"],
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
        np.zeros(shape),
        [_axis("model.c", C_VALS)],
        intrinsic_ts=ts,
        name="obs",
        cell_coords=cell_coords,
        dims=dims,
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
    da = _stacked((2, 1338, 200), dims=("time", "node"), cell_coords={"model.c": np.array([0.1, 0.3])})
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
        float(bare.sel({"Osc.omega": o, "Cpl.a": k, "network.v": v}).values) != enc(o, k, v) for o in OM for k in K for v in V
    )
    assert mism > 0


def test_full_grid_with_incomplete_cell_coords_raises():
    """Placement by value needs every declared axis in ``cell_coords`` — a missing one
    means the caller's coordinate readback failed (e.g. a seed axis whose grid column
    is the ``dynamics._noise_seed`` state leaf, not the declared label). Falling back
    to a positional reshape here is what once wrote a seed-major (seed, r_s) fan into
    an (r_s, seed)-labelled container, so the mismatch raises instead.
    """
    from tvbo.data.types import _stacked_to_dataarray

    axes = [_axis("model.c", [0.1, 0.2]), _axis("execution.random_seed", [0, 1, 2])]
    cell_coords = {
        "model.c": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        "dynamics._noise_seed": [0, 1, 2, 0, 1, 2],
    }
    with pytest.raises(ValueError, match="execution.random_seed"):
        _stacked_to_dataarray(np.arange(6.0), axes, name="obs", cell_coords=cell_coords)


def test_full_grid_with_colliding_cell_coords_raises():
    """Per-cell values that do not identify cells uniquely cannot place the grid."""
    from tvbo.data.types import _stacked_to_dataarray

    with pytest.raises(ValueError, match="uniquely"):
        _stacked_to_dataarray(
            np.zeros(2),
            [_axis("model.c", [0.1, 0.2])],
            name="obs",
            cell_coords={"model.c": [0.1, 0.1]},
        )


def _expl(cell_counts, axis_sizes, **kw):
    """An ExplorationResult carrying *cell_counts* cells over axes of *axis_sizes*."""
    axes = [Bunch(name=f"ax{i}", explored_values=np.arange(n, dtype=float), n=n) for i, n in enumerate(axis_sizes)]
    coords = {f"ax{i}": np.zeros(cell_counts) for i in range(len(axis_sizes))}
    return ExplorationResult(name="sweep", axes=axes, cell_coords=coords, **kw)


def test_a_whole_sweep_is_not_mistaken_for_an_hpc_shard():
    """`cell_coords` is set for every keyed sweep, so presence alone must not mean "shard".

    Reading it as a shard marker made `save()` skip the YAML provenance sidecar for
    every local sweep — silently, since the write is best-effort. The run then claimed
    to be self-describing while shipping only the .h5.
    """
    assert _is_partial_shard(_expl(6, [2, 3])) is False


def test_a_partial_slice_is_a_shard():
    """Fewer cells than the Cartesian product is what actually makes a run one task's slice."""
    assert _is_partial_shard(_expl(2, [2, 3])) is True


def test_an_undecidable_exploration_defaults_to_writing_provenance():
    """No axes means the cell count proves nothing; a redundant sidecar beats a missing one."""
    assert _is_partial_shard(ExplorationResult(name="sweep", cell_coords={"ax0": np.zeros(3)})) is False
    assert _is_partial_shard(ExplorationResult(name="sweep")) is False


def test_the_producer_declaration_beats_the_cell_count():
    """A branch shard's axis `n` comes from the already-sliced index, so counting says "whole run".

    Only the generated script knows — it holds `kwargs['shard']` — so a declared
    `is_shard` wins over the fallback in both directions.
    """
    assert _is_partial_shard(_expl(6, [2, 3], is_shard=True)) is True
    assert _is_partial_shard(_expl(2, [2, 3], is_shard=False)) is False


def test_scalar_cell_coords_do_not_take_the_run_down_with_them():
    """`is_shard` is computed before anything is written, so raising here discards the results."""
    scalar = ExplorationResult(name="sweep", axes=[Bunch(name="g", n=3)], cell_coords={"g": 0.5})
    assert _is_partial_shard(scalar) is False


def test_an_axis_is_read_however_the_producer_shaped_it():
    """Dict axes and values-only axes are both legal here, as they are everywhere else in the module."""
    as_dict = ExplorationResult(name="sweep", axes=[{"name": "g", "n": 20}], cell_coords={"g": np.zeros(5)})
    values_only = ExplorationResult(
        name="sweep", axes=[Bunch(name="g", explored_values=np.arange(20.0))], cell_coords={"g": np.zeros(5)}
    )
    assert _is_partial_shard(as_dict) is True
    assert _is_partial_shard(values_only) is True


def test_a_non_numeric_axis_still_labels_rather_than_raising():
    """Placement subtracts coordinates, which strings cannot do.

    Newly reachable: `cell_coords` is now set for every sweep, so a full grid over
    `integration.method` reaches the by-value placement it used to skip. The TypeError
    escaped `as_grid` entirely rather than falling back to the positional reshape.
    """
    from tvbo.data.types import _stacked_to_dataarray

    axes = [Bunch(name="integration.method", explored_values=np.array(["heun", "euler"], dtype=object), n=2)]
    coords = {"integration.method": np.array(["heun", "euler"], dtype=object)}
    da = _stacked_to_dataarray(np.zeros((2, 5)), axes, cell_coords=coords, name="obs")
    assert da is not None and da.dims[0] == "integration.method"


def _object_array(values):
    """An object-dtype array holding *values* — how a matrix-valued axis's points arrive."""
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


@pytest.mark.parametrize(
    "cells,grid",
    [
        (_object_array([np.full((2, 2), c) for c in (0.1, 0.2)]), np.arange(2)),
        (_object_array([np.zeros((2, 2))]), _object_array([np.zeros((2, 2)), np.ones((2, 2))])),
        (_object_array([np.zeros((3, 3))]), _object_array([np.zeros((2, 2)), np.ones((2, 2))])),
    ],
    ids=["index-declared-grid", "matrices-both-sides", "mismatched-width"],
)
def test_array_valued_points_are_refused_and_name_the_upstream_conversion(cells, grid):
    """A grid coordinate holds scalars, so array-valued points cannot be placed here at all.

    An axis of whole matrices coordinates on the point INDEX, and only the generated script holds
    the materialised points to convert against — so the container refuses and names that
    conversion. Matching the matrices here instead would place the cells against a coordinate they
    do not share, and the surface would come out keyed on something no reader can select by.
    """
    from tvbo.data.types import _axis_positions

    with pytest.raises(ValueError, match="different currencies"):
        _axis_positions(cells, grid, "network.edges.length", "theta")


def test_point_indices_are_placed_through_the_ordinary_numeric_path():
    """Converted upstream, a matrix axis is just an integer axis, placed by value like any other.

    Fed a scrambled cell order, the placement recovers the declared order rather than the arrival
    order — the property the whole by-value rule exists for.
    """
    from tvbo.data.types import _axis_positions

    order = [2, 0, 4, 1, 3]
    assert list(_axis_positions(np.asarray(order), np.arange(5), "network.edges.length", "theta")) == order
