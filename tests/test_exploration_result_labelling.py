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
