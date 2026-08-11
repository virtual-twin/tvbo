"""Tests for the ``ExplorationAxis.reduce`` capability.

An exploration axis marked ``reduce`` is collapsed by a statistic in the result container: the axis's named grid dimension is reduced across every observation that carries it (keyed by dim name, never positional), the reduced observations keep their names, and the collapsed axis drops out of the shape metadata.
Observations that do not carry the dim are left untouched. Without ``reduce`` the result container is unchanged.
"""

import numpy as np
import xarray as xr
import pytest

from tvbo.data.types import ExplorationResult
from tvbo.utils import Bunch


MU_VALS = np.array([0.1, 0.2, 0.3])
SEED_VALS = np.array([0, 1, 2, 3])


def _axes(reduce_stat=None):
    """A swept param axis + an ensemble (random_seed) axis, optionally reduced."""
    seed_kw = {"reduce": reduce_stat} if reduce_stat is not None else {}
    return [
        Bunch(name="MurrayWangDM.mu", n=len(MU_VALS), explored_values=MU_VALS.copy()),
        Bunch(
            name="execution.random_seed",
            n=len(SEED_VALS),
            explored_values=SEED_VALS.copy(),
            **seed_kw,
        ),
    ]


def _observations():
    """Two labelled observations; only ``decision`` carries the seed dim."""
    rng = np.random.default_rng(0)
    decision = xr.DataArray(
        rng.standard_normal((len(MU_VALS), len(SEED_VALS), 5)),
        dims=("MurrayWangDM.mu", "execution.random_seed", "node"),
        coords={"MurrayWangDM.mu": MU_VALS, "execution.random_seed": SEED_VALS},
        name="decision",
    )
    # `rate` does NOT carry the random_seed dim -> must be left untouched.
    rate = xr.DataArray(
        rng.standard_normal((len(MU_VALS), 5)),
        dims=("MurrayWangDM.mu", "node"),
        coords={"MurrayWangDM.mu": MU_VALS},
        name="rate",
    )
    return {"decision": decision, "rate": rate}


def test_reduce_removes_dim_and_drops_axis():
    """The reduced dim disappears from carrying obs; the axis leaves the metadata."""
    obs = _observations()
    res = ExplorationResult(name="t", axes=_axes("mean"), observations=obs)

    dec = res.observations["decision"]
    assert "execution.random_seed" not in dec.dims
    assert dec.dims == ("MurrayWangDM.mu", "node")
    # The reduced axis is dropped from the metadata / shape.
    assert [ExplorationResult._axis_name(a) for a in res.axes] == ["MurrayWangDM.mu"]
    assert res._grid_shape == (len(MU_VALS),)


@pytest.mark.parametrize("stat", ["mean", "sum", "std", "median", "sem"])
def test_reduce_values_match_xarray_by_name(stat):
    """Reduced values equal the xarray reduction over that dim (keyed by name)."""
    ref = _observations()["decision"]
    res = ExplorationResult(name="t", axes=_axes(stat), observations=_observations())
    got = res.observations["decision"]

    dim = "execution.random_seed"
    if stat == "sem":
        expected = ref.std(dim=dim) / np.sqrt(ref.sizes[dim])
    else:
        expected = getattr(ref, stat)(dim=dim)

    np.testing.assert_allclose(got.values, expected.values)
    assert got.name == "decision"


def test_other_obs_and_labels_preserved():
    """Obs without the dim are untouched; surviving coords/labels are preserved."""
    obs = _observations()
    rate_before = obs["rate"].copy(deep=True)
    res = ExplorationResult(name="t", axes=_axes("mean"), observations=obs)

    # `rate` never carried the seed dim -> identical values and dims.
    rate_after = res.observations["rate"]
    assert rate_after.dims == ("MurrayWangDM.mu", "node")
    np.testing.assert_array_equal(rate_after.values, rate_before.values)

    # Surviving `mu` coordinate labels are preserved on the reduced obs.
    dec = res.observations["decision"]
    np.testing.assert_array_equal(dec.coords["MurrayWangDM.mu"].values, MU_VALS)


def test_reduce_on_internally_labelled_observations():
    """Raw stacked arrays get labelled by ``_stacked_to_dataarray`` then reduced.

    Exercises the production path where ExplorationResult builds the DataArray itself (grid dim named by the axis ``name``) before reducing by that dim name.
    """
    rng = np.random.default_rng(1)
    # Flat leading dim = prod(grid) = 3 * 4, followed by a node dim.
    raw = rng.standard_normal((len(MU_VALS) * len(SEED_VALS), 5))
    res = ExplorationResult(name="t", axes=_axes("mean"), observations={"decision": raw})
    dec = res.observations["decision"]
    # Grid dims are named by the axis names; the reduced one is gone while the surviving grid axis stays leading. (The lone trailing spatial dim is named by _stacked_to_dataarray's own convention — not under test here.)
    assert "execution.random_seed" not in dec.dims
    assert dec.dims[0] == "MurrayWangDM.mu"
    assert "MurrayWangDM.mu" in dec.dims

    # Compare against the same array labelled without reduction, then meaned.
    ref = ExplorationResult(name="t", axes=_axes(None), observations={"decision": raw.copy()}).observations["decision"]
    np.testing.assert_allclose(dec.values, ref.mean(dim="execution.random_seed").values)


def test_no_reduce_leaves_observations_unchanged():
    """Without `reduce`, observations and axes are byte-identical to the input."""
    obs = _observations()
    dec_before = obs["decision"].copy(deep=True)
    res = ExplorationResult(name="t", axes=_axes(None), observations=obs)

    dec = res.observations["decision"]
    assert dec.dims == ("MurrayWangDM.mu", "execution.random_seed", "node")
    np.testing.assert_array_equal(dec.values, dec_before.values)
    # Both axes retained; full grid shape intact.
    assert [ExplorationResult._axis_name(a) for a in res.axes] == [
        "MurrayWangDM.mu",
        "execution.random_seed",
    ]
    assert res._grid_shape == (len(MU_VALS), len(SEED_VALS))
