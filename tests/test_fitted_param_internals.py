"""A fitted parameter set is written to the container; the solver's scaffolding beside it is not.

A backend hands back its whole parameter pytree, which carries the step size, the history indices and the stage bookkeeping it needs to run alongside the values it actually fitted. Writing those states them as outcomes of the run. One of them is a subtree named ``time``, and no group in the result tree can be called that, because the container's ``time`` coordinate is inherited by every group — so the leak is not cosmetic: it makes the container unreadable as a tree.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.data.experiment_result_io import result_tree
from tvbo.data.types import ExperimentResult

FITTED = {
    "dynamics": {"a": np.array([0.06, 0.05]), "b": np.array([0.05, 0.05])},
    "_internal": {
        "time": {"dt": 0.1, "t0": 0.0, "t1": 1000.0},
        "coupling": {"DelayedSigmoidalJansenRit": {"dt": 0.1, "max_delay_steps": 42}},
    },
}


@pytest.fixture
def written(tmp_path):
    result = ExperimentResult(optimizations={"spectral_gradient_fit": SimpleNamespace(fitted_params=FITTED, final_loss=0.062)})
    paths = result.save(str(tmp_path), compress=False, record_only=False)
    h5 = [p for p in paths if p.endswith(".h5")]
    assert h5, f"expected an .h5 result, got {paths}"
    return h5[0]


def test_the_fitted_values_are_written(written):
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(written, engine="h5netcdf") as ds:
        assert "optimization__spectral_gradient_fit__fitted__dynamics__a" in ds.data_vars


def test_the_solver_scaffolding_is_not(written):
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(written, engine="h5netcdf") as ds:
        leaked = [str(n) for n in ds.data_vars if "_internal" in str(n)]
    assert leaked == [], f"the container states solver internals as results: {leaked}"


def test_the_container_still_reads_back_as_a_tree(written):
    """The failure the leak caused: a group named `time` collides with the inherited `time` coordinate."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(written, engine="h5netcdf") as ds:
        tree = result_tree(ds.load())
    assert "a" in tree.optimizations.spectral_gradient_fit.fitted.dynamics.dataset.data_vars
