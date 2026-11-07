"""A per-node aggregation over a model variable is stored on labelled axes, and its streamed twin folds the same window.

Reduced after the scan, the value is ``(variable, node)``: a length-one variable axis, one column per node. Left undeclared, the exploration labeller would right-align a ``(node, mode)`` template onto it, naming the variable axis ``node`` and the nodes ``mode`` with no labels, so ``sel: {node: PFC}`` would find nothing while positional reads still worked. The streamed twin of the same observation folds the same trailing window, not the whole run, so both are checked here against the closed-form trajectory.
"""

import glob

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.templates.tvboptim.utils import observation_dims  # noqa: E402

_SPEC = """
id: 1
dynamics:
  name: TwoRate
  label: TwoRate
  parameters:
    tau: {value: 5.0}
    drive: {value: 1.0, heterogeneous: true, shape: "(n_nodes,)"}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(drive - x) / tau"}
      coupling_variable: true
    y:
      initial_value: 1.0
      equation: {rhs: "(x - y) / tau"}
  output: [x, y]
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: PPC, parameters: {drive: {value: 1.0}}}
    - {id: 1, label: PFC, parameters: {drive: {value: 3.0}}}
  edges: []
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
observations:
  x_last: {source: [x], aggregation: last STREAM}
  x_tail: {source: [x], aggregation: mean, tail_duration: 10.0 STREAM}
EXPLORE
execution: {random_seed: 0, precision: float64}
"""

_EXPLORE = """explorations:
  sweep:
    space:
      TwoRate.tau: {explored_values: [5.0, 10.0]}
"""

_DRIVE = {"PPC": 1.0, "PFC": 3.0}


def _spec(explored=True, streamed=False):
    return _SPEC.replace("EXPLORE\n", _EXPLORE if explored else "").replace(
        " STREAM", ", reduce: streaming" if streamed else ""
    )


def _saved(spec, tmp_path, key):
    SimulationExperiment.from_string(spec).run("tvboptim").save(tmp_path / key)
    (path,) = glob.glob(str(tmp_path / key / "**" / "*result.h5"), recursive=True)
    return xr.open_dataset(path, engine="h5netcdf").load()


def _euler_x(drive, tau, n):
    """``x`` after ``n`` Euler steps of ``dx/dt = (drive - x) / tau`` from 0 at ``dt = 1``."""
    return drive * (1.0 - (1.0 - 1.0 / tau) ** np.asarray(n, dtype=float))


def _expected(name, node, tau):
    samples = np.arange(1, 51)
    window = samples[-10:] if name == "x_tail" else samples[-1:]
    return _euler_x(_DRIVE[node], tau, window).mean()


def test_post_scan_aggregations_declare_variable_and_node():
    dims = observation_dims(SimulationExperiment.from_string(_spec()))
    assert dims["x_last"] == dims["x_tail"] == ("variable", "node")


@pytest.mark.parametrize("name", ["x_last", "x_tail"])
def test_an_explored_aggregation_is_selectable_by_node_label(tmp_path, name):
    da = _saved(_spec(), tmp_path, "explored")[name]
    assert da.sizes["variable"] == 1 and list(da["node"].values) == ["PPC", "PFC"]
    assert "mode" not in da.dims


@pytest.mark.parametrize("explored", [True, False], ids=["explored", "unexplored"])
@pytest.mark.parametrize("streamed", [False, True], ids=["post_scan", "streamed"])
@pytest.mark.parametrize("name", ["x_last", "x_tail"])
def test_the_labelled_value_is_the_declared_window(tmp_path, name, streamed, explored):
    ds = _saved(_spec(explored=explored, streamed=streamed), tmp_path, "run")
    da = ds[name] if explored else ds[f"observation__{name}"]
    taus = (5.0, 10.0) if explored else (5.0,)
    for node in _DRIVE:
        np.testing.assert_allclose(
            np.ravel(da.sel(node=node).values), [_expected(name, node, tau) for tau in taus], rtol=1e-12
        )


_STREAMED_FC = """  x_fc:
    source: [x]
    tail_duration: 10.0
    reduce: streaming
    pipeline:
      - callable: {name: compute_fc, module: tvboptim.observations.observation}
        arguments: {timeseries: {value: x}}
"""


def test_a_tail_on_a_reducer_that_cannot_honour_it_is_refused():
    spec = _spec(explored=False).replace("  x_last:", _STREAMED_FC + "  x_last:")
    with pytest.raises(Exception, match="trailing window"):
        SimulationExperiment.from_string(spec).run("tvboptim")
