"""The Python backend steps by the declared method on tvboptim's clock, a network synchronously and a single node alike.

tvboptim is the reference: the graph runner (`tvbo.run.compgraph`) must record the state after step k at time k·dt, read every node's coupling from the states at the start of the step, and so follow tvboptim step for step on a coupled network; a single node's `Dynamics.run` must step by the same `integration_step` rather than by SciPy's adaptive `odeint`. Every run starts from the model's initial values, which both backends read.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo import SimulationExperiment

METHODS = ["Euler", "Heun", "RungeKutta4thOrder"]
W = np.array([[0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 5.0], [0.0, 6.0, 0.0, 0.5]]) / 4.0
"""Source-by-target: ``W[i, j]`` is the weight of the edge ``i → j``; every node has an afferent."""
G2D = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}


def _network(scale=1.0):
    edges = [
        {
            "source": i,
            "target": j,
            "directed": True,
            "parameters": {"weight": {"name": "weight", "value": float(W[i, j] * scale)}},
        }
        for i in range(4)
        for j in range(4)
        if W[i, j]
    ]
    linear = {
        "name": "Linear",
        "iri": "tvbo:Linear",
        "delayed": False,
        "parameters": {"a": {"name": "a", "value": 0.3}, "b": {"name": "b", "value": 0.05}},
    }
    return {"number_of_nodes": 4, "nodes": [{"id": k} for k in range(4)], "edges": edges, "coupling": {"Linear": linear}}


def _experiment(method, network=None, duration=10.0):
    spec = {"dynamics": G2D, "integration": {"method": method, "step_size": 0.05, "duration": duration}}
    if network is not None:
        spec["network"] = network
    return SimulationExperiment(**spec)


def _data(result):
    return next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )


def _by_time(data):
    """*data* as ``{time: states}``, the states of every variable and node recorded at that time."""
    times = np.round(np.asarray(data["time"].values, dtype=float), 9)
    states = np.stack([data.sel(variable=v).values.reshape(len(times), -1) for v in ("V", "W")], axis=1)
    return dict(zip(times, states, strict=True))


def _gap(a, b):
    """The largest |a - b| over the times both record, and how many they share."""
    common = sorted(set(a) & set(b))
    return max(float(np.max(np.abs(a[t] - b[t]))) for t in common), len(common)


@pytest.mark.backend_tvboptim
def test_the_graph_runner_records_on_tvboptims_clock():
    pytest.importorskip("tvboptim")
    times = {
        backend: _data(_experiment("Heun", _network(), duration=1.0).run(format=backend))["time"].values
        for backend in ("python", "tvboptim")
    }

    np.testing.assert_allclose(times["python"], times["tvboptim"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(times["python"][[0, -1]], [0.05, 1.0], rtol=0, atol=1e-12)


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("method", METHODS)
def test_the_graph_runner_follows_tvboptim_on_a_coupled_network(method):
    """Every node reads its neighbours' states at the start of the step, so the runner matches tvboptim at every recorded time; without the coupling it would not."""
    pytest.importorskip("tvboptim")
    python = _by_time(_data(_experiment(method, _network()).run(format="python")))
    tvboptim = _by_time(_data(_experiment(method, _network()).run(format="tvboptim")))
    uncoupled = _by_time(_data(_experiment(method, _network(scale=0.0)).run(format="tvboptim")))

    gap, shared = _gap(python, tvboptim)
    assert shared == 200 and gap < 1e-12
    assert _gap(python, uncoupled)[0] > 1e-3


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("method", METHODS)
def test_a_single_node_run_steps_by_the_declared_method(method):
    """`Dynamics.run` records ``u_0`` at t = 0 and the state after step k at k·dt, so it shares tvboptim's times from dt on and matches it there under the same method, and not under another."""
    pytest.importorskip("tvboptim")
    experiment = _experiment(method)
    python = _by_time(_single_node(experiment))
    tvboptim = _by_time(_data(_experiment(method).run(format="tvboptim")))
    euler = _by_time(_data(_experiment("Euler").run(format="tvboptim")))

    gap, shared = _gap(python, tvboptim)
    assert shared >= 199 and gap < 1e-12
    if method != "Euler":
        assert _gap(python, euler)[0] > 1e-6


def _single_node(experiment):
    """`Dynamics.run` of *experiment*'s model by its integration, as a ``(time, variable)`` DataArray."""
    ts = experiment.dynamics.run(format="python", integration=experiment.integration, dt=0.05, duration=10.0, save=False)
    return xr.DataArray(
        np.asarray(ts.data)[:, :, 0, 0],
        dims=["time", "variable"],
        coords={"time": np.asarray(ts.time), "variable": list(experiment.dynamics.state_variables)},
    )


def test_a_single_node_run_refuses_a_method_with_no_update_expression():
    experiment = _experiment("Dopri5")
    with pytest.raises(ValueError, match="'Dopri5' has none"):
        experiment.dynamics.run(format="python", integration=experiment.integration, dt=0.05, duration=1.0, save=False)
