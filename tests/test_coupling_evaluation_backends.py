"""Each Python backend integrates the ``coupling_evaluation`` and method a recipe declares, or refuses the declaration by name.

``per_stage`` and ``per_step`` integrate different systems only for a multi-stage method over more than one node (`coupling_evaluation_in_effect`). tvboptim's native solvers honour both and are the reference here: the jax backend re-evaluates an undelayed coupling at each stage's state and must follow tvboptim under either declaration, while TVB and the Python graph runner compute the coupling once per step and refuse ``per_stage``. The graph runner steps each node by the declared method's own update expressions.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo import SimulationExperiment

W = np.array([[0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 5.0], [0.0, 6.0, 0.0, 0.5]]) / 4.0
"""Source-by-target: ``W[i, j]`` is the weight of the edge ``i → j``."""


def _experiment(method="Heun", evaluation=None, delayed=False, scale=1.0, duration=5.0):
    """Four Generic2dOscillator nodes from the model's initial values, coupled by Linear through `W` times *scale*; every edge declares a length, which the jax backend reads, and a delayed coupling runs at speed 1."""
    edges = [
        {
            "source": i,
            "target": j,
            "directed": True,
            "parameters": {
                "weight": {"name": "weight", "value": float(W[i, j] * scale)},
                "length": {"name": "length", "value": 2.0 if delayed else 0.0},
            },
        }
        for i in range(4)
        for j in range(4)
        if W[i, j]
    ]
    coupling = {
        "Linear": {
            "name": "Linear",
            "iri": "tvbo:Linear",
            "delayed": delayed,
            "parameters": {"a": {"name": "a", "value": 0.3}, "b": {"name": "b", "value": 0.0}},
        }
    }
    network = {
        "number_of_nodes": 4,
        "nodes": [{"id": k} for k in range(4)],
        "edges": edges,
        "coupling": coupling,
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": 1.0}},
    }
    integration = {"method": method, "step_size": 0.05, "duration": duration}
    if evaluation:
        integration["coupling_evaluation"] = evaluation
    return SimulationExperiment(
        dynamics={"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}, network=network, integration=integration
    )


def _states(experiment, backend):
    """*experiment*'s states on *backend* as ``(step, variable, node)``, the k-th row the state after the k-th step."""
    result = experiment.run(format=backend)
    data = next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )
    return np.stack([data.sel(variable=v).values.reshape(data.sizes["time"], -1) for v in ("V", "W")], axis=1)


def _gap(a, b):
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])))


def test_tvb_refuses_per_stage_where_it_changes_the_system():
    with pytest.raises(NotImplementedError, match="TVB's simulator computes it once per step"):
        _experiment("Heun", "per_stage").render_code(format="tvb")


def test_tvb_renders_per_stage_where_a_single_stage_makes_it_per_step():
    assert "def define_simulation" in _experiment("Euler", "per_stage").render_code(format="tvb")


@pytest.mark.parametrize(("evaluation", "restaged"), [("per_step", False), ("per_stage", True)])
def test_jax_reevaluates_the_coupling_at_each_stage_under_per_stage(evaluation, restaged):
    code = _experiment("RungeKutta4thOrder", evaluation).render_code(format="jax")

    assert all((f"cX{k} = jax.vmap(cfun" in code) is restaged for k in (1, 2, 3))
    assert all((f"dX{k} = dfun(X{k}, t, cX{k}, params_dfun)" in code) is restaged for k in (1, 2, 3))


def test_jax_refuses_per_stage_over_a_delayed_coupling():
    with pytest.raises(NotImplementedError, match="reads the delayed coupling Linear off its history buffer"):
        _experiment("Heun", "per_stage", delayed=True).render_code(format="jax")


@pytest.mark.backend_jax
@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("method", ["Heun", "RungeKutta4thOrder"])
def test_jax_follows_tvboptim_under_each_coupling_evaluation(method):
    """Both backends start from the model's initial values; the two declarations differ by far more than either pair does."""
    pytest.importorskip("tvboptim")
    runs = {(e, b): _states(_experiment(method, e), b) for e in ("per_step", "per_stage") for b in ("jax", "tvboptim")}

    for evaluation in ("per_step", "per_stage"):
        assert _gap(runs[evaluation, "jax"], runs[evaluation, "tvboptim"]) < 1e-12
    assert _gap(runs["per_step", "jax"], runs["per_stage", "jax"]) > 1e-6


def test_the_graph_runner_refuses_per_stage_and_a_method_with_no_update_expression():
    with pytest.raises(NotImplementedError, match="the Python graph runner computes each node's input once per step"):
        _experiment("Heun", "per_stage", duration=0.2).run(format="python")
    with pytest.raises(ValueError, match="'Dopri5' has none"):
        _experiment("Dopri5", duration=0.2).run(format="python")


@pytest.mark.backend_tvboptim
def test_the_graph_runner_steps_by_the_declared_method():
    """Uncoupled, so each node follows its own model and the runner's coupling path is out of the comparison: the runner matches tvboptim step for step under each method and not under another."""
    pytest.importorskip("tvboptim")
    runs = {
        m: (_states(_experiment(m, scale=0.0), "python"), _states(_experiment(m, scale=0.0), "tvboptim"))
        for m in ("Euler", "Heun", "RungeKutta4thOrder")
    }

    for python, tvboptim in runs.values():
        assert _gap(python, tvboptim) < 1e-12
    assert _gap(runs["Heun"][0], runs["Euler"][1]) > 1e-6
