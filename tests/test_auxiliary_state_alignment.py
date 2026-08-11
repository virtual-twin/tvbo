"""Recorded auxiliaries must describe the same instant as the recorded state.

A derived (auxiliary) variable is an algebraic function of the state, so at every recorded row it must equal that function evaluated at the state in the *same* row.
The native solver evaluates auxiliaries at the step-start state while recording the post-step state, which used to leave the auxiliary channel lagging the state by one integration step; the recorder re-evaluates the auxiliaries at the recorded state so the two align. Uses the pendulum, whose derived x = L·sin(theta) makes the alignment exact and independent of integrator order.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import Dynamics, SimulationExperiment

PENDULUM = """
name: PendulumSystem
parameters:
    c: {value: 0.001}
    omega0: {value: 0.01}
    L: {value: 1.0}
state_variables:
    theta:
        initial_value: 1.0
        equation: {rhs: omega}
    omega:
        initial_value: 0.0
        equation: {rhs: -c*omega - omega0**2 * sin(theta)}
derived_variables:
    x: {equation: {rhs: L * sin(theta)}}
    y: {equation: {rhs: -L * cos(theta)}}
output:
    - x
    - y
    - theta
    - omega
"""


@pytest.mark.parametrize("method,stages", [("Heun", 2), ("Euler", 1)])
def test_derived_variable_aligns_with_state(method, stages):
    """Both Euler and Heun record x on the same row as theta, and detectably not on the previous one.

    Without the shifted comparison, `same < 1e-12` is satisfied vacuously by a near-static run and a regression that reintroduced the lag would still pass; for a real motion the one-step shift error is orders of magnitude above the alignment tolerance.
    """
    exp = SimulationExperiment(dynamics=Dynamics.from_string(PENDULUM))
    exp.integration.method = method
    exp.integration.number_of_stages = stages
    g = exp.run(duration=50).integration.data

    theta = g.sel(variable="theta").values.ravel()
    x = g.sel(variable="x").values.ravel()
    y = g.sel(variable="y").values.ravel()

    same = np.max(np.abs(x - np.sin(theta)))
    lagged = np.max(np.abs(x[1:] - np.sin(theta[:-1])))
    assert same < 1e-12, f"{method}: x[t] != L*sin(theta[t]) (max {same:.2e})"
    assert lagged > 1e-6, f"{method}: trajectory too static to detect a one-step lag (lagged={lagged:.2e})"
    assert np.max(np.abs(y + np.cos(theta))) < 1e-12


COUPLED = {
    "label": "coupling-dependent-aux",
    "dynamics": {
        "name": "MiniOsc",
        "output": ["x", "recv"],
        "parameters": {"a": {"value": 0.5}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "x": {
                "equation": {"rhs": "-a*x + c"},
                "initial_value": 0.3,
                "coupling_variable": True,
            }
        },
        "derived_variables": {"recv": {"equation": {"rhs": "c"}}},
    },
    "network": {
        "number_of_nodes": 2,
        "edges": [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 1, "target": 0, "weight": 1.0},
        ],
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["x"],
                "pre_expression": {"rhs": "x"},
                "post_expression": {"rhs": "gx"},
            }
        },
    },
    "integration": {
        "method": "heun",
        "step_size": 0.1,
        "duration": 8.0,
        "transient_time": 0.0,
        "unit": "s",
    },
}
"""A 2-node network whose derived variable `recv` records the coupling input c = W @ x directly, so the recorded auxiliary is *coupling-dependent*. With W = [[0, 1], [1, 0]] the true coupling at any instant is W @ x at that instant, which lets us check exactly which state row the recorded coupling aligns with."""


def _run_coupled(coupling_evaluation):
    """Run the coupled model; return (W, x, recv) as (node, node), (time, node)."""
    spec = copy.deepcopy(COUPLED)
    spec["integration"]["coupling_evaluation"] = coupling_evaluation
    exp = SimulationExperiment(**spec)
    g = exp.run("tvboptim").integration.data
    W = np.asarray(exp.network.weights)
    x = np.atleast_2d(g.sel(variable="x").values)
    recv = np.atleast_2d(g.sel(variable="recv").values)
    if x.shape[0] < x.shape[1]:  # ensure (time, node)
        x, recv = x.T, recv.T
    return W, x, recv


@pytest.mark.xfail(
    reason="Coupling-dependent aux alignment under per_stage needs the coupling at the "
    "recorded (post-step) state, which only exists inside the tvboptim solver scan. The "
    "tvbo-side post-solve recompute (utils.state_only_recorded_aux) deliberately handles "
    "only state-only derived variables; the coupling-dependent case is fixed in the "
    "solver on the tvboptim `feat/auxiliary-state-alignment` branch. Remove this marker "
    "once that lands and the pin is bumped.",
    strict=False,
)
def test_coupling_dependent_aux_aligns_under_per_stage():
    """A coupling-dependent recorded auxiliary aligns with the recorded state's coupling only when the coupling is re-evaluated per stage (the §4.1e axiom).

    ``recv`` records c = W @ x. Under ``per_stage`` the recorded auxiliary equals ``W @ x[t]`` (the coupling at the recorded state); it must *not* equal the shifted ``W @ x[t-1]``, or the alignment would be indistinguishable from lag.

    Currently ``xfail``: the tvbo-side fix aligns only state-only derived variables (a coupling-dependent aux needs the in-scan coupling). See the marker for the tvboptim branch that resolves it.
    """
    W, x, recv = _run_coupled("per_stage")
    Wx = x @ W.T  # (W @ x[t])_i for every t
    same = np.max(np.abs(recv[1:] - Wx[1:]))
    lagged = np.max(np.abs(recv[1:] - Wx[:-1]))
    assert np.ptp(recv) > 1.0, "coupling has no dynamic range; the test is vacuous"
    assert same < 1e-10, f"per_stage: recv[t] != W@x[t] (max {same:.2e})"
    assert lagged > 1e-3, "per_stage: recv also matches the shifted state — not truly aligned"


def test_coupling_dependent_aux_lags_under_per_step():
    """Under the default per-step (frozen) coupling, the recorded coupling-dependent auxiliary carries the step-start coupling and lags the recorded state by one step — the documented limitation (§4.1e) that ``per_stage`` resolves."""
    W, x, recv = _run_coupled("per_step")
    Wx = x @ W.T
    same = np.max(np.abs(recv[1:] - Wx[1:]))
    lagged = np.max(np.abs(recv[1:] - Wx[:-1]))
    assert np.ptp(recv) > 1.0, "coupling has no dynamic range; the test is vacuous"
    assert lagged < 1e-10, f"per_step: recv[t] != W@x[t-1] (max {lagged:.2e})"
    assert same > 1e-3, "per_step: recv unexpectedly aligned with the same-step state"
