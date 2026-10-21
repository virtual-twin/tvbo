"""The numcont warm-up integrates by the ``initial_state.solver`` a continuation declares, and names a solver it cannot integrate by rather than settling by another.

The model is ``x' = -x`` from ``x = 1`` over two time units, so each reference is closed form: a fixed-step method multiplies the state by its stability polynomial at ``h`` every step, and an adaptive solver lands on ``exp(-2)`` within its tolerance.
"""

import numpy as np
import pytest

from tvbo import Continuation, Dynamics
from tvbo.adapters.numcont import _warmup_to_steady_state

H = 0.5
STEPS = 4
DECAY = Dynamics(
    name="Decay",
    parameters={"a": {"name": "a", "value": 1.0}},
    state_variables={"x": {"name": "x", "equation": {"rhs": "-a*x"}, "initial_value": 1.0}},
)


def _warm_up(solver=None):
    initial_state = {"method": "time_integration", "duration": H * STEPS}
    if solver is not None:
        initial_state["solver"] = solver
    return float(_warmup_to_steady_state(DECAY, Continuation(name="warm_up", initial_state=initial_state))[0])


@pytest.mark.parametrize(
    ("method", "factor"),
    [("Euler", 1 - H), ("Heun", 1 - H + H**2 / 2), ("rk4", 1 - H + H**2 / 2 - H**3 / 6 + H**4 / 24)],
)
def test_a_fixed_step_solver_steps_at_its_step_size(method, factor):
    assert _warm_up({"method": method, "step_size": H}) == pytest.approx(factor**STEPS, rel=1e-14, abs=0)


@pytest.mark.parametrize("solver", [None, {"method": "Dopri5"}, {"method": "RK23"}, {"method": "Radau"}], ids=str)
def test_an_adaptive_solver_settles_within_its_tolerance(solver):
    assert _warm_up(solver) == pytest.approx(np.exp(-H * STEPS), rel=1e-7, abs=0)


@pytest.mark.parametrize(
    ("solver", "refusal"),
    [
        ({"method": "Tsit5"}, "'Tsit5' is not a solver the numcont warm-up can integrate by"),
        ({"method": "Rodas5"}, "'Rodas5' is not a solver the numcont warm-up can integrate by"),
        ({"method": "Heun"}, "'Heun' is a fixed-step method, and initial_state.solver declares no step_size"),
        ({"method": "Dopri5", "step_size": 0.1}, "declares step_size 0.1 for Dopri5, an adaptive solver"),
    ],
    ids=["Tsit5", "Rodas5", "fixed-step-without-step", "adaptive-with-step"],
)
def test_a_solver_the_warm_up_cannot_honour_is_refused_by_name(solver, refusal):
    with pytest.raises(ValueError, match=refusal):
        _warm_up(solver)
