"""Recorded auxiliaries must describe the same instant as the recorded state.

A derived (auxiliary) variable is an algebraic function of the state, so at every
recorded row it must equal that function evaluated at the state in the *same* row.
The native solver evaluates auxiliaries at the step-start state while recording the
post-step state, which used to leave the auxiliary channel lagging the state by one
integration step; the recorder re-evaluates the auxiliaries at the recorded state so
the two align. Uses the pendulum, whose derived x = L·sin(theta) makes the alignment
exact and independent of integrator order.
"""

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
    exp = SimulationExperiment(dynamics=Dynamics.from_string(PENDULUM))
    exp.integration.method = method
    exp.integration.number_of_stages = stages
    g = exp.run(duration=50).integration.data

    theta = g.sel(variable="theta").values.ravel()
    x = g.sel(variable="x").values.ravel()
    y = g.sel(variable="y").values.ravel()

    # Same-row identity holds exactly; a one-step-shifted comparison must not.
    same = np.max(np.abs(x - np.sin(theta)))
    lagged = np.max(np.abs(x[1:] - np.sin(theta[:-1])))
    assert same < 1e-12, f"{method}: x[t] != L*sin(theta[t]) (max {same:.2e})"
    assert lagged > same, f"{method}: aux still aligns to the shifted state (lag not fixed)"
    assert np.max(np.abs(y + np.cos(theta))) < 1e-12
