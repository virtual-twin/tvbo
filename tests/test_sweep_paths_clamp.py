"""Every tvboptim solve of an experiment integrates with the experiment's own solver, so a declared clamp holds in its sweeps and analyses as in its run.

The adiabatic scan, the warm-started sweep and its per-value Lyapunov pass, the branch restart, and the Lyapunov, gradient and finite-difference analysis observations each built a bare solver of the declared method, dropping the `BoundedSolver` a clamped domain asks for and the noise covariance a correlated noise asks for. The model here drifts upward without bound (dx/dt = a + x, so its Lyapunov exponent is 1) and declares x clamped at 1, so an unclamped solve shows as x above 1 and an exponent near 1, and a clamped one pins x at 1, where the clip leaves no tangent to grow.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

HI = 1.0

DRIFT = {
    "id": 1,
    "label": "an upward drift clamped at x = 1",
    "dynamics": {
        "name": "Drift",
        "output": ["x"],
        "parameters": {"a": {"value": 0.5}},
        "state_variables": {
            "x": {"equation": {"rhs": "a + x"}, "initial_value": 0.0, "domain": {"lo": -10.0, "hi": HI, "enforce": "clamp"}}
        },
    },
    "integration": {"method": "Heun", "step_size": 0.01, "duration": 2.0, "transient_time": 0.0},
    "observations": {
        "x_peak": {"source": ["x"], "pipeline": [{"name": "peak", "callable": {"name": "max", "module": "jax.numpy"}}]},
        "lam": {
            "analysis": {
                "type": "lyapunov",
                "parameters": {"segment_time": {"value": 1.0}, "n": {"value": 3}, "k": {"value": 1}},
            }
        },
    },
}
"""x reaches 1 at t = ln 3 and, unclamped, about 3.2 at t = 2."""

SWEEP_AXIS = [{"parameter": "Drift.a", "domain": {"lo": 0.5, "hi": 1.0, "n": 2}}]

ADIABATIC = {
    "envelope": {
        "name": "envelope",
        "strategy": "adiabatic_scan",
        "space": SWEEP_AXIS,
        "parameters": {
            "signal": {"value": "x"},
            "segment_time": {"value": 2.0},
            "transient_time": {"value": 0.0},
            "bothways": {"value": 0},
        },
    }
}

WARMSTART = {
    "branch": {
        "name": "branch",
        "sweep_seeding": "from_previous",
        "sweep_direction": "up",
        "record": ["x_peak", "lam"],
        "space": SWEEP_AXIS,
    }
}


def _experiment(explorations=None):
    """The clamped drift, with *explorations* declared."""
    spec = copy.deepcopy(DRIFT)
    if explorations:
        spec["explorations"] = copy.deepcopy(explorations)
    return SimulationExperiment(**spec)


@pytest.mark.parametrize("explorations", [ADIABATIC, WARMSTART], ids=["adiabatic", "warmstart"])
def test_no_solve_builds_a_bare_solver(explorations):
    """Every scan and analysis solve the render emits takes `get_solver`, the solver the run integrates with."""
    code = _experiment(explorations).render_code("tvboptim")
    assert not re.search(r"(?:prepare\(_?network|_network),\s*Heun\(", code)
    assert "get_solver(solver_kwargs=" in code


def test_the_adiabatic_scan_holds_the_clamp():
    """The scan's envelope never passes the clamp."""
    envelope = _experiment(ADIABATIC).run("tvboptim").explorations["envelope"]
    assert float(np.max(np.asarray(envelope.observations["env_hi"]))) <= HI


def test_the_warmstart_sweep_holds_the_clamp():
    """Each swept value's settled rollout never passes the clamp."""
    branch = _experiment(WARMSTART).run("tvboptim").explorations["branch"]
    assert float(np.max(np.asarray(branch.observations["x_peak"]))) <= HI


def test_the_warmstart_lyapunov_pass_holds_the_clamp():
    """Seeded from a state pinned at the clamp, the per-value Lyapunov solve has no tangent to grow, where the unclamped drift has exponent 1."""
    branch = _experiment(WARMSTART).run("tvboptim").explorations["branch"]
    assert not np.any(np.asarray(branch.observations["lam"]) > 0.5)


def test_the_analysis_lyapunov_holds_the_clamp():
    """The run's own Lyapunov observation integrates on the clamped solver too."""
    result = _experiment().run("tvboptim", mode="simulation")
    assert not np.any(np.asarray(result.observations.lam) > 0.5)
