"""A stimulus aimed at a state variable is injected into that state's derivative, not substituted for the state.

An event's `target_variable` names what the drive enters. Two cases have to stay apart. When it names a symbol the model's equations already carry (Deco's `I_external`), binding that symbol is exactly right. When it names a *state variable* — the only thing to aim at in a model written without an input term, such as `tvbo:SupHopf` — the tvboptim dfun used to bind it under the same name, so `x = state[0]` was overwritten by the stimulus and the generated model integrated `x_stim * (a - x_stim**2 - y**2)` in place of the cubic. That is silent: the run finishes, the numbers are plausible, and the model is not the declared one.
"""

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

DECAY_TO_DRIVE = {
    "id": 1,
    "label": "stimulus-on-a-state-variable fixture",
    "dynamics": {
        "name": "MiniDecay",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 0.5}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": 3},
    "integration": {"method": "heun", "step_size": 0.5, "duration": 200.0, "transient_time": 0.0, "unit": "ms"},
    "events": {
        "drive": {
            "event_type": "stimulus",
            "target_variable": "x",
            "equation": {"rhs": "amplitude"},
            "parameters": {"amplitude": {"value": 1.0}},
            "nodes": [1],
            "weights": [1.0],
        }
    },
    "execution": {"backend": "tvboptim"},
}


def _experiment(**event_overrides):
    spec = copy.deepcopy(DECAY_TO_DRIVE)
    spec["events"]["drive"].update(event_overrides)
    exp = SimulationExperiment(**spec)
    exp.configure()
    return exp


def test_the_state_variable_keeps_its_meaning_in_the_derivative():
    """The regression: `x = jnp.atleast_1d(external.x)[0]` sat between `x = state[0]` and the derivative that reads `x`."""
    code = _experiment().render_code("tvboptim")
    assert "x = state[0]" in code
    assert "\n        x = jnp.atleast_1d(external.x)" not in code
    assert "_ext_x = jnp.atleast_1d(external.x)" in code
    assert "dx_dt = (-a) * x + _ext_x" in code


def test_a_symbol_the_equations_already_carry_is_still_bound_plainly():
    """The other half of the contract: an event named after a model symbol binds that symbol, with nothing added to any derivative."""
    spec = copy.deepcopy(DECAY_TO_DRIVE)
    spec["dynamics"]["state_variables"]["x"]["equation"]["rhs"] = "-a * x + I_ext"
    spec["events"]["drive"]["target_variable"] = "I_ext"
    exp = SimulationExperiment(**spec)
    exp.configure()
    code = exp.render_code("tvboptim")
    assert "I_ext = jnp.atleast_1d(external.I_ext)" in code
    assert "dx_dt = (-a) * x + I_ext" in code


def test_the_driven_node_settles_where_the_equation_says_it_should():
    """End to end: `dx/dt = -a x + s` on the targeted node has the fixed point `s / a`, which a substituted state cannot produce."""
    result = _experiment().run("tvboptim")
    trajectory = np.asarray(result.data)[:, 0, :]
    assert trajectory[-1, 1] == pytest.approx(1.0 / 0.5, rel=1e-3)
    for node in (0, 2):
        assert np.ptp(trajectory[:, node]) < 1e-12
