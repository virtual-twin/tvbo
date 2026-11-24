"""The seed a run draws from has one definition, `run_seed`, and the tvboptim render reads it from its context instead of resolving ``execution.random_seed`` itself.

A noisy run is used because the render then states its seed in one place, the default of the noise key in ``run_simulation``.
"""

from __future__ import annotations

import copy
import re

import pytest

from tvbo import SimulationExperiment

NOISY = {
    "id": 1,
    "label": "noisy decay",
    "dynamics": {
        "name": "Decay",
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x"}, "initial_value": 0.1}},
    },
    "integration": {
        "method": "Heun",
        "step_size": 0.1,
        "duration": 1.0,
        "noise": {"parameters": {"sigma": {"value": 0.01}}},
    },
}

KEY = re.compile(r"_noise_key = jax\.random\.key\(\s*jnp\.asarray\((\d+) if _rs is None")
"""Across whitespace: the rendered module is laid out by black, which breaks this line."""


def _experiment(execution=None):
    spec = copy.deepcopy(NOISY)
    if execution is not None:
        spec["execution"] = execution
    return SimulationExperiment(**spec)


@pytest.mark.parametrize("execution", [{"random_seed": 11}, None], ids=["declared", "undeclared"])
def test_the_tvboptim_render_keys_its_run_with_run_seed(execution):
    from tvbo.adapters.base import run_seed

    experiment = _experiment(execution)
    assert set(KEY.findall(experiment.render_code("tvboptim"))) == {str(run_seed(experiment))}


def test_the_tvboptim_render_takes_its_seed_from_the_context():
    """Handed another seed, the render emits that one: the template reads the value it is given and resolves none of its own."""
    assert set(KEY.findall(_experiment({"random_seed": 11}).render_code("tvboptim", random_seed=5))) == {"5"}
