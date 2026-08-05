"""Seed resolution for stochastic draws: ``distribution.seed`` overrides ``execution.random_seed``.

One rule governs every distribution-driven draw (initial conditions, parameter samples):

    seed = distribution.seed   if set
           else execution.random_seed   (default 0)

so a distribution's own ``seed`` is a *local* override of the experiment-global
``execution.random_seed``, and both the single-trial and the ``n_trials`` initial-condition
paths resolve it identically. The ``n_trials`` sampler also keys each trial off
``fold_in(base, i)`` so a trial's IC is independent of ``n_trials`` (adding a trial never
reshuffles the existing ones).
"""

import copy

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

# Single-node model whose state variable carries an IC distribution; `n_trials` makes the
# emitter draw per-trial ICs from it.
BASE = {
    "id": 1,
    "label": "seed-resolution fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {
            "x": {
                "equation": {"rhs": "-a * x"},
                "initial_value": 0.1,
                "domain": {"lo": 0.0, "hi": 1.0},
            },
        },
    },
    "integration": {
        "method": "heun", "step_size": 0.1, "duration": 1.0,
        "transient_time": 0.0, "unit": "s",
    },
    "explorations": {
        "ens": {
            "name": "ens",
            "mode": "product",
            "n_trials": 3,
            "record": ["x"],
            "space": [{"parameter": "MiniOsc.a", "explored_values": [1.0]}],
        }
    },
}


def _spec(dist_seed=None, exec_seed=None):
    spec = copy.deepcopy(BASE)
    dist = {"name": "Uniform", "domain": {"lo": 0.0, "hi": 1.0}}
    if dist_seed is not None:
        dist["seed"] = dist_seed
    spec["dynamics"]["state_variables"]["x"]["distribution"] = dist
    if exec_seed is not None:
        spec["execution"] = {"random_seed": exec_seed}
    return spec


def test_distribution_seed_overrides_execution_random_seed(unwrapped):
    """A distribution's own seed wins over execution.random_seed, and each trial folds it in."""
    code = SimulationExperiment(**_spec(dist_seed=7, exec_seed=99)).render_code("tvboptim")
    assert "jax.random.key(7)" in code                          # distribution.seed, not 99
    assert "jax.random.fold_in(jax.random.key(7)" in code       # per-trial key = fold_in(key(seed), i)
    assert unwrapped("jax.vmap(_sample_ics)(jnp.arange(_n_trials))") in unwrapped(code)
    assert "jax.random.key(99)" not in code                     # execution seed is overridden


def test_ic_sampler_inherits_execution_random_seed_when_distribution_unseeded():
    """No distribution.seed => the ensemble inherits the experiment-global execution.random_seed."""
    code = SimulationExperiment(**_spec(dist_seed=None, exec_seed=99)).render_code("tvboptim")
    assert "jax.random.key(99)" in code


def test_ic_sampler_defaults_to_zero_when_no_seed_anywhere():
    """No distribution.seed and no execution block => the single global default, 0 (not 42)."""
    code = SimulationExperiment(**_spec(dist_seed=None, exec_seed=None)).render_code("tvboptim")
    assert "jax.random.key(0)" in code
    assert "jax.random.key(42)" not in code


def test_each_distributed_variable_uses_its_own_seed():
    """With two distributed state variables carrying distinct seeds, BOTH seeds appear — the
    ensemble keys each variable off its own seed, not only the first (regression for the
    n_trials sampler that used a single base key)."""
    spec = copy.deepcopy(BASE)
    svs = spec["dynamics"]["state_variables"]
    svs["x"]["distribution"] = {"name": "Uniform", "seed": 3, "domain": {"lo": 0.0, "hi": 1.0}}
    svs["y"] = {
        "equation": {"rhs": "-a * y"}, "initial_value": 0.2, "domain": {"lo": 0.0, "hi": 1.0},
        "distribution": {"name": "Uniform", "seed": 5, "domain": {"lo": 0.0, "hi": 1.0}},
    }
    spec["dynamics"]["output"] = ["x", "y"]
    code = SimulationExperiment(**spec).render_code("tvboptim")
    assert "jax.random.key(3)" in code and "jax.random.key(5)" in code
