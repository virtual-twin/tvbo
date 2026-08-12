"""An `execution.random_seed` axis must have something that consumes the seed.

The axis reseeds the stochastic solver's PRNG key (``config.noise.key``). On a
deterministic experiment there is no key to reseed, so the swept leaf is read by
nothing and every grid cell returns an identical result — a silent no-op that still
shows up in the result container as a genuine-looking ensemble dimension. Codegen
rejects it instead, so a recipe cannot quietly produce a fake trial ensemble.
"""

import copy

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

# Minimal single-node model with an instantaneous coupling; noise is added per-test.
MINI_EXP = {
    "id": 1,
    "label": "seed-axis consumer unit fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "x": {
                "equation": {"rhs": "-a * x + c"},
                "initial_value": 0.1,
                "coupling_variable": True,
            },
        },
    },
    "network": {
        "number_of_nodes": 2,
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["x"],
                "pre_expression": {"rhs": "x"},
                "post_expression": {"rhs": "gx_0"},
            }
        },
    },
    "integration": {
        "method": "heun",
        "step_size": 0.1,
        "duration": 1.0,
        "transient_time": 0.0,
        "unit": "s",
    },
    "explorations": {
        "seed_sweep": {
            "name": "seed_sweep",
            "mode": "product",
            "record": ["x"],
            "space": [
                {
                    "parameter": "execution.random_seed",
                    "domain": {"lo": 0, "hi": 3, "n": 4},
                }
            ],
        }
    },
}


def _with_noise(spec, sigma=0.01):
    spec = copy.deepcopy(spec)
    spec["integration"]["noise"] = {"parameters": {"sigma": {"value": sigma}}}
    return spec


def test_seed_axis_without_noise_is_rejected():
    """No noise => nothing reads the seed => codegen error, not identical cells."""
    with pytest.raises(ValueError, match="has no consumer"):
        SimulationExperiment(**copy.deepcopy(MINI_EXP)).render_code("tvboptim")


def test_seed_axis_error_names_the_axis_and_a_way_forward():
    """The message must identify the axis and point at a mechanism that does apply."""
    with pytest.raises(ValueError) as excinfo:
        SimulationExperiment(**copy.deepcopy(MINI_EXP)).render_code("tvboptim")
    message = str(excinfo.value)
    assert "execution.random_seed" in message
    assert "n_trials" in message


def test_seed_axis_with_noise_still_renders():
    """The positive control: with noise the axis has a consumer and codegen proceeds."""
    code = SimulationExperiment(**_with_noise(MINI_EXP)).render_code("tvboptim")
    assert "_noise_seed" in code
    assert "noise.key" in code


def test_no_seed_axis_is_unaffected_without_noise():
    """A deterministic experiment with no seed axis must still render."""
    spec = copy.deepcopy(MINI_EXP)
    spec["explorations"]["seed_sweep"]["space"] = [{"parameter": "MiniOsc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}}]
    code = SimulationExperiment(**spec).render_code("tvboptim")
    assert "grid_state.dynamics.a" in code


def test_seed_axis_is_rejected_under_a_strategy_that_bypasses_the_grid():
    """Noise alone is not enough: the seed must also reach the per-cell grid binding.

    nsga2 / warm-start / branch-analysis bodies never execute the grid-binding block that
    applies the swept seed, so the axis is inert there even on a stochastic experiment —
    the same fake ensemble, just arrived at a different way.
    """
    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["strategy"] = "nsga2"
    with pytest.raises(ValueError, match="has no consumer"):
        SimulationExperiment(**spec).render_code("tvboptim")


def test_two_axis_seed_sweep_maps_the_noise_seed_leaf_to_its_label():
    """A (parameter x seed) product keys results by value, and the seed axis's grid
    column is the ``dynamics._noise_seed`` state leaf — codegen must map that bare name
    onto the declared ``execution.random_seed`` label, or cell placement cannot find
    the axis and the container assembly refuses rather than scrambling.
    """
    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["space"].insert(0, {"parameter": "MiniOsc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}})
    code = SimulationExperiment(**spec).render_code("tvboptim")
    squeezed = "".join(code.split()).replace('"', "'")
    assert "_bare_to_label.setdefault('_noise_seed',str(_a.name))" in squeezed
