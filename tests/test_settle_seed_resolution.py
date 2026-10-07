"""The settle's noise stream is declared, not inherited.

A settle that runs as its own scan draws its own noise. Whether that draw is the same one the measured window uses is a real choice with a measurable consequence, so it is stated in the recipe rather than left to whichever scan happens to be prepared first:

    settle seed = noise.settle_seed   if set
                  else a stream derived from noise.seed, distinct from it

Setting ``settle_seed`` equal to ``seed`` therefore puts both scans on one stream, which is what a setup integrating the whole window from a single draw does. That case is reproducible on purpose: where the two scans take the same number of steps, a shorter draw is a prefix of a longer one, so the measured noise repeats the settle's sample for sample.
"""

import copy

import pytest

from tvbo.datamodel.schema import Noise

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

SPEC = {
    "id": 1,
    "label": "settle-seed fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x"}, "initial_value": 0.1}},
    },
    "integration": {
        "method": "heun",
        "step_size": 0.1,
        "duration": 1.0,
        "transient_time": 0.5,
        "unit": "s",
        "noise": {"parameters": {"sigma": {"value": 0.01}}},
    },
}


def _code(settle_seed=None):
    spec = copy.deepcopy(SPEC)
    if settle_seed is not None:
        spec["integration"]["noise"]["settle_seed"] = settle_seed
    return SimulationExperiment(**spec).render_code("tvboptim")


def test_settle_seed_is_absent_by_default():
    """Absent means "derive a distinct stream", which is a different statement from any particular integer -- so the slot must not acquire a default that silently picks one."""
    assert Noise().settle_seed is None
    assert Noise().seed == 42


def test_settle_seed_round_trips_and_may_equal_seed():
    """Equal to `seed` is the one value that reproduces a single-draw setup, so it has to be expressible rather than rejected as redundant."""
    assert Noise(settle_seed=7).settle_seed == 7
    shared = Noise(seed=42, settle_seed=42)
    assert shared.settle_seed == shared.seed


def test_an_undeclared_settle_seed_emits_no_override():
    """Absent must leave the settle on the stream its solver hands it. Emitting any key here would put the backend on a different draw than the upstream warm start it is checked against byte for byte, which is a guarantee worth more than decorrelating a leak both measurements found null."""
    code = _code()
    assert "_settle_key" not in code
    assert "_tr_state.noise.key" not in code


def test_a_declared_settle_seed_keys_the_settle_scan():
    """Declared, it reaches the settle scan and only the settle scan -- the measured window keeps its own key."""
    code = _code(settle_seed=7)
    assert "_settle_key = jax.random.key(jnp.asarray(7" in code
    assert "_tr_state.noise.key = _settle_key" in code
    assert "state.noise.key = _noise_key" in code


def test_settle_seed_equal_to_the_measured_seed_is_expressible():
    """The aliased case has to be reachable, not prohibited: it is what a setup integrating one draw across both scans does, and reproducing such a setup is the point of the slot."""
    code = _code(settle_seed=0)
    assert "_settle_key = jax.random.key(jnp.asarray(0" in code
