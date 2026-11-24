"""tvboptim draws a stimulus parameter declared with `axis: time` by the reading every drawn parameter shares.

The per-step draw resolves its family, bounds, moments and seed through `start_draw` on the run's seed, as a parameter start does: a normal is drawn as a normal, a distribution without a seed follows the run's, and a family, bound or moment no draw takes is refused by name. The runs integrate ``dx/dt = u(t)`` under Euler (`STIMULATED`), so each step's increment over the step size is the sample that step read.
"""

from __future__ import annotations

import copy

import jax
import numpy as np
import pytest

from tvbo import SimulationExperiment

from .test_sweep_cell_settle import STIMULATED, _integration, _values


def _stimulated(distribution, seed=None):
    """`STIMULATED` with its stimulus parameter ``u`` declaring *distribution*, run under *seed* where given."""
    spec = copy.deepcopy(STIMULATED)
    spec["events"]["u"]["parameters"]["u"]["distribution"] = {**distribution, "axis": "time"}
    if seed is not None:
        spec["execution"] = {"random_seed": seed}
    return spec


def _samples(spec):
    """The sample each step of *spec*'s settle and window read, and the length of the sequence the stimulus pre-generates."""
    integ = _integration(spec)
    trajectory = np.concatenate([[0.0], _values(integ.transient).reshape(-1), _values(integ.data).reshape(-1)])
    dt = spec["integration"]["step_size"]
    return np.diff(trajectory) / dt, round((spec["integration"]["transient_time"] + spec["integration"]["duration"]) / dt) + 2


def _assert_drawn(spec, drawn):
    """Every step of *spec* read the step's entry of *drawn*, the sequence built from the draw's key and moments."""
    samples, n = _samples(spec)
    expected = np.asarray(drawn(n))[: samples.size]
    np.testing.assert_allclose(samples, expected, atol=5e-4)


def test_the_reservoir_driver_draws_as_before():
    """ReservoirComputing B's declaration (Uniform on [-1, 1], its own seed 7, `run_experimentB_mc.py:86-91`) draws the sequence it drew: uniform under ``jax.random.key(7)``."""
    spec = _stimulated({"name": "Uniform", "domain": {"lo": -1.0, "hi": 1.0}, "seed": 7}, seed=3)
    _assert_drawn(spec, lambda n: jax.random.uniform(jax.random.key(7), (n,), minval=-1.0, maxval=1.0))


def test_a_normal_stimulus_parameter_is_drawn_as_a_normal():
    """A Normal by its mean and std is drawn around the mean with that spread, under the run's seed since it declares none."""
    normal = {"name": "Normal", "parameters": {"mean": {"name": "mean", "value": 1.0}, "std": {"name": "std", "value": 0.5}}}
    _assert_drawn(_stimulated(normal, seed=11), lambda n: 1.0 + 0.5 * jax.random.normal(jax.random.key(11), (n,)))


def test_a_draw_without_its_own_seed_follows_the_run_seed():
    """A distribution that declares no seed is keyed by `execution.random_seed`."""
    spec = _stimulated({"name": "Uniform", "domain": {"lo": 0.0, "hi": 2.0}}, seed=11)
    _assert_drawn(spec, lambda n: jax.random.uniform(jax.random.key(11), (n,), minval=0.0, maxval=2.0))


@pytest.mark.parametrize(
    "distribution, message",
    [
        (
            {"name": "LogNormal", "domain": {"lo": 0.1, "hi": 3.0}},
            r"'u' of event 'u'.*no draw is implemented for the distribution 'LogNormal'",
        ),
        ({"name": "Uniform"}, r"'u' of event 'u'.*a uniform draw needs lo, hi, and the distribution declares no lo, hi"),
    ],
    ids=["other-family", "no-domain"],
)
def test_a_draw_the_shared_reading_refuses_is_refused_by_name(distribution, message):
    """A family no draw takes, and a uniform without the bounds it draws between, are refused naming the parameter and its event, as a parameter start's are."""
    with pytest.raises(ValueError, match=message):
        SimulationExperiment(**_stimulated(distribution)).render_code("tvboptim")
