"""The phase locking value and the phase lag index as curated reducer recipes folded over the whole run, and as the curated host functions of the same quantities.

An observation marked ``reduce: streaming`` whose pipeline step names ``tvbo:functions/PhaseLockingValue`` or ``tvbo:functions/PhaseLagIndex`` streams through the ``phase_locking_value`` or ``phase_lag_index`` recipe (``emit_kind: run``): each sample of a node-wise phase after the transient folds into a node-by-node state, and the matrix is read out once at finalize, so no trajectory is held. These tests resolve such an observation, render its reducer from the observation template, feed it a phase trajectory in blocks as a grid cell does, and check analytic cases: a constant phase difference locks fully, a difference spread over the circle does not, a constant lag strictly inside (0, pi) has a PLI of 1 and a zero lag a PLI of 0, both matrices are symmetric, and the streamed value equals the curated host function on the same samples.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo import Function
from tvbo.codegen.streaming_reducers import is_windowed_reducer, lookup_streaming_reducer
from tvbo.datamodel.schema import Observation
from tvbo.templates.tvboptim.utils import resolve_reduction

pytestmark = pytest.mark.skipif(
    os.environ.get("TVBO_STREAMING_REDUCERS") == "0", reason="streaming reducers disabled (TVBO_STREAMING_REDUCERS=0)"
)

OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")
RECIPES = {"PhaseLockingValue": "phase_locking_value", "PhaseLagIndex": "phase_lag_index"}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _fold(function, phase, skip=0, block_len=None):
    """The streamed value of the recipe that replaces the curated *function*, over *phase* (time, node) recorded as a cell's second variable, folded in blocks of *block_len* samples with the first *skip* left out."""
    observation = Observation(
        name="sync", source=["theta"], reduce="streaming", pipeline=[{"iri": f"tvbo:functions/{function}"}]
    )
    red = resolve_reduction(observation)
    assert red["kind"] == "run"
    source = OBS_TEMPLATE.get_def("render_reduction").render(red=red, name="sync", s_idx=1, dt=1.0)
    namespace = {"jnp": jnp, "jax": jax}
    exec(compile(source, "<run_reduction>", "exec"), namespace)
    init, update, finalize = namespace["_reduction_sync"](skip=skip)
    trajectory = jnp.asarray(np.stack([np.zeros_like(phase), phase], axis=1))
    acc = init(trajectory[0], trajectory.shape[0])
    step = block_len or trajectory.shape[0]
    for start in range(0, trajectory.shape[0], step):
        acc = jax.jit(update)(acc, trajectory[start : start + step])
    return np.asarray(finalize(acc))


def _oscillation(n_samples):
    """The phase of a 10 Hz oscillation sampled every millisecond, in radians."""
    return 2 * np.pi * 10.0 * np.arange(n_samples) * 1e-3


def _random_walk(rng, n_samples=500, n_nodes=6):
    """Phases of *n_nodes* oscillations near 10 Hz, each drifting by its own random walk from its own start, so every pair has a lag that varies without spreading evenly."""
    drift = np.cumsum(rng.normal(scale=0.05, size=(n_samples, n_nodes)), axis=0)
    return _oscillation(n_samples)[:, None] + drift + rng.uniform(-np.pi, np.pi, n_nodes)


def test_a_constant_phase_difference_gives_plv_one():
    """Two nodes whose phases keep one difference lock fully: the PLV is 1 off the diagonal, and 1 on it."""
    theta = _oscillation(2000)
    plv = _fold("PhaseLockingValue", np.stack([theta, theta + 1.2], axis=1))
    np.testing.assert_allclose(plv, np.ones((2, 2)), rtol=0, atol=1e-12)


def test_a_phase_difference_spread_over_the_circle_gives_plv_near_zero():
    """A phase difference drawn uniformly over the circle at each of N samples gives a PLV below 5 / sqrt(N): the modulus of the mean of N independent unit phasors exceeds a / sqrt(N) with probability exp(-a**2), about 1e-11 at a = 5."""
    n_samples = 20_000
    theta = _oscillation(n_samples)
    spread = np.random.default_rng(3).uniform(-np.pi, np.pi, n_samples)
    plv = _fold("PhaseLockingValue", np.stack([theta, theta + spread], axis=1))
    assert plv[0, 1] < 5 / np.sqrt(n_samples)


@pytest.mark.parametrize("lag", [0.3, np.pi / 2, 3.0])
def test_a_constant_lag_strictly_inside_zero_and_pi_gives_pli_one(lag):
    """A phase difference held at one value strictly inside (0, pi) gives its sine one sign at every sample, so the PLI is 1 off the diagonal and 0 on it."""
    theta = _oscillation(3000)
    pli = _fold("PhaseLagIndex", np.stack([theta, theta + lag], axis=1))
    np.testing.assert_array_equal(pli, [[0.0, 1.0], [1.0, 0.0]])


def test_a_zero_lag_gives_pli_zero():
    """Two nodes with the same phase at every sample have no lag to take the sign of, so the PLI is exactly 0 everywhere, the diagonal included."""
    theta = _oscillation(3000)
    pli = _fold("PhaseLagIndex", np.stack([theta, theta.copy()], axis=1))
    np.testing.assert_array_equal(pli, np.zeros((2, 2)))


@pytest.mark.parametrize("function", list(RECIPES))
def test_the_matrix_is_symmetric(function, rng):
    """Both measures read two nodes' phase difference through a modulus, so the matrix equals its transpose to rounding."""
    value = _fold(function, _random_walk(rng))
    np.testing.assert_allclose(value, value.T, rtol=0, atol=1e-12)


@pytest.mark.parametrize("function", list(RECIPES))
@pytest.mark.parametrize("block_len", [7, 64, 500])
def test_the_whole_run_emit_equals_the_curated_function(function, block_len, rng):
    """Folded in blocks past a transient that ends inside a block, the streamed emit equals the curated host function on the kept samples."""
    phase, skip = _random_walk(rng), 37
    streamed = _fold(function, phase, skip=skip, block_len=block_len)
    host = Function.from_string(f"iri: tvbo:functions/{function}").resolved_callable(phase[skip:], dim="time")
    assert host.dims == ("node", "node_j")
    np.testing.assert_allclose(streamed, host.values, rtol=0, atol=1e-12)


def test_a_run_recipe_is_not_a_windowed_reducer():
    """A recipe read out once over the run has no window, so the tuning loop's sliding window and the degenerate-window guard leave its callable alone, while compute_fc keeps both."""
    for name in RECIPES.values():
        assert lookup_streaming_reducer("tvboptim", "tvbo.analysis.statistics", name).emit_kind == "run"
        assert not is_windowed_reducer("tvbo.analysis.statistics", name)
    assert is_windowed_reducer("tvboptim.observations.observation", "compute_fc")
