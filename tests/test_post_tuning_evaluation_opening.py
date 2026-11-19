"""The evaluation after an algorithm has tuned opens at the experiment's own opening, its delay lines filled from it.

The tuning opens at the experiment's opening (an initial-condition draw, a node's declared state, a seeded ``from_experiment`` operating point), and the evaluation is the experiment at the tuned values, so it opens there too. Without a settle it opened from the state ``prepare`` handed it instead: the dynamics' ``INITIAL_STATE`` on every node, with the delay lines filled from it. Two probes read it. The leaky node of ``test_evaluation_own_settle`` is compared with the experiment run alone at the tuned value from the same opening. The delayed probe of ``test_delay_history_from_opening`` is held by a rule whose rate is zero, so its evaluation is the experiment at the declared values and reads its first max-delay window in closed form, ``y(t) = (settle + t) a W x(0)``.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvboptim.experimental.network_dynamics import prepare

from tvbo import SimulationExperiment
from tvbo.utils import yaml_loader

from .test_delay_history_from_opening import (
    DELAY,
    DT,
    DURATION,
    PROBE,
    A,
    W,
    _assert_reads_its_opening,
    _buffers,
    _built,
    _declared_per_node,
    _drawn,
    _sample_times,
)
from .test_evaluation_own_settle import _spec, _values

HOLD = {
    "hold": {
        "objective": {"type": "activity_target", "target_variable": "x", "target_value": 1.0},
        "observations": ["mean_x"],
        "update_rules": [
            {"name": "k_update", "target_parameter": {"name": "k"}, "equation": {"rhs": "k + eta * (mean_x - 1.0)"}}
        ],
        "hyperparameters": [{"name": "eta", "value": 0.0}],
        "n_iterations": 2,
        "simulation_period": DELAY,
    }
}
"""A rule at rate zero: the algorithm runs and evaluates, and ``k`` stays at the declared 0, so ``x`` keeps its opening."""

SETTLE = 6.0


def _leak(opening, **shape):
    """The leaky node of ``test_evaluation_own_settle`` without a settle, opening at 1.5 or a draw from [1, 2] instead of its declared 0.0, and the run options that opening needs."""
    spec = yaml_loader.load_as_dict(_spec(0.0, **shape))
    if opening == "drawn":
        spec["dynamics"]["state_variables"]["x"]["distribution"] = {"name": "Uniform", "domain": {"lo": 1.0, "hi": 2.0}}
    elif opening == "node-declared":
        spec["network"]["nodes"][0]["state"] = {"x": {"value": 1.5}}
    return spec, ({"seed_dynamics": {"x": np.array([1.5])}} if opening == "seeded" else {})


def _held(spec, settle=0.0):
    """The delayed probe *spec* with a settle of *settle* ms and the algorithm ``hold``, whose rule leaves ``k`` per node at 0."""
    spec = copy.deepcopy(spec)
    spec["dynamics"]["parameters"]["k"].update(heterogeneous=True, shape="(n_nodes,)")
    spec["observations"] = {"mean_x": {"source": ["x"], "aggregation": "mean"}}
    spec["algorithms"] = copy.deepcopy(HOLD)
    spec["integration"]["transient_time"] = settle
    return spec


def _evaluated(spec):
    """*spec* run through its generated module: the module, a network built by it, the result of ``hold``, and the state and function its evaluation was handed."""
    module, network = _built(spec)
    handed = {}
    run_hold = module.run_experiment.__globals__["run_hold"]

    def _recording(*args, **kwargs):
        handed.update(state=kwargs["post_state"], model_fn=kwargs["post_model_fn"])
        return run_hold(*args, **kwargs)

    module.run_experiment.__globals__["run_hold"] = _recording
    results = module.run_experiment(
        jnp.asarray(W), distances=jnp.asarray(np.where(W != 0, DELAY, 0.0)), mode="algorithms", quiet=True
    )
    return module, network, results["hold"], handed


def _evaluation_trace(hold):
    """The evaluation's ``x``, ``y`` and sample times, read by sample index as ``test_delay_history_from_opening`` reads a window."""
    trace = hold.post_tuning.data.transpose("time", "variable", "node")
    return trace.sel(variable="x").values, trace.sel(variable="y").values, _sample_times(trace.sizes["time"])


@pytest.mark.parametrize("opening", ["drawn", "node-declared", "seeded"])
def test_the_evaluation_without_a_settle_is_the_experiment_alone_from_its_own_opening(opening):
    """The evaluation is the experiment run alone at the tuned value from the same opening, from its first sample."""
    spec, options = _leak(opening)
    tune = SimulationExperiment.from_dict(spec).run("tvboptim", **options).algorithms["tune"]
    gain = float(np.asarray(tune.state.dynamics.g)[0])
    assert gain > 0.25, "the rule must have moved g off its untuned 0 for the comparison to mean anything"
    alone_spec, _ = _leak(opening, gain=gain, tuned=False)
    reference = _values(SimulationExperiment.from_dict(alone_spec).run("tvboptim", **options).integration.data)
    assert abs(reference[0]) > 0.9, "the opening must lie away from the declared 0.0 for the comparison to mean anything"
    evaluated = _values(tune.post_tuning)
    assert evaluated.shape == reference.shape
    np.testing.assert_allclose(evaluated, reference, rtol=0, atol=1e-9)


def test_the_evaluation_s_first_max_delay_window_reads_the_drawn_opening():
    """``x`` holds the experiment's own draw and ``y`` is what delay lines filled with that draw deliver from the first step."""
    module, network, hold, _ = _evaluated(_held(_drawn()))
    x, y, t = _evaluation_trace(hold)
    _assert_reads_its_opening(x, y, t)
    opening = np.asarray(module.run_simulation(network, run_main=False).opening)[module._STATE_INDEX["x"]]
    np.testing.assert_array_equal(x[0], opening)


def test_an_opening_at_the_declared_initial_values_evaluates_bit_for_bit_as_before():
    """The evaluation is handed the state ``prepare`` builds, opening and buffers alike, so it integrates the unpatched numbers.

    The unpatched evaluation is reconstructed rather than stored: before the fix nothing replaced the state ``prepare`` returned, so it is the same function on that state.
    """
    module, network, _, handed = _evaluated(_held(PROBE))
    _, prepared = prepare(network, module.get_solver(), t0=0.0, t1=DURATION, dt=DT)
    opened, built = handed["state"].initial_state.dynamics, prepared.initial_state.dynamics
    assert opened.dtype == built.dtype
    np.testing.assert_array_equal(np.asarray(opened), np.asarray(built))
    buffers = _buffers(handed["state"])
    assert buffers, "the probe must run a delayed coupling"
    for refilled, default in zip(buffers, _buffers(prepared), strict=True):
        assert refilled.dtype == default.dtype
        np.testing.assert_array_equal(refilled, default)
    unpatched = handed["state"].copy()
    unpatched.initial_state = prepared.initial_state
    solve = jax.jit(handed["model_fn"])
    np.testing.assert_array_equal(np.asarray(solve(handed["state"]).ys), np.asarray(solve(unpatched).ys))


def test_with_a_settle_the_evaluation_is_unchanged():
    """The evaluation's own settle rebuilds the delay lines from the opening, so the refill before it changes no number: the window reads ``(settle + t) a W x(0)`` and equals the one integrated from the buffer ``prepare`` built."""
    module, network, hold, handed = _evaluated(_held(_drawn(), settle=SETTLE))
    _assert_reads_its_opening(*_evaluation_trace(hold), settle=SETTLE)
    _, prepared = prepare(network, module.get_solver(), t0=0.0, t1=DURATION, dt=DT)
    unrefilled = handed["state"].copy()
    unrefilled.initial_state.coupling = prepared.initial_state.coupling
    solve = jax.jit(handed["model_fn"])
    for refilled, before in zip(solve(handed["state"]), solve(unrefilled), strict=True):
        np.testing.assert_array_equal(np.asarray(refilled.ys), np.asarray(before.ys))


def test_the_folded_evaluation_reads_each_node_s_declared_opening():
    """A streamed observation folds the evaluation in-carry, and its last ``y`` is ``T a W x(0)`` for the states the nodes declare."""
    spec = _held(_declared_per_node())
    spec["observations"]["y_last"] = {"source": ["y"], "aggregation": "last", "reduce": "streaming"}
    _, _, hold, _ = _evaluated(spec)
    expected = DURATION * A * (W @ np.array([1.5, -0.5]))
    np.testing.assert_allclose(_values(hold.post_tuning.observations.y_last), expected, rtol=1e-8, atol=1e-8)
