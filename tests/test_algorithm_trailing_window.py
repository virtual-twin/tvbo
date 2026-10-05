"""An online tuning rule reads its observation over the trailing window the recipe declares, and applies its update only from the iterations the recipe allows.

A `tail_duration` longer than the algorithm's `simulation_period` is a span of simulated time across iterations, and `update_start` and `apply_every` decide which iterations update. A window that cannot be built from whole periods is refused rather than cut to one, since a rule averaging over one period where the recipe says many settles somewhere else. Each run below is compared with a NumPy transcription of the same rule on the same Euler steps.
"""

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

PERIOD, STEP, TAU, ETA, TARGET = 2.0, 0.1, 5.0, 0.3, 0.5

# One deterministic leaky node whose offset `g` a homeostatic rule tunes towards a mean of 0.5; it relaxes over several periods, so the window length changes every update.
_SPEC = """
id: 1
dynamics:
  name: TunedLeak
  label: TunedLeak
  parameters:
    g: {value: 0.0, heterogeneous: true, free: true, shape: "(n_nodes,)"}
    h: {value: 0.0, heterogeneous: true, free: true, shape: "(n_nodes,)"}
    tau: {value: 5.0}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(1 - g - h - x) / tau"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 1
  nodes:
    - {id: 0, label: r0}
  edges: []
integration: {method: euler, step_size: 0.1, duration: 4.0, transient_time: 0.0, unit: ms}
observations:
  mean_x: {source: [x], aggregation: mean WINDOW}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [mean_x]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g + eta * (mean_x - 0.5)"}
    hyperparameters:
      - {name: eta, value: 0.3}
    n_iterations: 20
    simulation_period: 2.0CADENCE
execution: {random_seed: 0, precision: float64}
"""


def _spec(window="", cadence=""):
    return _SPEC.replace(" WINDOW", window).replace("CADENCE", cadence)


def _reference(n_iterations=20, window_samples=None, start=None, every=1):
    """The rule transcribed step by step: the window is the last `window_samples` states of the run so far, and an update applies when its period ends at or after `start` and is every `every`th."""
    per_period = int(round(PERIOD / STEP))
    window_samples = per_period if window_samples is None else window_samples
    x, g, states, means, gs = 0.0, 0.0, [], [], []
    for i in range(n_iterations):
        gs.append(g)
        for _ in range(per_period):
            x = x + STEP * (1 - g - x) / TAU
            states.append(x)
        end = len(states)
        mean = np.mean(states[max(0, end - window_samples) : end])
        means.append(mean)
        if (start is None or (i + 1) * PERIOD >= start) and (i + 1) % every == 0:
            g = g + ETA * (mean - TARGET)
    return np.array(means), np.array(gs), g


def _run(spec):
    tune = SimulationExperiment.from_string(spec).run("tvboptim").algorithms["tune"]
    return np.asarray(tune.history.mean_x), np.asarray(tune.history.g)[:, 0], float(np.asarray(tune.state.dynamics.g)[0])


def _assert_matches(spec, **reference):
    means, gs, g = _run(spec)
    ref_means, ref_gs, ref_g = _reference(**reference)
    np.testing.assert_allclose(means, ref_means, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gs, ref_gs, rtol=0, atol=1e-12)
    assert abs(g - ref_g) < 1e-12


def test_a_window_of_three_periods_averages_across_iterations():
    """Over the first two iterations fewer than three periods exist, so the mean is over all of them."""
    _assert_matches(_spec(window=", tail_duration: 6.0"), window_samples=60)


@pytest.mark.parametrize("start", [7.0, 8.0])
def test_updates_start_with_the_first_period_ending_at_or_after_update_start(start):
    """Either start puts the first update at the end of the fourth period, at 8.0: 7.0 falls inside it and 8.0 is its end."""
    _assert_matches(
        _spec(window=", tail_duration: 6.0", cadence=f"\n    update_start: {start}"), window_samples=60, start=start
    )


def test_apply_every_keeps_every_nth_update():
    _assert_matches(_spec(cadence="\n    apply_every: 3"), every=3)


def test_a_window_within_one_period_is_still_the_tail_of_that_period():
    _assert_matches(_spec(window=", tail_duration: 1.0"), window_samples=10)


@pytest.mark.parametrize(
    "window, cadence",
    [
        (", tail_duration: 5.0", ""),
        (", tail_samples: 30", ""),
        (", tail_duration: 6.0", "\n    stages:\n      - {n_iterations: 10}\n      - {n_iterations: 10}"),
    ],
    ids=["not-whole-periods", "samples-beyond-one-period", "window-across-stages"],
)
def test_a_window_that_cannot_be_honoured_is_refused(window, cadence):
    with pytest.raises(Exception, match="simulation period|stages"):
        SimulationExperiment.from_string(_spec(window=window, cadence=cadence)).render_code("tvboptim")


def test_an_included_rule_keeps_its_own_algorithms_start():
    outer = (
        "  outer:\n"
        "    observations: [mean_x]\n"
        "    includes: [{algorithm: tune}]\n"
        "    update_rules:\n"
        "      - name: h_update\n"
        "        target_parameter: {name: h}\n"
        "        equation: {rhs: 'h + eta * (mean_x - 0.5)'}\n"
        "    hyperparameters:\n"
        "      - {name: eta, value: 0.3}\n"
        "    n_iterations: 20\n"
        "    simulation_period: 2.0\n"
    )
    spec = _spec(cadence="\n    update_start: 7.0").replace("execution:", outer + "execution:")
    code = "".join(SimulationExperiment.from_string(spec).render_code("tvboptim").split())
    assert code.count("new_g=jnp.where((_i>=3),new_g,state.dynamics.g)") == 2
    assert "new_h=jnp.where(" not in code
