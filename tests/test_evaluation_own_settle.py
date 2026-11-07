"""The evaluation after an algorithm has tuned is the experiment at the tuned values: it runs the declared settle itself, under those values, before its measured window.

``integration.transient_time`` is settling the experiment declares, so it belongs to every run of the experiment, and the evaluation an algorithm reports is such a run. An evaluation that opens its window on the state the base run settled to, under the values the algorithm has since replaced, records the relaxation from the untuned operating point to the tuned one inside the measured window, and a kernel observation convolves its first support over activity the tuned model never produced.

The reference for every case here is the experiment run alone, without the algorithm, with the tuned value written in. That is what the evaluation is, so the comparison holds from the window's first sample.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
id: 1
dynamics:
  name: TunedLeak
  label: TunedLeak
  parameters:
    g: {value: GAIN, heterogeneous: true, free: true, shape: "(n_nodes,)"}
    tau: {value: TAU}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(1 - g - x) / tau"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 1
  nodes:
    - {id: 0, label: r0}
  edges: []
integration: {method: euler, step_size: STEP, duration: DURATION, transient_time: SETTLE, unit: ms}
observations:OBSERVATIONS
execution: {random_seed: 0, precision: float64}
"""
"""One leaky node relaxing to 1 - g with time constant tau, so a window that opens off the fixed point of its own g shows the relaxation for several tau."""

TUNING = """algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [mean_x]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g + eta * (mean_x - 0.5)"}
    hyperparameters:
      - {name: eta, value: 0.3}
    n_iterations: ITERATIONS
    simulation_period: PERIOD
"""
"""A homeostatic rule that moves g from 0 towards 0.5, the value at which the node rests at the target."""


def _spec(settle, *, gain=0.0, tuned=True, tau=5.0, step=0.1, duration=4.0, period=2.0, iterations=200, observation=""):
    """The experiment with its tuning algorithm, or alone at *gain*. Alone it observes only what the case compares, since the rule's own input is the algorithm's and not the experiment's."""
    text = SPEC + (TUNING.replace("ITERATIONS", str(iterations)).replace("PERIOD", str(period)) if tuned else "")
    for token, value in {"GAIN": gain, "TAU": tau, "STEP": step, "DURATION": duration, "SETTLE": settle}.items():
        text = text.replace(token, repr(value))
    observed = (["mean_x: {source: [x], aggregation: mean}"] if tuned or not observation else []) + (
        [f"bold: {{source: [x], {observation}}}"] if observation else []
    )
    return text.replace("OBSERVATIONS", "".join(f"\n  {line}" for line in observed))


def _values(view):
    return np.asarray(getattr(view, "data", view)).reshape(-1)


def _tuned(spec):
    tune = SimulationExperiment.from_string(spec).run("tvboptim").algorithms["tune"]
    return tune, float(np.asarray(tune.state.dynamics.g)[0])


@pytest.mark.parametrize("settle", [40.0, 0.0], ids=["declared-settle", "no-settle"])
def test_the_evaluation_window_is_the_experiment_alone_at_the_tuned_value(settle):
    """With a settle of eight time constants the evaluation records the tuned fixed point from its first sample; with none declared it opens at the initial condition, as the experiment alone does."""
    tune, gain = _tuned(_spec(settle))
    assert abs(gain - 0.5) < 1e-6, "the rule must have moved g off its untuned value for the comparison to mean anything"
    alone = SimulationExperiment.from_string(_spec(settle, gain=gain, tuned=False)).run("tvboptim").integration
    evaluated, reference = _values(tune.post_tuning), _values(alone.data)
    assert evaluated.shape == reference.shape
    np.testing.assert_allclose(evaluated, reference, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "observation",
    ["iri: 'tvbo:BOLD_HRF_strided', reduce: streaming", "iri: 'tvbo:BOLD_HRF_strided'"],
    ids=["streamed", "materialised"],
)
def test_a_kernel_observation_of_the_evaluation_opens_on_its_own_settle(observation):
    """The evaluation's BOLD is the BOLD of the experiment alone at the tuned value, from its first sample: the kernel is warmed on the settle the tuned model ran."""
    shape = {"tau": 4000.0, "step": 1.0, "duration": 40000.0, "period": 20000.0, "iterations": 40, "observation": observation}
    tune, gain = _tuned(_spec(60000.0, **shape))
    assert abs(gain - 0.5) < 1e-3
    alone = SimulationExperiment.from_string(_spec(60000.0, gain=gain, tuned=False, **shape)).run("tvboptim")
    reference = _values(alone.observations.bold)
    assert abs(reference).max() > 0, "the reference must carry signal for the comparison to mean anything"
    np.testing.assert_allclose(_values(tune.post_tuning.observations.bold), reference, rtol=1e-9)
