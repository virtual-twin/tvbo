"""An observer keeps its states from one window of a tuning loop to the next.

A sliding-window algorithm integrates one period at a time and hands each period to its monitors. An ``Observation.dynamics`` observer that emits every period, a haemodynamic model for one, advances states of its own, so the sample it reports for a period depends on every period before it. The contract pinned here is the one a plain run already keeps: with a deterministic model and a rule that moves nothing, the loop integrates the trajectory a plain run of the same length integrates, and the samples in its window buffer are that run's samples at the same instants. A monitor that opens every period on the observer's initial states reports, for each period, the response of a resting observer to that period alone.

The carry is across the loop's windows only: the evaluation after tuning is a run of its own and opens its observer fresh.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

PERIOD, STEP, WINDOW, N_ITERATIONS = 200.0, 10.0, 6, 9

OBSERVERS = {
    "one-state": f"""
    source: [x]
    period: {PERIOD}
    dynamics:
      name: Lag
      system_type: discrete
      parameters:
        tau_m: {{value: 900.0}}
      state_variables:
        m:
          initial_value: 0.0
          equation_type: recurrence
          equation: {{rhs: "m + dt * (x - m) / tau_m"}}
      derived_variables:
        out:
          record: true
          equation: {{rhs: "m"}}""",
    "state-read-after-its-update": f"""
    source: [x]
    period: {PERIOD}
    dynamics:
      name: Chain
      system_type: discrete
      parameters:
        tau_a: {{value: 600.0}}
        tau_b: {{value: 900.0}}
      state_variables:
        a:
          initial_value: 0.0
          equation_type: recurrence
          equation: {{rhs: "a_next"}}
        b:
          initial_value: 1.0
          equation_type: recurrence
          equation: {{rhs: "b + dt * (a_next - (b - 1.0)) / tau_b"}}
      derived_variables:
        a_next:
          equation: {{rhs: "a + dt * (x - a) / tau_a"}}
        out:
          record: true
          equation: {{rhs: "a + b"}}""",
    "balloon-windkessel": f"""
    iri: tvbo:BOLD_Balloon
    source: [x]
    period: {PERIOD}""",
}
"""An observer with one state of its own, one whose second state reads the first after its update, and the curated four-state haemodynamic one."""

SPEC = f"""
id: 1
dynamics:
  name: Relax
  label: Relax
  parameters:
    g: {{value: 0.6, heterogeneous: true, shape: "(n_nodes,)"}}
    tau: {{value: 1500.0}}
  state_variables:
    x:
      initial_value: 0.0
      equation: {{rhs: "(g - x) / tau"}}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 3
  nodes:
    - {{id: 0, label: r0}}
    - {{id: 1, label: r1}}
    - {{id: 2, label: r2}}
  edges: []
integration: {{method: euler, step_size: {STEP}, duration: DURATION, unit: ms}}
observations:
  signal:OBSERVER
execution: {{random_seed: 0, precision: float64}}
"""
"""Three uncoupled nodes relaxing to one level with no noise, so a run is fixed by its length alone."""

TUNING = f"""  windowed:
    source: [signal]
    dims: [node_i, node_j]
    pipeline:
      - callable: {{name: compute_fc, module: tvboptim.observations.observation}}
        arguments:
          timeseries: {{value: signal}}
          skip_t: {{value: 0}}
algorithms:
  tune:
    objective: {{type: activity_target, target_variable: x, target_value: 0.5}}
    observations: [signal, windowed]
    update_rules:
      - name: g_update
        target_parameter: {{name: g}}
        equation: {{rhs: "g - eta * (signal - 0.5)"}}
    hyperparameters:
      - {{name: eta, value: 0.0}}
      - {{name: window_size, value: {WINDOW}}}
    n_iterations: {N_ITERATIONS}
    simulation_period: {PERIOD}
"""
"""A rule with a zero rate over a window of six periods: the loop integrates six periods to fill the window and nine more, and changes no parameter."""


def _samples(observation):
    values = np.asarray(observation.data if hasattr(observation, "data") else observation)
    return values.reshape(values.shape[0], -1)


@pytest.mark.parametrize("observer", list(OBSERVERS))
def test_the_window_buffer_holds_the_samples_a_plain_run_reports_at_the_same_instants(observer):
    spec = SPEC.replace("OBSERVER", OBSERVERS[observer])
    periods = WINDOW + N_ITERATIONS
    plain = SimulationExperiment.from_string(spec.replace("DURATION", str(PERIOD * periods))).run("tvboptim", quiet=True)
    whole = _samples(plain.observations.signal)
    assert whole.shape[0] == periods
    tuned = SimulationExperiment.from_string(
        spec.replace("DURATION", str(PERIOD)).rstrip("\n").replace("\nexecution:", "\n" + TUNING + "execution:")
    )
    buffer = _samples(tuned.run("tvboptim", quiet=True).algorithms["tune"].signal_buffer)
    assert buffer.shape[0] == WINDOW
    cold = _samples(
        SimulationExperiment.from_string(spec.replace("DURATION", str(PERIOD))).run("tvboptim", quiet=True).observations.signal
    )
    assert np.abs(whole[-WINDOW:] - cold).max() > 1e-4, (
        "the fixture must take its observer far from the response of a resting one to a single period"
    )
    np.testing.assert_allclose(buffer, whole[-WINDOW:], rtol=0, atol=1e-12)
