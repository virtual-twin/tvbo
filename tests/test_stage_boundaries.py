"""What crosses the boundary between two stages of a tuning schedule, pinned bit for bit on a deterministic model whose loop carries an observer's states and a filled window.

A stage that declares ``reset_state`` opens where a run of the experiment opens: the declared initial condition on a delay history no settle has touched, its monitors on their cold start and its window empty, with only the update rules' tuned targets read back in. Nothing a stage before it integrated can then reach it except through those targets, and neither can the base run's settle. A stage that does not reset continues the one before it, so splitting a stage in two changes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.datamodel import AlgorithmStage, Parameter  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

WINDOW = 4

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
integration: {{method: euler, step_size: 10.0, duration: 200.0, transient_time: SETTLE, unit: ms}}
observations:
  signal:
    iri: tvbo:BOLD_Balloon
    source: [x]
    period: 200.0
  windowed:
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
        equation: {{rhs: "g - eta * (signal - 0.01)"}}
    hyperparameters:
      - {{name: eta, value: 0.0}}
      - {{name: window_size, value: {WINDOW}}}
    n_iterations: 1
    simulation_period: 200.0
execution: {{random_seed: 0, precision: float64}}
"""
"""Three uncoupled nodes relaxing towards g with no noise, observed through a haemodynamic observer whose states the loop carries from window to window, and a windowed FC that makes the loop fill and roll a window buffer."""


def _tuned_g(stages, settle=600.0):
    """The tuned ``g`` after the schedule *stages*, each ``(eta, n_iterations, reset_state)``."""
    exp = SimulationExperiment.from_string(SPEC.replace("SETTLE", repr(settle)))
    exp.algorithms["tune"].stages = [
        AlgorithmStage(
            n_iterations=n,
            reset_state=reset,
            arguments=[Parameter(name="eta", value=eta), Parameter(name="window_size", value=WINDOW)],
        )
        for eta, n, reset in stages
    ]
    return np.asarray(exp.run("tvboptim", mode="algorithms", quiet=True).algorithms["tune"].state.dynamics.g)


def test_a_reset_stage_after_one_that_tunes_nothing_computes_what_it_computes_alone():
    """A first stage at a rate of zero advances the model, the observer and the window and moves no target, so the reset stage after it is the whole schedule."""
    alone = _tuned_g([(0.5, 4, True)])
    assert np.abs(alone - 0.6).max() > 1e-6, "the rule must move g for the comparison to mean anything"
    np.testing.assert_array_equal(_tuned_g([(0.0, 3, True), (0.5, 4, True)]), alone)


def test_a_reset_schedule_does_not_see_the_base_runs_settle():
    """The settle belongs to the base run; a reset stage opens at the declared initial condition however long that settle ran."""
    np.testing.assert_array_equal(_tuned_g([(0.5, 4, True)], settle=400.0), _tuned_g([(0.5, 4, True)], settle=1200.0))


def test_a_schedule_that_does_not_reset_opens_on_the_settled_state():
    short, long = _tuned_g([(0.5, 4, False)], settle=400.0), _tuned_g([(0.5, 4, False)], settle=1200.0)
    assert np.abs(short - long).max() > 1e-9


def test_a_stage_boundary_without_a_reset_is_invisible():
    """Splitting a stage in two hands the second its state, observer states and window as the first left them, so the schedule is the single stage."""
    np.testing.assert_array_equal(_tuned_g([(0.5, 3, False), (0.5, 4, False)]), _tuned_g([(0.5, 7, False)]))
