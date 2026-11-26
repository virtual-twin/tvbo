"""An on-device cohort fit resets its stages exactly as a single fit does.

Every subject of an on-device cohort is fitted in one ``vmap`` lane by ``run_cohort_<algo>``, and a stage that declares ``reset_state`` reopens each lane where a run of the experiment opens: the declared initial condition, its monitors cold, its window empty and its noise from the lane's own key, carrying only the tuned targets. Pinned bit for bit on a deterministic model whose loop carries an observer's states and a filled window: a reset stage after one at a rate of zero is the schedule without it, in every lane.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import Network, SimulationExperiment  # noqa: E402
from tvbo.datamodel import AlgorithmStage, Parameter  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

SUBJECTS = ["01", "02"]
LABELS = ["A", "B", "C"]

SPEC = """
id: 1
dynamics:
  name: Relax
  label: Relax
  parameters:
    g: {value: 0.6, heterogeneous: true, shape: "(n_nodes,)"}
    tau: {value: 1500.0}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(g - x) / tau"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: A}
    - {id: 1, label: B}
    - {id: 2, label: C}
  edges: []
integration: {method: euler, step_size: 10.0, duration: 200.0, transient_time: 600.0, unit: ms}
dataset:
  dataset_id: toy_fc
  bids_root: BIDS_ROOT
  batch_mode: on_device
  subjects: [{subject_id: "01"}, {subject_id: "02"}]
observations:
  fc_target: {source: [dataset.subject.fc], query: {atlas: Toy, desc: FC, suffix: relmat}, reconcile: by_label}
  signal:
    iri: tvbo:BOLD_Balloon
    source: [x]
    period: 200.0
  windowed:
    source: [signal]
    dims: [node_i, node_j]
    pipeline:
      - callable: {name: compute_fc, module: tvboptim.observations.observation}
        arguments:
          timeseries: {value: signal}
          skip_t: {value: 0}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [signal, windowed, fc_target]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g - eta * (signal - 0.01)"}
    hyperparameters:
      - {name: eta, value: 0.0}
      - {name: window_size, value: 4}
    n_iterations: 1
    simulation_period: 200.0
execution: {random_seed: 0, precision: float64}
"""
"""Three uncoupled nodes relaxing towards g with no noise, observed through a haemodynamic observer the loop carries and a windowed FC that fills a window buffer; each subject's FC target is what makes the fit a cohort, one lane per subject."""


@pytest.fixture
def bids_root(tmp_path):
    """Two subjects' functional connectomes, the per-subject targets the cohort batches over."""
    for subject in SUBJECTS:
        net = Network.from_matrix(weights=np.eye(3), labels=LABELS)
        net.set_matrix("fc", np.full((3, 3), int(subject) / 10.0))
        folder = tmp_path / f"sub-{subject}" / "func"
        folder.mkdir(parents=True)
        net.save(folder / f"sub-{subject}_atlas-Toy_desc-FC_relmat.yaml")
    return tmp_path


def _tuned_g(bids_root, stages):
    """Every lane's tuned ``g`` after the schedule *stages*, each ``(eta, n_iterations, reset_state)``."""
    exp = SimulationExperiment.from_string(SPEC.replace("BIDS_ROOT", str(bids_root)))
    exp.algorithms["tune"].stages = [
        AlgorithmStage(
            n_iterations=n,
            reset_state=reset,
            arguments=[Parameter(name="eta", value=eta), Parameter(name="window_size", value=4)],
        )
        for eta, n, reset in stages
    ]
    assert exp.dataset_on_device()
    return np.asarray(exp.run("tvboptim", mode="algorithms", quiet=True).algorithms["tune"].cohort_state.dynamics.g)


def test_a_cohort_reset_stage_after_one_that_tunes_nothing_computes_what_it_computes_alone(bids_root):
    alone = _tuned_g(bids_root, [(0.5, 4, True)])
    assert alone.shape[0] == len(SUBJECTS)
    np.testing.assert_array_equal(_tuned_g(bids_root, [(0.0, 3, True), (0.5, 4, True)]), alone)
    assert np.abs(_tuned_g(bids_root, [(0.5, 4, False)]) - alone).max() > 1e-9, (
        "a schedule that does not reset opens on the settled state, so the reset must change the fit"
    )
