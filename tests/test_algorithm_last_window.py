"""A tuning algorithm's container holds what its loop computed over a sliding window, whole, as it stood at the last iteration.

An online rule scores the model on a window that slides while the parameters are still moving, and the evaluation after tuning scores a separate run with the parameters frozen. The two are different measurements of one fit, and a study that compares itself with a published in-loop score needs the first. The history keeps one mean per record of a windowed observation, so the matrix itself is written beside it as ``algorithm__<name>__last_window__<observation>``: the observation's own callable on the window the last iteration read.

The variable says so itself. Its ``iteration`` attribute is ``last``, and where a selection restored another iterate's tuned values it also carries that iterate as ``selected_iteration`` with a sentence, so the window is never read as the restored iterate's.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")
xr = pytest.importorskip("xarray")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.data.types import AlgorithmResult, ExperimentResult  # noqa: E402
from tvbo.utils import Bunch  # noqa: E402

SPEC = """
id: 1
dynamics:
  name: NoisyLeak
  label: NoisyLeak
  parameters:
    g: {value: 0.2, heterogeneous: true, shape: "(n_nodes,)"}
    tau: {value: 5.0}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(g - x) / tau"}
      coupling_variable: true
      noise: {parameters: {sigma: {value: 0.05}}}
  output: [x]
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
    - {id: 2, label: r2}
  edges: []
integration: {method: euler, step_size: 0.5, duration: 20.0, unit: ms, noise: {parameters: {sigma: {value: 0.05}}}}
observations:
  level: {source: [x], aggregation: mean}WINDOWED
execution: {random_seed: 0, precision: float64}
"""
"""Three uncoupled noisy nodes relaxing to their own g, so the correlation of their per-period levels over a short window is a full matrix with no degenerate entry."""

WINDOWED = """
  fc_window:
    source: [level]
    dims: [node_i, node_j]
    pipeline:
      - callable: {name: compute_fc, module: tvboptim.observations.observation}
        arguments:
          timeseries: {value: level}
          skip_t: {value: 0}"""

ALGORITHM = """algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [OBSERVED]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g - eta * (level - 0.5)"}
    hyperparameters:
      - {name: eta, value: 0.05}HYPER
    n_iterations: 9
    simulation_period: 4.0EXTRA
"""

STAGES = """
    stages:
      - {n_iterations: 5, arguments: [{name: eta, value: 0.05}, {name: window_size, value: 6}]}
      - {n_iterations: 5, arguments: [{name: eta, value: 0.02}, {name: window_size, value: 4}]}"""

SHAPES = {
    "one-call": "",
    "staged": STAGES,
    "selected": """
    selection: {criterion: level, mode: min}""",
    "selected-per-stage": STAGES
    + """
    selection: {criterion: level, mode: min, scope: stage}""",
}
"""The ways a windowed loop ends: one call, a staged schedule whose last window is shorter than the ring the stages share, a run that restores its best iterate, and a staged schedule that restores the best iterate of each stage."""

LAST_ITERATION = {"selected": 8, "selected-per-stage": 9}
"""The index of each selecting shape's last iteration on the count a selection records, which runs on across stages."""


def _spec(shape="one-call", windowed=True, algorithm=True):
    observed, hyper = ("level, fc_window", "\n      - {name: window_size, value: 6}") if windowed else ("level", "")
    tuning = ALGORITHM.replace("OBSERVED", observed).replace("HYPER", hyper).replace("EXTRA", SHAPES[shape])
    return SPEC.replace("WINDOWED", WINDOWED if windowed else "") + (tuning if algorithm else "")


def _written(result, tmp_path):
    """The algorithm variables of the container *result* writes, loaded."""
    paths = result.save(str(tmp_path), compress=False, record_only=False)
    with xr.open_dataset(next(p for p in paths if p.endswith(".h5")), engine="h5netcdf") as ds:
        return {str(k): ds[k].load() for k in ds.data_vars if str(k).startswith("algorithm__")}


def _window_fc(buffer):
    """The correlation matrix of a window buffer's node series with a zero diagonal, as the observation's callable defines it."""
    fc = np.corrcoef(np.asarray(buffer)[:, 0, :].T)
    np.fill_diagonal(fc, 0.0)
    return fc


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_container_holds_the_windowed_observation_of_the_last_iteration(shape, tmp_path):
    """The written matrix is the observation recomputed from the window buffer the algorithm returns, and its mean is the history's last record."""
    result = SimulationExperiment.from_string(_spec(shape)).run("tvboptim", quiet=True)
    tune = result.algorithms["tune"]
    stored = _written(result, tmp_path)["algorithm__tune__last_window__fc_window"]
    assert stored.dims == ("node_i", "node_j")
    assert np.abs(stored.values).max() > 0.05, "the window must hold signal for the comparison to mean anything"
    np.testing.assert_allclose(stored.values, _window_fc(tune.level_buffer), rtol=0, atol=1e-12)
    assert np.asarray(tune.level_buffer).shape[0] == (4 if "stages:" in SHAPES[shape] else 6)
    np.testing.assert_allclose(stored.values.mean(), np.asarray(tune.history.fc_window)[-1], rtol=1e-9)


@pytest.mark.parametrize("shape", ["one-call", "staged"])
def test_the_stored_window_states_that_it_is_the_last_iteration_s(shape, tmp_path):
    """Without a selection the tuned values are the last iteration's as well, so the variable states its iteration and nothing more."""
    result = SimulationExperiment.from_string(_spec(shape)).run("tvboptim", quiet=True)
    assert dict(_written(result, tmp_path)["algorithm__tune__last_window__fc_window"].attrs) == {"iteration": "last"}


@pytest.mark.parametrize("shape", ["selected", "selected-per-stage"])
def test_under_a_selection_the_stored_window_names_the_iteration_the_tuned_values_came_from(shape, tmp_path):
    """A selection restores an earlier iterate's tuned values and leaves the window where the loop ended, so the variable carries both: its own iteration, and the restored one as the algorithm's ``selected_iteration`` records it."""
    result = SimulationExperiment.from_string(_spec(shape)).run("tvboptim", quiet=True)
    written = _written(result, tmp_path)
    attrs = written["algorithm__tune__last_window__fc_window"].attrs
    restored = int(np.asarray(written["algorithm__tune__selected_iteration"].values).reshape(-1)[-1])
    assert 0 <= restored < LAST_ITERATION[shape], "the fixture must restore an earlier iterate"
    assert attrs["iteration"] == "last"
    assert attrs["selected_iteration"] == restored
    assert "algorithm__tune__selected_iteration" in attrs["description"]
    assert "estimate__" in attrs["description"]


def test_an_algorithm_without_a_windowed_observation_writes_nothing_new(tmp_path):
    experiment = SimulationExperiment.from_string(_spec(windowed=False))
    assert "last_window" not in experiment.render_code("tvboptim")
    result = experiment.run("tvboptim", quiet=True)
    assert result.algorithms["tune"].get("last_window") is None
    assert not [key for key in _written(result, tmp_path) if "last_window" in key]


def test_an_experiment_without_algorithms_renders_no_last_window():
    assert "last_window" not in SimulationExperiment.from_string(_spec(algorithm=False)).render_code("tvboptim")


MATRIX = np.array([[0.0, 0.4], [0.4, 0.0]])


def _hand_built(tmp_path, **algorithms):
    """The written algorithm variables of a result assembled by hand from ``name=extras`` pairs, each an algorithm with a two-record history."""
    from types import SimpleNamespace

    source = SimpleNamespace(network=SimpleNamespace(node_labels=["n0", "n1"]), dynamics=None, coupling=None, observations={})
    built = {
        name: AlgorithmResult(name=name, history=Bunch(score=np.array([0.1, 0.2])), **extras)
        for name, extras in algorithms.items()
    }
    return _written(ExperimentResult(algorithms=built, source=source), tmp_path)


def test_the_writer_names_the_variable_after_the_algorithm_and_the_observation(tmp_path):
    """The write is the result's `last_window`, whatever produced it: one variable per entry, and none for a result without one."""
    stored = _hand_built(tmp_path, a={"last_window": Bunch(fc=MATRIX)}, b={})
    np.testing.assert_array_equal(stored["algorithm__a__last_window__fc"].values, MATRIX)
    assert [key for key in stored if "last_window" in key] == ["algorithm__a__last_window__fc"]


@pytest.mark.parametrize(
    ("selected", "expected"),
    [(None, None), (np.array(3), 3), (np.array([2, -1]), -1)],
    ids=["no-selection", "one-selection", "per-stage-none-eligible-in-the-last"],
)
def test_the_writer_states_the_window_s_iteration_from_the_result_s_selection(selected, expected, tmp_path):
    """The annotation follows the result alone: ``iteration`` always, and the restored iterate with its sentence where the result records a selection, the last stage's entry where it records one per stage."""
    extras = {"last_window": Bunch(fc=MATRIX)} | ({} if selected is None else {"selected_iteration": selected})
    attrs = dict(_hand_built(tmp_path, a=extras)["algorithm__a__last_window__fc"].attrs)
    assert attrs.pop("iteration") == "last"
    if expected is None:
        assert attrs == {}
    else:
        assert attrs.pop("selected_iteration") == expected
        assert set(attrs) == {"description"}
