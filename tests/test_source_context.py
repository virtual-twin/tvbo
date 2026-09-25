"""The recipe being loaded is scoped context: relative paths resolve against it at any depth, and nothing of it survives the load."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from tvbo.utils.source import current_source_dir, current_source_file, loading_from

_WEIGHTS = np.array([[0.0, 2.0, 0.0], [0.5, 0.0, 3.0], [0.0, 0.0, 0.0]])

_DYNAMICS = """
    dynamics:
      name: Osc
      output: [x]
      parameters: {a: {value: 1.0}}
      state_variables:
        x:
          equation: {rhs: '-a*x'}
          initial_value: 0.1
    integration: {method: heun, step_size: 0.1, duration: 1.0, transient_time: 0.0, unit: s}
"""


def _write_matrix(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, _WEIGHTS, delimiter=",")


def test_a_load_states_its_file_for_the_block_only(tmp_path):
    recipe = tmp_path / "r.yaml"
    assert current_source_file() is None
    with loading_from(recipe) as source:
        assert source == str(recipe.resolve())
        assert current_source_dir() == str(tmp_path.resolve())
    assert current_source_file() is None


def test_a_nested_load_restores_the_outer_file(tmp_path):
    outer, inner = tmp_path / "study.yaml", tmp_path / "sub" / "exp.yaml"
    with loading_from(outer):
        with loading_from(inner):
            assert current_source_file() == str(inner.resolve())
        assert current_source_file() == str(outer.resolve())


def test_none_states_nothing_and_keeps_the_enclosing_file(tmp_path):
    outer = tmp_path / "study.yaml"
    with loading_from(None):
        assert current_source_file() is None
    with loading_from(outer), loading_from(None) as source:
        assert source == str(outer.resolve())


def test_a_failed_load_does_not_leak_its_file(tmp_path):
    with pytest.raises(RuntimeError), loading_from(tmp_path / "r.yaml"):
        raise RuntimeError
    assert current_source_file() is None


def test_another_thread_never_sees_this_threads_load(tmp_path):
    seen = []
    with loading_from(tmp_path / "r.yaml"):
        worker = threading.Thread(target=lambda: seen.append(current_source_file()))
        worker.start()
        worker.join()
    assert seen == [None]


def test_from_file_resolves_edge_matrix_files_against_the_recipe_not_the_cwd(tmp_path, monkeypatch):
    from tvbo import SimulationExperiment

    _write_matrix(tmp_path / "data" / "w.csv")
    recipe = tmp_path / "exp.yaml"
    recipe.write_text(
        "id: 1\nlabel: e\n" + _DYNAMICS.replace("\n    ", "\n") + "network:\n  edge_matrix_files: [data/w.csv]\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    exp = SimulationExperiment.from_file(str(recipe))

    assert exp.network.number_of_nodes == 3
    assert len(exp.network.edges) == int(np.count_nonzero(_WEIGHTS))
    assert exp._source_file == str(recipe.resolve())
    assert exp.network._source_dir == str(tmp_path.resolve())
    assert not hasattr(SimulationExperiment, "_pending_source_file")
    assert current_source_file() is None


def test_a_study_materialises_its_experiment_against_the_study_file(tmp_path, monkeypatch):
    from tvbo import SimulationStudy

    _write_matrix(tmp_path / "data" / "w.csv")
    spec = tmp_path / "study.yaml"
    spec.write_text(
        "key: T\nexperiments:\n  - id: 1\n    label: e\n" + _DYNAMICS + "    network:\n      edge_matrix_files: [data/w.csv]\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    exp = SimulationStudy.from_file(str(spec)).get_experiment(1)

    assert exp.network.number_of_nodes == 3
    assert exp._source_file == str(spec.resolve())
    assert current_source_file() is None


def test_a_network_built_outside_any_load_has_no_source_dir():
    from tvbo import Network

    net = Network(number_of_nodes=2)
    assert getattr(net, "_source_dir", None) is None
