"""Algorithm.selection: keep the best iterate of an online tuning run instead of the last one.

- absent, the rendered code is untouched (the default path stays byte-identical and so do its results);
- declared, the tuning scan carries (best value, best iteration, snapshot of every tuned target and the settled state), updates it where the criterion improves, tracks it across stages jointly, and restores the snapshot after the last stage before the post-tuning evaluation;
- the chosen iteration and value are recorded on the result and persisted beside the history.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.data.types import AlgorithmResult, ExperimentResult
from tvbo.datamodel import Algorithm, AlgorithmStage, Parameter, StateSelection
from tvbo.utils import Bunch

_EXP = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"
_SEL_TOKENS = ("_sel_better", "restore_best", "selected_iteration", "_sel_in", "_stage_selection")


def _stage(n_iter, eta, window):
    return AlgorithmStage(
        n_iterations=n_iter, arguments=[Parameter(name="eta", value=eta), Parameter(name="window_size", value=window)]
    )


def _experiment(n_iter=3, stages=2, selection=None, window=4):
    """FIC + FIC_EIB, the EIB loop on a tiny schedule: ``stages`` stages of ``n_iter`` (0 for a plain single call)."""
    exp = SimulationExperiment.from_file(_EXP)
    fe, fic = exp.algorithms["fic_eib"], exp.algorithms["fic"]
    if stages:
        fe.stages = [_stage(n_iter, 0.10, window), _stage(n_iter, 0.05, 2 * window)][:stages]
    else:
        for hp in fe.hyperparameters:
            if str(hp.name) == "window_size":
                hp.value = window
    fe.n_iterations, fic.n_iterations = n_iter, 3
    if selection is not None:
        fe.selection = selection
    return exp


def _tuned(algo):
    return {
        "J_i": np.asarray(algo.state.dynamics.J_i),
        "wLRE": np.asarray(algo.state.coupling.EIBLinearCoupling.wLRE),
        "wFFI": np.asarray(algo.state.coupling.EIBLinearCoupling.wFFI),
    }


def _run(exp):
    return exp.run("tvboptim", mode="algorithms", quiet=True).algorithms["fic_eib"]


def test_absent_selection_leaves_the_rendered_code_untouched():
    code = _experiment().render_code("tvboptim")
    for token in _SEL_TOKENS:
        assert token not in code, token


def test_the_slot_round_trips_with_its_defaults():
    algo = Algorithm(name="fic_eib", selection={"criterion": "fc_rmse"})
    assert str(algo.selection.criterion) == "fc_rmse"
    assert str(algo.selection.mode) == "min"
    assert int(algo.selection.from_iteration) == 0
    assert str(algo.selection.scope) == "run"
    algo = Algorithm(name="x", selection={"criterion": "fc_corr", "mode": "max", "scope": "stage"})
    assert str(algo.selection.mode) == "max" and str(algo.selection.scope) == "stage"


def test_the_slot_validates_against_the_shipped_json_schema(tmp_path):
    """`tvbo validate schema --class Algorithm` accepts the new block and refuses a mode outside the enum."""
    from typer.testing import CliRunner

    from tvbo.cli.validate import app

    good = tmp_path / "good.yaml"
    good.write_text("name: fic_eib\nselection: {criterion: fc_rmse, mode: max, from_iteration: 3, scope: stage}\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: fic_eib\nselection: {criterion: fc_rmse, mode: best}\n")
    runner = CliRunner()
    assert runner.invoke(app, ["schema", str(good), "--class", "Algorithm"]).exit_code == 0
    assert runner.invoke(app, ["schema", str(bad), "--class", "Algorithm"]).exit_code != 0


def test_declared_selection_emits_carry_restore_and_stage_handoff():
    code = _experiment(selection=StateSelection(criterion="fc_rmse", mode="max", from_iteration=2)).render_code("tvboptim")
    core = code[code.index("def _fic_eib_tuning_core_impl(") :]
    assert "_sel_in" in core[: core.index("):")]
    step = core[core.index("def _tuning_step(") : core.index("_ls_final, _ys_all = jax.lax.scan(")]
    assert "_sel_crit > _sel[" in step and "(_sel_gi >= 2)" in step
    for target in ("p__J_i", "p__wLRE", "p__wFFI"):
        assert f'"{target}"' in step
    run = code[code.index("def run_fic_eib(") : code.index("def _fic_eib_tuning_core_impl(")]
    assert "if restore_best:" in run
    assert '_selection["p__wLRE"], state.coupling.EIBLinearCoupling.wLRE' in run
    assert '_selection["ic"]' in run
    assert "selected_iteration=_selected_iteration" in run
    stage_loop = " ".join(code[code.index('if algorithm_name == "fic_eib":') :].split())
    assert "selection=_stage_selection," in stage_loop
    assert "restore_best=(_si == len(_stage_defs) - 1)," in stage_loop
    assert '_stage_selection = algo_result.get("selection", _stage_selection)' in stage_loop


def test_an_unknown_criterion_fails_at_render_time():
    exp = _experiment(selection=StateSelection(criterion="fc_rmse_post"))
    with pytest.raises(ValueError, match="selection.criterion 'fc_rmse_post'"):
        exp.render_code("tvboptim")


def test_the_selection_record_is_persisted_beside_the_history(tmp_path):
    xr = pytest.importorskip("xarray")
    observations = {"fc_rmse": SimpleNamespace(name="fc_rmse", dims=None, reduce=None, pipeline=None, source=None)}
    source = SimpleNamespace(
        network=SimpleNamespace(node_labels=["n0", "n1"]), dynamics=None, coupling=None, observations=observations
    )
    algo = AlgorithmResult(
        name="fic_eib",
        history=Bunch(fc_rmse=np.array([0.6, 0.5, 0.7])),
        selected_iteration=np.asarray(1),
        selected_value=np.asarray(0.5),
    )
    written = ExperimentResult(algorithms={"fic_eib": algo}, source=source).save(
        str(tmp_path), compress=False, record_only=False
    )
    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        assert int(ds["algorithm__fic_eib__selected_iteration"]) == 1
        assert float(ds["algorithm__fic_eib__selected_value"]) == pytest.approx(0.5)


@pytest.mark.slow
@pytest.mark.parametrize("mode, pick", [("min", np.argmin), ("max", np.argmax)])
def test_the_restored_state_is_the_snapshot_at_the_best_iteration(mode, pick):
    """Six online iterations whose in-loop RMSE is non-monotonic: the tuned state must be the recorded iterate at argmin (or argmax), not the last one."""
    pytest.importorskip("tvboptim")
    algo = _run(_experiment(n_iter=6, stages=0, selection=StateSelection(criterion="fc_rmse", mode=mode)))
    history = np.asarray(algo.history.fc_rmse)
    k = int(pick(history))
    assert k != len(history) - 1, f"fixture no longer non-monotonic: {history}"
    assert int(algo.selected_iteration) == k
    assert float(algo.selected_value) == pytest.approx(history[k])
    for name, value in _tuned(algo).items():
        np.testing.assert_array_equal(value, np.asarray(algo.history[name])[k], err_msg=name)


@pytest.mark.slow
def test_the_best_iterate_is_tracked_across_stages_jointly():
    """The carry crosses the stage boundary: a pick in stage 1 survives stage 2, `from_iteration` counts globally, and the restored parameters are stage 1's own record of that iteration."""
    pytest.importorskip("tvboptim")
    two = _run(_experiment(selection=StateSelection(criterion="fc_rmse", mode="max")))
    k = int(two.selected_iteration)
    assert 0 <= k < 3, f"the fixture's stage 1 no longer holds the largest RMSE (picked {k})"
    one = _run(_experiment(stages=1, selection=StateSelection(criterion="fc_rmse", mode="max")))
    assert int(one.selected_iteration) == k
    for name, value in _tuned(two).items():
        np.testing.assert_allclose(value, np.asarray(one.history[name])[k], rtol=1e-10, err_msg=name)
    later = _run(_experiment(selection=StateSelection(criterion="fc_rmse", mode="min", from_iteration=3)))
    history2 = np.asarray(later.history.fc_rmse)
    k2 = int(np.argmin(history2))
    assert int(later.selected_iteration) == 3 + k2
    assert float(later.selected_value) == pytest.approx(history2[k2])
    for name, value in _tuned(later).items():
        np.testing.assert_array_equal(value, np.asarray(later.history[name])[k2], err_msg=name)


def test_scope_run_renders_exactly_like_an_absent_scope():
    """`scope: run` is the default and must not change one byte of the generated code."""
    absent = _experiment(selection=StateSelection(criterion="fc_rmse")).render_code("tvboptim")
    explicit = _experiment(selection=StateSelection(criterion="fc_rmse", scope="run")).render_code("tvboptim")
    assert absent == explicit
    assert "restore_best=(_si == len(_stage_defs) - 1)," in " ".join(absent.split())


def test_scope_stage_restores_every_stage_and_counts_within_it():
    code = _experiment(selection=StateSelection(criterion="fc_rmse", scope="stage", from_iteration=1)).render_code("tvboptim")
    stage_loop = " ".join(code[code.index('if algorithm_name == "fic_eib":') :].split())
    assert "restore_best=True," in stage_loop
    core = code[code.index("def _fic_eib_tuning_core_impl(") :]
    assert "(_i >= 1) &" in core
    run = code[code.index("def run_fic_eib(") : code.index("def _fic_eib_tuning_core_impl(")]
    assert "_sel_stages" in run and 'selection["stages"]' in run


def test_the_per_stage_record_is_persisted_on_a_stage_axis(tmp_path):
    xr = pytest.importorskip("xarray")
    source = SimpleNamespace(network=SimpleNamespace(node_labels=["n0"]), dynamics=None, coupling=None, observations={})
    algo = AlgorithmResult(name="fic_eib", selected_iteration=np.array([1, 4]), selected_value=np.array([0.5, 0.4]))
    written = ExperimentResult(algorithms={"fic_eib": algo}, source=source).save(
        str(tmp_path), compress=False, record_only=False
    )
    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        assert ds["algorithm__fic_eib__selected_iteration"].dims == ("stage",)
        assert list(ds["algorithm__fic_eib__selected_iteration"].values) == [1, 4]
        assert list(ds["algorithm__fic_eib__selected_value"].values) == pytest.approx([0.5, 0.4])


@pytest.mark.slow
def test_scope_stage_equals_a_manual_stage_by_stage_argmin():
    """Stage 1's best iterate is restored before stage 2 starts, stage 2 ranks only its own iterations, and the record holds one global iteration and value per stage."""
    pytest.importorskip("tvboptim")
    sel = StateSelection(criterion="fc_rmse", mode="min", scope="stage")
    one = _run(_experiment(stages=1, selection=sel))
    h1 = np.asarray(one.history.fc_rmse)
    k1 = int(np.argmin(h1))
    two = _run(_experiment(selection=sel))
    h2 = np.asarray(two.history.fc_rmse)
    k2 = int(np.argmin(h2))
    assert k1 != 2 or k2 != 2, f"fixture no longer non-monotonic: {h1} {h2}"
    np.testing.assert_array_equal(np.asarray(two.selected_iteration), [k1, 3 + k2])
    np.testing.assert_allclose(np.asarray(two.selected_value), [h1[k1], h2[k2]], rtol=1e-12)
    for name, value in _tuned(two).items():
        np.testing.assert_array_equal(value, np.asarray(two.history[name])[k2], err_msg=name)
        np.testing.assert_allclose(
            np.asarray(two.history[name])[0],
            np.asarray(one.history[name])[k1],
            rtol=1e-10,
            err_msg=f"stage 2 did not start from stage 1's best {name}",
        )
