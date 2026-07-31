"""Tests for ``tvbo run`` engine dispatch helpers."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tvbo.cli import run as run_cli


@pytest.mark.parametrize(
    "engine,expected_workflow_file",
    [
        ("slurm", "run.sbatch"),
        ("snakemake", "Snakefile"),
        ("nextflow", "main.nf"),
    ],
)
def test_dispatch_to_engine_uses_kit_dir_not_file(monkeypatch, tmp_path: Path, engine: str, expected_workflow_file: str):
    """``tvbo run --engine`` emits the kit in-process into a directory.

    The emit runs in-process (no re-shelling ``tvbo``), so it must not depend on
    ``tvbo`` being on ``$PATH``; only the engine submission (sbatch/snakemake/
    nextflow) shells out. The artefact must land inside the kit directory.
    """
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    # Only the engine submission shells out; it lives in workflow now.
    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)

    kit_dir = tmp_path / "kit"
    run_cli._dispatch_to_engine(
        engine,
        spec="experiment:JR_MEG_FrequencyGradient_Optimization",
        backend="jax",
        experiment=None,
        container=None,
        out_dir=kit_dir,
    )

    # Kit emitted in-process into the directory (not a bare artefact file).
    assert (kit_dir / expected_workflow_file).is_file()
    # The submission shells out from the kit dir; none re-invokes `tvbo` on PATH.
    # Match the launcher by BASENAME: `_resolve_launcher` returns snakemake's
    # ABSOLUTE path when it sits next to the running interpreter (the venv-off-PATH
    # case — a cluster user runs `.venv/bin/tvbo` without activating), so cmd[0] may
    # be `/…/.venv/bin/snakemake`, not the bare name. This is exactly why it works
    # both locally and on HPC. (Filter out unrelated subprocess calls a library may
    # make, e.g. `uname -p`.)
    submits = [c for c in calls if Path(c["cmd"][0]).name in {"sbatch", "snakemake", "nextflow"}]
    assert submits, calls
    assert all(Path(c["cwd"]) == kit_dir for c in submits)
    assert all(Path(c["cmd"][0]).name != "tvbo" for c in calls)
    # JR_MEG dispatches as a single array task (chunk=1) — that one task IS the
    # whole result, so slurm submits just the array; no gather job is chained.
    if engine == "slurm":
        assert submits[0]["cmd"] == ["sbatch", "--parsable", "run.sbatch"]
        assert not (kit_dir / "finalize.sbatch").exists()
        assert not any("--dependency=afterok" in a for c in submits for a in c["cmd"])


def test_dispatch_to_engine_slurm_single_task_no_gather(monkeypatch, tmp_path: Path):
    """A single-task array (chunk=1) submits just the array — nothing to reassemble."""
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="27452074\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)

    kit_dir = tmp_path / "kit"
    run_cli._dispatch_to_engine(
        "slurm",
        spec="experiment:JR_MEG_FrequencyGradient_Optimization",
        backend="jax",
        experiment=None,
        container=None,
        out_dir=kit_dir,
    )

    submits = [c for c in calls if c["cmd"][0] == "sbatch"]
    assert len(submits) == 1
    assert submits[0]["cmd"] == ["sbatch", "--parsable", "run.sbatch"]
    assert Path(submits[0]["cwd"]) == kit_dir
    assert not (kit_dir / "finalize.sbatch").exists()


def test_unloadable_spec_reports_every_attempt(tmp_path: Path):
    """A spec that loads as nothing must say why, not blame the last fallback.

    ``_load_from_file`` tries study -> experiment -> dynamics. It used to swallow
    each failure, so the caller only ever saw the *dynamics* error — which, for a
    file that is plainly an experiment (e.g. one written by a newer tvbo than the
    one reading it), sends the reader chasing a malformed Dynamics that never was.
    """
    import typer

    from tvbo.cli import _common

    spec = tmp_path / "experiment.yaml"
    spec.write_text(
        "key: broken\n"
        "dynamics:\n"
        "  no_such_slot_for_any_class: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        _common.resolve_spec(str(spec))

    msg = str(excinfo.value)
    # Every interpretation tried is named, so nothing is hidden...
    assert "as experiment:" in msg
    assert "as dynamics:" in msg
    # ...and the original exception is chained rather than discarded.
    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# --pin: the fanned-exploration-axis per-cell restriction (the --subject sibling)
# ---------------------------------------------------------------------------
def _exp_with_sweep():
    """A 1-node experiment sweeping a dynamics param, so pinning is observable."""
    from tvbo import SimulationExperiment

    return SimulationExperiment(
        id=1, label="pin",
        dynamics={"name": "Osc", "system_type": "continuous", "output": ["x"],
                  "parameters": {"a": {"value": 1.0}},
                  "state_variables": {"x": {"equation": {"rhs": "-a*x"}, "initial_value": 0.1}}},
        network={"number_of_nodes": 1},
        integration={"method": "heun", "step_size": 0.1, "duration": 1.0,
                     "transient_time": 0.0, "unit": "s"},
        explorations={"sweep_a": {"name": "sweep_a", "mode": "product", "record": ["x"],
                                  "space": [{"parameter": "Osc.a",
                                             "domain": {"lo": 0.5, "hi": 1.5, "n": 3}}]}},
    )


def test_pin_sets_the_dynamics_param_and_drops_the_axis():
    """--pin must BOTH set the base parameter (so the representative run uses it) AND remove
    the axis from the sweep — else the exploration re-expands it and the cell is not a point."""
    exp = _exp_with_sweep()
    run_cli._apply_axis_pins(exp, ["Osc.a=0.5"])
    assert exp.dynamics.parameters["a"].value == 0.5          # base param set
    assert not (exp.explorations or {})                       # emptied exploration removed


def test_pin_leaves_other_axes_sweeping():
    """Pinning one axis of a multi-axis sweep collapses only that axis."""
    from tvbo import SimulationExperiment

    exp = SimulationExperiment(
        id=1, label="pin2",
        dynamics={"name": "Osc", "system_type": "continuous", "output": ["x"],
                  "parameters": {"a": {"value": 1.0}, "b": {"value": 2.0}},
                  "state_variables": {"x": {"equation": {"rhs": "-a*x + b"}, "initial_value": 0.1}}},
        network={"number_of_nodes": 1},
        integration={"method": "heun", "step_size": 0.1, "duration": 1.0,
                     "transient_time": 0.0, "unit": "s"},
        explorations={"g": {"name": "g", "mode": "product", "record": ["x"],
                            "space": [{"parameter": "Osc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}},
                                      {"parameter": "Osc.b", "domain": {"lo": 1.0, "hi": 3.0, "n": 3}}]}},
    )
    run_cli._apply_axis_pins(exp, ["Osc.a=0.5"])
    assert exp.dynamics.parameters["a"].value == 0.5
    remaining = list((exp.explorations["g"].space or {}))
    assert remaining and all("Osc.a" not in str(getattr(exp.explorations["g"].space[k], "parameter", k))
                             for k in remaining)


def test_pin_rejects_a_malformed_arg():
    exp = _exp_with_sweep()
    with pytest.raises(Exception, match="parameter=value"):
        run_cli._apply_axis_pins(exp, ["Osc.a"])   # no '='


# ── smoke iteration cap (`tvbo run --max-iterations` / `--smoke`) ─────────────────────────
from types import SimpleNamespace


def _algo(n_iterations, stages=None):
    return SimpleNamespace(n_iterations=n_iterations, stages=stages or [])


def test_max_iterations_caps_algorithms_and_stages_only_downward():
    """`--max-iterations N` caps every algorithm's and stage's `n_iterations` to N, and
    never RAISES a smaller count — it is a smoke ceiling, applied to the loaded object only."""
    exp = SimpleNamespace(
        algorithms={
            "fic": _algo(200),
            "fic_eib": _algo(2000, stages=[_algo(50000), _algo(50000)]),
        },
        optimizations={"grad": SimpleNamespace(max_iterations=66)},
    )
    run_cli._apply_max_iterations(exp, 1)
    assert exp.algorithms["fic"].n_iterations == 1
    assert exp.algorithms["fic_eib"].n_iterations == 1
    assert [s.n_iterations for s in exp.algorithms["fic_eib"].stages] == [1, 1]
    assert exp.optimizations["grad"].max_iterations == 1

    # A count already below the cap is left untouched.
    exp2 = SimpleNamespace(algorithms={"a": _algo(1)}, optimizations={})
    run_cli._apply_max_iterations(exp2, 5)
    assert exp2.algorithms["a"].n_iterations == 1


def test_max_iterations_none_is_a_no_op():
    exp = SimpleNamespace(algorithms={"a": _algo(200)}, optimizations={})
    run_cli._apply_max_iterations(exp, None)
    assert exp.algorithms["a"].n_iterations == 200


# ── study figure rendering (`tvbo run <study>` closes the replication loop) ──────────────
def test_render_study_figures_renders_into_base_figures_dir(monkeypatch, tmp_path: Path):
    """A study run renders its declarative `figures:` via the same path as `tvbo figure
    render`: base = the spec file's dir, output = <base>/figures — so the one-command
    result is interchangeable with a follow-up `tvbo figure render`."""
    seen = {}

    def _fake_render(figures, base_dir, out_dir):
        seen["figures"] = list(figures)
        seen["base"] = Path(base_dir)
        seen["out"] = Path(out_dir)
        return [Path(out_dir) / "f.png"]

    monkeypatch.setattr("tvbo.cli.figures.render_figures", _fake_render)

    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")
    study = SimpleNamespace(figures=[SimpleNamespace(name="Fig1")])

    run_cli._render_study_figures(study, str(spec), out_dir=tmp_path / "output" / "nc")

    assert seen["figures"] == list(study.figures)
    assert seen["base"] == tmp_path                     # spec dir, not the results out-dir
    assert seen["out"] == tmp_path / "figures"


def test_render_study_figures_no_figures_is_a_no_op(monkeypatch, tmp_path: Path):
    """A study without a `figures:` list never invokes the renderer."""
    called = False

    def _fake_render(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr("tvbo.cli.figures.render_figures", _fake_render)
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")

    run_cli._render_study_figures(SimpleNamespace(figures=None), str(spec), out_dir=None)
    run_cli._render_study_figures(SimpleNamespace(figures=[]), str(spec), out_dir=None)
    assert called is False


def test_render_study_figures_swallows_render_error(monkeypatch, tmp_path: Path):
    """A plotting failure must not fail a completed run — the results are already on disk."""
    def _boom(*a, **k):
        raise RuntimeError("no container")

    monkeypatch.setattr("tvbo.cli.figures.render_figures", _boom)
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")

    # Must not raise.
    run_cli._render_study_figures(
        SimpleNamespace(figures=[SimpleNamespace(name="Fig1")]), str(spec), out_dir=None
    )


def _die_raises(monkeypatch):
    """`_common.die` as an exception, so a refusal is observable in-process."""
    def _die(msg):
        raise SystemExit(msg)

    monkeypatch.setattr("tvbo.cli._common.die", _die)


def _run_kwargs(**over):
    """Explicit defaults for a direct `run()` call.

    Calling the typer-decorated function leaves every unpassed default an `OptionInfo`,
    which `is not None` — so the flag-conflict check would see every flag as given.
    """
    base = dict(engine="local", experiment=None, shard=None, rendered=None, limit=None,
                subject=None, duration=None, max_iterations=None, smoke=False,
                set_=[], pin=[], container=None, out_dir=None)
    base.update(over)
    return base


def test_analysis_is_refused_on_a_non_local_engine(monkeypatch, tmp_path: Path):
    """The kit fans out experiments, so dispatching would run the WHOLE study.

    Silently, and on a cluster — the same "exit 0 having simulated nothing" class the
    local guards refuse, inverted into simulating everything the user excluded.
    """
    _die_raises(monkeypatch)
    dispatched = False

    def _fake_dispatch(*a, **k):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(run_cli, "_dispatch_to_engine", _fake_dispatch)

    with pytest.raises(SystemExit, match="local-only"):
        run_cli.run(str(tmp_path / "Study.yaml"), analysis="fcd",
                    **_run_kwargs(engine="slurm"))
    assert dispatched is False


def test_analysis_is_refused_beside_any_simulation_flag(monkeypatch, tmp_path: Path):
    """Every flag that selects or reshapes simulation work, not just the first three."""
    _die_raises(monkeypatch)
    monkeypatch.setattr("tvbo.cli._common.resolve_spec",
                        lambda spec: ("study", SimpleNamespace(name="s")))

    for flag, over in (("--limit", {"limit": 4}), ("--pin", {"pin": ["G=2.1"]}),
                       ("--subject", {"subject": "100610"}), ("--smoke", {"smoke": True}),
                       ("--set", {"set_": ["integration.duration=8"]})):
        with pytest.raises(SystemExit, match=flag):
            run_cli.run(str(tmp_path / "Study.yaml"), analysis="fcd", **_run_kwargs(**over))


def test_analysis_is_refused_when_the_spec_is_an_experiment(monkeypatch, tmp_path: Path):
    """An experiment declares no `analyses:`, so the flag could only be ignored."""
    _die_raises(monkeypatch)
    monkeypatch.setattr("tvbo.cli._common.resolve_spec",
                        lambda spec: ("experiment", SimpleNamespace(name="e")))

    with pytest.raises(SystemExit, match="needs a study"):
        run_cli.run(str(tmp_path / "exp-3.yaml"), analysis="spectrum", **_run_kwargs())
