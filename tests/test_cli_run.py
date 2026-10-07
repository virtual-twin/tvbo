"""Tests for ``tvbo run`` engine dispatch helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tvbo.cli import run as run_cli
from tvbo.run import study as study_run


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

    The emit runs in-process (no re-shelling ``tvbo``), so it must not depend on ``tvbo`` being on ``$PATH``; only the engine submission (sbatch/snakemake/nextflow) shells out, and it does so from the kit directory. The artefact must land inside that directory.

    Submissions are recognised by launcher BASENAME: ``_resolve_launcher`` returns snakemake's absolute path when it sits next to the running interpreter (the venv-off-``PATH`` case — a cluster user runs ``.venv/bin/tvbo`` without activating), so ``cmd[0]`` may be ``/…/.venv/bin/snakemake`` rather than the bare name. Matching on the basename is also what filters out unrelated subprocess calls a library may make, such as ``uname -p``.
    """
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

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
    submits = [c for c in calls if Path(c["cmd"][0]).name in {"sbatch", "snakemake", "nextflow"}]
    assert submits, calls
    assert all(Path(c["cwd"]) == kit_dir for c in submits)
    assert all(Path(c["cmd"][0]).name != "tvbo" for c in calls)
    # JR_MEG dispatches as a single array task (chunk=1) — that one task IS the whole result, so slurm submits just the array; no gather job is chained.
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


def test_dispatch_to_engine_carries_the_container_and_the_results_root_into_the_plan(monkeypatch, tmp_path: Path):
    """``tvbo run --engine slurm --container IMG -o DIR`` emits a kit whose tasks run in IMG and write their results into DIR.

    Both flags reach the plan as the assignments ``tvbo workflow --set`` parses, so each must be spelled as a bare key: written flag-style (``--set=container=…``) the key parses as ``set`` and the kit runs bare, writing into its own ``derivatives/tvbo/``. ``-o`` is relative to where the command runs, as it is for a local run, even though the tasks run from the kit directory.
    """
    from tvbo.run import workflow as _workflow

    monkeypatch.setattr(
        "tvbo.cli.workflow.subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="12345\n")
    )
    plans = []
    real_plan = _workflow.plan
    monkeypatch.setattr(_workflow, "plan", lambda *a, **k: plans.append(real_plan(*a, **k)) or plans[-1])
    monkeypatch.chdir(tmp_path)

    image = "ghcr.io/the-virtual-brain/tvbo:0.7.0"
    run_cli._dispatch_to_engine(
        "slurm",
        spec="experiment:JR_MEG_FrequencyGradient_Optimization",
        backend="jax",
        experiment=None,
        container=image,
        out_dir=Path("run-001"),
    )

    results = (tmp_path / "run-001").resolve()
    (plan,) = plans
    assert plan.container == f"docker://{image}"
    assert plan.out_dir == str(results)
    sbatch = (results / "run.sbatch").read_text()
    assert f"docker://{image} " in sbatch
    assert f"-o {results}" in sbatch


def _captured_reexec(monkeypatch, *, singularity: bool) -> list[list[str]]:
    """Run `_reexec_in_container` with Singularity present or absent, capturing each command it would launch instead of launching it."""
    import subprocess
    from types import SimpleNamespace

    monkeypatch.delenv("SINGULARITY_BIND", raising=False)
    monkeypatch.setattr(
        run_cli.shutil, "which", lambda name: "/usr/bin/singularity" if singularity and name == "singularity" else None
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: launched.append(cmd) or SimpleNamespace(returncode=0))
    return launched


def test_container_reexec_hands_singularity_a_docker_reference_and_marks_the_inner_run(monkeypatch):
    """`--container ghcr.io/…:tag` reaches `singularity exec` as `docker://ghcr.io/…:tag`, which Singularity pulls rather than reading as a local file, and the inner run carries `TVBO_IN_CONTAINER=1` so it does not re-exec again."""
    launched = _captured_reexec(monkeypatch, singularity=True)
    with pytest.raises(SystemExit):
        run_cli._reexec_in_container("ghcr.io/the-virtual-brain/tvbo:0.7.0", ["run", "x.yaml"])
    (cmd,) = launched
    assert cmd[:4] == ["singularity", "exec", "--env", "TVBO_IN_CONTAINER=1"]
    assert "docker://ghcr.io/the-virtual-brain/tvbo:0.7.0" in cmd
    assert cmd[-3:] == ["tvbo", "run", "x.yaml"]


def test_container_reexec_hands_docker_the_reference_without_its_transport(monkeypatch):
    """Docker names a registry image without `docker://`, so a reference written either way runs the same image."""
    launched = _captured_reexec(monkeypatch, singularity=False)
    for image in ("ghcr.io/the-virtual-brain/tvbo:0.7.0", "docker://ghcr.io/the-virtual-brain/tvbo:0.7.0"):
        with pytest.raises(SystemExit):
            run_cli._reexec_in_container(image, ["run", "x.yaml"])
    assert all(
        "ghcr.io/the-virtual-brain/tvbo:0.7.0" in cmd and not any(c.startswith("docker://") for c in cmd) for cmd in launched
    )
    assert all(cmd[:2] == ["docker", "run"] and "TVBO_IN_CONTAINER=1" in cmd for cmd in launched)


@pytest.mark.parametrize("singularity, want", [(True, "docker://rocker/r-ver"), (False, "rocker/r-ver")])
def test_container_reexec_runs_a_third_party_image_at_its_registry_default_tag(monkeypatch, singularity, want):
    """An untagged image that is not tvbo's own is launched untagged on either runtime, never at tvbo's version, which is not a tag it carries."""
    monkeypatch.setenv("TVBO_CONTAINER_TAG", "9.9.9")
    launched = _captured_reexec(monkeypatch, singularity=singularity)
    with pytest.raises(SystemExit):
        run_cli._reexec_in_container("rocker/r-ver", ["run", "x.yaml"])
    (cmd,) = launched
    assert want in cmd and not any("9.9.9" in c for c in cmd)


def test_container_reexec_refuses_a_local_image_docker_cannot_run(monkeypatch):
    """Without Singularity a local `.sif` has no runtime, which is refused by name rather than handed to `docker run`."""
    import typer

    launched = _captured_reexec(monkeypatch, singularity=False)
    with pytest.raises(typer.Exit) as exc:
        run_cli._reexec_in_container("/images/tvbo.sif", ["run", "x.yaml"])
    assert exc.value.exit_code == 1
    assert launched == []


def _sweep_spec(tmp_path: Path) -> Path:
    """A one-node experiment on disk whose exploration sweeps a dynamics parameter."""
    import yaml

    spec = tmp_path / "sweep.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "id": 1,
                "label": "sweep",
                "dynamics": {
                    "name": "Osc",
                    "system_type": "continuous",
                    "output": ["x"],
                    "parameters": {"a": {"value": 1.0}},
                    "state_variables": {"x": {"equation": {"rhs": "-a*x"}, "initial_value": 0.1}},
                },
                "network": {"number_of_nodes": 1},
                "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0},
                "explorations": {
                    "sweep_a": {
                        "name": "sweep_a",
                        "mode": "product",
                        "record": ["x"],
                        "space": [{"parameter": "Osc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return spec


def test_a_shard_on_a_backend_the_registry_does_not_know_is_refused_by_name(tmp_path: Path):
    """The capability registry raises ``ValueError`` for an unknown backend, and a shard then has no axis it can slice.

    The refusal names the backend and every axis it would have to fan out, rather than a traceback from the registry lookup.
    """
    with pytest.raises(study_run.StudyRunError, match=r"'nosuch' is not in the backend capability registry.*Osc\.a"):
        study_run.run_experiment(
            _exp_with_sweep(),
            str(tmp_path / "sweep.yaml"),
            tmp_path / "out",
            options=study_run.RunOptions(backend="nosuch", shard=(0, 2)),
        )
    assert not (tmp_path / "out").exists()


def test_the_cli_reports_a_shard_on_an_unknown_backend_and_exits_1(tmp_path: Path):
    from typer.testing import CliRunner

    from tvbo.cli import app

    res = CliRunner().invoke(app, ["run", str(_sweep_spec(tmp_path)), "--shard", "0/2", "--backend", "nosuch"])
    assert res.exit_code == 1, res.output
    assert isinstance(res.exception, SystemExit)
    assert "is not in the backend capability registry" in res.output


def test_unloadable_spec_reports_every_attempt(tmp_path: Path):
    """A spec that loads as nothing must say why, not blame the last fallback.

    ``_load_from_file`` tries study -> experiment -> dynamics. It used to swallow each failure, so the caller only ever saw the *dynamics* error — which, for a file that is plainly an experiment (e.g. one written by a newer tvbo than the one reading it), sends the reader chasing a malformed Dynamics that never was.
    """
    import typer

    from tvbo.cli import _common

    spec = tmp_path / "experiment.yaml"
    spec.write_text(
        "key: broken\ndynamics:\n  no_such_slot_for_any_class: 1\n",
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


# --pin: the fanned-exploration-axis per-cell restriction (the --subject sibling)
def _exp_with_sweep():
    """A 1-node experiment sweeping a dynamics param, so pinning is observable."""
    from tvbo import SimulationExperiment

    return SimulationExperiment(
        id=1,
        label="pin",
        dynamics={
            "name": "Osc",
            "system_type": "continuous",
            "output": ["x"],
            "parameters": {"a": {"value": 1.0}},
            "state_variables": {"x": {"equation": {"rhs": "-a*x"}, "initial_value": 0.1}},
        },
        network={"number_of_nodes": 1},
        integration={"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0, "unit": "s"},
        explorations={
            "sweep_a": {
                "name": "sweep_a",
                "mode": "product",
                "record": ["x"],
                "space": [{"parameter": "Osc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}}],
            }
        },
    )


def test_pin_sets_the_dynamics_param_and_drops_the_axis():
    """--pin must BOTH set the base parameter (so the representative run uses it) AND remove the axis from the sweep — else the exploration re-expands it and the cell is not a point."""
    exp = _exp_with_sweep()
    study_run.apply_axis_pins(exp, [("Osc.a", 0.5)])
    assert exp.dynamics.parameters["a"].value == 0.5  # base param set
    assert not (exp.explorations or {})  # emptied exploration removed


def test_pin_leaves_other_axes_sweeping():
    """Pinning one axis of a multi-axis sweep collapses only that axis."""
    from tvbo import SimulationExperiment

    exp = SimulationExperiment(
        id=1,
        label="pin2",
        dynamics={
            "name": "Osc",
            "system_type": "continuous",
            "output": ["x"],
            "parameters": {"a": {"value": 1.0}, "b": {"value": 2.0}},
            "state_variables": {"x": {"equation": {"rhs": "-a*x + b"}, "initial_value": 0.1}},
        },
        network={"number_of_nodes": 1},
        integration={"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0, "unit": "s"},
        explorations={
            "g": {
                "name": "g",
                "mode": "product",
                "record": ["x"],
                "space": [
                    {"parameter": "Osc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}},
                    {"parameter": "Osc.b", "domain": {"lo": 1.0, "hi": 3.0, "n": 3}},
                ],
            }
        },
    )
    study_run.apply_axis_pins(exp, [("Osc.a", 0.5)])
    assert exp.dynamics.parameters["a"].value == 0.5
    remaining = list(exp.explorations["g"].space or {})
    assert remaining and all("Osc.a" not in str(getattr(exp.explorations["g"].space[k], "parameter", k)) for k in remaining)


def test_pin_rejects_a_malformed_arg():
    """A pin with no ``=`` is a usage error before any experiment runs."""
    import typer

    with pytest.raises(typer.BadParameter, match="parameter=value"):
        run_cli._assignments(["Osc.a"], "--pin", "parameter=value")


def test_an_assignment_arrives_as_the_type_it_spells():
    assert run_cli._assignments(["Osc.a=0.5", "network.name=x"], "--set", "path=value") == (
        ("Osc.a", 0.5),
        ("network.name", "x"),
    )


# ── smoke iteration cap (`tvbo run --max-iterations` / `--smoke`) ─────────────────────────
from types import SimpleNamespace


def _algo(n_iterations, stages=None):
    return SimpleNamespace(n_iterations=n_iterations, stages=stages or [])


def test_max_iterations_caps_algorithms_and_stages_only_downward():
    """`--max-iterations N` caps every algorithm's and stage's `n_iterations` to N, and never RAISES a smaller count — it is a smoke ceiling, applied to the loaded object only."""
    exp = SimpleNamespace(
        algorithms={
            "fic": _algo(200),
            "fic_eib": _algo(2000, stages=[_algo(50000), _algo(50000)]),
        },
        optimizations={"grad": SimpleNamespace(max_iterations=66)},
    )
    study_run.apply_max_iterations(exp, 1)
    assert exp.algorithms["fic"].n_iterations == 1
    assert exp.algorithms["fic_eib"].n_iterations == 1
    assert [s.n_iterations for s in exp.algorithms["fic_eib"].stages] == [1, 1]
    assert exp.optimizations["grad"].max_iterations == 1

    # A count already below the cap is left untouched.
    exp2 = SimpleNamespace(algorithms={"a": _algo(1)}, optimizations={})
    study_run.apply_max_iterations(exp2, 5)
    assert exp2.algorithms["a"].n_iterations == 1


def test_max_iterations_none_is_a_no_op():
    exp = SimpleNamespace(algorithms={"a": _algo(200)}, optimizations={})
    study_run.apply_max_iterations(exp, None)
    assert exp.algorithms["a"].n_iterations == 200


# ── study figure rendering (`tvbo run <study>` closes the replication loop) ──────────────
def test_render_study_figures_renders_into_the_layouts_figures_dir(monkeypatch, tmp_path: Path):
    """A study run renders its declarative `figures:` via the same path as `tvbo figure render`: base = the spec file's dir, output = the figures directory the layout record names — so the one-command result is interchangeable with a follow-up `tvbo figure render`."""
    seen = {}

    def _fake_render(figures, base_dir, out_dir):
        seen["figures"] = list(figures)
        seen["base"] = Path(base_dir)
        seen["out"] = Path(out_dir)
        return [Path(out_dir) / "f.png"]

    monkeypatch.setattr(study_run, "render_figures", _fake_render)

    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")
    study = SimpleNamespace(figures=[SimpleNamespace(name="Fig1")])

    study_run.render_study_figures(study, str(spec))

    from tvbo.utils.study_layout import study_path

    assert seen["figures"] == list(study.figures)
    assert seen["base"] == tmp_path  # spec dir, not the results out-dir
    assert seen["out"] == study_path("figures", root=tmp_path)


def test_render_study_figures_no_figures_is_a_no_op(monkeypatch, tmp_path: Path):
    """A study without a `figures:` list never invokes the renderer."""
    called = False

    def _fake_render(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr(study_run, "render_figures", _fake_render)
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")

    study_run.render_study_figures(SimpleNamespace(figures=None), str(spec))
    study_run.render_study_figures(SimpleNamespace(figures=[]), str(spec))
    assert called is False


def test_render_study_figures_swallows_render_error(monkeypatch, tmp_path: Path):
    """A plotting failure must not fail a completed run — the results are already on disk."""

    def _boom(*a, **k):
        raise RuntimeError("no container")

    monkeypatch.setattr(study_run, "render_figures", _boom)
    spec = tmp_path / "Study.yaml"
    spec.write_text("name: s\n", encoding="utf-8")

    # Must not raise.
    study_run.render_study_figures(SimpleNamespace(figures=[SimpleNamespace(name="Fig1")]), str(spec))


def _die_raises(monkeypatch):
    """`_common.die` as an exception, so a refusal is observable in-process."""

    def _die(msg):
        raise SystemExit(msg)

    monkeypatch.setattr("tvbo.cli._common.die", _die)


def _run_kwargs(**over):
    """Explicit defaults for a direct `run()` call.

    Calling the typer-decorated function leaves every unpassed default an `OptionInfo`, which `is not None` — so the flag-conflict check would see every flag as given.
    """
    base = dict(
        engine="local",
        experiment=None,
        shard=None,
        rendered=None,
        limit=None,
        subject=None,
        duration=None,
        max_iterations=None,
        smoke=False,
        set_=[],
        pin=[],
        container=None,
        out_dir=None,
    )
    base.update(over)
    return base


def test_analysis_is_refused_on_a_non_local_engine(monkeypatch, tmp_path: Path):
    """The kit fans out experiments, so dispatching would run the WHOLE study.

    Silently, and on a cluster — the same "exit 0 having simulated nothing" class the local guards refuse, inverted into simulating everything the user excluded.
    """
    _die_raises(monkeypatch)
    dispatched = False

    def _fake_dispatch(*a, **k):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(run_cli, "_dispatch_to_engine", _fake_dispatch)

    with pytest.raises(SystemExit, match="local-only"):
        run_cli.run(str(tmp_path / "Study.yaml"), analysis="fcd", **_run_kwargs(engine="slurm"))
    assert dispatched is False


def test_analysis_is_refused_beside_any_simulation_flag(monkeypatch, tmp_path: Path):
    """Every flag that selects or reshapes simulation work, not just the first three."""
    _die_raises(monkeypatch)
    monkeypatch.setattr("tvbo.cli._common.resolve_spec", lambda spec: ("study", SimpleNamespace(name="s")))

    for flag, over in (
        ("--limit", {"limit": 4}),
        ("--pin", {"pin": ["G=2.1"]}),
        ("--subject", {"subject": "100610"}),
        ("--smoke", {"smoke": True}),
        ("--set", {"set_": ["integration.duration=8"]}),
    ):
        with pytest.raises(SystemExit, match=flag):
            run_cli.run(str(tmp_path / "Study.yaml"), analysis="fcd", **_run_kwargs(**over))


def test_analysis_is_refused_when_the_spec_is_an_experiment(monkeypatch, tmp_path: Path):
    """An experiment declares no `analyses:`, so the flag could only be ignored."""
    _die_raises(monkeypatch)
    monkeypatch.setattr("tvbo.cli._common.resolve_spec", lambda spec: ("experiment", SimpleNamespace(name="e")))

    with pytest.raises(SystemExit, match="needs a study"):
        run_cli.run(str(tmp_path / "exp-3.yaml"), analysis="spectrum", **_run_kwargs())


# ------------------------------------------- flags that must not be silently ignored


@pytest.fixture
def collection_spec(tmp_path: Path) -> str:
    """A minimal study-of-studies on disk, with one nested study and one authored result."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "toy.yaml").write_text("title: Toy\nlabel: toy\nkey: toy\nexperiments: []\n", encoding="utf-8")
    spec = tmp_path / "collection.yaml"
    spec.write_text(
        "title: Demo\n"
        "studies:\n"
        "  - !include nested/toy.yaml\n"
        "results:\n"
        "  - {key: parcels, value: '379', source: Glasser2016}\n",
        encoding="utf-8",
    )
    return str(spec)


@pytest.mark.parametrize(
    "flag",
    [
        "--experiment=41",
        "--save-all",
        "--no-compress",
        "--limit=1",
        "--smoke",
        "--set=integration.duration=1",
    ],
)
def test_a_flag_a_collection_cannot_honour_is_refused(collection_spec, flag):
    """A study-of-studies runs every nested study with fixed save options.

    Accepting one of these and dropping it turns a one-container request into the whole study — hours of cluster time — or reports success for a ``--save-all`` that in fact wrote record-only. Each must fail fast, naming the flag.

    Driven through the CLI rather than by calling ``run()``: invoked directly, every typer default is an unresolved ``OptionInfo``, so the guard fires for every flag at once and the test passes without testing anything.
    """
    from typer.testing import CliRunner

    from tvbo.cli import app

    res = CliRunner().invoke(app, ["run", collection_spec, "--dry-run", flag])
    assert res.exit_code != 0
    assert "would be ignored" in res.output
    assert flag.split("=")[0] in res.output


def test_analysis_selects_a_collections_own_analyses(collection_spec):
    """A study-of-studies owns analyses reading what its nested studies committed, and `--analysis` names them.

    Refusing the flag would leave those analyses unreachable: the only alternative is running the whole tree, which is hours of work to re-derive numbers whose inputs are already on disk.
    """
    from typer.testing import CliRunner

    from tvbo.cli import app

    res = CliRunner().invoke(app, ["run", collection_spec, "--analysis", "nothing_declared"])
    assert "would be ignored" not in res.output


def test_a_plain_collection_run_is_not_refused(collection_spec):
    """The other half: ``--compress`` is the default, so its presence is not a choice.

    Treating a default as a passed flag would refuse every ordinary collection run.
    """
    from typer.testing import CliRunner

    from tvbo.cli import app

    res = CliRunner().invoke(app, ["run", collection_spec, "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "would be ignored" not in res.output
