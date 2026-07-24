"""Tests for ``tvbo workflow`` (C4) and ``tvbo validate`` stubs (C5)."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tvbo.cli import app
from tvbo.cli._backends import (
    BACKENDS,
    axis_kind_of,
    list_backends,
    resolve_backend,
)


runner = CliRunner()
EXP = "experiment:JR_MEG_FrequencyGradient_Optimization"


# ---------------------------------------------------------------------------
# Backend table (ontology-derived)
# ---------------------------------------------------------------------------

def test_backends_match_ontology_keys():
    keys = {b.name for b in list_backends()}
    assert keys == {"jax", "tvb", "pyrates", "tvboptim", "networkdynamics", "bifurcationkit", "numpy", "brian2"}


def test_jax_vectorizes_parameters_and_seeds():
    jax = resolve_backend("jax")
    assert jax.can_vectorize("parameters")
    assert jax.can_vectorize("noise_seed")
    assert jax.can_vectorize("initial_conditions")


def test_tvb_does_not_vectorize_anything():
    tvb = resolve_backend("tvb")
    assert tvb.vectorize_axes == frozenset()


def test_alias_resolution():
    assert resolve_backend("nd").name == "networkdynamics"


@pytest.mark.parametrize("path,kind", [
    ("integrator.noise.seed", "noise_seed"),
    ("model.initial_conditions", "initial_conditions"),
    ("subject", "subjects"),
    ("JansenRit.a", "parameters"),
])
def test_axis_kind_of(path, kind):
    assert axis_kind_of(path) == kind


# ---------------------------------------------------------------------------
# CLI: workflow backends / plan
# ---------------------------------------------------------------------------

def test_workflow_backends_lists_all():
    r = runner.invoke(app, ["workflow", "backends", "--json"])
    assert r.exit_code == 0
    import json
    data = json.loads(r.stdout)
    names = {row["name"] for row in data}
    assert names == set(BACKENDS)


def test_workflow_plan_jax_vectorizes_both_axes():
    r = runner.invoke(app, ["workflow", "plan", EXP, "--backend", "jax", "--json"])
    assert r.exit_code == 0, r.stdout
    import json
    p = json.loads(r.stdout)
    vec = {a["name"] for a in p["vectorize_axes"]}
    wf = {a["name"] for a in p["workflow_axes"]}
    assert vec == {"a", "b"}
    assert wf == set()
    assert p["n_workflow_cells"] == 1


def test_workflow_plan_tvb_fans_out_both_axes():
    r = runner.invoke(app, ["workflow", "plan", EXP, "--backend", "tvb", "--json"])
    assert r.exit_code == 0, r.stdout
    import json
    p = json.loads(r.stdout)
    vec = {a["name"] for a in p["vectorize_axes"]}
    wf = {a["name"] for a in p["workflow_axes"]}
    assert vec == set()
    assert wf == {"a", "b"}
    assert p["n_workflow_cells"] == 32 * 32


# ---------------------------------------------------------------------------
# CLI: workflow snakemake / slurm / nextflow kit emission
# ---------------------------------------------------------------------------

def test_workflow_snakemake_emits_kit(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "snakemake", EXP, "--backend", "jax", "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    assert (out / "Snakefile").is_file()
    assert (out / "README.md").is_file()
    # Each experiment is frozen into its own self-contained spec/<key>/ directory,
    # so one Snakefile fans a whole study (per experiment, per subject / sweep cell).
    frozen = list(out.glob("spec/*/experiment.yaml"))
    assert frozen, "expected a frozen spec/<key>/experiment.yaml"
    smk = (out / "Snakefile").read_text()
    assert "rule" in smk
    # A rule runs the frozen spec through `tvbo run` (the resolution layer needed
    # for per-subject targets), not the raw backend script.
    assert "tvbo run spec/" in smk
    # The README's Layout must describe where the specs actually are. The snakemake
    # emitter always writes spec/<key>/experiment.yaml — including for one
    # experiment — so a flat `spec/<key>.yaml` claim would send the reader nowhere.
    readme = (out / "README.md").read_text()
    assert "spec/<experiment>/experiment.yaml" in readme
    assert not any(line.startswith(f"- `spec/{p.parent.name}.yaml`")
                   for p in frozen for line in readme.splitlines())


def test_snakemake_rule_emits_the_resolved_backend_never_none(tmp_path: Path):
    """Emitting WITHOUT `--backend` must pin the plan's resolved backend on each rule,
    not the raw ``None``. The plan resolves an unset backend to the experiment's
    ``execution.backend`` (else tvboptim); if the raw None leaks through, the rule
    renders ``--backend=None`` and every cell dies at backend resolution
    (``experiment.run(format="None")`` -> ValueError)."""
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "snakemake", EXP, "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    smk = (out / "Snakefile").read_text()
    assert "--backend=None" not in smk
    assert "--backend=tvboptim" in smk  # EXP self-selects tvboptim when none is given


def test_study_rules_do_not_share_one_experiments_resources():
    """Declared resources belong to the rule that declared them, and to no other.

    Both directions have shipped broken: with no per-rule block every rule inherited
    whichever experiment the freeze loop ended on, and with an empty block treated as
    "unset" a heavyweight experiment's 128G/24h leaked onto every trivial sibling.
    """
    from tvbo.cli.workflow import _render_template

    heavy = {"cpus_per_task": 2, "mem": "128G", "time": "24:00:00",
             "env": [{"name": "OMP_NUM_THREADS", "value": "1"}]}

    def ep(key, block):
        return {"key": key, "rule_name": f"exp_{key}", "spec_relpath": f"spec/{key}/experiment.yaml",
                "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
                "container": None, "block": block, "axes": [], "depends_on": []}

    smk = _render_template("snakemake/study.smk.mako",
                           exp_plans=[ep("40", heavy), ep("1", {})], block={}, bundled_code=False)

    grid = smk[smk.index("rule exp_40:"):smk.index("rule exp_1:")]
    sheet = smk[smk.index("rule exp_1:"):]
    # 128 GiB, not 128_000 MB: Slurm sizes are binary, so a decimal conversion would
    # reserve ~2.4% less than the recipe declared.
    assert "mem_mb=131072" in grid and "runtime=1440" in grid
    assert "OMP_NUM_THREADS" in grid
    # The sheet experiment declared nothing: no reservation, no borrowed env.
    assert "mem_mb" not in sheet, f"exp 1 inherited exp 40's memory:\n{sheet}"
    assert "OMP_NUM_THREADS" not in sheet
    assert "threads: 1" in sheet


@pytest.mark.parametrize("walltime,minutes", [
    ("3-00:00:00", 4320),   # days-hours:minutes:seconds — the sbatch form that silently parsed to None
    ("7-00:00:00", 10080),
    ("1-06:30", 1830),      # days-hours:minutes
    ("2-12", 3600),         # days-hours
    ("08:00:00", 480),      # hours:minutes:seconds
    ("30:00", 30),          # minutes:seconds
    ("480", 480),           # bare minutes
    ("00:00:30", 1),        # a sub-minute request still reserves a whole minute
    (None, None),
    ("garbage", None),
])
def test_runtime_minutes_accepts_every_sbatch_walltime_spelling(walltime, minutes):
    """A declared walltime must survive into Snakemake's ``runtime`` resource.

    The day-prefixed spellings are the ones that matter: when ``3-00:00:00`` parsed
    to None the resource was omitted entirely and every job silently inherited the
    partition's default limit instead of the 3 days the study asked for.
    """
    from tvbo.cli._workflow import runtime_minutes

    assert runtime_minutes(walltime) == minutes


def test_declared_walltime_reaches_the_snakemake_rule():
    """End-to-end: a day-prefixed `time:` lands in the rule as `runtime=`."""
    from tvbo.cli.workflow import _render_template

    smk = _render_template(
        "snakemake/study.smk.mako", block={}, bundled_code=False,
        exp_plans=[{"key": "30", "rule_name": "exp_30", "spec_relpath": "spec/30/experiment.yaml",
                    "select": None, "backend": "tvboptim", "out_dir": "results",
                    "result_stem": "result", "container": None, "axes": [], "depends_on": [],
                    "block": {"cpus_per_task": 2, "mem": "8G", "time": "3-00:00:00"}}])
    assert "runtime=4320" in smk


def test_fanout_snakefile_is_executable_python():
    """The emitted Snakefile must actually *run*, not merely contain the right text.

    Paths are built with f-strings (to interpolate OUT_DIR), and an f-string eats
    single braces — so a wildcard written as `{subject}` is evaluated as a Python
    name and the Snakefile dies at parse time with `NameError: name 'subject' is not
    defined`, before Snakemake sees a single rule. Text assertions sail straight past
    that, so execute the module instead. `expand`/`f-string` must yield the literal
    `{subject}` a wildcard needs.
    """
    from tvbo.cli.workflow import _render_template

    smk = _render_template(
        "snakemake/study.smk.mako", block={}, bundled_code=False,
        exp_plans=[{"key": "30", "rule_name": "exp_30", "spec_relpath": "spec/30/experiment.yaml",
                    "select": None, "backend": "tvboptim", "out_dir": "results",
                    "result_stem": "result", "container": None, "block": {}, "depends_on": [],
                    "axes": [{"name": "subject", "parameter": "dataset.active_subject",
                              "values": ["100206", "100307"]}]}])

    # Evaluate every path f-string the Snakefile builds, with only OUT_DIR bound —
    # exactly the namespace Snakemake parses them in. A wildcard that leaked a single
    # brace resolves as a Python name here and raises NameError.
    literals = re.findall(r'f"[^"\n]*"', smk)
    assert literals, "expected f-string paths in the emitted Snakefile"
    resolved = [eval(lit, {"OUT_DIR": "results"}) for lit in literals]

    # ...and the wildcard must survive that evaluation as a literal for expand()/output:.
    assert "results/30/sub-{subject}_result.h5" in resolved


def _snakemake_cmd(tmp_path, monkeypatch, *, sbatch: bool, ship_profile: bool,
                   cores=None, profile=None):
    """Capture the snakemake argv `_execute_engine_artefact` would run for a kit.

    *sbatch* toggles whether a scheduler is discoverable; *ship_profile* whether the
    kit carries a SLURM `profile/`. Returns the launched command (minus argv[0]).
    """
    from tvbo.cli import workflow as wf

    kit = tmp_path / "kit"
    (kit / "profile").mkdir(parents=True) if ship_profile else kit.mkdir()
    if ship_profile:
        (kit / "profile" / "config.yaml").write_text("executor: slurm\n", encoding="utf-8")
    (kit / "Snakefile").write_text("rule all:\n    input: []\n", encoding="utf-8")

    # has_scheduler consults _resolve_launcher (which itself checks the venv sibling AND
    # $PATH), so mocking it fully controls both the launcher path and scheduler discovery.
    monkeypatch.setattr(wf, "_resolve_launcher",
                        lambda n: (f"/fake/{n}" if n == "snakemake"
                                   else (f"/fake/{n}" if (n == "sbatch" and sbatch) else None)))

    captured = {}

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(wf.subprocess, "run", _fake_run)
    wf._execute_engine_artefact("snakemake", kit / "Snakefile", profile=profile, cores=cores)
    return captured["cmd"]


def test_snakemake_explicit_cores_runs_local_ignoring_shipped_profile(tmp_path, monkeypatch):
    """`--cores` forces a LOCAL run even when the kit ships a SLURM profile and a scheduler exists."""
    cmd = _snakemake_cmd(tmp_path, monkeypatch, sbatch=True, ship_profile=True, cores="4")
    assert cmd[1:] == ["--cores", "4"]
    assert "--profile" not in cmd


def test_snakemake_shipped_profile_used_only_when_scheduler_present(tmp_path, monkeypatch):
    """With a scheduler present the shipped SLURM profile is used (submit each rule)."""
    cmd = _snakemake_cmd(tmp_path, monkeypatch, sbatch=True, ship_profile=True)
    assert cmd[1:] == ["--profile", "profile"]


def test_snakemake_falls_back_to_local_cores_without_a_scheduler(tmp_path, monkeypatch):
    """A kit's SLURM profile can't work on a machine with no `sbatch`; a bare run must
    still execute — auto-fall back to local cores so the SAME kit runs natively locally."""
    cmd = _snakemake_cmd(tmp_path, monkeypatch, sbatch=False, ship_profile=True)
    assert cmd[1:] == ["--cores", "all"]
    assert "--profile" not in cmd


def test_snakemake_explicit_profile_wins_over_shipped(tmp_path, monkeypatch):
    """`--profile cubi-v1` replaces the shipped profile regardless of scheduler discovery."""
    cmd = _snakemake_cmd(tmp_path, monkeypatch, sbatch=False, ship_profile=True, profile="cubi-v1")
    assert cmd[1:] == ["--profile", "cubi-v1"]


@pytest.mark.parametrize("engine,expected_tail", [
    ("snakemake", "--dry-run"),
    ("slurm", "--test-only"),
])
def test_submit_dry_run_reports_without_queueing(tmp_path: Path, monkeypatch, engine, expected_tail):
    """`--dry-run` must reach the engine and must not queue anything.

    Validating a kit before a large submission is the whole point, so the Slurm
    path must also skip the `finalize.sbatch` afterok chain — there is no array job
    id to depend on when nothing was submitted.
    """
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)
    # Pretend the launcher is installed; keep the basename so argv[0] stays recognisable.
    monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")

    out = tmp_path / "kit"
    assert runner.invoke(app, ["workflow", engine, EXP, "--backend", "jax",
                               "-o", str(out)]).exit_code == 0
    r = runner.invoke(app, ["workflow", "submit", str(out), "--dry-run"])
    assert r.exit_code == 0, r.stdout

    submits = [c for c in calls if c and Path(c[0]).name in {"sbatch", "snakemake", "nextflow"}]
    assert submits, "expected submit to invoke the engine"
    assert expected_tail in submits[0], f"{expected_tail} missing from {submits[0]}"
    assert not any("--dependency=afterok" in a for c in submits for a in c)


def test_scalar_set_override_lands_as_one_bind(tmp_path: Path):
    """`--set container_binds=/data/cephfs-1` is ONE bind, not one per character.

    A `--set` override arrives as a string into a multivalued slot. Strings are
    iterable, so a naive `list()` expands `/data/cephfs-1` into 14 single-character
    binds and the emitted flag reads `--bind /,d,a,t,a,…` — which apptainer accepts
    as nonsense mounts rather than failing loudly.
    """
    from tvbo.utils import as_list

    assert as_list("/data/cephfs-1") == ["/data/cephfs-1"]
    assert as_list(["/a", "/b"]) == ["/a", "/b"]
    assert as_list(None) == []

    from tvbo.cli.workflow import _write_snakemake_profile

    _write_snakemake_profile(tmp_path, {}, plan=_container_plan(binds=["/data/cephfs-1"]))
    cfg = (tmp_path / "profile" / "config.yaml").read_text()
    assert 'apptainer-args: "--bind /data/cephfs-1"' in cfg


def _container_plan(image="docker://ghcr.io/virtual-twin/tvbo:dev",
                    binds=("/data/cephfs-1",), args=None):
    """Minimal stand-in for the container fields the profile writer reads."""
    from tvbo.cli._workflow import WorkflowPlan

    return SimpleNamespace(
        container=image, container_binds=list(binds), container_args=args,
        container_exec_flags=WorkflowPlan.container_exec_flags.fget(
            SimpleNamespace(container_binds=list(binds), container_args=args)),
    )


def _layer_plan(container="/w/tvbo-dev.sif", reqs=({"package": "igl"},), binds=("/data/cephfs-1",)):
    """A plan-like stand-in exposing exactly what the layer templates read."""
    from tvbo.cli._workflow import WorkflowPlan

    p = SimpleNamespace(
        study_key="S", experiment_key="fig6", chunk=156, n_array_tasks=156,
        n_workflow_cells=1560, n_vectorize_cells=1560, vectorize_axes=[], workflow_axes=[],
        wildcards=[], overrides=[], container=container, container_binds=list(binds),
        container_args=None, requirements=list(reqs), experiment_selector=None,
        source_spec=None, out_dir="out/S/fig6", run_spec="experiment:fig6",
        backend=SimpleNamespace(name="tvboptim"),
        engine_block={"partition": "medium", "mem": "16G", "time": "02:00:00", "cpus_per_task": 4},
    )
    for prop in ("pip_specs", "needs_env_layer", "needs_container_layer",
                 "container_extras_venv", "container_exec_flags"):
        setattr(p, prop, getattr(WorkflowPlan, prop).fget(p))
    return p


def test_env_layer_provisions_requirements_container_or_not():
    """Requirements need provisioning wherever the tasks run: `needs_env_layer` is true as
    soon as requirements exist (a native venv, or one layered on a container). The narrower
    `needs_container_layer` only fires when a container is ALSO declared (the Slurm --env
    path); with requirements but no container it stays a native venv."""
    assert _layer_plan().needs_env_layer is True
    assert _layer_plan(container=None).needs_env_layer is True          # native venv
    assert _layer_plan(reqs=()).needs_env_layer is False                # nothing to provision

    assert _layer_plan().needs_container_layer is True
    assert _layer_plan(container=None).needs_container_layer is False   # no image to layer onto
    assert _layer_plan(reqs=()).needs_container_layer is False


def test_concrete_container_reference_passes_through_unchanged():
    """A pinned image — local .sif path or a tagged/digested registry ref — is the
    author's exact choice and must survive resolution verbatim."""
    from tvbo.cli._workflow import resolve_container_ref

    for ref in ("~/work/tvbo-dev.sif", "/abs/img.simg",
                "docker://ghcr.io/virtual-twin/tvbo:dev",
                "docker://ghcr.io/virtual-twin/tvbo:0.5.3",
                "docker://ghcr.io/virtual-twin/tvbo@sha256:abc"):
        assert resolve_container_ref(ref) == ref


def test_unpinned_reference_resolves_to_version_matched_image(monkeypatch):
    """A reference that leaves the version open — the symbolic `tvbo`, or a tvbo
    registry ref with no tag — pulls the image matching the running CLI, so the kit
    runs the tvbo it was emitted with instead of failing to resolve."""
    from tvbo.cli import _workflow

    monkeypatch.delenv("TVBO_CONTAINER", raising=False)
    monkeypatch.delenv("TVBO_CONTAINER_IMAGE", raising=False)
    monkeypatch.setenv("TVBO_CONTAINER_TAG", "9.9.9")
    want = "docker://ghcr.io/virtual-twin/tvbo:9.9.9"
    assert _workflow.resolve_container_ref("tvbo") == want
    assert _workflow.resolve_container_ref("default") == want
    assert _workflow.resolve_container_ref("docker://ghcr.io/virtual-twin/tvbo") == want


def test_no_container_means_none_even_with_requirements():
    """Requirements do NOT force a container: an undeclared container stays None (the deps
    are provisioned into a native venv by setup.sh), and no container + no requirements is
    a bare run. The `container` field alone chooses the substrate."""
    from tvbo.cli import _workflow

    assert _workflow.resolve_container_ref(None) is None
    assert _workflow.resolve_container_ref("") is None


def test_full_container_env_override_wins_verbatim(monkeypatch):
    """TVBO_CONTAINER supplies a complete reference — a site mirror, a pinned digest —
    that overrides both the default repository and tag."""
    from tvbo.cli import _workflow

    monkeypatch.setenv("TVBO_CONTAINER", "docker://mirror.local/tvbo:pinned")
    assert _workflow.resolve_container_ref("tvbo") == "docker://mirror.local/tvbo:pinned"


def test_fan_input_expr_expands_over_every_fanned_cell():
    """A figure (or cross-experiment dep) that reads a FANNED experiment must depend on
    ALL its cells, so it waits for the whole sweep — the input is the `expand()` over the
    fan's value lists. A group run (no axes) is its single result path."""
    from tvbo.cli._workflow import fan_input_expr

    fanned = {"key": "41", "rule_name": "exp_41",
              "result_stem": "exp-41_desc-Kuramoto_result",
              "axes": [{"name": "a", "parameter": "K.a", "values": [1, 2]},
                       {"name": "conduction_speed",
                        "parameter": "network.conduction_speed", "values": [6]}]}
    expr = fan_input_expr(fanned)
    assert expr.startswith("expand(")
    # the cell pattern (doubled wildcards for the carrying f-string) + the fan value lists
    assert ('f"{OUT_DIR}/41/a={{a}}/conduction_speed={{conduction_speed}}/'
            'exp-41_desc-Kuramoto_result.h5"') in expr
    assert "a=EXP_41_A" in expr and "conduction_speed=EXP_41_CONDUCTION_SPEED" in expr

    group = {"key": "9", "rule_name": "exp_9", "result_stem": "exp-9_result", "axes": []}
    assert fan_input_expr(group) == 'f"{OUT_DIR}/9/exp-9_result.h5"'


def test_setup_sh_layers_requirements_onto_the_image_via_system_site_venv():
    """setup.sh must build the venv WITH --system-site-packages (so pip installs only the
    delta and reuses the image's packages) and install with the IMAGE's own pip (so a
    native wheel like igl is ABI-correct) — not rebuild the SIF."""
    from tvbo.cli.workflow import _render_template

    sh = _render_template("setup.sh.mako", plan=_layer_plan())
    assert "python -m venv --system-site-packages" in sh
    assert "/w/tvbo-dev.sif" in sh and "--bind /data/cephfs-1" in sh
    # install runs through the container's interpreter, into the venv
    assert 'singularity exec' in sh and '${VENV}/bin/pip" install' in sh
    assert 'VENV=".tvbo-extras-venv"' in sh and "-r requirements.txt" in sh


def test_setup_sh_provisions_a_native_venv_when_no_container():
    """With requirements but NO container, setup.sh builds a NATIVE --system-site-packages
    venv (no `singularity exec` prefix) — so `tvbo workflow submit` provisions the study's
    deps on the host with no manual `pip install`."""
    from tvbo.cli.workflow import _render_template

    sh = _render_template("setup.sh.mako", plan=_layer_plan(container=None, binds=()))
    assert "python -m venv --system-site-packages" in sh
    assert "-r requirements.txt" in sh and 'VENV=".tvbo-extras-venv"' in sh
    assert "singularity exec" not in sh          # native: no container to exec into
    assert ".sif" not in sh


def test_snakemake_rule_prepends_the_native_venv_to_pythonpath():
    """A native run (no container) still prepends the requirements venv to PYTHONPATH — the
    Snakemake prepend is a plain shell export, substrate-agnostic — so a host observation's
    `import igl` resolves without any manual install."""
    from tvbo.cli.workflow import _render_template

    ep = {"key": "fig6", "rule_name": "exp_fig6", "spec_relpath": "spec/fig6/experiment.yaml",
          "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
          "container": None, "needs_env_layer": True, "extras_venv": ".tvbo-extras-venv",
          "block": {}, "axes": [], "depends_on": []}
    smk = _render_template("snakemake/study.smk.mako", exp_plans=[ep], block={}, bundled_code=False)
    assert "export PYTHONPATH=$(echo .tvbo-extras-venv/lib/python*/site-packages):" in smk
    assert "container:" not in smk               # native: no container directive


def test_run_sbatch_exposes_the_layer_via_pythonpath_and_guards_on_setup():
    """Each task must see the layered deps (PYTHONPATH into the venv site-packages) and
    fail loudly if setup.sh was never run, rather than crashing mid-import on the node."""
    from tvbo.cli.workflow import _render_template

    sb = _render_template("slurm/run.sbatch.mako", plan=_layer_plan(),
                          block=_layer_plan().engine_block, spec_relpath=None, bundled_code=False)
    assert 'if [ ! -d "${TVBO_EXTRAS}" ]' in sb            # guard: setup.sh must run first
    assert 'TVBO_EXTRAS=$(echo "$(pwd)/.tvbo-extras-venv"/lib/python*/site-packages)' in sb
    assert '--env PYTHONPATH="${TVBO_EXTRAS}${PYTHONPATH:+:$PYTHONPATH}"' in sb
    assert "singularity exec" in sb and "tvbo run" in sb


def test_snakemake_rule_prepends_the_container_layer_to_pythonpath():
    """The Snakemake fan-out runs each cell's `tvbo run` INSIDE the container, so the
    layered deps (setup.sh's venv) must reach it via PYTHONPATH on every rule — this is
    the per-cell path a host (igl) observation needs, where slurm's --shard vmaps a chunk.
    Double braces survive Snakemake's `.format()` (single braces are wildcards)."""
    from tvbo.cli.workflow import _render_template

    ep = {"key": "fig6", "rule_name": "exp_fig6", "spec_relpath": "spec/fig6/experiment.yaml",
          "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
          "container": "/w/tvbo-dev.sif", "needs_env_layer": True,
          "extras_venv": ".tvbo-extras-venv", "block": {}, "axes": [], "depends_on": []}
    smk = _render_template("snakemake/study.smk.mako", exp_plans=[ep], block={}, bundled_code=False)
    assert ('"export PYTHONPATH=$(echo .tvbo-extras-venv/lib/python*/site-packages):'
            '${{PYTHONPATH:-}} && "') in smk
    # the layer precedes the run and the rule execs in the image
    assert smk.index("PYTHONPATH=$(echo .tvbo-extras") < smk.index("tvbo run spec/fig6")
    assert "container:" in smk and "/w/tvbo-dev.sif" in smk


def test_snakemake_fans_a_model_param_axis_via_pin_not_set():
    """A fanned exploration axis must emit `--pin`, not `--set`: `--pin` sets the base
    parameter AND drops the axis so the cell is a single point (its host observation lands
    there), whereas `--set` on a swept model param neither resolves nor collapses the sweep.
    The dataset subject axis keeps `--subject`."""
    from tvbo.cli.workflow import _render_template

    ep = {"key": "fig6", "rule_name": "exp_fig6", "spec_relpath": "spec/fig6/experiment.yaml",
          "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
          "container": None, "block": {}, "depends_on": [],
          "axes": [{"name": "omega_mean_hz", "parameter": "Kuramoto.omega_mean_hz", "values": [10, 20]},
                   {"name": "conduction_speed", "parameter": "network.conduction_speed", "values": [3, 6]}]}
    smk = _render_template("snakemake/study.smk.mako", exp_plans=[ep], block={}, bundled_code=False)
    assert "--pin=Kuramoto.omega_mean_hz={wildcards.omega_mean_hz}" in smk
    assert "--pin=network.conduction_speed={wildcards.conduction_speed}" in smk
    assert "--set=" not in smk


def test_submit_provisions_the_container_layer_before_submitting(tmp_path, monkeypatch):
    """`tvbo workflow submit <archive>` must run setup.sh itself, so the whole cluster step
    is one command (no manual `bash setup.sh`). A layer failure aborts the submit."""
    from tvbo.cli import workflow as wf

    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "setup.sh").write_text("#!/bin/bash\necho layered\n")
    calls = []
    monkeypatch.setattr(wf.subprocess, "run",
                        lambda *a, **k: calls.append((a, k)) or type("R", (), {"returncode": 0})())
    wf._provision_env_layer(kit, dry_run=False)
    assert calls and calls[0][0][0] == ["bash", "setup.sh"] and calls[0][1]["cwd"] == str(kit)


def test_submit_without_a_layer_provisions_nothing(tmp_path, monkeypatch):
    """No setup.sh (kit declares no layer) → submit runs nothing extra."""
    from tvbo.cli import workflow as wf

    ran = []
    monkeypatch.setattr(wf.subprocess, "run", lambda *a, **k: ran.append(a))
    wf._provision_env_layer(tmp_path, dry_run=False)   # empty dir, no setup.sh
    assert not ran


def test_snakemake_rule_without_layer_has_no_pythonpath_injection():
    """The layer is strictly opt-in: an experiment that declares no layer emits a bare rule."""
    from tvbo.cli.workflow import _render_template

    ep = {"key": "e", "rule_name": "exp_e", "spec_relpath": "spec/e/experiment.yaml",
          "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
          "container": None, "block": {}, "axes": [], "depends_on": []}
    smk = _render_template("snakemake/study.smk.mako", exp_plans=[ep], block={}, bundled_code=False)
    assert ".tvbo-extras-venv" not in smk


def test_no_container_layer_means_no_pythonpath_injection():
    """Without a container the run line stays bare — the layer is strictly opt-in."""
    from tvbo.cli.workflow import _render_template

    plan = _layer_plan(container=None)
    sb = _render_template("slurm/run.sbatch.mako", plan=plan, block=plan.engine_block,
                          spec_relpath=None, bundled_code=False)
    assert "TVBO_EXTRAS" not in sb and "--env PYTHONPATH" not in sb
    assert "singularity exec" not in sb


def test_profile_carries_container_binds(tmp_path: Path):
    """``container_binds`` is what makes the container usable at a site.

    A container sees only ``$HOME`` and ``$PWD`` by default, so a home directory
    that symlinks into another filesystem dangles inside it — and a library that
    touches such a path at import time fails before the task starts. The binds
    reach the run as ``apptainer-args``.
    """
    from tvbo.cli.workflow import _write_snakemake_profile

    _write_snakemake_profile(tmp_path, {"partition": "medium"}, plan=_container_plan())
    cfg = (tmp_path / "profile" / "config.yaml").read_text()
    assert 'apptainer-args: "--bind /data/cephfs-1"' in cfg
    assert "slurm_partition: medium" in cfg


@pytest.mark.parametrize("mem,mib", [
    ("8G", 8192),        # binary, not 8000 — Slurm's --mem=8G is 8 GiB
    ("8GB", 8192),
    ("128G", 131072),
    ("512M", 512),
    ("1T", 1048576),     # every sbatch suffix parses; an unknown one used to vanish
    ("2000", 2000),      # bare number is already MiB
    ("512K", 1),         # sub-mebibyte rounds up; 0 would reserve nothing
    (None, None),
    ("garbage", None),
    ("8X", None),        # unrecognised suffix -> None, never a wrong number
])
def test_mem_mb_uses_binary_units_and_every_sbatch_suffix(mem, mib):
    """A declared memory size must reach Slurm as the size that was declared.

    Two failure modes this pins: a decimal conversion under-reserves against a
    binary ``--mem`` and OOM-kills a task sized to its own limit, and an
    unrecognised suffix returning None drops the resource entirely so the job
    silently inherits the partition default.
    """
    from tvbo.cli._workflow import mem_mb

    assert mem_mb(mem) == mib


def test_bind_paths_survive_spaces_and_commas():
    """Binds reach the runtime as distinct, shell-safe arguments.

    A comma cannot be escaped inside one ``--bind``, and the Slurm emitters splice
    these flags straight into a command line — so comma-joining or leaving a space
    unquoted turns one bind into two bogus arguments.
    """
    from tvbo.cli._workflow import WorkflowPlan

    flags = WorkflowPlan.container_exec_flags.fget(
        SimpleNamespace(container_binds=["/data/cephfs-1", "/my scratch"],
                        container_args=None))
    assert flags == "--bind /data/cephfs-1 --bind '/my scratch'"


def test_profile_quotes_container_args_containing_a_quote():
    """`container_args` is the verbatim escape hatch, so it may contain a quote.

    Interpolated raw into a double-quoted YAML scalar it closes the scalar early
    and the whole profile fails to parse.
    """
    import yaml

    from tvbo.cli.workflow import _write_snakemake_profile
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        plan = _container_plan(binds=["/data"], args='--env FOO="bar"')
        _write_snakemake_profile(Path(td), {"partition": "medium"}, plan=plan)
        cfg = yaml.safe_load((Path(td) / "profile" / "config.yaml").read_text())
    assert cfg["apptainer-args"] == '--bind /data --env FOO="bar"'


def test_profile_omits_default_resources_when_nothing_to_put_in_it():
    """A `default-resources:` key whose only content is a comment parses as null."""
    import yaml
    import tempfile

    from tvbo.cli.workflow import _write_snakemake_profile

    with tempfile.TemporaryDirectory() as td:
        _write_snakemake_profile(Path(td), {}, plan=_container_plan(image=None))
        cfg = yaml.safe_load((Path(td) / "profile" / "config.yaml").read_text())
    assert "default-resources" not in cfg


def test_per_experiment_partition_reaches_its_own_rule():
    """Partition is declarable per experiment, so it cannot live only in the profile.

    The profile carries one study-wide default; an experiment that overrides it
    (a long sweep needing a longer-walltime partition) must carry its own, or it
    is submitted to a sibling's partition and killed at that partition's cap.
    """
    from tvbo.cli.workflow import _render_template

    def ep(key, block):
        return {"key": key, "rule_name": f"exp_{key}", "spec_relpath": f"spec/{key}/experiment.yaml",
                "select": None, "backend": "tvboptim", "out_dir": "results", "result_stem": "result",
                "container": None, "block": block, "axes": [], "depends_on": []}

    smk = _render_template(
        "snakemake/study.smk.mako", block={}, bundled_code=False,
        exp_plans=[ep("1", {"partition": "short"}),
                   ep("30", {"partition": "medium", "time": "24:00:00"})])
    long_rule = smk[smk.index("rule exp_30:"):]
    assert 'slurm_partition="medium"' in long_rule
    assert 'slurm_partition="short"' not in long_rule


def test_snakemake_rule_activates_declared_venv_and_modules():
    """A rule's shell runs in a clean shell on the compute node.

    The kit is emitted by one interpreter and executed by another, so a declared
    `venv:`/`modules:` must be activated inside the rule or the job dies on
    `tvbo: command not found` seconds after it starts. The Slurm emitter has always
    done this; the Snakemake emitter silently ignored both.
    """
    from tvbo.cli.workflow import _render_template

    smk = _render_template(
        "snakemake/study.smk.mako", block={}, bundled_code=False,
        exp_plans=[{"key": "30", "rule_name": "exp_30", "spec_relpath": "spec/30/experiment.yaml",
                    "select": None, "backend": "tvboptim", "out_dir": "results",
                    "result_stem": "result", "container": None, "axes": [], "depends_on": [],
                    "block": {"venv": "/work/env", "modules": ["python/3.12"],
                              "setup": ["export FOO=1"]}}])
    shell = smk[smk.index("    shell:"):]
    assert "module load python/3.12 && " in shell
    assert "source /work/env/bin/activate && " in shell
    # Order matters: modules, then venv, then setup.
    assert shell.index("module load") < shell.index("source /work/env") < shell.index("export FOO=1")


def test_venv_path_with_a_space_is_shell_quoted():
    """An unquoted path splits into two shell words and `source` reads the wrong file."""
    from tvbo.cli.workflow import _render_template

    smk = _render_template(
        "snakemake/study.smk.mako", block={}, bundled_code=False,
        exp_plans=[{"key": "30", "rule_name": "exp_30", "spec_relpath": "spec/30/experiment.yaml",
                    "select": None, "backend": "tvboptim", "out_dir": "results",
                    "result_stem": "result", "container": None, "axes": [], "depends_on": [],
                    "block": {"venv": "/work/my env"}}])
    assert "source '/work/my env'/bin/activate && " in smk


def test_experiment_override_does_not_clear_inherited_list():
    """An override replaces only the slots its author filled in.

    LinkML gives an unfilled multivalued slot an empty list, not None — so an
    experiment that overrides just its walltime still carries
    `container_binds: []`. Merged naively that wipes the study's binds, and only
    that experiment's tasks run unbound: they die at import time, far from the
    walltime override that caused it.
    """
    from tvbo.cli._workflow import merge_workflow_spec
    from types import SimpleNamespace as NS

    study = NS(workflow=NS(container="img", container_binds=["/data/cephfs-1"],
                           container_args=None, requirements=[]))
    exp = NS(workflow=NS(container=None, container_binds=[], container_args=None,
                         requirements=[], slurm=NS(time="48:00:00")))

    merged = merge_workflow_spec(study, exp)
    assert merged["container_binds"] == ["/data/cephfs-1"]
    assert merged["container"] == "img"
    assert merged["slurm"]["time"] == "48:00:00"


def test_nextflow_enables_the_container_runtime():
    """`process.container` is inert unless a runtime is switched on.

    Without `singularity.enabled` Nextflow runs the task on the bare host and the
    declared image — and every bind — is silently ignored.
    """
    from tvbo.cli.workflow import _render_template

    plan = SimpleNamespace(
        container="docker://ghcr.io/virtual-twin/tvbo:dev",
        container_exec_flags="--bind /data/cephfs-1",
        backend=SimpleNamespace(name="tvboptim"), study_key="s", experiment_key="30",
        wildcards=[], vectorize_axes=[], workflow_axes=[], out_dir="results",
        n_array_tasks=1, chunk=1,
    )
    nf = _render_template("nextflow/main.nf.mako", plan=plan, block={}, bundled_code=False)
    assert "singularity.enabled = true" in nf
    assert "singularity.runOptions = '--bind /data/cephfs-1'" in nf


def test_profile_accepts_prebuilt_image_path(tmp_path: Path):
    """A path to an image already built on the target is a valid ``container``.

    It pins the exact image and needs no pull, which is why the schema carries no
    staging-directory slot: where a registry reference gets materialised is a
    run-time concern (``--apptainer-prefix``), not part of the study.
    """
    from tvbo.cli.workflow import _write_snakemake_profile

    plan = _container_plan(image="/data/containers/tvbo-dev.sif")
    _write_snakemake_profile(tmp_path, {}, plan=plan)
    cfg = (tmp_path / "profile" / "config.yaml").read_text()
    assert "software-deployment-method:\n  - apptainer" in cfg
    assert 'apptainer-args: "--bind /data/cephfs-1"' in cfg
    assert "apptainer-prefix" not in cfg


def test_profile_omits_apptainer_keys_when_no_container(tmp_path: Path):
    """No `container:` directive means no Apptainer section to configure."""
    from tvbo.cli.workflow import _write_snakemake_profile

    _write_snakemake_profile(tmp_path, {}, plan=_container_plan(image=None))
    cfg = (tmp_path / "profile" / "config.yaml").read_text()
    assert "apptainer" not in cfg


def test_study_readme_covers_every_experiment():
    """A whole-study README must describe the whole study, not its last experiment.

    ``_emit_snakemake_study`` freezes each experiment in a loop; deriving the README
    from the loop variable made an 18-experiment kit announce itself as its final
    experiment with ``workflow cells: 1``.
    """
    from types import SimpleNamespace as NS

    from tvbo.cli.workflow import _write_readme

    def plan(key, jobs, vec_cells):
        return NS(study_key="Koller2024", experiment_key=key, backend=NS(name="tvboptim", label="tvboptim"),
                  container=None, n_workflow_cells=jobs, n_vectorize_cells=vec_cells, chunk=1,
                  n_array_tasks=1, vectorize_axes=[], workflow_axes=[NS(name="K")], overrides=[])

    # Exp 40/48 vectorize their whole 4x39x10 grid into one job; exp 52 is a single run.
    plans = [plan("40", 1, 1560), plan("48", 1, 1560), plan("52", 1, 1)]

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_readme(out, engine="snakemake", plans=plans, script_relpath=None,
                      spec_layout="spec/<experiment>/experiment.yaml")
        readme = (out / "README.md").read_text()

    # Titled by the study, never "study / <last experiment>".
    assert readme.startswith("# Koller2024\n")
    assert "# Koller2024 / 52" not in readme
    # Every experiment is named, and the cell count is the total, not the last one's.
    for key in ("40", "48", "52"):
        assert f"`{key}`" in readme
    assert "experiments   : 3" in readme
    # A vectorized grid must not be reported as a single cell: 3 jobs, 3121 sims.
    assert "3121 simulation cells" in readme
    assert "| `40` | 1 | 1560 |" in readme


def test_workflow_slurm_emits_kit(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "slurm", EXP, "--backend", "jax", "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    assert (out / "run.sbatch").is_file()
    sbatch = (out / "run.sbatch").read_text()
    assert sbatch.startswith("#!/bin/bash")
    assert "#SBATCH --array=" in sbatch


def test_workflow_slurm_emits_env_exports(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(
        app,
        [
            "workflow",
            "slurm",
            EXP,
            "--backend",
            "jax",
            "-o",
            str(out),
            "--set",
            "slurm.env.XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--set",
            "slurm.env.OMP_NUM_THREADS=1",
        ],
    )
    assert r.exit_code == 0, r.stdout
    sbatch = (out / "run.sbatch").read_text()
    # Values are shell-quoted (shlex.quote): safe tokens need no quotes.
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in sbatch
    assert "export OMP_NUM_THREADS=1" in sbatch


def test_env_set_merges_by_name_not_replace():
    """--set slurm.env.X overrides one var in a YAML env list, keeping the rest.

    Guards the GPU footgun: a YAML env: [{name,value}] list and a --set mapping
    must merge by name, else overriding one XLA flag silently drops the others
    (e.g. losing XLA_PYTHON_CLIENT_PREALLOCATE=false grabs all VRAM on GPU).
    """
    from tvbo.cli._workflow import _canonicalize_engine_maps, _normalize_env
    from tvbo.utils import deep_merge

    yaml_side = _canonicalize_engine_maps(
        {"slurm": {"env": [{"name": "XLA_PYTHON_CLIENT_PREALLOCATE", "value": "false"},
                           {"name": "XLA_FLAGS", "value": "--host_device_count=1"}]}}
    )
    cli_side = {"slurm": {"env": {"XLA_FLAGS": ""}}}  # --set slurm.env.XLA_FLAGS=""
    merged = deep_merge(yaml_side, cli_side)["slurm"]["env"]
    names = {e["name"]: e["value"] for e in _normalize_env(merged)}
    assert names["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"  # preserved
    assert names["XLA_FLAGS"] == "''"                          # overridden (empty, quoted)


def test_workflow_nextflow_emits_kit(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "nextflow", EXP, "--backend", "jax", "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    nf = (out / "main.nf").read_text()
    assert "nextflow.enable.dsl = 2" in nf
    assert "process tvbo_run" in nf


@pytest.mark.parametrize(
    "engine,artefact,opt_needle,env_needle",
    [
        ("slurm", "run.sbatch", "#SBATCH --qos=normal", "export OMP_NUM_THREADS=2"),
        ("snakemake", "Snakefile", 'slurm_partition="gpu"', "export OMP_NUM_THREADS=2 &&"),
        ("nextflow", "main.nf", "process.clusterOptions = '--gres=gpu:1'", "export OMP_NUM_THREADS=2"),
    ],
)
def test_env_and_options_render_across_engines(tmp_path, engine, artefact, opt_needle, env_needle):
    """env + options are name-keyed passthroughs rendered by every engine's emitter,
    each in its native form (Slurm #SBATCH / Snakemake resources / Nextflow process)."""
    opt = {"slurm": "qos=normal", "snakemake": "slurm_partition=gpu",
           "nextflow": "clusterOptions=--gres=gpu:1"}[engine]
    out = tmp_path / "kit"
    r = runner.invoke(app, [
        "workflow", engine, EXP, "--backend", "jax", "-o", str(out),
        "--set", f"{engine}.env.OMP_NUM_THREADS=2",
        "--set", f"{engine}.options.{opt}",
    ])
    assert r.exit_code == 0, r.stdout
    text = (out / artefact).read_text()
    assert opt_needle in text
    assert env_needle in text


def test_frozen_spec_captures_merged_workflow_for_reproducibility(tmp_path: Path):
    """The emitted spec records the effective workflow config (study < experiment <
    --set), so re-emitting from it reproduces the run with no flags — one-click
    provenance rather than the overrides living only in run.sbatch."""
    k1 = tmp_path / "k1"
    r = runner.invoke(app, [
        "workflow", "slurm", EXP, "--backend", "jax", "-o", str(k1),
        "--set", "slurm.partition=gpu", "--set", "slurm.array_chunk=1",
        "--set", "slurm.env.FOO=bar",
    ])
    assert r.exit_code == 0, r.stdout
    spec_yaml = next(p for p in (k1 / "spec").glob("*.yaml") if p.name != "network.yaml")
    frozen = spec_yaml.read_text()
    assert "partition: gpu" in frozen and "array_chunk: 1" in frozen and "FOO" in frozen

    # Re-emit from the frozen spec with NO --set → the GPU sbatch is reproduced.
    k2 = tmp_path / "k2"
    r2 = runner.invoke(app, ["workflow", "slurm", str(spec_yaml), "--backend", "jax", "-o", str(k2)])
    assert r2.exit_code == 0, r2.stdout
    sbatch = (k2 / "run.sbatch").read_text()
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --array=0-0" in sbatch          # array_chunk=1 → single shard
    assert "export FOO=bar" in sbatch


def test_workflow_stdout_only_does_not_create_kit(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "snakemake", EXP, "--backend", "jax",
                            "-o", str(out), "--stdout"])
    assert r.exit_code == 0, r.stdout
    assert "rule" in r.stdout
    assert not out.exists()


@pytest.mark.parametrize(
    "engine,expected_cmd,expected_file",
    [
        # Slurm submits the array with --parsable (then chains a gather job).
        ("slurm", ["sbatch", "--parsable", "run.sbatch"], "run.sbatch"),
        # Snakemake ships a SLURM-executor profile, so with a scheduler present submit
        # runs the login-node orchestrator that dispatches each rule to it (not local
        # --cores). Without a scheduler it auto-falls back to --cores; a `sbatch` mock
        # below pins the HPC branch so this asserts the profile path deterministically.
        ("snakemake", ["snakemake", "--profile", "profile"], "Snakefile"),
        ("nextflow", ["nextflow", "run", "main.nf"], "main.nf"),
    ],
)
def test_workflow_run_emits_and_executes_engine(tmp_path: Path, monkeypatch, engine, expected_cmd, expected_file):
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)
    # Pin a discoverable scheduler so the snakemake kit uses its shipped SLURM profile
    # (the HPC branch) rather than the no-`sbatch` local-cores fallback — the test's intent.
    monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")

    out = tmp_path / "kit"
    r = runner.invoke(
        app,
        ["workflow", "run", engine, EXP, "--backend", "jax", "-o", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    assert (out / expected_file).is_file()
    # Filter to the engine's submission commands — a globally-patched subprocess.run
    # also captures an incidental platform `uname -p` (jax init) on some hosts.
    submits = [c for c in calls
               if c["cmd"] and Path(c["cmd"][0]).name in {"sbatch", "snakemake", "nextflow"}]
    assert submits, "expected workflow run to execute engine command"
    # argv[0] is an absolute path when the launcher is installed beside this
    # interpreter, so compare the program name plus the arguments.
    assert [Path(submits[0]["cmd"][0]).name, *submits[0]["cmd"][1:]] == expected_cmd
    assert Path(submits[0]["cwd"]) == out
    # EXP is a single array task (chunk=1): that one task IS the whole result, so
    # slurm submits just the array — no gather job to reassemble a single shard.
    if engine == "slurm":
        assert not (out / "finalize.sbatch").exists()
        assert not any("--dependency=afterok" in a for c in submits for a in c["cmd"])


def test_workflow_run_slurm_chains_gather_when_sharded(tmp_path: Path, monkeypatch):
    """A multi-shard array (chunk>1) emits and chains a dependent gather job."""
    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)
    out = tmp_path / "kit"
    r = runner.invoke(
        app,
        ["workflow", "run", "slurm", EXP, "--backend", "jax",
         "--set", "distribute.chunk=4", "-o", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    assert (out / "finalize.sbatch").is_file()
    submits = [c for c in calls if c and c[0] == "sbatch"]
    assert any("--dependency=afterok" in a for c in submits for a in c)


def test_workflow_submit_accepts_packaged_archive(tmp_path: Path, monkeypatch):
    """`tvbo workflow submit` extracts a packaged .tar.gz kit and submits it.

    The archive ships on its own (no kit directory next to it), so the submit path
    must auto-extract before launching — the "no manual unzip" workflow.
    """
    import shutil
    import tarfile

    calls = []

    def _fake_run(cmd, check=True, cwd=None, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_run)
    monkeypatch.setattr(shutil, "which", lambda _launcher: "/usr/bin/snakemake")

    # emit a kit, then package it into an archive that travels on its own
    emitted = tmp_path / "emit" / "kit"
    r = runner.invoke(app, ["workflow", "snakemake", EXP, "--backend", "jax", "-o", str(emitted)])
    assert r.exit_code == 0, r.stdout
    ship = tmp_path / "ship"
    ship.mkdir()
    archive = ship / "kit.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        t.add(emitted, arcname="kit")

    # submit the archive directly: must extract beside it, then launch the engine
    r = runner.invoke(app, ["workflow", "submit", str(archive)])
    assert r.exit_code == 0, r.stdout
    assert (ship / "kit" / "Snakefile").is_file(), "archive must auto-extract beside itself"
    # The launcher is resolved to an absolute path when one is found, so match on the
    # program name rather than the exact argv[0].
    submits = [c for c in calls if c and Path(c[0]).name == "snakemake"]
    assert submits, "expected the extracted kit to be submitted"


def test_slurm_pack_emits_only_tarball(tmp_path: Path):
    """`tvbo workflow slurm … --pack` writes ONLY <kit>.tar.gz and removes the loose
    kit directory (the tarball is the shippable artifact; submit/run re-extract it)."""
    kit = tmp_path / "mykit"
    r = runner.invoke(app, ["workflow", "slurm", EXP, "--backend", "jax",
                            "-o", str(kit), "--pack"])
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "mykit.tar.gz").is_file(), "expected the packed tarball"
    assert not kit.exists(), "--pack must remove the loose kit directory"


def test_pack_warns_on_machine_specific_bids_root(tmp_path: Path, monkeypatch):
    """A per-subject dataset fan-out kit bakes an absolute ``bids_root`` that will not
    resolve on another host — emitting/packing it must warn with the exact submit-time
    override, so a kit is never shipped silently wrong. (Capture ``_common.warn``
    directly: caplog's root propagation is polluted by the shared tvbo logging setup.)"""
    from tvbo.cli import workflow as workflow_cli

    warned: list[str] = []
    monkeypatch.setattr("tvbo.cli._common.warn", lambda m: warned.append(m))

    kit = tmp_path / "kit"
    (kit / "spec").mkdir(parents=True)
    (kit / "spec" / "30.yaml").write_text(
        "dataset:\n  dataset_id: hcpya\n  bids_root: /Volumes/bronkodata/hcp/fc\n",
        encoding="utf-8",
    )
    workflow_cli._warn_machine_specific_bids_root(kit)
    joined = "\n".join(warned)
    assert warned, "expected a portability warning for the baked bids_root"
    assert "/Volumes/bronkodata/hcp/fc" in joined
    assert "--set dataset.bids_root=" in joined


def test_no_bids_root_warning_without_dataset(tmp_path: Path, monkeypatch):
    """A kit whose frozen spec has no ``dataset.bids_root`` (e.g. a group-average fit)
    emits no portability warning."""
    from tvbo.cli import workflow as workflow_cli

    warned: list[str] = []
    monkeypatch.setattr("tvbo.cli._common.warn", lambda m: warned.append(m))

    kit = tmp_path / "kit"
    (kit / "spec").mkdir(parents=True)
    (kit / "spec" / "34.yaml").write_text("model: ReducedWongWangEIB\n", encoding="utf-8")
    workflow_cli._warn_machine_specific_bids_root(kit)
    assert not warned


def test_workflow_run_rejects_unknown_engine():
    r = runner.invoke(app, ["workflow", "run", "local", EXP, "--backend", "jax"])
    assert r.exit_code != 0
    combined = (r.stdout or "") + (r.stderr or "")
    assert "expects engine one of" in combined


def test_workflow_run_slurm_array_smoke(tmp_path: Path, monkeypatch):
    """--array 0 passed to workflow run must appear in the sbatch call."""
    sbatch_calls = []
    emitted = {}

    def _fake_sbatch(cmd, check=True, cwd=None, **kwargs):
        if cmd[0] == "sbatch":
            sbatch_calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_sbatch)

    out = tmp_path / "kit"
    from tvbo.cli import workflow as workflow_cli

    original_emit = workflow_cli._emit

    def _capture_emit(engine, *, spec, backend, experiment, output, override, stdout):
        emitted["overrides"] = list(override)
        return original_emit(
            engine,
            spec=spec,
            backend=backend,
            experiment=experiment,
            output=output,
            override=override,
            stdout=stdout,
        )

    monkeypatch.setattr("tvbo.cli.workflow._emit", _capture_emit)

    r = runner.invoke(
        app,
        ["workflow", "run", "slurm", EXP, "--backend", "jax", "-o", str(out), "--array", "0"],
    )
    assert r.exit_code == 0, r.stdout
    assert (out / "run.sbatch").is_file(), "kit was not emitted"
    assert sbatch_calls, "sbatch was never called"
    sbatch_cmd = sbatch_calls[0]["cmd"]
    assert "--array=0" in sbatch_cmd
    assert sbatch_cmd[-1] == "run.sbatch"
    assert "slurm.array_chunk=1024" in emitted["overrides"]

    sbatch_text = (out / "run.sbatch").read_text()
    assert "#SBATCH --array=0-1023" in sbatch_text


def test_workflow_run_slurm_array_throttle(tmp_path: Path, monkeypatch):
    sbatch_calls = []

    def _fake_sbatch(cmd, check=True, cwd=None, **kwargs):
        if cmd[0] == "sbatch":
            sbatch_calls.append({"cmd": cmd, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout="12345\n")

    monkeypatch.setattr("tvbo.cli.workflow.subprocess.run", _fake_sbatch)

    out = tmp_path / "kit"
    r = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "slurm",
            EXP,
            "--backend",
            "jax",
            "-o",
            str(out),
            "--array",
            "0-39",
            "--array-throttle",
            "1",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert sbatch_calls, "sbatch was never called"
    assert "--array=0-39%1" in sbatch_calls[0]["cmd"]


# ---------------------------------------------------------------------------
# CLI: validate stubs (C5)
# ---------------------------------------------------------------------------

def test_validate_sedml_stub_rejects_non_xml(tmp_path: Path):
    fp = tmp_path / "fake.sedml"
    fp.write_text("not sedml")
    r = runner.invoke(app, ["validate", "sedml", str(fp)])
    assert r.exit_code != 0


def test_validate_sedml_stub_accepts_minimal(tmp_path: Path):
    fp = tmp_path / "fake.sedml"
    fp.write_text("<sedML xmlns='http://sed-ml.org/sed-ml/level1/version4'/>")
    r = runner.invoke(app, ["validate", "sedml", str(fp)])
    assert r.exit_code == 0


def test_validate_omex_stub_rejects_non_zip(tmp_path: Path):
    fp = tmp_path / "fake.omex"
    fp.write_bytes(b"not a zip")
    r = runner.invoke(app, ["validate", "omex", str(fp)])
    assert r.exit_code != 0


def test_validate_omex_stub_accepts_zip_with_manifest(tmp_path: Path):
    import zipfile
    fp = tmp_path / "fake.omex"
    with zipfile.ZipFile(fp, "w") as zf:
        zf.writestr("manifest.xml", "<omexManifest/>")
    r = runner.invoke(app, ["validate", "omex", str(fp)])
    assert r.exit_code == 0


# ---------------------------------------------------------------------------
# Backend templates expose a runnable __main__ entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["jax", "tvb", "tvboptim"])
def test_python_experiment_script_has_main_block(fmt):
    """All Python backends that emit a full experiment script must be runnable.

    The workflow kit (`tvbo workflow ...`) freezes the rendered script and
    invokes it via `python scripts/<exp>.py`, so the script itself must define
    an `if __name__ == "__main__":` block.
    """
    from tvbo import SimulationExperiment

    exp = SimulationExperiment.from_db("JR_MEG_FrequencyGradient_Optimization")
    code = exp.render(fmt)
    assert 'if __name__ == "__main__":' in code, (
        f"backend {fmt!r} renders a script without a __main__ entry point"
    )


def test_pde_experiment_template_has_main_block():
    """PDE template requires field_dynamics experiments; verify __main__ in source."""
    from pathlib import Path

    src = Path("tvbo/templates/pde/tvbo-pde-fem.py.mako").read_text()
    assert 'if __name__ == "__main__":' in src


def test_bundler_carries_a_callables_local_helper_but_not_stdlib_or_installed(tmp_path, monkeypatch):
    """A recipe callable often imports a LOCAL helper of its own (e.g. Koller's
    wave_detection_methods, pulled in via a runtime sys.path insert). The kit must carry it
    — else `import <helper>` fails on the node and the kit is not self-contained. But stdlib
    and installed packages must NOT be swept in (they ship via requirements)."""
    import sys
    from tvbo.cli.workflow import _local_module_deps

    (tmp_path / "helper_b.py").write_text("VALUE = 1\n")
    (tmp_path / "helper_a.py").write_text("import os\nimport json\nimport helper_b\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    a = importlib.import_module("helper_a")
    try:
        deps = dict(_local_module_deps(a, set()))
        assert "helper_b" in deps and deps["helper_b"].endswith("helper_b.py")   # the local helper
        assert "os" not in deps and "json" not in deps                            # stdlib excluded
    finally:
        for m in ("helper_a", "helper_b"):
            sys.modules.pop(m, None)


def test_benchmark_and_smoke_reach_the_snakemake_rule():
    """`--benchmark` attaches Snakemake's native `benchmark:` directive (per-cell TSV next
    to the output), and `--max-iterations`/`--smoke` threads `tvbo run --max-iterations`.
    Both are run modifiers carried on the exp_plan, never in the frozen workflow block."""
    from tvbo.cli.workflow import _render_template

    ep = {"key": "34", "rule_name": "exp_34", "spec_relpath": "spec/34/experiment.yaml",
          "select": None, "backend": "tvboptim", "out_dir": "results",
          "result_stem": "result", "container": None, "axes": [], "depends_on": [],
          "block": {"cpus_per_task": 2, "mem": "120G"},
          "benchmark": True, "max_iterations": 1}
    smk = _render_template("snakemake/study.smk.mako", block={}, bundled_code=False, exp_plans=[ep])
    assert "benchmark:" in smk
    assert "result.benchmark.tsv" in smk
    assert "--max-iterations 1" in smk

    # Off by default: no benchmark directive, no cap flag.
    ep_off = {**ep, "benchmark": False, "max_iterations": None}
    smk_off = _render_template("snakemake/study.smk.mako", block={}, bundled_code=False, exp_plans=[ep_off])
    assert "benchmark:" not in smk_off
    assert "--max-iterations" not in smk_off
