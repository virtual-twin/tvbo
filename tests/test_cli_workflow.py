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
    assert keys == {"jax", "tvb", "pyrates", "tvboptim", "networkdynamics", "bifurcationkit", "numpy"}


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
        # Snakemake ships a SLURM-executor profile, so submit runs the login-node
        # orchestrator that dispatches each rule to the scheduler (not local --cores).
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
