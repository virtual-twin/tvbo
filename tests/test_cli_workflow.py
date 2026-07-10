"""Tests for ``tvbo workflow`` (C4) and ``tvbo validate`` stubs (C5)."""
from __future__ import annotations

from pathlib import Path
import subprocess

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
        ("snakemake", ["snakemake", "--cores", "all"], "Snakefile"),
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
    assert calls, "expected workflow run to execute engine command"
    assert calls[0]["cmd"] == expected_cmd
    assert Path(calls[0]["cwd"]) == out
    # Slurm chains a dependent gather job after the array.
    if engine == "slurm":
        assert any("--dependency=afterok" in a for c in calls[1:] for a in c["cmd"])


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
