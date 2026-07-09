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
    assert (out / "spec" / "experiment.yaml").is_file()
    assert (out / "scripts" / "experiment.py").is_file()
    smk = (out / "Snakefile").read_text()
    assert "rule" in smk
    # When a frozen script exists, the rule should call it (not `tvbo run`)
    assert "python scripts/experiment.py" in smk


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
    assert 'export XLA_PYTHON_CLIENT_PREALLOCATE="false"' in sbatch
    assert 'export OMP_NUM_THREADS="1"' in sbatch


def test_workflow_nextflow_emits_kit(tmp_path: Path):
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "nextflow", EXP, "--backend", "jax", "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    nf = (out / "main.nf").read_text()
    assert "nextflow.enable.dsl = 2" in nf
    assert "process tvbo_run" in nf


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
        ("slurm", ["sbatch", "run.sbatch"], "run.sbatch"),
        ("snakemake", ["snakemake", "--cores", "all"], "Snakefile"),
        ("nextflow", ["nextflow", "run", "main.nf"], "main.nf"),
    ],
)
def test_workflow_run_emits_and_executes_engine(tmp_path: Path, monkeypatch, engine, expected_cmd, expected_file):
    calls = []

    def _fake_run(cmd, check, cwd=None):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tvbo.cli.run.subprocess.run", _fake_run)

    out = tmp_path / "kit"
    r = runner.invoke(
        app,
        ["workflow", "run", engine, EXP, "--backend", "jax", "-o", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    assert (out / expected_file).is_file()
    assert calls, "expected workflow run to execute engine command"
    assert calls[0]["cmd"] == expected_cmd
    assert calls[0]["check"] is True
    assert Path(calls[0]["cwd"]) == out


def test_workflow_run_rejects_unknown_engine():
    r = runner.invoke(app, ["workflow", "run", "local", EXP, "--backend", "jax"])
    assert r.exit_code != 0
    combined = (r.stdout or "") + (r.stderr or "")
    assert "expects engine one of" in combined


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
