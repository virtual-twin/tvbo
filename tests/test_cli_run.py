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
    # Every shell-out is the engine submission from the kit dir; none re-invokes `tvbo`.
    assert calls and calls[0]["cmd"][0] in {"sbatch", "snakemake", "nextflow"}
    assert all(Path(c["cwd"]) == kit_dir for c in calls)
    assert all(c["cmd"][0] != "tvbo" for c in calls)
    # Slurm submits the array (--parsable) then chains a dependent gather job.
    if engine == "slurm":
        assert calls[0]["cmd"] == ["sbatch", "--parsable", "run.sbatch"]
        assert any("--dependency=afterok" in a for c in calls[1:] for a in c["cmd"])


def test_dispatch_to_engine_slurm_chains_gather_job(monkeypatch, tmp_path: Path):
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

    # Array job first (parsable), then the gather job with an afterok dependency.
    assert calls[0]["cmd"] == ["sbatch", "--parsable", "run.sbatch"]
    assert Path(calls[0]["cwd"]) == kit_dir
    assert len(calls) == 2
    assert "--dependency=afterok:27452074" in calls[1]["cmd"]
    assert calls[1]["cmd"][-1] == "finalize.sbatch"
    assert Path(calls[1]["cwd"]) == kit_dir
