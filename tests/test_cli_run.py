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

    def _fake_run(cmd, check, cwd=None):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0)

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
    # Exactly one shell-out: the engine submission, run from the kit dir.
    assert len(calls) == 1
    assert calls[0]["cmd"][0] in {"sbatch", "snakemake", "nextflow"}
    assert Path(calls[0]["cwd"]) == kit_dir
    # No call re-invokes `tvbo` on PATH.
    assert all(c["cmd"][0] != "tvbo" for c in calls)


def test_dispatch_to_engine_slurm_executes_emitted_script(monkeypatch, tmp_path: Path):
    calls = []

    def _fake_run(cmd, check, cwd=None):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0)

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

    assert len(calls) == 1
    assert calls[0]["cmd"] == ["sbatch", "run.sbatch"]
    assert Path(calls[0]["cwd"]) == kit_dir
