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
    """Regression: ``tvbo run --engine`` must pass a kit directory to workflow.

    Older code passed a file path to ``tvbo workflow <engine> -o ...``; since
    workflow expects an output directory, this produced an empty script path and
    Slurm failed with "Batch script is empty".
    """
    calls = []

    def _fake_run(cmd, check, cwd=None):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})
        # Mimic emission side-effect: create expected artefact in kit dir so the
        # follow-up execute step can reference it.
        if len(calls) == 1:
            out_idx = cmd.index("-o") + 1
            kit_dir = Path(cmd[out_idx])
            kit_dir.mkdir(parents=True, exist_ok=True)
            (kit_dir / expected_workflow_file).write_text("#!/bin/bash\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tvbo.cli.run.subprocess.run", _fake_run)

    run_cli._dispatch_to_engine(
        engine,
        spec="experiment:JR_MEG_FrequencyGradient_Optimization",
        backend="jax",
        experiment=None,
        container=None,
        out_dir=tmp_path / "kit",
    )

    assert len(calls) >= 1
    emit = calls[0]["cmd"]
    out_arg = Path(emit[emit.index("-o") + 1])
    # Critical assertion: output target is a directory (kit root), not artefact file.
    assert out_arg.name == "kit"
    assert out_arg.suffix == ""


def test_dispatch_to_engine_slurm_executes_emitted_script(monkeypatch, tmp_path: Path):
    calls = []

    def _fake_run(cmd, check, cwd=None):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})
        if len(calls) == 1:
            out_idx = cmd.index("-o") + 1
            kit_dir = Path(cmd[out_idx])
            kit_dir.mkdir(parents=True, exist_ok=True)
            (kit_dir / "run.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("tvbo.cli.run.subprocess.run", _fake_run)

    kit_dir = tmp_path / "kit"
    run_cli._dispatch_to_engine(
        "slurm",
        spec="experiment:JR_MEG_FrequencyGradient_Optimization",
        backend="jax",
        experiment=None,
        container=None,
        out_dir=kit_dir,
    )

    assert len(calls) >= 2
    assert calls[1]["cmd"] == ["sbatch", "run.sbatch"]
    assert Path(calls[1]["cwd"]) == kit_dir
