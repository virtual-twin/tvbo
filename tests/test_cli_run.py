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
    # (Filter out unrelated subprocess calls a library may make, e.g. `uname -p`.)
    submits = [c for c in calls if c["cmd"][0] in {"sbatch", "snakemake", "nextflow"}]
    assert submits, calls
    assert all(Path(c["cwd"]) == kit_dir for c in submits)
    assert all(c["cmd"][0] != "tvbo" for c in calls)
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
