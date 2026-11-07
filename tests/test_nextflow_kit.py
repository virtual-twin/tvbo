"""A Nextflow kit runs the command the other launchers run, and Nextflow can read what it holds.

The pipeline's one process calls ``tvbo run`` on the frozen spec with one flag per fanned axis (``--pin`` for an exploration axis, ``--subject`` for the dataset fan-out) and writes into its own work directory; publishing puts each result at the path the Snakemake kit writes (``tvbo.run.workflow.cell_out_relpath``). Configuration (container, runtime, executor, process directives) is in ``nextflow.config`` beside the pipeline, the only place Nextflow reads it from.

The emitted text is pinned here on every run. The last test runs a kit through Nextflow itself and is skipped where the launcher is not installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from tvbo.cli import app
from tvbo.cli.workflow import _render_template

runner = CliRunner()

RECIPE = """
id: 200
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
    shift: {name: shift, value: 0.0}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + shift + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  nodes: [{id: 0, label: r0}, {id: 1, label: r1}]
  edges: [{source: 0, target: 1, weight: 0.5}, {source: 1, target: 0, weight: 0.5}]
integration: {method: RungeKutta4thOrder, duration: 10.0, step_size: 1.0, transient_time: 0.0}
explorations:
  sweep:
    name: sweep
    mode: product
    space:
      - {parameter: Kuramoto.omega, explored_values: [0.05, 0.06, 0.07]}
      - {parameter: Kuramoto.shift, explored_values: [0.0, 0.01]}
"""
FAN = "workflow:\n  distribute:\n    workflow: [%s]\n"
STEM = "exp-200_model-Kuramoto_result"


def _kit(tmp_path: Path, fanned: str = "", *extra: str) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE + (FAN % fanned if fanned else ""))
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", "nextflow", str(recipe), "--backend", "tvboptim", "-o", str(out), *extra])
    assert r.exit_code == 0, r.output
    return out


def _task_command(pipeline: str) -> str:
    return pipeline[pipeline.index("tvbo run") : pipeline.index('"""', pipeline.index("tvbo run"))]


def test_a_task_passes_tvbo_run_only_flags_it_defines(tmp_path: Path):
    """Every option on the task's command line is one ``tvbo run`` declares, so a kit cannot ask for a flag that does not exist."""
    command = _task_command((_kit(tmp_path, "Kuramoto.omega, Kuramoto.shift") / "main.nf").read_text())
    passed = set(re.findall(r"(?<=\s)(--?[a-z][a-z-]*)", command))
    defined = {opt for param in typer.main.get_command(app).commands["run"].params for opt in param.opts}
    assert passed and passed <= defined, sorted(passed - defined)


def test_a_fanned_axis_is_one_pinned_task_per_value_at_the_path_the_snakemake_kit_writes(tmp_path: Path):
    """One task per value of the fanned axis, pinned there, its result under ``<axis>=<value>/``; the axis left to the backend is no task input."""
    kit = _kit(tmp_path, "Kuramoto.omega")
    nf = (kit / "main.nf").read_text()
    assert "channel.of('0.05', '0.06', '0.07')" in nf
    assert "        val omega\n" in nf and "shift" not in nf[nf.index("input:") : nf.index("output:")]
    command = _task_command(nf)
    assert "--pin=Kuramoto.omega=${omega} " in command and "Kuramoto.shift" not in command
    assert command.rstrip().endswith("-o ../out/omega=${omega}")
    assert "publishDir \"${projectDir}/derivatives/tvbo/200\", mode: 'copy', saveAs: { name -> name.substring(4) }" in nf
    assert "path 'out/**'" in nf

    recipe = tmp_path / "recipe.yaml"
    r = runner.invoke(app, ["workflow", "snakemake", str(recipe), "--backend", "tvboptim", "-o", str(tmp_path / "smk")])
    assert r.exit_code == 0, r.output
    assert f'f"{{OUT_DIR}}/200/omega={{{{omega}}}}/{STEM}.h5"' in (tmp_path / "smk" / "Snakefile").read_text()


def test_two_fanned_axes_are_run_over_their_whole_product(tmp_path: Path):
    """Two axes are combined, so every pair of values is one task; paired element by element they would run only as many tasks as the shorter axis has values."""
    nf = (_kit(tmp_path, "Kuramoto.omega, Kuramoto.shift") / "main.nf").read_text()
    assert "channel.of('0.05', '0.06', '0.07').combine(channel.of('0.0', '0.01'))" in nf
    assert "        tuple val(omega), val(shift)\n" in nf
    assert _task_command(nf).rstrip().endswith("-o ../out/omega=${omega}/shift=${shift}")


def test_a_run_with_no_fanned_axis_is_one_task_with_no_empty_section(tmp_path: Path):
    """A group run has no value channel: its one task takes the kit alone and writes straight into its output directory."""
    nf = (_kit(tmp_path) / "main.nf").read_text()
    assert nf[nf.index("input:") : nf.index("output:")].split() == ["input:", "path", "kit,", "stageAs:", "'kit/*'"]
    assert "    tvbo_run(kit)\n" in nf and "cells" not in nf
    assert _task_command(nf).rstrip().endswith("-o ../out")


def test_a_subject_fan_out_passes_each_subject_and_writes_beside_the_others():
    """The dataset fan-out is ``--subject``, not a pinned parameter, and every subject's result lands in one directory under its own ``sub-`` name."""
    axis = SimpleNamespace(name="subject", parameter="dataset.active_subject", values=("01", "02"), kind="subjects")
    plan = SimpleNamespace(
        container=None,
        backend=SimpleNamespace(name="tvboptim"),
        study_key="s",
        experiment_key="30",
        wildcards=["subject"],
        vectorize_axes=[],
        workflow_axes=[axis],
        out_dir="results",
        reference_root="results",
    )
    nf = _render_template(
        "nextflow/main.nf.mako", plan=plan, block={}, spec_relpath="spec/30.yaml", result_stem="exp-30_result"
    )
    command = _task_command(nf)
    assert "--subject=${subject} " in command and "--pin" not in command
    assert command.rstrip().endswith("-o ../out")
    assert "channel.of('01', '02')" in nf


def test_the_pipeline_holds_no_configuration_and_the_kit_carries_it_beside(tmp_path: Path):
    """``process.x = ...`` and a runtime switch are configuration statements, which a pipeline script cannot hold; the kit writes them to ``nextflow.config``."""
    kit = _kit(tmp_path, "", "--set", "nextflow.executor=slurm", "--set", "nextflow.queue=long")
    nf, config = (kit / "main.nf").read_text(), (kit / "nextflow.config").read_text()
    assert not re.search(r"^\s*(process|singularity|docker|executor)\.\w+\s*=", nf, flags=re.M)
    assert "process.executor = 'slurm'" in config and "process.queue = 'long'" in config


def test_shell_text_in_the_task_script_survives_the_scripts_own_interpolation(tmp_path: Path):
    """The task script is a Groovy string that reads ``$`` as its own, so an exported value holding one is escaped and reaches the shell as written."""
    nf = (_kit(tmp_path, "", "--set", "nextflow.env.SCRATCH=$TMPDIR/run") / "main.nf").read_text()
    assert "    export SCRATCH='\\$TMPDIR/run'\n" in nf


@pytest.mark.parametrize("slot", ["mem=16G", "time=02:00:00", "cpus_per_task=4", "venv=/opt/venv"])
def test_an_engine_slot_the_emitter_writes_nowhere_is_refused(tmp_path: Path, slot):
    """A resource or an activation declared for Nextflow and written into no file would leave the tasks running without it; the kit is refused, naming the slot and the passthrough that takes it."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE)
    r = runner.invoke(
        app,
        [
            "workflow",
            "nextflow",
            str(recipe),
            "--backend",
            "tvboptim",
            "-o",
            str(tmp_path / "kit"),
            "--set",
            f"nextflow.{slot}",
        ],
    )
    assert r.exit_code != 0
    assert slot.split("=")[0] in r.output and "options" in r.output
    assert not (tmp_path / "kit" / "main.nf").exists()


@pytest.mark.parametrize("slot", ["mem=16G", "time=02:00:00", "cpus_per_task=4", "venv=/opt/venv"])
def test_a_slurm_resource_does_not_make_a_nextflow_kit_refuse_itself(tmp_path: Path, slot):
    """A study's Slurm block is not the Nextflow block: its resources are not inherited, so they cannot trip the refusal of a Nextflow slot the emitter writes nowhere."""
    out = _kit(tmp_path, "", "--set", f"slurm.{slot}")
    assert (out / "main.nf").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("nextflow") is None or shutil.which("tvbo") is None,
    reason="needs the nextflow launcher and the tvbo command on PATH",
)
def test_nextflow_runs_an_emitted_kit_and_publishes_every_cell(tmp_path: Path):
    """Nextflow itself compiles the pipeline, runs one task per cell and publishes each result where the kit says; a second run with ``-resume`` finds every task cached."""
    kit = _kit(tmp_path, "Kuramoto.omega")
    env = {**os.environ, "NXF_HOME": os.environ.get("NXF_HOME", str(tmp_path / "nxf")), "NXF_DISABLE_CHECK_LATEST": "true"}
    for resume, expected in (((), "completed=3 failed=0 cached=0"), (("-resume",), "completed=0 failed=0 cached=3")):
        done = subprocess.run(
            ["nextflow", "run", "main.nf", "-ansi-log", "false", *resume], cwd=kit, env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert expected in done.stdout
    results = sorted(p.relative_to(kit).as_posix() for p in (kit / "derivatives").rglob("*.h5"))
    assert results == [f"derivatives/tvbo/200/omega={v}/{STEM}.h5" for v in ("0.05", "0.06", "0.07")]
