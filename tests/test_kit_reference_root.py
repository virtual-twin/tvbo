"""Every launcher tells its runs where the containers they read and do not write are.

A kit stages analysis containers and the results of experiments it does not run into its own results tree, and its own experiments write there too. A run looks under its own output directory unless told otherwise, so a launcher that does not pass ``--results-root`` leaves a warm start or a cohort aggregate unfound on the compute node.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tvbo.classes.experiment import SimulationExperiment
from tvbo.cli import app
from tvbo.cli.workflow import _render_template
from tvbo.run import study as _study_run
from tvbo.run import workflow as _workflow

runner = CliRunner()
EXP = "experiment:JR_MEG_FrequencyGradient_Optimization"
KIT_ROOT = "derivatives/tvbo"


def test_a_plan_reads_from_its_own_results_tree_unless_an_emitter_states_another():
    exp = SimulationExperiment.from_db("JR_MEG_FrequencyGradient_Optimization")
    planned = _workflow.plan(study_key="s", experiment=exp, backend="jax")
    assert planned.reference_root == planned.out_dir == KIT_ROOT


@pytest.mark.parametrize(
    ("engine", "artefact", "passed"),
    [
        ("slurm", "run.sbatch", ['RESULTS_ROOT="${TVBO_RESULTS_ROOT:-' + KIT_ROOT + '}"', '--results-root "${RESULTS_ROOT}"']),
        ("snakemake", "Snakefile", [f'"--results-root {KIT_ROOT} "']),
    ],
)
def test_a_kits_launcher_passes_the_kits_results_tree(tmp_path: Path, engine, artefact, passed):
    """An experiment that reads nothing from another run is told too: the rule is the launcher's, not a guess about what a run will read."""
    out = tmp_path / "kit"
    r = runner.invoke(app, ["workflow", engine, EXP, "--backend", "jax", "-o", str(out)])
    assert r.exit_code == 0, r.stdout
    text = (out / artefact).read_text()
    for line in passed:
        assert line in text


@pytest.mark.parametrize(
    ("root", "passed"),
    [("", "--results-root ${baseDir}/" + KIT_ROOT), ("/study/derivatives/tvbo", "--results-root /study/derivatives/tvbo")],
)
def test_a_nextflow_task_reads_relative_to_the_pipeline_and_not_to_its_work_directory(root, passed):
    """A task runs in a work directory of its own, so a kit-relative root is spelled from ``baseDir``; an absolute one is passed as it is."""
    exp = SimulationExperiment.from_db("JR_MEG_FrequencyGradient_Optimization")
    planned = _workflow.plan(study_key="s", experiment=exp, backend="jax", engine="nextflow")
    planned.reference_root = root or planned.reference_root
    nf = _render_template("nextflow/main.nf.mako", plan=planned, block={}, script_relpath=None)
    assert passed + " \\" in nf


@pytest.mark.parametrize(
    ("engine", "passed"),
    [("slurm", 'RESULTS_ROOT="${{TVBO_RESULTS_ROOT:-{root}}}"'), ("snakemake", '"--results-root {root} "')],
)
def test_an_artefact_printed_to_stdout_reads_the_studys_results(tmp_path: Path, monkeypatch, engine, passed):
    """No kit is written, so nothing is staged: the runs read where the study keeps its results."""
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["workflow", engine, EXP, "--backend", "jax", "--stdout"])
    assert r.exit_code == 0, r.stdout
    root = _study_run.results_root(EXP, None)
    assert root.is_absolute()
    assert passed.format(root=root) in r.stdout


def test_staging_for_an_experiment_without_a_network_constructs_none(tmp_path: Path, monkeypatch):
    """Where the study keeps its results follows from the recipe's location alone, so an experiment that has no network needs none built to find the run it warm-starts from."""
    from types import SimpleNamespace

    from tvbo.classes.network import Network
    from tvbo.cli import workflow as kit
    from tvbo.utils.study_layout import study_path

    (tmp_path / "dataset_description.json").write_text('{"Name": "Toy", "BIDSVersion": "1.9.0"}')
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    (results / "exp-4_model-Toy_result.h5").write_bytes(b"group fit")

    def refuse(self, **kwargs):
        raise AssertionError("staging constructed a Network")

    monkeypatch.setattr(Network, "__init__", refuse)
    experiment = SimpleNamespace(id=5, network=None, _source_file=str(tmp_path / "Toy.yaml"))
    staged_dir = tmp_path / "kit" / KIT_ROOT
    assert kit._stage_reference_containers(experiment, staged_dir, depends_on=["4"]) == ["exp-4"]
    assert (staged_dir / "exp-4_model-Toy_result.h5").read_bytes() == b"group fit"
