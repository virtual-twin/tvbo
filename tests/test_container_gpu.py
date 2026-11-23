"""A container kit hands the node's GPU to the experiments that declare one, and the published image carries the JAX that uses it.

An experiment that declares ``execution.accelerator: gpu`` is granted a GPU, but a task inside an Apptainer or Singularity container sees that GPU only when the runtime is called with ``--nv``, and a CPU-only JAX would not use it if it did. Each engine's container call (the Slurm script, the one ``apptainer-args`` of the Snakemake profile, Nextflow's ``singularity.runOptions``) is read from a kit emitted for a two-node experiment, and the image's CUDA JAX from the two files that build it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from typer.testing import CliRunner

from tvbo.cli import app

REPO_ROOT = Path(__file__).parent.parent
IMAGE = "docker://ghcr.io/virtual-twin/tvbo:dev"
ENGINES = ["slurm", "snakemake", "nextflow"]

EXPERIMENT = """id: ID
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5}
    - {source: 1, target: 0, weight: 0.5}
  coupling:
    KuramotoCoupling:
      label: KuramotoCoupling
      parameters:
        a: {value: 0.01}
        N: {value: 1.0}
      pre_expression: {rhs: "sin(theta_j - theta_i)"}
      post_expression: {rhs: "a * gx / N"}
      incoming_states: [theta]
      local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 20.0
  step_size: 1.0
  transient_time: 0.0
execution:
  accelerator: ACCELERATOR
"""


def _experiment_text(id_: int, accelerator: str) -> str:
    """The experiment numbered *id_*, declaring *accelerator*."""
    return EXPERIMENT.replace("ID", str(id_)).replace("ACCELERATOR", accelerator)


def _experiment(tmp_path: Path, accelerator: str) -> Path:
    """A two-node Kuramoto experiment declaring *accelerator*, written as a spec file."""
    spec = tmp_path / "experiment.yaml"
    spec.write_text(_experiment_text(1, accelerator), encoding="utf-8")
    return spec


def _study(tmp_path: Path, *accelerators: str) -> Path:
    """A study holding one such experiment per entry of *accelerators*, numbered from 1 in that order."""
    experiments = "".join(
        "  - " + _experiment_text(i, a).replace("\n", "\n    ").rstrip() + "\n" for i, a in enumerate(accelerators, 1)
    )
    spec = tmp_path / "study.yaml"
    spec.write_text(
        f"tvbo_class: tvbo:SimulationStudy\ncitekey: Gpu\ntitle: GPU kit\nexperiments:\n{experiments}", encoding="utf-8"
    )
    return spec


def _kit(tmp_path: Path, engine: str, spec: Path, *assignments: str) -> Path:
    """Emit *engine*'s kit for *spec* in the tvbo image, with each of *assignments* as a ``--set``."""
    out = tmp_path / engine
    sets = [arg for a in (f"container={IMAGE}", *assignments) for arg in ("--set", a)]
    result = CliRunner().invoke(app, ["workflow", engine, str(spec), "-o", str(out), *sets])
    assert result.exit_code == 0, result.output
    return out


def _container_flags(kit: Path, engine: str) -> str:
    """The flags the kit's container call passes ahead of the image, past the kit directory Nextflow binds of its own."""
    if engine == "slurm":
        call = next(line for line in (kit / "run.sbatch").read_text().splitlines() if "singularity exec" in line)
        return call.split("singularity exec", 1)[1].split(IMAGE, 1)[0].strip()
    if engine == "snakemake":
        return yaml.safe_load((kit / "profile" / "config.yaml").read_text()).get("apptainer-args", "")
    config = (kit / "nextflow.config").read_text()
    return re.search(r'^singularity\.runOptions = "--bind \$\{projectDir\} ?(.*)"$', config, re.MULTILINE).group(1)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_gpu_experiment_runs_its_container_with_the_gpu(tmp_path, engine):
    """``accelerator: gpu`` adds ``--nv`` after the declared binds, so the task's CUDA JAX finds the GPU the node grants."""
    kit = _kit(tmp_path, engine, _experiment(tmp_path, "gpu"), "container_binds=/data/cephfs-1")
    assert _container_flags(kit, engine) == "--bind /data/cephfs-1 --nv"


@pytest.mark.parametrize(
    "gres,flags", [("gpu:a100:1", "--nv"), ("gpu", "--nv"), ("tmpspace:10G,gpu:2", "--nv"), ("tmpspace:10G", "")]
)
@pytest.mark.parametrize("engine", ["slurm", "snakemake"])
def test_a_gpu_gres_runs_its_container_with_the_gpu(tmp_path, engine, gres, flags):
    """A GPU requested through ``gres`` is handed to the container even when the experiment leaves the accelerator to JAX; a resource that is not a GPU adds nothing."""
    kit = _kit(tmp_path, engine, _experiment(tmp_path, "auto"), f"slurm.gres={gres}")
    assert _container_flags(kit, engine) == flags


@pytest.mark.parametrize("accelerator", ["cpu", "auto"])
@pytest.mark.parametrize("engine", ENGINES)
def test_a_cpu_kit_keeps_its_flags(tmp_path, engine, accelerator):
    """A kit that asks for no GPU passes its declared binds and nothing else."""
    kit = _kit(tmp_path, engine, _experiment(tmp_path, accelerator), "container_binds=/data/cephfs-1")
    assert _container_flags(kit, engine) == "--bind /data/cephfs-1"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_nv_is_not_repeated(tmp_path, engine):
    """``container_args`` is passed as written, and an ``--nv`` it already holds is not added a second time."""
    kit = _kit(tmp_path, engine, _experiment(tmp_path, "gpu"), "container_args=--nv --cleanenv")
    assert _container_flags(kit, engine) == "--nv --cleanenv"


def test_one_gpu_experiment_gives_the_whole_snakemake_kit_the_gpu(tmp_path, caplog):
    """Snakemake passes one ``apptainer-args`` to every rule, so a study whose second experiment declares a GPU runs every container with ``--nv``: Apptainer only warns on a node without one, and the first experiment's declared ``cpu`` keeps its JAX on the CPU. Both experiments declare the same binds, so no divergence is reported."""
    with caplog.at_level("WARNING", logger="tvbo.cli"):
        kit = _kit(tmp_path, "snakemake", _study(tmp_path, "cpu", "gpu"), "container_binds=/data/cephfs-1")
    assert _container_flags(kit, "snakemake") == "--bind /data/cephfs-1 --nv"
    assert "--apptainer-args" not in caplog.text


def test_the_cuda_extra_is_jax_with_its_cuda_plugin_on_linux_only():
    """``tvbo[cuda]`` installs JAX's CUDA 12 build where its wheels exist, and ``tvbo[all]`` leaves it out because macOS has none."""
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    (jax,) = map(Requirement, extras["cuda"])
    assert jax.name == "jax" and jax.extras == {"cuda12"}
    assert jax.marker.evaluate({"sys_platform": "linux", "platform_system": "Linux"})
    assert not jax.marker.evaluate({"sys_platform": "darwin", "platform_system": "Darwin"})
    assert "cuda" not in Requirement(extras["all"][0]).extras


def test_the_image_installs_the_cuda_extra():
    """amd64 and arm64 build from the one Dockerfile, so the dependency layer's ``--extra cuda`` reaches both."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    install = next(
        line for line in dockerfile.splitlines() if line.startswith("RUN ") and "uv pip install -r pyproject.toml" in line
    )
    assert re.search(r"--extra cuda(\s|$)", install)
