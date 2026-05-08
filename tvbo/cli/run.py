"""``tvbo run`` — execute a Study or Experiment.

Implements the cardinal HPC contract from §5.1 of ``dev/tvbo-cli.md``:

* ``--engine slurm`` re-emits via :mod:`tvbo.cli.workflow` and submits
  through ``sbatch`` rather than running locally.
* ``--container IMAGE`` re-execs the run inside the named OCI image
  (Singularity if ``SINGULARITY_BIND`` is set in the environment, else
  Docker).
* ``--slurm-chunk i/N`` shards the study's sweep grid across array tasks:
  cell index ``j`` runs iff ``j %% N == i``. This is what the generated
  sbatch script invokes for every array index.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import typer

from . import _common


def run(
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name."),
    backend: str = typer.Option(
        "tvboptim", "--backend", "-b",
        help="Execution backend (tvboptim, tvb, jax, pyrates, networkdynamics, ...).",
    ),
    out_dir: Path = typer.Option(
        None, "--out-dir", "-o", help="Directory to write results into."
    ),
    experiment: str = typer.Option(
        None, "--experiment", help="When SPEC is a Study, run only this named experiment."
    ),
    duration: float = typer.Option(
        None, "--duration", help="Override integration.duration (ms)."
    ),
    engine: str = typer.Option(
        "local", "--engine", "-e",
        help="local | slurm | snakemake | nextflow. Non-local engines re-emit via `tvbo workflow ENGINE` and submit.",
    ),
    container: str = typer.Option(
        None, "--container",
        help="OCI image (e.g. ghcr.io/the-virtual-brain/tvbo:0.7.0); re-execs the same `tvbo run` inside it.",
    ),
    slurm_chunk: str = typer.Option(
        None, "--slurm-chunk",
        help="Shard sweep cells across array tasks: ``i/N`` runs cells where j%N==i.",
    ),
) -> None:
    if engine != "local":
        _dispatch_to_engine(engine, spec=spec, backend=backend,
                            experiment=experiment, container=container, out_dir=out_dir)
        return

    if container and os.environ.get("TVBO_IN_CONTAINER") != "1":
        _reexec_in_container(container, sys.argv[1:])
        return

    kind, obj = _common.resolve_spec(spec)

    kwargs: dict = {}
    if duration is not None:
        kwargs["duration"] = duration

    chunk_i = chunk_n = None
    if slurm_chunk:
        chunk_i, chunk_n = _parse_chunk(slurm_chunk)
        _common.info(f"sharding: cell j runs iff j%{chunk_n}=={chunk_i}")

    if kind == "study":
        exps = obj.experiments if hasattr(obj, "experiments") else obj.simulation_experiments
        items = list(exps.values()) if hasattr(exps, "values") else list(exps)
        if experiment is not None:
            items = [
                e for e in items
                if (getattr(e, "key", None) == experiment
                    or getattr(e, "name", None) == experiment
                    or getattr(e, "label", None) == experiment)
            ]
            if not items:
                _common.die(f"No experiment named {experiment!r} in study.")
        for exp in items:
            _common.info(f"running experiment: {getattr(exp, 'key', None) or getattr(exp, 'label', None)}")
            _run_one(exp, backend, out_dir, kwargs, chunk_i, chunk_n)
        return

    if kind == "experiment":
        _run_one(obj, backend, out_dir, kwargs, chunk_i, chunk_n)
        return

    _common.die(f"`tvbo run` does not yet support kind={kind!r}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_chunk(s: str) -> tuple[int, int]:
    if "/" not in s:
        raise typer.BadParameter("--slurm-chunk must be of the form i/N")
    i_s, n_s = s.split("/", 1)
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise typer.BadParameter(f"--slurm-chunk i={i} out of range [0,{n})")
    return i, n


def _run_one(experiment, backend: str, out_dir: Path | None,
             kwargs: dict, chunk_i: int | None, chunk_n: int | None) -> None:
    if chunk_n is not None:
        from ._workflow import extract_axes
        axes = extract_axes(experiment)
        if not axes:
            _common.info("no sweep axes on experiment; running once")
            _exec_one(experiment, backend, out_dir, kwargs)
            return
        from itertools import product
        grids = [ax.values for ax in axes]
        for j, combo in enumerate(product(*grids)):
            if j % chunk_n != chunk_i:
                continue
            cell_kwargs = dict(kwargs)
            for ax, v in zip(axes, combo):
                _common.info(f"  cell {j}: {ax.parameter}={v}")
            cell_out = out_dir / f"cell_{j:06d}" if out_dir else None
            _exec_one(experiment, backend, cell_out, cell_kwargs)
        return

    _exec_one(experiment, backend, out_dir, kwargs)


def _exec_one(experiment, backend: str, out_dir: Path | None, kwargs: dict) -> None:
    result = experiment.run(format=backend, **kwargs)
    _common.info(f"done: {type(result).__name__}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "save"):
            saved = result.save(str(out_dir))
            _common.info(f"wrote {saved}")
        else:
            _common.info(f"(result has no .save(); skipping write to {out_dir})")


def _dispatch_to_engine(engine: str, *, spec: str, backend: str,
                        experiment: str | None, container: str | None,
                        out_dir: Path | None) -> None:
    if engine not in {"slurm", "snakemake", "nextflow"}:
        _common.die(f"--engine {engine!r} not supported. Use local|slurm|snakemake|nextflow.")

    overrides: list[str] = []
    if container:
        overrides.append(f"--set=container={container}")
    if out_dir:
        overrides.append(f"--set=out_dir={out_dir}")

    artefact = Path(out_dir or ".") / {
        "slurm": "run.sbatch",
        "snakemake": "Snakefile",
        "nextflow": "main.nf",
    }[engine]

    cmd = ["tvbo", "workflow", engine, spec, "--backend", backend, "-o", str(artefact)]
    if experiment:
        cmd.extend(["--experiment", experiment])
    cmd.extend(overrides)
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    _common.info(f"emitted {artefact}")

    if engine == "slurm":
        sub = ["sbatch", str(artefact)]
        _common.info("$ " + " ".join(shlex.quote(c) for c in sub))
        subprocess.run(sub, check=True)


def _reexec_in_container(image: str, argv: list[str]) -> None:
    """Re-exec ``tvbo`` inside *image* via Singularity (preferred) or Docker."""
    use_singularity = bool(os.environ.get("SINGULARITY_BIND")) or _which("singularity")
    cwd = os.getcwd()
    if use_singularity:
        cmd = ["singularity", "exec", "--bind", f"{cwd}:{cwd}", image, "tvbo", *argv]
    else:
        cmd = ["docker", "run", "--rm",
               "-e", "TVBO_IN_CONTAINER=1",
               "-v", f"{cwd}:{cwd}",
               "-w", cwd,
               image, "tvbo", *argv]
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    raise SystemExit(subprocess.run(cmd).returncode)


def _which(prog: str) -> str | None:
    from shutil import which
    return which(prog)
