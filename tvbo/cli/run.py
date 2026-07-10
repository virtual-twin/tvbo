"""``tvbo run`` — execute a Study or Experiment.

Implements the cardinal HPC contract from §5.1 of ``dev/tvbo-cli.md``:

* ``--engine slurm`` re-emits via :mod:`tvbo.cli.workflow` and submits
  through ``sbatch`` rather than running locally.
* ``--container IMAGE`` re-execs the run inside the named OCI image
  (Singularity if ``SINGULARITY_BIND`` is set in the environment, else
  Docker).
* ``--shard i/N`` runs one shard of the sweep in-process (no scheduler):
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
    shard: str = typer.Option(
        None, "--shard", "--slurm-chunk",
        help="Run one shard of the sweep in-process: ``i/N`` runs cells where j%N==i "
             "(no scheduler needed). ``--slurm-chunk`` is a deprecated alias.",
    ),
) -> None:
    """Run a SPEC (experiment or study) in the selected backend.

    Resolves *spec* to a `SimulationExperiment` or `SimulationStudy`, executes
    via *backend* on *engine*, and optionally writes results to `--out-dir`.
    Non-local engines re-emit the run through `tvbo workflow ENGINE` and submit.
    """
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
    if shard:
        chunk_i, chunk_n = _parse_chunk(shard)
        _common.info(f"sharding: cell j runs iff j%{chunk_n}=={chunk_i}")

    if kind == "study":
        exps = obj.experiments if hasattr(obj, "experiments") else obj.simulation_experiments
        items = list(exps.values()) if hasattr(exps, "values") else list(exps)
        if experiment is not None:
            # Accept one id/name or a comma-separated list ("2,3,20,30" runs all).
            wanted = {s.strip() for s in str(experiment).split(",") if s.strip()}
            items = [e for e in items if wanted & _common.experiment_ids(e)]
            if not items:
                _common.die(f"No experiment(s) matching {experiment!r} in study.")
        for exp in items:
            _common.info(f"running experiment: {getattr(exp, 'key', None) or getattr(exp, 'label', None)}")
            # Resolve to the runtime experiment (has .run) rather than the datamodel object.
            if not hasattr(exp, "run") and hasattr(obj, "get_experiment"):
                sel = (getattr(exp, "id", None) or getattr(exp, "key", None)
                       or getattr(exp, "name", None) or getattr(exp, "label", None))
                try:
                    exp = obj.get_experiment(sel)
                except Exception as e:
                    _common.die(
                        f"Could not resolve experiment {sel!r} to a runnable object: {e}\n"
                        "If the recipe references custom builder/analysis modules "
                        "(e.g. `module: my_networks`), make them importable — run from "
                        "their directory or set PYTHONPATH."
                    )
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
        raise typer.BadParameter("--shard must be of the form i/N")
    i_s, n_s = s.split("/", 1)
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise typer.BadParameter(f"--shard i={i} out of range [0,{n})")
    return i, n


def _run_one(experiment, backend: str, out_dir: Path | None,
             kwargs: dict, chunk_i: int | None, chunk_n: int | None) -> None:
    if chunk_n is not None:
        from ._backends import resolve_backend
        from ._workflow import extract_axes
        axes = extract_axes(experiment)
        if not axes:
            _common.info("no sweep axes on experiment; running once")
            _exec_one(experiment, backend, out_dir, kwargs)
            return
        # A sweep is shardable in-process only where the backend vectorises every
        # swept axis — the same ontology capability the planner uses to decide
        # vectorised-vs-fanned (``BackendSpec.can_vectorize``). Slicing the shard
        # then just indexes that vectorised batch (tvboptim: ``Space[i::N]``).
        # Axes the backend cannot vectorise have no in-process batch to slice;
        # they are fanned into per-cell tasks at the workflow layer instead.
        try:
            spec = resolve_backend(backend)
        except KeyError:
            spec = None
        fanned = [ax for ax in axes if spec is None or not spec.can_vectorize(ax.kind)]
        if fanned:
            names = ", ".join(f"{ax.parameter} ({ax.kind})" for ax in fanned)
            _common.die(
                f"--shard shards a sweep by slicing the backend's vectorised "
                f"batch, but backend {backend!r} does not vectorise: {names}. Emit "
                f"the kit with snakemake/nextflow (which fan these axes into per-cell "
                f"tasks), or use a backend that vectorises them (e.g. tvboptim)."
            )
        n_cells = 1
        for ax in axes:
            n_cells *= len(ax.values)
        per_task = -(-n_cells // chunk_n)  # ceil
        _common.info(
            f"sharding: task {chunk_i}/{chunk_n} runs ~{per_task} of {n_cells} "
            f"sweep cells (backend-vectorised, Space[{chunk_i}::{chunk_n}])"
        )
        shard_kwargs = dict(kwargs)
        shard_kwargs["shard"] = (chunk_i, chunk_n)
        _exec_one(experiment, backend, out_dir, shard_kwargs)
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
    """Emit a workflow kit for *engine* and submit/execute it, all in-process.

    Shares the emit + execute path with ``tvbo workflow <engine>`` rather than
    re-shelling ``tvbo`` (which needs it on ``$PATH`` — fragile under venv /
    module / container setups on HPC) and rather than building the plan twice.
    """
    from . import workflow as _workflow_cmd

    if engine not in _workflow_cmd._ARTEFACT_NAME:
        supported = "|".join(_workflow_cmd._ARTEFACT_NAME)
        _common.die(f"--engine {engine!r} not supported. Use local|{supported}.")

    overrides: list[str] = []
    if container:
        overrides.append(f"--set=container={container}")
    if out_dir:
        overrides.append(f"--set=out_dir={out_dir}")

    kit_dir = _workflow_cmd._emit(engine, spec=spec, backend=backend,
                                  experiment=experiment, output=out_dir,
                                  override=overrides, stdout=False)
    if kit_dir is None:
        _common.die("failed to emit workflow kit")
    _workflow_cmd._execute_emitted(engine, kit_dir)


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
