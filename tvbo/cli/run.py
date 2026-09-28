"""``tvbo run`` — execute a Study or Experiment.

Implements the cardinal HPC contract:

* ``--engine slurm`` re-emits via :mod:`tvbo.cli.workflow` and submits through ``sbatch`` rather than running locally.
* ``--container IMAGE`` re-execs the run inside the named OCI image (Singularity if ``SINGULARITY_BIND`` is set in the environment, else Docker).
* ``--shard i/N`` runs one shard of the sweep in-process (no scheduler): cell index ``j`` runs iff ``j %% N == i``. This is what the generated sbatch script invokes for every array index.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from tvbo.run import study as _study_run

from . import _common


def run(
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name."),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend (tvboptim, tvb, jax, brian2, pyrates, networkdynamics, ...). "
        "Default: each experiment's declared execution.backend, else tvboptim.",
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Directory to write results into. With a non-local --engine it is also where the workflow kit is written.",
    ),
    results_root: Path = typer.Option(
        None,
        "--results-root",
        help="Directory searched for a sibling run's saved result when this experiment's "
        "initial_state.method=from_experiment (state / parameter warm-start). Defaults "
        "to the output dir's parent; set it to point at another run's output — e.g. the "
        "group fit's results dir for a per-subject warm-start (Run A → Run B).",
    ),
    experiment: str = typer.Option(None, "--experiment", help="When SPEC is a Study, run only this named experiment."),
    analysis: str = typer.Option(
        None,
        "--analysis",
        help="When SPEC is a Study, run only these named `analyses:` (comma-separated) and "
        "no experiments — for re-deriving a container after editing its callable, "
        "which no cache invalidates on its own. An input analysis is re-run only when "
        "it has no container yet; existing ones are read as they are. Figures are not "
        "redrawn — follow with `tvbo figure render`. Local engine only, and not "
        "combinable with any flag that selects or reshapes simulation work.",
    ),
    duration: float = typer.Option(None, "--duration", help="Override integration.duration (ms)."),
    engine: str = typer.Option(
        "local",
        "--engine",
        "-e",
        help="local | slurm | snakemake | nextflow. Non-local engines re-emit via `tvbo workflow ENGINE` and submit.",
    ),
    container: str = typer.Option(
        None,
        "--container",
        help="OCI image (e.g. ghcr.io/the-virtual-brain/tvbo:0.7.0); re-execs the same `tvbo run` inside it.",
    ),
    shard: str = typer.Option(
        None,
        "--shard",
        "--slurm-chunk",
        help="Run one shard of the sweep in-process: ``i/N`` runs cells where j%N==i "
        "(no scheduler needed). ``--slurm-chunk`` is a deprecated alias.",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        min=1,
        help="Run at most N cells of the sweep (a spread sample) — a quick look "
        "without needing to know the grid size. Ignored when --shard is given.",
    ),
    subject: str = typer.Option(
        None,
        "--subject",
        help="Active subject ID for a per-subject dataset experiment: resolves and "
        "injects that subject's empirical target (e.g. their FC). Set per shard "
        "by the workflow fan-out.",
    ),
    rendered: Path = typer.Option(
        None,
        "--rendered",
        help="Run a PRE-RENDERED backend script instead of generating code at run time. "
        "The spec is still loaded for orchestration (subject/dataset resolution, "
        "seeds, network observations, output layout) — only the backend code source "
        "changes: the frozen script is executed as-is. Lets a workflow kit run on a "
        "stock tvbo runtime with no codegen step. The script must match the run's "
        "backend and the (single) experiment being run.",
    ),
    set_: list[str] = typer.Option(
        [],
        "--set",
        help="Override an experiment metadata field for THIS run only (the recipe file is "
        "not modified), e.g. --set integration.duration=8 --set integration.step_size=0.05. "
        "Repeatable; dotted keys traverse attributes and keyed collections. Lets one "
        "recipe stay the single source of truth while the CLI runs it with test settings.",
    ),
    pin: list[str] = typer.Option(
        [],
        "--pin",
        help="Pin an exploration axis to a single value for THIS run, e.g. "
        "--pin Kuramoto.omega_mean_hz=20 --pin network.conduction_speed=6. The workflow "
        "fan-out emits one --pin per fanned axis per cell: it sets the axis's parameter "
        "AND drops the axis from the sweep, so the cell is a single run at that point (its "
        "base run — and every declared observation — computed there). The model-scope "
        "sibling of --subject. Repeatable.",
    ),
    compress: bool = typer.Option(
        True,
        "--compress/--no-compress",
        help="gzip-deflate the result HDF5 (default on; grids compress well). "
        "--no-compress writes uncompressed for maximum write speed.",
    ),
    save_all: bool = typer.Option(
        False,
        "--save-all",
        help="Persist every observation, including intermediates. By default only "
        "recorded outputs are saved (leaves + `record: true`); this keeps the "
        "scaffolding (e.g. a raw BOLD feeding an FC) for debugging.",
    ),
    max_iterations: int = typer.Option(
        None,
        "--max-iterations",
        min=1,
        help="Smoke cap: run at most N tuning iterations per algorithm AND per stage for "
        "THIS run (the recipe is untouched). A fit's post-tuning evaluation — the "
        "memory- and time-critical part of a long-horizon fit — is independent of how "
        "many tuning iterations preceded it, so `--max-iterations 1` reaches it in "
        "minutes to verify it runs/streams within memory.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Shorthand for --max-iterations 1: the quickest run that still reaches the "
        "post-tuning evaluation (verify a fit executes / streams end to end).",
    ),
    figures: bool = typer.Option(
        True,
        "--figures/--no-figures",
        help="After a Study's experiments finish, render its declarative `figures:` "
        "(emit each render script and run it), so one command produces results AND "
        "figures. On by default; --no-figures skips rendering (e.g. a partial/smoke "
        "run whose panels would be placeholders). No effect on an experiment spec or "
        "a study without figures.",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="When SPEC nests studies, skip these sub-studies by label, at any depth, "
        "comma-separated/repeatable — their committed figures/results are reused as-is. "
        "Skipping a study skips the studies it holds.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="When SPEC nests studies, list the studies, analyses and result keys that "
        "WOULD run (honouring --skip) and emit nothing.",
    ),
    manifest_only: bool = typer.Option(
        False,
        "--manifest-only",
        help="When SPEC nests studies, emit the results manifest from existing containers "
        "and authored values only — run no study or experiment. The fast refresh for the "
        "two-tier build (the heavy `tvbo run` produces the containers; this restamps the "
        "manifest the manuscript reads).",
    ),
) -> None:
    """Run a SPEC (experiment or study) in the selected backend.

    Resolves *spec* to a `SimulationExperiment` or `SimulationStudy`, executes via *backend* on *engine*, and optionally writes results to `--out-dir`.
    Non-local engines re-emit the run through `tvbo workflow ENGINE` and submit. The run itself is :mod:`tvbo.run.study`; this command turns the flags into its options and its refusals into exit codes.
    """
    if engine != "local":
        if analysis is not None:
            _common.die(
                f"--analysis is a local-only mode: the workflow kit fans out experiments, "
                f"so --engine {engine} would run the WHOLE study instead of the named "
                f"analyses. Re-derive them locally with `tvbo run {spec} --analysis "
                f"{analysis}`."
            )
        _dispatch_to_engine(engine, spec=spec, backend=backend, experiment=experiment, container=container, out_dir=out_dir)
        return

    if container and os.environ.get("TVBO_IN_CONTAINER") != "1":
        _reexec_in_container(container, sys.argv[1:])
        return

    kind, obj = _common.resolve_spec(spec)

    # Flags that select or reshape SIMULATION work.
    _sim_flags = [
        f
        for f, v in (
            ("--experiment", experiment),
            ("--shard", shard),
            ("--rendered", rendered),
            ("--limit", limit),
            ("--subject", subject),
            ("--duration", duration),
            ("--max-iterations", max_iterations),
            ("--smoke", smoke or None),
            ("--set", set_ or None),
            ("--pin", pin or None),
        )
        if v is not None
    ]

    if _study_run.has_nested(obj) and analysis is None:
        # A study-of-studies runs every nested study end to end with the recipe's own settings, so any of these would be silently dropped — turning a one-container request into the whole tree, or reporting success for a --save-all that saved record-only.
        rejected = _sim_flags + [
            f
            for f, v in (
                ("--save-all", save_all or None),
                ("--no-compress", None if compress else True),
                ("--results-root", results_root),
            )
            if v is not None
        ]
        if rejected:
            _common.die(
                f"{spec} nests studies, so it runs every one of them end to end; "
                f"{', '.join(rejected)} would be ignored. Point the flag at the "
                "study that declares the work."
            )
        with _common.fatal():
            _study_run.run_tree(
                obj,
                spec,
                out_dir,
                backend=backend,
                figures=figures,
                skip=skip,
                dry_run=dry_run,
                manifest_only=manifest_only,
            )
        return

    if analysis is not None:
        # Ignoring one would exit 0 having simulated nothing, which a cluster reads as success.
        if _sim_flags:
            _common.die(
                f"--analysis runs no experiments, so {', '.join(_sim_flags)} would be ignored. Run them as a separate command."
            )
        if kind != "study":
            _common.die(
                f"--analysis needs a study: {spec} resolves to a {kind}, which "
                f"declares no `analyses:`. Point it at the study that does."
            )
        with _common.fatal():
            _study_run.run_named_analyses(obj, spec, analysis, out_dir)
        return

    options = _run_options(
        backend=backend,
        set_=set_,
        pin=pin,
        max_iterations=max_iterations,
        smoke=smoke,
        duration=duration,
        subject=subject,
        results_root=results_root,
        rendered=rendered,
        limit=limit,
        shard=shard,
        compress=compress,
        save_all=save_all,
    )
    with _common.fatal():
        if kind == "study":
            _study_run.run_study(obj, spec, out_dir, options=options, experiment=experiment, figures=figures)
            return
        if kind == "experiment":
            _study_run.run_experiment(obj, spec, out_dir, options=options)
            return

    _common.die(f"`tvbo run` does not yet support kind={kind!r}.")


# Helpers


def _run_options(
    *,
    backend: str | None,
    set_: list[str],
    pin: list[str],
    max_iterations: int | None,
    smoke: bool,
    duration: float | None,
    subject: str | None,
    results_root: Path | None,
    rendered: Path | None,
    limit: int | None,
    shard: str | None,
    compress: bool,
    save_all: bool,
) -> _study_run.RunOptions:
    """The flags that shape each experiment's run, as the library's :class:`~tvbo.run.study.RunOptions`.

    A malformed flag is a usage error here, before any experiment starts. ``--smoke`` is shorthand for ``--max-iterations 1``, and an explicit ``--max-iterations`` wins. The ``--rendered`` script is read once, here, and handed to every experiment run as the code it executes in place of generated code.
    """
    rendered_code = None
    if rendered is not None:
        try:
            rendered_code = Path(rendered).read_text(encoding="utf-8")
        except OSError as e:
            _common.die(f"--rendered: cannot read {rendered}: {e}")
    chunk = None
    if shard:
        chunk = _parse_chunk(shard)
        _common.info(f"sharding: cell j runs iff j%{chunk[1]}=={chunk[0]}")
    return _study_run.RunOptions(
        backend=backend,
        overrides=_assignments(set_, "--set", "path=value"),
        pins=_assignments(pin, "--pin", "parameter=value"),
        max_iterations=max_iterations if max_iterations is not None else (1 if smoke else None),
        duration=duration,
        subject=subject,
        results_root=results_root,
        rendered_code=rendered_code,
        limit=limit,
        shard=chunk,
        compress=compress,
        record_only=not save_all,
    )


def _assignments(items: list[str], flag: str, form: str) -> tuple[tuple[str, Any], ...]:
    """Each ``KEY=VALUE`` of a repeatable *flag* as a ``(key, value)`` pair, the value coerced to the type it spells."""
    pairs = (_common.parse_assignment(raw, flag, form) for raw in items)
    return tuple((key, _common.coerce_value(value)) for key, value in pairs)


def _parse_chunk(s: str) -> tuple[int, int]:
    if "/" not in s:
        raise typer.BadParameter("--shard must be of the form i/N")
    i_s, n_s = s.split("/", 1)
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise typer.BadParameter(f"--shard i={i} out of range [0,{n})")
    return i, n


def _dispatch_to_engine(
    engine: str, *, spec: str, backend: str, experiment: str | None, container: str | None, out_dir: Path | None
) -> None:
    """Emit a workflow kit for *engine* and submit/execute it, all in-process.

    Shares the emit + execute path with ``tvbo workflow <engine>`` rather than re-shelling ``tvbo`` (which needs it on ``$PATH`` — fragile under venv / module / container setups on HPC) and rather than building the plan twice.

    The flags reach the plan as the ``--set`` assignments ``tvbo workflow`` takes: *container* is the image every task runs in, and *out_dir* is the results root, resolved against the directory the command runs in exactly as ``-o`` is for a local run. *out_dir* is also where the kit itself is written.
    """
    from . import workflow as _workflow_cmd

    if engine not in _workflow_cmd._ARTEFACT_NAME:
        supported = "|".join(_workflow_cmd._ARTEFACT_NAME)
        _common.die(f"--engine {engine!r} not supported. Use local|{supported}.")

    overrides: list[str] = []
    if container:
        overrides.append(f"container={container}")
    if out_dir:
        # Absolute, because the tasks run from the kit directory rather than from here.
        overrides.append(f"out_dir={Path(out_dir).resolve()}")

    kit_dir = _workflow_cmd._emit(
        engine, spec=spec, backend=backend, experiment=experiment, output=out_dir, override=overrides, stdout=False
    )
    if kit_dir is None:
        _common.die("failed to emit workflow kit")
    _workflow_cmd._execute_emitted(engine, kit_dir)


def _reexec_in_container(image: str, argv: list[str]) -> None:
    """Re-exec ``tvbo`` inside *image* via Singularity (preferred) or Docker.

    *image* is resolved the way a workflow kit resolves its ``container`` (:func:`tvbo.cli._workflow.resolve_container_ref`), so ``tvbo`` and a bare ``ghcr.io/…:tag`` name the same image on both paths. Docker takes the reference without its ``docker://`` transport and cannot run a local ``.sif`` or a non-Docker transport at all, which is refused by name. Both paths set ``TVBO_IN_CONTAINER=1`` so the inner run does not re-exec again.
    """
    from ._workflow import resolve_container_ref

    ref = resolve_container_ref(image)
    use_singularity = bool(os.environ.get("SINGULARITY_BIND")) or shutil.which("singularity")
    cwd = os.getcwd()
    if use_singularity:
        cmd = ["singularity", "exec", "--env", "TVBO_IN_CONTAINER=1", "--bind", f"{cwd}:{cwd}", ref, "tvbo", *argv]
    elif ref.startswith("docker://"):
        ref = ref.removeprefix("docker://")
        cmd = ["docker", "run", "--rm", "-e", "TVBO_IN_CONTAINER=1", "-v", f"{cwd}:{cwd}", "-w", cwd, ref, "tvbo", *argv]
    else:
        _common.die(f"--container {image}: Docker runs registry images only, and singularity is not on PATH to run {ref}.")
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    raise SystemExit(subprocess.run(cmd).returncode)
