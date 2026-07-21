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
        None, "--backend", "-b",
        help="Execution backend (tvboptim, tvb, jax, brian2, pyrates, networkdynamics, ...). "
             "Default: each experiment's declared execution.backend, else tvboptim.",
    ),
    out_dir: Path = typer.Option(
        None, "--out-dir", "-o", help="Directory to write results into."
    ),
    results_root: Path = typer.Option(
        None, "--results-root",
        help="Directory searched for a sibling run's saved result when this experiment's "
             "initial_state.method=from_experiment (state / parameter warm-start). Defaults "
             "to the output dir's parent; set it to point at another run's output — e.g. the "
             "group fit's results dir for a per-subject warm-start (Run A → Run B).",
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
    limit: int = typer.Option(
        None, "--limit", min=1,
        help="Run at most N cells of the sweep (a spread sample) — a quick look "
             "without needing to know the grid size. Ignored when --shard is given.",
    ),
    subject: str = typer.Option(
        None, "--subject",
        help="Active subject ID for a per-subject dataset experiment: resolves and "
             "injects that subject's empirical target (e.g. their FC). Set per shard "
             "by the workflow fan-out.",
    ),
    set_: list[str] = typer.Option(
        [], "--set",
        help="Override an experiment metadata field for THIS run only (the recipe file is "
             "not modified), e.g. --set integration.duration=8 --set integration.step_size=0.05. "
             "Repeatable; dotted keys traverse attributes and keyed collections. Lets one "
             "recipe stay the single source of truth while the CLI runs it with test settings.",
    ),
    pin: list[str] = typer.Option(
        [], "--pin",
        help="Pin an exploration axis to a single value for THIS run, e.g. "
             "--pin Kuramoto.omega_mean_hz=20 --pin network.conduction_speed=6. The workflow "
             "fan-out emits one --pin per fanned axis per cell: it sets the axis's parameter "
             "AND drops the axis from the sweep, so the cell is a single run at that point (its "
             "base run — and every declared observation — computed there). The model-scope "
             "sibling of --subject. Repeatable.",
    ),
    compress: bool = typer.Option(
        True, "--compress/--no-compress",
        help="gzip-deflate the result HDF5 (default on; grids compress well). "
             "--no-compress writes uncompressed for maximum write speed.",
    ),
    save_all: bool = typer.Option(
        False, "--save-all",
        help="Persist every observation, including intermediates. By default only "
             "recorded outputs are saved (leaves + `record: true`); this keeps the "
             "scaffolding (e.g. a raw BOLD feeding an FC) for debugging.",
    ),
    max_iterations: int = typer.Option(
        None, "--max-iterations", min=1,
        help="Smoke cap: run at most N tuning iterations per algorithm AND per stage for "
             "THIS run (the recipe is untouched). A fit's post-tuning evaluation — the "
             "memory- and time-critical part of a long-horizon fit — is independent of how "
             "many tuning iterations preceded it, so `--max-iterations 1` reaches it in "
             "minutes to verify it runs/streams within memory.",
    ),
    smoke: bool = typer.Option(
        False, "--smoke",
        help="Shorthand for --max-iterations 1: the quickest run that still reaches the "
             "post-tuning evaluation (verify a fit executes / streams end to end).",
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

    # --smoke is shorthand for --max-iterations 1; an explicit --max-iterations wins.
    eff_max_iterations = max_iterations if max_iterations is not None else (1 if smoke else None)

    kwargs: dict = {}
    if duration is not None:
        kwargs["duration"] = duration
    if subject is not None:
        kwargs["active_subject"] = subject
    kwargs["compress"] = compress
    kwargs["record_only"] = not save_all
    if results_root is not None:
        kwargs["results_root"] = str(results_root)

    chunk_i = chunk_n = None
    if shard:
        chunk_i, chunk_n = _parse_chunk(shard)
        _common.info(f"sharding: cell j runs iff j%{chunk_n}=={chunk_i}")
    # --limit is a cell budget; it becomes a per-experiment shard in _run_one,
    # which knows each experiment's grid size (a study's experiments can differ).

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
            _apply_metadata_overrides(exp, set_)
            _apply_axis_pins(exp, pin)
            _apply_max_iterations(exp, eff_max_iterations)
            _run_one(exp, _effective_backend(exp, backend), out_dir, kwargs, chunk_i, chunk_n, limit)
        return

    if kind == "experiment":
        _apply_metadata_overrides(obj, set_)
        _apply_axis_pins(obj, pin)
        _apply_max_iterations(obj, eff_max_iterations)
        _run_one(obj, _effective_backend(obj, backend), out_dir, kwargs, chunk_i, chunk_n, limit)
        return

    _common.die(f"`tvbo run` does not yet support kind={kind!r}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_backend(experiment, cli_backend: str | None) -> str:
    """Resolve which backend runs *experiment*.

    An explicit ``--backend`` wins for the whole run; otherwise each experiment
    self-selects via its declared ``execution.backend`` (e.g. a spiking network
    sets ``brian2``), falling back to ``tvboptim``. This lets one study mix a
    mean-field sweep and a spiking column and run each on the right engine.
    """
    if cli_backend:
        return cli_backend
    return getattr(getattr(experiment, "execution", None), "backend", None) or "tvboptim"

def _parse_chunk(s: str) -> tuple[int, int]:
    if "/" not in s:
        raise typer.BadParameter("--shard must be of the form i/N")
    i_s, n_s = s.split("/", 1)
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise typer.BadParameter(f"--shard i={i} out of range [0,{n})")
    return i, n


def _coerce_scalar(v: str):
    """Coerce a ``--set`` value string to bool/int/float/JSON, else leave a string."""
    low = v.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v[:1] in "[{":               # JSON list/object, e.g. [0,2] or ["xi","freq"]
        import json
        try:
            return json.loads(v)
        except ValueError:
            pass
    return v


def _apply_metadata_overrides(experiment, overrides: list[str]) -> None:
    """Apply ``--set dotted.path=value`` overrides to a resolved experiment in place.

    Traverses attributes and keyed collections (LinkML keyed dicts) so one recipe can
    stay the single source of truth while a run uses test settings. Mutates the loaded
    object only — the recipe file is untouched.
    """
    def _step(cur, seg):
        if isinstance(cur, dict) and seg in cur:
            return cur[seg]
        if hasattr(cur, seg):
            return getattr(cur, seg)
        try:                        # LinkML keyed collection (dict-like __getitem__)
            return cur[seg]
        except Exception:
            _common.die(f"--set: cannot resolve {seg!r} on {type(cur).__name__}")

    for raw in overrides:
        s = raw.lstrip("-")
        if "=" not in s:
            raise typer.BadParameter(f"--set {raw!r} must be of the form path=value")
        path, _, val = s.partition("=")
        segs = [p for p in path.split(".") if p]
        cur = experiment
        for seg in segs[:-1]:
            cur = _step(cur, seg)
        leaf, value = segs[-1], _coerce_scalar(val)
        if isinstance(cur, dict):
            cur[leaf] = value
        elif hasattr(cur, leaf):
            setattr(cur, leaf, value)
        else:
            try:
                cur[leaf] = value
            except Exception:
                setattr(cur, leaf, value)
        _common.info(f"--set {path} = {value!r}")


def _apply_max_iterations(experiment, n: int) -> None:
    """Cap every algorithm's and stage's ``n_iterations`` (and any optimization's
    ``max_iterations``) to *n* for THIS run — a smoke override, the recipe untouched.

    The post-tuning evaluation of a fit — the memory- and time-critical part of a
    long-horizon run — is independent of how many tuning iterations preceded it, so a
    handful of iterations is enough to verify the fit executes and its long-horizon
    post-tuning observables stream within memory. Mirrors ``--set``: it mutates only the
    loaded object, so one recipe stays the single source of truth.
    """
    if n is None:
        return
    _capped = 0

    def _cap(holder):
        nonlocal _capped
        if isinstance(holder, dict):
            cur = holder.get("n_iterations")
            if isinstance(cur, int) and cur > n:
                holder["n_iterations"] = n
                _capped += 1
        else:
            cur = getattr(holder, "n_iterations", None)
            if isinstance(cur, int) and cur > n:
                setattr(holder, "n_iterations", n)
                _capped += 1

    algos = getattr(experiment, "algorithms", None) or {}
    for algo in (algos.values() if hasattr(algos, "values") else algos):
        _cap(algo)
        for stage in (getattr(algo, "stages", None) or []):
            _cap(stage)

    opts = getattr(experiment, "optimizations", None) or {}
    for opt in (opts.values() if hasattr(opts, "values") else opts):
        cur = getattr(opt, "max_iterations", None)
        if isinstance(cur, int) and cur > n:
            setattr(opt, "max_iterations", n)
            _capped += 1

    _common.info(f"--max-iterations {n}: capped {_capped} iteration count(s)")


def _apply_axis_pins(experiment, pins: list[str]) -> None:
    """Pin fanned exploration axes to single values for THIS run — the workflow fan-out's
    per-cell restriction (the model-scope sibling of ``--subject``).

    For each ``parameter=value``: set the axis's parameter on the experiment so the base
    (representative) run uses it — every DECLARED observation, host or not, is computed on
    that run, so this is what makes a fanned cell's host observation land at the cell's
    coordinates — AND drop that axis from every exploration so the sweep does not re-expand
    it. An exploration left with no axes is removed, collapsing the run to a single point.
    """
    for raw in pins:
        s = raw.lstrip("-")
        if "=" not in s:
            raise typer.BadParameter(f"--pin {raw!r} must be of the form parameter=value")
        parameter, _, val = s.partition("=")
        value = _coerce_scalar(val)
        _set_axis_parameter(experiment, parameter, value)
        _drop_exploration_axis(experiment, parameter)
        _common.info(f"--pin {parameter} = {value!r}")


def _set_axis_parameter(experiment, parameter: str, value) -> None:
    """Write an exploration axis's value onto its parameter target on the experiment.

    Mirrors the codegen axis classifier (tvbo-tvboptim-experiment.py.mako): ``network.<p>``
    is a network scalar; ``<coupling-name>.<p>`` is a coupling parameter; anything else
    ``<x>.<p>`` (or a bare ``<p>``) is a dynamics parameter; and an experiment-scoped path
    (``execution.random_seed``, ``integration.<p>``) falls back to the ``--set`` attribute
    walk, which resolves those correctly. Kept in step with that classifier so a pinned run
    and the swept grid write the same target.
    """
    def _set_in(coll, name) -> bool:
        if coll is None:
            return False
        try:
            entry = coll[name] if name in coll else None
        except TypeError:
            entry = getattr(coll, name, None)
        if entry is None:
            return False
        setattr(entry, "value", value)
        return True

    if parameter.startswith("network."):
        leaf = parameter[len("network."):]
        net = getattr(experiment, "network", None)
        if net is not None and _set_in(getattr(net, "parameters", None), leaf):
            return
        _common.die(f"--pin: cannot resolve network parameter {leaf!r} on the network.")
    if "." in parameter:
        prefix, name = parameter.rsplit(".", 1)
        cpl = getattr(experiment, "coupling", None)
        if cpl is not None and getattr(cpl, "name", None) == prefix \
                and _set_in(getattr(cpl, "parameters", None), name):
            return
        net = getattr(experiment, "network", None)
        net_cpl = getattr(net, "coupling", None) if net is not None else None
        entry = net_cpl.get(prefix) if hasattr(net_cpl, "get") else None
        if entry is not None and _set_in(getattr(entry, "parameters", None), name):
            return
        dyn = getattr(experiment, "dynamics", None)
        if dyn is not None and _set_in(getattr(dyn, "parameters", None), name):
            return
        # Experiment-scoped axis (execution.random_seed, integration.<p>, initial_conditions):
        # its parameter path IS an attribute path, so the --set walk resolves it.
        _apply_metadata_overrides(experiment, [f"{parameter}={value}"])
        return
    dyn = getattr(experiment, "dynamics", None)
    if dyn is not None and _set_in(getattr(dyn, "parameters", None), parameter):
        return
    _common.die(f"--pin: cannot resolve axis parameter {parameter!r}.")


def _drop_exploration_axis(experiment, parameter: str) -> None:
    """Remove the axis with this ``parameter`` from every exploration; drop an exploration
    left with no axes so a fully-pinned run collapses to a single point (no empty sweep)."""
    explorations = getattr(experiment, "explorations", None) or {}
    expl_items = list(explorations.items()) if hasattr(explorations, "items") \
        else list(enumerate(list(explorations)))
    emptied = []
    for key, expl in expl_items:
        space = getattr(expl, "space", None)
        if not space:
            continue
        if hasattr(space, "items"):        # keyed by parameter (LinkML keyed collection)
            for axk in list(space.keys()):
                if str(getattr(space[axk], "parameter", axk)) == parameter:
                    del space[axk]
            if len(space) == 0:
                emptied.append(key)
        else:                              # plain list of axes
            expl.space = [ax for ax in space if str(getattr(ax, "parameter", None)) != parameter]
            if len(expl.space) == 0:
                emptied.append(key)
    # Reverse order so deleting by positional index from a list-form explorations does not
    # shift later indices (dict keys are order-independent).
    for key in reversed(emptied):
        try:
            del explorations[key]
        except Exception:
            pass


def _run_one(experiment, backend: str, out_dir: Path | None,
             kwargs: dict, chunk_i: int | None, chunk_n: int | None,
             limit: int | None = None) -> None:
    # --limit N is a cell budget: turn it into a stride over this experiment's own
    # grid so ``Space[0::stride]`` yields ~N spread cells — no need to know the size.
    if limit is not None and chunk_n is None:
        import math

        from ._workflow import extract_axes
        n_cells = 1
        for ax in extract_axes(experiment):
            n_cells *= len(ax.values)
        if n_cells > limit:
            chunk_i, chunk_n = 0, math.ceil(n_cells / limit)
            _common.info(f"--limit {limit}: running ~{-(-n_cells // chunk_n)} of "
                         f"{n_cells} cells (Space[0::{chunk_n}])")

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
        if any(getattr(ax, "runtime_sized", False) for ax in axes):
            # Branch-restart sweep: the cell count comes from the source run's recorded
            # branch (read at run time), so this task just slices its share of it.
            _common.info(
                f"sharding: task {chunk_i}/{chunk_n} runs its slice of a runtime-sized "
                f"branch (Space[{chunk_i}::{chunk_n}]; cell count known at run time)"
            )
        else:
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
    compress = kwargs.pop("compress", True)          # save options, not backend-run kwargs
    record_only = kwargs.pop("record_only", True)
    # initial_state.from_experiment seeds from a sibling experiment's saved result;
    # search the output dir's parent (covers results/<key> and output/nc/exp<id> alike).
    # Explicit --results-root (in kwargs) wins; else default to the output dir's parent.
    results_root = kwargs.pop("results_root", None) or (out_dir.parent if out_dir is not None else None)
    result = experiment.run(format=backend, results_root=results_root, **kwargs)
    _common.info(f"done: {type(result).__name__}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "save"):
            saved = result.save(str(out_dir), compress=compress, record_only=record_only)
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
