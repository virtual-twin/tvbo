"""``tvbo run`` — execute a Study or Experiment.

Implements the cardinal HPC contract from §5.1 of ``dev/tvbo-cli.md``:

* ``--engine slurm`` re-emits via :mod:`tvbo.cli.workflow` and submits
  through ``sbatch`` rather than running locally.
* ``--container IMAGE`` re-execs the run inside the named OCI image
  (Singularity if ``SINGULARITY_BIND`` is set in the environment, else Docker).
* ``--shard i/N`` runs one shard of the sweep in-process (no scheduler):
  cell index ``j`` runs iff ``j %% N == i``. This is what the generated sbatch script invokes for every array index.
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
        None,
        "--backend",
        "-b",
        help="Execution backend (tvboptim, tvb, jax, brian2, pyrates, networkdynamics, ...). "
        "Default: each experiment's declared execution.backend, else tvboptim.",
    ),
    out_dir: Path = typer.Option(None, "--out-dir", "-o", help="Directory to write results into."),
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
        help="When SPEC is a StudyCollection, skip these member studies (by label or recipe "
        "stem), comma-separated/repeatable — their committed figures/results are reused "
        "as-is. Members flagged `optional:` are skipped by default; pass --all-members "
        "to include them.",
    ),
    all_members: bool = typer.Option(
        False,
        "--all-members",
        help="When SPEC is a StudyCollection, also run members flagged `optional:` (the heavy studies skipped by default).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="When SPEC is a StudyCollection, list the members, analyses and result keys that "
        "WOULD run (honouring --skip / --all-members) and emit nothing.",
    ),
    manifest_only: bool = typer.Option(
        False,
        "--manifest-only",
        help="When SPEC is a StudyCollection, emit the results manifest from existing containers "
        "and authored values only — run no members or experiments. The fast refresh for the "
        "two-tier build (the heavy `tvbo run` produces the containers; this restamps the "
        "manifest the manuscript reads).",
    ),
) -> None:
    """Run a SPEC (experiment or study) in the selected backend.

    Resolves *spec* to a `SimulationExperiment` or `SimulationStudy`, executes via *backend* on *engine*, and optionally writes results to `--out-dir`.
    Non-local engines re-emit the run through `tvbo workflow ENGINE` and submit.
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

    if kind == "study_collection":
        # A StudyCollection runs every member end to end with fixed save options, so any of these is silently dropped — turning a one-container request into the whole study, or reporting success for a --save-all that saved record-only.
        rejected = _sim_flags + [
            f
            for f, v in (
                ("--analysis", analysis),
                ("--save-all", save_all or None),
                ("--no-compress", None if compress else True),
                ("--results-root", results_root),
            )
            if v is not None
        ]
        if rejected:
            _common.die(
                f"{spec} is a StudyCollection, which runs every member study end to end; "
                f"{', '.join(rejected)} would be ignored. Point the flag at the member "
                "study that declares the work."
            )
        _run_study_collection(
            obj,
            spec,
            out_dir,
            backend=backend,
            figures=figures,
            skip=skip,
            all_members=all_members,
            dry_run=dry_run,
            manifest_only=manifest_only,
        )
        return

    if analysis is not None:
        # Ignoring one would exit 0 having simulated nothing, which a cluster reads as success.
        given = _sim_flags
        if given:
            _common.die(
                f"--analysis runs no experiments, so {', '.join(given)} would be ignored. Run them as a separate command."
            )
        if kind != "study":
            _common.die(
                f"--analysis needs a study: {spec} resolves to a {kind}, which "
                f"declares no `analyses:`. Point it at the study that does."
            )

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
    # A pre-rendered backend script replaces codegen for this run only; every other flag (subject/shard/-o/--set/--experiment) keeps its meaning. Read once here and thread it through to Experiment.run as the rendered_code override.
    if rendered is not None:
        try:
            kwargs["rendered_code"] = Path(rendered).read_text(encoding="utf-8")
        except OSError as e:
            _common.die(f"--rendered: cannot read {rendered}: {e}")

    chunk_i = chunk_n = None
    if shard:
        chunk_i, chunk_n = _parse_chunk(shard)
        _common.info(f"sharding: cell j runs iff j%{chunk_n}=={chunk_i}")
    if kind == "study":
        _import_figure_code_modules(obj)
        analyses_before, analyses_after = _study_analysis_stages(obj)
        if analysis is not None:
            _run_named_analyses(analyses_before + analyses_after, analysis, spec, out_dir)
            return
        whole_study = experiment is None and shard is None
        if whole_study:
            _run_study_analyses(analyses_before, spec, out_dir, stage="before")
        exps = obj.experiments if hasattr(obj, "experiments") else obj.simulation_experiments
        items = list(exps.values()) if hasattr(exps, "values") else list(exps)
        if experiment is not None:
            # Accept one id/name or a comma-separated list ("2,3,20,30" runs all).
            wanted = {s.strip() for s in str(experiment).split(",") if s.strip()}
            items = [e for e in items if wanted & _common.experiment_ids(e)]
            if not items:
                _common.die(f"No experiment(s) matching {experiment!r} in study.")
        # A single frozen script belongs to a single experiment; refuse to run it against several (each experiment renders its own code). The workflow always drives one experiment per rule, so this only guards manual misuse.
        if rendered is not None and len(items) > 1:
            _common.die(
                f"--rendered is a single pre-rendered experiment script but {len(items)} "
                f"experiments matched; pass --experiment to select exactly one."
            )
        for exp in items:
            _common.info(f"running experiment: {getattr(exp, 'key', None) or getattr(exp, 'label', None)}")
            # Resolve to the runtime experiment (has .run) rather than the datamodel object.
            if not hasattr(exp, "run") and hasattr(obj, "get_experiment"):
                sel = (
                    getattr(exp, "id", None)
                    or getattr(exp, "key", None)
                    or getattr(exp, "name", None)
                    or getattr(exp, "label", None)
                )
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
        ok = _run_study_analyses(analyses_after, spec, out_dir, stage="after") if whole_study else True
        if not whole_study and shard is None:
            # Not per shard: an array task holds one slice of one sweep, so every task would repeat the warning and its "refresh now" remedy would run on a half-done grid.
            _warn_stale_analyses(
                analyses_before + analyses_after,
                spec,
                out_dir,
                experiments={i for exp in items for i in _common.experiment_ids(exp)},
            )
        if figures and whole_study and ok:
            _render_study_figures(obj, spec, out_dir)
        return

    if kind == "experiment":
        _apply_metadata_overrides(obj, set_)
        _apply_axis_pins(obj, pin)
        _apply_max_iterations(obj, eff_max_iterations)
        _run_one(obj, _effective_backend(obj, backend), out_dir, kwargs, chunk_i, chunk_n, limit)
        return

    _common.die(f"`tvbo run` does not yet support kind={kind!r}.")


# Helpers


def _import_figure_code_modules(study) -> None:
    """Import a study's figure ``code_modules`` so their registered transforms/panels are available before its experiments run.

    A figure ``Layer.transform`` and a builder/parameter ``used:`` transform name the same ``bsplot.register_transform`` registry, but the latter is resolved during the experiment run, before any figure renders. Importing the declared modules up front (the study loader has already put ``code/`` on the path) fires their ``register_*`` decorators once, study- wide. Import errors are swallowed here — a genuinely broken module is reported with full context when a figure that needs it renders; this pass only pre-populates the registry.
    """
    import importlib

    figures = getattr(study, "figures", None) or []
    seen: list[str] = []
    for fig in figures.values() if hasattr(figures, "values") else figures:
        for m in getattr(fig, "code_modules", None) or []:
            if str(m) not in seen:
                seen.append(str(m))
    for m in seen:
        try:
            importlib.import_module(m)
        except Exception:
            pass


def _study_analysis_stages(study) -> tuple[list, list]:
    """A study's ``analyses:`` split into the stages that run before / after its experiments.

    Returns two empty lists when the study declares none, so callers need no guard. A malformed schedule (duplicate name, unknown or circular ``used:``) raises here, before any experiment runs, rather than half way through a long study.
    """
    from tvbo.data.analysis_io import schedule, study_analyses

    analyses = study_analyses(study)
    return schedule(analyses) if analyses else ([], [])


def _warn_stale_analyses(analyses, spec: str, out_dir: Path | None, *, experiments=(), recomputed=()) -> None:
    """Name the containers a partial run just invalidated but did not recompute.

    Both partial modes need this and they need it identically. ``--experiment`` re-runs a simulation, ``--analysis`` re-derives a container, and in each case everything downstream keeps the PREVIOUS numbers while the thing it reads is fresh. Nothing raises, so a figure or report built next mixes the two.

    ``recomputed`` names what this run actually produced, which both seeds the walk and drops out of its result. It is the CLOSURE, not what was asked for on the command line:
    a named analysis pulls a never-produced upstream in with it, and seeding on the request would miss every dependent of that upstream.
    """
    from tvbo.data.analysis_io import container_path, dependents_of

    if not analyses:
        return
    root = _container_root(spec, out_dir)
    produced = set(recomputed)
    stale = [
        n
        for n in dependents_of(analyses, experiments=experiments, changed_analyses=produced)
        if n not in produced and container_path(n, root).exists()
    ]
    if not stale:
        return
    _common.warn(
        f"{len(stale)} analysis container(s) read something this run re-computed and were "
        f"NOT recomputed themselves: {', '.join(stale)}. Any figure or report built now "
        f"mixes fresh results with those stale reductions. Refresh them with "
        f"`tvbo run {spec} --analysis {','.join(stale)}`, or a whole-study "
        f"`tvbo run {spec}`, or delete them under {root / 'results'} first."
    )


def _run_named_analyses(analyses, wanted: str, spec: str, out_dir: Path | None) -> None:
    """Run only the named ``analyses:``, plus whatever they read, in dependency order.

    The counterpart to ``--experiment`` on the derivation side. It exists because an analysis container is content-addressed on its INPUTS: editing the callable that produces it changes nothing a cache can see, so the only way to refresh one is to ask for it by name.
    Its own upstream analyses are pulled in — a container that has never been produced cannot be read — while the experiments are left alone, which is the point.
    """
    from tvbo.data.analysis_io import analysis_closure, analysis_name, container_path

    names = [s.strip() for s in str(wanted).split(",") if s.strip()]
    if not names:
        _common.die("--analysis was given no names. Pass one or more declared analysis names, comma-separated.")
    # Through `analysis_name`, not `getattr`: the loader may hand these over as Mappings.
    by_name = {analysis_name(a): a for a in analyses}
    missing = [n for n in names if n not in by_name]
    if missing:
        _common.die(f"No analysis named {', '.join(missing)} in {spec}. Declared: {', '.join(sorted(by_name)) or '(none)'}")

    root = _container_root(spec, out_dir)
    needed = analysis_closure(analyses, names, exists=lambda n: container_path(n, root).exists())
    ordered = [a for a in analyses if analysis_name(a) in needed]
    if len(ordered) > len(names):
        _common.info(f"also producing {len(ordered) - len(names)} upstream analysis container(s) that do not exist yet")
    _run_study_analyses(ordered, spec, out_dir, stage="named")
    _warn_stale_analyses(analyses, spec, out_dir, recomputed=needed)
    _common.info(
        f"figures were NOT re-rendered; run `tvbo figure render {spec}` to redraw them from the refreshed container(s)."
    )


def _spec_base(spec: str) -> Path:
    """The study file's own directory — the root ``used:`` references resolve against."""
    spec_path = Path(spec)
    return spec_path.resolve().parent if spec_path.is_file() else Path.cwd()


def _container_root(spec: str, out_dir: Path | None) -> Path:
    """The directory holding THIS run's result containers.

    ``<root>/results/<name>/result.h5`` for an analysis, ``<root>/exp-N_*.h5`` for an experiment; figures resolve them under ``<root's parent>/output/…``, so the root is always a directory named ``output``. The single place ``--out-dir`` is mapped onto that layout, because the analysis WRITER and the figure READER disagreeing is invisible:
    with the documented ``-o output/nc`` the analyses landed in ``output/nc/results/`` while the figures looked in ``output/results/``, so one command rendered this run's experiments against a previous run's analyses.
    """
    base = _spec_base(spec)
    if out_dir is None:
        return base / "output"
    od = Path(out_dir).resolve()
    if od.name == "nc" and od.parent.name == "output":
        return od.parent
    if od.name == "output":
        return od
    return od / "output"


def _run_study_analyses(analyses, spec: str, out_dir: Path | None, *, stage: str) -> bool:
    """Execute one stage of a study's declarative ``analyses:``; True when the stage held.

    Each writes ``<root>/results/<name>/result.h5`` — the container a figure layer or a later analysis binds with ``used: {analysis: <name>}``, at the root :func:`_container_root` resolves for this run.

    A failure is only ever SWALLOWED when there are completed experiments to protect. The ``before`` stage raises — nothing has run yet, and an experiment may source the missing analysis. A ``named`` stage (``--analysis``) raises too: it ran no experiments, the analysis is the whole of what was asked for, and a warning there would exit zero on a job that produced nothing. Only the ``after`` stage reports and returns False, because the experiments already succeeded and must not be lost to a reduction; the figures that would read the missing container are then skipped rather than drawn from absent data.
    """
    from tvbo.data.analysis_io import run_analyses

    if not analyses:
        return True
    base = _spec_base(spec)
    root = _container_root(spec, out_dir)
    try:
        run_analyses(
            analyses,
            root,
            on_start=lambda n: _common.info(f"running analysis: {n}"),
            on_done=lambda n, p: _common.info(f"  wrote {p.relative_to(base) if p.is_relative_to(base) else p}"),
        )
    except Exception as e:
        if stage in ("before", "named"):
            raise
        _common.warn(
            f"analysis stage failed after the experiments completed ({type(e).__name__}: {e}). "
            f"The runs are saved under {root}; skipping figures, which would read a "
            f"container that was not written. Re-run to retry the analyses and figures."
        )
        return False
    return True


def _render_study_figures(study, spec: str, out_dir: Path | None) -> None:
    """Render a study's declarative ``figures:`` after its experiments have run.

    Reuses the exact ``tvbo figure render`` path (``figures.render_figures``), so the images and render scripts a one-command ``tvbo run`` produces are byte-identical to a follow-up ``tvbo figure render`` — the study run just fuses the two steps. ``base_dir`` is the study file's directory (the root each layer's ``used`` IRI resolves against, ``<base>/output/…``); figures land in ``<base>/figures`` to match the render command's default. A study with no ``figures:`` is a silent no-op. A render failure is reported but does not fail the run — the experiments already succeeded and their results are on disk.
    """
    from tvbo.utils import as_list

    from .figures import render_figures

    figs = as_list(getattr(study, "figures", None))
    if not figs:
        return

    base = _spec_base(spec)
    # A run persists results only when --out-dir is given (see _exec_one). With no -o and no results already on disk under <base>/output, every layer would resolve to nothing and each panel would be an empty placeholder — skip and tell the user how to get real figures, rather than emit blank plots. Prior results under output/ still render.
    if out_dir is None:
        if not (base / "output").is_dir():
            _common.info(
                f"skipping figures: this run saved no results (pass -o to persist them, e.g. `tvbo run {spec} -o output`)."
            )
            return
        # output/ exists but THIS run did not persist (no -o): the figures below render from a PREVIOUS run's results, which may be stale. Warn rather than silently mislead.
        _common.warn(
            f"rendering figures from existing results under {base}/output — this run did not "
            f"persist (no -o), so the figures reflect a PREVIOUS run, not this one. Pass -o "
            f"(e.g. `tvbo run {spec} -o output/nc`) to render this run's own results."
        )
    # Figures resolve result containers under <fig_base>/output/… — the SAME mapping the analysis stage writes through, so the two cannot disagree about where this run's containers are.
    fig_base = _container_root(spec, out_dir).parent
    out_figs = base / "figures"
    _common.info(f"rendering {len(figs)} figure(s) -> {out_figs}")
    try:
        render_figures(figs, fig_base, out_figs)
    except Exception as e:  # noqa: BLE001 - never lose a completed run over a plotting error
        _common.info(
            f"figure rendering failed ({type(e).__name__}: {e}); the experiment "
            f"results are saved. Re-run `tvbo figure render {spec}` to retry."
        )


def _run_whole_study(obj, spec: str, out_dir: Path | None, *, backend: str | None = None, figures: bool = True) -> bool:
    """Run a study's WHOLE pipeline — every experiment, its before/after analyses, its figures.

    The subset of the ``kind == "study"`` branch with no per-run selectors (no --experiment /
    --shard / --pin / --set), reused for each StudyCollection member and for the collection's
    own demo content. Returns whether the after-analysis stage held (figures are skipped if it did not), mirroring the study branch.
    """
    _import_figure_code_modules(obj)
    analyses_before, analyses_after = _study_analysis_stages(obj)
    _run_study_analyses(analyses_before, spec, out_dir, stage="before")
    exps = getattr(obj, "experiments", None)
    if exps is None:
        exps = getattr(obj, "simulation_experiments", None) or []
    items = list(exps.values()) if hasattr(exps, "values") else list(exps)
    for exp in items:
        _common.info(f"running experiment: {getattr(exp, 'key', None) or getattr(exp, 'label', None)}")
        if not hasattr(exp, "run") and hasattr(obj, "get_experiment"):
            sel = (
                getattr(exp, "id", None)
                or getattr(exp, "key", None)
                or getattr(exp, "name", None)
                or getattr(exp, "label", None)
            )
            try:
                exp = obj.get_experiment(sel)
            except Exception as e:
                _common.die(f"Could not resolve experiment {sel!r} to a runnable object: {e}")
        _run_one(exp, _effective_backend(exp, backend), out_dir, {"compress": True, "record_only": True}, None, None, None)
    ok = _run_study_analyses(analyses_after, spec, out_dir, stage="after")
    if figures and ok:
        _render_study_figures(obj, spec, out_dir)
    return ok


def _run_study_collection(
    inv,
    spec: str,
    out_dir: Path | None,
    *,
    backend: str | None = None,
    figures: bool = True,
    skip=(),
    all_members: bool = False,
    dry_run: bool = False,
    manifest_only: bool = False,
) -> None:
    """Run a StudyCollection: every member study, the collection's own demo content, then emit the results manifest the manuscript reads.

    Optional members are skipped unless ``all_members``; ``skip`` drops named members (by label or recipe stem); ``dry_run`` lists what would run and emits nothing. The manifest lands at ``<collection-dir>/manuscript_results.yml`` — a committed derived artifact (the seam Quarto reads as ``{{< meta results.* >}}``), so the build never needs the generated run containers;
    an unresolved result key hard-fails the run.
    """
    from tvbo.data.analysis_io import analysis_name, study_analyses
    from tvbo.data.study_collection import MANIFEST_NAME, emit_manifest
    from tvbo.utils import as_list

    base = Path(getattr(inv, "_source_file", spec)).resolve().parent
    skip_set = {s.strip() for part in skip for s in str(part).split(",") if s.strip()}
    members = inv.member_recipes(base, include_optional=all_members)
    to_run = [(label, p) for label, p in members if label not in skip_set and Path(p).stem not in skip_set]
    skipped = [label for label, p in members if (label, p) not in to_run]
    own_analyses = [analysis_name(a) for a in study_analyses(inv)]
    n_results = len(as_list(getattr(inv, "results", None)))

    if dry_run:
        _common.info(f"[dry-run] study collection: {getattr(inv, 'title', None) or spec}")
        for label, p in to_run:
            _common.info(f"  member: {label}  ({p})")
        for label in skipped:
            _common.info(f"  member skipped: {label}")
        if own_analyses:
            _common.info(f"  own analyses: {', '.join(own_analyses)}")
        _common.info(f"  own figures: {len(as_list(getattr(inv, 'figures', None)))}")
        _common.info(f"  would emit {MANIFEST_NAME} with {n_results} result key(s)")
        return

    inv_out = Path(out_dir) if out_dir is not None else (base / "output" / "nc")
    results_root = _container_root(spec, inv_out)

    def _emit() -> None:
        out_path, problems = emit_manifest(inv, results_root, base / MANIFEST_NAME)
        if problems:
            _common.die("results manifest has unresolved key(s):\n  - " + "\n  - ".join(problems))
        _common.info(f"wrote results manifest: {out_path} ({n_results} key(s))")

    if manifest_only:
        _emit()
        return

    # A failed analysis stage means the containers the manifest reads are stale or absent, so emitting from them would report a number the run did not produce. Members are all attempted first — one broken member should not hide the state of the others.
    failed: list[str] = []
    for label, p in to_run:
        _common.info(f"=== member: {label} ({p}) ===")
        kind, mobj = _common.resolve_spec(str(p))
        if kind not in ("study", "study_collection"):
            _common.warn(f"member {label!r} resolves to a {kind}, not a study; skipping.")
            continue
        member_out = Path(p).resolve().parent / "output" / "nc"
        if not _run_whole_study(mobj, str(p), member_out, backend=backend, figures=figures):
            failed.append(label)

    if not _run_whole_study(inv, spec, inv_out, backend=backend, figures=figures):
        failed.append(getattr(inv, "title", None) or spec)
    if failed:
        _common.die(
            "analysis stage failed for: "
            + ", ".join(failed)
            + "\nThe results manifest was NOT written; it would have reported "
            "numbers from stale or missing containers."
        )
    _emit()


def _effective_backend(experiment, cli_backend: str | None) -> str:
    """Resolve which backend runs *experiment*.

    An explicit ``--backend`` wins for the whole run; otherwise each experiment self-selects via its declared ``execution.backend`` (e.g. a spiking network sets ``brian2``), falling back to ``tvboptim``. This lets one study mix a mean-field sweep and a spiking column and run each on the right engine.
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
    if v[:1] in "[{":  # JSON list/object, e.g. [0,2] or ["xi","freq"]
        import json

        try:
            return json.loads(v)
        except ValueError:
            pass
    return v


def _apply_metadata_overrides(experiment, overrides: list[str]) -> None:
    """Apply ``--set dotted.path=value`` overrides to a resolved experiment in place.

    Traverses attributes and keyed collections (LinkML keyed dicts) so one recipe can stay the single source of truth while a run uses test settings. Mutates the loaded object only — the recipe file is untouched.

    Anything on the path that has already MATERIALISED from its declaration is invalidated, so it rebuilds from the new value. Without this an override of, say, a graph generator's connectome is reported and then ignored — the network resolved at load time and keeps the matrix it built — and the run completes, looks right, and is not the run that was asked for.
    """

    def _step(cur, seg):
        if isinstance(cur, dict) and seg in cur:
            return cur[seg]
        if hasattr(cur, seg):
            return getattr(cur, seg)
        try:  # LinkML keyed collection (dict-like __getitem__)
            return cur[seg]
        except Exception:
            _common.die(f"--set: cannot resolve {seg!r} on {type(cur).__name__}")

    for raw in overrides:
        s = raw.lstrip("-")
        if "=" not in s:
            raise typer.BadParameter(f"--set {raw!r} must be of the form path=value")
        path, _, val = s.partition("=")
        segs = [p for p in path.split(".") if p]
        cur, chain = experiment, [experiment]
        for seg in segs[:-1]:
            cur = _step(cur, seg)
            chain.append(cur)
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
        rebuilt = _invalidate_on_path(chain)
        _common.info(f"--set {path} = {value!r}" + (f"  ({rebuilt} rebuilds from it)" if rebuilt else ""))


def _invalidate_on_path(chain: list):
    """Invalidate the innermost object on *chain* that materialises from its declaration.

    Returns the name of what was invalidated, or ``None``. Innermost wins: an override inside one network's generator must not rebuild an unrelated network beside it.
    """
    for obj in reversed(chain):
        invalidate = getattr(obj, "invalidate_resolution", None)
        if callable(invalidate):
            invalidate()
            return type(obj).__name__
    return None


def _apply_max_iterations(experiment, n: int) -> None:
    """Cap every algorithm's and stage's ``n_iterations`` (and any optimization's ``max_iterations``) to *n* for THIS run — a smoke override, the recipe untouched.

    The post-tuning evaluation of a fit — the memory- and time-critical part of a long-horizon run — is independent of how many tuning iterations preceded it, so a handful of iterations is enough to verify the fit executes and its long-horizon post-tuning observables stream within memory. Mirrors ``--set``: it mutates only the loaded object, so one recipe stays the single source of truth.
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
                holder.n_iterations = n
                _capped += 1

    algos = getattr(experiment, "algorithms", None) or {}
    for algo in algos.values() if hasattr(algos, "values") else algos:
        _cap(algo)
        for stage in getattr(algo, "stages", None) or []:
            _cap(stage)

    opts = getattr(experiment, "optimizations", None) or {}
    for opt in opts.values() if hasattr(opts, "values") else opts:
        cur = getattr(opt, "max_iterations", None)
        if isinstance(cur, int) and cur > n:
            opt.max_iterations = n
            _capped += 1

    _common.info(f"--max-iterations {n}: capped {_capped} iteration count(s)")


def _apply_axis_pins(experiment, pins: list[str]) -> None:
    """Pin fanned exploration axes to single values for THIS run — the workflow fan-out's per-cell restriction (the model-scope sibling of ``--subject``).

    For each ``parameter=value``: set the axis's parameter on the experiment so the base (representative) run uses it — every DECLARED observation, host or not, is computed on that run, so this is what makes a fanned cell's host observation land at the cell's coordinates — AND drop that axis from every exploration so the sweep does not re-expand it. An exploration left with no axes is removed, collapsing the run to a single point.
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

    Mirrors the codegen axis classifier (tvbo-tvboptim-experiment.py.mako): ``network.<p>`` is a network scalar; ``<coupling-name>.<p>`` is a coupling parameter; anything else ``<x>.<p>`` (or a bare ``<p>``) is a dynamics parameter; and an experiment-scoped path (``execution.random_seed``, ``integration.<p>``) falls back to the ``--set`` attribute walk, which resolves those correctly. Kept in step with that classifier so a pinned run and the swept grid write the same target.
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
        entry.value = value
        return True

    if parameter.startswith("network."):
        leaf = parameter[len("network.") :]
        net = getattr(experiment, "network", None)
        if net is not None and _set_in(getattr(net, "parameters", None), leaf):
            return
        _common.die(f"--pin: cannot resolve network parameter {leaf!r} on the network.")
    if "." in parameter:
        prefix, name = parameter.rsplit(".", 1)
        cpl = getattr(experiment, "coupling", None)
        if cpl is not None and getattr(cpl, "name", None) == prefix and _set_in(getattr(cpl, "parameters", None), name):
            return
        net = getattr(experiment, "network", None)
        net_cpl = getattr(net, "coupling", None) if net is not None else None
        entry = net_cpl.get(prefix) if hasattr(net_cpl, "get") else None
        if entry is not None and _set_in(getattr(entry, "parameters", None), name):
            return
        dyn = getattr(experiment, "dynamics", None)
        if dyn is not None and _set_in(getattr(dyn, "parameters", None), name):
            return
        # An experiment-scoped axis has an attribute path, so the --set walk resolves it.
        _apply_metadata_overrides(experiment, [f"{parameter}={value}"])
        return
    dyn = getattr(experiment, "dynamics", None)
    if dyn is not None and _set_in(getattr(dyn, "parameters", None), parameter):
        return
    _common.die(f"--pin: cannot resolve axis parameter {parameter!r}.")


def _drop_exploration_axis(experiment, parameter: str) -> None:
    """Remove the axis with this ``parameter`` from every exploration; drop an exploration left with no axes so a fully-pinned run collapses to a single point (no empty sweep)."""
    explorations = getattr(experiment, "explorations", None) or {}
    expl_items = list(explorations.items()) if hasattr(explorations, "items") else list(enumerate(list(explorations)))
    emptied = []
    for key, expl in expl_items:
        space = getattr(expl, "space", None)
        if not space:
            continue
        if hasattr(space, "items"):  # keyed by parameter (LinkML keyed collection)
            for axk in list(space.keys()):
                if str(getattr(space[axk], "parameter", axk)) == parameter:
                    del space[axk]
            if len(space) == 0:
                emptied.append(key)
        else:  # plain list of axes
            expl.space = [ax for ax in space if str(getattr(ax, "parameter", None)) != parameter]
            if len(expl.space) == 0:
                emptied.append(key)
    # Reverse order so deleting by positional index from a list-form explorations does not shift later indices (dict keys are order-independent).
    for key in reversed(emptied):
        try:
            del explorations[key]
        except Exception:
            pass


def _run_one(
    experiment,
    backend: str,
    out_dir: Path | None,
    kwargs: dict,
    chunk_i: int | None,
    chunk_n: int | None,
    limit: int | None = None,
) -> None:
    if limit is not None and chunk_n is None:
        import math

        from ._workflow import extract_axes

        n_cells = 1
        for ax in extract_axes(experiment):
            n_cells *= len(ax.values)
        if n_cells > limit:
            chunk_i, chunk_n = 0, math.ceil(n_cells / limit)
            _common.info(f"--limit {limit}: running ~{-(-n_cells // chunk_n)} of {n_cells} cells (Space[0::{chunk_n}])")

    if chunk_n is not None:
        from ._backends import resolve_backend
        from ._workflow import extract_axes

        axes = extract_axes(experiment)
        if not axes:
            _common.info("no sweep axes on experiment; running once")
            _exec_one(experiment, backend, out_dir, kwargs)
            return
        # Shardable in-process only where the backend vectorises every swept axis (BackendSpec.can_vectorize).
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
            # Branch-restart sweep: the cell count comes from the source run's recorded branch (read at run time), so this task just slices its share of it.
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
    compress = kwargs.pop("compress", True)  # save options, not backend-run kwargs
    record_only = kwargs.pop("record_only", True)
    # The output dir's parent covers results/<key> and output/nc/exp<id> alike; --results-root wins.
    results_root = kwargs.pop("results_root", None) or (out_dir.parent if out_dir is not None else None)
    result = experiment.run(format=backend, results_root=results_root, **kwargs)
    _common.info(f"done: {type(result).__name__}")
    if out_dir is None:
        _common.warn(
            f"result NOT saved: no --out-dir, so this run computed {type(result).__name__} "
            f"and discarded it. Any figures/report built afterwards will read STALE results "
            f"from a previous run, not this one. Pass -o to persist, e.g. "
            f"`tvbo run <spec> -o output/nc`."
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "save"):
        saved = result.save(str(out_dir), compress=compress, record_only=record_only)
        _common.info(f"wrote {saved}")
    else:
        _common.info(f"(result has no .save(); skipping write to {out_dir})")


def _dispatch_to_engine(
    engine: str, *, spec: str, backend: str, experiment: str | None, container: str | None, out_dir: Path | None
) -> None:
    """Emit a workflow kit for *engine* and submit/execute it, all in-process.

    Shares the emit + execute path with ``tvbo workflow <engine>`` rather than re-shelling ``tvbo`` (which needs it on ``$PATH`` — fragile under venv / module / container setups on HPC) and rather than building the plan twice.
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

    kit_dir = _workflow_cmd._emit(
        engine, spec=spec, backend=backend, experiment=experiment, output=out_dir, override=overrides, stdout=False
    )
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
        cmd = ["docker", "run", "--rm", "-e", "TVBO_IN_CONTAINER=1", "-v", f"{cwd}:{cwd}", "-w", cwd, image, "tvbo", *argv]
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    raise SystemExit(subprocess.run(cmd).returncode)


def _which(prog: str) -> str | None:
    from shutil import which

    return which(prog)
