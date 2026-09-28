"""Run a `SimulationStudy` end to end: its experiments, its analyses, its figures, and the provenance of every container it writes.

`tvbo run` and [`SimulationStudy.run()`](/api/classes/study.qmd#tvbo.classes.study.SimulationStudy.run) both call this module, so a run from a shell and a run from a notebook take one path and leave one record. A study that declares `studies:` runs as a tree: every nested study first, each in its own directory, then the holder's own content, then the results manifest the manuscript reads.

Nothing here exits the process. A run that refuses its input or cannot finish raises `StudyRunError` with a message written for the person who asked for the run; `tvbo run` prints that message and exits 1, and the Python API lets it propagate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StudyRunError(RuntimeError):
    """A run refused its input or could not finish, for the reason its message gives.

    Raised only where that message is the whole story, so it is addressed to the person who asked for the run rather than to a traceback reader.
    """


@dataclass(frozen=True)
class RunOptions:
    """How each experiment of one run executes and saves; the recipe itself is never modified.

    The defaults run the recipe as written and save its recorded outputs compressed. Each field is the `tvbo run` flag named beside it.

    Attributes:
        backend: Backend for every experiment, over each one's declared `execution.backend` (`--backend`).
        overrides: `(dotted.path, value)` assignments applied to each experiment before it runs (`--set`).
        pins: `(axis parameter, value)` pairs, each setting that parameter and dropping its exploration axis (`--pin`).
        max_iterations: Cap on every algorithm's and stage's `n_iterations` (`--max-iterations`, `--smoke`).
        duration: `integration.duration` in ms (`--duration`).
        subject: Active subject of a per-subject dataset experiment (`--subject`).
        results_root: Where a `from_experiment` warm start looks for its source run, in place of this run's results directory (`--results-root`).
        rendered_code: A pre-rendered backend script run as-is in place of generated code, for a single experiment (`--rendered`).
        limit: Run at most this many cells of a sweep, a spread sample; ignored under `shard` (`--limit`).
        shard: `(i, n)`, running the sweep cells `j` with `j % n == i` (`--shard`).
        compress: gzip the result containers (`--compress/--no-compress`).
        record_only: Save the recorded outputs only, not every intermediate observation (cleared by `--save-all`).
    """

    backend: str | None = None
    overrides: Sequence[tuple[str, Any]] = ()
    pins: Sequence[tuple[str, Any]] = ()
    max_iterations: int | None = None
    duration: float | None = None
    subject: str | None = None
    results_root: Path | str | None = None
    rendered_code: str | None = None
    limit: int | None = None
    shard: tuple[int, int] | None = None
    compress: bool = True
    record_only: bool = True

    def backend_kwargs(self) -> dict:
        """The keyword arguments `SimulationExperiment.run` takes from these options."""
        kwargs: dict = {}
        if self.duration is not None:
            kwargs["duration"] = self.duration
        if self.subject is not None:
            kwargs["active_subject"] = self.subject
        if self.rendered_code is not None:
            kwargs["rendered_code"] = self.rendered_code
        return kwargs


# Where a run reads and writes


def spec_dir(spec: str) -> Path | None:
    """The directory a spec's relative paths mean, or ``None`` when the spec is not a file.

    A spec may also be a CURIE (``study:Deco2014``), a bare database name or a ``file://`` URL; ``Path(spec).parent`` on any of those silently yields the cwd, which would resolve relative paths against whatever directory the run happened to start in.
    """
    raw = spec[len("file://") :] if spec.startswith("file://") else spec
    path = Path(raw).expanduser()
    return path.resolve().parent if path.is_file() else None


def spec_base(spec: str) -> Path:
    """The study file's own directory — the root ``used:`` references resolve against — or the cwd for a spec that is not a file."""
    return spec_dir(spec) or Path.cwd()


def study_path_for(role: str, base: Path) -> Path:
    """The directory *base* gives *role*, per the study-layout record."""
    from tvbo.utils.study_layout import study_path

    return study_path(role, root=base)


def results_root(spec: str, out_dir: Path | None, *, base: Path | None = None) -> Path:
    """The directory holding THIS run's result containers.

    The results directory of the study root the run writes into (:mod:`tvbo.utils.study_layout`, role ``results``), holding ``exp-<id>_*_result.h5`` for a run and ``ana-<name>_result.h5`` for an analysis, flat. That root is *base*, defaulting to the recipe's own directory. *out_dir* (``-o``) overrides the location and nothing else. Writer and reader ask this one function, because two independent answers disagreeing is invisible: a run would render this run's experiments against a previous run's analyses and report success.
    """
    if out_dir is not None:
        return Path(out_dir).resolve()
    return study_path_for("results", Path(base).resolve() if base is not None else spec_base(spec))


# Which experiments a run covers


def experiment_ids(exp: Any) -> set[str]:
    """The identifiers an experiment can be selected by.

    Its ``key``, ``name``, ``label``, and stringified ``id`` (dropping the empty ones), plus the bare numeric id those spell — ``exp-3``, ``exp3`` and ``3`` name one experiment, and an experiment carrying only ``key: exp-3`` must still answer to ``3``.
    Normalising HERE and in ``analysis_io.dependencies`` is what lets the two sides of the staleness walk intersect; normalising one side only makes every dotted spelling match nothing, and an empty stale set reads exactly like a clean one.

    Shared by ``tvbo run`` and ``tvbo workflow`` so ``--experiment`` matches the same way in both.
    """
    from tvbo.data.dataref import experiment_id

    spellings = {
        getattr(exp, "key", None),
        getattr(exp, "name", None),
        getattr(exp, "label", None),
        str(getattr(exp, "id", "")),
    } - {None, ""}
    return spellings | {experiment_id(s) for s in spellings} - {None}


def select_experiments(study: Any, ids: str | None = None) -> list:
    """The experiment records of *study*, or the comma-separated *ids* subset of them.

    An id is matched against every spelling :func:`experiment_ids` accepts, so ``--experiment`` selects the same way in every verb; a selector that matches nothing raises `StudyRunError`. The records are datamodel objects — :func:`runtime_experiment` turns each into one that can ``run`` and ``render``, which callers do one at a time so a long study never holds every materialised experiment at once.
    """
    from tvbo.utils import as_list

    records = as_list(getattr(study, "experiments", None))
    if ids is None:
        return records
    wanted = {s.strip() for s in str(ids).split(",") if s.strip()}
    records = [e for e in records if wanted & experiment_ids(e)]
    if not records:
        raise StudyRunError(f"No experiment(s) matching {ids!r} in study.")
    return records


def runtime_experiment(study: Any, record: Any) -> Any:
    """*record* materialised through ``study.get_experiment``, which is keyed on its ``id``; a `StudyRunError` names what usually stops one from loading."""
    if hasattr(record, "run") or not hasattr(study, "get_experiment"):
        return record
    record_id = getattr(record, "id", None)
    try:
        return study.get_experiment(record_id)
    except Exception as e:
        raise StudyRunError(
            f"Could not resolve experiment {record_id!r} to a runnable object: {e}\nIf the recipe references custom builder/analysis modules (e.g. `module: my_networks`), make them importable — run from their directory or set PYTHONPATH."
        ) from e


def figure_code_modules(figures: Any) -> list[str]:
    """The ``code_modules`` the *figures* declare, deduplicated in declaration order.

    Importing one registers the custom panels and transforms its figures name, which is why a study run imports them before its experiments and ``tvbo workflow`` bundles them into a kit's ``code/``.
    """
    from tvbo.utils import as_list

    return list(dict.fromkeys(str(m) for fig in as_list(figures) for m in as_list(getattr(fig, "code_modules", None))))


def has_nested(obj: Any) -> bool:
    """Whether *obj* is a study that aggregates others, which is the only thing that changes how a run proceeds."""
    from tvbo.utils import as_list

    return bool(as_list(getattr(obj, "studies", None)))


# Provenance: every container records the run that wrote it


def provenance_root(spec: str, out_dir: Path | None, *, base: Path | None = None) -> Path:
    """The root a recorded ``used:`` edge names its containers relative to.

    The STUDY the spec belongs to, even when ``-o`` sends the data elsewhere: both ends of an edge have to spell a container identically, and the study is the frame the reader resolves them in. A spec that is in no study — a curated experiment run straight out of the installed database — has no such root, so the results themselves are the frame: *out_dir* when the results were redirected, else *base*, the root the run writes into, which defaults to the spec's own directory. *out_dir* is ``-o`` as the caller gave it, not the directory it resolves to: the question is whether the results were redirected, and a resolved default answers that yes every time.
    """
    from tvbo.utils.study_layout import study_root

    own = spec_base(spec)
    try:
        return study_root(own)
    except FileNotFoundError:
        if out_dir is not None:
            return Path(out_dir).resolve()
        return Path(base).resolve() if base is not None else own


def provenance_ctx(spec: str, owner: Any, out_dir: Path | None = None, *, base: Path | None = None) -> dict:
    """What a run needs in hand to record itself beside each container it writes; *owner* is the study (or lone experiment) the records name.

    Always built. The record goes into the container's own sidecar, which is a product of the run and never tracked, so remembering what happened costs one key in a file the run was writing anyway — there is nothing for a study to switch off.
    """
    from tvbo.data import provenance

    return {
        "study_root": provenance_root(spec, out_dir, base=base),
        "study": str(getattr(owner, "citekey", None) or getattr(owner, "key", None) or Path(spec).stem),
        "started_at": provenance.now(),
        "requires": tuple(getattr(owner, "requires", None) or ()),
    }


def _emit_provenance(ctx: dict | None, container: Path, produced_by: str, outputs=(), used=()) -> None:
    """Record one container's run, reporting rather than raising if it cannot be written.

    A run that computed its result has succeeded; failing it afterwards over its own bookkeeping would throw away the compute. The warning names what is missing so the gap is visible rather than silent.
    """
    if ctx is None:
        return
    from tvbo.data import provenance

    try:
        provenance.emit(
            container=Path(container),
            produced_by=produced_by,
            outputs=outputs,
            used=used,
            started_at=ctx.get("started_at"),
            requires=ctx.get("requires", ()),
        )
    except Exception as e:  # noqa: BLE001 — the result stands; only its record is missing
        logger.warning(f"provenance for {Path(container).name} not written ({type(e).__name__}: {e})")


def _analysis_inputs(analysis, root, ctx: dict | None):
    """The containers an analysis's ``used:`` arguments read, so its activity records what it consumed and not only what it produced.

    Without these the records describe each run in isolation; with them a study's provenance is the derivation graph its analyses actually form.
    """
    if analysis is None or ctx is None:
        return ()
    from tvbo.data.provenance import input_containers
    from tvbo.utils import as_list

    refs = [getattr(arg, "used", None) for arg in as_list(getattr(analysis, "arguments", None))]
    return input_containers(refs, results_root=root, study_root=ctx["study_root"])


def _container_vars(path: Path) -> list[str]:
    """The data variables a written container actually holds, read back from it."""
    import xarray as xr

    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return sorted(str(v) for v in ds.data_vars)
    except Exception:  # noqa: BLE001 — an unreadable container is reported by whoever needs its data
        return []


def _record_run(ctx: dict | None, experiment, saved) -> None:
    """Describe the container a run just wrote, in that container's own sidecar.

    The written paths are the authority on which file to describe — a run that shards, fans over subjects, or writes nothing at all is then recorded as what it actually produced rather than as what the stem predicted.
    """
    if ctx is None:
        return
    containers = [
        Path(p) for p in ([saved] if isinstance(saved, (str, Path)) else list(saved or [])) if str(p).endswith(".h5")
    ]
    eid = getattr(experiment, "id", None) or getattr(experiment, "name", None)
    for container in containers:
        _emit_provenance(ctx, container, f"tvbo:exp/{ctx['study']}/exp-{eid}", _container_vars(container))


# One experiment


def apply_overrides(experiment, overrides: Iterable[tuple[str, Any]]) -> None:
    """Apply ``(dotted.path, value)`` overrides to a resolved experiment in place.

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
        except Exception as e:
            raise StudyRunError(f"--set: cannot resolve {seg!r} on {type(cur).__name__}") from e

    for path, value in overrides:
        segs = [p for p in path.split(".") if p]
        cur, chain = experiment, [experiment]
        for seg in segs[:-1]:
            cur = _step(cur, seg)
            chain.append(cur)
        leaf = segs[-1]
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
        logger.info(f"--set {path} = {value!r}" + (f"  ({rebuilt} rebuilds from it)" if rebuilt else ""))


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


def apply_max_iterations(experiment, n: int | None) -> None:
    """Cap every algorithm's and stage's ``n_iterations`` (and any optimization's ``max_iterations``) to *n* for THIS run — a smoke override, the recipe untouched.

    The post-tuning evaluation of a fit — the memory- and time-critical part of a long-horizon run — is independent of how many tuning iterations preceded it, so a handful of iterations is enough to verify the fit executes and its long-horizon post-tuning observables stream within memory. Mirrors :func:`apply_overrides`: it mutates only the loaded object, so one recipe stays the single source of truth.
    """
    if n is None:
        return
    capped = 0

    def _cap(holder):
        nonlocal capped
        if isinstance(holder, dict):
            cur = holder.get("n_iterations")
            if isinstance(cur, int) and cur > n:
                holder["n_iterations"] = n
                capped += 1
        else:
            cur = getattr(holder, "n_iterations", None)
            if isinstance(cur, int) and cur > n:
                holder.n_iterations = n
                capped += 1

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
            capped += 1

    logger.info(f"--max-iterations {n}: capped {capped} iteration count(s)")


def apply_axis_pins(experiment, pins: Iterable[tuple[str, Any]]) -> None:
    """Pin fanned exploration axes to single values for THIS run — the workflow fan-out's per-cell restriction (the model-scope sibling of ``--subject``).

    For each ``(parameter, value)``: set the axis's parameter on the experiment so the base (representative) run uses it — every DECLARED observation, host or not, is computed on that run, so this is what makes a fanned cell's host observation land at the cell's coordinates — AND drop that axis from every exploration so the sweep does not re-expand it. An exploration left with no axes is removed, collapsing the run to a single point.
    """
    for parameter, value in pins:
        _set_axis_parameter(experiment, parameter, value)
        _drop_exploration_axis(experiment, parameter)
        logger.info(f"--pin {parameter} = {value!r}")


def _set_axis_parameter(experiment, parameter: str, value) -> None:
    """Write an exploration axis's value onto its parameter target on the experiment.

    Mirrors the codegen axis classifier (tvbo-tvboptim-experiment.py.mako): ``network.<p>`` is a network scalar; ``<coupling-name>.<p>`` is a coupling parameter; anything else ``<x>.<p>`` (or a bare ``<p>``) is a dynamics parameter; and an experiment-scoped path (``execution.random_seed``, ``integration.<p>``) falls back to the :func:`apply_overrides` attribute walk, which resolves those correctly. Kept in step with that classifier so a pinned run and the swept grid write the same target.
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
        raise StudyRunError(f"--pin: cannot resolve network parameter {leaf!r} on the network.")
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
        # An experiment-scoped axis has an attribute path, so the override walk resolves it.
        apply_overrides(experiment, [(parameter, value)])
        return
    dyn = getattr(experiment, "dynamics", None)
    if dyn is not None and _set_in(getattr(dyn, "parameters", None), parameter):
        return
    raise StudyRunError(f"--pin: cannot resolve axis parameter {parameter!r}.")


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


def run_experiment(
    experiment,
    spec: str,
    out_dir: Path | None = None,
    *,
    options: RunOptions | None = None,
    owner: Any = None,
    base: Path | None = None,
) -> None:
    """Run one experiment under *options*, persist its container, and record the run in the container's sidecar.

    Args:
        experiment: A runnable experiment; the overrides, pins and iteration cap in *options* are applied to it in place.
        spec: The recipe the run was asked for, which roots its paths and its provenance.
        out_dir: Results directory, in place of *base*'s.
        options: How the experiment executes and saves; the defaults run it as written.
        owner: The study the experiment belongs to, which its provenance record names; the experiment itself when omitted.
        base: The study root the run writes into; defaults to the recipe's own directory.

    Raises:
        StudyRunError: An override or pin names nothing on the experiment, or a shard asks a backend to slice a sweep it does not vectorise.
    """
    from tvbo.cli._backends import effective_backend

    options = options or RunOptions()
    apply_overrides(experiment, options.overrides)
    apply_axis_pins(experiment, options.pins)
    apply_max_iterations(experiment, options.max_iterations)
    ctx = provenance_ctx(spec, experiment if owner is None else owner, out_dir, base=base)
    _run_one(experiment, effective_backend(experiment, options.backend), results_root(spec, out_dir, base=base), options, ctx)


def _run_one(experiment, backend: str, out_dir: Path, options: RunOptions, prov_ctx: dict | None) -> None:
    """Run *experiment* whole, or the slice of its sweep that ``options.shard`` or ``options.limit`` selects."""
    import math

    shard = options.shard
    if options.limit is not None and shard is None:
        from tvbo.cli._workflow import extract_axes

        n_cells = math.prod(len(ax.values) for ax in extract_axes(experiment))
        if n_cells > options.limit:
            shard = (0, math.ceil(n_cells / options.limit))
            logger.info(
                f"--limit {options.limit}: running ~{-(-n_cells // shard[1])} of {n_cells} cells (Space[0::{shard[1]}])"
            )

    if shard is None:
        _exec_one(experiment, backend, out_dir, options, prov_ctx)
        return

    from tvbo.cli._backends import resolve_backend
    from tvbo.cli._workflow import extract_axes

    chunk_i, chunk_n = shard
    axes = extract_axes(experiment)
    if not axes:
        logger.info("no sweep axes on experiment; running once")
        _exec_one(experiment, backend, out_dir, options, prov_ctx)
        return
    # Shardable in-process only where the backend vectorises every swept axis (BackendSpec.can_vectorize).
    try:
        capabilities = resolve_backend(backend)
    except ValueError:
        capabilities = None
    fanned = [ax for ax in axes if capabilities is None or not capabilities.can_vectorize(ax.kind)]
    if fanned:
        names = ", ".join(f"{ax.parameter} ({ax.kind})" for ax in fanned)
        why = (
            "does not vectorise"
            if capabilities is not None
            else "is not in the backend capability registry, so it vectorises none of"
        )
        raise StudyRunError(
            f"--shard shards a sweep by slicing the backend's vectorised "
            f"batch, but backend {backend!r} {why}: {names}. Emit "
            f"the kit with snakemake/nextflow (which fan these axes into per-cell "
            f"tasks), or use a backend that vectorises them (e.g. tvboptim)."
        )
    if any(getattr(ax, "runtime_sized", False) for ax in axes):
        # Branch-restart sweep: the cell count comes from the source run's recorded branch (read at run time), so this task just slices its share of it.
        logger.info(
            f"sharding: task {chunk_i}/{chunk_n} runs its slice of a runtime-sized "
            f"branch (Space[{chunk_i}::{chunk_n}]; cell count known at run time)"
        )
    else:
        n_cells = math.prod(len(ax.values) for ax in axes)
        per_task = -(-n_cells // chunk_n)  # ceil
        logger.info(
            f"sharding: task {chunk_i}/{chunk_n} runs ~{per_task} of {n_cells} "
            f"sweep cells (backend-vectorised, Space[{chunk_i}::{chunk_n}])"
        )
    # Names this task's slice `split-<i>`, so N tasks writing one directory cannot overwrite each other.
    experiment._active_split = chunk_i
    _exec_one(experiment, backend, out_dir, options, prov_ctx, shard=(chunk_i, chunk_n))


def _exec_one(experiment, backend: str, out_dir: Path, options: RunOptions, prov_ctx: dict | None, shard=None) -> None:
    """Run one experiment and persist its container into ``out_dir``.

    ``out_dir`` is the study's results directory, already resolved by :func:`results_root`: a run always persists, so nothing here has to warn that the figures a caller is about to draw came from a previous run. ``options.results_root`` still wins as the place a warm start reads from, for a run whose cross-experiment sources live elsewhere.
    """
    kwargs = options.backend_kwargs()
    if shard is not None:
        kwargs["shard"] = shard
    source_root = str(options.results_root) if options.results_root is not None else out_dir
    result = experiment.run(format=backend, results_root=source_root, **kwargs)
    logger.info(f"done: {type(result).__name__}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "save"):
        saved = result.save(str(out_dir), compress=options.compress, record_only=options.record_only)
        logger.info(f"wrote {saved}")
        _record_run(prov_ctx, experiment, saved)
    else:
        logger.info(f"(result has no .save(); skipping write to {out_dir})")


# A study's analyses


def _study_analysis_stages(study) -> tuple[list, list]:
    """A study's ``analyses:`` split into the stages that run before / after its experiments.

    Returns two empty lists when the study declares none, so callers need no guard. A malformed schedule (duplicate name, unknown or circular ``used:``) raises here, before any experiment runs, rather than half way through a long study.
    """
    from tvbo.data.analysis_io import schedule, study_analyses

    analyses = study_analyses(study)
    return schedule(analyses) if analyses else ([], [])


def _import_figure_code_modules(study) -> None:
    """Import a study's figure ``code_modules`` so their registered transforms/panels are available before its experiments run.

    A figure ``Layer.transform`` and a builder/parameter ``used:`` transform name the same ``bsplot.register_transform`` registry, but the latter is resolved during the experiment run, before any figure renders. Importing the declared modules up front (the study loader has already put ``code/`` on the path) fires their ``register_*`` decorators once, study-wide. Import errors are swallowed here — a genuinely broken module is reported with full context when a figure that needs it renders; this pass only pre-populates the registry.
    """
    import importlib

    for m in figure_code_modules(getattr(study, "figures", None)):
        try:
            importlib.import_module(m)
        except Exception:
            pass


def warn_stale_analyses(analyses, spec: str, out_dir: Path | None, *, experiments=(), recomputed=()) -> None:
    """Name the containers a partial run just invalidated but did not recompute.

    Both partial modes need this and they need it identically. ``--experiment`` re-runs a simulation, ``--analysis`` re-derives a container, and in each case everything downstream keeps the PREVIOUS numbers while the thing it reads is fresh. Nothing raises, so a figure or report built next mixes the two.

    ``recomputed`` names what this run actually produced, which both seeds the walk and drops out of its result. It is the CLOSURE, not what was asked for on the command line: a named analysis pulls a never-produced upstream in with it, and seeding on the request would miss every dependent of that upstream.
    """
    from tvbo.data.analysis_io import container_path, dependents_of

    if not analyses:
        return
    root = results_root(spec, out_dir)
    produced = set(recomputed)
    stale = [
        n
        for n in dependents_of(analyses, experiments=experiments, changed_analyses=produced)
        if n not in produced and container_path(n, root).exists()
    ]
    if not stale:
        return
    logger.warning(
        f"{len(stale)} analysis container(s) read something this run re-computed and were "
        f"NOT recomputed themselves: {', '.join(stale)}. Any figure or report built now "
        f"mixes fresh results with those stale reductions. Refresh them with "
        f"`tvbo run {spec} --analysis {','.join(stale)}`, or a whole-study "
        f"`tvbo run {spec}`, or delete them under {root / 'results'} first."
    )


def _run_study_analyses(
    study, analyses, spec: str, root: Path, *, stage: str, out_dir: Path | None = None, base: Path | None = None
) -> bool:
    """Execute one stage of *study*'s declarative ``analyses:`` into the results directory *root*; True when the stage held.

    Each writes ``<root>/ana-<name>_result.h5`` — the container a figure layer or a later analysis binds with ``used: {analysis: <name>}``. Each container's provenance record names *study* as the one it belongs to, rooted by `provenance_root` from *out_dir* (``-o`` as the caller gave it) exactly as the experiments' records are. *base* is the study root the run writes into, which the progress lines name each container relative to.

    A failure is only ever SWALLOWED when there are completed experiments to protect. The ``before`` stage raises — nothing has run yet, and an experiment may source the missing analysis. A ``named`` stage (``--analysis``) raises too: it ran no experiments, the analysis is the whole of what was asked for, and a warning there would exit zero on a job that produced nothing. Only the ``after`` stage reports and returns False, because the experiments already succeeded and must not be lost to a reduction; the figures that would read the missing container are then skipped rather than drawn from absent data.
    """
    from tvbo.data.analysis_io import run_analyses

    if not analyses:
        return True
    shown_from = Path(base).resolve() if base is not None else spec_base(spec)
    ctx = provenance_ctx(spec, study, out_dir, base=base)

    by_name = {str(getattr(a, "name", "")): a for a in analyses}

    def _done(name, path):
        logger.info(f"  wrote {path.relative_to(shown_from) if path.is_relative_to(shown_from) else path}")
        _emit_provenance(
            ctx,
            path,
            f"tvbo:ana/{ctx['study']}/{name}",
            _container_vars(path),
            _analysis_inputs(by_name.get(name), root, ctx),
        )

    try:
        run_analyses(
            analyses,
            root,
            on_start=lambda n: logger.info(f"running analysis: {n}"),
            on_done=_done,
        )
    except Exception as e:
        if stage in ("before", "named"):
            raise
        logger.warning(
            f"analysis stage failed after the experiments completed ({type(e).__name__}: {e}). "
            f"The runs are saved under {root}; skipping figures, which would read a "
            f"container that was not written. Re-run to retry the analyses and figures."
        )
        return False
    return True


def run_named_analyses(study, spec: str, wanted: str, out_dir: Path | None = None) -> None:
    """Run only the named ``analyses:`` (comma-separated *wanted*), plus whatever they read, in dependency order.

    The counterpart to ``--experiment`` on the derivation side. It exists because an analysis container is content-addressed on its INPUTS: editing the callable that produces it changes nothing a cache can see, so the only way to refresh one is to ask for it by name.
    Its own upstream analyses are pulled in — a container that has never been produced cannot be read — while the experiments are left alone, which is the point.

    Raises:
        StudyRunError: *wanted* names nothing, or names an analysis the study does not declare.
    """
    from tvbo.data.analysis_io import analysis_closure, analysis_name, container_path

    _import_figure_code_modules(study)
    before, after = _study_analysis_stages(study)
    analyses = before + after
    names = [s.strip() for s in str(wanted).split(",") if s.strip()]
    if not names:
        raise StudyRunError("--analysis was given no names. Pass one or more declared analysis names, comma-separated.")
    # Through `analysis_name`, not `getattr`: the loader may hand these over as Mappings.
    by_name = {analysis_name(a): a for a in analyses}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise StudyRunError(
            f"No analysis named {', '.join(missing)} in {spec}. Declared: {', '.join(sorted(by_name)) or '(none)'}"
        )

    root = results_root(spec, out_dir)
    needed = analysis_closure(analyses, names, exists=lambda n: container_path(n, root).exists())
    ordered = [a for a in analyses if analysis_name(a) in needed]
    if len(ordered) > len(names):
        logger.info(f"also producing {len(ordered) - len(names)} upstream analysis container(s) that do not exist yet")
    _run_study_analyses(study, ordered, spec, root, stage="named", out_dir=out_dir)
    warn_stale_analyses(analyses, spec, root, recomputed=needed)
    logger.info(
        f"figures were NOT re-rendered; run `tvbo figure render {spec}` to redraw them from the refreshed container(s)."
    )


# A study's figures


def figure_outputs(figure, out_dir: Path) -> tuple[str, Path, Path]:
    """``(name, image, script)`` for *figure* under *out_dir* — where the render writes, named once.

    The renderer and every consumer that has to find a rendered figure afterwards ask this, so a figure's file name is derived in one place rather than re-spelled wherever it is looked up.
    """
    from tvbo.adapters import bsplot
    from tvbo.utils import sanitize_name

    name = getattr(figure, "name", None) or "figure"
    return name, out_dir / f"{name}.{bsplot.output_format(figure)}", out_dir / "scripts" / f"plot_{sanitize_name(name)}.py"


def register_figure_code(base: Path) -> None:
    """Put a root's ``code/`` on ``sys.path``, so a figure rendering against it can import the modules its ``code_modules`` names.

    Called for whichever root a figure actually renders against: its own, or — for a record ``!include``d from another study — that study's, which the including spec knows nothing about. Without it a figure resolves against the right root and still cannot find its panels.
    """
    from tvbo.utils import register_recipe_code_paths

    code_dir = study_path_for("code", Path(base))
    if code_dir.is_dir():
        register_recipe_code_paths(None, {"path": str(code_dir)})


def render_figures(figures, base_dir: Path, out_dir: Path, origins: dict[str, Path] | None = None) -> list[Path]:
    """Emit + run each figure's render script and return the written images.

    The single home for the per-figure render loop, shared by the ``tvbo figure render`` command and by a study run (which renders a study's figures after its experiments, so one command closes the replication loop). ``base_dir`` is the study root each layer's ``used`` IRI resolves against, whose results directory holds the containers; ``origins`` overrides it per figure, for a record ``!include``d from another study (see :func:`tvbo.cli.figures.figure_origins`).

    Every figure is attempted before anything is raised, so one broken declaration reports itself alongside the others rather than hiding the thirteen behind it; the run still fails, naming all of them.

    The image lands directly in ``out_dir`` — the one place the report and every other consumer reads a figure from — while its self-contained, editable ``plot_<name>.py`` goes to ``out_dir/scripts/``. Both are regenerable and gitignored together; separating them just keeps a study with many figures from interleaving twice as many files in the directory people actually browse. The subdirectory is deliberately NOT called ``code``: in a study that name means the authored, tracked, importable code the recipe references by bare module name, which this is not.
    """
    from tvbo.adapters import bsplot

    out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = out_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failed: list[str] = []
    attempted = 0
    for figure in figures:
        attempted += 1  # counted here, not re-derived: `figures` may be an iterator the loop has consumed
        name, outfile, script_path = figure_outputs(figure, out_dir)
        base = (origins or {}).get(name, base_dir)
        register_figure_code(Path(base))
        try:
            bsplot.render(figure, base_dir=str(base), outfile=str(outfile), script_path=str(script_path))
        except Exception as e:  # noqa: BLE001 — every figure is attempted, then the whole set of failures is raised at once
            failed.append(f"{name}: {type(e).__name__}: {e}")
            logger.info(f"{name} FAILED ({type(e).__name__}: {e})")
            continue
        logger.info(f"wrote {outfile}")
        logger.info(f"wrote {script_path}")
        written.append(outfile)
        # Caption partial beside the image, for `{{< include >}}` in the prose.
        try:
            cap = bsplot.write_caption(figure, out_dir, name=name) if bsplot.compose_caption(figure) else None
            if cap:
                logger.info(f"wrote {cap}")
        except Exception as e:  # noqa: BLE001 — a caption must never lose a rendered figure
            logger.info(f"caption for {name} not written ({type(e).__name__}: {e})")
    if failed:
        raise RuntimeError("{} of {} figures did not render:\n  {}".format(len(failed), attempted, "\n  ".join(failed)))
    return written


def render_study_figures(study, spec: str, *, base: Path | None = None) -> None:
    """Render a study's declarative ``figures:`` after its experiments have run.

    Reuses the exact ``tvbo figure render`` path (:func:`render_figures`), so the images and render scripts a one-command run produces are byte-identical to a follow-up ``tvbo figure render`` — the study run just fuses the two steps. ``base`` is the study root each layer's ``used`` IRI resolves against and whose figures directory the images land in, defaulting to the recipe's own directory; ``-o`` moves the containers and not the figures, so there is no results directory to pass here. A study with no ``figures:`` is a silent no-op. A render failure is reported but does not fail the run — the experiments already succeeded and their results are on disk.
    """
    from tvbo.utils import as_list

    figs = as_list(getattr(study, "figures", None))
    if not figs:
        return

    base = Path(base).resolve() if base is not None else spec_base(spec)
    out_figs = study_path_for("figures", base)
    logger.info(f"rendering {len(figs)} figure(s) -> {out_figs}")
    try:
        # Layers resolve against the study root, whose results directory is where the run and the analysis stage just wrote.
        render_figures(figs, base, out_figs)
    except Exception as e:  # noqa: BLE001 - never lose a completed run over a plotting error
        # At WARNING because the default level suppresses INFO: a failure reported below the threshold is one the run never mentions, and what follows is a reader failing on a figure that was never drawn.
        logger.warning(
            f"figure rendering failed ({type(e).__name__}: {e}); the experiment "
            f"results are saved. Re-run `tvbo figure render {spec}` to retry."
        )


# A whole study, and a study of studies


def run_study(
    study,
    spec: str,
    out_dir: Path | None = None,
    *,
    base: Path | None = None,
    options: RunOptions | None = None,
    experiment: str | None = None,
    figures: bool = True,
) -> bool:
    """Run *study*: its before-analyses, its experiments, its after-analyses and its figures, each container recording the run that wrote it.

    The one path every study run takes — `tvbo run <recipe>`, each study of a tree, and `SimulationStudy.run()` — so the flags a run honours and the provenance it records cannot depend on how it was launched.

    A run narrowed by *experiment* or by ``options.shard`` runs only those experiments: no analysis stage and no figures, since both would read a half-refreshed set of containers. An unsharded narrowed run then names the analysis containers it left stale.

    Args:
        study: The loaded study.
        spec: The recipe it was loaded from, which roots its relative paths and its provenance.
        out_dir: Results directory, in place of *base*'s.
        base: The study root this run writes into — containers land in its results directory, figures in its figures directory; defaults to the recipe's own directory.
        options: How each experiment executes and saves; the defaults run the recipe as written.
        experiment: Comma-separated ids of the experiments to run, instead of all of them.
        figures: Render the declared ``figures:`` once a whole-study run's analyses hold.

    Returns:
        Whether the after-experiment analysis stage held; figures are skipped when it did not.

    Raises:
        StudyRunError: *experiment* matches nothing, a pre-rendered script is given for several experiments, or an experiment cannot be materialised, overridden or sharded as asked.
    """
    options = options or RunOptions()
    base = Path(base).resolve() if base is not None else spec_base(spec)
    root = results_root(spec, out_dir, base=base)
    _import_figure_code_modules(study)
    before, after = _study_analysis_stages(study)
    whole = experiment is None and options.shard is None
    if whole:
        _run_study_analyses(study, before, spec, root, stage="before", out_dir=out_dir, base=base)
    records = select_experiments(study, experiment)
    # A single frozen script belongs to a single experiment; each experiment renders its own code.
    if options.rendered_code is not None and len(records) > 1:
        raise StudyRunError(
            f"--rendered is a single pre-rendered experiment script but {len(records)} "
            f"experiments matched; pass --experiment to select exactly one."
        )
    for record in records:
        logger.info(f"running experiment: {getattr(record, 'key', None) or getattr(record, 'label', None)}")
        run_experiment(runtime_experiment(study, record), spec, out_dir, options=options, owner=study, base=base)
    if not whole:
        if options.shard is None:
            # Not per shard: an array task holds one slice of one sweep, so every task would repeat the warning and its "refresh now" remedy would run on a half-done grid.
            warn_stale_analyses(before + after, spec, root, experiments={i for r in records for i in experiment_ids(r)})
        return True
    ok = _run_study_analyses(study, after, spec, root, stage="after", out_dir=out_dir, base=base)
    if figures and ok:
        render_study_figures(study, spec, base=base)
    return ok


def nested_root(base: Path, recipe_dir: Path, label: str) -> Path | None:
    """Where the study the tree knows as *label* writes, for a tree writing into *base*.

    ``None`` — the study's own directory — whenever the run has not been redirected, which is the CLI's behaviour and keeps a nested study's results in its own layout. A redirected run gives each study a subdirectory named for its label, so a run that must not write into the source tree does not write into a nested study's either. The label rather than the recipe stem, because a sub-study written inline has no recipe of its own and inherits the holding study's: naming the directory after that would land every inline sub-study of one recipe in a single directory, each overwriting the last.
    """
    return None if Path(base) == Path(recipe_dir) else Path(base) / label


def _tree_to_run(inv, skip: tuple) -> tuple[list, list[str]]:
    """The studies a run covers and the labels it drops.

    Walks the whole tree depth first, so a study's own analyses and figures run after everything they may read. A skipped label takes the studies it holds with it: dropping a group without dropping what it holds would run those studies while claiming the group was skipped.
    """
    skip_set = {s.strip() for part in skip for s in str(part).split(",") if s.strip()}
    if not skip_set:
        return list(inv.walk_studies(include_self=False)), []
    kept, dropped = [], []

    def walk(study) -> None:
        for label, nested in study.nested_studies():
            if label in skip_set:
                dropped.append(label)
                continue
            walk(nested)
            kept.append((label, nested))

    walk(inv)
    return kept, dropped


def run_tree(
    inv,
    spec: str,
    out_dir: Path | None = None,
    *,
    backend: str | None = None,
    figures: bool = True,
    skip=(),
    dry_run: bool = False,
    manifest_only: bool = False,
    base: Path | None = None,
) -> None:
    """Run a study-of-studies: every nested study depth first, then this study's own content, then the results manifest the manuscript reads.

    Each study runs through :func:`run_study` with the recipe's own settings and *backend*, in its own directory, so its results land in its own layout rather than the aggregating study's. ``skip`` drops named studies at any depth (and everything they hold); ``dry_run`` lists what would run and emits nothing; ``manifest_only`` runs nothing and emits the manifest from the containers already on disk. The manifest lands at ``<study-dir>/manuscript_results.yml`` — a committed derived artifact (the seam Quarto reads as ``{{< meta results.* >}}``), so the build never needs the generated run containers.

    ``base`` redirects everything this run WRITES, leaving what it READS alone: nested recipes still resolve against their own directories, while each one's results land in a subdirectory of ``base`` named for its label. Defaults to the recipe's directory, which is the CLI's behaviour. *out_dir* moves this study's own results only.

    Raises:
        StudyRunError: A study's analysis stage failed, or a result key does not resolve; the manifest is not written in either case.
    """
    from tvbo.data.analysis_io import analysis_name, study_analyses
    from tvbo.data.study_manifest import MANIFEST_NAME, emit_manifest
    from tvbo.utils import as_list

    recipe_dir = Path(getattr(inv, "_source_file", spec)).resolve().parent
    base = Path(base).resolve() if base is not None else recipe_dir
    to_run, skipped = _tree_to_run(inv, tuple(skip))
    own_analyses = [analysis_name(a) for a in study_analyses(inv)]
    n_results = len(as_list(getattr(inv, "results", None)))

    if dry_run:
        logger.info(f"[dry-run] study of studies: {getattr(inv, 'title', None) or spec}")
        for label, nested in to_run:
            logger.info(f"  study: {label}  ({getattr(nested, '_source_file', '?')})")
        for label in skipped:
            logger.info(f"  study skipped: {label}")
        if own_analyses:
            logger.info(f"  own analyses: {', '.join(own_analyses)}")
        logger.info(f"  own figures: {len(as_list(getattr(inv, 'figures', None)))}")
        logger.info(f"  would emit {MANIFEST_NAME} with {n_results} result key(s)")
        return

    manifest_results = results_root(spec, out_dir, base=base)

    def _emit() -> None:
        out_path, problems = emit_manifest(inv, manifest_results, base / MANIFEST_NAME)
        if problems:
            raise StudyRunError("results manifest has unresolved key(s):\n  - " + "\n  - ".join(problems))
        logger.info(f"wrote results manifest: {out_path} ({n_results} key(s))")

    if manifest_only:
        _emit()
        return

    options = RunOptions(backend=backend)
    # A failed analysis stage means the containers the manifest reads are stale or absent, so emitting from them would report a number the run did not produce. Every study is attempted first — one broken study should not hide the state of the others.
    failed: list[str] = []
    for label, nested in to_run:
        source = getattr(nested, "_source_file", None) or spec
        logger.info(f"=== study: {label} ({source}) ===")
        if not run_study(nested, str(source), base=nested_root(base, recipe_dir, label), options=options, figures=figures):
            failed.append(label)

    if not run_study(inv, spec, out_dir, base=base, options=options, figures=figures):
        failed.append(getattr(inv, "title", None) or spec)
    if failed:
        raise StudyRunError(
            "analysis stage failed for: "
            + ", ".join(failed)
            + "\nThe results manifest was NOT written; it would have reported "
            "numbers from stale or missing containers."
        )
    _emit()
