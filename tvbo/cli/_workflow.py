"""Workflow planner.

Given a Study + Experiment + workflow spec + (resolved) backend, produce
a :class:`WorkflowPlan` describing:

* which sweep axes the backend will vectorize internally,
* which axes the workflow engine must fan out as wildcards / array tasks,
* the resulting cell count, chunking, and per-cell command line.

The planner is intentionally backend-aware. It consults
:mod:`tvbo.cli._backends` (mirrored from ``ontology/tvb-o-axioms.ttl``)
so the same ``study.yaml`` produces a *different* DAG when re-rendered
against a different backend — exactly as §4.10.1 of ``dev/tvbo-cli.md``
requires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ._backends import BackendSpec, axis_kind_of, resolve_backend


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepAxis:
    """One sweep dimension extracted from an Exploration."""
    name: str               # short identifier used as wildcard (e.g. ``G``)
    parameter: str          # dotted path (e.g. ``ReducedWongWang.G``)
    values: tuple[float, ...]
    kind: str               # AXIS_KIND (parameters | initial_conditions | ...)
    placement: str = "auto"  # auto | vectorize | workflow
    # A branch-restart axis (initial_state source_point='branch') whose cell count is
    # known only at run time, from the source run's recorded branch. It fans into a
    # fixed number of array shards (``chunk``); each task slices its share of the loaded
    # branch. ``values`` is empty and ``n`` is unknown until the source result is read.
    runtime_sized: bool = False

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class WorkflowPlan:
    """Result of planning. Drives all emitters."""
    study_key: str
    experiment_key: str
    backend: BackendSpec
    engine: str                       # local | slurm | snakemake | nextflow
    out_dir: str                      # template; may contain ``{wildcard}``
    container: str | None
    retries: int
    rng: str
    provenance: bool

    vectorize_axes: list[SweepAxis] = field(default_factory=list)
    workflow_axes: list[SweepAxis] = field(default_factory=list)

    chunk: int = 1                    # workflow-fanned: cells per array task;
                                      # fully-vectorized sweep: number of array shards
    engine_block: dict[str, Any] = field(default_factory=dict)
    overrides: list[dict[str, Any]] = field(default_factory=list)
    requirements: list[dict[str, Any]] = field(default_factory=list)  # normalized pip/conda deps
    source_spec: str = ""             # SPEC arg for `tvbo run` (recipe path / CURIE / DB name)
    experiment_selector: str | None = None  # --experiment value picking this experiment in a study
    workflow_spec: dict[str, Any] = field(default_factory=dict)  # effective merged workflow config (study < experiment < --set)
    depends_on: list[str] = field(default_factory=list)  # experiment keys whose result seeds this run (initial_state.from_experiment)

    # ---- derived helpers --------------------------------------------------

    @property
    def n_workflow_cells(self) -> int:
        n = 1
        for ax in self.workflow_axes:
            n *= ax.n
        return n

    @property
    def n_vectorize_cells(self) -> int:
        n = 1
        for ax in self.vectorize_axes:
            # Runtime-sized (branch-restart) axes have an unknown cell count at plan
            # time; they don't contribute a static factor to the vectorised total.
            if getattr(ax, "runtime_sized", False):
                continue
            n *= ax.n
        return n

    @property
    def n_array_tasks(self) -> int:
        if self.workflow_axes:
            # Fanned axes → one array task per ``chunk`` workflow cells.
            return max(1, (self.n_workflow_cells + self.chunk - 1) // self.chunk)
        # A runtime-sized (branch-restart) sweep is fanned into exactly ``chunk`` array
        # shards: the cell count is known only when the source branch is read, so each
        # task slices its share of the loaded branch (``_branch_p[i::N]``). No cell-count
        # cap, because there is no static count to cap against.
        if any(getattr(ax, "runtime_sized", False) for ax in self.vectorize_axes):
            return max(1, self.chunk)
        # Fully backend-vectorized sweep (no fanned axes): ``chunk`` is the number
        # of SLURM array shards. Each task runs ``tvbo run --slurm-chunk=$i/N`` over
        # 1/N of the sweep cells (the backend vmap/pmap-s its own share). Capped at
        # the cell count so we never emit more tasks than there are cells.
        return max(1, min(self.chunk, self.n_vectorize_cells))

    @property
    def pip_specs(self) -> list[str]:
        """Requirements as pip-installable strings (``pkg>=x`` or a source URL)."""
        out: list[str] = []
        for r in self.requirements:
            url = r.get("source_url") or r.get("url")
            if url:
                out.append(str(url)); continue
            pkg = r.get("package") or r.get("name")
            if pkg:
                out.append(f"{pkg}{r.get('version_spec') or ''}")
        return out

    @property
    def run_spec(self) -> str:
        """SPEC argument for ``tvbo run`` — the source recipe path/CURIE if known,
        else the ``experiment:<key>`` fallback."""
        return self.source_spec or f"experiment:{self.experiment_key}"

    @property
    def wildcards(self) -> list[str]:
        return [ax.name for ax in self.workflow_axes]

    def cell_iter(self) -> Iterable[dict[str, float]]:
        """Iterate over the cartesian product of *workflow* axes."""
        if not self.workflow_axes:
            yield {}
            return
        from itertools import product
        names = [ax.name for ax in self.workflow_axes]
        for combo in product(*(ax.values for ax in self.workflow_axes)):
            yield dict(zip(names, combo))


# ---------------------------------------------------------------------------
# Sweep extraction
# ---------------------------------------------------------------------------

def _axis_values(ax) -> tuple[float, ...]:
    """Materialise an ExplorationAxis into a tuple of values."""
    vals = list(getattr(ax, "explored_values", None) or [])
    if vals:
        return tuple(float(v) for v in vals)
    dom = getattr(ax, "domain", None)
    if dom is None:
        return ()
    lo = float(getattr(dom, "lo", 0.0))
    hi = float(getattr(dom, "hi", 0.0))
    n = int(getattr(dom, "n", 1) or 1)
    log = bool(getattr(dom, "log_scale", False))
    if n <= 1:
        return (lo,)
    if log:
        return tuple(float(v) for v in np.logspace(np.log10(lo), np.log10(hi), n))
    return tuple(float(v) for v in np.linspace(lo, hi, n))


def _short_name(parameter: str) -> str:
    """Pick a wildcard-friendly short name from a dotted parameter path."""
    return parameter.rsplit(".", 1)[-1] or parameter.replace(".", "_")


def extract_axes(experiment) -> list[SweepAxis]:
    """Collect every ExplorationAxis declared on *experiment*.

    The axis kind is inferred from the parameter path (see
    :func:`tvbo.cli._backends.axis_kind_of`); placement defaults to
    ``"auto"`` and is resolved by :func:`plan`.
    """
    explorations = getattr(experiment, "explorations", None) or {}
    if hasattr(explorations, "values"):
        explorations = list(explorations.values())
    else:
        explorations = list(explorations)

    # A from_experiment:branch experiment restarts an analysis over a sibling run's whole
    # recorded branch: its exploration axes carry no domain (values come from the branch at
    # run time), so they are runtime-sized shard axes rather than statically-valued grids.
    _ini = getattr(experiment, "initial_state", None)
    _is_branch = (_ini is not None
                  and str(getattr(_ini, "method", "") or "") == "from_experiment"
                  and str(getattr(_ini, "source_point", "") or "endpoint") == "branch")

    out: list[SweepAxis] = []
    seen: set[str] = set()
    for expl in explorations:
        for ax in _as_list(getattr(expl, "space", None)):
            param = str(getattr(ax, "parameter", "") or "")
            if not param:
                continue
            name = _short_name(param)
            # Disambiguate clashing wildcard names.
            uniq = name
            i = 2
            while uniq in seen:
                uniq = f"{name}{i}"
                i += 1
            seen.add(uniq)
            _branch_axis = (_is_branch and getattr(ax, "domain", None) is None
                            and not getattr(ax, "explored_values", None))
            out.append(SweepAxis(
                name=uniq,
                parameter=param,
                values=() if _branch_axis else _axis_values(ax),
                kind=axis_kind_of(param),
                runtime_sized=_branch_axis,
            ))
    return out


def _dataset_subject_axis(experiment) -> "SweepAxis | None":
    """A workflow-fanned ``subject`` axis when the experiment has a per-subject target.

    Values are the cohort subject IDs (from ``experiment.dataset_subject_ids()``);
    each fanned cell runs ``tvbo run … --subject <sub>`` so the run resolves that
    subject's empirical target. Returns ``None`` when the experiment declares no
    dataset-sourced observation.
    """
    ids_fn = getattr(experiment, "dataset_subject_ids", None)
    if not callable(ids_fn):
        return None
    try:
        subjects = list(ids_fn())
    except Exception:
        return None
    if not subjects:
        return None
    return SweepAxis(
        name="subject",
        parameter="dataset.active_subject",
        values=tuple(str(s) for s in subjects),
        kind="subjects",
        placement="workflow",
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _norm_requirement(item) -> dict[str, Any]:
    """Normalize a dep (``'libigl>=2.5'`` string, a dict, or a SoftwareRequirement
    object) into ``{package, version_spec, source_url}`` for the env-file emitters."""
    if item is None:
        return {}
    if isinstance(item, str):
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$", item)
        return {"package": m.group(1), "version_spec": (m.group(2).strip() or None)} if m else {}
    get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
    return {"package": get("package") or get("name"),
            "version_spec": get("version_spec"),
            "source_url": get("source_url") or get("url")}


def _normalize_env(raw) -> list[dict[str, str]]:
    """Canonicalise an engine block's ``env`` into a shell-ready list.

    Accepts the YAML list form ``[{name, value}]`` and the mapping form
    ``{NAME: value}`` produced by ``--set slurm.env.NAME=value``. Booleans lower
    to ``true``/``false``; every value is shell-quoted so the template can emit
    ``export NAME=value`` verbatim without branching on shape or escaping.
    """
    import shlex

    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        for name, value in raw.items():
            # schema-inlined EnvironmentVariable carries {name, value}; --set is scalar.
            pairs.append((name, value.get("value") if isinstance(value, dict) else value))
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                pairs.append((item.get("name"), item.get("value")))
    out: list[dict[str, str]] = []
    for name, value in pairs:
        if name is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        out.append({"name": str(name), "value": shlex.quote(str(value))})
    return out


def _pairs_to_map(items) -> dict[str, Any]:
    """Flatten a name-keyed slot (YAML ``[{name, value}]`` list or schema/CLI map)
    into a plain ``{name: value}`` dict."""
    if isinstance(items, dict):
        return {k: (v.get("value") if isinstance(v, dict) else v) for k, v in items.items()}
    out: dict[str, Any] = {}
    if isinstance(items, (list, tuple)):
        for item in items:
            if isinstance(item, dict) and item.get("name") is not None:
                out[item["name"]] = item.get("value")
    return out


#: Engine-block slots that are name-keyed lists in YAML but must merge by name.
_ENGINE_MAP_SLOTS = ("env", "options")


def _canonicalize_engine_maps(block: dict) -> dict:
    """Rewrite each engine block's name-keyed slots (env, options) to maps in place.

    The YAML author writes ``env: [{name, value}]`` (a list) while ``--set
    slurm.env.X=v`` yields a mapping; representing both as a name-keyed map lets
    the workflow merge (study < experiment < --set) override single entries by
    name instead of replacing the whole list. The plan later lowers each map back
    to a list via :func:`_normalize_env` / :func:`_normalize_directives`.
    """
    for engine in ("slurm", "snakemake", "nextflow"):
        eng = block.get(engine)
        if isinstance(eng, dict):
            for slot in _ENGINE_MAP_SLOTS:
                if slot in eng:
                    eng[slot] = _pairs_to_map(eng[slot])
    return block


def _normalize_directives(raw) -> list[dict[str, str]]:
    """Canonicalise an engine block's ``options`` into a ``[{name, value}]`` list.

    Same name-keyed shapes as :func:`_normalize_env`, but the values are scheduler
    directive tokens (e.g. a Slurm ``#SBATCH --<name>=<value>`` line), not shell
    words, so they are emitted verbatim rather than shell-quoted.
    """
    src = raw.items() if isinstance(raw, dict) else (
        ((i.get("name"), i.get("value")) for i in raw if isinstance(i, dict))
        if isinstance(raw, (list, tuple)) else ())
    out: list[dict[str, str]] = []
    for name, value in src:
        if name is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        out.append({"name": str(name), "value": str(value)})
    return out


def _as_lines(raw) -> list[str]:
    """Normalize a shell-line field (``setup``) to a list of strings.

    A string (or any scalar) becomes a single line; a list/tuple is stringified
    per element. So ``--set slurm.setup="conda activate env"`` yields one line, not
    one line per character, and a bare scalar does not raise.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def plan(
    *,
    study_key: str,
    experiment,
    backend: str,
    engine: str = "local",
    workflow_spec: dict[str, Any] | None = None,
    overrides: list[dict[str, Any]] | None = None,
    source_spec: str = "",
    experiment_selector: str | None = None,
) -> WorkflowPlan:
    """Compute a :class:`WorkflowPlan` from an Experiment + spec.

    *workflow_spec* mirrors the Study-level ``workflow:`` block from
    ``study.yaml`` (§4.10.1). Missing keys use sensible defaults.
    """
    spec = dict(workflow_spec or {})
    bk = resolve_backend(backend)

    distribute = dict(spec.get("distribute") or {})
    explicit_vec = set(distribute.get("vectorize") or [])
    explicit_wf = set(distribute.get("workflow") or [])
    by = (distribute.get("by") or "auto").lower()

    axes = extract_axes(experiment)

    # A per-subject empirical target adds an implicit "subject" fan-out axis: one
    # shard per subject, each resolving its own target. It reuses the same
    # wildcard/array machinery as sweep axes, so no separate per-subject path.
    subject_axis = _dataset_subject_axis(experiment)
    if subject_axis is not None:
        axes = [subject_axis, *axes]

    vectorize: list[SweepAxis] = []
    workflow: list[SweepAxis] = []
    for ax in axes:
        # An axis constructed with an explicit placement (e.g. the subject
        # fan-out) is honoured as-is rather than re-decided by the auto rule.
        if ax.placement in ("workflow", "vectorize"):
            (workflow if ax.placement == "workflow" else vectorize).append(ax)
            continue
        forced_vec = ax.parameter in explicit_vec or ax.name in explicit_vec
        forced_wf = ax.parameter in explicit_wf or ax.name in explicit_wf
        if forced_vec and forced_wf:
            raise ValueError(
                f"Axis {ax.parameter!r} is in both distribute.vectorize and distribute.workflow."
            )
        if forced_vec:
            if not bk.can_vectorize(ax.kind):
                raise ValueError(
                    f"Backend {bk.name!r} cannot vectorize axis kind {ax.kind!r} "
                    f"(parameter {ax.parameter!r}). Drop it from distribute.vectorize "
                    f"or pick a different backend."
                )
            vectorize.append(SweepAxis(**{**ax.__dict__, "placement": "vectorize"}))
            continue
        if forced_wf:
            workflow.append(SweepAxis(**{**ax.__dict__, "placement": "workflow"}))
            continue
        # auto / default
        if by == "workflow" or not bk.can_vectorize(ax.kind):
            workflow.append(SweepAxis(**{**ax.__dict__, "placement": "workflow"}))
        else:
            vectorize.append(SweepAxis(**{**ax.__dict__, "placement": "vectorize"}))

    chunk = int(distribute.get("chunk") or spec.get("chunk") or 1)
    engine_block = dict(spec.get(engine) or {})
    if "array_chunk" in engine_block:
        chunk = int(engine_block["array_chunk"])
    if "env" in engine_block:
        engine_block["env"] = _normalize_env(engine_block["env"])
    if "options" in engine_block:
        engine_block["options"] = _normalize_directives(engine_block["options"])
    if "setup" in engine_block:
        engine_block["setup"] = _as_lines(engine_block["setup"])

    # Software dependencies come from the experiment's schema-native
    # environment.requirements (overridable via workflow_spec["requirements"]).
    _exp_env = getattr(experiment, "environment", None)
    _req_raw = (spec.get("requirements")
                or (getattr(_exp_env, "requirements", None) if _exp_env is not None else None)
                or [])
    _reqs = [r for r in (_norm_requirement(x) for x in _as_list(_req_raw))
             if r.get("package") or r.get("source_url")]

    from ._common import experiment_key as _experiment_key  # canonical (id-first) key

    experiment_key = _experiment_key(experiment)
    # Results land in a kit-relative ``results/`` by default (the emitted scripts
    # run from the kit dir, which already encodes study/experiment/engine — like
    # ``logs/``). An explicit out_dir (relative or absolute) overrides it; the
    # {study}/{experiment} placeholders still resolve for custom templates.
    out_dir = str(spec.get("out_dir") or "results")
    out_dir = out_dir.replace("{study}", study_key).replace("{experiment}", experiment_key)

    # A ``from_experiment`` initial state makes this experiment depend on another
    # experiment's completed result (its operating point). Recorded as an ordering
    # edge so DAG engines run the source first (Snakemake input; SLURM afterok).
    depends_on: list[str] = []
    _ini = getattr(experiment, "initial_state", None)
    if _ini is not None and str(getattr(_ini, "method", "") or "") == "from_experiment":
        _src = getattr(_ini, "source_experiment", None)
        if _src is not None:
            # ``source_experiment`` is referenced by identifier; keep the raw
            # reference (id, key, or name) as a string. The emitter resolves it to
            # the source's canonical workflow key, so a non-numeric key/name here
            # does not crash and an explicit ``key`` still matches its rule/output.
            depends_on.append(str(getattr(_src, "id", None) or getattr(_src, "name", None) or _src))

    return WorkflowPlan(
        study_key=study_key,
        experiment_key=experiment_key,
        backend=bk,
        engine=engine,
        out_dir=out_dir,
        container=(spec.get("container") or None),
        retries=int(spec.get("retries") or 0),
        rng=str(spec.get("rng") or "deterministic"),
        provenance=bool(spec.get("provenance", True)),
        vectorize_axes=vectorize,
        workflow_axes=workflow,
        chunk=max(1, chunk),
        engine_block=engine_block,
        overrides=list(overrides or []),
        requirements=_reqs,
        source_spec=source_spec or "",
        experiment_selector=experiment_selector,
        workflow_spec=spec,
        depends_on=depends_on,
    )


def workflow_config_from_spec(spec: dict) -> Any:
    """Rebuild a datamodel ``WorkflowConfig`` from the merged workflow spec dict.

    Lets an emitted kit freeze the *effective* configuration (study < experiment
    < ``--set``) into its spec, so the spec re-emits identically without the flags
    being re-supplied — full, self-contained provenance. Returns ``None`` when the
    spec carries no workflow settings.
    """
    from tvbo import datamodel as dm

    if not spec:
        return None
    wc = dm.WorkflowConfig()
    for key in ("out_dir", "container", "retries", "rng", "provenance", "chunk"):
        if spec.get(key) is not None:
            setattr(wc, key, spec[key])
    dist = spec.get("distribute")
    if isinstance(dist, dict):
        dc = dm.DistributionConfig()
        for key in ("by", "chunk"):
            if dist.get(key) is not None:
                setattr(dc, key, dist[key])
        if dist.get("vectorize"):
            dc.vectorize = list(dist["vectorize"])
        if dist.get("workflow"):
            dc.workflow = list(dist["workflow"])
        wc.distribute = dc
    for engine in ("slurm", "snakemake", "nextflow"):
        blk = spec.get(engine)
        if isinstance(blk, dict) and blk:
            setattr(wc, engine, _engine_config_from_dict(blk))
    return wc


def _engine_config_from_dict(blk: dict) -> Any:
    """Rebuild a ``WorkflowEngineConfig`` from a merged engine block (env/options as
    name-keyed maps or lists, values raw so they re-quote cleanly on the next emit)."""
    from tvbo import datamodel as dm

    ec = dm.WorkflowEngineConfig()
    for key in ("cpus_per_task", "mem", "time", "gres", "partition", "account",
                "cores", "executor", "queue", "venv", "mail_type", "mail_user",
                "array_chunk"):
        if blk.get(key) is not None:
            setattr(ec, key, blk[key])
    if blk.get("modules"):
        ec.modules = list(blk["modules"])
    if blk.get("setup"):
        ec.setup = _as_lines(blk["setup"])
    env_map = _pairs_to_map(blk.get("env")) if blk.get("env") else {}
    if env_map:
        ec.env = [dm.EnvironmentVariable(name=str(n), value=str(v)) for n, v in env_map.items()]
    opt_map = _pairs_to_map(blk.get("options")) if blk.get("options") else {}
    if opt_map:
        ec.options = [dm.SchedulerDirective(name=str(n), value=str(v)) for n, v in opt_map.items()]
    return ec


def merge_workflow_spec(study, experiment=None) -> dict[str, Any]:
    """Merge the study's ``workflow`` defaults with an experiment's ``workflow``.

    The experiment block refines the study block: only the fields it sets take
    precedence, the rest are inherited. Pass the experiment object directly — it
    need not carry a ``key``. With no experiment, only the study block is returned.
    """
    base = _canonicalize_engine_maps(_as_plain_dict(getattr(study, "workflow", None)))
    override = _canonicalize_engine_maps(
        _as_plain_dict(getattr(experiment, "workflow", None))
        if experiment is not None else {})
    return _deep_merge(base, override)


def _as_plain_dict(obj) -> dict[str, Any]:
    """Convert a (possibly nested) LinkML object into a plain dict tree.

    Unset scalar fields (``None``) are dropped so an experiment's ``workflow``
    block overrides only the keys it names when merged onto the study default;
    empty multivalued fields serialize as ``[]`` and are harmless. Always returns
    a dict (an empty one for ``None``).
    """
    plain = _plainify(obj)
    return plain if isinstance(plain, dict) else {}


def _plainify(obj):
    """Recursively turn LinkML objects / containers into plain Python values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _plainify(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_plainify(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _plainify(v) for k, v in vars(obj).items()
                if not k.startswith("_") and v is not None}
    return obj


from tvbo.utils import deep_merge as _deep_merge, as_list as _as_list  # noqa: E402  (late-imported shared utils)
