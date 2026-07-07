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
            n *= ax.n
        return n

    @property
    def n_array_tasks(self) -> int:
        if self.workflow_axes:
            # Fanned axes → one array task per ``chunk`` workflow cells.
            return max(1, (self.n_workflow_cells + self.chunk - 1) // self.chunk)
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

    def per_cell_command(self, *, run_cmd: str = "tvbo run") -> str:
        """Render a `tvbo run` command line for one workflow cell."""
        spec = f"experiment:{self.experiment_key}"
        parts = [run_cmd, spec, f"--backend={self.backend.name}"]
        if self.container:
            parts.append(f"--container={self.container}")
        # Workflow-fanned axes become explicit --override flags
        for ax in self.workflow_axes:
            parts.append(f"--override={ax.parameter}={{wildcards.{ax.name}}}")
        # Vectorized axes are passed as a sweep range so the backend packs them
        for ax in self.vectorize_axes:
            parts.append(f"--sweep={ax.parameter}={','.join(repr(v) for v in ax.values)}")
        parts.append(f"-o {self.out_dir}")
        return " ".join(parts)


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
            out.append(SweepAxis(
                name=uniq,
                parameter=param,
                values=_axis_values(ax),
                kind=axis_kind_of(param),
            ))
    return out


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

    vectorize: list[SweepAxis] = []
    workflow: list[SweepAxis] = []
    for ax in axes:
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

    # Software dependencies come from the experiment's schema-native
    # environment.requirements (overridable via workflow_spec["requirements"]).
    _exp_env = getattr(experiment, "environment", None)
    _req_raw = (spec.get("requirements")
                or (getattr(_exp_env, "requirements", None) if _exp_env is not None else None)
                or [])
    _reqs = [r for r in (_norm_requirement(x) for x in _as_list(_req_raw))
             if r.get("package") or r.get("source_url")]

    return WorkflowPlan(
        study_key=study_key,
        experiment_key=str(getattr(experiment, "key", None) or getattr(experiment, "name", None) or "experiment"),
        backend=bk,
        engine=engine,
        out_dir=str(spec.get("out_dir") or "out/{study}/{experiment}"),
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
    )


def merge_workflow_spec(study, experiment_key: str | None = None) -> dict[str, Any]:
    """Merge ``study.workflow`` with ``experiment.workflow_overrides``.

    Returns the effective spec dict for the named experiment. When
    *experiment_key* is None, only the Study-level block is returned.
    """
    base = _as_plain_dict(getattr(study, "workflow", None))
    if experiment_key is None:
        return base
    exps = getattr(study, "experiments", None) or getattr(study, "simulation_experiments", None) or []
    items = list(exps.values()) if hasattr(exps, "values") else list(exps)
    for e in items:
        ek = getattr(e, "key", None) or getattr(e, "name", None) or getattr(e, "label", None)
        if ek == experiment_key:
            override = _as_plain_dict(getattr(e, "workflow_overrides", None))
            return _deep_merge(base, override)
    return base


def _as_plain_dict(obj) -> dict[str, Any]:
    """Best-effort conversion of LinkML-ish objects into plain dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _as_plain_dict(v) if hasattr(v, "__dict__") and not isinstance(v, (str, int, float, bool)) else v
                for k, v in obj.items()}
    if hasattr(obj, "_as_dict"):
        return obj._as_dict
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


from tvbo.utils import deep_merge as _deep_merge, as_list as _as_list  # noqa: E402  (late-imported shared utils)
