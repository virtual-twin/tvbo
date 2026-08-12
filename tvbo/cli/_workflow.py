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

import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ._backends import BackendSpec, axis_kind_of, resolve_backend


# Canonical published tvbo image. CI (``.github/workflows/docker.yml``) pushes this
# on every ``main``/``dev`` commit, tagged ``:<branch>``, ``:<version>``, ``:<sha>``
# and ``:latest`` (default branch), so a registry reference tracks the source rather
# than a local file that goes stale.
DEFAULT_CONTAINER_IMAGE = "ghcr.io/virtual-twin/tvbo"


def _default_container_tag() -> str:
    """Tag matching the running CLI, so a kit pulls the image built from its source.
    Overridable with ``TVBO_CONTAINER_TAG``."""
    from tvbo import __version__

    return os.environ.get("TVBO_CONTAINER_TAG") or __version__


def default_container_ref() -> str:
    """The tvbo container reference used when a recipe asks for container-based
    execution without pinning a concrete image.

    The tag matches the running CLI's version (see :func:`_default_container_tag`) so
    a kit runs against the image built from the same source it was emitted with.
    Every part is overridable from the environment: ``TVBO_CONTAINER`` supplies a
    full reference verbatim, otherwise ``TVBO_CONTAINER_IMAGE`` sets the repository.
    """
    full = os.environ.get("TVBO_CONTAINER")
    if full:
        return full
    image = os.environ.get("TVBO_CONTAINER_IMAGE") or DEFAULT_CONTAINER_IMAGE
    return f"docker://{image}:{_default_container_tag()}"


def resolve_container_ref(raw: Any) -> str | None:
    """Resolve a recipe's declared ``container`` into an engine-ready reference.

    A concrete image — a local ``.sif``/``.simg`` path or a registry reference that
    already carries a ``:tag`` or ``@digest`` — passes through unchanged: the author
    pinned it. Anything that leaves the version open is filled in with
    :func:`default_container_ref` so an unpinned reference pulls the version-matched
    image rather than failing to resolve:

    - the symbolic requests ``tvbo`` / ``default``;
    - a tvbo registry reference with no tag (``docker://…/tvbo``).

    No container declared ⇒ ``None``: tasks run in the surrounding environment (bare, or
    the requirements venv ``setup.sh`` provisions — see :attr:`WorkflowPlan.needs_env_layer`).
    ``requirements`` are provisioned by whichever substrate the ``container`` field selects;
    they do NOT force a container of their own.
    """
    val = str(raw or "").strip()
    if not val:
        return None
    if val in ("tvbo", "default"):
        return default_container_ref()
    if val.startswith("docker://"):
        # A tag or digest lives in the final path segment; its absence means the
        # reference names an image stream without pinning a version.
        last = val[len("docker://") :].rsplit("/", 1)[-1]
        if ":" not in last and "@" not in last:
            return f"{val}:{_default_container_tag()}"
    return val


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepAxis:
    """One sweep dimension extracted from an Exploration."""

    name: str  # short identifier used as wildcard (e.g. ``G``)
    parameter: str  # dotted path (e.g. ``ReducedWongWang.G``)
    values: tuple[float, ...]
    kind: str  # AXIS_KIND (parameters | initial_conditions | ...)
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
    engine: str  # local | slurm | snakemake | nextflow
    out_dir: str  # template; may contain ``{wildcard}``
    container: str | None
    container_binds: list[str]
    container_args: str | None
    retries: int
    rng: str
    provenance: bool

    vectorize_axes: list[SweepAxis] = field(default_factory=list)
    workflow_axes: list[SweepAxis] = field(default_factory=list)
    cohort_subjects: list[str] = field(
        default_factory=list
    )  # on_device: the whole cohort runs as ONE job producing these per-subject results
    cohort_result_files: list[str] = field(
        default_factory=list
    )  # canonical per-subject result filenames (build_result_path), one per cohort_subjects entry

    chunk: int = 1  # workflow-fanned: cells per array task;
    # fully-vectorized sweep: number of array shards
    engine_block: dict[str, Any] = field(default_factory=dict)
    overrides: list[dict[str, Any]] = field(default_factory=list)
    requirements: list[dict[str, Any]] = field(default_factory=list)  # normalized pip/conda deps
    source_spec: str = ""  # SPEC arg for `tvbo run` (recipe path / CURIE / DB name)
    experiment_selector: str | None = None  # --experiment value picking this experiment in a study
    workflow_spec: dict[str, Any] = field(
        default_factory=dict
    )  # effective merged workflow config (study < experiment < --set)
    depends_on: list[str] = field(
        default_factory=list
    )  # experiment keys whose result seeds this run (initial_state.from_experiment)

    @property
    def container_exec_flags(self) -> str:
        """``container_binds``/``container_args`` as flags for an ``exec`` call.

        Apptainer and Singularity share this command line, so the Slurm and
        Nextflow emitters (which build the exec call themselves) and the
        Snakemake emitter (which hands the same string to ``--apptainer-args``)
        render them identically. Empty when nothing is declared, so callers can
        concatenate unconditionally.

        Each bind gets its own ``--bind`` rather than joining them with the
        comma separator: a comma is not escapable inside one ``--bind``, so a
        path containing one could not be expressed at all. Paths are shell-quoted
        because the Slurm emitters interpolate this straight into a command line,
        where an unquoted space would split one bind into two arguments.
        """
        parts = ["--bind " + shlex.quote(b) for b in self.container_binds]
        if self.container_args:
            parts.append(self.container_args)
        return " ".join(parts)

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
                out.append(str(url))
                continue
            pkg = r.get("package") or r.get("name")
            if pkg:
                out.append(f"{pkg}{r.get('version_spec') or ''}")
        return out

    @property
    def needs_env_layer(self) -> bool:
        """Whether the kit must provision declared ``requirements`` into a run environment.

        ``requirements`` names what the study's code needs (e.g. a callable that imports
        ``igl``) beyond a bare tvbo. ``setup.sh`` provisions them into a
        ``--system-site-packages`` venv and each task prepends it to ``PYTHONPATH`` — pip
        resolves against the surrounding interpreter (installing only the delta) and
        compiles native wheels with it, so the layer is ABI-correct. This holds when the
        surrounding environment is a base interpreter: a conda env, or a ``container``'s
        image python (the venv is built via ``singularity exec`` on the image).

        It does NOT hold when a run ``venv`` is provided (``slurm.venv``): a
        ``--system-site-packages`` venv layered on another venv does not inherit that venv's
        packages (nested venvs don't chain), so pip re-resolves the FULL stack and its CPU
        wheels shadow the run venv's — a plain ``jaxlib`` masking ``jax[cuda]`` silently
        forces a CPU run. A provided venv IS the declared environment, so the kit trusts it
        and skips the layer (no ``setup.sh``, no ``PYTHONPATH`` prepend).
        """
        if str(self.engine_block.get("venv") or "").strip():
            return False
        return bool(self.pip_specs)

    @property
    def needs_container_layer(self) -> bool:
        """A :attr:`needs_env_layer` that layers onto a declared ``container`` specifically.

        The Slurm emitter injects the layer into the task's ``singularity exec`` via
        ``--env`` (container-only); the Snakemake emitter's plain ``PYTHONPATH`` prepend is
        substrate-agnostic and keys on :attr:`needs_env_layer` instead.
        """
        return bool(self.container and self.pip_specs)

    @property
    def container_extras_venv(self) -> str:
        """Kit-relative dir for the requirements venv (see :attr:`needs_env_layer`)."""
        return ".tvbo-extras-venv"

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
    _is_branch = (
        _ini is not None
        and str(getattr(_ini, "method", "") or "") == "from_experiment"
        and str(getattr(_ini, "source_point", "") or "endpoint") == "branch"
    )

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
            _branch_axis = _is_branch and getattr(ax, "domain", None) is None and not getattr(ax, "explored_values", None)
            out.append(
                SweepAxis(
                    name=uniq,
                    parameter=param,
                    values=() if _branch_axis else _axis_values(ax),
                    kind=axis_kind_of(param),
                    runtime_sized=_branch_axis,
                )
            )
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
    return {
        "package": get("package") or get("name"),
        "version_spec": get("version_spec"),
        "source_url": get("source_url") or get("url"),
    }


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
    src = (
        raw.items()
        if isinstance(raw, dict)
        else (((i.get("name"), i.get("value")) for i in raw if isinstance(i, dict)) if isinstance(raw, (list, tuple)) else ())
    )
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


#: Slurm's ``--mem`` suffixes, as multiples of a mebibyte. Slurm sizes are
#: binary (``--mem=8G`` reserves 8 GiB = 8192 MiB), and Snakemake's ``mem_mb``
#: is handed back to it as a bare number in the same unit — so a decimal
#: conversion would reserve ~2.4% less than the recipe asked for and OOM-kill a
#: task sized to its own declared limit.
_MEM_UNIT_MIB = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}


def mem_mb(mem) -> int | None:
    """``'8G'``/``'8GB'``/``'512M'``/``'2000'`` -> integer mebibytes.

    Feeds Snakemake's ``mem_mb`` resource, which the SLURM executor renders as
    ``--mem``. Every suffix ``sbatch --mem`` accepts is understood; an
    unrecognised one returns ``None`` rather than a wrong number, and the caller
    omits the resource. A sub-mebibyte request rounds up to 1, since 0 would
    reserve nothing.
    """
    if not mem:
        return None
    s = str(mem).strip().upper().rstrip("B")
    factor = _MEM_UNIT_MIB.get(s[-1:], None)
    try:
        if factor is not None:
            return max(1, int(float(s[:-1]) * factor))
        return max(1, int(float(s)))
    except ValueError:
        return None


def runtime_minutes(t) -> int | None:
    """A SLURM walltime -> integer minutes, for Snakemake's ``runtime`` resource.

    Accepts every spelling ``sbatch --time`` documents: ``minutes``,
    ``minutes:seconds``, ``hours:minutes:seconds``, ``days-hours``,
    ``days-hours:minutes`` and ``days-hours:minutes:seconds``. The day-prefixed
    forms matter — without them a ``3-00:00:00`` walltime parses as nothing, the
    ``runtime`` resource is omitted, and jobs silently inherit the partition
    default instead of the declared limit. Any leftover seconds round up to a
    whole minute. Returns ``None`` when unset or unparseable.
    """
    if not t:
        return None
    s = str(t).strip()
    day_part, sep, clock = s.partition("-")
    try:
        days = int(day_part) if sep else 0
        parts = [int(x) for x in (clock if sep else s).split(":")]
        if sep:  # days-hours[:minutes[:seconds]]
            hours, minutes, seconds = (parts + [0, 0])[:3]
        elif len(parts) == 3:  # hours:minutes:seconds
            hours, minutes, seconds = parts
        elif len(parts) == 2:  # minutes:seconds
            hours, minutes, seconds = 0, parts[0], parts[1]
        else:  # bare minutes
            hours, minutes, seconds = 0, parts[0], 0
        return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)
    except (ValueError, IndexError):
        return None


def _wildcard(name: str) -> str:
    """A Snakemake wildcard placeholder, doubled for the f-string that carries it.

    Every emitted path lands inside an f-string that interpolates ``OUT_DIR``; a single
    brace would make the f-string evaluate the wildcard name as a Python variable. Doubling
    leaves the literal ``{name}`` that Snakemake's ``expand()`` / ``output:`` need.
    """
    return "{{" + name + "}}"


def cell_out_relpath(ep: dict) -> str:
    """Per-cell output path under ``results/<key>/`` — exactly the file ``tvbo run`` writes.

    A dataset (subject) fan-out writes ``sub-<subject>_<stem>.h5``; an exploration fan writes
    one nested ``<axis>=<val>/…/<stem>.h5`` per cell; a group run writes ``<stem>.h5``.
    Wildcards are doubled (see :func:`_wildcard`) so the carrying f-string leaves them intact.
    This is the single source of truth for the fanned-cell path — the Snakefile's ``rule all``
    and each figure rule that depends on a fanned experiment resolve to the same pattern.
    """
    axes = ep["axes"]
    stem = ep["result_stem"]
    if len(axes) == 1 and axes[0]["parameter"] == "dataset.active_subject":
        return "sub-" + _wildcard("subject") + "_" + stem + ".h5"
    if axes:
        return "/".join("%s=%s" % (a["name"], _wildcard(a["name"])) for a in axes) + "/" + stem + ".h5"
    return stem + ".h5"


def _cohort_result_files(experiment, subjects: list[str]) -> list[str]:
    """Canonical per-subject result filenames for an on_device cohort, one per subject.

    Built through the same :func:`tvbo.adapters.bids.build_result_path` that
    :meth:`ExperimentResult.save` writes through — the subject is injected as the
    ``_active_subject`` entity, exactly as the per-subject save does — so the rule's
    declared outputs cannot drift from the files the cohort job actually produces.
    """
    from tvbo.adapters.bids import build_result_path

    saved = getattr(experiment, "_active_subject", None)
    files: list[str] = []
    try:
        for sid in subjects:
            experiment._active_subject = str(sid)
            files.append(build_result_path(experiment, extension=".h5"))
    finally:
        experiment._active_subject = saved
    return files


def cohort_out_relpaths(ep: dict) -> list[str]:
    """Per-subject output relpaths an on_device cohort job writes (one per subject).

    The whole cohort runs as ONE vectorised job that saves one result per subject (the
    same filenames the fan-out produces), so the rule declares them all as its outputs.
    The filenames are the canonical :func:`_cohort_result_files` set computed at plan
    time. Empty for a non-cohort experiment.
    """
    return list(ep.get("cohort_result_files") or [])


def fan_expand_kwargs(ep: dict) -> str:
    """``axis=EXP_<RULE>_<AXIS>`` kwargs binding an ``expand()`` to the fan's value lists.

    The value lists (``EXP_<RULE>_<AXIS> = [...]``) are emitted at the top of the Snakefile,
    so any rule in the same Snakefile (including an ``include:``-d figure rule) can reference
    them to expand a fanned experiment's whole grid of cells.
    """
    return ", ".join("%s=%s" % (a["name"], ep["rule_name"].upper() + "_" + a["name"].upper()) for a in ep["axes"])


def fan_input_expr(ep: dict) -> str:
    """A Snakemake input expression matching ALL of *ep*'s output files.

    A group run (no axes) is the single ``f"{OUT_DIR}/<key>/<stem>.h5"``; a fanned experiment
    is the ``expand()`` over its wildcard-value lists — every cell. Emitted verbatim into a
    rule's ``input:``, so a figure that reads a fanned experiment depends on its whole grid.
    """
    cohort = cohort_out_relpaths(ep)
    if cohort:
        return "[" + ", ".join('f"{OUT_DIR}/%s/%s"' % (ep["key"], r) for r in cohort) + "]"
    single = 'f"{OUT_DIR}/%s/%s"' % (ep["key"], cell_out_relpath(ep))
    if not ep["axes"]:
        return single
    return "expand(%s, %s)" % (single, fan_expand_kwargs(ep))


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
    # No explicit backend → the experiment self-selects via execution.backend
    # (a spiking network declares 'brian2'), defaulting to tvboptim.
    if not backend:
        backend = getattr(getattr(experiment, "execution", None), "backend", None) or "tvboptim"
    bk = resolve_backend(backend)

    distribute = dict(spec.get("distribute") or {})
    explicit_vec = set(distribute.get("vectorize") or [])
    explicit_wf = set(distribute.get("workflow") or [])
    by = (distribute.get("by") or "auto").lower()

    axes = extract_axes(experiment)

    # dataset.batch_mode: on_device runs the whole cohort as one job; fan_out shards per subject.
    on_device = bool(getattr(experiment, "dataset_on_device", lambda: False)())
    cohort_subjects: list[str] = []
    cohort_result_files: list[str] = []
    if on_device:
        cohort_subjects = [str(s) for s in experiment.dataset_subject_ids()]
        if not cohort_subjects:
            raise ValueError(
                f"Experiment {getattr(experiment, 'id', None)!r} sets dataset.batch_mode: on_device "
                f"but no subjects were discovered, so the single cohort job's per-subject outputs "
                f"cannot be planned. Check dataset.bids_root / dataset.subjects."
            )
        cohort_result_files = _cohort_result_files(experiment, cohort_subjects)
    else:
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
            raise ValueError(f"Axis {ax.parameter!r} is in both distribute.vectorize and distribute.workflow.")
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

    if on_device and workflow:
        _fanned = ", ".join(sorted(ax.parameter for ax in workflow))
        raise ValueError(
            f"An on_device cohort runs the whole cohort as ONE job, so it cannot also fan "
            f"a workflow axis ({_fanned}). Vectorise the axis (distribute.vectorize) so it "
            f"stays inside the single job, or set dataset.batch_mode: fan_out to shard per "
            f"subject instead."
        )

    chunk = int(distribute.get("chunk") or spec.get("chunk") or 1)
    engine_block = dict(spec.get(engine) or {})
    # Resource requirements (cpus_per_task, mem, time, env, setup, …) map across
    # engines (WorkflowEngineConfig doc): a Snakemake run on a cluster orchestrates
    # Slurm via its executor, so it needs the same partition/time/mem/cpus. When the
    # engine's own block omits them, inherit from the Slurm block (the de-facto
    # resource spec); the engine block still wins where it sets a key.
    if engine != "slurm":
        # Engine-agnostic resources map across engines (WorkflowEngineConfig doc).
        _shared = ["cpus_per_task", "mem", "time", "modules", "venv", "env", "setup"]
        # Snakemake orchestrates Slurm through its executor, so it also needs the
        # Slurm scheduler identity; other engines (Nextflow) have their own and must
        # not inherit these.
        if engine == "snakemake":
            _shared += ["partition", "account", "gres"]
        _slurm = spec.get("slurm") or {}
        for _k in _shared:
            if _k not in engine_block and _k in _slurm:
                engine_block[_k] = _slurm[_k]
    if "array_chunk" in engine_block:
        chunk = int(engine_block["array_chunk"])
    if "env" in engine_block:
        engine_block["env"] = _normalize_env(engine_block["env"])
    if "options" in engine_block:
        engine_block["options"] = _normalize_directives(engine_block["options"])
    if "setup" in engine_block:
        engine_block["setup"] = _as_lines(engine_block["setup"])

    # A 'gpu' accelerator ⇒ the scheduler must actually allocate a GPU (Slurm gres),
    # unless the workflow block already pins one; the site-specific partition stays explicit.
    _accel = str(getattr(getattr(experiment, "execution", None), "accelerator", "") or "").lower()
    if _accel == "gpu" and engine in ("slurm", "snakemake"):
        engine_block.setdefault("gres", "gpu:1")

    # Software dependencies come from the experiment's schema-native
    # environment.requirements (overridable via workflow_spec["requirements"]).
    _exp_env = getattr(experiment, "environment", None)
    _req_raw = spec.get("requirements") or (getattr(_exp_env, "requirements", None) if _exp_env is not None else None) or []
    _reqs = [r for r in (_norm_requirement(x) for x in _as_list(_req_raw)) if r.get("package") or r.get("source_url")]

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
            _sid = getattr(_src, "id", None)
            depends_on.append(str(_sid if _sid is not None else (getattr(_src, "name", None) or _src)))

    # A post-hoc experiment (e.g. Fig 4 input-statistics) reads a prior fit's
    # recorded parameters/observations by sourcing ``<study>.exp<id>`` on one of its
    # observations or parameters. That is a result dependency exactly like
    # ``from_experiment``, so record the referenced experiment id here — otherwise a
    # DAG engine schedules the analysis before the fit it consumes.
    # Match ``exp<id>`` as a whole dotted segment: the bare ref (``…exp30``) and a
    # sub-reference into a prior result (``…exp30.observations.fc``) both depend on
    # experiment 30. Anchoring on ``$`` alone would miss the dotted sub-reference.
    _exp_ref = re.compile(r"(?:^|\.)exp(\d+)(?:\.|$)")

    def _record_source_deps(container):
        _items = container.values() if hasattr(container, "values") else (container or [])
        for _it in _items:
            _src = getattr(_it, "source", None)
            if not _src:
                continue
            for _s in _src if isinstance(_src, (list, tuple)) else [_src]:
                _sn = _s if isinstance(_s, str) else (getattr(_s, "name", None) or str(_s))
                _m = _exp_ref.search(str(_sn))
                if _m and _m.group(1) != str(getattr(experiment, "id", "")) and _m.group(1) not in depends_on:
                    depends_on.append(_m.group(1))

    _record_source_deps(getattr(experiment, "observations", None))
    _dyn = getattr(experiment, "dynamics", None)
    if _dyn is not None:
        _record_source_deps(getattr(_dyn, "parameters", None))
    _record_source_deps(getattr(experiment, "parameters", None))

    # A ``used:`` DataRef (Parameter.used or an exploration-builder Argument.used) that
    # names an in-study experiment is the same result dependency: the PROV ``used`` edge
    # is the ordering edge. Record the referenced experiment id so the DAG runs it first.
    def _dep_from_used(ref):
        if ref is None:
            return
        _exp = getattr(ref, "experiment", None)
        if _exp is not None:
            _id = str(getattr(_exp, "id", _exp))
        else:
            # Same WHERE-parsing rule as the runtime resolver (dataref.locate_container):
            # only a last iri segment that *is* an experiment token (``exp-30`` / ``exp30`` /
            # ``30``) names an in-study dependency. A curated / dataset iri that merely
            # contains digits (``tvbo:dataset/HCP1200``, ``rec-avgMatrix_atlas-HCPMMP1``)
            # yields None here, so it never registers a phantom edge on a non-existent
            # experiment (which would deadlock the DAG on a rule that is never emitted).
            from tvbo.data.dataref import experiment_id

            _id = experiment_id(getattr(ref, "iri", None))
        if _id and _id != str(getattr(experiment, "id", "")) and _id not in depends_on:
            depends_on.append(_id)

    def _record_used_param_deps(container):
        _items = container.values() if hasattr(container, "values") else (container or [])
        for _it in _items:
            _dep_from_used(getattr(_it, "used", None))

    if _dyn is not None:
        _record_used_param_deps(getattr(_dyn, "parameters", None))
    _record_used_param_deps(getattr(experiment, "parameters", None))
    _net = getattr(experiment, "network", None)
    for _cpl in (
        list((getattr(_net, "coupling", None) or {}).values())
        if hasattr(getattr(_net, "coupling", None), "values")
        else _as_list(getattr(_net, "coupling", None) or [])
    ):
        _record_used_param_deps(getattr(_cpl, "parameters", None))
    # Exploration-builder arguments (ExplorationAxis.builder → Argument.used).
    _expls = getattr(experiment, "explorations", None)
    for _expl in list(_expls.values()) if hasattr(_expls, "values") else _as_list(_expls or []):
        _space = getattr(_expl, "space", None)
        for _axis in list(_space.values()) if hasattr(_space, "values") else _as_list(_space or []):
            _bargs = getattr(getattr(_axis, "builder", None), "arguments", None)
            for _barg in list(_bargs.values()) if hasattr(_bargs, "values") else _as_list(_bargs or []):
                _dep_from_used(getattr(_barg, "used", None))

    # An explicit run venv wins over a declared container, with a notice.
    _container = resolve_container_ref(spec.get("container"))
    _run_venv = str(engine_block.get("venv") or "").strip()
    if _container and _run_venv:
        from ._common import info as _info

        _info(f"slurm.venv set ({_run_venv}) → running in the venv; ignoring the declared container ({_container})")
        _container = None

    return WorkflowPlan(
        study_key=study_key,
        experiment_key=experiment_key,
        backend=bk,
        engine=engine,
        out_dir=out_dir,
        container=_container,
        container_binds=[str(b) for b in _as_list(spec.get("container_binds") or [])],
        container_args=(spec.get("container_args") or None),
        retries=int(spec.get("retries") or 0),
        rng=str(spec.get("rng") or "deterministic"),
        provenance=bool(spec.get("emit_provenance", True)),
        vectorize_axes=vectorize,
        workflow_axes=workflow,
        cohort_subjects=cohort_subjects,
        cohort_result_files=cohort_result_files,
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
    for key in ("out_dir", "container", "container_binds", "container_args", "retries", "rng", "emit_provenance", "chunk"):
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
    for key in (
        "cpus_per_task",
        "mem",
        "time",
        "gres",
        "partition",
        "account",
        "cores",
        "executor",
        "queue",
        "venv",
        "mail_type",
        "mail_user",
        "array_chunk",
    ):
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
        _as_plain_dict(getattr(experiment, "workflow", None)) if experiment is not None else {}
    )
    return _deep_merge(base, override)


def _as_plain_dict(obj) -> dict[str, Any]:
    """Convert a (possibly nested) LinkML object into a plain dict tree.

    Unset fields are dropped so an experiment's ``workflow`` block overrides only
    the keys it names when merged onto the study default. LinkML spells an unset
    scalar ``None`` and an unset multivalued slot ``[]`` — both mean "not
    declared", and both must be dropped: an experiment that overrides only its
    walltime still carries ``container_binds: []``, which would otherwise replace
    the study's binds with nothing and strip the mounts off that experiment's
    tasks. An empty container is therefore never distinguishable from an absent
    one here, so a list cannot be *cleared* by an override, only replaced.
    Always returns a dict (an empty one for ``None``).
    """
    plain = _plainify(obj)
    return plain if isinstance(plain, dict) else {}


def _unset(v) -> bool:
    """True when *v* carries no declaration — ``None``, or an empty container.

    See :func:`_as_plain_dict`: an override must not overwrite an inherited value
    with a slot its author never filled in, and LinkML gives an unfilled
    multivalued slot an empty list rather than ``None``.
    """
    return v is None or (isinstance(v, (list, tuple, dict)) and not v)


def _plainify(obj):
    """Recursively turn LinkML objects / containers into plain Python values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _plainify(v) for k, v in obj.items() if not _unset(v)}
    if isinstance(obj, (list, tuple)):
        return [_plainify(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _plainify(v) for k, v in vars(obj).items() if not k.startswith("_") and not _unset(v)}
    return obj


from tvbo.utils import deep_merge as _deep_merge, as_list as _as_list  # noqa: E402  (late-imported shared utils)
