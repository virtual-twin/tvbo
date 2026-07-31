"""Run a study's declarative ``analyses:`` and persist each as its own container.

A study reports quantities that no simulation produced — a basis decomposition of
empirical maps, a spectrum, a permutation test — and reductions that span several
experiments. ``SimulationStudy.analyses`` declares each as the same ``FunctionCall`` a
``Parameter.producer`` uses, and this module is what executes one: resolve its
arguments (a literal ``value``, or a ``used:`` DataRef reading an experiment, another
analysis, or a dataset), call it, and write the result to
``<results_root>/results/<name>/result.h5`` beside a ``result.yaml`` provenance sidecar.

That path is the container convention every consumer already resolves, so a figure
layer binds an analysis exactly as it binds a run. Arrays are written through xarray
untouched: an invocation that returns labelled ``DataArray``s keeps its dims and
coordinates, which is what lets a grammar panel name them in an ``encoding``.

Which backend renders an analysis is declared, not assumed — ``Analysis.execution``
carries the same choice a ``SimulationExperiment`` makes. :data:`RENDERERS` maps a
backend name to the renderer that runs it; a backend with no renderer raises rather
than being quietly run by another.

The ``used:`` edges also order the work. :func:`schedule` splits the analyses into
those that run before the experiments (an experiment may source one through a parameter)
and those that run after (they read an experiment's result), each topologically ordered.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tvbo.data import dataref as _dref
from tvbo.utils import as_list

__all__ = [
    "RENDERERS",
    "analysis_closure",
    "container_path",
    "dependencies",
    "dependents_of",
    "dependents_of_experiments",
    "render_inprocess",
    "run_analyses",
    "run_analysis",
    "schedule",
    "study_analyses",
]


def _slot(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or mapping item ``name``, tolerating both object and dict spellings."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    return default if value is None else value


def _name(analysis) -> str:
    """The analysis's declared name — its container key and its identity in a ``used:``."""
    name = _slot(analysis, "name")
    if not name:
        raise ValueError(
            "A study `analyses:` entry needs a `name:` — it keys the analysis's result "
            "container and is how a figure or another analysis references it."
        )
    return str(name)


def _arg_items(arguments) -> list[tuple[str, Any]]:
    """``(name, Argument)`` pairs from the keyed dict or the list spelling."""
    if not arguments:
        return []
    if hasattr(arguments, "items"):
        return [(str(k), v) for k, v in arguments.items()]
    return [(str(_slot(a, "name")), a) for a in arguments]


# --------------------------------------------------------------------------- WHERE


def container_path(name: str, results_root=None) -> Path:
    """Path an analysis named ``name`` writes, whether or not it exists yet.

    The layout itself is defined once, by
    :func:`tvbo.data.dataref.locate_analysis_container`; this is its write-side
    counterpart, which must not raise on a container that has not been produced.
    """
    return _dref.analysis_container_path(results_root, name)


# ---------------------------------------------------------------------------- order


def dependencies(analysis) -> dict:
    """What this analysis consumes, read off its arguments' ``used:`` edges.

    Returns ``{"experiments": {...}, "analyses": {...}}`` — the ids/names that must have
    run before it. A literal argument contributes nothing.

    An ``experiment:`` edge contributes BOTH the spelling the recipe used and its bare
    numeric id, because a recipe may write ``exp-3``, ``exp3`` or ``3`` for the same
    experiment while the runtime identifies it by ``{key, name, label, id}``. Keeping only
    the literal string makes every dotted spelling silently match nothing — an empty stale
    set reads exactly like a clean one.
    """
    exps: set[str] = set()
    anas: set[str] = set()
    for _, arg in _arg_items(_slot(analysis, "arguments")):
        used = _slot(arg, "used")
        if used is None:
            continue
        exp = _slot(used, "experiment")
        if exp is not None:
            exps.add(str(exp))
            eid = _dref.experiment_id(exp)
            if eid is not None:
                exps.add(eid)
        ana = _slot(used, "analysis")
        if ana is not None:
            anas.add(str(ana))
        iri = _slot(used, "iri")
        if iri is not None:
            eid = _dref.experiment_id(iri)
            if eid is not None:
                exps.add(str(eid))
    return {"experiments": exps, "analyses": anas}


def dependents_of(analyses, *, experiments=(), changed_analyses=()) -> list[str]:
    """Every analysis downstream of the given experiments or analyses, in declaration order.

    An analysis container records no link back to the run it was derived from, so re-running
    part of a study leaves each downstream container holding the PREVIOUS run's numbers with
    nothing to raise. This names the set to invalidate, and it must be transitive: the
    second-order analyses (a landscape built from a per-cell reduction, a correlation built
    from that) are exactly the ones a hand-written list forgets.

    The seed is either kind of node, because both partial-run modes exist: ``--experiment``
    re-runs a simulation, ``--analysis`` re-runs a derivation, and what goes stale downstream
    is found by the same walk over the study's own ``used:`` edges. Analyses named in
    ``changed_analyses`` come back in the result — a caller that just ran them drops them.
    """
    wanted = {str(e) for e in experiments}
    stale = {str(a) for a in changed_analyses}
    items = as_list(analyses)
    changed = True
    while changed:
        changed = False
        for analysis in items:
            name = _name(analysis)
            if name in stale:
                continue
            deps = dependencies(analysis)
            if deps["experiments"] & wanted or deps["analyses"] & stale:
                stale.add(name)
                changed = True
    return [_name(a) for a in items if _name(a) in stale]


def dependents_of_experiments(analyses, experiment_ids) -> list[str]:
    """Every analysis that reads those experiments, transitively, in declaration order."""
    return dependents_of(analyses, experiments=experiment_ids)


def analysis_closure(analyses, names, exists) -> set[str]:
    """The named analyses plus the upstream ones that have no container yet.

    The upstream counterpart of :func:`dependents_of`, and it belongs beside it: both walk
    the study's ``used:`` edges, and a caller that needs one usually needs the other.

    Asking for an analysis by name means asking for it to be RE-derived, so its own inputs
    are left alone wherever they already exist — re-running them would be the whole study,
    which is what asking by name is an alternative to. An input that has never been produced
    is different: it cannot be read, so it is pulled in, and transitively, because the first
    missing container's own inputs may be missing too.

    Args:
        analyses: The study's declared analyses.
        names: The names asked for.
        exists: Predicate saying whether an analysis's container is already on disk.
    """
    by_name = {_name(a): a for a in as_list(analyses)}
    needed, queue = set(names), list(names)
    while queue:
        for dep in dependencies(by_name[queue.pop()])["analyses"]:
            if dep in by_name and dep not in needed and not exists(dep):
                needed.add(dep)
                queue.append(dep)
    return needed


def _toposort(analyses) -> list:
    """Analyses in dependency order, declaration order breaking every tie.

    Also enforces the two things a name-keyed container makes structural: names are
    unique (two analyses sharing one would overwrite each other's result), and every
    ``used: {analysis: …}`` names one that exists. Raises on a cycle, listing the
    analyses left unresolved.
    """
    items = list(analyses)
    names = [_name(a) for a in items]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"study `analyses:` reuse the name(s) {duplicates} — a name keys the "
            "analysis's result container, so duplicates would overwrite each other."
        )
    known = set(names)
    for a in items:
        missing = sorted(dependencies(a)["analyses"] - known)
        if missing:
            raise KeyError(
                f"analysis {_name(a)!r} sources {missing} through a `used: {{analysis: …}}`, "
                f"which this study does not declare (have {sorted(known)})."
            )

    pending = list(items)
    done: set[str] = set()
    ordered: list = []
    while pending:
        ready = [a for a in pending if dependencies(a)["analyses"] <= done]
        if not ready:
            raise ValueError(
                "study `analyses:` have a circular `used:` dependency among "
                f"{sorted(_name(a) for a in pending)} — an analysis cannot consume its "
                "own result, directly or through a chain."
            )
        for a in ready:
            ordered.append(a)
            done.add(_name(a))
            pending.remove(a)
    return ordered


def schedule(analyses) -> tuple[list, list]:
    """Split the analyses into ``(before_experiments, after_experiments)``.

    An analysis that reads an experiment result runs after the experiments, and so does
    everything downstream of it. The rest run first, so an experiment's own parameter can
    source one through a ``used:``. Each list is topologically ordered.
    """
    ordered = _toposort(analyses)
    deferred: set[str] = set()
    for a in ordered:
        deps = dependencies(a)
        if deps["experiments"] or (deps["analyses"] & deferred):
            deferred.add(_name(a))
    before = [a for a in ordered if _name(a) not in deferred]
    after = [a for a in ordered if _name(a) in deferred]
    return before, after


# --------------------------------------------------------------------------- execute


def _import_attr(module: str, attr: str, name: str):
    """``module.attr``, with an error that says how a study makes its code importable."""
    if not module or not attr:
        raise ValueError(
            f"analysis {name!r}: needs both `module` and `name` "
            f"(got module={module!r}, name={attr!r})."
        )
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"analysis {name!r}: cannot import module {module!r} ({e}). The study's code "
            "directory is put on the path when the recipe loads — check `code_source:`, "
            "or that the module sits in `code/` beside the recipe."
        ) from e
    obj = getattr(mod, attr, None)
    if obj is None:
        raise AttributeError(f"analysis {name!r}: {module} has no {attr!r}.")
    return obj


def render_inprocess(analysis, kwargs):
    """Renderer that performs the declared invocation in this interpreter.

    ``callable:`` calls the function with the resolved arguments. ``class_call:``
    instantiates the class with its ``constructor_args`` and calls the instance with the
    resolved arguments, so an analysis carried by a class (a fitted estimator, a stateful
    reduction) declares the same way a monitor does. The ``function:`` and ``equation:``
    forms are refused rather than half-executed — they describe a step inside an
    observation pipeline, whose inputs come from a running solve, not a container.

    It imposes no array library: whatever the declared code uses is what runs. The
    written container is xarray either way.
    """
    name = _name(analysis)

    call = _slot(analysis, "callable")
    if call is not None:
        fn = _import_attr(str(_slot(call, "module", "")), str(_slot(call, "name", "")), name)
        if not callable(fn):
            raise TypeError(f"analysis {name!r}: {_slot(call, 'name')} is not callable.")
        return fn(**kwargs)

    cls_ref = _slot(analysis, "class_call")
    if cls_ref is not None:
        cls = _import_attr(str(_slot(cls_ref, "module", "")), str(_slot(cls_ref, "name", "")), name)
        ctor = {str(_slot(a, "name")): _slot(a, "value")
                for a in as_list(_slot(cls_ref, "constructor_args"))}
        return cls(**ctor)(**kwargs)

    declared = [k for k in ("function", "equation") if _slot(analysis, k) is not None]
    raise ValueError(
        f"analysis {name!r}: declares "
        + (f"only `{declared[0]}:`" if declared else "no invocation")
        + ". An analysis over saved data needs a `callable:` (module + name) or a "
        "`class_call:`; the `function:`/`equation:` forms belong to an observation "
        "pipeline, which is evaluated inside a run."
    )


RENDERERS = {"inprocess": render_inprocess}
"""Backend name -> renderer ``fn(analysis, kwargs) -> produced``.

``Analysis.execution.backend`` selects one, exactly as it selects an experiment's
engine. A backend that renders experiments does not automatically render analyses:
this registry is what makes the difference explicit, so an unrendered backend raises
instead of being silently run by another.
"""

_DEFAULT_RENDERER = "inprocess"


def _render(analysis, kwargs):
    """Dispatch the declared invocation to the renderer its ``execution.backend`` names."""
    backend = _slot(_slot(analysis, "execution"), "backend")
    key = str(backend).lower() if backend else _DEFAULT_RENDERER
    renderer = RENDERERS.get(key)
    if renderer is None:
        raise NotImplementedError(
            f"analysis {_name(analysis)!r} declares `execution.backend: {backend}`, which "
            f"has no analysis renderer (have {sorted(RENDERERS)}). An analysis is not run "
            "by a different backend than the one it asks for — either register a renderer "
            "for it or declare one that exists."
        )
    return renderer(analysis, kwargs)


def _kwargs_of(analysis, results_root) -> dict:
    """The analysis's arguments as kwargs — literals as written, ``used:`` refs resolved.

    A sourced argument arrives as a labelled ``xarray.DataArray`` (sliced and reconciled
    by :func:`tvbo.data.dataref.resolve_dataref`), never a bare positional array, so the
    callable selects by name.
    """
    name = _name(analysis)
    out: dict = {}
    for key, arg in _arg_items(_slot(analysis, "arguments")):
        used = _slot(arg, "used")
        value = _slot(arg, "value")
        if used is not None and value is not None:
            raise ValueError(
                f"analysis {name!r}, argument {key!r}: `value` and `used` are mutually "
                "exclusive — an argument is either a literal or a sourced array."
            )
        if used is None:
            out[key] = value
            continue
        try:
            out[key] = _dref.resolve_dataref(used, results_root=results_root)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"analysis {name!r}, argument {key!r}: {e}") from e
    return out


def _as_dataset(name: str, produced):
    """The callable's return as an xarray ``Dataset`` of ``observation__<key>`` variables.

    A mapping contributes one variable per key; a ``Dataset`` keeps its own variable
    names; anything else is a single array keyed by the analysis name. A labelled
    ``DataArray`` passes through with its dims and coordinates intact — that fidelity is
    the point, since a figure's ``encoding`` names them.
    """
    import numpy as np
    import xarray as xr

    if isinstance(produced, xr.Dataset):
        items = list(produced.data_vars.items())
    elif isinstance(produced, Mapping):
        items = list(produced.items())
    else:
        items = [(name, produced)]

    data_vars = {}
    for key, value in items:
        key = str(key)
        if isinstance(value, xr.DataArray):
            da = value
        else:
            arr = np.asarray(value)
            if arr.dtype == object:
                raise TypeError(
                    f"analysis {name!r}: output {key!r} is not numeric (dtype=object). "
                    "Return arrays, DataArrays, or scalars — a result container holds "
                    "labelled numeric arrays."
                )
            da = xr.DataArray(arr) if arr.ndim == 0 else xr.DataArray(
                arr, dims=[f"{key}_d{i}" for i in range(arr.ndim)]
            )
        data_vars[f"observation__{key}"] = da.rename(f"observation__{key}")
    if not data_vars:
        raise ValueError(f"analysis {name!r}: the callable returned nothing to persist.")
    return xr.Dataset(data_vars)


def _provenance(analysis, produced_keys: Iterable[str]) -> dict:
    """The sidecar record: what was called, with what, producing which arrays."""
    call = _slot(analysis, "callable")
    args = {}
    for key, arg in _arg_items(_slot(analysis, "arguments")):
        used = _slot(arg, "used")
        if used is None:
            args[key] = {"value": _slot(arg, "value")}
            continue
        ref = {k: _slot(used, k) for k in ("experiment", "analysis", "iri", "output", "transform")}
        args[key] = {"used": {k: v for k, v in ref.items() if v is not None}}
    cls_ref = _slot(analysis, "class_call")
    invoked = call if call is not None else cls_ref
    backend = _slot(_slot(analysis, "execution"), "backend") or _DEFAULT_RENDERER
    return {
        "name": _name(analysis),
        "label": _slot(analysis, "label"),
        "description": _slot(analysis, "description"),
        ("class_call" if call is None and cls_ref is not None else "callable"): {
            "module": _slot(invoked, "module"), "name": _slot(invoked, "name"),
        },
        "backend": str(backend),
        "arguments": args,
        "outputs": sorted(str(k) for k in produced_keys),
    }


def run_analysis(analysis, results_root=None, *, compress: bool = True) -> Path:
    """Execute one declared analysis and persist its result. Returns the container path."""
    import yaml

    name = _name(analysis)
    produced = _render(analysis, _kwargs_of(analysis, results_root))
    ds = _as_dataset(name, produced)

    path = container_path(name, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = ({v: {"zlib": True, "complevel": 4} for v in ds.data_vars} if compress else None)
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    record = _provenance(analysis, (str(v)[len("observation__"):] for v in ds.data_vars))
    (path.parent / "result.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def run_analyses(analyses, results_root=None, *, compress: bool = True,
                 on_start=None, on_done=None) -> list[Path]:
    """Execute ``analyses`` in the given order, returning the containers written.

    ``on_start(name)`` / ``on_done(name, path)`` report progress to a caller's logger
    without this module choosing an output style.
    """
    written: list[Path] = []
    for analysis in as_list(analyses):
        name = _name(analysis)
        if on_start:
            on_start(name)
        path = run_analysis(analysis, results_root, compress=compress)
        written.append(path)
        if on_done:
            on_done(name, path)
    return written


def study_analyses(study) -> list:
    """A study's declared analyses as a list, whatever spelling the loader produced."""
    return as_list(_slot(study, "analyses"))
