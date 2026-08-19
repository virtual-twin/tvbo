"""Run a study's declarative ``analyses:`` and persist each as its own container.

A study reports quantities that no simulation produced — a basis decomposition of empirical maps, a spectrum, a permutation test — and reductions that span several experiments. ``SimulationStudy.analyses`` declares each as the same ``FunctionCall`` a ``Parameter.producer`` uses, and this module is what executes one: resolve its arguments (a literal ``value``, or a ``used:`` DataRef reading an experiment, another analysis, or a dataset), call it, and write the result to ``<results_root>/results/<name>/result.h5`` beside a ``result.yaml`` provenance sidecar.

That path is the container convention every consumer already resolves, so a figure layer binds an analysis exactly as it binds a run. Arrays are written through xarray untouched: an invocation that returns labelled ``DataArray``s keeps its dims and coordinates, which is what lets a grammar panel name them in an ``encoding``.

Which backend renders an analysis is declared, not assumed — ``Analysis.execution`` carries the same choice a ``SimulationExperiment`` makes. :data:`RENDERERS` maps a backend name to the renderer that runs it; a backend with no renderer raises rather than being quietly run by another.

The ``used:`` edges also order the work. :func:`schedule` splits the analyses into those that run before the experiments (an experiment may source one through a parameter) and those that run after (they read an experiment's result), each topologically ordered.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tvbo.data import dataref as _dref
from tvbo.utils import as_list

logger = logging.getLogger("tvbo.run")

__all__ = [
    "RENDERERS",
    "analysis_closure",
    "analysis_name",
    "container_path",
    "dependencies",
    "dependents_of",
    "render_inprocess",
    "render_tvboptim",
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


def analysis_name(analysis) -> str:
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

    The layout itself is defined once, by :func:`tvbo.data.dataref.locate_analysis_container`; this is its write-side counterpart, which must not raise on a container that has not been produced.
    """
    return _dref.analysis_container_path(results_root, name)


# ---------------------------------------------------------------------------- order


def dependencies(analysis) -> dict:
    """What this analysis consumes, read off its arguments' ``used:`` edges.

    Returns ``{"experiments": {...}, "analyses": {...}}`` — the ids/names that must have run before it. A literal argument contributes nothing.

    An ``experiment:`` edge contributes BOTH the spelling the recipe used and its bare numeric id, because a recipe may write ``exp-3``, ``exp3`` or ``3`` for the same experiment while the runtime identifies it by ``{key, name, label, id}``. Keeping only the literal string makes every dotted spelling silently match nothing — an empty stale set reads exactly like a clean one.
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

    An analysis container records no link back to the run it was derived from, so re-running part of a study leaves each downstream container holding the PREVIOUS run's numbers with nothing to raise. This names the set to invalidate, and it must be transitive: the second-order analyses (a landscape built from a per-cell reduction, a correlation built from that) are exactly the ones a hand-written list forgets.

    The seed is either kind of node, because both partial-run modes exist: ``--experiment`` re-runs a simulation, ``--analysis`` re-runs a derivation, and what goes stale downstream is found by the same walk over the study's own ``used:`` edges. Analyses named in ``changed_analyses`` come back in the result — a caller that just ran them drops them.
    """
    wanted = {str(e) for e in experiments}
    stale = {str(a) for a in changed_analyses}
    names = [analysis_name(a) for a in as_list(analyses)]
    deps = {n: dependencies(a) for n, a in zip(names, as_list(analyses), strict=True)}
    changed = True
    while changed:
        changed = False
        for name in names:
            if name in stale:
                continue
            if deps[name]["experiments"] & wanted or deps[name]["analyses"] & stale:
                stale.add(name)
                changed = True
    return [n for n in names if n in stale]


def analysis_closure(analyses, names, exists) -> set[str]:
    """The named analyses plus the upstream ones that have no container yet.

    The upstream counterpart of :func:`dependents_of`, and it belongs beside it: both walk the study's ``used:`` edges, and a caller that needs one usually needs the other.

    Asking for an analysis by name means asking for it to be RE-derived, so its own inputs are left alone wherever they already exist — re-running them would be the whole study, which is what asking by name is an alternative to. An input that has never been produced is different: it cannot be read, so it is pulled in, and transitively, because the first missing container's own inputs may be missing too.

    Args:
        analyses: The study's declared analyses.
        names: The names asked for.
        exists: Predicate saying whether an analysis's container is already on disk.
    """
    by_name = {analysis_name(a): a for a in as_list(analyses)}
    needed, queue = set(names), list(names)
    while queue:
        for dep in dependencies(by_name[queue.pop()])["analyses"]:
            if dep in by_name and dep not in needed and not exists(dep):
                needed.add(dep)
                queue.append(dep)
    return needed


def _toposort(analyses) -> list:
    """Analyses in dependency order, declaration order breaking every tie.

    Also enforces the two things a name-keyed container makes structural: names are unique (two analyses sharing one would overwrite each other's result), and every ``used: {analysis: …}`` names one that exists. Raises on a cycle, listing the analyses left unresolved.
    """
    items = list(analyses)
    names = [analysis_name(a) for a in items]
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
                f"analysis {analysis_name(a)!r} sources {missing} through a `used: {{analysis: …}}`, "
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
                f"{sorted(analysis_name(a) for a in pending)} — an analysis cannot consume its "
                "own result, directly or through a chain."
            )
        for a in ready:
            ordered.append(a)
            done.add(analysis_name(a))
            pending.remove(a)
    return ordered


def schedule(analyses) -> tuple[list, list]:
    """Split the analyses into ``(before_experiments, after_experiments)``.

    An analysis that reads an experiment result runs after the experiments, and so does everything downstream of it. The rest run first, so an experiment's own parameter can source one through a ``used:``. Each list is topologically ordered.
    """
    ordered = _toposort(analyses)
    deferred: set[str] = set()
    for a in ordered:
        deps = dependencies(a)
        if deps["experiments"] or (deps["analyses"] & deferred):
            deferred.add(analysis_name(a))
    before = [a for a in ordered if analysis_name(a) not in deferred]
    after = [a for a in ordered if analysis_name(a) in deferred]
    return before, after


# --------------------------------------------------------------------------- execute


def _import_attr(module: str, attr: str, name: str):
    """``module.attr``, with an error that says how a study makes its code importable."""
    if not module or not attr:
        raise ValueError(f"analysis {name!r}: needs both `module` and `name` (got module={module!r}, name={attr!r}).")
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

    ``callable:`` calls the function with the resolved arguments. ``class_call:`` instantiates the class with its ``constructor_args`` and calls the instance with the resolved arguments, so an analysis carried by a class (a fitted estimator, a stateful reduction) declares the same way a monitor does. The ``function:`` and ``equation:`` forms are refused rather than half-executed — they describe a step inside an observation pipeline, whose inputs come from a running solve, not a container.

    It imposes no array library: whatever the declared code uses is what runs. The written container is xarray either way.
    """
    name = analysis_name(analysis)

    call = _slot(analysis, "callable")
    if call is not None:
        fn = _import_attr(str(_slot(call, "module", "")), str(_slot(call, "name", "")), name)
        if not callable(fn):
            raise TypeError(f"analysis {name!r}: {_slot(call, 'name')} is not callable.")
        return fn(**kwargs)

    cls_ref = _slot(analysis, "class_call")
    if cls_ref is not None:
        cls = _import_attr(str(_slot(cls_ref, "module", "")), str(_slot(cls_ref, "name", "")), name)
        ctor = {str(_slot(a, "name")): _slot(a, "value") for a in as_list(_slot(cls_ref, "constructor_args"))}
        return cls(**ctor)(**kwargs)

    declared = [k for k in ("function", "equation") if _slot(analysis, k) is not None]
    raise ValueError(
        f"analysis {name!r}: declares "
        + (f"only `{declared[0]}:`" if declared else "no invocation")
        + ". An analysis over saved data needs a `callable:` (module + name) or a "
        "`class_call:`; the `function:`/`equation:` forms belong to an observation "
        "pipeline, which is evaluated inside a run."
    )


def _labelled(value):
    """``(dims, array)`` for a labelled input, or ``((), value)`` for a scalar/bare array."""
    dims = tuple(getattr(value, "dims", ()) or ())
    return dims, (value.values if dims else value)


_EXPRESSION_NAMESPACES = {
    "jax": lambda: {"jnp": __import__("jax.numpy", fromlist=["numpy"])},
    "numpy": lambda: {"np": __import__("numpy"), "numpy": __import__("numpy")},
}
"""Backend name -> the module bindings its printer's output expects at call time.

A printer and the namespace its code runs in are one choice, not two: printing
``numpy.tanh`` into a scope holding only ``jnp`` compiles fine and raises ``NameError``
on the first call, so the pair is registered together and an unknown *fmt* is refused
here rather than at the call site.
"""


def _expression_fn(analysis, names, fmt="jax"):
    """Compile the declared ``equation:`` into a function of *names*, for backend *fmt*.

    The RHS is parsed against the same vocabulary a streaming reducer uses — tvbo's array functions plus one symbol per declared argument — and printed by the *fmt* printer, so the one declaration lowers to whichever backend renders it. Nothing about the analysis is hard-coded here; the recipe is the data.
    """
    import sympy as sp

    from tvbo.codegen.code import get_printer
    from tvbo.parse.expression import ARRAY_FUNCTIONS

    name = analysis_name(analysis)
    rhs = _slot(_slot(analysis, "equation"), "rhs")
    if rhs is None:
        raise ValueError(f"analysis {name!r}: `equation:` needs an `rhs`.")
    if str(fmt) not in _EXPRESSION_NAMESPACES:
        raise ValueError(
            f"analysis {name!r}: no expression namespace registered for {fmt!r} "
            f"(known: {sorted(_EXPRESSION_NAMESPACES)}). Register the printer's module "
            "bindings alongside it, or render the analysis on a backend that is."
        )
    vocab = {**ARRAY_FUNCTIONS, "pi": sp.pi, **{n: sp.Symbol(n) for n in names}}
    src = get_printer(fmt).doprint(sp.sympify(str(rhs), locals=vocab))

    ns: dict = {}
    exec(f"def _analysis({', '.join(names)}):\n    return {src}\n", _EXPRESSION_NAMESPACES[str(fmt)](), ns)
    return ns["_analysis"], src


def _aligned(value, arg_dims, dims, mapped):
    """``(array, is_mapped)`` — one argument laid out on the expression's own axis order.

    An elementwise expression combines its arguments positionally once they are arrays, but the containers they come from are keyed, so two inputs may carry the same axes in different orders — ``(node, time)`` and ``(time, node)`` are the same data. Ordering each argument by *dims* is what makes the expression mean what it reads as; without it numpy pairs the axes by position and silently multiplies node against time whenever the lengths happen to agree.

    An axis an argument does not carry becomes a length-1 axis in the right place rather than being left to numpy's right-alignment, which would land a per-node vector on the time axis of a ``(node, time)`` result.
    """
    import numpy as np

    order = tuple(d for d in dims if d in arg_dims)
    is_mapped = bool(mapped) and mapped in arg_dims
    arr = np.asarray(value.transpose(*order).values)
    inner = dims[1:] if mapped else dims
    rest = order[1:] if is_mapped else order
    if rest != inner:
        idx = ((slice(None),) if is_mapped else ()) + tuple(slice(None) if d in rest else None for d in inner)
        arr = arr[idx]
    return arr, is_mapped


def _elementwise_dims(kwargs, mapped) -> tuple:
    """Axis names an elementwise expression over *kwargs* produces, mapped axis first.

    Order is first-mention across the labelled inputs, which is what makes the derivation reproducible from the recipe rather than from any one input's storage layout.
    """
    order: list = []
    for value in kwargs.values():
        for d in _labelled(value)[0]:
            if d not in order:
                order.append(d)
    if mapped and mapped not in order:
        raise ValueError(
            f"`apply_on_dimension: {mapped}` names an axis no argument carries "
            f"(they carry {order}). The mapped axis must exist on at least one input."
        )
    rest = [d for d in order if d != mapped]
    return tuple(([mapped] if mapped else []) + rest)


def _pin_accelerator(execution) -> None:
    """Pin JAX to the declared ``execution.accelerator``, or say why it could not be.

    Effective only before JAX initialises, which is why it is attempted here rather than at import: an analyses-only run touches JAX for the first time in this renderer, so the declaration still lands. Once JAX is up the platform is fixed for the process and the declaration cannot be honoured retroactively — that is reported, not silently dropped, because the analysis then ran somewhere it did not ask for.
    """
    from tvbo.templates.tvboptim.utils import jax_platform

    declared = str(_slot(execution, "accelerator", "auto") or "auto").strip().lower()
    if declared == "auto":
        return
    if "jax" not in sys.modules:
        os.environ.setdefault("JAX_PLATFORMS", jax_platform(declared))
        return

    import jax

    live = {str(getattr(d, "platform", "")).lower() for d in jax.devices()}
    if declared not in live:
        logger.warning(
            "execution.accelerator=%r cannot be applied: JAX is already initialised on %s. "
            "The platform is process-wide and fixed at first import — set JAX_PLATFORMS=%s "
            "in the environment to run this analysis on %s.",
            declared,
            sorted(live) or ["an unknown platform"],
            jax_platform(declared),
            declared,
        )


def _device_plan(analysis, n_items, per_lane_bytes):
    """``(n_pmap, n_vmap)`` for the mapped axis — how many shards, how wide each batch.

    ``n_workers`` is the shard count, exactly as it is for an experiment: the devices the mapped axis spreads across. It is clamped to the devices that actually exist (and to the item count), and a shortfall is logged rather than raised — a recipe declaring 8 shards must still run on a one-device laptop, just not silently as if it had 8. ``batch_size`` is the per-device vmap width; unset means ``auto``, which resolves against the same per-cell memory budget an exploration's ``n_parallel:
    auto`` uses. The live batch is ``n_pmap x n_vmap`` lanes in whatever RAM those devices share, which is what :func:`shared_ram_device_count` accounts for on host-replicated CPU devices.
    """
    import jax

    from tvbo.templates.tvboptim.callbacks import resolve_n_vmap, shared_ram_device_count

    execution = _slot(analysis, "execution")
    workers = max(1, int(_slot(execution, "n_workers", 1) or 1))
    n_pmap = max(1, min(workers, jax.local_device_count(), int(n_items)))
    if workers > n_pmap:
        logger.warning(
            "analysis %r declares execution.n_workers=%d but only %d device(s) and %d item(s) "
            "are available; sharding the %r axis %d ways.",
            analysis_name(analysis),
            workers,
            jax.local_device_count(),
            int(n_items),
            str(_slot(analysis, "apply_on_dimension")),
            n_pmap,
        )
    per_shard = -(-int(n_items) // n_pmap)
    spec = _slot(execution, "batch_size") or "auto"
    n_vmap = resolve_n_vmap(spec, per_shard, per_lane_bytes, n_pmap=shared_ram_device_count())
    return n_pmap, max(1, min(int(n_vmap), per_shard))


def _map_over(analysis, fn, names, args, in_axes, mapped):
    """Evaluate *fn* over every element of the mapped axis, honouring ``execution``.

    One device by default (``jax.vmap`` under ``jit``). When the analysis declares ``n_workers`` or ``batch_size``, the axis instead runs through tvboptim's own :class:`ParallelExecution` — the pmap-of-``lax.map`` an experiment's grid runs on — so a cohort-sized analysis spreads across devices and chunks within one instead of holding every lane at once. Reusing that path rather than re-deriving it is what keeps the padding, shard reshape and trim identical to an experiment's.

    Parallelism is declared, never inferred: an analysis with no ``execution`` block takes the single-device path and produces the same array it always did.
    """
    import jax
    import numpy as np

    execution = _slot(analysis, "execution")
    workers = max(1, int(_slot(execution, "n_workers", 1) or 1))
    batch = _slot(execution, "batch_size")

    if not mapped or (workers <= 1 and batch is None):
        if not mapped and (workers > 1 or batch is not None):
            logger.warning(
                "analysis %r declares execution.n_workers/batch_size but no "
                "`apply_on_dimension:` — there is no axis to shard, so the expression "
                "is evaluated whole.",
                analysis_name(analysis),
            )
        kernel = jax.vmap(fn, in_axes=tuple(in_axes)) if mapped else fn
        return np.asarray(jax.jit(kernel)(*args))

    from tvboptim.execution import ParallelExecution
    from tvboptim.types.spaces import DataAxis, Space

    from tvbo.templates.tvboptim.callbacks import estimate_per_cell_bytes

    n_items = int(args[in_axes.index(0)].shape[0])
    lane = tuple(a[0] if ax == 0 else a for a, ax in zip(args, in_axes, strict=True))
    per_lane = estimate_per_cell_bytes(lambda one: fn(*one), lane) if batch is None else None
    n_pmap, n_vmap = _device_plan(analysis, n_items, per_lane)

    space = Space(
        {n: (DataAxis(a) if ax == 0 else a) for n, a, ax in zip(names, args, in_axes, strict=True)},
        mode="zip",
    )
    logger.debug(
        "analysis %r: %d %r lanes over n_pmap=%d x n_vmap=%d (per-lane %s B)",
        analysis_name(analysis),
        n_items,
        mapped,
        n_pmap,
        n_vmap,
        per_lane,
    )
    result = ParallelExecution(
        lambda lane_state: fn(**{n: lane_state[n] for n in names}),
        space,
        n_vmap=n_vmap,
        n_pmap=n_pmap,
    ).run()
    sharded = np.asarray(result.results)
    return sharded.reshape((-1,) + sharded.shape[2:])[:n_items]


def render_tvboptim(analysis, kwargs):
    """Renderer that lowers a declarative ``equation:`` analysis to JAX and vmaps it.

    This is the metadata-native form: instead of pointing at arbitrary ``code/`` Python, the analysis states an expression over its named arguments and the axis to map it over, and tvbo emits the realization — so the framework, not the study, owns the parallelism. ``apply_on_dimension`` becomes a ``jax.vmap`` over that axis of every argument carrying it (arguments without it are broadcast, not copied per element), and ``aggregate`` reduces a named axis of the result.

    The expression is ELEMENTWISE over the inputs' axes; ``aggregate`` is what reduces.
    Output axes are derived from that contract — ``apply_on_dimension`` first, then the inputs' own axes minus ``aggregate.over`` — and cross-checked against the result's rank, never read off its shape. An expression that reshapes or reduces on its own (an outer product, a correlation) fails that check and must declare ``dims:``, which is honoured as written.

    How that map is spread is declared too. By default it is one ``jit``-ed vmap on one device; ``execution.n_workers`` shards the mapped axis across devices and ``execution.batch_size`` bounds how many lanes are live per device, both through the same tvboptim machinery an experiment's grid uses (see :func:`_map_over`).
    ``execution.accelerator`` pins the platform when JAX has not yet initialised.

    Host-only work stays host-only: a sparse eigensolve, a CIFTI read or a spin-permutation generator is not JAX-expressible and belongs on the ``inprocess`` renderer. This renderer refuses a ``callable:``/``class_call:`` analysis rather than silently jitting code that was never written to be traced.
    """
    import numpy as np
    import xarray as xr

    name = analysis_name(analysis)
    if _slot(analysis, "equation") is None:
        raise ValueError(
            f"analysis {name!r}: `execution.backend: tvboptim` renders the declarative "
            "`equation:` form. A `callable:` analysis is arbitrary Python, which this "
            "renderer will not trace — declare an `equation:`, or render it `inprocess`."
        )

    names = list(kwargs)
    _pin_accelerator(_slot(analysis, "execution"))
    fn, src = _expression_fn(analysis, names)
    mapped = _slot(analysis, "apply_on_dimension")
    mapped = str(mapped).split(".")[-1] if mapped else None

    agg = _slot(analysis, "aggregate")
    over = _slot(agg, "over")
    over = str(over).split(".")[-1] if over else None
    dims = _elementwise_dims(kwargs, mapped)
    if over and over not in dims:
        raise ValueError(
            f"analysis {name!r}: `aggregate.over: {over}` names an axis the "
            f"expression does not produce (it produces {list(dims)})."
        )

    in_axes, args = [], []
    for key in names:
        arg_dims, array = _labelled(kwargs[key])
        if not arg_dims:
            in_axes.append(None)
            args.append(array)
            continue
        arr, is_mapped = _aligned(kwargs[key], arg_dims, dims, mapped)
        in_axes.append(0 if is_mapped else None)
        args.append(arr)

    produced = _map_over(analysis, fn, names, args, in_axes, mapped)

    if over:
        how = str(_slot(agg, "type", "mean")).split(".")[-1]
        if how != "none":
            produced = getattr(np, how)(produced, axis=dims.index(over))
            dims = tuple(d for d in dims if d != over)

    declared = _slot(analysis, "dims")
    if declared:
        dims = tuple(str(d) for d in declared)
    if produced.ndim != len(dims):
        raise ValueError(
            f"analysis {name!r}: the expression returned a rank-{produced.ndim} result "
            f"where its declaration implies {len(dims)} axes {list(dims)}. Axis names are "
            "declared, never read off a shape — if the expression reshapes or reduces on "
            f"its own, name the output axes with `dims:`. Lowered expression: {src}"
        )

    coords = {}
    for key in names:
        value = kwargs[key]
        for d in _labelled(value)[0]:
            if d in dims and d not in coords and d in getattr(value, "coords", {}):
                coords[d] = value.coords[d].values
    return xr.DataArray(produced, dims=list(dims), coords=coords)


RENDERERS = {"inprocess": render_inprocess, "tvboptim": render_tvboptim}
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
            f"analysis {analysis_name(analysis)!r} declares `execution.backend: {backend}`, which "
            f"has no analysis renderer (have {sorted(RENDERERS)}). An analysis is not run "
            "by a different backend than the one it asks for — either register a renderer "
            "for it or declare one that exists."
        )
    return renderer(analysis, kwargs)


def _kwargs_of(analysis, results_root) -> dict:
    """The analysis's arguments as kwargs — literals as written, ``used:`` refs resolved.

    A sourced argument arrives as a labelled ``xarray.DataArray`` (sliced and reconciled by :func:`tvbo.data.dataref.resolve_dataref`), never a bare positional array, so the callable selects by name.
    """
    name = analysis_name(analysis)
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

    A mapping contributes one variable per key; a ``Dataset`` keeps its own variable names; a ``DataFrame`` contributes one variable per COLUMN, all sharing a single row dimension, so a table survives the round trip as a table rather than as an unlabelled block; anything else is a single array keyed by the analysis name. A labelled ``DataArray`` passes through with its dims and coordinates intact — that fidelity is the point, since a figure's ``encoding`` names them.
    """
    import numpy as np
    import xarray as xr

    row_dim, row_index = None, None
    if isinstance(produced, xr.Dataset):
        items = list(produced.data_vars.items())
    elif _is_dataframe(produced):
        row_dim, row_index = f"{name}_row", _row_labels(produced)
        columns = [str(c) for c in produced.columns]
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"analysis {name!r}: the returned table has repeated column names "
                f"({sorted({c for c in columns if columns.count(c) > 1})}) — one column is "
                "one stored array, so a repeat would silently keep only the last."
            )
        items = [(str(c), produced[c].to_numpy()) for c in produced.columns]
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
                arr = _strings_or_raise(name, key, arr)
            if arr.ndim == 0:
                da = xr.DataArray(arr)
            elif row_dim is not None and arr.ndim == 1:
                da = xr.DataArray(arr, dims=[row_dim])
            else:
                da = xr.DataArray(arr, dims=[f"{key}_d{i}" for i in range(arr.ndim)])
        data_vars[f"observation__{key}"] = da.rename(f"observation__{key}")
    if not data_vars:
        raise ValueError(f"analysis {name!r}: the callable returned nothing to persist.")
    coords = {row_dim: row_index} if row_index is not None else None
    return xr.Dataset(data_vars, coords=coords)


def _is_dataframe(obj) -> bool:
    """True for a pandas DataFrame, without importing pandas to ask."""
    return type(obj).__name__ == "DataFrame" and hasattr(obj, "columns") and hasattr(obj, "to_numpy")


def _row_labels(frame):
    """A table's index as a coordinate, or ``None`` where it is the plain 0..n-1 counter.

    A table keyed by region, parameter or condition carries its keys in the index; dropping it stores the columns without the names that say which row is which.
    """
    import numpy as np

    idx = np.asarray(frame.index.to_numpy())
    if idx.dtype.kind in "iu" and np.array_equal(idx, np.arange(len(idx))):
        return None
    return idx.astype(str) if idx.dtype == object else idx


def _strings_or_raise(name: str, key: str, arr):
    """An object array of strings becomes a string array; anything else is not storable."""
    flat = arr.ravel()
    if all(isinstance(v, str) for v in flat):  # vacuously true for an empty column, which is storable
        return arr.astype(str)
    raise TypeError(
        f"analysis {name!r}: output {key!r} is not numeric (dtype=object). "
        "Return arrays, DataArrays, strings, or scalars — a result container holds "
        "labelled numeric arrays."
    )


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
        "name": analysis_name(analysis),
        "label": _slot(analysis, "label"),
        "description": _slot(analysis, "description"),
        ("class_call" if call is None and cls_ref is not None else "callable"): {
            "module": _slot(invoked, "module"),
            "name": _slot(invoked, "name"),
        },
        "backend": str(backend),
        "arguments": args,
        "outputs": sorted(str(k) for k in produced_keys),
    }


def run_analysis(analysis, results_root=None, *, compress: bool = True) -> Path:
    """Execute one declared analysis and persist its result. Returns the container path."""
    import yaml

    name = analysis_name(analysis)
    produced = _render(analysis, _kwargs_of(analysis, results_root))
    ds = _as_dataset(name, produced)

    path = container_path(name, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars} if compress else None
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    record = _provenance(analysis, (str(v)[len("observation__") :] for v in ds.data_vars))
    (path.parent / "result.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    # Digest of the declaration this container came from, so staleness is per analysis rather than "the spec file was touched" (see study_collection._stale_or_missing_analyses).
    from tvbo.data.study_collection import _analysis_fingerprint

    (path.parent / ".fingerprint").write_text(_analysis_fingerprint(analysis))
    return path


def run_analyses(analyses, results_root=None, *, compress: bool = True, on_start=None, on_done=None) -> list[Path]:
    """Execute ``analyses`` in the given order, returning the containers written.

    ``on_start(name)`` / ``on_done(name, path)`` report progress to a caller's logger without this module choosing an output style.
    """
    written: list[Path] = []
    for analysis in as_list(analyses):
        name = analysis_name(analysis)
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
