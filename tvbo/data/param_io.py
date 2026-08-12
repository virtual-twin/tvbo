"""Resolve a :class:`Parameter`'s value from its declared provenance.

A parameter says where its value comes from in exactly one of three ways, and the
choice is by provenance, never by size:

``value:``
    A YAML literal. Returned as-is; codegen inlines it.

``source:`` (+ ``measure:``)
    WHERE existing bytes live — a curated entity's IRI, or a path resolved against
    the declaring spec's directory. ``measure`` selects one array out of a source
    holding several; its address space is the source's own (an HDF5/Zarr dataset
    key, a Network's per-node measure name).

``producer:``
    HOW derived bytes are made — a :class:`FunctionCall` naming the callable, its
    arguments, and (via ``output``) which entry to take from a callable returning
    several named arrays.

Sourced and produced values are **never materialised into ``Parameter.value``**, so
loading a spec stays cheap no matter how large the array is, and a dumper never sees
bytes it would try to serialise back into YAML. Resolution happens here, on demand,
and the result is cached in this module — the ``Parameter`` object is never mutated.

That is deliberate: ``Parameter`` appears at 16 nesting sites in the schema, and
LinkML always constructs the *declared range* class, so a tvbo subclass carrying lazy
behaviour would need a hand-written re-wrap at every one of them (and would diverge
between the dataclass and pydantic flavours). Keeping resolution outside the class
means the generated datamodel is untouched and both flavours behave identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

# Resolved arrays, keyed by a content-addressed key (see `_source_key` /
# `_producer_key`) rather than by object identity: two Parameter objects naming the
# same array share one entry, and a rebuilt spec hits the cache rather than re-reading
# or recomputing. Never keyed by id() — CPython reuses ids of collected objects, which
# would silently serve one parameter's array for another's.
_CACHE: dict[tuple, Any] = {}

# module name -> digest of the source that module was LOADED from, pinned for the process
_SOURCE_DIGESTS: dict[str, str] = {}


def _default_cache_dir() -> Path:
    """On-disk home for materialised produced constants, alongside the network cache
    (``~/.tvbo/networks``). Resolved per call, not at import, so a harness that sets
    ``HOME`` after importing tvbo still writes where it expects."""
    return Path.home() / ".tvbo" / "constants"


def clear_cache() -> None:
    """Drop every resolved array. Mainly for tests and long-lived processes."""
    _CACHE.clear()
    _SOURCE_DIGESTS.clear()


# --------------------------------------------------------------------------- helpers


def _slot(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _resolve_path(source: str, source_dir: Optional[Path]) -> Optional[Path]:
    """A path source, resolved against the declaring spec's directory when relative.

    Mirrors how ``Network.bids_dir`` / ``Network.data_file`` resolve, so a spec means
    the same thing wherever it is loaded from and a kit that carries its companion
    alongside the spec keeps working after it is moved.
    """
    p = Path(str(source))
    if not p.is_absolute() and source_dir is not None:
        p = (Path(source_dir) / p).resolve()
    return p if p.exists() else None


def _fingerprint(path: Path) -> tuple:
    """Cheap staleness key for a file (the mtime+size half of ReferenceFingerprint)."""
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _source_key(path: Path, measure: Optional[str]) -> tuple:
    return ("source", _fingerprint(path), measure)


def _readonly(value: Any) -> Any:
    """A read-only view of a resolved array (recursing into a bundle dict).

    One buffer is shared by every parameter naming the same array, so an in-place write
    by one consumer would silently corrupt the others — action at a distance that would
    surface in an unrelated run. A resolved constant is conceptually immutable (it IS the
    declared value), so a read-only view turns an accidental write into a ValueError at
    the offending line.

    Crucially this returns a *view* and does not freeze the input in place: a producer
    may hand back an array the recipe still owns (a module-level cache, or an echoed
    argument), and setting ``write=False`` on that object would break the recipe's own
    later use of it from a line that never asked for a constant.
    """
    if isinstance(value, np.ndarray):
        view = value.view()
        view.setflags(write=False)
        return view
    if isinstance(value, dict):
        return {k: _readonly(v) for k, v in value.items()}
    return value


# ------------------------------------------------------------------------ references

# Only a fully-qualified `network.*` value is a reference; anything else is a literal
# (a bare `weight` is a state variable, not the connectome) — the same rule the pipeline
# argument path uses, so a producer and a pipeline step read arguments identically.
_REF_PREFIX = "network."


def _mesh_array(net: Any, field: str) -> np.ndarray:
    """A mesh array off the Network's lazy runtime caches (set by the h5 load path)."""
    attr = {
        "vertices": "_mesh_vertices",
        "elements": "_mesh_elements",
        "faces": "_mesh_elements",
        "normals": "_mesh_normals",
    }.get(field)
    if attr is None:
        raise ValueError(f"network.mesh.{field}: unknown mesh array; expected one of vertices, elements/faces, normals.")
    try:
        return np.asarray(object.__getattribute__(net, attr))
    except AttributeError:
        raise ValueError(
            f"network.mesh.{field}: the network carries no mesh (nothing loaded into "
            f"{attr}); does its companion have a mesh/ group?"
        ) from None


_NODE_MEASURES = ("positions", "instrength")


def resolve_network_node(net: Any, measure: str) -> Optional[np.ndarray]:
    """Per-node vector for a ``network.<measure>`` reference — the single definition shared
    by the producer-argument path (``_resolve_ref``) and the observation-embedding path
    (``utils.collect_network_node_arrays``), so both resolve ``network.positions`` /
    ``network.instrength`` identically. ``positions`` → region centroids ``(n_nodes, 3)``;
    ``instrength`` → weighted in-degree ``matrix('weight').sum(axis=1)`` (row sum = incoming,
    the TVB/Koller convention). Returns None when the measure is unknown or unbuildable.
    """
    if measure == "positions" and hasattr(net, "node_positions"):
        return np.asarray(net.node_positions(), dtype=float)
    if measure == "instrength" and hasattr(net, "matrix"):
        w = net.matrix("weight")
        return np.asarray(w, dtype=float).sum(axis=1) if w is not None else None
    return None


def is_reference(value: Any) -> bool:
    """True when an argument value points at an entity rather than being a literal."""
    return isinstance(value, str) and value.startswith(_REF_PREFIX)


def _resolve_ref(ref: str, context: Any, where: str) -> Any:
    """Resolve a dotted ``network.*`` reference against the owning experiment."""
    if context is None:
        raise ValueError(
            f"{where}: {ref!r} references an entity but no context was given; pass the "
            f"owning experiment as `context=` so the reference can be resolved."
        )
    # The context is the owning experiment, or a network directly (handy for tests and
    # for a caller that already has one). Discriminate on the slot, not on whether nodes
    # happen to be populated — an empty network is still a network.
    net = context.network if hasattr(context, "network") else context
    if net is None:
        raise ValueError(f"{where}: {ref!r} needs a network but the context has none.")

    rest = ref[len(_REF_PREFIX) :]
    if rest == "nodes.position":
        rest = "positions"  # legacy spelling of network.positions
    if rest in _NODE_MEASURES:
        vec = resolve_network_node(net, rest)
        if vec is None:
            raise ValueError(f"{where}: {ref!r} needs a network that can build {rest!r}, got {type(net).__name__}.")
        return vec
    if rest.startswith("mesh."):
        return _mesh_array(net, rest.split(".", 1)[1])
    if rest.startswith("edges."):
        label = rest.split(".", 1)[1]
        mat = net.matrix(label) if hasattr(net, "matrix") else None
        if mat is None:
            raise ValueError(f"{where}: the network has no {label!r} matrix.")
        return np.asarray(mat, dtype=float)
    raise ValueError(
        f"{where}: unsupported reference {ref!r}; supported are network.positions, "
        f"network.instrength, network.mesh.<vertices|elements|normals>, network.edges.<label>."
    )


def _hashable(value: Any) -> Any:
    """A stable, hashable rendering of an argument value (lists/dicts included)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, np.ndarray):
        # dtype and shape too: `tobytes()` alone collides (2,3) with (3,2), and float32
        # with the int32 of the same bytes, onto one key and one artifact path.
        return (value.dtype.str, value.shape, value.tobytes())
    return value


def _producer_key(module: str, name: str, kwargs: dict) -> tuple:
    """Keyed on the CALL and on the producing SOURCE, deliberately not on ``output``.

    One producer typically returns a bundle of named arrays (a precompute emitting every
    mesh operator at once). Keying per-output would re-run that call once per parameter
    reading it — the expensive thing the cache exists to avoid.

    The source digest belongs HERE rather than only in the artifact path, because the two
    must key on the same thing. Keyed apart, a process that materialises, has its producer
    edited underneath it, and materialises again computes the NEW path from the new source
    while the in-memory cache still answers on the old one — writing pre-edit arrays under a
    digest that asserts they are post-edit. Every later run then reads that file and trusts
    it, which is worse than the stale hit this digest exists to prevent.
    """
    return ("producer", module, name, _hashable(kwargs), _module_source_digest(module))


def _module_source_digest(module: str) -> str:
    """A digest of the source the DEFINING MODULE is currently LOADED from.

    Without this the on-disk companion is keyed on ``(module, function, kwargs)`` alone, and
    editing the callable changes nothing the key can see: the run reads the array from before
    the edit while a direct call to the same function returns the new value. Hashing the whole
    module rather than the function catches an edit to a helper the producer delegates to —
    which is where that bug actually landed — but only for a helper in the SAME file. An edit
    to a sibling module in ``code/`` still changes no byte here and is still invisible; a
    producer that delegates across files must be re-derived deliberately.

    The digest describes the source the module was LOADED from, so it is taken once per
    process and then pinned. Python does not re-execute an imported module, so re-reading
    the file would let an edit rename the artifact while the stale function still fills it —
    writing pre-edit arrays under a digest asserting they are post-edit, which every later
    process then finds and trusts. That is worse than the stale hit this exists to prevent,
    because a stale hit clears on restart and a mislabelled artifact never does. Pinning
    keeps key and code in step without reloading the module, which would re-run its
    import-time side effects (a ``code_modules`` module registers panels and transforms on
    import). An in-session edit therefore needs a restart to take effect — exactly as the
    edited function itself does. A module with no readable source contributes nothing,
    keeping the previous behaviour for anything not backed by a file.
    """
    import hashlib
    import importlib
    import sys

    if module in _SOURCE_DIGESTS:
        return _SOURCE_DIGESTS[module]
    try:
        mod = sys.modules.get(module) or importlib.import_module(module)
        source = Path(getattr(mod, "__file__", "") or "")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16] if source.is_file() else ""
    except Exception:
        # Uncached, so a transient read failure degrades this one lookup rather than
        # pinning the digest-free key for the rest of the process.
        return ""
    _SOURCE_DIGESTS[module] = digest
    return digest


# --------------------------------------------------------------------------- sources


def _read_source(path: Path, measure: Optional[str]) -> Any:
    """Read ``measure`` out of a binary store, or the whole array when it holds one."""
    from tvbo.data.matrix_io import LazyArrayStore

    store = LazyArrayStore(path, {})
    if measure:
        return store.read_dataset(measure)
    arrays = store.arrays
    if len(arrays) == 1:
        return next(iter(arrays.values()))
    raise ValueError(f"{path} holds {len(arrays)} arrays {sorted(arrays)}; the parameter must name one with `measure:`.")


def read_artifact(path: Any, key: Optional[str] = None) -> Any:
    """Load a materialised ``(path, key)`` artifact back to its array — for codegen probes
    that need a produced/sourced constant's shape (e.g. a partition's group count) without
    re-running the producer."""
    return _read_source(Path(path), key)


# ------------------------------------------------------------------------- producers


def _argument_values(producer: Any, context: Any, where: str) -> dict:
    """The producer's arguments as plain kwargs, entity references resolved.

    A `network.*` value is resolved against the context; everything else is a literal.
    Same rule as a pipeline step's arguments, so a producer reads no differently.
    """
    args = _slot(producer, "arguments", None) or {}
    items = args.items() if hasattr(args, "items") else [(_slot(a, "name"), a) for a in args]
    out = {}
    for k, a in items:
        value = _slot(a, "value", a)
        out[str(k)] = _resolve_ref(value, context, where) if is_reference(value) else value
    return out


def _producer_spec(producer: Any, param_name: str, context: Any) -> tuple:
    """The producer's ``(module, name, kwargs)`` — its identity and its inputs."""
    call = _slot(producer, "callable", None)
    if call is None:
        raise ValueError(f"Parameter {param_name!r} declares a producer with no `callable:`.")
    module, name = str(_slot(call, "module", "")), str(_slot(call, "name", ""))
    if not module or not name:
        raise ValueError(
            f"Parameter {param_name!r}: producer callable needs both `module` and `name` "
            f"(got module={module!r}, name={name!r})."
        )
    kwargs = _argument_values(producer, context, f"Parameter {param_name!r} producer")
    return module, name, kwargs


def _producer_bundle(
    producer: Any,
    param_name: str,
    context: Any,
    spec: Optional[tuple] = None,
    key: Optional[tuple] = None,
) -> Any:
    """Everything the producer returns, cached on the CALL — no output selection.

    Kept separate from selection so a caller that needs the whole result (writing the
    cache artifact) gets every named array, not just the one entry some parameter asked
    for; writing only that entry would make each sibling output a fresh re-run.

    ``spec`` is an already-resolved ``_producer_spec``; pass it when the caller has one,
    since resolving arguments again would rebuild every referenced array (a whole node
    position matrix) only to discard the copy. ``key`` is the matching ``_producer_key``,
    for the same reason one step further on: hashing it re-runs ``tobytes()`` over every
    array argument.

    The callable resolves by bare module name against the recipe's ``code_source``
    (already on ``sys.path`` once the study is loaded), so a study's own code produces
    its own derived constants without that code living in core tvbo.
    """
    import importlib

    module, name, kwargs = spec or _producer_spec(producer, param_name, context)
    key = key if key is not None else _producer_key(module, name, kwargs)
    if key in _CACHE:
        return _CACHE[key]
    try:
        fn = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Parameter {param_name!r}: cannot import producer {module}.{name} — is the "
            f"study's `code_source` registered? ({exc})"
        ) from exc
    produced = _readonly(fn(**kwargs))
    _CACHE[key] = produced
    return produced


def _call_producer(producer: Any, param_name: str, context: Any) -> Any:
    """The producer's result, narrowed to this parameter's ``output`` entry."""
    spec = _producer_spec(producer, param_name, context)
    module, name, _ = spec
    produced = _producer_bundle(producer, param_name, context, spec)
    output = _slot(producer, "output", None)

    if output:
        # A callable returning several named arrays (e.g. a bundle of mesh operators)
        # is selected by `output`, so one precompute serves many parameters and runs once.
        try:
            produced = produced[str(output)]
        except (TypeError, KeyError, IndexError) as exc:
            keys = sorted(produced) if hasattr(produced, "keys") else type(produced).__name__
            raise ValueError(
                f"Parameter {param_name!r}: producer {module}.{name} has no output {output!r} (it returned {keys})."
            ) from exc
    elif isinstance(produced, dict):
        # Hand back a shallow copy of the bundle: its arrays are read-only, but the dict
        # itself is the cache entry, and rebinding a key in it would poison every other
        # parameter reading this producer.
        return dict(produced)
    return produced


# ------------------------------------------------------------------------------- API


def _declared_name(obj: Any) -> str:
    """What to call this thing in a cache key and an error message.

    Everything carrying the provenance triple is named, but not all of it is named
    ``name``: an ``Edge`` is identified by its ``label`` (the matrix it supplies).
    """
    return str(_slot(obj, "name") or _slot(obj, "label") or "<unnamed>")


def _provenance(param: Any) -> Optional[str]:
    """Which of ``value``/``source``/``producer`` this parameter declares.

    The schema states the three are mutually exclusive; enforce it here rather than let
    each entry point pick its own precedence. Two that disagreed (``resolve`` preferring
    the producer while ``materialise`` preferred the source) would hand the host and the
    generated code different values for one parameter, silently.
    """
    declared = [n for n in ("value", "source", "producer") if _slot(param, n) is not None]
    if len(declared) > 1:
        raise ValueError(
            f"{_declared_name(param)!r} declares "
            f"{declared}; value/source/producer are mutually exclusive — a literal, "
            f"bytes to read, and a recipe to compute are three different claims about "
            f"where the value comes from."
        )
    return declared[0] if declared else None


def is_lazy(param: Any) -> bool:
    """True when this parameter's value is resolved rather than inlined.

    The one rule codegen branches on: a literal is materialised and inlined; anything
    obtained (``source``) or derived (``producer``) is read at run time and must never
    be embedded in generated source.
    """
    return _provenance(param) in ("source", "producer")


def _source_file(param: Any, source_dir: Optional[Path], name: str) -> Path:
    """The existing file a ``source:`` parameter names, resolved against the spec dir."""
    source = _slot(param, "source")
    path = _resolve_path(str(source), source_dir)
    if path is None:
        raise ValueError(
            f"Parameter {name!r}: source {source!r} does not resolve to an existing file (source_dir={source_dir})."
        )
    return path


def materialise(
    param: Any,
    source_dir: Optional[Path] = None,
    context: Any = None,
    cache_dir: Optional[Path] = None,
) -> tuple:
    """The ``(file, key)`` a backend reads this parameter's array from.

    Codegen emits this pair rather than the bytes, so a large constant never enters the
    spec or the generated source and is read at run time instead. Generated modules are
    ``exec``'d in memory as often as they are written to disk, so the path is absolute
    (resolved against the declaring spec's directory) — the same way ``bids_dir`` is
    emitted. A kit stays correct without rewriting code: its emitter rewrites the *spec*
    to point at the companion it staged and re-renders, exactly as it already does for
    ``network.h5``.

    A ``source:`` parameter already lives in a file, so nothing is written. A
    ``producer:`` parameter is computed once and cached content-addressed under
    ``~/.tvbo/constants``, keyed by the producing call — so it survives across runs and
    every parameter naming that producer shares the one artifact.

    Raises for a literal (there is nothing to read; it inlines) or for a parameter with
    no declared value.
    """
    name = _declared_name(param)
    kind = _provenance(param)
    if kind not in ("source", "producer"):
        raise ValueError(
            f"Parameter {name!r} declares no `source`/`producer`; it has no file to read "
            f"(a literal `value:` is inlined instead)."
        )

    if kind == "source":
        path = _source_file(param, source_dir, name)
        measure = _slot(param, "measure")
        if not measure:
            raise ValueError(
                f"Parameter {name!r}: a sourced constant must name its array with `measure:` for a backend to read it back."
            )
        return path, _checked_key(path, str(measure), name)

    # The artifact is content-addressed on the SAME key the in-memory cache uses.
    import hashlib

    producer = _slot(param, "producer")
    module, fname, kwargs = _producer_spec(producer, name, context)
    key = _producer_key(module, fname, kwargs)
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]

    root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{module}.{fname}.{digest}.h5"

    if not path.exists():
        _write_bundle(path, _producer_bundle(producer, name, context, (module, fname, kwargs), key))
    return path, _checked_key(path, _slot(producer, "output", None), name)


def _checked_key(path: Path, key: Optional[str], name: str) -> str:
    """The dataset ``key`` names in ``path``, verified to exist.

    Checked here rather than left to run time: codegen bakes this key into the generated
    module, so an unverified one turns a typo into a failure inside a simulation, far
    from the declaration that caused it. A cache hit skips the write entirely, so this is
    the only place the key is ever seen against the artifact.
    """
    import h5py

    with h5py.File(path, "r") as f:
        keys = sorted(f)
        if key is None:
            # No `output`/`measure`: unambiguous only when the artifact holds one array.
            if len(keys) == 1:
                return keys[0]
            raise ValueError(
                f"Parameter {name!r}: {path.name} holds {len(keys)} arrays {keys}; name "
                f"the one to read with `output:` (produced) or `measure:` (sourced)."
            )
        if str(key) not in f:
            raise ValueError(f"Parameter {name!r}: {path.name} has no array {str(key)!r} (it holds {keys}).")
    return str(key)


def _write_bundle(path: Path, produced: Any) -> None:
    """Write a producer's result to its cache artifact.

    The whole bundle is written, not just one entry: a precompute typically emits every
    operator at once, so writing them together makes the next parameter naming a sibling
    output a cache hit rather than a re-run.
    """
    import os

    import h5py

    arrays = produced if isinstance(produced, dict) else {"value": produced}
    # A temp path unique to this process: several cluster shards materialise the same
    # artifact concurrently, and a shared temp would let one truncate another's file
    # mid-write and then both rename, publishing a corrupt artifact that every later
    # shard reads as a valid cache hit. The rename itself is atomic, so readers only
    # ever see a complete file.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(str(k), data=np.asarray(v))
        tmp.replace(path)
    finally:
        # A failed write must not strand a temp file in the cache dir.
        if tmp.exists():
            tmp.unlink()


def _declared_producers(root: Any, _seen: Optional[set] = None):
    """Every object under *root* that declares a ``producer:``, wherever it sits.

    Walked generically rather than from a list of the places producers are allowed: a
    parameter, a noise covariance and a coupling weight all declare one, and a reclaimer
    that enumerated those three would silently spare — that is, treat as dead — every
    artifact belonging to the fourth.
    """
    _seen = set() if _seen is None else _seen
    if root is None or id(root) in _seen or isinstance(root, (str, bytes, int, float, bool)):
        return
    _seen.add(id(root))
    if isinstance(root, dict):
        children = list(root.values())
    elif isinstance(root, (list, tuple, set)):
        children = list(root)
    else:
        if _slot(root, "producer") is not None:
            yield root
        children = list(getattr(root, "__dict__", {}).values())
    for child in children:
        yield from _declared_producers(child, _seen)


def live_artifacts(root: Any, cache_dir: Optional[Path] = None) -> tuple[set, set]:
    """``(paths, producers)`` this study still reaches in the produced-constant store.

    *producers* is every ``module.function`` whose liveness could be decided; a producer
    whose arguments cannot be resolved is left out of BOTH sets, so its artifacts are
    never judged dead on the strength of a failure to look at them.
    """
    import hashlib

    root_dir = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    paths, producers = set(), set()
    for owner in _declared_producers(root):
        name = str(_slot(owner, "name", "<unnamed>"))
        try:
            module, fname, kwargs = _producer_spec(_slot(owner, "producer"), name, root)
            digest = hashlib.sha256(repr(_producer_key(module, fname, kwargs)).encode()).hexdigest()[:16]
        except Exception:
            continue
        producers.add(f"{module}.{fname}")
        paths.add(root_dir / f"{module}.{fname}.{digest}.h5")
    return paths, producers


def superseded_artifacts(root: Any, cache_dir: Optional[Path] = None) -> list:
    """Artifacts of THIS study's producers that it no longer reaches, newest first.

    Superseded, not merely old: the content address keys on the producing call *and* on
    its module's source, so an artifact of a producer this study uses, at a digest this
    study no longer computes, can only be a version left behind by an edit or by a
    changed argument. Files belonging to producers not seen here are never listed — they
    may well be another study's, and this reads one study.
    """
    root_dir = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    if not root_dir.is_dir():
        return []
    keep, producers = live_artifacts(root, cache_dir)
    dead = [p for p in sorted(root_dir.glob("*.h5")) if p not in keep and p.name.rsplit(".", 2)[0] in producers]
    return sorted(dead, key=lambda p: -p.stat().st_size)


def resolve(param: Any, source_dir: Optional[Path] = None, context: Any = None) -> Any:
    """This parameter's value, resolving ``source``/``producer`` on demand.

    Returns the literal ``value`` untouched when there is one, so a scalar costs
    nothing. Returns ``None`` for a parameter that declares no value at all (a free
    parameter, say) rather than raising — the caller decides whether that is an error.

    ``context`` is the owning experiment (or a network); a producer argument naming an
    entity — ``positions: network.nodes.position`` — resolves against it. Resolved
    arrays are cached and returned **read-only**: they are shared, and a resolved
    constant is not the caller's to modify.
    """
    kind = _provenance(param)
    if kind == "value":
        return _slot(param, "value")
    if kind is None:
        return None

    name = _declared_name(param)
    if kind == "producer":
        return _call_producer(_slot(param, "producer"), name, context)

    path = _source_file(param, source_dir, name)
    measure = _slot(param, "measure")
    key = _source_key(path, measure)
    if key not in _CACHE:
        _CACHE[key] = _readonly(_read_source(path, measure))
    return _CACHE[key]
