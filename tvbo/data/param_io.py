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

# On-disk home for materialised produced constants, alongside the network cache
# (`~/.tvbo/networks`). A derived constant is deterministic in its producing call, so it
# is cached across runs and shared by every parameter naming that call.
CACHE_DIR = Path.home() / ".tvbo" / "constants"


def clear_cache() -> None:
    """Drop every resolved array. Mainly for tests and long-lived processes."""
    _CACHE.clear()


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
    """Mark a cached array read-only.

    One buffer is shared by every parameter naming the same array, so an in-place write
    by one consumer would silently corrupt the others — action at a distance that would
    surface in an unrelated run. A resolved constant is conceptually immutable (it IS
    the declared value), so flagging it turns that into a ValueError at the offending
    line. A caller that genuinely needs to modify copies explicitly.
    """
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for v in value.values():
            _readonly(v)
    return value


# ------------------------------------------------------------------------ references

# Only a fully-qualified `network.*` value is a reference; anything else is a literal
# (a bare `weight` is a state variable, not the connectome) — the same rule the pipeline
# argument path uses, so a producer and a pipeline step read arguments identically.
_REF_PREFIX = "network."


def _network_positions(net: Any) -> np.ndarray:
    """Node coordinates as (n_nodes, 3), in declared node order."""
    out = []
    for node in (_slot(net, "nodes", None) or []):
        pos = _slot(node, "position")
        if pos is None:
            raise ValueError(
                f"network.nodes.position: node {_slot(node, 'id', '?')!r} has no position."
            )
        out.append([pos.x, pos.y, getattr(pos, "z", 0.0) or 0.0])
    return np.asarray(out, dtype=float)


def _mesh_array(net: Any, field: str) -> np.ndarray:
    """A mesh array off the Network's lazy runtime caches (set by the h5 load path)."""
    attr = {"vertices": "_mesh_vertices", "elements": "_mesh_elements",
            "faces": "_mesh_elements", "normals": "_mesh_normals"}.get(field)
    if attr is None:
        raise ValueError(
            f"network.mesh.{field}: unknown mesh array; expected one of "
            f"vertices, elements/faces, normals."
        )
    try:
        return np.asarray(object.__getattribute__(net, attr))
    except AttributeError:
        raise ValueError(
            f"network.mesh.{field}: the network carries no mesh (nothing loaded into "
            f"{attr}); does its companion have a mesh/ group?"
        ) from None


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

    rest = ref[len(_REF_PREFIX):]
    if rest == "nodes.position":
        return _network_positions(net)
    if rest.startswith("mesh."):
        return _mesh_array(net, rest.split(".", 1)[1])
    if rest.startswith("edges."):
        label = rest.split(".", 1)[1]
        mat = net.matrix(label) if hasattr(net, "matrix") else None
        if mat is None:
            raise ValueError(f"{where}: the network has no {label!r} matrix.")
        return np.asarray(mat, dtype=float)
    raise ValueError(
        f"{where}: unsupported reference {ref!r}; supported are network.nodes.position, "
        f"network.mesh.<vertices|elements|normals>, network.edges.<label>."
    )


def _hashable(value: Any) -> Any:
    """A stable, hashable rendering of an argument value (lists/dicts included)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tobytes()
    return value


def _producer_key(module: str, name: str, kwargs: dict) -> tuple:
    """Keyed on the CALL, deliberately not on ``output``.

    One producer typically returns a bundle of named arrays (a precompute emitting every
    mesh operator at once). Keying per-output would re-run that call once per parameter
    reading it — the expensive thing the cache exists to avoid.
    """
    return ("producer", module, name, _hashable(kwargs))


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
    raise ValueError(
        f"{path} holds {len(arrays)} arrays {sorted(arrays)}; the parameter must name "
        f"one with `measure:`."
    )


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
        raise ValueError(
            f"Parameter {param_name!r} declares a producer with no `callable:`."
        )
    module, name = str(_slot(call, "module", "")), str(_slot(call, "name", ""))
    if not module or not name:
        raise ValueError(
            f"Parameter {param_name!r}: producer callable needs both `module` and `name` "
            f"(got module={module!r}, name={name!r})."
        )
    kwargs = _argument_values(producer, context, f"Parameter {param_name!r} producer")
    return module, name, kwargs


def _producer_bundle(producer: Any, param_name: str, context: Any) -> Any:
    """Everything the producer returns, cached on the CALL — no output selection.

    Kept separate from selection so a caller that needs the whole result (writing the
    cache artifact) gets every named array, not just the one entry some parameter asked
    for; writing only that entry would make each sibling output a fresh re-run.

    The callable resolves by bare module name against the recipe's ``code_source``
    (already on ``sys.path`` once the study is loaded), so a study's own code produces
    its own derived constants without that code living in core tvbo.
    """
    import importlib

    module, name, kwargs = _producer_spec(producer, param_name, context)
    key = _producer_key(module, name, kwargs)
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
    module, name, _ = _producer_spec(producer, param_name, context)
    produced = _producer_bundle(producer, param_name, context)
    output = _slot(producer, "output", None)

    if output:
        # A callable returning several named arrays (e.g. a bundle of mesh operators)
        # is selected by `output`, so one precompute serves many parameters and runs once.
        try:
            produced = produced[str(output)]
        except (TypeError, KeyError, IndexError) as exc:
            keys = sorted(produced) if hasattr(produced, "keys") else type(produced).__name__
            raise ValueError(
                f"Parameter {param_name!r}: producer {module}.{name} has no output "
                f"{output!r} (it returned {keys})."
            ) from exc
    elif isinstance(produced, dict):
        # Hand back a shallow copy of the bundle: its arrays are read-only, but the dict
        # itself is the cache entry, and rebinding a key in it would poison every other
        # parameter reading this producer.
        return dict(produced)
    return produced


# ------------------------------------------------------------------------------- API

def is_lazy(param: Any) -> bool:
    """True when this parameter's value is resolved rather than inlined.

    The one rule codegen branches on: a literal is materialised and inlined; anything
    obtained (``source``) or derived (``producer``) is read at run time and must never
    be embedded in generated source.
    """
    return _slot(param, "value") is None and bool(
        _slot(param, "source") or _slot(param, "producer")
    )


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
    name = str(_slot(param, "name", "<unnamed>"))
    if not is_lazy(param):
        raise ValueError(
            f"Parameter {name!r} declares no `source`/`producer`; it has no file to read "
            f"(a literal `value:` is inlined instead)."
        )

    source = _slot(param, "source")
    if source:
        path = _resolve_path(str(source), source_dir)
        if path is None:
            raise ValueError(
                f"Parameter {name!r}: source {source!r} does not resolve to an existing "
                f"file (source_dir={source_dir})."
            )
        measure = _slot(param, "measure")
        if not measure:
            raise ValueError(
                f"Parameter {name!r}: a sourced constant must name its array with "
                f"`measure:` for a backend to read it back."
            )
        return path, str(measure)

    # Produced: materialise once, content-addressed on the producing call so an edited
    # argument (a different k_ring) is a different artifact rather than a stale hit.
    import hashlib

    producer = _slot(param, "producer")
    module, fname, kwargs = _producer_spec(producer, name, context)
    digest = hashlib.sha256(
        repr(_producer_key(module, fname, kwargs)).encode()
    ).hexdigest()[:16]

    root = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{module}.{fname}.{digest}.h5"
    key = str(_slot(producer, "output", None) or "value")

    if not path.exists():
        _write_bundle(path, _producer_bundle(producer, name, context), key)
    return path, key


def _write_bundle(path: Path, produced: Any, key: str) -> None:
    """Write a producer's result to its cache artifact.

    The whole bundle is written, not just the requested entry: one precompute typically
    emits every operator, and writing them together means the next parameter naming a
    sibling output is a cache hit rather than a re-run.
    """
    import h5py

    arrays = produced if isinstance(produced, dict) else {key: produced}
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(str(k), data=np.asarray(v))
    # Rename last: a concurrent reader (a cluster shard racing on the same artifact)
    # must never observe a half-written file at the real path.
    tmp.replace(path)


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
    value = _slot(param, "value")
    if value is not None:
        return value

    name = str(_slot(param, "name", "<unnamed>"))
    producer = _slot(param, "producer")
    if producer is not None:
        return _call_producer(producer, name, context)

    source = _slot(param, "source")
    if not source:
        return None

    path = _resolve_path(str(source), source_dir)
    if path is None:
        raise ValueError(
            f"Parameter {name!r}: source {source!r} does not resolve to an existing "
            f"file (source_dir={source_dir})."
        )
    measure = _slot(param, "measure")
    key = _source_key(path, measure)
    if key not in _CACHE:
        _CACHE[key] = _readonly(_read_source(path, measure))
    return _CACHE[key]
