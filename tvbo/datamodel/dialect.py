"""The TVBO YAML dialect — one implementation, both construction paths.

TVBO authors write a dialect the schema does not describe on its own: a slot may be
written under a declared alias (``dt`` for ``step_size``), an object with one obvious
field may be written as a bare scalar (``omega: 0.0628`` for ``omega: {value: 0.0628}``),
and an entity may be named by ``iri`` instead of spelled out. None of the three is
something LinkML's loaders apply — ``aliases:`` is documentation to them, and
``simple_dict_value`` is specified but unimplemented.

Both are therefore applied here, from tables read off the schema at build time
(:mod:`tvbo.datamodel.dialect_tables`). The generated dataclasses install
:func:`install_on_dataclasses`, which folds the dialect in ``__init__``; the Pydantic
models call :func:`normalize` from a ``mode="before"`` validator. Sharing this module is
what keeps them from drifting: the dialect used to live twice, and the copy the Pydantic
path carried had the aliases but not the scalar shortcuts, so a model written the way the
README shows validated on one path and was rejected on the other.

Every function here takes and returns plain data for ONE level of nesting. Recursion
belongs to the caller — Pydantic already descends into members, and the dataclass
``__init__`` wrapper is reached once per constructed object.
"""

from __future__ import annotations

import warnings
from functools import cache

from tvbo.datamodel.dialect_tables import SCALAR_SHORTCUTS, SLOT_ALIASES

__all__ = [
    "SCALAR_SHORTCUTS",
    "SLOT_ALIASES",
    "expand_iri",
    "fold_aliases",
    "is_literal",
    "lift_scalar",
    "normalize",
    "install_on_dataclasses",
]

_SCALARS = (str, int, float, bool)


def is_literal(value) -> bool:
    """A bare value the shortcut may lift: a scalar, or a (nested) list of scalars.

    An array literal counts — a coordinate list to select, a coefficient matrix — because
    the slot it lifts into holds arrays as well as scalars. A list of MAPPINGS does not:
    that is the list spelling of a keyed collection, whose members lift individually.
    """
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_literal(v) for v in value)
    return False


def lift_scalar(value, target, multivalued, keyed=False):
    """``0.0628`` -> ``{'value': 0.0628}``, leaving an already-written mapping alone.

    On a multivalued slot the members are lifted, not the collection: ``{omega: 0.0628}``
    is a keyed collection of one Parameter, not a Parameter. A ``keyed`` collection's LIST
    spelling (``arguments: [v]``) is a list of member identifiers, not values, so its bare
    scalars are left for the loader to key on; only a non-keyed list
    (``additional_equations: ["x = -x"]`` -> ``[{rhs: "x = -x"}]``) lifts its elements.
    """
    if not multivalued:
        return {target: value} if is_literal(value) else value
    if isinstance(value, dict):
        return {k: ({target: v} if is_literal(v) else v) for k, v in value.items()}
    if isinstance(value, list):
        if keyed:
            return value
        return [({target: v} if is_literal(v) else v) for v in value]
    return value


def fold_aliases(cls_name: str, data: dict) -> dict:
    """Rename *cls_name*'s declared aliases to their canonical slots, in place.

    Class-scoped on purpose: an alias is only an alias where its class declares it.
    ``target_variable`` is an ``Edge`` alias for ``target_var`` but the canonical slot on
    a stimulus ``Event``, so a table keyed by slot name alone would rename it in the one
    place it must not be.
    """
    for alias, canonical in SLOT_ALIASES.get(cls_name, {}).items():
        if alias not in data:
            continue
        value = data.pop(alias)
        if canonical in data:
            warnings.warn(
                f"{cls_name} got both {alias!r} and its canonical slot {canonical!r}; ignoring {alias!r}.",
                stacklevel=4,
            )
        else:
            data[canonical] = value
    return data


@cache
def _curated_entry(cls_name: str, iri: str) -> dict | None:
    """The curated record *iri* names, alias-folded and ready to merge, or ``None``.

    Cached because a recipe naming the same entity twice — every node of a homogeneous
    network — would otherwise re-read and re-parse the same file per object. The result is
    only ever fed to :func:`tvbo.utils.deep_merge`, which mutates neither side, so callers
    share one instance safely.

    The entry's own ``iri`` is dropped: keeping it would make the expanded record ask to be
    expanded again on every later construction, and a self-referential entry would not
    terminate.
    """
    import yaml

    from tvbo.data.registry import local_name, resolve

    try:
        path = resolve(cls_name, local_name(iri))
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        return None
    entry.pop("iri", None)
    return fold_aliases(cls_name, entry)


def expand_iri(cls_name: str, data: dict) -> dict:
    """Fill *data* from the curated record its ``iri`` names, letting the recipe win.

    Naming an entity by ``iri`` instead of spelling it out is the same kind of dialect as
    an alias or a bare scalar: a spelling the schema does not describe. It is resolved here,
    before validation, because this is the only point that still knows which keys the recipe
    actually wrote — after construction every slot carrying a schema default reads as though
    it had been authored, and "the recipe did not say" becomes unanswerable. That is why a
    curated ``delayed:`` was never applied, and why an explicit value equal to a default
    could be overwritten by the entry.

    Only the curated database is consulted. It is a local file read and its content is
    ordered, whereas the ontology answers with an unordered set — a different parameter
    order per process, which no frozen record can be written against. Reaching it is
    :meth:`IriEnrichable.enrich`, which the caller asks for.

    Only a *reference* expands. A record that also states its own ``name`` is a definition,
    and its ``iri`` is grounding — "this model is a ReducedWongWang in the ontology" — not
    an instruction to inherit. Fifty curated files are written that way, and expanding them
    would re-derive a definition from a name lookup: ``ReducedWongWangFunc.yml`` declares
    ``name: ReducedWongWang``, so it would be overwritten by the canonical
    ``ReducedWongWang.yaml`` it merely relates to.

    An ``iri`` naming nothing is left alone: it may point at an entity that exists only in
    the ontology, and this pass cannot tell that from a typo. ``tvbo validate`` is where a
    name that resolves nowhere is reported.
    """
    iri = data.get("iri")
    if not isinstance(iri, str) or data.get("name") is not None:
        return data
    entry = _curated_entry(cls_name, iri)
    if entry is None:
        return data

    from tvbo.utils import deep_merge

    merged = deep_merge(entry, data)
    data.clear()
    data.update(merged)
    return data


def normalize(cls_name: str, data: dict) -> dict:
    """Fold *cls_name*'s dialect into *data*, in place: aliases, ``iri``, shortcuts.

    Aliases fold first, so the recipe and the curated record are keyed alike before they
    are merged and the shortcut pass can see every value under the name it looks for.
    Lifting first left ``BoundaryCondition(value="0")`` — the older spelling of
    ``equation`` — a bare string where the generated ``__post_init__`` wanted a mapping,
    and it raised.
    """
    fold_aliases(cls_name, data)
    expand_iri(cls_name, data)

    for slot, (target, multivalued, keyed) in SCALAR_SHORTCUTS.get(cls_name, {}).items():
        if data.get(slot) is not None:
            data[slot] = lift_scalar(data[slot], target, multivalued, keyed)
    return data


def install_on_dataclasses(namespace: dict) -> None:
    """Wrap each generated dataclass's ``__init__`` so it accepts the dialect.

    ``__init__`` is the one place where a keyword is known to name a slot of this class,
    so an alias resolves without guessing whether a mapping is an instance or a keyed
    collection, and a free-form key (a parameter literally named ``dt``) is never mistaken
    for one. Every construction path — the LinkML loaders, ``cls(**data)``, nested and
    inlined members, subclasses — goes through it.
    """

    def wrap(cls):
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            normalize(cls.__name__, kwargs)
            original(self, *args, **kwargs)

        cls.__init__ = __init__

    for name in set(SLOT_ALIASES) | set(SCALAR_SHORTCUTS):
        cls = namespace.get(name)
        if cls is not None:
            wrap(cls)
