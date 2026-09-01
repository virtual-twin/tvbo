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

from tvbo.datamodel.dialect_tables import KEYED_COLLECTIONS, SCALAR_SHORTCUTS, SLOT_ALIASES

__all__ = [
    "KEYED_COLLECTIONS",
    "SCALAR_SHORTCUTS",
    "SEMANTIC_FOLDS",
    "SLOT_ALIASES",
    "curated_entry",
    "expand_iri",
    "fold_aliases",
    "is_literal",
    "key_members",
    "lift_scalar",
    "normalize",
    "install_on_dataclasses",
    "peer_module",
]

_SCALARS = (str, int, float, bool)


def peer_module(instance):
    """The generated module *instance*'s class comes from.

    A record is filled with members — an ``Equation``, a ``Parameter`` — and those have to
    be of the same generated form as the record itself: the strict Pydantic models validate
    on assignment and reject a LinkML dataclass where they want their own peer. Behaviour
    that builds members reads the peer off the instance rather than importing one form,
    which is what lets one implementation serve both.
    """
    import importlib

    return importlib.import_module(type(instance).__module__)


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
def curated_entry(cls_name: str, name: str) -> dict | None:
    """The curated *cls_name* record called *name*, alias-folded and ready to merge.

    ``None`` when the database holds no such record — including when *cls_name* is not a
    category it keeps at all.

    Cached because a recipe naming the same entity twice — every node of a homogeneous
    network — would otherwise re-read and re-parse the same file per object. Callers must
    not mutate the result: :func:`expand_iri` only feeds it to
    :func:`tvbo.utils.deep_merge`, which mutates neither side, and
    :meth:`IriEnrichable._from_database` only reads it into a constructor.

    The entry's own ``iri`` is dropped: keeping it would make the expanded record ask to be
    expanded again on every later construction, and a self-referential entry would not
    terminate. So is its envelope: :func:`normalize` strips the recipe's own ``tvbo_class``
    before expanding, and a curated record that carries one would put it back. Every network
    sidecar in the database opens with ``tvbo_class: tvbo:Network``, so a recipe naming a
    curated network by ``iri`` reached ``Network.__init__`` with a keyword it has no slot for.
    """
    import yaml

    from tvbo.data.registry import resolve
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS

    try:
        path = resolve(cls_name, name)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        return None
    entry.pop("iri", None)
    for envelope_key in ENVELOPE_KEYS:
        entry.pop(envelope_key, None)
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
    :meth:`tvbo.behaviour._enrich.IriEnrichable.enrich`, which the caller asks for.

    Only a *reference* expands. A record that also states its own ``name`` is a definition,
    and its ``iri`` is grounding — "this model is a ReducedWongWang in the ontology" — not
    an instruction to inherit. Fifty curated files are written that way, and expanding them
    would re-derive a definition from a name lookup: ``ReducedWongWangFunc.yaml`` states its
    own name and grounds on ``tvbo:ReducedWongWang``, so expanding it would replace a
    distinct record with the canonical ``ReducedWongWang.yaml`` it merely relates to.

    An ``iri`` naming nothing is left alone: it may point at an entity that exists only in
    the ontology, and this pass cannot tell that from a typo. ``tvbo validate`` is where a
    name that resolves nowhere is reported.

    Once a reference has expanded, the ``iri`` survives only for a class that keeps one as a
    slot, where it is grounding worth recording. ``Network`` does not: its curated record is
    reached through ``iri`` but its own connectivity is the ``data_file`` the expansion just
    merged in, so keeping the key would hand ``Network.__init__`` a keyword it has no slot
    for. That is what stopped a study from naming a curated connectome by ``iri`` at all.
    """
    from tvbo.data.registry import local_name

    iri = data.get("iri")
    if not isinstance(iri, str) or data.get("name") is not None:
        return data
    entry = curated_entry(cls_name, local_name(iri))
    if entry is None:
        return data

    from tvbo.utils import deep_merge

    merged = deep_merge(entry, data)
    if not _keeps_iri_slot(cls_name):
        merged.pop("iri", None)
    data.clear()
    data.update(merged)
    return data


def _keeps_iri_slot(cls_name: str) -> bool:
    """Whether the generated *cls_name* declares ``iri`` as a slot of its own.

    Unknown classes answer True, so a name this module cannot resolve is left exactly as the recipe wrote it.
    """
    import dataclasses

    from tvbo.datamodel import schema

    cls = getattr(schema, cls_name, None)
    if cls is None or not dataclasses.is_dataclass(cls):
        return True
    return any(field.name == "iri" for field in dataclasses.fields(cls))


def key_members(cls_name: str, data: dict) -> dict:
    """Give a keyed collection the mapping spelling, each member named by its key.

    ``parameters: {TR: {value: 720.0}}`` means a Parameter *called* ``TR``, so writing the
    name a second time inside the member is the redundancy this project's records are
    written without. The generated dataclasses fill it from the key in ``__post_init__``;
    the generated Pydantic models leave it missing and reject the member as incomplete, so
    a record that loaded on one form failed on the other.

    The same collection may equally be written as a LIST — of bare identifiers
    (``arguments: [v]``) or of whole members — which the dataclasses key and the Pydantic
    models reject outright as not a mapping. Both spellings arrive as the mapping here, so
    which one a record uses stops being a question of which form is loading it. A list
    whose members state no identifier is left alone: there is nothing to key it by, and
    that is a record to reject rather than to guess at.

    Members are rebuilt rather than mutated: the mapping may be a cached curated entry,
    shared by every object that names it.
    """
    for slot, identifier in KEYED_COLLECTIONS.get(cls_name, {}).items():
        members = data.get(slot)
        if isinstance(members, list):
            members = _as_mapping(members, identifier)
        if not isinstance(members, dict):
            continue
        data[slot] = {
            key: ({identifier: key, **member} if isinstance(member, dict) and identifier not in member else member)
            for key, member in members.items()
        }
    return data


def _as_mapping(members: list, identifier: str) -> list | dict:
    """A list-spelled keyed collection as its mapping, or unchanged if it cannot be keyed."""
    keyed = {}
    for member in members:
        if isinstance(member, str):
            keyed[member] = {identifier: member}
        elif isinstance(member, dict) and isinstance(member.get(identifier), str):
            keyed[member[identifier]] = member
        else:
            return members
    return keyed


def _fold_state_variable_domain(data: dict) -> None:
    """``range``/``boundaries`` mean more than ``domain`` spelled differently.

    ``boundaries`` is a hard clamp and ``range`` is descriptive, so folding either one by
    a plain rename would silently promote a description into an enforced bound. The fold
    that keeps them apart lives with the loader; this is where the Python construction
    path reaches it.
    """
    from tvbo.utils.yaml_loader import _fold_one_state_variable_domain

    _fold_one_state_variable_domain(data)


def _reject_output_as_definitions(data: dict) -> None:
    """``output`` lists names. Definitions there are accepted and quietly ruined.

    The slot is a list of strings, so a mapping of variable definitions is coerced to
    ``["JsonObj(x=JsonObj(equation=...))"]`` — a model that constructs, renders and runs
    while naming an output that does not exist.
    """
    output = data.get("output")
    if not isinstance(output, dict):
        return
    first = next(iter(output.values()), None)
    if isinstance(first, dict) and ("equation" in first or "rhs" in first):
        raise ValueError(
            "'output' should be a list of variable names, not variable definitions. "
            "Did you mean 'derived_variables'?\n\n"
            "Change:\n"
            "  output:\n"
            "    x:\n"
            "      equation: ...\n\n"
            "To:\n"
            "  derived_variables:\n"
            "    x:\n"
            "      equation: ...\n"
            "  output: [x]  # optional: list of outputs to include"
        )


def _reject_unknown_output(data: dict) -> None:
    """An ``output`` entry names a channel of this model, so a name it does not declare is a typo.

    Checked here because it is the one place that is deterministic, local and cheap: two
    key lookups against collections already in hand. Parsing every equation is what used to
    answer it, from inside a constructor that no longer does any work — so between that
    constructor emptying and this fold the check was simply absent, and a model naming a
    channel that does not exist built without complaint.
    """
    declared = set(data.get("derived_variables") or ()) | set(data.get("state_variables") or ())
    for name in data.get("output") or ():
        if isinstance(name, str) and name not in declared:
            raise ValueError(f"Output variable '{name}' not found in derived_variables or state_variables")


def _fold_dynamics(data: dict) -> None:
    """The two things a `Dynamics` record cannot say for itself."""
    _reject_output_as_definitions(data)
    _reject_unknown_output(data)


SEMANTIC_FOLDS = {
    "StateVariable": _fold_state_variable_domain,
    "Dynamics": _fold_dynamics,
}
"""Per-class dialect a table of renames cannot express, keyed like the other tables.

Applied after the members are keyed, so a fold sees each collection under the names its
own class will, whichever spelling the record was written in.
"""


def normalize(cls_name: str, data: dict) -> dict:
    """Fold *cls_name*'s dialect into *data*, in place: aliases, ``iri``, shortcuts, keys.

    Aliases fold first, so the recipe and the curated record are keyed alike before they
    are merged and the shortcut pass can see every value under the name it looks for.
    Lifting first left ``BoundaryCondition(value="0")`` — the older spelling of
    ``equation`` — a bare string where the generated ``__post_init__`` wanted a mapping,
    and it raised. Keying comes last, once every member is a mapping that can carry a name.

    The semantic folds come last, after keying, so a fold reads each collection under the
    names the class will — a record spelling one as a list is not a different case to it.
    The terse ``distribution`` lift follows the domain fold, since a clamp folded out of
    ``boundaries`` can leave one behind for it to complete.

    The document envelope goes first. ``tvbo_class`` states which class a *file* holds,
    which is a fact about the file and never a slot, so every constructor route drops it.
    """
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS, _lift_one_distribution

    for envelope_key in ENVELOPE_KEYS:
        data.pop(envelope_key, None)

    fold_aliases(cls_name, data)
    expand_iri(cls_name, data)

    for slot, (target, multivalued, keyed) in SCALAR_SHORTCUTS.get(cls_name, {}).items():
        if data.get(slot) is not None:
            data[slot] = lift_scalar(data[slot], target, multivalued, keyed)
    key_members(cls_name, data)

    semantic = SEMANTIC_FOLDS.get(cls_name)
    if semantic is not None:
        semantic(data)
    _lift_one_distribution(data)
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

    for name in set(SLOT_ALIASES) | set(SCALAR_SHORTCUTS) | set(SEMANTIC_FOLDS):
        cls = namespace.get(name)
        if cls is not None:
            wrap(cls)
