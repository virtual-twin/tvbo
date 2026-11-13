"""The TVBO YAML dialect, folded into every record before it is validated.

TVBO authors write a dialect the schema does not describe on its own: a slot may be written under a declared alias (``dt`` for ``step_size``), an object with one obvious field may be written as a bare scalar (``omega: 0.0628`` for ``omega: {value: 0.0628}``), a keyed collection names each member by its key, and an entity may be named by ``iri`` instead of spelled out. None of these is something LinkML's generators apply — ``aliases:`` is documentation to them, and ``simple_dict_value`` is specified but unimplemented.

They are therefore applied here, from tables read off the schema at build time (:mod:`tvbo.datamodel.dialect_tables`): the generated models call :func:`normalize` from a ``mode="before"`` validator, so every construction path — a loaded document, ``cls(**data)``, a nested member — folds the same dialect.

Every function here takes and returns plain data for ONE level of nesting. Recursion belongs to the caller: Pydantic descends into members, and the validator is reached once per constructed object.
"""

from __future__ import annotations

import os
import warnings
from functools import cache

from tvbo.datamodel.dialect_tables import (
    KEYED_COLLECTIONS,
    LIST_SLOTS,
    SCALAR_SHORTCUTS,
    SLOT_ALIASES,
    SLOT_DEFAULTS,
    UNIT_SLOTS,
)

__all__ = [
    "identifier_field",
    "KEYED_COLLECTIONS",
    "LIST_SLOTS",
    "RETIRED_SLOTS",
    "SCALAR_SHORTCUTS",
    "SEMANTIC_FOLDS",
    "SLOT_ALIASES",
    "UNIT_SLOTS",
    "curated_entry",
    "expand_iri",
    "fold_aliases",
    "fold_node_mapping",
    "fold_slot",
    "is_literal",
    "key_members",
    "lift_scalar",
    "lift_shortcuts",
    "normalize",
    "paths_as_text",
    "refuse_dotted_parent",
    "refuse_retired",
]

_SCALARS = (str, int, float, bool)


def is_literal(value) -> bool:
    """A bare value the shortcut may lift: a scalar, or a (nested) list of scalars.

    An array literal counts — a coordinate list to select, a coefficient matrix — because the slot it lifts into holds arrays as well as scalars. A list of MAPPINGS does not: that is the list spelling of a keyed collection, whose members lift individually.
    """
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_literal(v) for v in value)
    return False


def lift_scalar(value, target, multivalued, keyed=False):
    """``0.0628`` -> ``{'value': 0.0628}``, leaving an already-written mapping alone.

    On a multivalued slot the members are lifted, not the collection: ``{omega: 0.0628}`` is a keyed collection of one Parameter, not a Parameter. A ``keyed`` collection's LIST spelling (``arguments: [v]``) is a list of member identifiers, not values, so its bare scalars are left for the loader to key on; only a non-keyed list (``additional_equations: ["x = -x"]`` -> ``[{rhs: "x = -x"}]``) lifts its elements.
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


def lift_shortcuts(cls_name: str, data: dict) -> dict:
    """Lift *cls_name*'s scalar shortcuts in *data* to the mappings they abbreviate, in place (see :func:`lift_scalar`)."""
    for slot, (target, multivalued, keyed) in SCALAR_SHORTCUTS.get(cls_name, {}).items():
        if data.get(slot) is not None:
            data[slot] = lift_scalar(data[slot], target, multivalued, keyed)
    return data


def fold_aliases(cls_name: str, data: dict) -> dict:
    """Rename *cls_name*'s declared aliases to their canonical slots, in place.

    Class-scoped on purpose: an alias is only an alias where its class declares it. ``target_variable`` is an ``Edge`` alias for ``target_var`` but the canonical slot on a stimulus ``Event``, so a table keyed by slot name alone would rename it in the one place it must not be.
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
    """The curated *cls_name* record called *name*, alias-folded, shortcut-lifted and ready to merge.

    ``None`` when the database holds no such record — including when *cls_name* is not a category it keeps at all. A name several records answer to raises, naming them (:func:`tvbo.data.registry.resolve`), and so does a missing database.

    Cached because a recipe naming the same entity twice — every node of a homogeneous network — would otherwise re-read and re-parse the same file per object. Callers must not mutate the result: :func:`expand_iri` only feeds it to :func:`tvbo.utils.deep_merge`, which mutates neither side, and :meth:`IriEnrichable._from_database` only reads it into a constructor.

    The entry's own ``iri`` is dropped: keeping it would make the expanded record ask to be expanded again on every later construction, and a self-referential entry would not terminate. So is its envelope: :func:`normalize` strips the recipe's own ``tvbo_class`` before expanding, and a curated record that carries one would put it back. Every network sidecar in the database opens with ``tvbo_class: tvbo:Network``, so a recipe naming a curated network by ``iri`` would otherwise carry the curated file's envelope into the object built from the recipe.
    """
    import yaml

    from tvbo.data.registry import _CATEGORIES, resolve
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS

    if cls_name not in _CATEGORIES:
        return None
    try:
        path = resolve(cls_name, name)
    except FileNotFoundError:
        return None
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        return None
    entry.pop("iri", None)
    for envelope_key in ENVELOPE_KEYS:
        entry.pop(envelope_key, None)
    return lift_shortcuts(cls_name, fold_aliases(cls_name, entry))


def expand_iri(cls_name: str, data: dict) -> dict:
    """Fill *data* in place from the curated record its ``iri`` names, the recipe winning at the leaf.

    An ``iri`` points at the curated database, and whatever else the record states overrides that entry field by field, ``name`` included: ``parameters: {a: 2.0}`` replaces only ``a``'s value and keeps the curated definition and domain. Both sides are alias-folded and shortcut-lifted before the merge so they meet at the same depth. No ``name`` is needed beside an ``iri``: the entry supplies it.

    Naming an entity by ``iri`` instead of spelling it out is the same kind of dialect as an alias or a bare scalar: a spelling the schema does not describe. It is resolved here, before validation, because this is the only point that still knows which keys the recipe actually wrote — after construction every slot carrying a schema default reads as though it had been authored, and "the recipe did not say" becomes unanswerable. That is why a curated ``delayed:`` was never applied, and why an explicit value equal to a default could be overwritten by the entry.

    Only the curated database is consulted. It is a local file read and its content is ordered, whereas the ontology answers with an unordered set — a different parameter order per process, which no frozen record can be written against. Reaching it is :meth:`tvbo.behaviour._enrich.IriEnrichable.enrich`, which the caller asks for.

    The record the ``iri`` addresses is :func:`tvbo.data.registry.curated_key`'s answer: a scoped ``tvbo:observation_models/bold_tvb`` names ``bold_tvb`` among observations, an unscoped ``tvbo:BOLD_TVB`` its local name. An ``iri`` naming no curated record merges nothing: it may point at an entity that exists only in the ontology, and this pass cannot tell that from a typo. The record then takes that name as its ``name`` when it states none, and ``tvbo validate`` is where a name that resolves nowhere is reported. A scope of another class, or a ``result/`` reference, is left exactly as written.

    An invocation names what it invokes the same way (:func:`_invoked_class`): a ``FunctionCall``, and every class that is one, written with ``iri: tvbo:functions/<name>`` takes every slot of that curated function it declares itself except the name, its equation or callable and its arguments among them, and states only its own arguments beside it. Its ``iri`` is read as a ``Function`` reads its own, so every other spelling :func:`~tvbo.data.registry.curated_key` accepts for a function names it too, the unscoped ``tvbo:<name>`` among them. A scoped ``iri`` naming no curated function is an error, since the scope says a function was meant and the invocation would have nothing to invoke. An unscoped one naming none merges nothing, as everywhere else, and is an error only where the invocation states no ``equation``, ``callable``, ``class_call``, ``function`` or ``type`` of its own either, since it would then run nothing; ``type`` is how an ``Analysis`` of the solve names what it runs. An invocation runs one thing, so one that names a curated function and also states an ``equation``, ``callable``, ``class_call`` or ``function`` of a kind that function does not record is refused rather than left with two.

    Once an ``iri`` has expanded, it survives only for a class that keeps one as a slot, where it is grounding worth recording. ``Network`` does not: its curated record is reached through ``iri`` but its own connectivity is the ``data_file`` the expansion just merged in, so keeping the key would hand ``Network.__init__`` a keyword it has no slot for.
    """
    from tvbo.data.registry import curated_key
    from tvbo.utils import deep_merge

    iri = data.get("iri")
    if not isinstance(iri, str):
        return data
    addressed = _invoked_class(cls_name)
    key = curated_key(iri, addressed)
    if key is None:
        return data
    entry = curated_entry(addressed, key)
    if addressed != cls_name:
        entry = _invoked_entry(cls_name, iri, addressed, key, entry, data)
    if entry is None:
        if data.get("name") is None and _has_slot(cls_name, "name"):
            data["name"] = key
        return data
    merged = deep_merge(entry, lift_shortcuts(cls_name, fold_aliases(cls_name, data)))
    if not _has_slot(cls_name, "iri"):
        merged.pop("iri", None)
    data.clear()
    data.update(merged)
    return data


INVOCATION_FORMS = ("equation", "callable", "class_call", "function")
"""The slots through which an invocation says what it runs; a record states one of them."""


def _listed(slots, joiner: str = "and", article: bool = True) -> str:
    """Slot names as backticked text in a sentence, the last joined by *joiner* and each with its article unless *article* is off."""
    names = [f"{('an ' if slot[0] in 'aeiou' else 'a ') if article else ''}`{slot}`" for slot in slots]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {joiner} {names[-1]}"


def _invoked_class(cls_name: str) -> str:
    """The class of the curated records an ``iri`` on a *cls_name* record addresses: ``Function`` for an invocation, and *cls_name* itself for every other class.

    An invocation is not filed in the database itself, so its ``iri`` cannot address a record of its own class; it addresses the function it invokes. A ``FunctionCall`` is an invocation, and so is every class that is one, a study-level ``Analysis`` among them.
    """
    from tvbo.datamodel import pydantic

    cls, call = getattr(pydantic, cls_name, None), getattr(pydantic, "FunctionCall", None)
    return "Function" if isinstance(cls, type) and isinstance(call, type) and issubclass(cls, call) else cls_name


def _invoked_entry(cls_name: str, iri: str, addressed: str, key: str, entry: dict | None, data: dict) -> dict | None:
    """The slots of the curated *addressed* record *entry* that the *cls_name* invocation *data* takes, or ``None`` when its *iri* matched no record and it states something of its own to run.

    Raises:
        ValueError: when a scoped *iri* names no curated record, when an unscoped one names none and *data* states no form and no ``type`` of its own either, or when *data* states a form of a kind *entry* does not record.
    """
    from tvbo.data.registry import iri_target

    if entry is None:
        if iri_target(iri) is not None:
            raise ValueError(f"{cls_name} names {iri!r}, and the curated database holds no {addressed} called {key!r}.")
        if data.get("type") is None and all(data.get(form) is None for form in INVOCATION_FORMS):
            raise ValueError(
                f"{cls_name} names {iri!r}, which matched no curated function, and states no {_listed(INVOCATION_FORMS, 'or', article=False)} of its own, so it has nothing to run."
            )
        return None
    recorded = [form for form in INVOCATION_FORMS if entry.get(form) is not None]
    own = [form for form in INVOCATION_FORMS if data.get(form) is not None and form not in recorded]
    if own and recorded:
        raise ValueError(
            f"{cls_name} names {iri!r}, which records {_listed(recorded)}, and states {_listed(own)} of its own; an invocation runs one of them. State only arguments beside the `iri`, or drop the `iri`."
        )
    return {slot: value for slot, value in entry.items() if slot != "name" and _has_slot(cls_name, slot)}


def _has_slot(cls_name: str, slot: str) -> bool:
    """Whether the generated *cls_name* declares *slot* as a slot of its own.

    Unknown classes answer True, so a name this module cannot resolve is left exactly as the recipe wrote it.
    """
    from tvbo.datamodel import pydantic

    cls = getattr(pydantic, cls_name, None)
    if cls is None or not hasattr(cls, "model_fields"):
        return True
    return slot in cls.model_fields


@cache
def identifier_field(model_cls) -> str | None:
    """Name of the slot a keyed collection's key maps onto, for a generated Pydantic model.

    A non-``name``/``id`` key slot marks itself with a ``collection_key`` annotation, because LinkML consumes ``identifier``/``key`` into the field's required-ness and drops them from the emitted metadata, so they cannot be read back here. Otherwise the universal TVBO conventions ``name`` then ``id``. ``None`` when no identifier slot can be determined — the member is then left as it is.
    """
    fields = model_cls.model_fields
    for fname, info in fields.items():
        extra = info.json_schema_extra
        meta = extra.get("linkml_meta") if isinstance(extra, dict) else None
        if isinstance(meta, dict) and "collection_key" in (meta.get("annotations") or {}):
            return fname
    for candidate in ("name", "id"):
        if candidate in fields:
            return candidate
    return None


def key_members(cls_name: str, data: dict) -> dict:
    """Give a keyed collection the mapping spelling, each member named by its key.

    ``parameters: {TR: {value: 720.0}}`` means a Parameter *called* ``TR``, so writing the name a second time inside the member is the redundancy this project's records are written without; the generated models would otherwise reject the member as incomplete.

    The same collection may equally be written as a LIST — of bare identifiers (``arguments: [v]``), of whole members, or of records already built — which the generated models would reject as not a mapping. Both spellings arrive as the mapping here, so a record may use either. A list whose members state no identifier is left alone: there is nothing to key it by, and that is a record to reject rather than to guess at.

    Members are rebuilt rather than mutated: the mapping may be a cached curated entry, shared by every object that names it.
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
        elif not isinstance(member, dict) and isinstance(getattr(member, identifier, None), str):
            keyed[getattr(member, identifier)] = member
        else:
            return members
    return keyed


def _fold_state_variable_domain(data: dict) -> None:
    """``range``/``boundaries`` mean more than ``domain`` spelled differently.

    ``boundaries`` is a hard clamp and ``range`` is descriptive, so folding either one by a plain rename would silently promote a description into an enforced bound. The fold that keeps them apart lives with the loader; this is where the Python construction path reaches it.
    """
    from tvbo.utils.yaml_loader import _fold_one_state_variable_domain

    _fold_one_state_variable_domain(data)


def _reject_output_as_definitions(data: dict) -> None:
    """``output`` lists names: a mapping of variable definitions there is refused by name, rather than as the list slot's one bare member."""
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

    Checked here because it is the one place that is deterministic, local and cheap: two key lookups against collections already in hand. Parsing every equation is what used to answer it, from inside a constructor that no longer does any work — so between that constructor emptying and this fold the check was simply absent, and a model naming a channel that does not exist built without complaint.
    """
    declared = set(data.get("derived_variables") or ()) | set(data.get("state_variables") or ())
    for name in data.get("output") or ():
        if isinstance(name, str) and name not in declared:
            raise ValueError(f"Output variable '{name}' not found in derived_variables or state_variables")


def _fold_dynamics(data: dict) -> None:
    """The two things a `Dynamics` record cannot say for itself."""
    _reject_output_as_definitions(data)
    _reject_unknown_output(data)


UNASSIGNED_NODE_ID = -1
"""The ``id`` a node declared by ``label`` alone carries until the network that materialises it assigns one."""


def _per_node_slots() -> list[str]:
    """The slots ``Node`` adds to ``NodeTemplate`` (``id``, ``label``, ``position``, ``region``), which the network's data gives each node."""
    from tvbo.datamodel import pydantic

    return [name for name in pydantic.Node.model_fields if name not in pydantic.NodeTemplate.model_fields]


def _refuse_per_node_slots(data: dict) -> None:
    """Refuse a ``node_template`` that sets a per-node slot (:func:`_per_node_slots`): the network's data gives each node its own, so the template's value would be ignored."""
    own = [name for name in _per_node_slots() if data.get(name) is not None]
    if own:
        raise ValueError(
            f"`node_template` sets {', '.join(f'`{name}`' for name in own)}; the network's data gives each node its own, and a template sets only what every node shares, so declare a node's own value on that node."
        )


def _fold_node_template(data: dict) -> None:
    """Read a ``node_template`` mapping: drop the ``id: -1`` (:data:`UNASSIGNED_NODE_ID`) on the template of a network an older tvbo saved, and only that value, so such a network still reads, then refuse a per-node slot (:func:`_refuse_per_node_slots`)."""
    if data.get("id") == UNASSIGNED_NODE_ID:
        del data["id"]
    _refuse_per_node_slots(data)


def _fold_label_keyed_node(data: dict) -> None:
    """Give a node that states no ``id`` the unassigned id, so it constructs.

    Such a node is one declared by ``label`` alone, which attaches values to the node of that label in a network whose data materialises its nodes, or a ``node_template``, which every node is built from; only the network knows the index. :meth:`tvbo.classes.network.Network._apply_declared_nodes` replaces the sentinel with it, and raises when no node carries the label.
    """
    if data.get("id") is None:
        data["id"] = UNASSIGNED_NODE_ID


def _fold_network(data: dict) -> None:
    """Refuse a ``node_template`` given as a built ``Node`` by the per-node slots it sets, where pydantic names only the wrong class, then read a ``nodes`` mapping by :func:`fold_node_mapping`."""
    template = data.get("node_template")
    if template is not None and not isinstance(template, dict):
        _refuse_per_node_slots({name: getattr(template, name, None) for name in _per_node_slots()})
    fold_node_mapping(data)


def fold_node_mapping(data: dict) -> None:
    """Read a ``nodes`` mapping as the list it abbreviates: each key is the ``label`` of its node, so ``{L.A1: {...}}`` is ``[{label: L.A1, ...}]``.

    Every key is a label, an integer one included (``{0: {...}}`` is the node labelled ``"0"``), because a mapping names its nodes and a position is what a name avoids; a node that should carry an ``id`` states it in its entry, and one that does not gets the unassigned id here, so an entry holding nothing else still reaches its class whole. The mapping is keyed when each of its entries is a mapping or empty; one that holds a node's own fields (``{id: 0, label: x}``) is the single node it spells and is left as written. An entry stating a ``label`` other than its key is an error.
    """
    nodes = data.get("nodes")
    if not isinstance(nodes, dict) or not all(node is None or isinstance(node, dict) for node in nodes.values()):
        return
    folded = []
    for key, node in nodes.items():
        spec = dict(node or {})
        if spec.get("label") not in (None, key, str(key)):
            raise ValueError(f"Node {key!r} is keyed by its label, so its entry cannot state the label {spec['label']!r}.")
        spec["label"] = str(key)
        _fold_label_keyed_node(spec)
        folded.append(spec)
    data["nodes"] = folded


def refuse_dotted_parent(data: dict) -> None:
    """An edge reaches the network its network is a subnetwork of as ``parent``, and that network's parent as ``parent/parent``; a path spelled with ``..`` segments is refused, naming the edge and the spelling."""
    for slot in ("source_network", "target_network"):
        steps = str(data.get(slot) or "").split("/")
        if set(steps) == {".."}:
            raise ValueError(
                f"Edge {data.get('label')!r}: `{slot}: {data[slot]}` is spelled `{slot}: {'/'.join('parent' for _ in steps)}`."
            )


SEMANTIC_FOLDS = {
    "StateVariable": _fold_state_variable_domain,
    "Dynamics": _fold_dynamics,
    "Node": _fold_label_keyed_node,
    "NodeTemplate": _fold_node_template,
    "Network": _fold_network,
    "Edge": refuse_dotted_parent,
}
"""Per-class dialect a table of renames cannot express, keyed like the other tables.

Applied after the members are keyed, so a fold sees each collection under the names its own class will, whichever spelling the record was written in.
"""


_ACTED_ON = "name the parameter where it is acted on: a fit's `free_parameters`, an update rule's `target_parameter`, a continuation's `free_parameters` or `held_parameters`, or an exploration's `space`"

RETIRED_SLOTS: dict[str, dict[str, str]] = {
    "Coupling": {
        "inner_coupling": "write the coupling on the subnetwork",
        "region_mapping": "write `node_mapping` on the network",
        "regional_connectivity": "the parent network's connectivity is the regional coupling's graph",
        "aggregation": "write the aggregation as an up edge's coupling",
        "distribution": "write the distribution as a down edge's coupling",
    },
    **dict.fromkeys(("Parameter", "DerivedParameter"), {"free": _ACTED_ON}),
}
"""Slots a class no longer has, each with where its meaning is written now. None has a counterpart on its own class to fold into, so a record that still carries one is refused at load rather than renamed or dropped."""


def refuse_retired(cls_name: str, data: dict) -> None:
    """Raise when *data* carries a slot *cls_name* no longer has, naming the slot and where its meaning is written now."""
    for slot, instead in RETIRED_SLOTS.get(cls_name, {}).items():
        if slot in data:
            raise ValueError(f"{cls_name}.{slot} was retired in tvbo 0.9.0: {instead}.")


def canonicalize_units(cls_name: str, data: dict) -> dict:
    """Give each unit slot of *cls_name* in *data* the canonical spelling of a curated unit, in place.

    ``mV/ms`` and ``mV_per_ms`` are one unit, and the published record spells it one way. An uncurated unit stays as written: the open slot records it, `normalize_unit` answers ``None`` for it, and a dimensional check involving it reports ``underdetermined`` rather than guessing.
    """
    from tvbo.utils.units import normalize_unit

    for slot in UNIT_SLOTS.get(cls_name, ()):
        unit = data.get(slot)
        if isinstance(unit, str):
            data[slot] = normalize_unit(unit) or unit
    return data


def paths_as_text(data: dict) -> dict:
    """Give each path in *data*, a slot's value or a member of a list slot, as its text, in place: a record holds the text a recipe writes for a path, never a path object."""
    for slot, value in data.items():
        if _is_path(value):
            data[slot] = os.fspath(value)
        elif isinstance(value, (list, tuple)) and any(_is_path(v) for v in value):
            data[slot] = [os.fspath(v) if _is_path(v) else v for v in value]
    return data


def _is_path(value) -> bool:
    """Whether *value* is a path; a scalar is answered without the slower ``os.PathLike`` check, which a long numeric list would pay per member."""
    return not isinstance(value, _SCALARS) and isinstance(value, os.PathLike)


def wrap_lists(cls_name: str, data: dict) -> dict:
    """Give each list-valued slot of *cls_name* whose one member is written bare the list spelling, in place.

    ``output: x`` is ``output: [x]``, and a ``nodes`` mapping that spells one node's own fields is that one node. A value that is neither a scalar, a mapping nor a record is left as written, so an array stays an array and is refused as itself.
    """
    for slot in LIST_SLOTS.get(cls_name, ()):
        value = data.get(slot)
        if value is None or isinstance(value, (list, tuple)):
            continue
        if isinstance(value, (*_SCALARS, dict)) or hasattr(value, "model_fields"):
            data[slot] = [value]
    return data


def fold_slot(cls_name: str, slot: str, value):
    """*value*, assigned to *cls_name*'s *slot* after construction, in the spelling construction would have given it: its unit canonical, its shortcut lifted, its members keyed, its terse distribution completed and its one bare member wrapped.

    Pydantic validates an assignment against the value assigned, not against the mapping the class's ``mode="before"`` validator returns, so what :func:`normalize` folds there never reaches the slot. The folds that read the one slot apply here. Those that need the whole record or the construction around it stay with :func:`normalize`: ``iri`` expansion, the defaults that fill after it, and the semantic folds, among them reading ``nodes`` as a mapping, whose label-keyed nodes only the network's construction gives ids.
    """
    from tvbo.utils.yaml_loader import _lift_one_distribution

    data = paths_as_text({slot: value})
    canonicalize_units(cls_name, data)
    lift_shortcuts(cls_name, data)
    key_members(cls_name, data)
    _lift_one_distribution(data)
    wrap_lists(cls_name, data)
    return data[slot]


def normalize(cls_name: str, data: dict) -> dict:
    """Fold *cls_name*'s dialect into *data*, in place: aliases, ``iri``, defaults, shortcuts, keys.

    Aliases fold and scalar shortcuts lift first, on the recipe here and on the curated record in :func:`curated_entry`, so the two meet at the same depth when merged: a recipe's ``parameters: {a: 2.0}`` overrides the curated ``a``'s value and keeps its definition and domain, where merging the bare scalar replaced the whole parameter. Folding precedes lifting so the shortcut pass sees every value under the name it looks for. Lifting first left ``BoundaryCondition(value="0")`` — the older spelling of ``equation`` — a bare string where the model wanted a mapping, and it raised. Keying comes last, once every member is a mapping that can carry a name.

    Schema-declared defaults (``SLOT_DEFAULTS``, for slots whose class range ``ifabsent`` cannot serve) fill after the ``iri`` expansion, so a curated record's own value is never overridden by the default, and before construction, so the record carries the default the schema states.

    The semantic folds come after keying, so a fold reads each collection under the names the class will — a record spelling one as a list is not a different case to it. The terse ``distribution`` lift follows the domain fold, since a clamp folded out of ``boundaries`` can leave one behind for it to complete. The bare member of a list slot is wrapped last, once a mapping of members has been read as the list it abbreviates, so what is left as a mapping is the one member it spells.

    The document envelope goes first. ``tvbo_class`` states which class a *file* holds, which is a fact about the file rather than the object: the schema declares it (the ``Document`` mixin) so plain LinkML can validate the file, and every constructor route drops it so no object carries it.
    """
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS, _lift_one_distribution

    for envelope_key in ENVELOPE_KEYS:
        data.pop(envelope_key, None)

    refuse_retired(cls_name, data)
    paths_as_text(data)
    fold_aliases(cls_name, data)
    canonicalize_units(cls_name, data)
    lift_shortcuts(cls_name, data)
    expand_iri(cls_name, data)
    for slot, default in SLOT_DEFAULTS.get(cls_name, {}).items():
        if data.get(slot) is None:
            data[slot] = default

    key_members(cls_name, data)

    semantic = SEMANTIC_FOLDS.get(cls_name)
    if semantic is not None:
        semantic(data)
    _lift_one_distribution(data)
    wrap_lists(cls_name, data)
    return data
