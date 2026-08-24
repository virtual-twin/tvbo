"""The TVBO YAML dialect — one implementation, both construction paths.

TVBO authors write a dialect the schema does not describe on its own: a slot may be
written under a declared alias (``dt`` for ``step_size``), and an object with one
obvious field may be written as a bare scalar (``omega: 0.0628`` for
``omega: {value: 0.0628}``). Neither is something LinkML's loaders apply — ``aliases:``
is documentation to them, and ``simple_dict_value`` is specified but unimplemented.

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

from tvbo.datamodel.dialect_tables import SCALAR_SHORTCUTS, SLOT_ALIASES

__all__ = [
    "SCALAR_SHORTCUTS",
    "SLOT_ALIASES",
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


def normalize(cls_name: str, data: dict) -> dict:
    """Fold *cls_name*'s scalar shortcuts and slot aliases into *data*, in place.

    Class-scoped on purpose: an alias is only an alias where its class declares it.
    ``target_variable`` is an ``Edge`` alias for ``target_var`` but the canonical slot on
    a stimulus ``Event``, so a table keyed by slot name alone would rename it in the one
    place it must not be.

    Aliases fold first: the shortcut table is keyed by canonical slot, so a value written
    under an alias is not yet under a name the shortcut pass can see. Lifting first left
    ``BoundaryCondition(value="0")`` — the older spelling of ``equation`` — a bare string
    where the generated ``__post_init__`` wanted a mapping, and it raised.
    """
    for alias, canonical in SLOT_ALIASES.get(cls_name, {}).items():
        if alias not in data:
            continue
        value = data.pop(alias)
        if canonical in data:
            warnings.warn(
                f"{cls_name} got both {alias!r} and its canonical slot {canonical!r}; ignoring {alias!r}.",
                stacklevel=3,
            )
        else:
            data[canonical] = value

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
