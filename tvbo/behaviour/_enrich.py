"""Filling a record's gaps from the entity it names.

An entity written by ``iri`` — or by a name the curated database knows — is expanded at
construction by :func:`tvbo.datamodel.dialect.expand_iri`, which reads one local file and
nothing else. Everything beyond that is an act the caller asks for, and this is the verb
for it: ``enrich()``.

Which classes carry it is read off the schema. A class the schema gives an ``iri`` slot is
a class whose records may name an entity elsewhere, so ``hatch_build`` makes this a base
of it in both generated forms; subclasses inherit. Nothing lists them, and adding the slot
to a class is what makes its records enrichable.

Which sources answer is read off the class. ``database`` is implemented here and applies to
every enrichable record; ``ontology`` is a ``_from_ontology`` a behaviour mixin defines,
so a class reaches the ontology by knowing how to, rather than by being named in a table
here. They are ranked rather than combined: the first that resolves the entity answers
alone. The database is a local file whose content is curated and ordered, and the ontology
answers with an unordered set, so a curated record topped up from the ontology would gain
whatever the curators left out — and only in a process where the ontology happens to be
loaded, which is one record enriching to two different things.

The mixin is deliberately not named ``*Behaviour``: that name attaches a class by its stem
(``EventBehaviour`` -> ``Event``), and this one attaches by a schema rule instead.
"""

from __future__ import annotations

from jsonasobj2 import JsonObj

#: Consulted in order; each names a ``_from_<source>(key) -> bool`` reader.
SOURCES = ("database", "ontology")

#: The container types a slot may hold; anything else is a value, array included.
#: ``JsonObj`` is the LinkML dataclasses' own container and goes when they do.
_CONTAINERS = (dict, list, tuple, JsonObj)


def _unset(value) -> bool:
    """Whether *value* is a gap: nothing, or a container holding nothing."""
    return value is None or (isinstance(value, _CONTAINERS) and not value)


def _schema_class(cls) -> str:
    """The schema class *cls* is, which is not always what Python calls it.

    Every table read here is keyed by the schema — the curated database's categories, the
    keyed collections — and a runtime subclass is still the schema class it extends:
    ``DynamicalSystem`` is a ``Dynamics``. The dataclasses state it outright; for the
    Pydantic models the generated class is the one that names it.
    """
    declared = getattr(cls, "class_name", None)
    if declared:
        return str(declared)
    for base in cls.__mro__:
        if base.__module__.startswith("tvbo.datamodel."):
            return base.__name__
    return cls.__name__


class IriEnrichable:
    """Gap-filling from the entity this record names."""

    def enrich(self, source: str | None = None, key: str | None = None):
        """Fill this record's gaps from the entity it names, and return it.

        Gap-filling only: a slot this record already carries is never overwritten, and a
        keyed collection gains the members it lacks while keeping the ones it has. That is
        the whole contract, and it is why this can run after construction — unlike the
        ``iri`` expansion in the dialect, which has to distinguish an authored value from a
        schema default and so must run before one is applied.

        Args:
            source: Consult only this source. By default the sources are tried in
                :data:`SOURCES` order and the FIRST that resolves the entity answers
                alone. Topping a curated record up from the ontology would add whatever
                the curators left out and make the result depend on whether the ontology
                happens to be loaded — the same record enriching to a different thing in
                two processes.
            key: The entity to enrich *from*, when this record does not name it itself —
                a ``network.coupling`` entry named for its role in the network, say,
                declaring the coupling function it is an instance of. Defaults to what
                this record names: its ``iri``'s local name, else its ``name``.

        Raises:
            ValueError: *source* is not a source this class has.
            LookupError: the entity was named by a pointer — an explicit *key*, or this
                record's ``iri`` — and no source resolves it. Falling back to a bare
                ``name`` that resolves nowhere is not an error: a self-contained record
                that happens to be called something is the normal case.
        """
        wanted = SOURCES if source is None else (source,)
        pointer = key or getattr(self, "iri", None)
        key = key or self._named_entity()
        cls_name = _schema_class(type(self))
        found = False
        for name in wanted:
            reader = getattr(self, f"_from_{name}", None)
            if reader is None:
                if source is not None:
                    raise ValueError(f"{cls_name} has no {name!r} to enrich from.")
                continue
            if key and reader(key):
                found = True
                break
        if not found and pointer:
            raise LookupError(f"{cls_name} names {pointer!r}, which no source resolves.")
        return self

    def _named_entity(self) -> str | None:
        """The entity this record names: its ``iri``'s local name, else its ``name``."""
        from tvbo.data.registry import local_name

        iri = getattr(self, "iri", None)
        return local_name(iri) if iri else getattr(self, "name", None)

    def _from_database(self, key: str) -> bool:
        """Gap-fill from the curated record *key* names. True when one was found.

        The entry is raw YAML, so it is handed to this class's own constructor rather than
        read slot by slot: that types every member, in this object's own generated form,
        and applies the dialect the entry itself is written in.
        """
        from tvbo.datamodel.dialect import curated_entry

        entry = curated_entry(_schema_class(type(self)), key)
        if entry is None:
            return False
        self._gap_fill(type(self)(**entry))
        return True

    def _gap_fill(self, other) -> None:
        """Take from *other* every slot this record leaves unset.

        A keyed collection is never assigned, only mutated member by member — so a record
        declaring one parameter of ten gains the other nine and keeps its own, and, just as
        importantly, the container the class built is the one that stays. Assigning a plain
        Python container into a LinkML slot is what produces a ``JsonObj``: the setter wraps
        whatever it is handed, and the typed members inside it stop reading as themselves.
        Which slots are keyed collections is read off the schema rather than sniffed from
        the value, so this holds for both generated forms and for an empty collection too.
        """
        from tvbo.datamodel.dialect import KEYED_COLLECTIONS
        from tvbo.utils import keyed_items

        collections = KEYED_COLLECTIONS.get(_schema_class(type(self)), {})
        for slot in _slots(type(self)):
            theirs = getattr(other, slot, None)
            if _unset(theirs):
                continue
            mine = getattr(self, slot, None)
            if slot not in collections:
                if _unset(mine):
                    setattr(self, slot, theirs)
                continue
            if mine is None:
                setattr(self, slot, {})
                mine = getattr(self, slot)
            for member, value in keyed_items(theirs, slot):
                if member not in mine:
                    mine[member] = value


def _slots(cls) -> tuple[str, ...]:
    """*cls*'s schema slots, whichever generated form it is."""
    fields = getattr(cls, "__dataclass_fields__", None)
    return tuple(fields) if fields is not None else tuple(cls.model_fields)
