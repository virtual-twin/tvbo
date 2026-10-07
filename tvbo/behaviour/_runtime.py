#
# Module: behaviour/_runtime.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""The bases a behaviour builds on without attaching to anything by name: the runtime bookkeeping it keeps out of the record it writes, the YAML constructors every document class shares, the curated-database lookups of the classes the registry files entries of, and copy hooks that hold on either generated form."""

from __future__ import annotations

import os
from typing import ClassVar


class RuntimeAttributes:
    """Hides every leading-underscore instance attribute from the LinkML dumpers.

    A behaviour may plant bookkeeping on the record it serves — a loaded flag, a resolved companion path, a cache of read arrays — through ``object.__setattr__``. Schema slot names are never underscore-prefixed, so hiding by rule, the way ``Network._items`` hides its arrays, keeps that state out of the YAML the record publishes and off the constructor that reads it back; the Pydantic form needs nothing, since `record_dict` reads schema fields alone. Not a behaviour itself but a base for the behaviours that keep such state, so it attaches to nothing by name.
    """

    def _items(self):
        """The record's slots as the dumpers see them, every runtime attribute left out.

        The dataclasses answer through ``JsonObj._items``; the Pydantic models have no such method, so their own attribute mapping stands in rather than the ``AttributeError`` a bare ``super()`` call would raise on them.
        """
        inherited = getattr(super(), "_items", None)
        for key, value in inherited() if inherited is not None else vars(self).items():
            if not str(key).startswith("_"):
                yield key, value


class YamlDocument:
    """Construction from YAML — a file or a string — and serialisation back to it.

    Not a behaviour itself but a base for the behaviours of every class a YAML document describes, so, like `RuntimeAttributes`, it attaches to nothing by name. A class the curated database also files entries of takes `Catalogued` instead, which adds the database lookups; one it files nothing of stays here, so it offers no lookup that could only ever fail.
    """

    @classmethod
    def from_file(cls, path: str | os.PathLike):
        """Load one from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), target_class=cls)

    @classmethod
    def from_string(cls, yaml_string: str):
        """Load one from a YAML string."""
        from tvbo.utils import yaml_loader

        return yaml_loader.loads(yaml_string, target_class=cls)

    def to_yaml(self, filepath: str | None = None):
        """Serialise this record to YAML.

        Args:
            filepath: Where to write the YAML; when omitted it is only returned.

        Returns:
            The written path when *filepath* is given, otherwise the YAML string.
        """
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)


class Catalogued(YamlDocument):
    """A `YamlDocument` the curated database files entries of: loading one by name and listing them.

    Only for a class the registry has a category for. The database category is the schema class the record is, read off the class by `_schema_class` rather than typed into each behaviour, so both generated forms and any runtime subclass look in the one directory the registry files that class under. A class whose entries are filed under another class's category names that category in `CATEGORY`.
    """

    CATEGORY: ClassVar[str | None] = None
    """The registry category this class's entries are filed under, when it is not the schema class itself."""

    @classmethod
    def _database_category(cls) -> str:
        """The registry category this class is looked up in: `CATEGORY`, else its schema class."""
        from tvbo.behaviour._enrich import _schema_class

        return cls.CATEGORY or _schema_class(cls)

    @classmethod
    def from_db(cls, name: str):
        """Load the curated database entry *name*."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve(cls._database_category(), name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """The names of this class's curated database entries."""
        from tvbo.data.registry import list_entries

        return list_entries(cls._database_category())


class Copyable:
    """Shallow and deep copies of a record, through its generated form's own copy protocol where it has one.

    Pydantic implements the copy hooks itself and ``model_copy()`` routes through them, so building a clone with ``cls.__new__`` leaves ``__pydantic_extra__`` unset and the first assignment raises; the LinkML dataclasses have no hook, hence the manual path. Not a behaviour itself but a base for the records a caller copies and then edits, so it attaches to nothing by name.
    """

    def copy(self, **overrides):
        """A deep copy of this record, with each of *overrides* set on the copy.

        Errors are not swallowed; if a field can't be copied, an exception is raised.
        """
        import copy as _copy

        new_obj = _copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def __copy__(self):
        """A shallow copy: the generated form's own hook, else a clone sharing this record's attribute values."""
        inherited = getattr(super(), "__copy__", None)
        if inherited is not None:
            return inherited()
        cls = self.__class__
        clone = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(clone, k, v)
        return clone

    def __deepcopy__(self, memo):
        """A deep copy: the generated form's own hook, else a clone built through the constructor from a deep copy of every declared field.

        Every dataclass field is copied, not only those in ``__dict__``, since a field still holding its default may be absent from it, and the constructor applies every default the copy does not state.
        """
        import copy as _copy
        import dataclasses

        inherited = getattr(super(), "__deepcopy__", None)
        if inherited is not None:
            return inherited(memo)
        if dataclasses.is_dataclass(self):
            data = {field.name: _copy.deepcopy(getattr(self, field.name, None), memo) for field in dataclasses.fields(self)}
        else:
            data = {k: _copy.deepcopy(v, memo) for k, v in self.__dict__.items()}
        clone = self.__class__(**data)
        memo[id(self)] = clone
        return clone
