#
# Module: behaviour/_runtime.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2
#
"""The bases a behaviour builds on without attaching to anything by name: the generated datamodel reached lazily, the YAML constructors every document class shares, the curated-database lookups of the classes the registry files entries of, and the edited copy of a record."""

from __future__ import annotations

import os
from typing import ClassVar


def _dm():
    """The generated datamodel, imported when first needed: the generated module imports the behaviour modules, so none of them can import it at module scope."""
    from tvbo import datamodel

    return datamodel


class YamlDocument:
    """Construction from YAML — a file or a string — and serialisation back to it.

    Not a behaviour itself but a base for the behaviours of every class a YAML document describes, so it attaches to nothing by name. A class the curated database also files entries of takes `Catalogued` instead, which adds the database lookups; one it files nothing of stays here, so it offers no lookup that could only ever fail.
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

    @classmethod
    def from_datamodel(cls, record):
        """A record of this class restating the slots *record* states, each as its own copy.

        The generated class is the class, so this is a copy through the constructor; a default *record* never stated is not restated, and a member is copied rather than shared, so the two records are edited apart.
        """
        import copy

        return cls(**{slot: copy.deepcopy(getattr(record, slot)) for slot in record.model_fields_set})

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

    Only for a class the registry has a category for. The database category is the schema class the record is, read off the class by `_schema_class` rather than typed into each behaviour, so a runtime subclass looks in the one directory the registry files that class under. A class whose entries are filed under another class's category names that category in `CATEGORY`.
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
    """A deep copy of a record with some slots changed. Not a behaviour itself but a base for the records a caller copies and then edits, so it attaches to nothing by name."""

    def copy(self, **overrides):
        """A deep copy of this record, with each of *overrides* set on the copy.

        Errors are not swallowed; if a field can't be copied, an exception is raised.
        """
        import copy as _copy

        new_obj = _copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj
