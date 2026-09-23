#
# Module: behaviour/_runtime.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Runtime bookkeeping a behaviour keeps on a record, kept out of the record it writes."""

from __future__ import annotations


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
