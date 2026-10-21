"""A text slot holds text: a structured value written where the schema declares a string is refused, not stringified.

LinkML's generated dataclasses coerce a non-``str`` value in a ``string`` slot, or in an identifier slot typed by an ``extended_str`` subclass, with ``str()``, which stores a mapping as the text ``"JsonObj(value=...)"`` and lets it surface later as an axis label or a name nobody wrote. ``hatch_build.py`` routes every such coercion through :func:`text_value`, which passes text and scalars on to ``str()`` and raises for anything with structure. The Pydantic models refuse these values too, so both construction paths agree.
"""

from __future__ import annotations

import reprlib
from collections.abc import Iterator, Mapping, Sequence, Set

from jsonasobj2 import JsonObj
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from linkml_runtime.utils.yamlutils import YAMLRoot


def text_value(owner, slot: str, value):
    """*value* unchanged when ``str()`` renders it as the text it stands for; a ``TypeError`` naming ``Class.slot`` when it carries structure.

    Structure is a mapping, a sequence other than text, a set, an iterator, an array with at least one dimension, or a record. An enum member is a scalar despite descending from both ``YAMLRoot`` and ``JsonObj``: its ``str()`` is its permissible value's text. So are numbers, booleans, paths and symbolic expressions.
    """
    if isinstance(value, EnumDefinitionImpl):
        return value
    sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if sequence or isinstance(value, (Mapping, Set, Iterator, JsonObj, YAMLRoot)) or getattr(value, "ndim", 0) > 0:
        raise TypeError(
            f"{type(owner).__name__}.{slot} takes text, but was given the {type(value).__name__} {reprlib.repr(value)}. "
            "Write the text itself: storing its str() would put that repr where the text belongs."
        )
    return value
