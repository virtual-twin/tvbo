"""The one JAX pytree registration in tvbo: a class declares which attributes are leaves and what is static, and this module flattens and rebuilds it.

A leaf is anything JAX may trace — an array, a tracer, a nested pytree such as a `Network` — and travels as a child keyed by attribute name, so the order a class lists them in never decides a treedef. Everything static travels as one canonical JSON string, which is hashable and compares by value, so two objects that differ only in their arrays share a compiled trace and two that differ in metadata do not. `register` is the only entry point; `Pytree` supplies the declaration by field name, and a class whose leaves are not fixed attributes — `Network`, whose leaves are whatever it has materialised — implements the three hooks itself.
"""

from __future__ import annotations

import functools
import json

import numpy as np


def jsonable(value):
    """*value* as JSON can carry it: arrays to lists, numpy scalars to Python ones, enums and generated records to their dict form, anything else to its string."""
    from jax import Array as JaxArray
    from jsonasobj2 import as_dict
    from linkml_runtime.utils.enumerations import EnumDefinitionImpl
    from linkml_runtime.utils.yamlutils import YAMLRoot

    if isinstance(value, JaxArray):
        value = np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, EnumDefinitionImpl):
        return str(value)
    if isinstance(value, YAMLRoot):
        return as_dict(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)


def static_spec(record, held_out=()) -> str:
    """A generated record's fields as canonical JSON, without *held_out* and without ``None``s: the static half of its pytree, which its constructor takes back. An `Equation`'s ``_code`` collapses to its text for the same reason."""
    from jsonasobj2 import as_dict

    def strip(obj):
        if isinstance(obj, dict):
            if "_code" in obj and isinstance(obj["_code"], dict) and "text" in obj["_code"]:
                return obj["_code"]["text"]
            return {k: strip(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [strip(x) for x in obj]
        return obj

    meta = as_dict(record)
    if not isinstance(meta, dict):
        meta = dict(meta) if hasattr(meta, "__iter__") else {}
    held = set(held_out)
    return json.dumps(
        strip({k: v for k, v in meta.items() if not str(k).startswith("_") and k not in held}),
        sort_keys=True,
        default=jsonable,
    )


class Pytree:
    """Declares a class's pytree, and registers every subclass as one when the subclass is created.

    `LEAVES` are the children, `STATIC` the metadata, and the class is rebuilt by handing both back to its constructor as keywords. A class whose leaves are not fixed attributes — a mapping, or a `Network` whose leaves are whatever it has materialised — inherits this and overrides the three hooks instead; it still registers itself by being a subclass. Registration is on the exact type, so a subclass of a pytree registers again, which is what JAX requires of it.

    Subclassing is the whole registration mechanism, so a generated class becomes a pytree by taking a behaviour mixin that inherits this, at the moment the class is created, rather than by a decorator some import has to reach first.
    """

    LEAVES: tuple[str, ...] = ()
    STATIC: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    def _pytree_leaves(self) -> dict:
        """The children, keyed by attribute name."""
        return {name: getattr(self, name, None) for name in self.LEAVES}

    def _pytree_static(self) -> str:
        """The metadata as one canonical JSON string."""
        return json.dumps({name: getattr(self, name, None) for name in self.STATIC}, sort_keys=True, default=jsonable)

    @classmethod
    def _pytree_build(cls, static: str, leaves: dict):
        """The object again, from what `_pytree_static` and `_pytree_leaves` gave."""
        return cls(**leaves, **json.loads(static))


def _flatten(obj):
    return (obj._pytree_leaves(),), (obj._pytree_static(),)


def _unflatten(cls, aux, children):
    (static,), (leaves,) = aux, children
    return cls._pytree_build(static, leaves)


def register(cls):
    """Register *cls* with JAX through the one flatten/unflatten pair.

    JAX registers exact types, so a subclass registers itself again — which `Pytree.__init_subclass__` does for it. Registering the same class twice raises, so this is called once per class creation and nowhere else.
    """
    from jax.tree_util import register_pytree_node

    register_pytree_node(cls, _flatten, functools.partial(_unflatten, cls))
    return cls
