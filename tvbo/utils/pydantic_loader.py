"""Pydantic loader / validator for TVBO YAML — the trustworthy validation path.

TVBO authors write experiments in a human-friendly YAML dialect where collections are *keyed dicts* and the key doubles as the member's identifier::

    parameters:
      a: {value: 0.27}        # name == "a"
      b: {value: 0.108}       # name == "b"

The LinkML runtime loader injects that key into each member's identifier slot (``name``) automatically. The generated Pydantic datamodel
(:mod:`tvbo.datamodel.pydantic`) does **not** — it expects ``name`` to be present inside every member — so validating raw TVBO YAML against the Pydantic
models fails with a flood of ``name Field required`` errors that are purely an artefact of the keyed-dict convention, not real schema violations.

This module performs exactly that key→identifier normalization and then validates with the strict (``extra="forbid"``) Pydantic models, giving callers
a single, trustworthy "is this a valid TVBO object?" entry point. The YAML preprocessing for ``<<:`` merge keys and ``!include`` directives is delegated to
:mod:`tvbo.utils.yaml_loader` so there is one implementation of those idioms.

Typical use::

    from tvbo.utils import pydantic_loader

    exp = pydantic_loader.load("experiment.yaml")          # -> SimulationExperiment
    exp = pydantic_loader.loads(yaml_text)                 # from a string
    exp = pydantic_loader.validate(some_dict)              # from an already-parsed dict
    yaml_text = pydantic_loader.dump(exp)                  # canonical YAML round-trip

All three loaders raise :class:`pydantic.ValidationError` on genuinely invalid input, so they are real validators rather than lenient coercers.
"""

from __future__ import annotations

import copy
import re
import warnings
from functools import lru_cache
from typing import Any, Type, Union, get_args, get_origin

import yaml
from pydantic import BaseModel

from tvbo.datamodel import pydantic as _dm
from tvbo.utils import yaml_loader

__all__ = ["load", "loads", "validate", "normalize", "dump", "DEFAULT_TARGET"]

#: Default datamodel class used when a caller does not specify ``target_class``.
DEFAULT_TARGET = "SimulationExperiment"

#: File-envelope metadata keys that annotate a serialized object's class and
#: schema version but are not datamodel slots. TVBO strips these at construction
#: (see ``tvbo.classes.phenotype``), so we drop them before validation too.
_ENVELOPE_KEYS = yaml_loader.ENVELOPE_KEYS


# --------------------------------------------------------------------------- #
# Target-class resolution
# --------------------------------------------------------------------------- #
def _resolve_target(target_class: Union[str, Type[BaseModel], None]) -> Type[BaseModel]:
    """Resolve ``target_class`` (a class, a class name, or ``None``) to a model.

    Strings are looked up in :mod:`tvbo.datamodel.pydantic` so the platform can pass ``"SimulationExperiment"`` / ``"Dynamics"`` / ... straight through.
    """
    if target_class is None:
        target_class = DEFAULT_TARGET
    if isinstance(target_class, str):
        cls = getattr(_dm, target_class, None)
        if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise ValueError(f"Unknown TVBO datamodel class: {target_class!r}")
        return cls
    if not (isinstance(target_class, type) and issubclass(target_class, BaseModel)):
        raise TypeError(f"target_class must be a Pydantic model or its name, got {target_class!r}")
    return target_class


# --------------------------------------------------------------------------- #
# Annotation helpers
# --------------------------------------------------------------------------- #
def _candidates(annotation: Any):
    """Yield concrete type candidates, unwrapping ``Optional``/``Union`` nesting."""
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if arg is not type(None):
                yield from _candidates(arg)
    else:
        yield annotation


def _first_model(annotation: Any) -> Type[BaseModel] | None:
    """Return the first Pydantic-model candidate within ``annotation``, if any."""
    for cand in _candidates(annotation):
        if isinstance(cand, type) and issubclass(cand, BaseModel):
            return cand
    return None


@lru_cache(maxsize=None)
def _identifier_field(model_cls: Type[BaseModel]) -> str | None:
    """Name of the slot the inlined-dict key maps onto.

    A non-``name``/``id`` key slot marks itself with a ``collection_key`` annotation, because LinkML consumes ``identifier``/``key`` into the field's
    required-ness and drops them from the emitted metadata (so they can't be read back here). Otherwise falls back to the universal TVBO conventions ``name``
    then ``id``. Returns ``None`` when no identifier slot can be determined (the member is then left untouched).
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


@lru_cache(maxsize=None)
def _slot_alias_map(model_cls: Type[BaseModel]) -> dict[str, str]:
    """This class's ``{alias: canonical}`` slot-alias map.

    Reused verbatim from the LinkML dataclass path (``schema._SLOT_ALIASES``) so this validator accepts exactly the keys that loader does. The dataclasses fold
    these aliases in ``__init__``; the Pydantic models cannot, so :func:`_inject` folds them here, class-scoped — the same context that keeps ``target_variable``
    (an ``Edge`` alias, but the canonical slot on a stimulus ``Event``) from being renamed where it must not be.
    """
    from tvbo.datamodel.schema import _SLOT_ALIASES

    return _SLOT_ALIASES.get(model_cls.__name__, {})


# --------------------------------------------------------------------------- #
# Key -> identifier injection
# --------------------------------------------------------------------------- #
def _inject(model_cls: Type[BaseModel], data: Any) -> Any:
    """Recursively inject keyed-dict keys into each member's identifier slot."""
    if not isinstance(data, dict) or not hasattr(model_cls, "model_fields"):
        return data

    # Drop file-envelope metadata wherever a model is constructed (mirrors TVBO).
    for envelope_key in _ENVELOPE_KEYS:
        data.pop(envelope_key, None)

    # Fold this class's slot aliases (``dt``->``step_size``, ``righthandside``->``rhs``,
    # ...), class-scoped, exactly as the generated dataclasses fold them in ``__init__``.
    # The Pydantic models cannot, so a raw alias key would otherwise be rejected by
    # ``extra='forbid'``. Runs before the field walk so an aliased key is seen under its canonical name.
    for alias, canonical in _slot_alias_map(model_cls).items():
        if alias not in data:
            continue
        if canonical in data:
            warnings.warn(
                f"{model_cls.__name__} got both {alias!r} and its canonical slot {canonical!r}; ignoring {alias!r}.",
                stacklevel=2,
            )
            data.pop(alias)
        else:
            data[canonical] = data.pop(alias)

    for fname, info in model_cls.model_fields.items():
        key = fname if fname in data else (info.alias if info.alias and info.alias in data else None)
        if key is None:
            continue
        value = data[key]

        for cand in _candidates(info.annotation):
            origin = get_origin(cand)

            # dict[str, SomeModel] — keyed collection. Accept either the keyed form (inject the key as the member's identifier) or a list form (Odoo many2many and JS arrays use lists) which we coerce into a keyed dict using each member's identifier value.
            if origin is dict and isinstance(value, (dict, list)):
                args = get_args(cand)
                member_cls = _first_model(args[1]) if len(args) == 2 else None
                if member_cls is not None:
                    id_slot = _identifier_field(member_cls)
                    if isinstance(value, dict):
                        for member_key, member in value.items():
                            if isinstance(member, dict):
                                if id_slot and id_slot not in member:
                                    member[id_slot] = member_key
                                _inject(member_cls, member)
                    else:
                        coerced = {}
                        for member in value:
                            if not isinstance(member, dict):
                                continue
                            member_key = member.get(id_slot) if id_slot else None
                            if member_key in (None, ""):
                                member_key = f"item_{len(coerced)}"
                            _inject(member_cls, member)
                            coerced[str(member_key)] = member
                        data[key] = coerced
                    break

            # list[...] — recurse into model members, or split a newline/ bulleted Text blob (how Odoo stores list[scalar]) into a list.
            elif origin in (list, tuple, set):
                args = get_args(cand)
                member_cls = _first_model(args[0]) if args else None
                if isinstance(value, list):
                    if member_cls is not None:
                        for member in value:
                            _inject(member_cls, member)
                    else:
                        # Scalar list (e.g. list[str]); Odoo many2many can yield full objects -> collapse each to its identifier.
                        data[key] = [(m.get("name") or m.get("id")) if isinstance(m, dict) else m for m in value]
                    break
                if isinstance(value, str) and member_cls is None:
                    parts = [re.sub(r"^[-*]\s*", "", p.strip()) for p in re.split(r"[\r\n]+", value)]
                    data[key] = [p for p in parts if p]
                    break

            # Nested single model.
            elif isinstance(cand, type) and issubclass(cand, BaseModel) and isinstance(value, dict):
                _inject(cand, value)
                break

            # Scalar string slot that received an object (Odoo many2one for a by-name reference, e.g. FunctionCall.function or Continuation.dynamics)
            # -> collapse to the referenced identifier.
            elif cand is str and isinstance(value, dict):
                data[key] = value.get("name") or value.get("id")
                break

    return data


def normalize(data: dict, target_class: Union[str, Type[BaseModel], None] = None) -> dict:
    """Return a copy of ``data`` with keyed-dict keys injected as identifiers.

    Pure data transformation — performs no validation. Useful when a caller wants the normalized dict (e.g. to merge with other state) without building
    a model instance.
    """
    target = _resolve_target(target_class)
    out = copy.deepcopy(data)
    _inject(target, out)
    return out


# --------------------------------------------------------------------------- #
# Public validation / load API
# --------------------------------------------------------------------------- #
def _strip_unknown(model_cls: Type[BaseModel], data: Any) -> None:
    """Recursively drop keys not declared by ``model_cls`` (by field name or alias). Assumes ``data`` is already normalized (collections as keyed dicts)."""
    if not isinstance(data, dict) or not hasattr(model_cls, "model_fields"):
        return
    known = {}
    for fname, info in model_cls.model_fields.items():
        known[fname] = info
        if info.alias:
            known[info.alias] = info
    for key in list(data.keys()):
        info = known.get(key)
        if info is None:
            del data[key]
            continue
        value = data[key]
        for cand in _candidates(info.annotation):
            origin = get_origin(cand)
            if origin is dict and isinstance(value, dict):
                args = get_args(cand)
                member_cls = _first_model(args[1]) if len(args) == 2 else None
                if member_cls is not None:
                    for item in value.values():
                        _strip_unknown(member_cls, item)
                break
            if origin in (list, tuple, set) and isinstance(value, list):
                args = get_args(cand)
                member_cls = _first_model(args[0]) if args else None
                if member_cls is not None:
                    for item in value:
                        _strip_unknown(member_cls, item)
                break
            if isinstance(cand, type) and issubclass(cand, BaseModel) and isinstance(value, dict):
                _strip_unknown(cand, value)
                break


def validate(data: dict, target_class: Union[str, Type[BaseModel], None] = None, *, drop_unknown: bool = False) -> BaseModel:
    """Validate an already-parsed ``dict`` and return a model instance.

    Raises :class:`pydantic.ValidationError` if ``data`` does not conform. With
    ``drop_unknown=True``, keys not declared by the schema are discarded before validation instead of being rejected — used by the TVBO platform's database
    export to ignore Odoo-only fields (e.g. portal ``visibility``/``owner``) while hand-authored input still rejects unknown keys.
    """
    target = _resolve_target(target_class)
    data = normalize(data, target)
    if drop_unknown:
        _strip_unknown(target, data)
    return target.model_validate(data)


def loads(
    source: str, target_class: Union[str, Type[BaseModel], None] = None, *, drop_unknown: bool = False, **kwargs: Any
) -> BaseModel:
    """Parse a YAML string (with ``<<:`` / ``!include`` support) and validate it.

    ``drop_unknown`` is forwarded to :func:`validate` (see its docstring); the remaining ``kwargs`` go to the YAML loader.
    """
    target = _resolve_target(target_class)
    data = yaml_loader.load_as_dict(source, **kwargs) or {}
    return validate(data, target, drop_unknown=drop_unknown)


def load(
    source: Any, target_class: Union[str, Type[BaseModel], None] = None, *, drop_unknown: bool = False, **kwargs: Any
) -> BaseModel:
    """Load YAML from a path / stream / string and validate it.

    ``!include`` paths are resolved relative to ``source``'s directory when it is a path, matching :func:`tvbo.utils.yaml_loader.load`. ``drop_unknown`` is
    forwarded to :func:`validate`; the remaining ``kwargs`` go to the YAML loader.
    """
    target = _resolve_target(target_class)
    data = yaml_loader.load_as_dict(source, **kwargs) or {}
    return validate(data, target, drop_unknown=drop_unknown)


def dump(obj: Union[BaseModel, dict], *, exclude_none: bool = True, sort_keys: bool = False) -> str:
    """Serialise a validated model (or plain dict) to canonical TVBO YAML."""
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json", by_alias=True, exclude_none=exclude_none)
    else:
        data = obj
    return yaml.safe_dump(data, sort_keys=sort_keys, allow_unicode=True)
