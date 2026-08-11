"""TVB-O Data Model.

Auto-generated from LinkML schema.

Usage:
    from tvbo.datamodel.schema import Dynamics, Parameter, Equation
    from tvbo.datamodel.pydantic import Dynamics as PydanticDynamics
"""

from tvbo.datamodel.schema import Network  # noqa: E402

# Defined here rather than in the generated file, so it survives `make gen-linkml`.
Network.number_of_regions = property(
    lambda self: self.number_of_nodes,
    lambda self, v: setattr(self, "number_of_nodes", v),
)

# Patched to canonicalise any input: the generated coercions reject aliases and leak PermissibleValue.
from tvbo.datamodel.schema import UnitEnum as _UnitEnum  # noqa: E402
from tvbo.utils.units import normalize_unit as _normalize_unit  # noqa: E402

from .schema import *  # noqa: E402, F401, F403

# Register slash-notation aliases (mm/ms → mm_per_ms) so both work in YAML
for _alias, _canon in {
    "mm/ms": "mm_per_ms",
    "m/s": "m_per_s",
    "mV/ms": "mV_per_ms",
    "mV/s": "mV_per_s",
    "Hz/nA": "Hz_per_nA",
    "S/m": "S_per_m",
    "H/m": "H_per_m",
    "rad/ms": "rad_per_ms",
}.items():
    setattr(_UnitEnum, _alias, getattr(_UnitEnum, _canon))
del _alias, _canon

_UnitEnum_orig_init = _UnitEnum.__init__


def _unit_text(obj):
    """Extract the plain text key from any unit-like object."""
    if isinstance(obj, str):
        return obj
    # UnitEnum: str() returns just the key
    if isinstance(obj, _UnitEnum):
        return str(obj)
    # PermissibleValue or any object with .text
    text = getattr(obj, "text", None)
    if text is not None:
        return text
    # JsonObj from as_dict() round-trip: obj._code.text
    inner = getattr(obj, "_code", None)
    if inner is not None:
        text = getattr(inner, "text", None)
        if text is not None:
            return text
    # dict from as_dict(): {'_code': {'text': 'mm', ...}}
    if isinstance(obj, dict):
        inner = obj.get("_code", obj)
        if isinstance(inner, dict):
            return inner.get("text", str(obj))
    return str(obj)


def _unit_enum_init(self, code):
    code = _unit_text(code)
    code = _normalize_unit(code) or code
    _UnitEnum_orig_init(self, code)


_UnitEnum.__init__ = _unit_enum_init

# So `getattr(UnitEnum, "mm")` yields a UnitEnum, not a PermissibleValue that as_dict() blows up.
_UnitEnumMeta = type(_UnitEnum)
_meta_orig_getattribute = _UnitEnumMeta.__getattribute__

from linkml_runtime.linkml_model.meta import PermissibleValue as _PermissibleValue


def _unit_meta_getattribute(cls, item):
    try:
        result = _meta_orig_getattribute(cls, item)
    except AttributeError:
        if item.startswith("__") and item.endswith("__"):
            raise
        # Dotted names like "a.u." can't be Python attributes; construct instead
        canonical = _normalize_unit(item) or item
        return cls(canonical)
    # Wrap bare PermissibleValue in a proper UnitEnum instance
    if isinstance(result, _PermissibleValue) and not isinstance(result, _UnitEnum):
        canonical = _normalize_unit(item) or item
        return cls(canonical)
    return result


_UnitEnumMeta.__getattribute__ = _unit_meta_getattribute

# Backward-compat aliases for old module names
import sys

from tvbo.datamodel import pydantic as tvbopydantic  # noqa: E402, F401
from tvbo.datamodel import schema as tvbo_datamodel  # noqa: E402, F401

sys.modules["tvbo.datamodel.tvbo_datamodel"] = tvbo_datamodel
sys.modules["tvbo.datamodel.tvbopydantic"] = tvbopydantic

# Attach user-facing helper methods (.plot(), alias-aware __init__) to the generated ``schema.Event`` class. Done here — where the schema is first imported — rather than at ``import tvbo`` so bare imports and the CLI do not pay the datamodel import cost. Runs last, after schema/pydantic are bound.
from tvbo.classes import event as _event_helpers  # noqa: E402, F401
