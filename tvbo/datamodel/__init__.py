"""TVB-O Data Model.

Generated from the LinkML schema. `tvbo.datamodel.pydantic` is the runtime form and this package re-exports it; `tvbo.datamodel.schema`, the LinkML dataclass form, is generated beside it until gen-python is retired.

Usage:
    from tvbo.datamodel import Dynamics, Parameter, Equation
"""

import sys

from tvbo.datamodel import pydantic as _pydantic
from tvbo.datamodel.pydantic import Network, UnitEnum
from tvbo.utils.units import normalize_unit as _normalize_unit

from .pydantic import *  # noqa: F401, F403

# Defined here rather than in the generated file, so it survives `make gen-linkml`.
Network.number_of_regions = property(
    lambda self: self.number_of_nodes,
    lambda self, v: setattr(self, "number_of_nodes", v),
)


def _unit_from_alias(cls, value):
    """`UnitEnum("mV/ms")` is `UnitEnum.mV_per_ms`: a curated unit under any spelling `normalize_unit` knows; an uncurated one is no member."""
    canonical = _normalize_unit(value) if isinstance(value, str) else None
    return cls(canonical) if canonical and canonical != value else None


UnitEnum._missing_ = classmethod(_unit_from_alias)
UnitEnum.__str__ = lambda self: self.value

tvbo_datamodel = tvbopydantic = _pydantic
"""The runtime form under the names earlier releases imported it by."""
sys.modules["tvbo.datamodel.tvbo_datamodel"] = _pydantic
sys.modules["tvbo.datamodel.tvbopydantic"] = _pydantic
