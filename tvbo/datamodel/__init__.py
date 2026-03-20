"""
TVB-O Data Model
================
Auto-generated from LinkML schema.

Usage:
    from tvbo.datamodel.schema import Dynamics, Parameter, Equation
    from tvbo.datamodel.pydantic import Dynamics as PydanticDynamics
"""

from tvbo.datamodel.schema import Network  # noqa: E402

# number_of_regions is a deprecated alias for number_of_nodes.
# Defined here (not in the generated file) so it survives make gen-linkml.
Network.number_of_regions = property(
    lambda self: self.number_of_nodes,
    lambda self, v: setattr(self, 'number_of_nodes', v),
)

from .schema import *  # noqa: E402, F401, F403

# ── UnitEnum: register slash-notation aliases ────────────────────────
# Scientists naturally write "mm/ms", "S/m" etc. The canonical enum values
# use underscores (mm_per_ms, S_per_m) because they must be valid Python
# identifiers. This block registers the slash forms so both are accepted
# in YAML files and Python code.
from tvbo.datamodel.schema import UnitEnum as _UnitEnum  # noqa: E402

_UNIT_ALIASES = {
    "mm/ms": "mm_per_ms",
    "m/s": "m_per_s",
    "mV/ms": "mV_per_ms",
    "mV/s": "mV_per_s",
    "Hz/nA": "Hz_per_nA",
    "S/m": "S_per_m",
    "H/m": "H_per_m",
    "rad/ms": "rad_per_ms",
}

for _alias, _canonical in _UNIT_ALIASES.items():
    setattr(_UnitEnum, _alias, getattr(_UnitEnum, _canonical))

del _alias, _canonical  # clean up module namespace

# Backward-compat aliases for old module names
import sys
from tvbo.datamodel import schema as tvbo_datamodel  # noqa: E402, F401
from tvbo.datamodel import pydantic as tvbopydantic  # noqa: E402, F401
sys.modules['tvbo.datamodel.tvbo_datamodel'] = tvbo_datamodel
sys.modules['tvbo.datamodel.tvbopydantic'] = tvbopydantic
