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

# Backward-compat aliases for old module names
from tvbo.datamodel import schema as tvbo_datamodel  # noqa: E402, F401
from tvbo.datamodel import pydantic as tvbopydantic  # noqa: E402, F401
