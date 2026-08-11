# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2


"""
Data Module
===========

Access and manage TVB-O data.
"""

from .db import *  # noqa: F403  # database submodule re-exports
from .tvbo_data import ATLAS_DIR

_LAZY_FROM_TYPES = (
    "SimulationResult",
    "AlgorithmResult",
    "OptimizationResult",
    "ExplorationResult",
    "reassemble_shards",
)

__all__ = [*_LAZY_FROM_TYPES, "ATLAS_DIR"]


def __getattr__(name):
    if name in _LAZY_FROM_TYPES:
        from . import types

        value = getattr(types, name)
        globals()[name] = value  # cache: later access resolves as a normal global
        return value
    raise AttributeError(f"module 'tvbo.data' has no attribute {name!r}")
