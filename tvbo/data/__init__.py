#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
Data Module
===========

Access and manage TVB-O data.
"""

from .db import *  # noqa: F403  # database submodule re-exports
from .tvbo_data import ATLAS_DIR

# Names served lazily from the (heavy: jax/xarray) ``.types`` module, so
# ``import tvbo.data`` stays cheap for callers that only need ``db`` / ATLAS_DIR.
# Single source of truth — add a symbol here and it flows to __all__ + __getattr__.
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
