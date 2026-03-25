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
from .db import *
from .tvbo_data import ATLAS_DIR

__all__ = [
    "SimulationResult",
    "AlgorithmResult",
    "OptimizationResult",
    "ExplorationResult",
    "ATLAS_DIR",
]


def __getattr__(name):
    if name in ("SimulationResult", "AlgorithmResult",
                "OptimizationResult", "ExplorationResult"):
        from .types import (
            SimulationResult,
            AlgorithmResult,
            OptimizationResult,
            ExplorationResult,
        )
        globals().update({
            "SimulationResult": SimulationResult,
            "AlgorithmResult": AlgorithmResult,
            "OptimizationResult": OptimizationResult,
            "ExplorationResult": ExplorationResult,
        })
        return globals()[name]
    raise AttributeError(f"module 'tvbo.data' has no attribute {name!r}")
