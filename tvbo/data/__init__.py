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
from .tvbo_data import *

from .tvbo_data.connectomes import Network
from .tvbo_data.atlases import Atlas

# Result classes for simulation experiments
from .types import (
    SimulationResult,
    AlgorithmResult,
    OptimizationResult,
    ExplorationResult,
)
