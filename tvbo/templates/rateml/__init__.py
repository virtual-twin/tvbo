# -*- coding: utf-8 -*-
#
# RateML Templates
# ================
#
# Generate TVB-compatible model code (Python/Numba) and GPU kernels (CUDA) from TVBO SimulationExperiment specifications.
#
# This module provides templates for generating code in the style of RateML (https://github.com/the-virtual-brain/tvb-root/tree/master/tvb_contrib/tvb/contrib/rateML) but using TVBO's YAML-based model specifications instead of LEMS XML.
#

"""RateML-style code generation templates for TVBO models.

Provides templates that render TVB-compatible model code (Python/Numba) and
CUDA GPU kernels from TVBO `SimulationExperiment` specifications, following the conventions of TVB's RateML while sourcing dynamics from TVBO's YAML model
specifications instead of LEMS XML.
"""

from . import utils

__all__ = ["utils"]
