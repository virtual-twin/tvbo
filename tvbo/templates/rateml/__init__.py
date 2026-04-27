# -*- coding: utf-8 -*-
#
# RateML Templates
# ================
#
# Generate TVB-compatible model code (Python/Numba) and GPU kernels (CUDA)
# from TVBO SimulationExperiment specifications.
#
# This module provides templates for generating code in the style of RateML
# (https://github.com/the-virtual-brain/tvb-root/tree/master/tvb_contrib/tvb/contrib/rateML)
# but using TVBO's YAML-based model specifications instead of LEMS XML.
#

from . import utils

__all__ = ["utils"]
