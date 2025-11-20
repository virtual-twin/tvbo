# -*- coding: utf-8 -*-
#
# Auto-generated file

import argparse
import scipy
import numpy as np
import pandas as pd
from numba import float64
from numba import float64 as f64
from numba import guvectorize
from numpy import *
from tvb.basic.neotraits.api import Attr, Final, HasTraits, List, NArray, Range
from tvb.simulator.simulator import Simulator
from tvb.simulator.common import simple_gen_astr
from tvb.simulator.coupling import Coupling, SparseCoupling
from tvb.simulator.history import SparseHistory
from tvb.simulator.models.base import Model
from tvb.simulator.noise import Additive, Multiplicative
from tvb.simulator.integrators import Integrator, IntegratorStochastic
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
