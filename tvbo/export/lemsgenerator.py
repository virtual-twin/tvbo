#  lemsgenerator.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin, Konstantin Bülau
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
# LEMS-Generator
Create LEMS model from TVB-O
"""

# %%
import fileinput
import numpy as np

import lems.api as lems
import sympy

from tvbo.knowledge import config, ontology
from tvbo.knowledge.ontology import *
from tvbo.knowledge.simulation import equations, network


# WD = abspath(dirname(__file__))
np.random.seed(1312)


# %% Functions for generating a LEMS model
def setup_lems_model():
    model = lems.Model()

    model.add(lems.Dimension("voltage", m=1, l=2, t=-3, i=-1))
    model.add(lems.Dimension("time", t=1))
    model.add(lems.Dimension("current", i=1))
    model.add(lems.Unit("second", "s", "time", 1))
    model.add(lems.Unit("milliVolt", "mV", "voltage", -3))
    model.add(lems.Unit("milliSecond", "ms", "time", -3))
    model.add(lems.Unit("milliAmpere", "mA", "current", -12))

    return model
