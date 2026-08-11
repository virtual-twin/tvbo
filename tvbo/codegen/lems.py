#  lemsgenerator.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin, Konstantin Bülau
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
# LEMS-Generator
Create LEMS model from TVB-O"""

# %%
import numpy as np

import lems.api as lems

np.random.seed(1312)


# %% Functions for generating a LEMS model
def setup_lems_model():
    """Create a LEMS model preloaded with base dimensions and units.

    Builds an empty `lems.Model` and registers the physical dimensions (`voltage`, `time`, `current`) and their SI-prefixed units (`second`, `milliVolt`, `milliSecond`, `milliAmpere`) that TVB-O component definitions are expressed in.

    Returns:
        The initialized LEMS model ready to have components added to it.
    """
    model = lems.Model()

    model.add(lems.Dimension("voltage", m=1, l=2, t=-3, i=-1))
    model.add(lems.Dimension("time", t=1))
    model.add(lems.Dimension("current", i=1))
    model.add(lems.Unit("second", "s", "time", 1))
    model.add(lems.Unit("milliVolt", "mV", "voltage", -3))
    model.add(lems.Unit("milliSecond", "ms", "time", -3))
    model.add(lems.Unit("milliAmpere", "mA", "current", -12))

    return model
