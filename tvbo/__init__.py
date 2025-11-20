# Copyright Berlin Institute of Health / Charité University Medicine Berlin
# Department of Neurology and Experimental Neurology
# Brain Simulation Section

"""
Welcome to the TVB-O project!
==============================
TVB-O is a Python package for understanding and generating large-scale brain network models.
"""

import logging
import os
import shutil

ROOT = os.path.dirname(__file__)

tempdir = os.path.join(os.path.dirname(__file__), ".temp")
os.makedirs(tempdir, exist_ok=True)

logging.disable(logging.CRITICAL)

__authors__ = [
    "Leon K. Martin",
    "Marius Pille",
    "Konstantin Bülau",
    "Leon Stefanovski",
    "Petra Ritter",
]

__version__ = "0.2.3"
__maintainer__ = "Leon K. Martin (leon.martin@bih-charite.de)"
__contact__ = "petra.ritter@charite.de"
__status__ = "beta"

__copyright__ = (
    "Copyright (c) 2023, "
    "Brain Simulation Section"
    "Charité Universitätsmedizin Berlin"
)
__license__ = "EUPL-1.2-or-later"


def clean_temp():
    shutil.rmtree(tempdir, ignore_errors=True)
    os.makedirs(tempdir)


from .data import tvbo_data
from .data.tvbo_data.connectomes import Connectome
from .data.tvbo_data.atlases import Atlas
from .export.experiment import SimulationExperiment
from .knowledge.study import SimulationStudy
from .knowledge.simulation import localdynamics
from .knowledge.simulation.localdynamics import Dynamics
from .knowledge.simulation.network import Coupling
from .knowledge.simulation.integration import Noise
