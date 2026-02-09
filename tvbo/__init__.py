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
import tempfile

ROOT = os.path.dirname(__file__)

# Use system temp directory for compatibility with containerized environments
tempdir = os.path.join(tempfile.gettempdir(), "tvbo")
os.makedirs(tempdir, exist_ok=True)

logging.disable(logging.CRITICAL)

__authors__ = [
    "Leon K. Martin",
    "Marius Pille",
    "Konstantin Bülau",
    "Leon Stefanovski",
    "Petra Ritter",
]

__version__ = "0.2.6"
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


# ---------------------------------------------------------------------------
# JAX backend configuration
# ---------------------------------------------------------------------------
# jax-metal (Apple GPU) plugin versions <= 0.1.1 are incompatible with
# JAX >= 0.7 and crash with "UNIMPLEMENTED: default_memory_space is not
# supported".  Detect this early and fall back to the CPU backend so that
# *every* downstream JAX call works out of the box.
# Users can override by setting JAX_PLATFORMS or jax_default_device before
# importing tvbo.
def _configure_jax_backend():
    """Fall back to CPU when the Metal plugin is broken."""
    # Respect explicit user override
    if "JAX_PLATFORMS" in os.environ:
        return
    try:
        import jax                                     # noqa: E402
        if jax.default_backend().upper() == "METAL":
            # Quick smoke-test: try the simplest device operation
            try:
                jax.numpy.zeros(1)
            except Exception:
                import warnings
                warnings.warn(
                    "jax-metal plugin detected but incompatible with the "
                    "installed JAX version. Falling back to CPU. "
                    "Uninstall jax-metal or upgrade it to fix this: "
                    "  pip uninstall jax-metal",
                    RuntimeWarning,
                    stacklevel=2,
                )
                jax.config.update(
                    "jax_default_device", jax.devices("cpu")[0]
                )
    except ImportError:
        pass  # JAX not installed – nothing to configure


_configure_jax_backend()
# ---------------------------------------------------------------------------

from .data import tvbo_data
from .data.tvbo_data.connectomes import Connectome, Network
from .data.tvbo_data.atlases import Atlas
from .export.experiment import SimulationExperiment
from .knowledge.study import SimulationStudy
from .knowledge.simulation import localdynamics
from .knowledge.simulation.localdynamics import Dynamics
from .knowledge.simulation.network import Coupling
from .knowledge.simulation.integration import Noise
from .knowledge.function import Function, LossFunction
