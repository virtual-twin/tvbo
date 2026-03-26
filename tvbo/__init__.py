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
import warnings

# ---------------------------------------------------------------------------
# Suppress harmless requests dependency warning
# ---------------------------------------------------------------------------
# requests 2.32.x checks chardet<6 but pyshex installs chardet 6.x.
# requests itself uses charset-normalizer (which *is* compatible); the
# warning is a false positive.  Suppress it before anything imports
# requests so the user never sees it.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3.*or chardet.*doesn't match a supported version",
    category=Warning,
    module=r"requests",
)
# ---------------------------------------------------------------------------

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

__version__ = "0.3.4"
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

# ---------------------------------------------------------------------------
# PyRates / networkx compatibility
# ---------------------------------------------------------------------------
def _patch_pyrates_networkx_backend():
    """Fix networkx 3.4+ backend dispatch conflict with PyRates.

    networkx ≥ 3.4 decorates ``MultiDiGraph.__new__`` with
    ``@nx._dispatchable`` which intercepts a ``backend`` keyword argument.
    PyRates' ``ComputeGraph(backend='default')`` triggers this and raises
    ``ImportError: 'default' backend is not installed``.

    We replace ``ComputeGraph.__new__`` with plain ``object.__new__`` so
    the decorator is removed.  Applied at tvbo import time so that all
    code paths (adapter, export, doc notebooks) benefit.
    """
    try:
        from pyrates.backend.computegraph import (
            ComputeGraph, ComputeGraphBackProp,
        )
    except ImportError:
        return

    def _plain_new(cls, *_args, **_kwargs):
        return object.__new__(cls)

    # Check if the __new__ is wrapped by networkx dispatch
    for klass in (ComputeGraph, ComputeGraphBackProp):
        try:
            klass(backend='default')
        except (ImportError, TypeError):
            klass.__new__ = _plain_new


_patch_pyrates_networkx_backend()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lazy public API — imports happen on first attribute access
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    "database_path": ".data.registry",
    "Connectome": ".classes.network",
    "Network": ".classes.network",
    "Atlas": ".classes.atlas",
    "SimulationExperiment": ".classes",
    "SimulationStudy": ".classes",
    "Dynamics": ".classes.dynamics",
    "Continuation": ".classes.continuation",
    "Coupling": ".classes.coupling",
    "Noise": ".classes.noise",
    "Observation": ".classes.observation",
    "Function": ".classes.function",
    "LossFunction": ".classes.function",
    "ExperimentResult": ".data.types",
    "TimeSeries": ".data.types",
}

# Special renames for backward compatibility
_LAZY_ALIASES = {
    "Model": ("Dynamics", ".classes.dynamics"),
    "LocalDynamics": ("Dynamics", ".classes.dynamics"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        attr = getattr(mod, name if name != "database_path" else "DATABASE_ROOT")
        globals()[name] = attr
        return attr
    if name in _LAZY_ALIASES:
        real_name, module = _LAZY_ALIASES[name]
        import importlib
        mod = importlib.import_module(module, __name__)
        attr = getattr(mod, real_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'tvbo' has no attribute {name!r}")
