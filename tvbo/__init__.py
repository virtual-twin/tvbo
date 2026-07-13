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

__version__ = "0.5.3"
__maintainer__ = "Leon K. Martin (leon.martin@bih-charite.de)"
__contact__ = "petra.ritter@charite.de"
__status__ = "beta"

__copyright__ = "Copyright (c) 2026, Brain Simulation SectionCharité Universitätsmedizin Berlin"
__license__ = "EUPL-1.2-or-later"


def clean_temp():
    """Remove and recreate the TVBO temporary working directory.

    Deletes the entire `tvbo` scratch directory under the system temp
    location (ignoring errors if it is missing) and recreates it empty,
    clearing any cached or generated artifacts from previous runs.
    """
    shutil.rmtree(tempdir, ignore_errors=True)
    os.makedirs(tempdir)


# ---------------------------------------------------------------------------
# JAX backend configuration
# ---------------------------------------------------------------------------
# jax-metal (Apple GPU) plugin versions <= 0.1.1 are incompatible with
# JAX >= 0.7 and crash with "UNIMPLEMENTED: default_memory_space is not
# supported".  Detect this and fall back to the CPU backend so that
# *every* downstream JAX call works out of the box.
# Users can override by setting JAX_PLATFORMS or jax_default_device before
# importing tvbo.
_JAX_CONFIGURED = False


def _configure_jax_backend():
    """Fall back to CPU when the Metal plugin is broken.

    Invoked lazily the first time TVBO touches JAX for a simulation (from the
    compute modules), not at ``import tvbo``: importing JAX eagerly added
    ~0.3 s to every CLI invocation and to bare ``import tvbo``. Idempotent, so
    the guard runs at most once regardless of how many entry points call it.
    """
    global _JAX_CONFIGURED
    if _JAX_CONFIGURED:
        return
    _JAX_CONFIGURED = True
    # Respect explicit user override
    if "JAX_PLATFORMS" in os.environ:
        return
    try:
        import jax  # noqa: E402

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
                jax.config.update("jax_default_device", jax.devices("cpu")[0])
    except ImportError:
        pass  # JAX not installed – nothing to configure
# ---------------------------------------------------------------------------


# PyRates is an optional backend. Its networkx-dispatch monkeypatch lives in
# ``tvbo.adapters.pyrates`` and is applied there when the adapter actually runs,
# so importing tvbo (and the CLI) no longer pulls in pyrates at all.

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
    "DynamicalSystem": ".classes.dynamics",
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


# Helper methods (.plot(), …) are attached to the auto-generated schema classes
# by ``tvbo.datamodel`` when the datamodel is first imported, alongside the sibling
# Network/UnitEnum patches. Keeping that out of ``import tvbo`` avoids eagerly
# pulling the datamodel (pydantic, sympy, pandas) into every CLI invocation.
