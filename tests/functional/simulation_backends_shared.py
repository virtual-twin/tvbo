"""Shared fixtures/utilities for simulation backend functional tests."""

import functools
import importlib


def _has(package: str) -> bool:
    """Return True if *package* can be imported."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ModuleNotFoundError, ValueError):
        return False


_HAVE_TVB = _has("tvb.simulator")
_HAVE_PYRATES = _has("pyrates")
_HAVE_TVBOPTIM = _has("tvboptim")
_HAVE_JULIACALL = _has("juliacall")
_HAVE_LEMS = _has("lems")
_HAVE_NEUROML = _has("pyneuroml")
_HAVE_NEURON = _has("neuron")
_HAVE_BRIAN2 = _has("brian2")
_HAVE_NETPYNE = _has("netpyne")
_HAVE_EDEN = _has("eden_simulator")


from tvbo import database_path

DATABASE_MODELS_DIR = database_path / "models"
JULIA_MODELS_DIR = DATABASE_MODELS_DIR / "julia"


def get_model_files():
    """Collect all model YAML files from database/models and database/models/julia."""
    model_files = []
    for filepath in DATABASE_MODELS_DIR.glob("*.yaml"):
        model_files.append(filepath)
    if JULIA_MODELS_DIR.exists():
        for filepath in JULIA_MODELS_DIR.glob("*.yaml"):
            model_files.append(filepath)
    return model_files


@functools.cache
def _meta(model_file):
    """The model file's YAML, read once."""
    import yaml

    with open(model_file) as fh:
        return yaml.safe_load(fh)


def _global_inputs_and_coupling_variables(model_file):
    """The number of global coupling inputs the model declares, its local ones (`is_local_coupling`) left out, and the number of its coupling variables."""
    from types import SimpleNamespace

    from tvbo.templates.base.utils import is_local_coupling

    meta = _meta(model_file)
    inputs = sum(
        1
        for name, spec in (meta.get("coupling_inputs") or {}).items()
        if not is_local_coupling(name, SimpleNamespace(local=(spec or {}).get("local", False)))
    )
    variables = sum(
        1 for sv in (meta.get("state_variables") or {}).values() if isinstance(sv, dict) and sv.get("coupling_variable")
    )
    return inputs, variables


def _tvb_compatible(model_file):
    """Return True if model can run on the TVB backend.

    TVB sizes its coupling array by the number of coupling variables (cvar), so a model with more *global* coupling inputs than coupling variables cannot run there; local coupling inputs are a separate dfun argument and do not count.
    """
    meta = _meta(model_file)
    if meta.get("autonomous") is False or meta.get("system_type") == "discrete":
        return False
    inputs, variables = _global_inputs_and_coupling_variables(model_file)
    return inputs <= max(variables, 1)


_SKIP_NON_TVB = {"ZerlautAdaptationSecondOrder"}
"""Models skipped on the non-TVB backends but kept on TVB. The second-order Zerlaut model's erfc/quadrature transfer functions produce enormous expressions that compile very slowly under JAX/XLA (and the other backends), exceeding the CI per-test timeout. TVB has the erfc + compilation path for it, so it stays covered there."""

_ALL_MODEL_FILES = get_model_files()

MODEL_FILES = [filepath for filepath in _ALL_MODEL_FILES if filepath.stem not in _SKIP_NON_TVB]
MODEL_IDS = [filepath.stem for filepath in MODEL_FILES]


def _couples(model_file):
    """Return True if the model's global coupling inputs, where it declares any, have a coupling variable to read, as a backend coupling nodes through their state needs; a spiking cell whose inputs are spike trains a synapse delivers declares none."""
    inputs, variables = _global_inputs_and_coupling_variables(model_file)
    return not inputs or variables > 0


COUPLED_MODEL_FILES = [filepath for filepath in MODEL_FILES if _couples(filepath)]
"""The model files the rate backends' single-node run tests take (`_couples`); the spiking cells left out are the event-driven backends' to run."""
COUPLED_MODEL_IDS = [filepath.stem for filepath in COUPLED_MODEL_FILES]


# Built from the full list so TVB still covers the _SKIP_NON_TVB models.
TVB_MODEL_FILES = [filepath for filepath in _ALL_MODEL_FILES if _tvb_compatible(filepath)]
TVB_MODEL_IDS = [filepath.stem for filepath in TVB_MODEL_FILES]


def _assert_timeseries(result, model):
    """Shared assertions for any TimeSeries result."""
    assert result is not None
    assert hasattr(result, "data")
    assert hasattr(result, "time")
    assert result.data.shape[0] > 0
