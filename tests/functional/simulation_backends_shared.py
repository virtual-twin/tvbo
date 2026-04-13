"""Shared fixtures/utilities for simulation backend functional tests."""

import importlib
from pathlib import Path


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


def _tvb_compatible(model_file):
    """Return True if model can run on the TVB backend."""
    import yaml

    with open(model_file) as fh:
        meta = yaml.safe_load(fh)
    if meta.get("autonomous") is False:
        return False
    if meta.get("system_type") == "discrete":
        return False
    # TVB coupling array size = number of coupling variables (cvar).
    # Models with more *global* coupling inputs than coupling variables can't
    # run on TVB.  local_coupling / lc_* are separate dfun arguments.
    ci = meta.get("coupling_inputs", {})
    n_global = sum(
        1 for name in ci
        if name != "local_coupling" and not name.startswith("lc_")
    )
    n_coupling_vars = sum(
        1 for sv in meta.get("state_variables", {}).values()
        if isinstance(sv, dict) and sv.get("coupling_variable")
    )
    if n_global > max(n_coupling_vars, 1):
        return False
    return True


MODEL_FILES = get_model_files()
MODEL_IDS = [filepath.stem for filepath in MODEL_FILES]

TVB_MODEL_FILES = [filepath for filepath in MODEL_FILES if _tvb_compatible(filepath)]
TVB_MODEL_IDS = [filepath.stem for filepath in TVB_MODEL_FILES]


def _assert_timeseries(result, model):
    """Shared assertions for any TimeSeries result."""
    assert result is not None
    assert hasattr(result, "data")
    assert hasattr(result, "time")
    assert result.data.shape[0] > 0
