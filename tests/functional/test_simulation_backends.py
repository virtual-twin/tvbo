"""Test single-node simulation across all backends.

Goal: verify that a single YAML specification file drives complete, correct
single-node simulations for every supported backend.

Planned split (TODO):
    1. Single-node (this file) — SimulationExperiment(local_dynamics=model)
    2. Multi-node — SimulationExperiment.from_file(...) with full network YAML

Backends covered
----------------
Core (always available):
    - jax          : JAX / autodiff backend
Python optional (importorskip):
    - tvb          : The Virtual Brain simulator
    - pyrates      : PyRates rate-equations backend
    - tvboptim     : tvboptim JAX network dynamics backend
Julia optional (importorskip "juliacall"):
    - julia        : DifferentialEquations.jl via juliacall
    - networkdynamics : NetworkDynamics.jl via juliacall
    - mtk          : ModelingToolkit.jl via juliacall
"""
import pytest
import os
import importlib
from pathlib import Path

from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.export.experiment import SimulationExperiment


def _has(package: str) -> bool:
    """Return True if *package* can be imported."""
    return importlib.util.find_spec(package) is not None


# Evaluate once at collection time so parametrized tests are skipped in bulk
# rather than triggering 63 individual importorskip calls.
_HAVE_TVB = _has("tvb.simulator")
_HAVE_PYRATES = _has("pyrates")
_HAVE_TVBOPTIM = _has("tvboptim")
_HAVE_JULIACALL = _has("juliacall")


# ---------------------------------------------------------------------------
# Collect model YAML files
# ---------------------------------------------------------------------------
DATABASE_MODELS_DIR = Path(__file__).parent.parent.parent / "database" / "models"
JULIA_MODELS_DIR = DATABASE_MODELS_DIR / "julia"


def get_model_files():
    """Collect all model YAML files from database/models and database/models/julia."""
    model_files = []
    for f in DATABASE_MODELS_DIR.glob("*.yaml"):
        model_files.append(f)
    if JULIA_MODELS_DIR.exists():
        for f in JULIA_MODELS_DIR.glob("*.yaml"):
            model_files.append(f)
    return model_files


def _tvb_compatible(model_file):
    """Return True if model can run on the TVB backend.

    TVB's dfun has no explicit time argument (no non-autonomous support)
    and only handles continuous ODE/SDE (no discrete maps).
    """
    import yaml
    with open(model_file) as fh:
        meta = yaml.safe_load(fh)
    if meta.get('autonomous') is False:
        return False
    if meta.get('system_type') == 'discrete':
        return False
    return True


MODEL_FILES = get_model_files()
MODEL_IDS = [f.stem for f in MODEL_FILES]

TVB_MODEL_FILES = [f for f in MODEL_FILES if _tvb_compatible(f)]
TVB_MODEL_IDS = [f.stem for f in TVB_MODEL_FILES]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_timeseries(result, model):
    """Shared assertions for any TimeSeries result."""
    assert result is not None
    assert hasattr(result, 'data')
    assert hasattr(result, 'time')
    assert result.data.shape[0] > 0


# ---------------------------------------------------------------------------
# Core: model loading & experiment creation
# ---------------------------------------------------------------------------

class TestSimulationBackends:
    """Test simulation backends for all models."""

    @pytest.fixture(scope="class")
    def loaded_models(self):
        """Load all models once for the test class."""
        models = {}
        for filepath in MODEL_FILES:
            try:
                model = Dynamics.from_file(filepath)
                models[filepath.stem] = model
            except Exception as e:
                models[filepath.stem] = e
        return models

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_model_loads(self, model_file):
        """Test that each model loads without error."""
        model = Dynamics.from_file(model_file)
        assert model is not None
        assert model.name is not None
        assert len(model.state_variables) > 0

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_experiment_creation(self, model_file):
        """Test that each model can be wrapped in a SimulationExperiment."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        assert exp is not None
        assert exp.local_dynamics is not None
        assert exp.integration is not None
        assert exp.coupling is not None

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_jax(self, model_file):
        """Test running simulation with JAX backend (single-node)."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        result = exp.run('jax')

        _assert_timeseries(result, model)

        if model.output:
            expected_n_vars = len(model.output)
        else:
            expected_n_vars = len(model.state_variables)

        assert result.data.shape[1] == expected_n_vars, \
            f"Expected {expected_n_vars} output variables, got {result.data.shape[1]}"


# ---------------------------------------------------------------------------
# JAX output-label detail
# ---------------------------------------------------------------------------

class TestJAXBackendDetailed:
    """Detailed tests for JAX backend with output verification."""

    @pytest.mark.parametrize("model_file", MODEL_FILES[:5], ids=MODEL_IDS[:5])
    def test_output_variables_match(self, model_file):
        """Test that output labels match the model's output specification."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('jax')

        if hasattr(result, 'labels_dimensions') and result.labels_dimensions:
            output_labels = result.labels_dimensions.get("State Variable", [])
            if model.output:
                assert output_labels == list(model.output), \
                    f"Labels {output_labels} don't match output spec {model.output}"
            else:
                expected = list(model.state_variables.keys())
                assert output_labels == expected, \
                    f"Labels {output_labels} don't match state variables {expected}"


# ---------------------------------------------------------------------------
# Optional Python backends — each class skipped as a unit if pkg missing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_TVB, reason="tvb-library not installed")
class TestTVBBackend:
    """Single-node tests for The Virtual Brain backend.

    Non-autonomous and discrete models are excluded at collection time
    via TVB_MODEL_FILES since TVB's dfun has no explicit time argument
    and only supports continuous ODE/SDE.
    """

    @pytest.mark.parametrize("model_file", TVB_MODEL_FILES, ids=TVB_MODEL_IDS)
    def test_run_tvb(self, model_file):
        """Run single-node simulation with The Virtual Brain backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('tvb')
        assert result is not None


@pytest.mark.skipif(not _HAVE_PYRATES, reason="pyrates not installed")
class TestPyRatesBackend:
    """Single-node tests for the PyRates backend."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_pyrates(self, model_file):
        """Run single-node simulation with PyRates backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('pyrates')
        assert result is not None


@pytest.mark.skipif(not _HAVE_TVBOPTIM, reason="tvboptim not installed")
class TestTvboptimBackend:
    """Single-node tests for the tvboptim JAX backend."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_tvboptim(self, model_file):
        """Run single-node simulation with tvboptim JAX backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('tvboptim')
        assert result is not None


# ---------------------------------------------------------------------------
# Optional Julia backends
# ---------------------------------------------------------------------------

@pytest.mark.xdist_group("julia")  # all Julia tests run on the same worker
@pytest.mark.skipif(not _HAVE_JULIACALL, reason="juliacall not installed")
class TestJuliaBackends:
    """Single-node tests for Julia-based simulation backends.

    juliacall only supports one Julia runtime per OS process. The
    xdist_group("julia") marker ensures all tests in this class are sent to
    the same xdist worker, so Julia is initialised exactly once.  All other
    backend tests still run on the remaining workers in parallel.

    The class-level skipif deselects all 3×N parametrized tests in one step
    when juliacall is absent, instead of generating N individual skip records.
    """

    # ------------------------------------------------------------------
    # julia — DifferentialEquations.jl
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_julia(self, model_file):
        """Run single-node simulation via DifferentialEquations.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        result = exp.run('julia')
        _assert_timeseries(result, model)

    # ------------------------------------------------------------------
    # NetworkDynamics.jl
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_networkdynamics(self, model_file):
        """Run single-node simulation via NetworkDynamics.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        result = exp.run('networkdynamics')
        _assert_timeseries(result, model)

    # ------------------------------------------------------------------
    # ModelingToolkit.jl
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_mtk(self, model_file):
        """Run single-node simulation via ModelingToolkit.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        result = exp.run('mtk')
        _assert_timeseries(result, model)


# Standalone test for quick verification
def test_basic_jax_simulation():
    """Quick sanity check that at least one model runs with JAX."""
    model_file = DATABASE_MODELS_DIR / "Jansen1995.yaml"
    if not model_file.exists():
        model_file = next(DATABASE_MODELS_DIR.glob("*.yaml"))

    model = Dynamics.from_file(model_file)
    exp = SimulationExperiment(local_dynamics=model)
    result = exp.run('jax')

    assert result is not None
    assert result.data.shape[0] > 0


def test_empty_output_returns_all_state_variables():
    """Verify that models without output specification return all state variables."""
    import tempfile

    yaml_content = """
name: TestEmptyOutput
parameters:
    a:
        value: 1.0
state_variables:
    x:
        equation:
            rhs: a * x
    y:
        equation:
            rhs: -a * y
    z:
        equation:
            rhs: x - y
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        model = Dynamics.from_file(temp_path)
        assert model.output == [], "Model should have empty output"

        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('jax')

        # Should have all 3 state variables
        assert result.data.shape[1] == 3, \
            f"Expected 3 state variables, got {result.data.shape[1]}"

        if hasattr(result, 'labels_dimensions') and result.labels_dimensions:
            labels = result.labels_dimensions.get("State Variable", [])
            assert labels == ['x', 'y', 'z'], f"Expected all state vars, got {labels}"
    finally:
        os.unlink(temp_path)


def test_output_specification_filters_variables():
    """Verify that output specification filters to only specified variables."""
    import tempfile

    yaml_content = """
name: TestOutputFilter
parameters:
    a:
        value: 1.0
state_variables:
    x:
        equation:
            rhs: a * x
    y:
        equation:
            rhs: -a * y
    z:
        equation:
            rhs: x - y
derived_variables:
    sum_xy:
        equation:
            rhs: x + y
output:
    - x
    - sum_xy
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        model = Dynamics.from_file(temp_path)
        assert model.output == ['x', 'sum_xy']

        exp = SimulationExperiment(local_dynamics=model)
        result = exp.run('jax')

        # Should have only 2 output variables (x and sum_xy)
        assert result.data.shape[1] == 2, \
            f"Expected 2 output variables, got {result.data.shape[1]}"

        if hasattr(result, 'labels_dimensions') and result.labels_dimensions:
            labels = result.labels_dimensions.get("State Variable", [])
            assert labels == ['x', 'sum_xy'], f"Expected ['x', 'sum_xy'], got {labels}"
    finally:
        os.unlink(temp_path)
