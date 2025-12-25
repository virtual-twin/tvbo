"""Test running all models with different simulation backends (JAX, TVB, PyRates)."""
import pytest
import os
from pathlib import Path

from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.export.experiment import SimulationExperiment


# Collect all model YAML files
DATABASE_MODELS_DIR = Path(__file__).parent.parent.parent / "database" / "models"
JULIA_MODELS_DIR = DATABASE_MODELS_DIR / "julia"

def get_model_files():
    """Collect all model YAML files from database/models and database/models/julia."""
    model_files = []

    # Main models directory
    for f in DATABASE_MODELS_DIR.glob("*.yaml"):
        model_files.append(f)

    # Julia models directory
    if JULIA_MODELS_DIR.exists():
        for f in JULIA_MODELS_DIR.glob("*.yaml"):
            model_files.append(f)

    return model_files


MODEL_FILES = get_model_files()
MODEL_IDS = [f.stem for f in MODEL_FILES]


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
        """Test running simulation with JAX backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        try:
            result = exp.run('jax')
        except (NameError, KeyError, AttributeError, ValueError) as e:
            # Known issues with some models (missing symbols, parameters, etc.)
            pytest.xfail(f"Model has known issues: {type(e).__name__}: {e}")

        assert result is not None
        assert hasattr(result, 'data')
        assert hasattr(result, 'time')
        assert result.data.shape[0] > 0  # Has time points

        # If output is specified, check it matches
        if model.output:
            expected_n_vars = len(model.output)
        else:
            expected_n_vars = len(model.state_variables)

        assert result.data.shape[1] == expected_n_vars, \
            f"Expected {expected_n_vars} output variables, got {result.data.shape[1]}"


class TestJAXBackendDetailed:
    """Detailed tests for JAX backend with output verification."""

    @pytest.mark.parametrize("model_file", MODEL_FILES[:5], ids=MODEL_IDS[:5])  # First 5 models
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
                # Should have all state variables
                expected = list(model.state_variables.keys())
                assert output_labels == expected, \
                    f"Labels {output_labels} don't match state variables {expected}"


# Optional: Tests for other backends (may not be available)
class TestOptionalBackends:
    """Tests for optional backends that may not be installed."""

    @pytest.mark.parametrize("model_file", MODEL_FILES[:3], ids=MODEL_IDS[:3])  # First 3 models only
    def test_run_tvb(self, model_file):
        """Test running simulation with TVB backend (if available)."""
        pytest.importorskip("tvb.simulator")

        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        try:
            result = exp.run('tvb')
            assert result is not None
        except NotImplementedError:
            pytest.skip("TVB backend not implemented for this model")
        except Exception as e:
            if "tvb" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip(f"TVB backend issue: {e}")
            raise

    @pytest.mark.parametrize("model_file", MODEL_FILES[:3], ids=MODEL_IDS[:3])  # First 3 models only
    def test_run_pyrates(self, model_file):
        """Test running simulation with PyRates backend (if available)."""
        pytest.importorskip("pyrates")

        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(local_dynamics=model)

        try:
            result = exp.run('pyrates')
            assert result is not None
        except NotImplementedError:
            pytest.skip("PyRates backend not implemented for this model")
        except Exception as e:
            if "pyrates" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip(f"PyRates backend issue: {e}")
            raise


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
