"""Core functional tests for simulation backends (model loading + JAX)."""

import os
import tempfile

import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.classes.dynamics import Dynamics
from tests.functional.simulation_backends_shared import (
    DATABASE_MODELS_DIR,
    MODEL_FILES,
    MODEL_IDS,
)


class TestSimulationBackendsCore:
    """Core tests shared by all backend workflows."""

    @pytest.mark.backend_core
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_model_loads(self, model_file):
        """Test that each model loads without error."""
        model = Dynamics.from_file(model_file)
        assert model is not None
        assert model.name is not None
        assert len(model.state_variables) > 0

    @pytest.mark.backend_core
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_experiment_creation(self, model_file):
        """Test that each model can be wrapped in a SimulationExperiment."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        assert exp is not None
        assert exp.dynamics is not None
        assert exp.integration is not None
        if model.coupling_inputs:
            # Coupling resolution is deferred to configure(); the canonical location is network.coupling (mirrored to exp.coupling there).
            exp.configure()
            assert exp.network.coupling, "network.coupling should be populated"
            assert exp.coupling is not None

    @pytest.mark.backend_jax
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_jax(self, model_file):
        """Test running simulation with JAX backend (single-node)."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("jax")

        assert result is not None
        assert hasattr(result, "data")
        assert hasattr(result, "time")
        assert result.data.shape[0] > 0

        if model.output:
            expected_n_vars = len(model.output)
        else:
            expected_n_vars = len(model.state_variables)

        assert result.data.shape[1] == expected_n_vars, (
            f"Expected {expected_n_vars} output variables, got {result.data.shape[1]}"
        )


class TestJAXBackendDetailed:
    """Detailed tests for JAX backend with output verification."""

    @pytest.mark.backend_jax
    @pytest.mark.parametrize("model_file", MODEL_FILES[:5], ids=MODEL_IDS[:5])
    def test_output_variables_match(self, model_file):
        """Test that output labels match the model's output specification."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("jax")

        if hasattr(result, "labels_dimensions") and result.labels_dimensions:
            output_labels = result.labels_dimensions.get("State Variable", [])
            if model.output:
                assert output_labels == list(model.output), f"Labels {output_labels} don't match output spec {model.output}"
            else:
                expected = list(model.state_variables.keys())
                assert output_labels == expected, f"Labels {output_labels} don't match state variables {expected}"


@pytest.mark.backend_jax
def test_basic_jax_simulation():
    """Quick sanity check that at least one model runs with JAX."""
    model_file = DATABASE_MODELS_DIR / "Jansen1995.yaml"
    if not model_file.exists():
        model_file = next(DATABASE_MODELS_DIR.glob("*.yaml"))

    model = Dynamics.from_file(model_file)
    exp = SimulationExperiment(dynamics=model)
    result = exp.run("jax")

    assert result is not None
    assert result.data.shape[0] > 0


@pytest.mark.backend_jax
def test_empty_output_returns_all_state_variables():
    """Verify that models without output specification return all state variables."""
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        model = Dynamics.from_file(temp_path)
        assert model.output == [], "Model should have empty output"

        exp = SimulationExperiment(dynamics=model)
        result = exp.run("jax")

        assert result.data.shape[1] == 3, f"Expected 3 state variables, got {result.data.shape[1]}"

        if hasattr(result, "labels_dimensions") and result.labels_dimensions:
            labels = result.labels_dimensions.get("State Variable", [])
            assert labels == ["x", "y", "z"], f"Expected all state vars, got {labels}"
    finally:
        os.unlink(temp_path)


@pytest.mark.backend_jax
def test_output_specification_filters_variables():
    """Verify that output specification filters to only specified variables."""
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        model = Dynamics.from_file(temp_path)
        assert model.output == ["x", "sum_xy"]

        exp = SimulationExperiment(dynamics=model)
        result = exp.run("jax")

        assert result.data.shape[1] == 2, f"Expected 2 output variables, got {result.data.shape[1]}"

        if hasattr(result, "labels_dimensions") and result.labels_dimensions:
            labels = result.labels_dimensions.get("State Variable", [])
            assert labels == ["x", "sum_xy"], f"Expected ['x', 'sum_xy'], got {labels}"
    finally:
        os.unlink(temp_path)
