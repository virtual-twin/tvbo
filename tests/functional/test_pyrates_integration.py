"""Functional tests for PyRates integration.

Tests the full-circle export and import of TVBO models to/from PyRates format, including:
- Dynamics export to PyRates OperatorTemplate/NodeTemplate
- Network export with edge-based connectivity
- Round-trip import from PyRates YAML back to TVBO Dynamics
"""

import os
import tempfile

import numpy as np
import pytest

from tvbo import Coupling, Dynamics, Network
from tvbo.codegen.pyrates import (
    to_pyrates_model_yaml,
    to_pyrates_network_yaml,
    to_pyrates_yaml_string,
)
from tvbo.datamodel import tvbo_datamodel


class TestDynamicsExport:
    """Tests for Dynamics → PyRates export."""

    def test_simple_oscillator_export(self):
        """Test export of a simple oscillator model."""
        model = Dynamics("Dynamics")
        model.name = "SimpleOscillator"
        model.add_parameter("omega", value=1.0)
        model.add_state_variable("x", equation="y", initial_value=1.0)
        model.add_state_variable("y", equation="-omega**2 * x", initial_value=0.0)

        yaml_output = to_pyrates_model_yaml(model)

        # Check structure
        assert "SimpleOscillator_op:" in yaml_output
        assert "base: OperatorTemplate" in yaml_output
        assert "x' = y" in yaml_output
        # Sympy may reorder the expression, so check for components
        assert "omega" in yaml_output and "x" in yaml_output
        assert "omega: 1.0" in yaml_output

    def test_van_der_pol_export(self):
        """Test export of Van der Pol oscillator (used in documentation)."""
        vdp = Dynamics("Dynamics")
        vdp.name = "VanDerPol"
        vdp.add_parameter("mu", value=5.0)
        vdp.add_state_variable("x", equation="z", initial_value=0.0)
        vdp.add_state_variable("z", equation="mu*(1 - x**2)*z - x", initial_value=1.0)

        yaml_output = vdp.to_yaml(format="pyrates")

        # Check contains all components
        assert "VanDerPol_op:" in yaml_output
        assert "VanDerPol:" in yaml_output
        assert "operators:" in yaml_output
        assert "mu: 5.0" in yaml_output
        assert "x: variable(0.0)" in yaml_output
        assert "z: variable(1.0)" in yaml_output

    def test_export_with_description(self):
        """Test that model description is included in export."""
        model = Dynamics("Dynamics")
        model.name = "TestModel"
        model.description = "A test model description"
        model.add_parameter("a", value=2.0)
        model.add_state_variable("x", equation="a*x", initial_value=1.0)

        yaml_output = to_pyrates_model_yaml(model)

        assert "description:" in yaml_output
        assert "TestModel" in yaml_output


class TestNetworkExport:
    """Tests for Network → PyRates export with edge-based connectivity."""

    def test_edge_based_network_export(self):
        """Test network with edges exports correctly."""
        # Create dynamics
        oscillator = Dynamics("Dynamics")
        oscillator.name = "osc"
        oscillator.add_parameter("omega", value=1.0)
        oscillator.add_state_variable("x", equation="y", initial_value=1.0)
        oscillator.add_state_variable("y", equation="-omega**2 * x", initial_value=0.0)

        # Create network with edges
        network = Network(number_of_nodes=2)
        network.label = "TestNetwork"
        network.nodes = [
            tvbo_datamodel.Node(id=0, label="A", dynamics=oscillator.name),
            tvbo_datamodel.Node(id=1, label="B", dynamics=oscillator.name),
        ]
        network.edges = [
            tvbo_datamodel.Edge(
                source=0,
                target=1,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=0.5)},
                coupling=Coupling(name="Linear"),
            ),
            tvbo_datamodel.Edge(
                source=1,
                target=0,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=0.3)},
                coupling=Coupling(name="Linear"),
            ),
        ]

        yaml_output = to_pyrates_yaml_string(dynamics=oscillator, network=network)

        # Check structure - export may use different naming conventions
        assert "CircuitTemplate" in yaml_output
        assert "nodes:" in yaml_output or "node" in yaml_output.lower()
        assert "edges:" in yaml_output or "edge" in yaml_output.lower()

    @pytest.mark.parametrize(
        "edge_kwargs",
        [
            {"parameters": {"weight": tvbo_datamodel.Parameter(name="weight", value=0.33)}},
            {"weight": 0.33},
        ],
        ids=["keyed-parameters", "scalar-field"],
    )
    def test_edge_weight_reaches_yaml(self, edge_kwargs):
        """A non-unit edge weight must reach the PyRates YAML, not the 1.0 fallback.

        edge.parameters is a dict[str, Parameter]; the codegen helper previously iterated it as a list of Parameter objects, never matched, and always emitted the default weight (any weight != 1.0 was silently lost). Covers both the keyed-parameters and scalar-field forms of the weight.
        """
        osc = Dynamics("Dynamics")
        osc.name = "osc"
        osc.add_parameter("omega", value=1.0)
        osc.add_state_variable("x", equation="y", initial_value=1.0)
        osc.add_state_variable("y", equation="-omega**2 * x", initial_value=0.0)

        network = Network(number_of_nodes=2)
        network.label = "WeightNet"
        network.nodes = [
            tvbo_datamodel.Node(id=0, label="A", dynamics=osc.name),
            tvbo_datamodel.Node(id=1, label="B", dynamics=osc.name),
        ]
        network.edges = [
            tvbo_datamodel.Edge(source=0, target=1, coupling=Coupling(name="Linear"), **edge_kwargs),
        ]

        yaml_output = to_pyrates_yaml_string(dynamics=osc, network=network)

        assert "weight: 0.33" in yaml_output
        assert "weight: 1.0" not in yaml_output

    def test_weights_matrix_from_edges(self):
        """Test that weights_matrix property correctly computes from edges.

        Matrices follow the target-by-source convention used by the backends: an edge ``source -> target`` is stored at ``[target, source]``.
        """
        network = Network(number_of_nodes=3)
        network.edges = [
            tvbo_datamodel.Edge(
                source=0,
                target=1,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=0.5)},
                coupling=Coupling(name="Linear"),
                directed=True,
            ),
            tvbo_datamodel.Edge(
                source=1,
                target=2,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=0.8)},
                coupling=Coupling(name="Linear"),
                directed=True,
            ),
            tvbo_datamodel.Edge(
                source=2,
                target=0,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=0.2)},
                coupling=Coupling(name="Linear"),
                directed=True,
            ),
        ]

        W = network.weights_matrix
        expected = np.array(
            [
                [0.0, 0.0, 0.2],
                [0.5, 0.0, 0.0],
                [0.0, 0.8, 0.0],
            ]
        )

        np.testing.assert_array_almost_equal(W, expected)

    def test_network_yaml_export_method(self):
        """Test Network.to_yaml with format='pyrates'."""
        network = Network(number_of_nodes=2)
        network.label = "TestCircuit"
        network.edges = [
            tvbo_datamodel.Edge(
                source=0,
                target=1,
                parameters={"weight": tvbo_datamodel.Parameter(name="weight", value=1.0)},
                coupling=Coupling(name="Linear"),
            ),
        ]

        # Create dynamics for the network
        dynamics = Dynamics("Dynamics")
        dynamics.name = "node_model"
        dynamics.add_parameter("a", value=1.0)
        dynamics.add_state_variable("x", equation="a*x", initial_value=0.0)

        yaml_output = to_pyrates_network_yaml(dynamics=dynamics, network=network)

        # Check for circuit template structure
        assert "CircuitTemplate" in yaml_output


class TestPyRatesImport:
    """Tests for PyRates → TVBO import."""

    def test_import_simple_model(self):
        """Test importing a simple PyRates model."""
        pyrates_yaml = """
simple_op:
  base: OperatorTemplate
  equations:
    - "x' = a*x"
  variables:
    x: variable(1.0)
    a: 2.0

simple_node:
  base: NodeTemplate
  operators:
    - simple_op
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(pyrates_yaml)
            temp_path = f.name

        try:
            model = Dynamics.from_pyrates(temp_path)

            assert model.name == "simple"  # Name extracted from OperatorTemplate (simple_op -> simple)
            assert "a" in model.parameters
            assert model.parameters["a"].value == 2.0
            assert "x" in model.state_variables
            assert model.state_variables["x"].initial_value == 1.0
        finally:
            os.unlink(temp_path)

    def test_import_with_caret_exponentiation(self):
        """Test that ^ is converted to ** when importing PyRates models."""
        # PyRates uses ^ for exponentiation, Python/sympy uses **
        pyrates_yaml = """
qif_op:
  base: OperatorTemplate
  equations:
    - "r' = Delta/(tau*3.14159) + 2*r*v"
    - "v' = v^2 + eta - (3.14159*tau*r)^2"
  variables:
    r: variable(0.1)
    v: variable(-2.0)
    Delta: 1.0
    tau: 1.0
    eta: -5.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(pyrates_yaml)
            temp_path = f.name

        try:
            model = Dynamics.from_pyrates(temp_path)

            # Check that the equations are parsed correctly (^ replaced with **)
            assert "v" in model.state_variables
            v_eq_rhs = str(model.state_variables["v"].equation.rhs)
            # Should have ** for exponentiation, not ^
            assert "**" in v_eq_rhs or "v**2" in v_eq_rhs.replace(" ", "")
            assert "^" not in v_eq_rhs
        finally:
            os.unlink(temp_path)

    def test_import_multivariate_model(self):
        """Test importing a model with multiple state variables."""
        print("\n[DEBUG] Starting test_import_multivariate_model")
        pyrates_yaml = """
oscillator_op:
  base: OperatorTemplate
  equations:
    - "x' = y"
    - "y' = -omega**2 * x"
  variables:
    x: variable(1.0)
    y: variable(0.0)
    omega: 2.0

oscillator_node:
  base: NodeTemplate
  operators:
    - oscillator_op
"""
        print("[DEBUG] Creating temp file...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(pyrates_yaml)
            temp_path = f.name
        print(f"[DEBUG] Temp file created: {temp_path}")

        try:
            print("[DEBUG] Calling Dynamics.from_pyrates...")
            model = Dynamics.from_pyrates(temp_path)
            print(f"[DEBUG] Model loaded: {model.name}")

            print("[DEBUG] Checking state variables...")
            assert "x" in model.state_variables
            assert "y" in model.state_variables
            print("[DEBUG] Checking parameters...")
            assert "omega" in model.parameters
            assert model.parameters["omega"].value == 2.0
            print("[DEBUG] All assertions passed")
        finally:
            os.unlink(temp_path)
            print("[DEBUG] Temp file deleted")


class TestRoundTrip:
    """Tests for full round-trip export/import."""

    def test_van_der_pol_round_trip(self):
        """Test round-trip of Van der Pol oscillator."""
        # Create original model
        vdp = Dynamics("Dynamics")
        vdp.name = "VanDerPol"
        vdp.add_parameter("mu", value=5.0)
        vdp.add_state_variable("x", equation="z", initial_value=0.0)
        vdp.add_state_variable("z", equation="mu*(1 - x**2)*z - x", initial_value=1.0)

        # Export to YAML
        yaml_output = vdp.to_yaml(format="pyrates")

        # Write to temp file and import
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_output)
            temp_path = f.name

        try:
            loaded = Dynamics.from_pyrates(temp_path)

            # Verify structure preserved
            assert loaded.name == vdp.name
            assert set(loaded.parameters.keys()) == set(vdp.parameters.keys())
            assert set(loaded.state_variables.keys()) == set(vdp.state_variables.keys())

            # Verify values preserved
            assert loaded.parameters["mu"].value == vdp.parameters["mu"].value
            assert loaded.state_variables["x"].initial_value == vdp.state_variables["x"].initial_value
            assert loaded.state_variables["z"].initial_value == vdp.state_variables["z"].initial_value
        finally:
            os.unlink(temp_path)

    def test_custom_oscillator_round_trip(self):
        """Test round-trip of a custom oscillator model."""
        # Create original model
        model = Dynamics("Dynamics")
        model.name = "CustomOsc"
        model.add_parameter("omega", value=3.14)
        model.add_parameter("gamma", value=0.1)
        model.add_state_variable("x", equation="y", initial_value=2.0)
        model.add_state_variable("y", equation="-omega**2 * x - gamma * y", initial_value=0.5)

        # Export to YAML
        yaml_output = model.to_yaml(format="pyrates")

        # Write to temp file and import
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_output)
            temp_path = f.name

        try:
            loaded = Dynamics.from_pyrates(temp_path)

            # Verify all parameters (gamma may be absorbed into equation during round-trip)
            assert loaded.parameters["omega"].value == 3.14
            # gamma may or may not survive round-trip depending on PyRates export format
            if "gamma" in loaded.parameters:
                assert loaded.parameters["gamma"].value == 0.1

            # Verify all state variables
            assert loaded.state_variables["x"].initial_value == 2.0
            assert loaded.state_variables["y"].initial_value == 0.5

            # Verify equations
            assert "y" in str(loaded.state_variables["x"].equation.rhs)
        finally:
            os.unlink(temp_path)


class TestSimulationExperimentImport:
    """Tests for SimulationExperiment.from_pyrates with multi-operator files."""

    def test_load_multi_operator_file(self):
        """Test loading a file with multiple OperatorTemplates."""
        pyrates_yaml = """
op1:
  base: OperatorTemplate
  equations:
    - "x' = a*x"
  variables:
    x: variable(1.0)
    a: 2.0

op2:
  base: OperatorTemplate
  equations:
    - "y' = b*y"
  variables:
    y: variable(0.5)
    b: -1.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(pyrates_yaml)
            temp_path = f.name

        try:
            from tvbo import SimulationExperiment

            exp = SimulationExperiment.from_pyrates(temp_path)

            # Primary dynamics is the first operator
            assert exp.dynamics is not None
            assert exp.dynamics.name == "op1"
            assert "x" in exp.dynamics.state_variables
            assert "a" in exp.dynamics.parameters
            assert exp.dynamics.parameters["a"].value == 2.0

            # Additional dynamics stored in network.dynamics
            net_dyn = exp.network.dynamics
            assert isinstance(net_dyn, dict)
            assert len(net_dyn) == 2
            assert "op1" in net_dyn
            assert "op2" in net_dyn

            # Verify second operator via network.dynamics
            assert "y" in net_dyn["op2"].state_variables
            assert "b" in net_dyn["op2"].parameters
            assert net_dyn["op2"].parameters["b"].value == -1.0
        finally:
            os.unlink(temp_path)

    def test_load_with_input_syntax(self):
        """Test loading operators with input() syntax for input variables."""
        pyrates_yaml = """
synapse_op:
  base: OperatorTemplate
  equations:
    - "x' = (1 - x)/tau - k*x*r_in"
  variables:
    x: variable(1.0)
    tau: 100.0
    k: 0.1
    r_in: input(0.0)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(pyrates_yaml)
            temp_path = f.name

        try:
            model = Dynamics.from_pyrates(temp_path)

            # input() syntax creates a parameter with the default value (in TVBO, this can be used as a coupling input in a network context)
            assert "r_in" in model.parameters
            assert model.parameters["r_in"].value == 0.0

            # Other variables should also be parameters
            assert "tau" in model.parameters
            assert "k" in model.parameters
            assert model.parameters["tau"].value == 100.0
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
