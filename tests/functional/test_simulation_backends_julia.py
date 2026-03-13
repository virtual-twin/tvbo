"""Functional tests for Julia-based backends."""

import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.classes.dynamics import Dynamics
from tests.functional.simulation_backends_shared import (
    MODEL_FILES,
    MODEL_IDS,
    _HAVE_JULIACALL,
    _assert_timeseries,
)


@pytest.mark.backend_julia
@pytest.mark.xdist_group("julia")
@pytest.mark.skipif(not _HAVE_JULIACALL, reason="juliacall not installed")
class TestJuliaBackends:
    """Single-node tests for Julia-based simulation backends."""

    @pytest.mark.backend_julia_diffeq
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_julia(self, model_file):
        """Run single-node simulation via DifferentialEquations.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("julia")
        _assert_timeseries(result, model)

    @pytest.mark.backend_networkdynamics
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_networkdynamics(self, model_file):
        """Run single-node simulation via NetworkDynamics.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("networkdynamics")
        _assert_timeseries(result, model)

    @pytest.mark.backend_mtk
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_mtk(self, model_file):
        """Run single-node simulation via ModelingToolkit.jl."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("mtk")
        _assert_timeseries(result, model)
