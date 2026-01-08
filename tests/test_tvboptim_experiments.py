"""Test tvboptim experiment execution for all experiments in database/experiments."""

import pytest
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from tvbo import SimulationExperiment

EXPERIMENTS = [
    "EI_Tuning_FIC_EIB_Optimization",
    "JR_MEG_FrequencyGradient_Optimization",
    "RWW_BOLD_FC_Optimization",
]


@pytest.mark.parametrize("experiment_name", EXPERIMENTS)
def test_experiment_runs(experiment_name):
    """Test that experiment runs with minimal iterations."""
    exp = SimulationExperiment.from_file(f"database/experiments/{experiment_name}.yaml")
    results = exp.run(mode="all", n_iterations=2, max_steps=2, format="tvboptim")

    assert results is not None
    assert "integration" in results
    assert results.integration is not None
