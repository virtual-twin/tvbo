"""Test tvboptim experiment execution for all experiments in database/experiments."""

import pytest
import os

pytest.importorskip("tvboptim", reason="tvboptim not installed")

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from tvbo import SimulationExperiment, database_path

EXPERIMENTS_DIR = database_path / "experiments"

EXPERIMENTS = [
    "EI_Tuning_FIC_EIB_Optimization",
    "JR_MEG_FrequencyGradient_Optimization",
    "RWW_BOLD_FC_Optimization",
]


@pytest.mark.parametrize("experiment_name", EXPERIMENTS)
def test_experiment_runs(experiment_name):
    """Test that experiment runs with minimal iterations."""
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / f"{experiment_name}.yaml"))

    # Shrink any exploration grids so CI stays fast.
    if exp.explorations:
        for expl in exp.explorations.values():
            if expl.parameters:
                for param in expl.parameters.values():
                    if param.domain is not None and getattr(param.domain, "n", None):
                        param.domain.n = 2

    results = exp.run(mode="all", n_iterations=2, max_steps=2, format="tvboptim")

    assert results is not None
    assert "integration" in results
    assert results.integration is not None
