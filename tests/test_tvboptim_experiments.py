"""Test tvboptim experiment execution for all experiments in database/experiments."""

import pytest
import os

pytest.importorskip("tvboptim", reason="tvboptim not installed")

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from tvbo import SimulationExperiment, database_path
from tvbo.utils import as_list

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
            for axis in as_list(expl.space):
                if axis.domain is not None and getattr(axis.domain, "n", None):
                    axis.domain.n = 2

    # Shrink algorithm and optimization iteration counts so CI stays fast.
    if exp.algorithms:
        for alg in exp.algorithms.values():
            if getattr(alg, "n_iterations", None):
                alg.n_iterations = 2
    if exp.optimizations:
        for opt in exp.optimizations.values():
            if getattr(opt, "max_iterations", None):
                opt.max_iterations = 2
            stages = getattr(opt, "stages", None) or []
            stages_iter = stages.values() if hasattr(stages, "values") else stages
            for stage in stages_iter:
                if getattr(stage, "max_iterations", None):
                    stage.max_iterations = 2

    results = exp.run(mode="all", n_iterations=2, max_steps=2, format="tvboptim")

    assert results is not None
    assert "integration" in results
    assert results.integration is not None
