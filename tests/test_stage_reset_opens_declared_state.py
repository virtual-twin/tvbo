"""AlgorithmStage.reset_state: a reset stage opens from the model's declared initial conditions.

The slot promises that a stage declaring it starts "from the model's declared initial conditions rather than the previous stage's endpoint", carrying only the update rules' tuned targets across: state variables, delay and coupling history, observation window buffers and monitors all reset, and the noise restarts from the run's seed. A dependency that tunes nothing therefore cannot change what a reset schedule computes, however long it ran, because everything it leaves behind is dynamical state. That is the behaviour of a simulator restarted once per stage with only its learned parameters read back in.
"""

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.datamodel import AlgorithmStage, Parameter

_EXP = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"
_SCHEDULE = ((0.10, 4), (0.05, 4))


def _experiment(dependency_iterations, reset, stages):
    """FIC then FIC_EIB on a tiny schedule, the FIC dependency at a learning rate of zero so that it tunes nothing."""
    exp = SimulationExperiment.from_file(_EXP)
    fe, fic = exp.algorithms["fic_eib"], exp.algorithms["fic"]
    fe.stages = [
        AlgorithmStage(
            n_iterations=3,
            reset_state=reset,
            arguments=[Parameter(name="eta", value=eta), Parameter(name="window_size", value=window)],
        )
        for eta, window in _SCHEDULE[:stages]
    ]
    fe.n_iterations, fic.n_iterations = 3, dependency_iterations
    for hp in fic.hyperparameters:
        if str(hp.name) == "eta":
            hp.value = 0.0
    return exp


def _tuned(algo):
    return {
        "J_i": np.asarray(algo.state.dynamics.J_i),
        "wLRE": np.asarray(algo.state.coupling.EIBLinearCoupling.wLRE),
        "wFFI": np.asarray(algo.state.coupling.EIBLinearCoupling.wFFI),
    }


def _run(dependency_iterations, reset, stages):
    algorithms = _experiment(dependency_iterations, reset, stages).run("tvboptim", mode="algorithms", quiet=True).algorithms
    return _tuned(algorithms["fic"]), _tuned(algorithms["fic_eib"])


@pytest.mark.parametrize("stages", [1, 2], ids=["first_stage", "every_stage"])
def test_a_reset_schedule_does_not_depend_on_how_long_a_dependency_that_tunes_nothing_ran(stages):
    short_dependency, short = _run(2, True, stages)
    long_dependency, long = _run(5, True, stages)
    np.testing.assert_array_equal(
        short_dependency["J_i"], long_dependency["J_i"], err_msg="the dependency tuned something, so the case tests nothing"
    )
    for name in ("wLRE", "wFFI", "J_i"):
        np.testing.assert_array_equal(
            short[name], long[name], err_msg=f"{name} of a reset schedule moved with the dependency's length"
        )


def test_a_continuous_schedule_continues_from_the_dependency():
    _, short = _run(2, False, 2)
    _, long = _run(5, False, 2)
    assert np.abs(short["wLRE"] - long["wLRE"]).max() > 1e-6
