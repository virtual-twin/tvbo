"""A backend that emits no observation refuses an experiment that declares one, instead of returning the raw trajectory as if it had computed it.

The python backend accepted a declared `mean` observation and returned an empty observation set, and julia (DiffEq) and the heterogeneous tvboptim path dropped theirs the same way, each reporting success. NetworkDynamics.jl already refused; all four now share `refuse_observations`.
"""

import pytest

from tvbo import SimulationExperiment
from tvbo.adapters.base import refuse_observations, require_observations
from tvbo.data.types import ExperimentResult, SimulationResult

DYNAMICS = """
label: Dropped observation
dynamics:
  name: Relax
  parameters: {k: {value: 0.5}}
  state_variables:
    x: {equation: {rhs: "k - x"}, initial_value: 0.0}
integration: {method: euler, step_size: 1.0, duration: 10.0}
"""
OBSERVED = DYNAMICS + "observations:\n  m: {source: [x], aggregation: mean}\n"


def test_the_python_backend_refuses_a_declared_observation():
    with pytest.raises(NotImplementedError, match="python backend emits no observation, so m would be dropped"):
        SimulationExperiment.from_string(OBSERVED).run("python")


def test_an_experiment_without_observations_still_runs_on_python():
    assert SimulationExperiment.from_string(DYNAMICS).run("python").integration.data is not None


def test_the_refusal_names_every_declared_observation():
    experiment = SimulationExperiment.from_string(OBSERVED + "  s: {source: [x], aggregation: std}\n")
    with pytest.raises(NotImplementedError, match="julia backend emits no observation, so m, s would be dropped"):
        refuse_observations(experiment, "julia")


def test_a_result_without_any_observation_is_refused():
    experiment = SimulationExperiment.from_string(OBSERVED)
    with pytest.raises(NotImplementedError, match="pyrates backend emits no observation, so m would be dropped"):
        require_observations(experiment, ExperimentResult(integration=SimulationResult(observations={})), "pyrates")


def test_a_result_carrying_observations_or_no_trajectory_passes():
    experiment = SimulationExperiment.from_string(OBSERVED)
    require_observations(experiment, ExperimentResult(integration=SimulationResult(observations={"m": 1.0})), "tvb")
    require_observations(experiment, ExperimentResult(), "bifurcationkit")


@pytest.mark.backend_pyrates
def test_the_dispatcher_refuses_a_backend_that_never_reads_observations():
    pytest.importorskip("pyrates")
    with pytest.raises(NotImplementedError, match="pyrates backend emits no observation, so m would be dropped"):
        SimulationExperiment.from_string(OBSERVED).run("pyrates")
