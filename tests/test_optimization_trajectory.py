"""An optimization that names `trajectory_observations` stores them at every saved iterate, along the step axis."""

import jax.numpy as jnp
import pytest
from tvboptim.types.parameter import Parameter
from tvboptim.types.stateutils import collect_parameters, partition_state

from tvbo.data.types import Bunch, OptimizationResult


def _state(a):
    return {"a": Parameter(jnp.asarray(a)), "k": jnp.asarray(3.0)}


def test_observe_iterates_simulates_each_saved_iterate_with_the_static_part_of_the_fit():
    iterates = [partition_state(_state(a))[0] for a in (0.0, 1.0)]
    fit = OptimizationResult(name="fit", state=_state(2.0), history={"parameters": Bunch(save=iterates)})

    def observe(state):
        values = collect_parameters(state)
        return {"y": values["a"] * values["k"] * jnp.ones(2), "unused": jnp.zeros(1)}

    fit.observe_iterates(observe, ["y"])

    assert list(fit.trajectory_observations) == ["y"]
    assert fit.trajectory_observations["y"].tolist() == [[0.0, 0.0], [3.0, 3.0]]


def test_the_iterates_are_mapped_in_batches_that_agree_with_one_at_a_time():
    iterates = [partition_state(_state(a))[0] for a in (0.0, 1.0, 2.0, 3.0, 4.0)]

    def observe(state):
        values = collect_parameters(state)
        return {"y": jnp.sin(values["a"]) * values["k"] * jnp.ones(3)}

    def observed(batch_size):
        fit = OptimizationResult(name="fit", state=_state(2.0), history={"parameters": Bunch(save=iterates)})
        fit.observe_iterates(observe, ["y"], batch_size=batch_size)
        return fit.trajectory_observations["y"]

    assert observed(2).shape == (5, 3)
    assert observed(2).tolist() == observed(None).tolist()


def test_a_fit_that_saved_no_iterate_says_so():
    fit = OptimizationResult(name="fit", state=_state(2.0), history={"loss": Bunch(save=[1.0])})
    with pytest.raises(ValueError, match="saved no parameter iterate"):
        fit.observe_iterates(lambda s: {"y": jnp.zeros(1)}, ["y"])


def _curated(name):
    from tvbo import SimulationExperiment, database_path

    return SimulationExperiment.from_file(str(database_path / "experiments" / f"{name}.yaml"))


@pytest.mark.backend_tvboptim
def test_a_fit_observes_every_saved_iterate_and_its_last_is_the_fitted_point():
    import numpy as np

    experiment = _curated("JR_MEG_FrequencyGradient_Optimization")
    experiment.explorations = None
    fit = experiment.optimizations["spectral_gradient_fit"]
    fit.max_iterations = 4
    fit.trajectory_observations = ["peak_frequencies"]
    result = experiment.run(mode="optimization", max_steps=4, format="tvboptim").optimizations["spectral_gradient_fit"]
    stacked = result.trajectory_observations["peak_frequencies"]
    fitted = result.simulation.observations["peak_frequencies"]
    assert stacked.shape[0] == len(result.state_trajectory)
    assert np.array_equal(stacked[-1], np.asarray(getattr(fitted, "data", fitted)))


def test_a_multi_stage_fit_refuses_trajectory_observations_before_any_backend_renders():
    experiment = _curated("RWW_BOLD_FC_Optimization")
    experiment.optimizations["fc_fitting"].trajectory_observations = ["fc"]
    with pytest.raises(ValueError, match="single-stage fit"):
        experiment.configure()


def test_a_fit_refining_an_exploration_front_refuses_trajectory_observations():
    experiment = _curated("Hopf_Pareto_ParallelOpt")
    experiment.optimizations["refine"].trajectory_observations = list(experiment.observations)[:1]
    with pytest.raises(ValueError, match="refines the front of exploration 'ga_presearch'"):
        experiment.configure()


def test_a_trajectory_observation_the_experiment_does_not_declare_is_refused():
    experiment = _curated("JR_MEG_FrequencyGradient_Optimization")
    experiment.explorations = None
    experiment.optimizations["spectral_gradient_fit"].trajectory_observations = ["peak_period"]
    with pytest.raises(ValueError, match="'peak_period', which the experiment does not declare"):
        experiment.configure()


@pytest.mark.backend_tvboptim
def test_an_iterate_traces_only_what_its_observations_read_and_refuses_a_host_callable():
    from tvbo.adapters.tvboptim import TvboptimAdapter

    experiment = _curated("JR_MEG_FrequencyGradient_Optimization")
    experiment.explorations = None
    experiment.optimizations["spectral_gradient_fit"].trajectory_observations = ["simulated_psd"]
    assert TvboptimAdapter(experiment).resolve_trajectory_observations() == (["simulated_psd"], ["simulated_psd"])

    experiment.observations["simulated_psd"].pipeline[-1].callable.traceable = False
    with pytest.raises(ValueError, match="simulated_psd runs a host callable"):
        TvboptimAdapter(experiment).resolve_trajectory_observations()
