import numpy as np
import pytest
from linkml_runtime.loaders import yaml_loader

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment, database_path
from tvbo.datamodel.schema import Observation as SchemaObservation
from tvbo.templates.tvboptim.utils import get_observation_refs


def _reduced_wong_wang_without_local_coupling_input():
    dynamics = Dynamics.from_db("ReducedWongWang")
    dynamics.local_coupling_term = "local_coupling"
    dynamics.coupling_inputs.pop("local_coupling", None)
    return dynamics


def _two_node_bold_experiment(duration=1000.0):
    network = Network.from_matrix(
        weights=np.array([[0.0, 0.2], [0.1, 0.0]]),
        lengths=np.zeros((2, 2)),
        labels=["left", "right"],
    )
    network.coupling["Linear"] = Coupling.from_db("Linear")
    return SimulationExperiment(
        dynamics=_reduced_wong_wang_without_local_coupling_input(),
        network=network,
        integration={"method": "Heun", "duration": duration, "step_size": 0.1},
        observations=[
            Observation.from_db("TemporalAverage"),
            Observation.from_db("Bold_TVB"),
        ],
    )


def _aligned_observation_data(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim == right.ndim - 1 and right.shape[-1] == 1:
        right = right[..., 0]
    if right.ndim == left.ndim - 1 and left.shape[-1] == 1:
        left = left[..., 0]
    if left.shape[0] == right.shape[0] + 1:
        left = left[1:]
    elif right.shape[0] == left.shape[0] + 1:
        right = right[1:]
    return left, right


def test_bold_tvb_metadata_resolves_keyed_parameter_names():
    path = database_path / "observation_models" / "bold_tvb.yaml"
    observation = yaml_loader.load(str(path), target_class=SchemaObservation)

    assert observation.name == "BOLD_TVB"
    assert observation.period == 720.0
    assert observation.parameters["TR"].name == "TR"
    assert observation.pipeline[1].equation.parameters["tau_f"].name == "tau_f"
    assert observation.pipeline[-1].equation.parameters["V_0"].name == "V_0"


def test_tvb_adapter_resolves_bold_period_from_metadata():
    pytest.importorskip("tvb")
    from tvbo.adapters.tvb import to_tvb_monitor

    monitor = to_tvb_monitor(Observation.from_db("Bold_TVB"))

    assert monitor.period == 720.0


def test_tvboptim_uses_native_monitors_for_tvb_class_references():
    pytest.importorskip("tvboptim")
    from tvbo.templates.tvboptim.observations import TVBBold, TVBTemporalAverage

    experiment = _two_node_bold_experiment(duration=10.0)

    _, observation_names = get_observation_refs(experiment.observations)
    assert observation_names == ["TemporalAverage", "BOLD_TVB"]

    code = experiment.render_code("tvboptim")

    assert "from tvbo.templates.tvboptim.observations import TVBBold as _ExtTVBBold" in code
    assert "from tvbo.templates.tvboptim.observations import TVBTemporalAverage as _ExtTVBTemporalAverage" in code
    assert "'hrf_kernel': 'FirstOrderVolterra'" not in code

    namespace = experiment.execute("tvboptim")
    temporal_average = namespace.Temporalaverage(dt=0.1)
    bold = namespace.BoldTvb(dt=0.1)

    assert isinstance(temporal_average._monitor, TVBTemporalAverage)
    assert isinstance(bold._monitor, TVBBold)
    assert bold._monitor.period == 720.0
    assert bold._monitor.downsample_period == 4.0
    assert bold._monitor.k_1 == 5.6
    assert bold._monitor.V_0 == 0.02


def test_tvboptim_allows_unbound_declared_coupling_inputs_as_zero():
    pytest.importorskip("tvboptim")

    result = SimulationExperiment(dynamics=Dynamics.from_db("Kuramoto2")).run(
        "tvboptim",
        duration=10,
        quiet=True,
    )

    signal = result.sel(variable="signal")
    assert signal.data.shape[0] > 0


def test_tvboptim_observations_match_native_tvb():
    pytest.importorskip("tvb")
    pytest.importorskip("tvboptim")

    experiment = _two_node_bold_experiment()

    tvboptim_result = experiment.run("tvboptim")
    tvb_result = experiment.run("tvb")

    assert set(tvboptim_result.integration.observations.keys()) == {"TemporalAverage", "BOLD_TVB"}
    assert set(tvb_result.integration.observations.keys()) == {"temporalaverage", "bold_tvb"}

    comparisons = [
        (
            tvboptim_result.integration.observations["TemporalAverage"].data,
            tvb_result.integration.observations["temporalaverage"].data,
        ),
        (
            tvboptim_result.integration.observations["BOLD_TVB"].data,
            tvb_result.integration.observations["bold_tvb"].data,
        ),
    ]
    for tvboptim_data, tvb_data in comparisons:
        tvboptim_data, tvb_data = _aligned_observation_data(tvboptim_data, tvb_data)
        np.testing.assert_allclose(tvboptim_data, tvb_data, rtol=0.0, atol=1e-9)