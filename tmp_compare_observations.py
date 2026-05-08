import numpy as np

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment
from tvboptim.experimental.network_dynamics.result import NativeSolution


network = Network.from_matrix(
    weights=np.array([[0.0, 0.2], [0.1, 0.0]]),
    lengths=np.zeros((2, 2)),
    labels=["left", "right"],
)
network.coupling["Linear"] = Coupling.from_db("Linear")
observations = [
    Observation.from_db("Raw"),
    Observation.from_db("TemporalAverage"),
    Observation.from_db("Bold_TVB"),
]
experiment = SimulationExperiment(
    dynamics=Dynamics.from_db("ReducedWongWang"),
    network=network,
    integration={"method": "Heun", "duration": 1000.0, "step_size": 0.1},
    observations=observations,
)
tvb_result = experiment.run("tvb")
raw = tvb_result.integration.data
solution = NativeSolution(ts=np.asarray(raw.coords["time"]), ys=np.asarray(raw)[..., 0], dt=0.1)
namespace = experiment.execute("tvboptim")

for class_name, tvb_key in [("Temporalaverage", "temporalaverage"), ("BoldTvb", "bold_tvb")]:
    generated = np.asarray(getattr(namespace, class_name)()(solution).data)
    reference = np.asarray(tvb_result.integration.observations[tvb_key].data)[..., 0]
    print(class_name, generated.shape, reference.shape, np.max(np.abs(generated - reference)))
    print(np.array_equal(generated, reference))