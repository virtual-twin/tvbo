"""A pipeline argument takes a value, never data bound with ``used:``; one bound to data is refused at codegen by observation, step and argument, rather than leaving the observation out of the run without a word."""

import numpy as np
import pytest

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment
from tvbo.datamodel import Argument, DataRef

pytestmark = pytest.mark.backend_tvboptim


def test_a_data_bound_pipeline_argument_is_refused_by_name():
    observation = Observation.from_db("BOLD_HRF_strided")
    observation.source = ["S"]
    step = next(s for s in observation.pipeline if str(s.function) == "strided_hrf")
    step.arguments["kernel"] = Argument(name="kernel", used=DataRef(experiment=34, output="observation__hrf"))
    network = Network.from_matrix(weights=np.array([[0.0, 0.2], [0.1, 0.0]]), lengths=np.zeros((2, 2)))
    network.coupling["Linear"] = Coupling.from_db("Linear")
    experiment = SimulationExperiment(
        dynamics=Dynamics.from_db("ReducedWongWang"),
        network=network,
        integration={"method": "Heun", "duration": 1000.0, "step_size": 4.0},
        observations=[observation],
    )
    with pytest.raises(
        ValueError, match=r"Observation 'BOLD_HRF_strided', pipeline step 'strided_hrf': argument 'kernel' is bound to data"
    ):
        experiment.render_code("tvboptim")
