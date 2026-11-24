"""A network that declares several couplings keys their parameters by role; the JAX backend hands the coupling a model reads its own parameters, not the whole keyed collection."""

import numpy as np
import pytest

pytest.importorskip("jax")

from tvbo import SimulationExperiment


@pytest.mark.parametrize(
    "coupling",
    [
        {"Linear": {"name": "Linear", "iri": "tvbo:coupling_functions/Linear"}},
        {"c_glob": {"iri": "tvbo:coupling_functions/Linear"}},
    ],
)
def test_a_model_reads_its_coupling_on_a_network_with_several(coupling):
    exp = SimulationExperiment(
        dynamics={"name": "ReducedWongWangExcInh", "iri": "tvbo:models/ReducedWongWangExcInh"},
        network={"iri": "network:example_3node_network", "coupling": coupling},
        integration={"method": "Heun", "duration": 20, "noise": None},
    )
    assert len(exp.network.coupling) > 1
    result = exp.run("jax")
    assert np.isfinite(np.asarray(result.data)).all()
