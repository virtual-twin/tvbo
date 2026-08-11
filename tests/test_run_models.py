import pytest

from tvbo.api import ontology_api
from tvbo.ontology.owl import get_models


MODELS = sorted(get_models().keys())


@pytest.mark.parametrize("model", MODELS)
def test_simulation_experiment(model):
    """Every ontology model configures a SimulationExperiment and runs a short simulation.

    The run length is passed as ``duration``, the kwarg the default (tvboptim) run path honors. ``simulation_length`` is TVB-backend-only and is ignored here, so passing it instead leaves every model running the full 1000 ms default and blows the CI timeout on the slow ones (e.g. Epileptor).
    """
    api = ontology_api.OntologyAPI()

    metadata = {
        "dynamics": model,
        "coupling": "Linear",
        "network": {
            "parcellation": {
                "atlas": {
                    "name": "DesikanKilliany",
                }
            },
            "tractogram": {
                "name": "dTOR",
            },
        },
        "integration": {"method": "Heun", "noise": None},
    }

    api.configure_simulation_experiment(metadata)
    api.experiment.run(duration=10)
