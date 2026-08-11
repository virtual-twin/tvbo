import pytest

from tvbo.api import ontology_api
from tvbo.ontology.owl import get_models


MODELS = sorted(get_models().keys())


@pytest.mark.parametrize("model", MODELS)
def test_simulation_experiment(model):
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
    # `duration` is the kwarg the default (tvboptim) run path honors; the old
    # `simulation_length` is TVB-backend-only and was silently ignored here, so every model ran the full 1000 ms default and slow models (e.g. Epileptor) blew the CI timeout.
    api.experiment.run(duration=10)
