import unittest
import warnings
from tqdm import tqdm
from itertools import product
from tvbo.api.ontology_api import OntologyAPI
from tvbo.knowledge.simulation import integration, localdynamics, network
import sys
import os


class TestSimulationConfiguration(unittest.TestCase):

    def test_simulation_experiment(self):
        api = OntologyAPI()

        metadata = {
            "model": {"label": "Generic2dOscillator"},
            "connectivity": {
                "parcellation": {"atlas": {"name": "Destrieux"}},
                "tractogram": {"label": "PPMI85"},
            },
            "coupling": {"label": "SigmoidalJansenRit"},
            "integration": {"method": "Heun", "noise": None},
        }

        # Prepare noise configurations
        noise_configs = [
            None,
            {"correlated": False, "additive": True},
            {"correlated": True, "additive": False},
        ]

        # Suppress all warnings
        warnings.filterwarnings("ignore")

        # Redirect stdout and stderr to null to suppress output
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull

            try:
                # Create a single loop with all combinations
                combinations = product(
                    localdynamics.available_neural_mass_models,
                    integration.available_integrators,
                    network.available_coupling_functions,
                    noise_configs,
                )

                # Loop through the combinations with tqdm progress bar
                for NM, INT, CF, NOISE in tqdm(
                    combinations,
                    desc="Running simulations",
                ):
                    if CF.name == "SigmoidalJansenRit":
                        continue

                    # Update metadata for each combination
                    metadata["model"]["label"] = NM
                    metadata["integration"]["method"] = INT
                    metadata["coupling"]["label"] = CF
                    metadata["integration"]["noise"] = NOISE

                    # Ensure the configuration step works without raising exceptions
                    try:
                        api.configure_simulation_experiment(metadata)
                        api.experiment.run(simulation_length=2)
                    except Exception as e:
                        self.fail(f"Simulation failed with exception: {e}")

            finally:
                # Restore stdout and stderr after test execution
                sys.stdout = original_stdout
                sys.stderr = original_stderr


if __name__ == "__main__":
    unittest.main()
