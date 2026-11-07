"""Test minimal SimulationExperiment assembly without running full simulation."""

from tvbo.classes.dynamics import Dynamics
from tvbo.classes.experiment import SimulationExperiment
from tvbo.ontology import owl as ontology


def test_simulation_experiment_auto_components():
    oc = ontology.get_model("JansenRit")
    dyn = Dynamics.from_ontology(oc)
    exp = SimulationExperiment(dynamics=dyn)
    # Coupling resolution is deferred to configure() (execution boundary)
    exp.configure()

    # Auto-filled components
    assert exp.integration is not None, "Integrator should be auto-created"
    assert exp.coupling is not None, "Coupling should be auto-created"
    assert exp.network is not None, "Network/connectome should be present"
    # Parameter collection should include at least one known parameter
    param_collection = exp.parameters
    assert any(k for k in param_collection.keys()), "Parameter collection must not be empty"
