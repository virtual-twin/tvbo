"""Test minimal SimulationExperiment assembly without running full simulation."""
from tvbo.knowledge import ontology
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.export.experiment import SimulationExperiment


def test_simulation_experiment_auto_components():
    oc = ontology.get_model("JansenRit")
    dyn = Dynamics.from_ontology(oc)
    exp = SimulationExperiment(local_dynamics=dyn)

    # Auto-filled components
    assert exp.integration is not None, "Integrator should be auto-created"
    assert exp.coupling is not None, "Coupling should be auto-created"
    assert exp.network is not None, "Network/connectome should be present"
    # Parameter collection should include at least one known parameter
    param_collection = exp.parameters
    assert any(k for k in param_collection.keys()), "Parameter collection must not be empty"
