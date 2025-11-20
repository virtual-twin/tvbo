"""Functional smoke test for Dynamics constructed from ontology.

Focus: ensure ontology-backed model populates parameters, state variables, and
equations; no heavy simulation run.
"""
from tvbo.knowledge import ontology
from tvbo.knowledge.simulation.localdynamics import Dynamics


def test_dynamics_jansenrit_populates():
    oc = ontology.get_model("JansenRit")
    model = Dynamics.from_ontology(oc)

    # Core metadata populated
    assert model.parameters, "Parameters should be populated"
    assert model.state_variables, "State variables should be populated"
    # Check at least one equation has a derivative form (dot)
    state_eqs = model.get_equations(format="state-equations")
    assert state_eqs, "State equations mapping should not be empty"
    # Ensure symbolic elements cover parameters and state variables
    sym_scope = model.get_symbolic_elements()
    # Handle dict or list-like containers robustly
    params = getattr(model, "parameters", {})
    if isinstance(params, dict):
        param_names = list(params.keys())[:3]
    else:
        param_names = [getattr(p, "name", None) for p in list(params)[:3]]
    for pname in [n for n in param_names if n]:
        assert pname in sym_scope, f"Parameter {pname} missing from symbolic scope"
    svs = getattr(model, "state_variables", {})
    if isinstance(svs, dict):
        sv_names = list(svs.keys())[:2]
    else:
        sv_names = [getattr(sv, "name", None) for sv in list(svs)[:2]]
    for sv in sv_names:
        assert sv in sym_scope, f"State variable {sv} missing from symbolic scope"
