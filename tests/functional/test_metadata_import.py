"""Test importing a minimal Dynamics into the ontology via metadata.import_yaml_model."""
from tvbo.datamodel import tvbo_datamodel as dm
from tvbo.parse import metadata
from tvbo.knowledge import ontology


def test_import_minimal_dynamics_into_ontology():
    # Minimal one-state, one-parameter model constructed via kwargs
    sv = dm.StateVariable(
        name="X",
        equation=dm.Equation(lhs="X", rhs="-a*X"),
        domain=dm.Range(lo=-1.0, hi=1.0),
        initial_value=0.1,
    )
    param_a = dm.Parameter(name="a", value=0.5)
    dyn = dm.Dynamics(
        name="ToyModel",
        state_variables={"X": sv},
        parameters={"a": param_a},
    )

    oc = metadata.import_yaml_model(dyn, model_name="ToyModel")
    assert oc is not None

    oc2 = ontology.onto.search_one(label="ToyModel")
    assert oc2 is not None, "Imported class should be found in ontology by label"
