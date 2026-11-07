"""A slot assigned after construction holds what the same value gives at construction (`tvbo.datamodel.dialect.fold_slot`)."""

import pytest

from tvbo import datamodel as dm

SPELLINGS = {
    "keyed bare scalars": {"V": 0.1, "W": 0.2},
    "keyed mappings naming themselves": {"V": {"name": "V", "value": 0.1}, "W": {"name": "W", "value": 0.2}},
    "keyed mappings named by their key": {"V": {"value": 0.1}, "W": {"value": 0.2}},
    "list of named mappings": [{"name": "V", "value": 0.1}, {"name": "W", "value": 0.2}],
}


@pytest.mark.parametrize("slot", ["state", "parameters"])
@pytest.mark.parametrize("spelling", SPELLINGS.values(), ids=SPELLINGS)
def test_an_assigned_node_collection_equals_the_constructed_one(slot, spelling):
    node = dm.Node(id=0)
    setattr(node, slot, spelling)
    built = getattr(node, slot)
    assert built == getattr(dm.Node(id=0, **{slot: spelling}), slot)
    assert {key: (member.name, member.value) for key, member in built.items()} == {"V": ("V", 0.1), "W": ("W", 0.2)}


def test_an_assigned_event_is_named_by_its_key():
    node = dm.Node(id=0)
    node.events = {"spike": {}}
    assert isinstance(node.events["spike"], dm.Event)
    assert node.events["spike"].name == "spike"


def test_a_scalar_shortcut_and_a_unit_fold_on_assignment():
    parameter = dm.Parameter(name="a")
    parameter.equation = "x + 2"
    parameter.unit = "millivolt"
    assert parameter.equation.rhs == "x + 2"
    assert parameter.unit == dm.Parameter(name="a", unit="millivolt").unit == "mV"


def test_a_bare_list_member_is_wrapped_on_assignment():
    dynamics = dm.Dynamics(name="d", state_variables={"x": {}})
    dynamics.output = "x"
    assert dynamics.output == ["x"]


def test_an_assignment_is_recorded_as_written():
    node = dm.Node(id=0, state={"V": 0.1})
    node.state = None
    assert node.state is None
    assert "state" in node.model_fields_set
    node.parameters = {}
    assert node.parameters == {}
    assert "parameters" in node.model_fields_set


def test_an_assignment_leaves_the_other_slots_as_they_were():
    coupling = dm.Coupling(iri="tvbo:Linear")
    removed = next(iter(coupling.parameters))
    del coupling.parameters[removed]
    coupling.description = "d"
    assert removed not in coupling.parameters
    assert all(isinstance(parameter, dm.Parameter) for parameter in coupling.parameters.values())


def test_an_assignment_is_checked_against_the_record():
    dynamics = dm.Dynamics(name="d", state_variables={"x": {}})
    with pytest.raises(ValueError, match="not found"):
        dynamics.output = "y"
