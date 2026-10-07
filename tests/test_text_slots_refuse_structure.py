"""A text slot refuses a structured value instead of storing its ``str()``.

The generated dataclasses coerced a non-``str`` value in a text slot with ``str()``, so a Figure spec that wrote ``xlabel: {value: ...}`` rendered the axis label ``JsonObj(value=...)`` and a recipe that wrote ``source: {BOLD: ...}`` named a state variable nobody declared, with nothing raised on either construction path the CLI uses.
"""

from pathlib import Path

import numpy as np
import pytest
import sympy
from jsonasobj2 import JsonObj

import tvbo.datamodel.schema as schema
from tvbo.datamodel.schema import Dynamics, Equation, Observation, Panel, Parameter, StateVariableName, SystemType


def _panel(xlabel):
    """A cartesian panel whose x-axis label is *xlabel*."""
    return Panel(panel_key="a", kind="cartesian", xlabel=xlabel)


@pytest.mark.parametrize(
    "value",
    [{"value": "time (s)"}, JsonObj(value="time (s)"), ["time", "(s)"], ("time",), {"time"}, np.arange(3)],
    ids=["dict", "JsonObj", "list", "tuple", "set", "array"],
)
def test_a_structured_value_in_a_text_slot_is_refused_by_name(value):
    with pytest.raises(TypeError, match=r"Panel\.xlabel takes text"):
        _panel(value)


def test_a_record_in_a_text_slot_is_refused():
    with pytest.raises(TypeError, match=r"Equation\.rhs takes text, but was given the Parameter"):
        Equation(rhs=Parameter(name="a", value=1.0))


def test_each_item_of_a_multivalued_text_slot_is_checked():
    with pytest.raises(TypeError, match=r"Observation\.source takes text, but was given the dict"):
        Observation(name="bold", source=["x", {"BOLD": 1}])


def test_an_identifier_slot_is_checked_too():
    with pytest.raises(TypeError, match=r"Dynamics\.name takes text"):
        Dynamics(name={"value": "JansenRit"})


@pytest.mark.parametrize(
    ("value", "text"),
    [(3, "3"), (0.5, "0.5"), (True, "True"), (Path("a/b"), "a/b"), (sympy.Symbol("x") + 1, "x + 1"), (np.float64(0.5), "0.5")],
)
def test_a_scalar_keeps_its_str(value, text):
    assert _panel(value).xlabel == text


def test_an_enum_member_keeps_its_text():
    assert Dynamics(name="Tent", system_type=SystemType("discrete")).system_type == "discrete"


def test_text_and_its_subclasses_pass_through_unchanged():
    name = StateVariableName("V")
    assert Observation(name="bold", source=[name]).source[0] is name
    assert _panel("time (s)").xlabel == "time (s)"


def test_the_generated_datamodel_routes_every_text_coercion():
    source = Path(schema.__file__).read_text(encoding="utf-8")
    assert "self.label = str(self.label)" not in source
    assert "else str(v) for v in self." not in source
    assert source.count("_text_value(self, ") > 500
