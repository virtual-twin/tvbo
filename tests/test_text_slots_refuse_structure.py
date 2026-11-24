"""A text slot refuses a structured value instead of storing its ``str()``, and reads a number or a path as its text.

A Figure spec that wrote ``xlabel: {value: ...}`` once rendered the axis label ``JsonObj(value=...)``, and a recipe that wrote ``source: {BOLD: ...}`` named a state variable nobody declared. The generated models refuse both, naming the slot, on construction and on assignment.
"""

from pathlib import Path

import numpy as np
import pytest
import sympy
from pydantic import ValidationError

from tvbo.datamodel import Dataset, Dynamics, Equation, Exploration, Observation, Panel, Parameter, SystemType


def _panel(xlabel):
    """A cartesian panel whose x-axis label is *xlabel*."""
    return Panel(panel_key="a", kind="cartesian", xlabel=xlabel)


@pytest.mark.parametrize(
    "value",
    [{"value": "time (s)"}, ["time", "(s)"], ("time",), {"time"}, np.arange(3), True, sympy.Symbol("t") + 1],
    ids=["dict", "list", "tuple", "set", "array", "bool", "sympy"],
)
def test_a_value_that_is_not_text_is_refused_by_name(value):
    with pytest.raises(ValidationError, match=r"xlabel\n  Input should be a valid string"):
        _panel(value)
    panel = _panel("time (s)")
    with pytest.raises(ValidationError, match=r"xlabel\n  Input should be a valid string"):
        panel.xlabel = value
    assert panel.xlabel == "time (s)"


def test_a_record_in_a_text_slot_is_refused():
    with pytest.raises(ValidationError, match=r"rhs\n  Input should be a valid string"):
        Equation(rhs=Parameter(name="a", value=1.0))


def test_each_item_of_a_multivalued_text_slot_is_checked():
    with pytest.raises(ValidationError, match=r"source\.1\n  Input should be a valid string"):
        Observation(name="bold", source=["x", {"BOLD": 1}])


def test_the_text_branch_of_a_union_slot_is_checked():
    with pytest.raises(ValidationError, match=r"unit"):
        Parameter(name="a", unit={"value": "ms"})
    parameter = Parameter(name="a", unit="ms")
    with pytest.raises(ValidationError, match=r"unit"):
        parameter.unit = {"value": "ms"}


def test_a_union_slot_keeps_the_branch_a_value_matches_and_refuses_a_mapping_none_takes():
    assert Exploration(name="s", n_parallel="auto").n_parallel == "auto"
    assert Exploration(name="s", n_parallel=4).n_parallel == 4
    with pytest.raises(ValidationError, match=r"n_parallel"):
        Exploration(name="s", n_parallel={"value": 4})


def test_an_identifier_slot_is_checked_too():
    with pytest.raises(ValidationError, match=r"name\n  Input should be a valid string"):
        Dynamics(name={"value": "JansenRit"})


@pytest.mark.parametrize(("value", "text"), [(3, "3"), (0.5, "0.5"), (np.float64(0.5), "0.5"), (Path("a/b"), "a/b")], ids=str)
def test_a_number_or_a_path_is_read_as_its_text(value, text):
    assert _panel(value).xlabel == text


def test_a_path_is_its_text_in_a_list_and_on_assignment():
    dataset = Dataset(dataset_id="d", bids_root=Path("/data/bids"), phenotype_file=Path("scores.tsv"))
    assert (dataset.bids_root, dataset.phenotype_file) == ("/data/bids", "scores.tsv")
    dataset.bids_root = Path("/data/other")
    assert dataset.bids_root == "/data/other"
    assert Dynamics(name="m", references=[Path("a.bib"), "b.bib"]).references == ["a.bib", "b.bib"]


def test_an_enum_member_keeps_its_text():
    assert Dynamics(name="Tent", system_type=SystemType("discrete")).system_type == "discrete"
