"""A text slot refuses a structured value instead of storing its ``str()``.

The generated dataclasses coerced a non-``str`` value in a text slot with ``str()``, so a Figure spec that wrote ``xlabel: {value: ...}`` rendered the axis label ``JsonObj(value=...)`` and a recipe that wrote ``source: {BOLD: ...}`` named a state variable nobody declared, with nothing raised on either construction path the CLI uses.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sympy
from jsonasobj2 import JsonObj

import tvbo.datamodel.schema as schema
from tvbo.datamodel.schema import (
    Coupling,
    Dynamics,
    Equation,
    Function,
    Observation,
    Panel,
    Parameter,
    StateVariableName,
    SystemType,
)

_spec = importlib.util.spec_from_file_location("hatch_build", Path(__file__).resolve().parents[1] / "hatch_build.py")
hatch_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hatch_build)

UNION = "        for _coerce in (lambda: UnitEnum(value), lambda: str(value)):"


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


def test_the_text_branch_of_a_union_slot_is_checked():
    with pytest.raises(ValueError, match=r"from JsonObj\(value='ms'\) for slot unit"):
        Parameter(name="a", unit={"value": "ms"})


def test_a_mapping_in_a_union_slot_whose_text_branch_comes_first_builds_its_record():
    aggregation = Coupling(name="c", aggregation={"name": "mean"}).aggregation
    assert isinstance(aggregation, Function) and aggregation.name == "mean"
    assert Coupling(name="c", aggregation="mean").aggregation == "mean"


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
    assert "lambda: str(value)" not in source
    assert source.count("_text_value(self, ") > 500


def _generated(*lines: str) -> str:
    """A generated module holding one text coercion of each form LinkML writes, followed by *lines*."""
    return "\n".join(
        [
            '"""Generated."""',
            'metamodel_version = "1.7.0"',
            "class Panel:",
            "    def __post_init__(self):",
            "        self.xlabel = str(self.xlabel)",
            "        self.source = [v if isinstance(v, str) else str(v) for v in self.source]",
            "        value = self.unit",
            UNION,
            *lines,
        ]
    )


@pytest.mark.parametrize("call", ["str(value)", "str(v)", "str(self.name)"])
def test_a_text_call_that_coerces_no_slot_passes_the_build(call):
    """A helper LinkML generates may call ``str()`` on a name a coercion also uses; it coerces no slot, so the build routes the three coercions and keeps the helper as it is."""
    code = hatch_build._with_strict_text(_generated("    def _describe(self, value, v):", f"        return {call}"))
    assert f"        return {call}" in code
    assert 'lambda: str(_text_value(self, "unit", value))' in code


@pytest.mark.parametrize(
    "code",
    [
        _generated("        self.label = str(self.label)  # a comment LinkML now writes"),
        _generated(
            "        self.tags = [v if isinstance(v, str) else str(v) for v in self.tags]  # a comment LinkML now writes"
        ),
        _generated(
            "        value = self.kind",
            "        # a line LinkML now writes between",
            "        for _coerce in (lambda: str(value),):",
        ),
        _generated().replace(UNION, "        self.unit = str(value)"),
    ],
    ids=[
        "a single-valued coercion left unrouted",
        "a multivalued coercion left unrouted",
        "a candidate lambda left unrouted",
        "any_of re-spelt without candidate lambdas",
    ],
)
def test_a_text_coercion_that_escapes_the_check_fails_the_build(code):
    with pytest.raises(RuntimeError, match=r"update hatch_build\._with_strict_text"):
        hatch_build._with_strict_text(code)
