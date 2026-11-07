"""A parameter read from another run's result is reported by its source, never by the placeholder value beside it.

The run replaces a sourced parameter's declared `value` with the other run's output, so a Methods table that printed the value would state a number no run integrated, and a variant that differs from its base only in where a parameter comes from would not be reported as a variant at all.
"""

from tvbo.datamodel.schema import DataRef, Parameter
from tvbo.utils.report import _param_signature, _value_text


def _sourced():
    return Parameter(name="a", value=-0.5, used=DataRef(experiment=6, output="optimization__refine__fitted__dynamics__a"))


def test_a_sourced_parameter_is_reported_by_its_source():
    assert _value_text(_sourced()) == "from exp 6"


def test_a_sweep_still_replaces_the_source():
    assert _value_text(_sourced(), swept="[0, 1], n=3") == "[0, 1], n=3"


def test_a_set_parameter_is_reported_by_its_value():
    assert _value_text(Parameter(name="a", value=-0.5)) == "-0.5"


def test_the_source_makes_a_variant():
    assert _param_signature(_sourced()) != _param_signature(Parameter(name="a", value=-0.5))
