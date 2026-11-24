"""Every reader of a declared expression reads a comparison as `parse_eq` does.

A comparison that arithmetic applies to, ``(v > 0) * v``, is the 0/1 field it denotes, and ``==`` compares values, in a ``Piecewise`` condition as anywhere else. Each reader below parses against its own vocabulary, so each is checked against `parse_eq`, or against numpy where its output runs, or refuses by name where it cannot take the term.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import tvbo
from tvbo.codegen import render_expression
from tvbo.codegen.pyrates import operator_template
from tvbo.codegen.reducers import resolve_streaming_reducer
from tvbo.codegen.streaming_reducers import StreamingReducerSpec
from tvbo.data.types import stimulus_signals
from tvbo.datamodel import DerivedVariable, Dynamics, Equation, Function, Observation, StateVariable
from tvbo.parse.expression import parse_eq
from tvbo.templates.pde import utils as pde
from tvbo.templates.tvboptim.utils import resolve_reduction


def test_an_equality_in_a_piecewise_condition_keeps_its_branch():
    """`k == level` compares the values; read as a test of the two symbols it would be False, and the branch would vanish."""
    k = np.array([0.0, 1.0, 2.0])
    code = render_expression("Piecewise((a, k == level), (b, True))", format="numpy", parameters=["a", "b", "k", "level"])

    got = eval(code, {"np": np}, {"a": 10.0, "b": 20.0, "k": k, "level": 1.0})

    np.testing.assert_array_equal(np.asarray(got), np.where(k == 1.0, 10.0, 20.0))


def test_a_streaming_reducer_reads_a_comparison_in_arithmetic():
    spec = StreamingReducerSpec(state=("s",), add=(("s", "s + (v > 0) * v"),), evict=(), resync=(), emit="s")
    ((_, src),) = resolve_streaming_reducer(spec, "numpy")["add"]
    v = np.array([-1.0, 2.0])

    got = eval(src, {"np": np}, {"s": 0.0, "v": v})

    np.testing.assert_array_equal(np.asarray(got), (v > 0) * v)


def test_an_observer_function_body_reads_a_comparison_in_arithmetic():
    relu = Function(name="relu", arguments={"u": {"name": "u"}}, equation=Equation(rhs="(u > 0) * u"))
    obs = Observation(
        name="rectified",
        source=["p"],
        dynamics=Dynamics(
            name="rectified",
            functions={"relu": relu},
            state_variables={"acc": StateVariable(name="acc", equation=Equation(rhs="acc + p"), equation_type="recurrence")},
            derived_variables={"r": DerivedVariable(name="r", equation=Equation(rhs="relu(p)"))},
            output=["r"],
        ),
    )

    body = resolve_reduction(obs)["functions"]["relu"]["expr"]

    assert body == parse_eq("(u > 0) * u", parameters=["u"])


def test_a_field_equation_keeps_a_comparison_in_its_explicit_remainder():
    """The indicator of a field is no coefficient the implicit blocks can assemble, so the term is evaluated explicitly each step."""
    rhs = pde._sympify("(x0 > 0) * u", ["u"], {"x0"})

    blocks, explicit = pde._linear_blocks(rhs, 0, ["u"], {}, {"x0"})

    assert blocks == []
    assert sp.Add(*explicit) == parse_eq("(x0 > 0) * u", parameters=["x0", "u"])


def test_a_field_equation_refuses_a_comparison_on_a_differential_operator_by_name():
    rhs = pde._sympify("(x0 > 0) * laplacian(u)", ["u"], {"x0"})

    with pytest.raises(pde.FieldPlanError, match=r"still applies \['laplacian'\]"):
        pde._linear_blocks(rhs, 0, ["u"], {}, {"x0"})


def test_brian2_writes_a_comparison_in_arithmetic_as_int():
    """Brian2's equations have no conditional expression; `int(condition)` is how they turn a comparison into 0 or 1."""
    from tvbo.adapters import brian2

    expr = brian2._parse("(v > thresh) * g", brian2._symbols(["thresh", "g"]))
    code = render_expression(str(expr), format="brian2")

    assert "int(" in code and " if " not in code
    assert [eval(code, {}, {"v": v, "thresh": 0.5, "g": 2.0}) for v in (1.0, 0.0)] == [2.0, 0.0]


def test_a_stimulus_waveform_reads_a_comparison_in_arithmetic():
    """A pulse written as the product of its window's two comparisons is the amplitude inside the window and 0 outside it."""
    pulse = SimpleNamespace(
        name="I",
        event_type="stimulus",
        equation=SimpleNamespace(rhs="(t >= on) * (t < off) * amp"),
        parameters={"on": SimpleNamespace(value=1.0), "off": SimpleNamespace(value=2.0), "amp": SimpleNamespace(value=3.0)},
        dataLocation=None,
    )
    t = np.arange(0.0, 3.0, 0.5)

    signal = stimulus_signals(SimpleNamespace(events={"I": pulse}), t)["event__I"]

    np.testing.assert_array_equal(signal.values, np.where((t >= 1.0) & (t < 2.0), 3.0, 0.0))


def test_the_pyrates_codegen_reads_a_comparison_in_arithmetic():
    """PyRates has no conditional, so the operator carries the 0/1 indicator as `sign` arithmetic, which takes the declared equation's values."""
    model = tvbo.Dynamics.from_string(
        "name: Gate\n"
        "parameters: {thresh: {value: 0.5}, g: {value: 2.0}}\n"
        "state_variables:\n"
        "  v: {equation: {rhs: '-v + (v > thresh) * g'}, initial_value: 0.0}"
    )

    (equation,) = operator_template(model)["equations"]

    assert equation.startswith("v' = ") and "Piecewise" not in equation
    rhs = sp.sympify(equation.removeprefix("v' = "))
    declared = parse_eq("-v + (v > thresh) * g", parameters=["v", "thresh", "g"])
    for v in (0.0, 1.0):
        values = {"v": v, "thresh": 0.5, "g": 2.0}
        assert float(rhs.subs(values)) == float(declared.subs(values))
