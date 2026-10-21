"""An equation step's axes follow its expression by name, and an axis argument may be a name.

A derived observation that reshapes its input (``sum_axis``, ``transpose``, ``matmul``, ``welch``) used to be treated as elementwise, so its result inherited the source's axis names positionally and ``(variable, node)`` binding broke the moment a step dropped an axis. :mod:`tvbo.codegen.dims` follows the axes through the expression tree instead, the primitives taking an axis accept the axis's NAME where the operand's axes are known, and the tvboptim backend declares the derived observation's axes from that. Where the axes cannot be followed the observation stays undeclared, for its author to say ``dims:``; a name no axis carries is refused at codegen.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy as sp

from tvbo.codegen import render_expression
from tvbo.codegen.dims import AXIS_ARGUMENTS, axis_symbols, expression_dims, resolve_axis
from tvbo.parse.expression import ARRAY_FUNCTIONS, parse_eq
from tvbo.templates.tvboptim.utils import equation_dims, observation_dims, resolve_step_expression

TVN = ("time", "variable", "node")


def _expr(rhs, *names):
    return parse_eq(rhs, parameters=list(names) or ["x"])


@pytest.mark.parametrize(
    "rhs,expected",
    [
        ("x", TVN),
        ("x ** 2 + 1", TVN),
        ("x * c", TVN),
        ("sum_axis(x, node)", ("time", "variable")),
        ("sum_axis(x, 2)", ("time", "variable")),
        ("sum_axis(x, -1)", ("time", "variable")),
        ("sum_axis(x, time)", ("variable", "node")),
        ("transpose(sum_axis(x, time))", ("node", "variable")),
        ("matmul(transpose(sum_axis(x, variable)), sum_axis(x, variable))", ("node", "node")),
        ("outer(sum_axis(sum_axis(x, time), variable), sum_axis(sum_axis(x, time), variable))", ("node", "node")),
        ("rankdata(x, node)", TVN),
        ("slice_axis(x, variable, 0, 1)", TVN),
        ("welch(x, 100, 1.0)", ("frequency", "variable", "node")),
        ("rfftfreq(100, 1.0)", ("frequency",)),
        ("pearson(x, x)", ()),
        ("shape(x, node)", ()),
        ("x[0]", ("variable", "node")),
        ("x[0, 1]", ("node",)),
        ("concatenate(x, x, 0)", TVN),
        ("sum_axis(x, 0) + sum_axis(x, time)", ("variable", "node")),
    ],
)
def test_the_axes_of_an_expression_follow_its_operations(rhs, expected):
    """Every rule in the table, from a (time, variable, node) input; a scalar constant broadcasts without adding an axis."""
    env = {"x": TVN, "c": ()}
    assert expression_dims(_expr(rhs, "x", "c"), env) == expected


@pytest.mark.parametrize(
    "rhs",
    ["upper_triangle(x, 1)", "diag(x)", "hann(10)", "x + y", "sum_axis(x, 0) + sum_axis(x, 1)", "f(x)"],
)
def test_axes_nothing_can_name_are_unknown_not_guessed(rhs):
    """A primitive without a rule, an operand with unknown axes, or two operands whose axes disagree leave the value unknown, never a positional guess."""
    assert expression_dims(_expr(rhs, "x"), {"x": TVN}) is None


def test_every_axis_taking_primitive_has_a_rule_or_is_unknown():
    """The table must cover the vocabulary: a primitive that takes an axis either follows a rule or is explicitly unknown, so adding one to the parser cannot silently fall into the elementwise default."""
    for name in AXIS_ARGUMENTS:
        assert name in ARRAY_FUNCTIONS, name
        value = expression_dims(sp.Function(name)(sp.Symbol("x"), sp.Integer(0)), {"x": TVN})
        assert value is None or isinstance(value, tuple)


def test_an_axis_name_resolves_to_its_position_and_a_literal_stays():
    assert resolve_axis(sp.Integer(1), TVN, "sum_axis(x, axis)") == 1
    assert resolve_axis(sp.Integer(-1), None, "sum_axis(x, axis)") == -1
    assert resolve_axis(sp.Symbol("node"), TVN, "sum_axis(x, axis)") == 2


def test_a_name_that_is_not_an_axis_is_refused_with_the_axes_listed():
    with pytest.raises(ValueError, match=r"'mode' is not an axis of the operand; its axes are \['time', 'variable', 'node'\]"):
        resolve_axis(sp.Symbol("mode"), TVN, "sum_axis(x, axis)")


def test_a_name_on_an_operand_of_unknown_axes_names_the_way_out():
    with pytest.raises(ValueError, match="declare `dims:` on the source"):
        resolve_axis(sp.Symbol("node"), None, "sum_axis(x, axis)")


def test_a_symbolic_non_name_axis_is_refused():
    with pytest.raises(ValueError, match="integer literal or an axis name"):
        resolve_axis(sp.Symbol("k") + 1, TVN, "sum_axis(x, axis)")


def test_axis_symbols_are_the_names_in_axis_positions_only():
    assert axis_symbols(_expr("sum_axis(x, node) + rankdata(x, time) * c", "x", "c")) == {"node", "time"}
    assert axis_symbols(_expr("sum_axis(x, 1)", "x")) == set()


@pytest.mark.parametrize("fmt,mod", [("jax", "jnp"), ("numpy", "np")])
def test_a_named_axis_renders_as_its_position(fmt, mod):
    expr = parse_eq("sum_axis(x, node)", parameters=["x"])
    assert render_expression(expr, format=fmt, dims={"x": TVN}) == f"{mod}.sum(x, axis=2)"


def test_a_named_axis_follows_through_a_nested_operand():
    """The operand's axes are those of the sub-expression, not the source: after summing over time, `variable` is axis 0."""
    expr = parse_eq("rankdata(sum_axis(x, time), variable)", parameters=["x"])
    assert "axis=0" in render_expression(expr, format="jax", dims={"x": TVN})


def test_rendering_a_name_without_known_axes_is_refused():
    expr = parse_eq("sum_axis(x, node)", parameters=["x"])
    with pytest.raises(ValueError, match="not known here"):
        render_expression(expr, format="jax")


def test_a_diagonal_offset_takes_only_a_literal():
    """`upper_triangle`'s k is an offset, not an axis, so no name can stand for it."""
    with pytest.raises(ValueError, match="k must be an integer literal"):
        render_expression(parse_eq("upper_triangle(x, node)", parameters=["x"]), format="jax", dims={"x": TVN})


def test_a_step_expression_does_not_mistake_an_axis_name_for_a_free_variable():
    """`resolve_step_expression` raises on an unresolved symbol; an axis name is not one."""
    expr = resolve_step_expression("sum_axis(data, node)", "_d", {}, {}, {"dt": 0.1, "period": 0.1}, "obs")
    assert str(expr) == "sum_axis(_d, node)"
    with pytest.raises(ValueError, match=r"\['c'\] left unresolved"):
        resolve_step_expression("c * data", "_d", {}, {}, {"dt": 0.1, "period": 0.1}, "obs")


def test_equation_dims_reads_the_input_under_a_generic_or_the_source_name():
    assert equation_dims("sum_axis(data, node)", TVN) == ("time", "variable")
    assert equation_dims("sum_axis(x_tail, node)", TVN, ["x_tail"]) == ("time", "variable")
    assert equation_dims("x_tail * 2", TVN, ["x_tail"]) == TVN
    assert equation_dims("sum_axis(x, 0)", None) is None


def _observation(name, **kwargs):
    return SimpleNamespace(
        **{
            "name": name,
            "reduce": None,
            "pipeline": None,
            "source": None,
            "dims": None,
            "aggregation": None,
            "dynamics": None,
            "class_reference": None,
            **kwargs,
        }
    )


def _equation_step(rhs):
    return SimpleNamespace(equation=SimpleNamespace(rhs=rhs), callable=None, function=None, name="step")


def _experiment(**observations):
    return SimpleNamespace(observations=observations, network=SimpleNamespace(node_labels=["A", "B"]))


def test_a_derived_observation_declares_the_axes_its_equation_leaves():
    """The defect this fixes: a reducing step no longer inherits (time, variable, node) positionally."""
    exp = _experiment(
        x_tail=_observation("x_tail", source=["x"]),
        x_summed=_observation("x_summed", source=["x_tail"], pipeline=[_equation_step("sum_axis(x_tail, node)")]),
        x_t=_observation("x_t", source=["x_tail"], pipeline=[_equation_step("transpose(sum_axis(x_tail, variable))")]),
        x_sq=_observation("x_sq", source=["x_tail"], pipeline=[_equation_step("x_tail ** 2")]),
    )
    dims = observation_dims(exp)
    assert dims["x_summed"] == ("time", "variable")
    assert dims["x_t"] == ("node", "time")
    assert dims["x_sq"] == TVN


def test_a_plain_source_lends_its_trajectory_axes_without_being_declared():
    """The plain observation stays positional in the container; only what is derived from it is declared."""
    exp = _experiment(
        x_tail=_observation("x_tail", source=["x"]),
        x_sq=_observation("x_sq", source=["x_tail"], pipeline=[_equation_step("x_tail ** 2")]),
    )
    assert "x_tail" not in observation_dims(exp)


def test_a_step_whose_axes_cannot_be_followed_leaves_the_observation_undeclared():
    exp = _experiment(
        x_tail=_observation("x_tail", source=["x"]),
        tri=_observation("tri", source=["x_tail"], pipeline=[_equation_step("upper_triangle(x_tail[0], 1)")]),
    )
    assert "tri" not in observation_dims(exp)


def test_a_wrong_axis_name_is_refused_naming_the_observation_and_step():
    exp = _experiment(
        x_tail=_observation("x_tail", source=["x"]),
        bad=_observation("bad", source=["x_tail"], pipeline=[_equation_step("sum_axis(x_tail, mode)")]),
    )
    with pytest.raises(ValueError, match=r"Observation 'bad' step 'step': sum_axis\(x, axis\): axis 'mode' is not an axis"):
        observation_dims(exp)


def test_a_monitor_pipeline_over_the_trajectory_is_followed_from_its_axes():
    """A pipeline on a plain source reads the (time, variable, node) window itself, so its equation steps are declared exactly as a derived observation's are."""
    exp = _experiment(
        x_mon_t=_observation("x_mon_t", source=["x"], pipeline=[_equation_step("transpose(sum_axis(data, variable))")]),
        x_mon_fc=_observation(
            "x_mon_fc",
            source=["x"],
            pipeline=[SimpleNamespace(equation=None, callable=SimpleNamespace(name="compute_fc"), function=None, name="fc")],
        ),
        x_mon_opaque=_observation(
            "x_mon_opaque",
            source=["x"],
            pipeline=[SimpleNamespace(equation=None, callable=SimpleNamespace(name="mystery"), function=None, name="m")],
        ),
    )
    dims = observation_dims(exp)
    assert dims["x_mon_t"] == ("node", "time")
    assert dims["x_mon_fc"] == ("node", "node_j")
    assert "x_mon_opaque" not in dims


def test_a_scalar_operand_reports_no_axes_rather_than_unknown_ones():
    with pytest.raises(ValueError, match=r"its axes are \[\]"):
        resolve_axis(sp.Symbol("node"), (), "sum_axis(x, axis)")


SPEC = """
label: Node-summed derived observation
dynamics:
  name: Ramp
  parameters:
    k: {value: 1.0}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 10.0, unit: ms}
observations:
  x_tail:
    source: [x]
    tail_duration: 5.0
  x_summed:
    source: [x_tail]
    pipeline:
      - equation: {rhs: "sum_axis(x_tail, node)"}
  x_t:
    source: [x_tail]
    pipeline:
      - equation: {rhs: "transpose(sum_axis(x_tail, variable))"}
  x_mon_t:
    source: [x]
    pipeline:
      - name: t
        equation: {rhs: "transpose(sum_axis(data, variable))"}
"""


@pytest.mark.backend_tvboptim
def test_a_tvboptim_run_keys_a_reshaped_derived_observation_by_name():
    """End to end: the named axis renders as its position, and the result is selectable by node label after the reshape."""
    pytest.importorskip("tvboptim")
    import numpy as np

    from tvbo import SimulationExperiment

    exp = SimulationExperiment.from_string(SPEC)
    code = exp.render_code("tvboptim")
    assert "obs.x_summed = jnp.sum(x_tail, axis=2)" in code
    assert "obs.x_t = jnp.sum(x_tail, axis=1).T" in code
    assert "_t = jnp.sum(_data, axis=1).T" in code
    res = exp.run("tvboptim")
    x_tail = np.asarray(res.observations["x_tail"].data)
    summed, transposed = res.observations["x_summed"], res.observations["x_t"]
    assert summed.dims == ("time", "variable")
    assert transposed.dims == ("node", "time")
    np.testing.assert_allclose(np.asarray(summed), x_tail.sum(axis=2))
    np.testing.assert_allclose(np.asarray(transposed.sel(node="B")), x_tail[:, 0, 1])
    monitor = res.observations["x_mon_t"].data
    assert monitor.dims == ("node", "time")
    assert monitor.shape[0] == 2
