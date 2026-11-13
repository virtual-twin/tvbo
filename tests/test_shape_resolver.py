"""One reader for a declared shape, the sizes it names, and the element axes a per-node equation reads from it.

A shape is a Python tuple of integer literals, ``n_nodes`` and sizes, where a size is a parameter of the same dynamics holding an integer that nothing changes. :mod:`tvbo.codegen.dims` parses it, sizes it, writes it into generated code and types a per-node equation by the element axes it gives each operand, by name, so a readout of shape ``(n_nodes, n)`` is never confused with a matrix of shape ``(n, n)`` even when the network has ``n`` nodes. The parser reads ``a @ b`` as the ``matmul`` the array vocabulary names.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from tvbo import Dynamics
from tvbo.codegen.dims import dynamics_sizes, per_node_dims, resolve_shape, shape_code, shape_dims, shape_terms
from tvbo.parse.expression import parse_eq


@pytest.mark.parametrize(
    "shape, terms",
    [
        ("(n_nodes,)", ("n_nodes",)),
        ("(n_nodes)", ("n_nodes",)),
        ("(n_nodes, n_nodes)", ("n_nodes", "n_nodes")),
        ("(2, 2)", (2, 2)),
        ("(n_nodes, n)", ("n_nodes", "n")),
        ("()", ()),
        (None, None),
        ("", None),
    ],
)
def test_a_shape_is_read_as_its_terms(shape, terms):
    assert shape_terms(shape) == terms


@pytest.mark.parametrize("shape", ["(n_nodes, 2.5)", "(n_nodes, -1)", "(n+1,)", "n_nodes x n", "(True,)"])
def test_a_shape_that_is_not_a_tuple_of_integers_and_names_is_refused(shape):
    with pytest.raises(ValueError, match="shape"):
        shape_terms(shape)


def test_a_shape_is_sized_by_the_node_count_and_the_sizes():
    assert resolve_shape("(n_nodes, n)", owner="W_out", sizes={"n": 5}, n_nodes=68) == (68, 5)
    assert resolve_shape("(n, n)", owner="W", sizes={"n": 4}, n_nodes=4) == (4, 4)
    assert resolve_shape("()", owner="a") == ()
    with pytest.raises(ValueError, match=r"W_out: shape '\(n_nodes, K\)' names 'K', which is neither `n_nodes` nor a size"):
        resolve_shape("(n_nodes, K)", owner="W_out", sizes={"n": 5}, n_nodes=68)
    with pytest.raises(ValueError, match="names `n_nodes`, and no network gives a node count here"):
        resolve_shape("(n_nodes,)", owner="k")


def test_a_shape_is_written_into_code_without_spaces_and_with_each_size_as_its_integer():
    assert shape_code("(n_nodes, n_nodes)") == "(n_nodes,n_nodes)"
    assert shape_code("(n_nodes,)") == "(n_nodes,)"
    assert shape_code("(n_nodes, n)", {"n": 100}) == "(n_nodes,100)"
    assert shape_code("()") == "()"


def test_a_shape_names_the_node_axis_and_one_element_axis_per_term():
    assert shape_dims("(n_nodes, n)") == ("node", "element_n")
    assert shape_dims("(n, n)") == ("element_n", "element_n")
    assert shape_dims("(2,)") == ("element_2",)


def _reservoir(**size):
    """A dynamics whose readout and recurrence are shaped by the size ``n``."""
    return Dynamics(
        name="Reservoir",
        parameters={
            "n": {"value": 5, **size},
            "W": {"value": 0.0, "shape": "(n, n)"},
            "W_out": {"value": 0.0, "shape": "(n_nodes, n)"},
        },
        state_variables={"x": {"equation": {"rhs": "-x"}}},
    )


def test_the_sizes_of_a_dynamics_are_the_parameters_its_shapes_name():
    assert dynamics_sizes(_reservoir()) == {"n": 5}
    assert dynamics_sizes(Dynamics(name="Plain", parameters={"a": {"value": 1.0, "shape": "(n_nodes,)"}})) == {}


@pytest.mark.parametrize(
    "size, why",
    [
        (
            {"distribution": {"name": "Uniform", "parameters": {"lo": {"value": 1}, "hi": {"value": 9}}}},
            "is drawn from a distribution",
        ),
        ({"measure": "n"}, "is sourced"),
        ({"value": 5.5}, r"holds 5.5, not an integer"),
    ],
    ids=["drawn", "sourced", "not-an-integer"],
)
def test_a_size_that_is_not_a_fixed_integer_is_refused_naming_its_reader(size, why):
    with pytest.raises(ValueError, match=rf"size 'n', which the shape of parameter 'W' names, {why}"):
        dynamics_sizes(_reservoir(**size))


@pytest.mark.parametrize(
    "experiment",
    [
        SimpleNamespace(
            optimizations={
                "fit": SimpleNamespace(
                    free_parameters=None, stages=[SimpleNamespace(free_parameters=[SimpleNamespace(parameter="n")])]
                )
            },
            explorations=None,
        ),
        SimpleNamespace(optimizations=None, explorations={"sweep": SimpleNamespace(space={"Reservoir.n": SimpleNamespace()})}),
    ],
    ids=["fitted by fit", "swept by sweep"],
)
def test_a_size_something_acts_on_is_refused_naming_what_acts(request, experiment):
    with pytest.raises(ValueError, match=f"size 'n', which the shape of parameter 'W' names, is {request.node.callspec.id}; "):
        dynamics_sizes(_reservoir(), experiment)


def _dims(rhs, **env):
    return per_node_dims(parse_eq(rhs), env)


N, M = ("element_n",), ("element_m",)


@pytest.mark.parametrize(
    "rhs, env, expected",
    [
        ("W @ x", {"W": N + N, "x": N}, N),
        ("W_out @ x", {"W_out": N, "x": N}, ()),
        ("W_in * c", {"W_in": N, "c": ()}, N),
        ("mean(x) + sum(x) + max(x) + min(x)", {"x": N}, ()),
        ("tanh(v) / tau", {"v": N, "tau": ()}, N),
        ("(1/tau) * (-x + tanh(W @ x + W_in * c))", {"tau": (), "x": N, "W": N + N, "W_in": N, "c": ()}, N),
    ],
)
def test_a_per_node_equation_is_typed_by_the_element_axes_of_its_operands(rhs, env, expected):
    assert _dims(rhs, **env) == expected


def test_a_per_region_readout_and_a_shared_matrix_are_told_apart_by_name_when_their_sizes_agree():
    readout = shape_dims("(n_nodes, n)")[1:]
    recurrence = shape_dims("(n, n)")
    assert resolve_shape("(n_nodes, n)", owner="W_out", sizes={"n": 4}, n_nodes=4) == resolve_shape(
        "(n, n)", owner="W", sizes={"n": 4}
    )
    assert _dims("W_out @ x", W_out=readout, x=N) == ()
    assert _dims("W @ x", W=recurrence, x=N) == N


@pytest.mark.parametrize(
    "rhs, env, message",
    [
        ("x + z", {"x": N, "z": M}, r"x \['element_n'\] and z \['element_m'\] have different element axes"),
        ("W @ z", {"W": N + N, "z": M}, r"W \['element_n', 'element_n'\] @ z \['element_m'\] contracts no shared axis"),
        ("x[0] + 1", {"x": N}, "indexing an element, x\\[0\\], is not read in a per-node equation"),
        ("y + 1", {}, "'y' has no declared axes here"),
    ],
    ids=["different-elements", "matmul-mismatch", "index", "unknown-symbol"],
)
def test_a_per_node_equation_whose_operands_do_not_fit_is_refused_naming_them(rhs, env, message):
    with pytest.raises(ValueError, match=message):
        _dims(rhs, **env)


def test_the_parser_reads_at_as_matmul():
    matmul = sp.Function("matmul")
    W, x, a, b, c = sp.symbols("W x a b c")
    assert parse_eq("W @ x") == matmul(W, x)
    assert parse_eq("a @ b @ c") == matmul(matmul(a, b), c)
    assert parse_eq("W @ x + 1") == matmul(W, x) + 1


SIZED = """
label: A dynamics with a size
dynamics:
  name: Sized
  parameters:
    n: {value: 3}
    k: {value: 1.0}
    W: {value: 0.5, shape: "(n, n)"}
  state_variables:
    x: {equation: {rhs: "-x / n + k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Sized}, {id: 1, label: B, dynamics: Sized}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 10.0, unit: ms}
"""


@pytest.mark.backend_tvboptim
def test_a_size_is_an_integer_constant_of_the_module_and_never_a_traced_parameter():
    pytest.importorskip("tvboptim")
    from tvbo import SimulationExperiment

    experiment = SimulationExperiment.from_string(SIZED)
    code = experiment.render_code("tvboptim")
    default_params = code[code.index("DEFAULT_PARAMS = Bunch(") : code.index("COUPLING_INPUTS")]
    assert "k=" in default_params and "W=" in default_params and "n=" not in default_params
    assert "        n = 3\n" in code and "params.n" not in code
    node_a = np.asarray(experiment.run("tvboptim").integration.data)[-1, 0, 0]
    assert node_a == pytest.approx(3.0 * (1.0 - (2.0 / 3.0) ** 10), rel=1e-6)


SOURCED = SIZED.replace(
    '    W: {value: 0.5, shape: "(n, n)"}\n',
    '    W_out: {value: 0.0, shape: "(n_nodes, n)", used: {experiment: 7, output: W}}\n',
)


@pytest.mark.parametrize("stored, kept", [((2, 3), (2, 3)), ((1, 2, 3), (2, 3))], ids=["as-declared", "leading-axis"])
def test_a_sourced_parameter_is_checked_against_its_sized_shape(monkeypatch, stored, kept):
    import xarray as xr

    from tvbo import SimulationExperiment
    from tvbo.data import dataref

    monkeypatch.setattr(dataref, "resolve_dataref", lambda spec, **kw: xr.DataArray(np.ones(stored)))
    assert SimulationExperiment.from_string(SOURCED)._resolve_from_experiment_params()["W_out"].shape == kept


def test_a_sourced_parameter_of_another_shape_is_refused(monkeypatch):
    import xarray as xr

    from tvbo import SimulationExperiment
    from tvbo.data import dataref

    monkeypatch.setattr(dataref, "resolve_dataref", lambda spec, **kw: xr.DataArray(np.ones((2, 4))))
    with pytest.raises(
        ValueError, match=r"Parameter 'W_out' declares shape \(n_nodes, n\), \(2, 3\), but its `used:` reference"
    ):
        SimulationExperiment.from_string(SOURCED)._resolve_from_experiment_params()
