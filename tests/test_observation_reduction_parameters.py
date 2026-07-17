"""An observation reduction observer binds its ``Dynamics.parameters`` as named constants.

A model ``Dynamics`` declares its constants as ``parameters`` (scalar, or array-valued for
an operator such as a mode-coupling matrix). An observer ``Dynamics`` — the co-integrated
auxiliary that computes an observation as a time recurrence — is the same class and so
declares constants the same way. These tests pin that the reduction resolver honours them:
they enter the symbolic vocabulary, survive into the rendered triple, and cannot silently
shadow a state variable.
"""

import pytest
from mako.template import Template

from tvbo.datamodel.schema import (
    DerivedVariable,
    Dynamics,
    Equation,
    Observation,
    Parameter,
    StateVariable,
)
from tvbo.templates.tvboptim.utils import resolve_reduction


def _observer(state_rhs, output_rhs, parameters=None):
    """An observation whose value is a recurrence over source ``x``."""
    return Observation(
        name="obs",
        source=["x"],
        dynamics=Dynamics(
            name="observer",
            parameters=parameters or {},
            state_variables={
                "acc": StateVariable(
                    name="acc",
                    equation=Equation(rhs=state_rhs),
                    equation_type="recurrence",
                )
            },
            derived_variables={"out": DerivedVariable(name="out", equation=Equation(rhs=output_rhs))},
            output=["out"],
        ),
    )


def test_scalar_parameter_is_in_the_symbolic_vocabulary():
    obs = _observer("acc + g * x", "acc / count", {"g": Parameter(name="g", value=0.5)})
    red = resolve_reduction(obs)

    assert red["parameters"]["g"]["value"] == 0.5
    # The constant survives as a free symbol rather than being folded away or rejected.
    assert "g" in {str(s) for s in red["states"][0]["update"].free_symbols}


def test_array_valued_parameter_is_kept_as_an_array_not_coerced_to_scalar():
    """An operator constant (here a matrix) must reach codegen intact."""
    op = [[1.0, 2.0], [3.0, 4.0]]
    obs = _observer("acc + matmul(A, x)", "acc / count", {"A": Parameter(name="A", value=op)})
    red = resolve_reduction(obs)

    assert red["parameters"]["A"]["value"] == op


def test_parameter_shape_is_carried_through():
    obs = _observer(
        "acc + matmul(A, x)",
        "acc / count",
        {"A": Parameter(name="A", value=[[1.0, 2.0], [3.0, 4.0]], shape="(2, 2)")},
    )
    red = resolve_reduction(obs)

    assert red["parameters"]["A"]["shape"] == "(2, 2)"


def test_an_undeclared_symbol_is_still_rejected():
    """The vocabulary widened to include parameters — it did not become permissive."""
    obs = _observer("acc + typo * x", "acc / count", {"g": Parameter(name="g", value=0.5)})

    with pytest.raises(ValueError, match="typo"):
        resolve_reduction(obs)


def test_a_parameter_may_not_shadow_a_state_variable():
    """One symbol cannot denote both a per-step state and a fixed constant."""
    obs = _observer("acc + x", "acc / count", {"acc": Parameter(name="acc", value=1.0)})

    with pytest.raises(ValueError, match="both a state variable and a parameter"):
        resolve_reduction(obs)


def test_an_observer_without_parameters_resolves_unchanged():
    """Regression: the shipped parameterless observers (e.g. effective frequency)."""
    obs = _observer("x", "(x - acc) / dt", None)
    red = resolve_reduction(obs)

    assert red["parameters"] == {}
    assert red["source"] == "x"


# --- emission: the rendered init/update/finalize triple ------------------------------

_TEMPLATE = "tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako"


def _render(obs):
    red = resolve_reduction(obs)
    tmpl = Template(filename=_TEMPLATE).get_def("render_reduction")
    return tmpl.render(red=red, name="obs", s_idx=0, dt=0.1)


def test_constants_are_bound_in_the_rendered_triple():
    """Both kinds of constant bind by name in the closure the triple shares, and an
    array operator reaches the backend as an array rather than a Python list."""
    obs = _observer(
        "acc + g * matmul(A, x)",
        "acc / count",
        {
            "A": Parameter(name="A", value=[[1.0, 2.0], [3.0, 4.0]]),
            "g": Parameter(name="g", value=0.5),
        },
    )
    code = _render(obs)

    assert "A = jnp.array([[1.0, 2.0], [3.0, 4.0]])" in code
    assert "g = 0.5" in code
    # The array-op handler lowers matmul through the reduction path. Operands are
    # parenthesized so a compound operand (e.g. matmul(A, x/c)) keeps `@` binding
    # tighter than `/` -> `A @ (x/c)`, not `(A @ x)/c`.
    assert "(A) @ (x)" in code


def test_a_parameterless_observer_emits_no_constants_block():
    """The shipped observers' output is unchanged by the parameter binding."""
    code = _render(_observer("x", "(x - acc) / dt", None))

    assert "Observer constants" not in code


# --- lazy constants: sourced/produced operators emit a load, never the bytes ---------

class _Experiment:
    """Minimal render context: only what resolve_reduction reads for materialisation."""

    def __init__(self, source_file, network=None):
        self._source_file = str(source_file)
        self.network = network


def test_a_sourced_operator_emits_a_lazy_load_not_its_bytes(tmp_path):
    """A large operator must never enter the generated source — the reduction reads it
    from the companion at run time instead."""
    pytest.importorskip("h5py")
    import h5py

    with h5py.File(tmp_path / "ops.h5", "w") as f:
        f.create_dataset("grad_op", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    obs = _observer(
        "acc + matmul(A, x)",
        "acc / count",
        {"A": Parameter(name="A", source="ops.h5", measure="grad_op")},
    )
    red = resolve_reduction(obs, _Experiment(tmp_path / "study.yaml"))
    code = Template(filename=_TEMPLATE).get_def("render_reduction").render(
        red=red, name="obs", s_idx=0, dt=0.1
    )

    assert "_load_constant(" in code
    assert "grad_op" in code
    # the array itself is nowhere in the source
    assert "1.0, 2.0, 3.0" not in code and "jnp.array([[" not in code
    assert red["parameters"]["A"]["lazy"][1] == "grad_op"


def test_the_existence_check_does_not_materialise(tmp_path, monkeypatch):
    """resolve_reduction is also called bare as an 'is this a streaming reducer?' predicate;
    that must not run the producer or write to disk as a side effect."""
    import tvbo.data.param_io as param_io

    called = []
    monkeypatch.setattr(param_io, "materialise", lambda *a, **k: called.append(1))

    obs = _observer(
        "acc + matmul(A, x)",
        "acc / count",
        {"A": Parameter(name="A", source="ops.h5", measure="grad_op")},
    )
    red = resolve_reduction(obs)  # bare: the predicate path

    assert red is not None                       # still recognised as a reducer
    assert called == []                          # but nothing materialised
    assert red["parameters"]["A"]["lazy"] is None  # deferred, not resolved


def test_the_emitted_path_is_absolute(tmp_path):
    """Generated modules are exec'd in memory, so a relative path has nothing to anchor
    to — codegen resolves it against the spec dir, exactly as bids_dir is emitted."""
    pytest.importorskip("h5py")
    import h5py

    with h5py.File(tmp_path / "ops.h5", "w") as f:
        f.create_dataset("grad_op", data=[[1.0, 2.0]])

    obs = _observer(
        "acc + matmul(A, x)",
        "acc / count",
        {"A": Parameter(name="A", source="ops.h5", measure="grad_op")},
    )
    red = resolve_reduction(obs, _Experiment(tmp_path / "study.yaml"))

    from pathlib import Path

    assert Path(red["parameters"]["A"]["lazy"][0]).is_absolute()
