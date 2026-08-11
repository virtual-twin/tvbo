"""A boundary condition states an equation, and the FEM lowering reads it.

`BoundaryCondition` held its value in a slot called `value` whose range is `Equation` —
the one spelling that already means "a number" on `Parameter`. The FEM template then read
it under `righthandside`, the alias the dialect folds away at construction, so `getattr`
returned the `Equation` object itself, `float()` of it raised, and a bare `except` turned
every Dirichlet boundary into 0.0. Nothing in the emitted code said so.
"""

import pytest
import yaml

from tvbo.classes.experiment import SimulationExperiment
from tvbo.datamodel import tvbo_datamodel as dm


def _experiment(bc_equation, tmp_path):
    """A minimal FEM diffusion experiment whose one Dirichlet BC states *bc_equation*.

    ``None`` omits the slot entirely, which is a boundary that states no value at all.
    """
    condition = {"label": "BC", "bc_type": "Dirichlet"}
    if bc_equation is not None:
        condition["equation"] = bc_equation
    spec = {
        "label": "BC probe",
        "field_dynamics": {
            "label": "Heat",
            "mesh": {"label": "m", "element_type": "triangle"},
            "parameters": {"D": {"name": "D", "value": 1.0}},
            "state_variables": [{
                "name": "u", "label": "u", "initial_value": 0.0,
                "boundary_conditions": [condition],
                "equation": {"lhs": "u_t", "rhs": "D * laplacian(u)"},
            }],
            "operators": [{"label": "Diff", "operator_type": "laplacian", "coefficient": "D"}],
            "solver": {"label": "FEM", "discretization": "FEM",
                       "method": "implicit Euler", "dt": 1.0},
        },
        "integration": {"duration": 10},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump(spec))
    return SimulationExperiment.from_file(str(path))


@pytest.mark.parametrize("written", [
    {"equation": "2.5"},
    {"equation": {"rhs": "2.5"}},
    {"value": "2.5"},
    {"value": {"rhs": "2.5"}},
])
def test_every_spelling_of_the_value_reaches_the_same_equation(written):
    """`equation` is canonical, `value` the older spelling, either bare or as a mapping.

    The bare-scalar forms are the ones that regressed: the dialect lifted shortcuts before
    it folded aliases, so a value written under `value` was still under a name the shortcut
    table — keyed by canonical slot — could not see, and reached `__post_init__` as a str.
    """
    bc = dm.BoundaryCondition(bc_type="Dirichlet", **written)

    assert bc.equation.rhs == "2.5"
    assert float(bc.equation.rhs) == 2.5


def test_a_nonzero_dirichlet_value_survives_into_the_generated_code(tmp_path):
    """The whole point: it used to arrive as 0.0 whatever the recipe said."""
    code = _experiment("2.5", tmp_path).render_code("pde")

    assert "DIRICHLET_VALUE: float = 2.5" in code


def test_a_zero_dirichlet_value_is_still_zero(tmp_path):
    """Guards against a fix that mistakes 'unset' for 'zero' or vice versa."""
    code = _experiment("0", tmp_path).render_code("pde")

    assert "DIRICHLET_VALUE: float = 0.0" in code


def test_a_boundary_held_at_a_declared_parameter_resolves_to_its_value(tmp_path):
    """The slot promises a constant, a parameter, or an expression; `D` is the second."""
    code = _experiment("D", tmp_path).render_code("pde")

    assert "DIRICHLET_VALUE: float = 1.0" in code


def test_a_boundary_condition_this_lowering_cannot_hold_says_so(tmp_path):
    """An expression silently became 0.0, which reads as a physically meaningful answer.

    Failing loudly is the point: a reader cannot tell a zero boundary from a dropped one.
    """
    with pytest.raises(ValueError, match="not at an expression"):
        _experiment("sin(t)", tmp_path).render_code("pde")


def test_a_dirichlet_condition_that_states_no_value_stays_homogeneous(tmp_path):
    """An unstated boundary is the homogeneous one, not an error."""
    code = _experiment(None, tmp_path).render_code("pde")

    assert "DIRICHLET_VALUE: float = 0.0" in code
