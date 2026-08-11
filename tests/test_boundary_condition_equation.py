"""A boundary condition states an equation, and the FEM lowering reads it.

`BoundaryCondition` held its value in a slot called `value` whose range is `Equation` —
the one spelling that already means "a number" on `Parameter`. The FEM template then read
it under `righthandside`, the alias the dialect folds away at construction, so `getattr`
returned the `Equation` object itself, `float()` of it raised, and a bare `except` turned
every Dirichlet boundary into 0.0. Nothing in the emitted code said so.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from tvbo.classes.experiment import SimulationExperiment
from tvbo.datamodel import tvbo_datamodel as dm


def _experiment(bc_equation):
    """A minimal FEM diffusion experiment whose one Dirichlet BC states *bc_equation*."""
    spec = {
        "label": "BC probe",
        "field_dynamics": {
            "label": "Heat",
            "mesh": {"label": "m", "element_type": "triangle"},
            "parameters": {"D": {"name": "D", "value": 1.0}},
            "state_variables": [{
                "name": "u", "label": "u", "initial_value": 0.0,
                "boundary_conditions": [
                    {"label": "BC", "bc_type": "Dirichlet", "equation": bc_equation}
                ],
                "equation": {"lhs": "u_t", "rhs": "D * laplacian(u)"},
            }],
            "operators": [{"label": "Diff", "operator_type": "laplacian", "coefficient": "D"}],
            "solver": {"label": "FEM", "discretization": "FEM",
                       "method": "implicit Euler", "dt": 1.0},
        },
        "integration": {"duration": 10},
    }
    path = Path(tempfile.mkdtemp()) / "experiment.yaml"
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


def test_a_nonzero_dirichlet_value_survives_into_the_generated_code():
    """The whole point: it used to arrive as 0.0 whatever the recipe said."""
    code = _experiment("2.5").render_code("pde")

    assert "DIRICHLET_VALUE: float = 2.5" in code


def test_a_zero_dirichlet_value_is_still_zero():
    """Guards against a fix that mistakes 'unset' for 'zero' or vice versa."""
    code = _experiment("0").render_code("pde")

    assert "DIRICHLET_VALUE: float = 0.0" in code


def test_a_boundary_condition_that_is_not_a_constant_says_so():
    """This lowering holds a boundary at a constant; an expression silently became 0.0.

    Failing loudly is the point — a zero boundary is a physically meaningful answer, so a
    reader has no way to tell it apart from a value that was dropped.
    """
    with pytest.raises(ValueError, match="constant only"):
        _experiment("sin(t)").render_code("pde")
