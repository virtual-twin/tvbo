"""An equation may state its right-hand side, or only its conditional branches.

Both spellings are valid in the schema and `parse_eq` resolves both. Only the first is
exercised by the curated database: every conditional model there also carries an `rhs`,
so **no** curated model proves the second works. That is the gap this closes.

It is not hypothetical. Reaching past an equation to `.rhs` yields `None` when the branches
are the whole definition, and `None` fails differently in each consumer — the Julia emitter
omitted a `NaNMath` import the emitted model then used, the NeuroML templates skipped the
element and emitted a component with **no dynamics at all**, and the report raised
`SympifyError: None`. None of those is a loud failure at the point of the mistake.

The model here is deliberately small and synthetic rather than added to the database: it
exists to be a shape no curated model has, and putting it in `tvbo/database/` would make it
part of the published record.
"""

from __future__ import annotations

import pytest

from tvbo.classes.dynamics import Dynamics

FORMATS = ("jax", "numpy", "julia", "tvb", "tvboptim", "neuroml")

_BRANCHES = [
    {"condition": "v > thr", "expression": "-a*v"},
    {"condition": "v <= thr", "expression": "b"},
]

SPEC = {
    "name": "BranchesOnly",
    "label": "Branches-only equations",
    "description": "One branches-only equation in each equation-bearing collection.",
    "parameters": {
        "a": {"name": "a", "value": 0.5},
        "b": {"name": "b", "value": 0.25},
        "thr": {"name": "thr", "value": 1.0},
    },
    "state_variables": {
        "v": {
            "name": "v",
            "initial_value": 0.0,
            "equation": {"lhs": "dv/dt", "conditionals": list(_BRANCHES)},
        }
    },
    "derived_variables": {"gated": {"name": "gated", "equation": {"lhs": "gated", "conditionals": list(_BRANCHES)}}},
    "derived_parameters": {
        "scale": {
            "name": "scale",
            "equation": {
                "lhs": "scale",
                "conditionals": [
                    {"condition": "a > 0", "expression": "1/a"},
                    {"condition": "a <= 0", "expression": "0"},
                ],
            },
        }
    },
    "output": ["gated"],
}


@pytest.fixture
def model() -> Dynamics:
    return Dynamics(**SPEC)


def test_the_fixture_really_has_no_right_hand_side(model: Dynamics):
    """Guards the premise. If a normalisation ever fills `rhs` in, every other test here
    silently stops testing anything."""
    stated = {
        "state_variables.v": model.state_variables["v"].equation,
        "derived_variables.gated": model.derived_variables["gated"].equation,
        "derived_parameters.scale": model.derived_parameters["scale"].equation,
    }
    for where, equation in stated.items():
        assert equation.rhs is None, f"{where} gained an rhs: {equation.rhs!r}"
        assert equation.conditionals, f"{where} lost its branches"


@pytest.mark.backend_core
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_emitter_keeps_the_branches(model: Dynamics, fmt: str):
    """Each emitter renders all three equations, branch condition included.

    `thr` appears only inside a branch condition, so a lowering that dropped the branches
    would emit a model that never mentions it.
    """
    code = model.render_code(format=fmt)
    for name in ("gated", "scale", "thr"):
        assert name in code, f"{fmt} output does not mention {name}"


@pytest.mark.backend_core
def test_the_report_renders_the_branches(model: Dynamics):
    """The report shows a branches-only equation as LaTeX cases, not as a crash."""
    text = str(model.generate_report())
    assert text.count(r"\begin{cases}") == 3, text
    assert "thr < v" in text


@pytest.mark.backend_core
def test_equation_groups_carry_every_collection(model: Dynamics):
    """`report.model_equation_groups` resolves all three, so no template needs to parse."""
    from tvbo.utils import report

    groups = report.model_equation_groups(model)
    assert [report.equation_name(eq) for eq in groups["state"]] == ["v"]
    assert [report.equation_name(eq) for eq in groups["derived"]] == ["gated"]
    assert [report.equation_name(eq) for eq in groups["derived_parameters"]] == ["scale"]
    for kind in ("state", "derived", "derived_parameters"):
        assert groups[kind][0].rhs.is_Piecewise, f"{kind} did not resolve to a Piecewise"
