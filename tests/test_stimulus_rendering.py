"""A rendered stimulus must be something TVB can actually evaluate.

TVBO used to render a stimulus three different ways depending on how its equation happened
to be written down, and two of the three produced a string TVB rejects:

* a `Piecewise` stated in `rhs` was printed as a Python `a if c else b`, which `numexpr` —
  TVB's evaluator — refuses, and left the time symbol as `t` when TVB binds `var`;
* an equation stated as `conditionals` was post-processed into `where(...)` by a regex over
  the printed Python, keeping the `and` that `numexpr` also refuses.

Only the third, a plain right-hand side, worked. The single stimulus in the database is of
the first kind, so the code TVBO emitted for it could not run.

All three now parse once and print once, through the `tvb` printer. These tests assert the
outcome rather than the spelling: the emitted string is handed to TVB's own `Equation` and
its values compared against SymPy's, which is what "the stimulus means what it says" means.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tvbo.datamodel import schema as tvbo_schema

pytestmark = pytest.mark.backend_tvb

VAR = np.linspace(0.0, 20.0, 201)
PARAMETERS = {"amplitude": 0.4, "onset": 10.0, "duration": 1.0}

# The same pulse — amplitude on [onset, onset+duration), zero elsewhere — in each spelling
# the schema allows. Only the second was ever emitted correctly.
SPELLINGS = {
    "piecewise in rhs": dict(
        rhs="Piecewise((amplitude, (t >= onset) & (t < onset + duration)), (0.0, True))",
    ),
    "conditional branches": dict(
        conditionals=[
            tvbo_schema.ConditionalBlock(
                condition="(t >= onset) & (t < onset + duration)",
                expression="amplitude",
            )
        ],
        rhs="0.0",
    ),
}

EXPECTED = np.where((VAR >= PARAMETERS["onset"]) & (VAR < PARAMETERS["onset"] + PARAMETERS["duration"]),
                    PARAMETERS["amplitude"], 0.0)


def _stimulus(**equation):
    from tvbo.classes.perturbation import Stimulus

    return Stimulus(
        label="Pulse",
        description="Rectangular pulse.",
        equation=tvbo_schema.Equation(**equation),
        parameters={
            name: tvbo_schema.Parameter(name=name, value=value) for name, value in PARAMETERS.items()
        },
    )


def _evaluate(stimulus) -> np.ndarray:
    """Render for TVB, execute the module, and let TVB evaluate its own equation."""
    namespace: dict = {}
    exec(stimulus.render_code(format="tvb"), namespace)  # noqa: S102
    return np.asarray(namespace["PulseEquation"]().evaluate(VAR), dtype=float)


def _default_of(stimulus) -> str:
    """The equation string the rendered TVB class carries."""
    code = stimulus.render_code(format="tvb")
    return next(line for line in code.splitlines() if "default=" in line).split('"')[1]


@pytest.mark.parametrize("spelling", list(SPELLINGS), ids=list(SPELLINGS))
def test_every_spelling_of_a_pulse_evaluates_in_tvb(spelling: str):
    """However the pulse is written down, TVB evaluates it to the pulse."""
    np.testing.assert_allclose(_evaluate(_stimulus(**SPELLINGS[spelling])), EXPECTED, atol=0)


@pytest.mark.parametrize("spelling", list(SPELLINGS), ids=list(SPELLINGS))
def test_no_spelling_emits_python_the_evaluator_rejects(spelling: str):
    """Both branch spellings take the one printer path, so neither emits Python syntax.

    A Python conditional or a Python boolean connective is not a stylistic difference here:
    `numexpr` raises on both, which is exactly how the old three-way branch shipped code
    that could not run.
    """
    default = _default_of(_stimulus(**SPELLINGS[spelling]))
    assert default.startswith("where("), default
    assert not any(token in default for token in (" if ", " else ", " and ", " or ", " not ")), default


def test_a_stimulus_names_time_the_way_tvb_binds_it():
    """TVB calls `evaluate(var)`; an emitted `t` would silently resolve to nothing."""
    default = _default_of(_stimulus(rhs="amplitude * sin(t / 10)"))
    names = set(re.findall(r"[A-Za-z_]\w*", default))
    assert "var" in names and "t" not in names, default


def test_an_authored_pycode_wins_over_an_equation_tvbo_cannot_parse():
    """`Equation.pycode` is the escape hatch, so it must be consulted *before* parsing.

    The right-hand side here is deliberately unparseable, because that is the only
    situation `pycode` exists for. Written with a *parseable* `rhs` — as this test first
    was — it passes whether or not the escape hatch works, which is how a template that
    parsed unconditionally shipped green: it raised on exactly the equations `pycode` was
    there to rescue.
    """
    stimulus = _stimulus(rhs="some_undeclared_thing(var, ??)", pycode="where(var > 5, 1.0, 0.0)")
    assert 'default="where(var > 5, 1.0, 0.0)"' in stimulus.render_code(format="tvb")


def test_an_equation_may_not_state_a_branch_as_a_where_call():
    """The ontology's `where(...)` spelling is rejected, not silently reinterpreted.

    It used to be rewritten into a `Piecewise` by a hand-written string splitter — a third
    producer, and one that no equation in the shipped ontology or database actually needs.
    """
    from tvbo.classes.equation import sympify_value

    class _OntologyEquation:
        has_function = has_parameter = has_state_variable = []
        value = label = type("_First", (), {"first": staticmethod(lambda: "where(t > 1, 1, 0)")})()

    with pytest.raises(ValueError, match="conditionals"):
        sympify_value(_OntologyEquation())
