"""What the report says about units, and what it refuses to say.

The report is the published record. A unit it prints is a claim the paper makes, so the two things that matter here are that a declared unit reaches the page typeset as a unit rather than as a product of italic variables, and that a unit nobody declared is never printed as though somebody had.
"""

from __future__ import annotations

import pytest
import sympy as sp

from tvbo.classes.dynamics import Dynamics
from tvbo.utils import report
from tvbo.utils.units import unit_expression, unit_named


@pytest.fixture(scope="module")
def jansen():
    return Dynamics.from_file("tvbo/database/models/Jansen1995.yaml")


class TestNamingADerivedUnit:
    """Propagation yields an expression; a reader wants a name where one exists."""

    def test_a_propagated_expression_gets_its_ordinary_name_back(self):
        """`kilogram*meter**2/(1000*ampere*second**3)` is `mV`, and reads better so."""
        assert unit_named(unit_expression("mV")) == "mV"
        assert report.unit_latex(unit_expression("mV")) == r"\mathrm{mV}"

    def test_an_ambiguous_dimension_is_not_given_one_of_its_names(self):
        """`Hz`, `per_s` and `rad_per_s` are all exactly `1/second`.

        Naming a derived quantity `Hz` would assert it is a frequency on evidence that says only "per second". The base expression is the weaker claim and the true one.
        """
        assert unit_named(unit_expression("Hz")) is None
        assert unit_named(unit_expression("per_s")) is None
        assert unit_named(unit_expression("dimensionless")) is None

    def test_the_ambiguity_is_a_property_of_the_vocabulary_not_a_gap(self):
        """Five groups of curated units share an expression; that is the whole set."""
        from collections import defaultdict

        from tvbo.utils.units import _vendored

        groups = defaultdict(list)
        for unit in _vendored():
            try:
                expression = unit_expression(unit)
            except ValueError:
                continue
            if expression is not None:
                groups[expression].append(unit)
        shared = {frozenset(names) for names in groups.values() if len(names) > 1}

        assert shared == {
            frozenset({"Hz", "per_s", "rad_per_s"}),
            frozenset({"kHz", "per_ms", "rad_per_ms"}),
            frozenset({"arbitrary_unit", "dimensionless", "rad"}),
            frozenset({"m_per_s", "mm_per_ms"}),
            frozenset({"Hz_per_nA", "per_nC"}),
        }

    def test_a_unit_with_no_name_at_all_is_still_typeset(self):
        """`mV/ms²` is a real unit that the vocabulary does not carry."""
        odd = unit_expression("mV") / unit_expression("ms") ** 2

        assert unit_named(odd) is None
        assert report.unit_latex(odd)


class TestDeclaredVersusDerived:
    def test_a_derived_unit_is_parenthesised_not_stated(self):
        """`I` is forced by the millivolt beside it, not declared by the model.

        Printed bare it would put a unit nobody wrote into the published record;
        the parentheses are what separate the model's claim from an inference.
        """
        model = Dynamics(name="UnitProbe", label="UnitProbe")
        model.add_parameter(name="tau", value=10.0, unit="ms")
        model.add_parameter(name="I", value=1.0)
        model.add_state_variable("v", "(-v + I) / tau", initial_value=0.0, unit="mV")

        derived = report.derived_units(report.unit_verdicts(model))
        table = report.parameter_table(model.parameters, derived=derived)

        assert derived["I"] == r"\mathrm{mV}"
        assert r"($\mathrm{mV}$)" in table
        assert r"| $\tau$ | 10 | $\mathrm{ms}$ |" in table

    def test_a_declared_unit_still_wins_over_a_derived_one(self, jansen):
        """The model's own claim is never overwritten by an inference."""
        table = report.parameter_table(jansen.parameters, derived={"A": r"\mathrm{nonsense}"})

        assert r"$\mathrm{mV}$" in table
        assert "nonsense" not in table

    def test_no_marking_at_all_without_the_argument(self, jansen):
        """Every existing caller renders exactly what it did before."""
        assert report.parameter_table(jansen.parameters) == report.param_table(jansen.parameters, name_header="Parameter")


class TestVerdictTable:
    def test_each_equation_is_named_in_the_reports_own_notation(self, jansen):
        r"""`\dot{y_0}`, not `Derivative(y0(t), t)` and not `\dot{y_0(t)}`."""
        table = report.unit_verdict_table(report.unit_verdicts(jansen))

        assert r"$\dot{y_{0}}$" in table
        assert "Derivative" not in table
        assert r"\left(t \right)" not in table

    def test_a_first_derivative_is_order_one_even_with_assumptions(self):
        r"""`Symbol("t")` and `Symbol("t", real=True)` print alike and compare unequal.

        Counting by identity read order 0 for every equation in the analysis view, which typesets as the nonsense `\frac{d^0}{d t^0}`.
        """
        t = sp.Symbol("t", real=True)
        y = sp.Function("y", real=True)(t)

        assert report.time_order(sp.Derivative(y, t)) == 1
        assert report.derivative_latex("y", report.time_order(sp.Derivative(y, t))) == r"\dot{y}"

    def test_the_three_verdicts_are_reported_as_three(self, jansen):
        """`underdetermined` is an answer, not a soft failure, and says so."""
        table = report.unit_verdict_table(report.unit_verdicts(jansen))

        assert "consistent" in table
        assert "underdetermined" in table

    def test_an_inconsistency_is_marked_and_explained(self):
        """`CakanObermayer` adds a voltage rate to a voltage; the report names both."""
        model = Dynamics.from_file("tvbo/database/models/CakanObermayer.yaml")

        table = report.unit_verdict_table(report.unit_verdicts(model))

        assert "**inconsistent**" in table
        assert "mu_se" in table
        assert "mu_se(t)" not in table


def test_a_unit_is_typeset_upright(jansen):
    """Units are not variables: italic `mV` reads as `m` times `V`."""
    table = report.parameter_table(jansen.parameters)

    assert r"$\mathrm{mV}$" in table
    assert r"$\mathrm{ms}^{-1}$" in table
