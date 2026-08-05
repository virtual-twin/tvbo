"""Dimensional verdicts on a model's equations.

Three-valued by design: `consistent`, `inconsistent`, `underdetermined`. Collapsing the
last two is how a checker becomes noise — 24 of the 39 curated models declare no units
at all, and calling those "wrong" would pressure fake declarations into the published
record, which is worse than no units.

Treating an undeclared quantity as dimensionless is the specific failure being excluded:
it silently corrupts the answer rather than reporting that it cannot be computed.
"""

from __future__ import annotations

import pytest
import sympy as sp

from tvbo.analysis.units import (
    CONSISTENT,
    INCONSISTENT,
    UNDERDETERMINED,
    check_units,
    declared_units,
)
from tvbo.classes.dynamics import Dynamics


def _model(state, parameters):
    """A one-equation model with the declared units under test."""
    model = Dynamics(name="UnitProbe", label="UnitProbe")
    for name, (value, unit) in parameters.items():
        model.add_parameter(name=name, value=value, unit=unit)
    for name, (initial, unit, rhs) in state.items():
        model.add_state_variable(name, rhs, initial_value=initial, unit=unit)
    model.update_metadata()
    return model


def test_a_homogeneous_equation_is_consistent():
    """`dv/dt = -v/tau` in mV and ms: both sides are mV/ms."""
    model = _model(
        state={"v": (0.0, "mV", "-v / tau")},
        parameters={"tau": (10.0, "ms")},
    )

    (verdict,) = check_units(model)

    assert verdict.status == CONSISTENT, verdict.detail
    assert verdict


def test_adding_two_different_quantities_is_inconsistent():
    """A voltage plus a current is not a quantity, and the ratio says which is which."""
    model = _model(
        state={"v": (0.0, "mV", "(-v + I) / tau")},
        parameters={"tau": (10.0, "ms"), "I": (1.0, "pA")},
    )

    (verdict,) = check_units(model)

    assert verdict.status == INCONSISTENT
    assert "addends disagree" in verdict.detail
    assert not verdict


def test_an_undeclared_quantity_reports_underdetermined_not_dimensionless():
    """The failure this exists to prevent: an undeclared symbol assumed dimensionless.

    Treated as dimensionless, this equation would be *reported consistent* while being
    unverified — which is how a 92%-declared model came to report `dv/dt` as
    `1/capacitance` with no error raised.
    """
    model = _model(
        state={"v": (0.0, "mV", "(-v + I) / tau")},
        parameters={"tau": (10.0, "ms"), "I": (1.0, None)},
    )

    (verdict,) = check_units(model)

    assert verdict.status == UNDERDETERMINED
    assert "I" in verdict.detail


def test_an_undeclared_addend_is_inferred_from_the_ones_beside_it():
    """Additive homogeneity forces the unknown, exactly, rather than giving up.

    `I` added to a millivolt must be a millivolt; propagation says so without solving
    anything, which is the bound this checker keeps to.
    """
    model = _model(
        state={"v": (0.0, "mV", "(-v + I) / tau")},
        parameters={"tau": (10.0, "ms"), "I": (1.0, None)},
    )

    from tvbo.utils.units import unit_expression

    (verdict,) = check_units(model)
    inferred = {str(k): v for k, v in verdict.inferred.items()}

    assert "I" in inferred
    assert sp.simplify(inferred["I"] / unit_expression("mV")) == 1


def test_dimensional_and_exact_disagree_where_only_the_scale_differs():
    """`V/s` and `V/ms` are the same dimension and different quantities.

    The clock is `ms`, so the left side is `V/ms` while `tau` in seconds makes the right
    `V/s`. `dimensional` accepts the pair — both are voltage over time — and `exact`
    refuses it and reports the ratio `1/1000`. The ratio is the number that names the
    bug; a boolean does not.
    """
    model = _model(
        state={"v": (0.0, "V", "-v / tau")},
        parameters={"tau": (10.0, "s")},
    )

    dimensional = check_units(model, strictness="dimensional")[0]
    exact = check_units(model, strictness="exact")[0]

    assert dimensional.status == CONSISTENT
    assert exact.status == INCONSISTENT
    assert exact.ratios["right/left"] == sp.Rational(1, 1000)


def test_a_function_argument_must_be_dimensionless():
    """`exp(v)` with `v` in millivolts is meaningless, and is refused."""
    model = _model(
        state={"v": (0.0, "mV", "exp(v) / tau")},
        parameters={"tau": (10.0, "ms")},
    )

    (verdict,) = check_units(model)

    assert verdict.status == INCONSISTENT
    assert "dimensionless" in verdict.detail


def test_an_unknown_strictness_is_refused():
    with pytest.raises(ValueError, match="strictness"):
        check_units(_model(state={"v": (0.0, "mV", "-v")}, parameters={}), strictness="approximate")


def test_units_are_a_projection_keyed_like_the_others():
    """`declared_units` is keyed by the scope's own symbols, like `symbolic["parameters"]`."""
    model = _model(state={"v": (0.0, "mV", "-v / tau")}, parameters={"tau": (10.0, "ms")})

    units = declared_units(model)
    scope = model.get_symbolic_elements(time_dependent=True)

    assert units[scope["tau"]] == "ms"
    assert scope["tau"] in units


class TestCuratedModels:
    """The checker against the shipped database, where the interesting cases live."""

    def test_a_model_function_is_inlined_before_checking(self):
        """`Jansen1995` is consistent only once `Sigm(...)` is expanded.

        A user function carries no unit, so a call to one is opaque to propagation and
        the sigmoid's argument cannot be shown dimensionless. Inlining resolves it
        exactly — `r` is `per_mV` and `v0` is `mV` — which is why no `Function.unit`
        slot is needed. Four of the model's six equations then check out; the remaining
        two name genuinely undeclared parameters rather than claiming a contradiction.
        """
        verdicts = check_units(Dynamics.from_file("tvbo/database/models/Jansen1995.yaml"))
        statuses = [v.status for v in verdicts]

        assert statuses.count(CONSISTENT) == 4, [(v.name, v.status, v.detail) for v in verdicts]
        assert INCONSISTENT not in statuses
        assert {v.detail.split()[0] for v in verdicts if v.status == UNDERDETERMINED} == {"c_glob", "C4"}

    def test_a_real_declaration_error_is_found_with_its_ratio(self):
        """`CakanObermayer` adds `mu_se` (mV/ms) to `E_A` (mV).

        The ratio is exactly `1000/second`, which is `mV_per_ms / mV` — so the report
        names the discrepancy rather than only flagging one. This is a defect in the
        curated model's declarations, not in the checker: one of the two units is wrong,
        and which one is a modelling question. Asserted here so that fixing the model
        fails this test deliberately.
        """
        verdicts = check_units(Dynamics.from_file("tvbo/database/models/CakanObermayer.yaml"))
        inconsistent = [v for v in verdicts if v.status == INCONSISTENT]

        assert len(inconsistent) == 1
        assert "mu_se" in inconsistent[0].detail
        assert "1000/second" in inconsistent[0].detail

    def test_every_verdict_is_one_of_the_three(self):
        """A crash is not a fourth answer.

        Coverage of the whole database — all 106 registered models, including the
        subdirectories a top-level glob misses — is frozen in
        `test_unit_verdict_corpus.py`; this only pins the return type.
        """
        verdicts = check_units(Dynamics.from_file("tvbo/database/models/ZetterbergJansen.yaml"))

        assert verdicts
        assert all(v.status in (CONSISTENT, INCONSISTENT, UNDERDETERMINED) for v in verdicts)
