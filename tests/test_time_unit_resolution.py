"""One resolver decides what a time-valued number means, at every scope.

TVBO used to answer "what unit is this duration in?" separately everywhere it was asked. The `time_scale` slot carried `ifabsent: ms`, so it was never unset — which made an inherited unit impossible to express and masked an explicitly declared `s` one scope out — and six call sites in the NeuroML adapter re-derived it anyway, with fallbacks that had drifted apart: one emitter defaulted to `s` where the rest defaulted to `ms`, and `SimulationExperiment.to_lems` appended `ms` unconditionally without reading the declaration at all. The database contains a model declaring `s`, so those disagreed on real data.

`time_unit_of` is now the only reader and the only place the `ms` fallback is stated.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from tvbo.utils.units import DEFAULT_TIME_UNIT, time_unit_factor, time_unit_of


class _Scope:
    """A scope carrying whichever spelling of the declaration is under test."""

    def __init__(self, **declared):
        for name, value in declared.items():
            setattr(self, name, value)


def test_an_undeclared_scope_falls_back_to_milliseconds():
    """The fallback exists once, here, rather than at each call site."""
    assert time_unit_of(_Scope()) == DEFAULT_TIME_UNIT == "ms"


def test_no_scopes_at_all_still_answers():
    assert time_unit_of() == "ms"
    assert time_unit_of(None, None) == "ms"


def test_a_declaration_beats_the_fallback():
    assert time_unit_of(_Scope(time_unit="s")) == "s"


def test_the_alias_spelling_is_read_too():
    """`time_scale` is the older spelling and stays readable."""
    assert time_unit_of(_Scope(time_scale="us")) == "us"


def test_the_innermost_declaring_scope_wins():
    """Scopes are passed innermost first, and the first declaration ends the walk."""
    inner, outer = _Scope(time_unit="s"), _Scope(time_unit="ms")

    assert time_unit_of(inner, outer) == "s"


def test_an_undeclared_inner_scope_inherits_from_an_outer_one():
    """The case `ifabsent: ms` made unrepresentable: silence means "ask my parent".

    With a default on the slot the inner scope always answered `ms`, so an outer `s` could never be inherited — a 1000x error in every derived rate, and silent.
    """
    assert time_unit_of(_Scope(), _Scope(time_unit="s")) == "s"


def test_a_none_scope_is_skipped_rather_than_answered():
    """Callers pass optional scopes straight through without pre-filtering."""
    assert time_unit_of(None, _Scope(time_unit="s"), None) == "s"


def test_an_empty_declaration_is_not_a_declaration():
    assert time_unit_of(_Scope(time_unit=None), _Scope(time_unit="s")) == "s"


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [("s", "ms", 1000), ("ms", "s", Fraction(1, 1000)), ("ms", "ms", 1), ("us", "ms", Fraction(1, 1000))],
)
def test_cross_scale_conversion_is_exact(source: str, target: str, expected):
    """A subnetwork in `s` folding into a parent in `ms` scales by exactly 1000.

    Exactly: the factor is a `Fraction`, because a factor of 1000 living only in a modeller's head is invisible in every backend's output, and one that arrives as 999.9999999999999 is worse than none.
    """
    factor = time_unit_factor(_Scope(time_unit=source), _Scope(time_unit=target))

    assert factor == expected
    assert isinstance(factor, Fraction)


def test_a_scale_that_needs_no_conversion_is_exactly_one():
    """Every network in the database declares `ms`, so this is today's only case — and it must be the identity, or the codegen corpus would move."""
    assert time_unit_factor(_Scope(), _Scope()) == 1


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("min", "ms", 60_000),
        ("h", "s", 3600),
        ("day", "h", 24),
        ("year", "day", Fraction(1461, 4)),
        ("ns", "us", Fraction(1, 1000)),
    ],
)
def test_a_slow_network_can_state_its_own_scale(source: str, target: str, expected):
    """`ns`, `min`, `h`, `day` and `year` are curated, so a non-brain model can declare one.

    The vocabulary used to stop at `ms`/`s`/`us`, which left a slow network unable to say what its clock meant. `year` is the Julian year, and converting it to days gives exactly 1461/4 — a ratio no float representation states exactly.
    """
    assert time_unit_factor(_Scope(time_unit=source), _Scope(time_unit=target)) == expected


def test_converting_from_something_that_is_not_a_time_is_refused():
    """`mV` is not a duration, and scaling by it would be silently meaningless."""
    with pytest.raises(ValueError, match="not a time unit"):
        time_unit_factor(_Scope(time_unit="mV"), _Scope(time_unit="ms"))


def test_the_database_still_declares_one_non_default_scale():
    """The `s` declaration that the disagreeing defaults used to get wrong survives.

    It is the reason this resolver exists rather than a constant, so if it ever disappears from the database this test should be deleted deliberately, not silently satisfied by a fallback.
    """
    import yaml

    from tests.database_corpus import DB

    declared = {
        path.relative_to(DB).as_posix(): text
        for path in DB.rglob("*.yaml")
        if (text := _declared_time_unit(yaml.safe_load(path.read_text()) or {})) and text not in (DEFAULT_TIME_UNIT,)
    }

    assert declared, "no non-ms time unit in the database — see this test's docstring"


def _declared_time_unit(data):
    """The first `time_unit`/`time_scale` anywhere in a loaded YAML document."""
    if isinstance(data, dict):
        for key in ("time_unit", "time_scale"):
            if isinstance(data.get(key), str):
                return data[key]
        for value in data.values():
            if found := _declared_time_unit(value):
                return found
    elif isinstance(data, list):
        for value in data:
            if found := _declared_time_unit(value):
                return found
    return None
