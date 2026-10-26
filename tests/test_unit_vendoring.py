"""Every curated unit resolves to a real SymPy quantity — the hard gate.

The failure this guards is silent. TVBO's unit map used to be a 37-entry display table in which 25 entries did not resolve: `mV`, `nS`, `pA`, `pF`, `per_ms`, `mm_per_ms`, `kHz`, `degC` and `dimensionless` all became *free symbols named after themselves*, which compare equal to nothing, raise nothing, and quietly make every dimensional statement about them vacuous. A separate 19-entry SI-factor table had the same shape: `.get(unit, 1.0)`, so `mm` converted as though it were metres.

Both are now read from `tvbo/data/ontology/unit_facts.json`, vendored from QUDT by `scripts/ontology/gen_units.py`. These tests assert the property that made the old tables wrong — that a unit TVBO claims to know is one it can actually compute with.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from sympy import Integer, Rational, simplify
from sympy.physics import units as sympy_units

from tvbo.utils.units import (
    unit_dimensions,
    unit_expression,
    unit_facts,
    unit_multiplier,
    unit_to_si_factor,
)

REPO = Path(__file__).resolve().parents[1]

ENUM = list(yaml.safe_load((REPO / "schema" / "units.yaml").read_text())["enums"]["UnitEnum"]["permissible_values"])
VENDORED = json.loads((REPO / "tvbo" / "data" / "ontology" / "unit_facts.json").read_text())["units"]
CURATED = [name for name in ENUM if VENDORED.get(name, {}).get("curated")]
UNCURATED = [name for name in ENUM if not VENDORED.get(name, {}).get("curated")]


def test_every_enum_value_has_a_vendored_record():
    """No `UnitEnum` value may be absent from the vendored file, curated or not."""
    assert set(ENUM) == set(VENDORED), set(ENUM).symmetric_difference(VENDORED)


@pytest.mark.parametrize("unit", CURATED)
def test_a_curated_unit_resolves_to_a_sympy_expression(unit: str):
    """The gate: a curated unit is one TVBO can put in an expression.

    A free symbol is the failure mode being excluded — it would satisfy "is not None" while carrying no dimensional content at all.
    """
    if VENDORED[unit].get("offset"):
        pytest.skip("affine unit — refused by design, covered by its own test")

    expression = unit_expression(unit)

    assert expression is not None, f"{unit} is curated but has no expression"
    assert not expression.free_symbols, f"{unit} resolved to free symbols {expression.free_symbols}"


@pytest.mark.parametrize("unit", CURATED)
def test_a_curated_unit_states_its_dimensions_and_scale(unit: str):
    """Every curated unit carries both halves of `Q = {Q}·[Q]`."""
    assert unit_multiplier(unit) is not None
    assert unit_dimensions(unit) is not None


def test_an_uncurated_unit_says_so_rather_than_guessing():
    """`per_unit` is in the enum but has no unit facts, and must not invent any."""
    assert UNCURATED == ["per_unit"], UNCURATED
    assert unit_facts("per_unit") is None
    assert unit_expression("per_unit") is None
    assert unit_dimensions("per_unit") is None


def test_an_unknown_unit_is_not_silently_dimensionless():
    """An unrecognised spelling reports "unknown", which is not "dimensionless"."""
    assert unit_facts("furlongs_per_fortnight") is None
    assert unit_dimensions("furlongs_per_fortnight") is None
    assert unit_dimensions("dimensionless") == {}


def test_an_affine_unit_is_refused_rather_than_approximated():
    """`degC` is kelvin plus an offset, so no single multiplicative expression states it."""
    assert unit_dimensions("degC") == {"kelvin": Fraction(1)}
    with pytest.raises(ValueError, match="affine"):
        unit_expression("degC")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("s", sympy_units.second),
        ("ms", sympy_units.second / 1000),
        ("mV", sympy_units.kilogram * sympy_units.meter**2 / (1000 * sympy_units.ampere * sympy_units.second**3)),
        ("per_ms", 1000 / sympy_units.second),
        ("dimensionless", Integer(1)),
    ],
)
def test_a_unit_expression_is_the_quantity_it_names(unit: str, expected):
    assert simplify(unit_expression(unit) - expected) == 0


def test_the_same_quantity_under_two_spellings_is_one_expression():
    """`mm/ms` and `m/s` are the same velocity, and must be exactly equal.

    Exactly, not approximately: the multiplier is composed as a `Fraction`, so this is `1`, where composing it in floating point gives 0.9999999999999999 and turns "identical" into "inconsistent by 1e-16" for every check downstream.
    """
    assert unit_multiplier("mm_per_ms") == Fraction(1)
    assert simplify(unit_expression("mm_per_ms") - unit_expression("m_per_s")) == 0


def test_a_prefixed_unit_keeps_its_prefix_exactly():
    """The distinction the old table lost: `mV` is not `V`, and the ratio is exact."""
    ratio = simplify(unit_expression("mV") / unit_expression("V"))

    assert ratio == Rational(1, 1000)


@pytest.mark.parametrize(("unit", "factor"), [("mm", 1e-3), ("cm", 1e-2), ("um3", 1e-18), ("percent", 1e-2)])
def test_si_factors_the_old_table_defaulted_to_one(unit: str, factor: float):
    """21 of 61 units fell through `.get(unit, 1.0)`; `mm` converted as metres."""
    assert unit_to_si_factor(unit) == pytest.approx(factor)


def test_lems_time_normalisation_is_not_the_base_dimension_vector():
    """`unit_has_time_dimension` asks a narrower question than the dimension vector.

    It drives NeuroML's `/ SEC` normalisation: does this quantity already express a rate. That is not "does the SI decomposition contain a second" — under the base vector `mV` is `kg·m²·s⁻³·A⁻¹` and would qualify, as would `V`, `nS`, `ohm` and `W`; 24 of the 62 units differ between the two readings. Deriving this predicate from the vendored vector would therefore silently change emitted LEMS, so it stays its own vocabulary, and this test records why.
    """
    from tvbo.utils.units import unit_has_time_dimension

    assert unit_has_time_dimension("ms") and unit_has_time_dimension("per_ms")
    assert not unit_has_time_dimension("mV")
    assert unit_dimensions("mV")["second"] == Fraction(-3)


def test_an_uncurated_unit_is_recorded_rather_than_rejected():
    """The `unit` range is open, so a unit nobody curated yet can still be written down.

    `nM` used to raise `ValueError: Unknown UnitEnum enumeration code`, which left an author two options — misdeclare the quantity as something in the enum, or declare nothing. Both lose more than an unrecognised string does. It is recorded verbatim and carries no dimensional claim, so anything reasoning about it reports underdetermined instead of guessing.
    """
    from tvbo.datamodel import schema

    parameter = schema.Parameter(name="concentration", value=1.0, unit="nM")

    assert str(parameter.unit) == "nM"
    assert unit_facts("nM") is None
    assert unit_dimensions("nM") is None


def test_opening_the_range_leaves_a_curated_unit_untouched():
    """`unit: mV` means what it meant, which is why the dump corpus does not move."""
    from tvbo.datamodel import schema

    assert str(schema.Parameter(name="v", value=1.0, unit="mV").unit) == "mV"
    assert unit_dimensions("mV") is not None


def test_the_vendored_file_is_current():
    """A `UnitEnum` change without `make gen-units` leaves the two out of step."""
    stale = set(ENUM).symmetric_difference(VENDORED)

    assert not stale, f"run `make gen-units` — {sorted(stale)} differ between schema and vendored facts"
