"""Every alias must be a true statement about the unit it maps to.

`_LEGACY_TO_ENUM` translates what an author wrote into TVBO's curated name for it.
A row is not a preference — it asserts that two spellings are the *same quantity* —
so each one is checkable against the vendored facts, and six rows were not true:
they dropped a milli prefix from a denominator, so `m/ms` was recorded as `m/s`
(×1000), `kg/ms²` as `N/m` (×10⁶), and `mM` as `mmol/m³` (×1000).

Silent is the operative word. A rejected unit is a visible error, but a unit
quietly rewritten to a neighbouring one leaves a model that loads, validates,
renders and runs — a thousand times off.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from tvbo.utils.units import _LEGACY_TO_ENUM, _UNIT_EQUIVALENTS, normalize_unit, unit_expression

pytestmark = pytest.mark.backend_core

SUPERSCRIPTS = str.maketrans({"²": "2", "³": "3", "⁻": "-"})


def _expression(name):
    """The vendored expression for *name*, or `None` where there cannot be one."""
    try:
        return unit_expression(name)
    except ValueError:
        return None


def _atom(text):
    """One factor of a compound — a curated unit, optionally raised to a power."""
    text = text.strip("() ")
    if text in ("", "1"):
        return sp.Integer(1)
    match = re.fullmatch(r"([A-Za-zµμΩ°]+)\^?(-?\d+)?", text)
    if not match:
        return None
    base, exponent = match.groups()
    expression = _expression(base)
    return None if expression is None else expression ** int(exponent or 1)


def compose(alias):
    """*alias* built from its curated atoms, or `None` if one of them is uncurated.

    Deliberately independent of `normalize_unit`: asking the alias table to
    check itself would confirm whatever it already says.
    """
    text = alias.translate(SUPERSCRIPTS).replace("·", "*").replace("**", "^")
    if not re.search(r"[/*^]", text):
        return None
    composed = sp.Integer(1)
    for position, group in enumerate(text.split("/")):
        for part in group.split("*"):
            factor = _atom(part)
            if factor is None:
                return None
            composed = composed / factor if position else composed * factor
    return composed


ROWS = sorted({**_LEGACY_TO_ENUM, **_UNIT_EQUIVALENTS}.items())
CHECKABLE = [(alias, target) for alias, target in ROWS if compose(alias) is not None]


@pytest.mark.parametrize(("alias", "target"), CHECKABLE, ids=[alias for alias, _ in CHECKABLE])
def test_a_compound_alias_equals_the_unit_it_maps_to(alias, target):
    """Composed from its own atoms, an alias must come out as exactly its target."""
    ratio = sp.nsimplify(sp.simplify(compose(alias) / _expression(target)))

    assert ratio == 1, f"`{alias}` -> `{target}` is off by {ratio}"


def test_the_check_still_reaches_the_rows_it_used_to():
    """Coverage that shrinks silently is the same as no coverage.

    A row becomes uncheckable when one of its atoms is not curated on its own —
    `uA/cm2` cannot be composed because `uA` is not a unit in its own right — so
    this number moves when the vocabulary does, and never quietly downwards.
    """
    assert len(CHECKABLE) == 27


def test_millimolar_is_mole_per_cubic_metre():
    """`mM` is `mol/m³`, not `mmol/m³`.

    One mM is one mmol per litre and a litre is 10⁻³ m³, so the two prefixes
    cancel. NeuroML says the same in `NeuroMLCoreDimensions.xml`, giving `mM` and
    `mol_per_m3` the same power — cited as a second, independent source, since the
    table under test is not evidence for itself.
    """
    from tvbo.adapters.neuroml import UNITS

    assert normalize_unit("mM") == "mol_per_m3"
    assert UNITS["mM"] == UNITS["mol_per_m3"]


def test_every_class_declaring_the_open_slot_canonicalizes_it():
    """Which classes get canonicalized is read off the generated annotations.

    So a class that gains the slot later is covered without anyone remembering,
    and — the reason this is asserted — a change in the shape of those annotations
    fails here instead of silently switching canonicalization off everywhere.
    """
    from tvbo.datamodel import _open_unit_slot_classes

    covered = {cls.__name__ for cls in _open_unit_slot_classes()}

    assert {"Parameter", "StateVariable", "DerivedParameter", "Observation", "Edge"} <= covered
    assert "ToolUnit" not in covered


def test_a_real_but_uncurated_unit_is_not_rounded_to_a_curated_neighbour():
    """`kg/ms²` is a real unit TVBO has not curated, and 10⁶ times `N/m`.

    Answering `None` sends it down the open-range path, where it is recorded as
    written and reported underdetermined. That is the whole reason the range was
    opened: the alternative on offer was never "curated or rejected", it was
    "curated or silently changed".
    """
    assert normalize_unit("kg/ms^2") is None
    assert normalize_unit("m/ms") is None
    assert normalize_unit("kg/s^2") == "N_per_m"
