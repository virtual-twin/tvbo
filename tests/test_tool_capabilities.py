"""A backend's unit vocabulary lives on its software entry, not in a shared table.

The tables this replaced sat in `tvbo.utils.units` as though "what LEMS calls a `per_time` quantity" were a fact about the unit. It is a fact about LEMS, and it is the reason a second backend could not state its own without editing a module every other backend imports.
"""

from __future__ import annotations

import pytest

from tvbo.classes.software import SimulationTool
from tvbo.utils.units import unit_to_lems_dimension, unit_to_lems_symbol


@pytest.fixture(scope="module")
def lems():
    return SimulationTool.for_format("neuroml")


def test_the_format_resolves_to_the_tool_that_declares_it(lems):
    """`codegen_format` is the join: an export-registry key names one tool."""
    assert lems is not None
    assert lems.name == "LEMS"
    assert lems.codegen_format == "neuroml"


def test_an_unclaimed_format_is_none_not_an_error():
    """Most formats have no tool entry, and the caller's fallback is the same as for a tool that is merely silent on a unit."""
    assert SimulationTool.for_format("yaml") is None


def test_a_tool_states_its_own_dimension_names(lems):
    """LEMS's names, read from LEMS's entry."""
    assert lems.dimension_of("mV") == "voltage"
    assert lems.dimension_of("pF") == "capacitance"
    assert lems.dimension_of("Hz") == "per_time"


def test_a_tools_spelling_can_differ_from_the_unit_it_translates(lems):
    """`Hz` and `kHz` are written `per_s` and `per_ms` — which is exactly why the vocabulary cannot be derived from the unit and has to be declared."""
    assert lems.symbol_of("Hz") == "per_s"
    assert lems.symbol_of("kHz") == "per_ms"
    assert lems.symbol_of("mV") == "mV"


def test_a_unit_the_tool_does_not_know_is_dimensionless_and_bare(lems):
    """`mm_per_ms` is a perfectly good unit that LEMS has no name for.

    It has to come back `("none", "")` rather than raise: an emitter asking how to write a value needs an answer for every value.
    """
    assert lems.dimension_of("mm_per_ms") == "none"
    assert lems.symbol_of("mm_per_ms") == ""
    assert lems.dimension_of(None) == "none"
    assert lems.symbol_of(None) == ""


def test_only_units_the_tool_names_are_recorded(lems):
    """The record states what LEMS knows, not the whole enum restated.

    24 of the 54 entries in the tables this replaced said `("none", "")` — the default spelled out — so they claimed nothing and are gone.
    """
    assert "mm_per_ms" not in lems.units
    assert "dimensionless" not in lems.units
    assert len(lems.units) == 30


def test_a_tool_may_name_units_outside_the_curated_vocabulary(lems):
    """`uA`, `uF` and `K` are LEMS units that TVBO does not curate.

    They are kept because the `unit` slot's range is open (U14): a model may declare `uA`, and LEMS knows how to write it even though TVBO holds no QUDT record for it. Dropping them would silently downgrade such a model to unitless.
    """
    from tvbo.utils.units import unit_facts

    outside = {u for u in lems.units if unit_facts(u) is None}

    assert outside == {"uA", "uF", "mS", "kohm", "M", "K"}
    assert all(lems.symbol_of(u) == u for u in outside)


def test_an_alias_keeps_the_spelling_the_tool_asked_for(lems):
    """`mM` normalises onto the curated `mol_per_m3`, and LEMS still writes `mM`.

    The two are separate entries because they are separate questions: what the quantity *is* (concentration, from QUDT) and what this backend *writes*. A tool's key is therefore left exactly as the tool spells it, which is also why `mM` and `mol_per_m3` can both appear in one vocabulary.
    """
    from tvbo.utils.units import normalize_unit

    assert normalize_unit("mM") == "mol_per_m3"
    assert lems.symbol_of("mM") == "mM"
    assert lems.dimension_of("mol_per_m3") == "concentration"


def test_the_module_level_helpers_read_the_record(lems):
    """The 19 call sites are unchanged; the data underneath them moved."""
    assert unit_to_lems_dimension("mV") == lems.dimension_of("mV") == "voltage"
    assert unit_to_lems_symbol("Hz") == lems.symbol_of("Hz") == "per_s"


def test_physical_dimension_is_gone():
    """`PhysicalDimension` restated LEMS's dimension names in TVBO's schema.

    Nothing declared a slot of that range — it was reachable only by importing it — so the names now live on the one tool that actually uses them.
    """
    from tvbo.datamodel import schema

    assert not hasattr(schema, "PhysicalDimension")
