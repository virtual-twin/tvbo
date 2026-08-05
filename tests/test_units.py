import pytest

from tvbo import Dynamics
from tvbo.datamodel.schema import UnitEnum
from tvbo.utils.units import normalize_unit, unit_to_symbol


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("mV", "mV"),
        ("millivolt", "mV"),
        ("ms^-1", "per_ms"),
        ("ms⁻¹", "per_ms"),
        ("mV/ms", "mV_per_ms"),
        ("mV*ms^-1", "mV_per_ms"),
        ("m/s²", "m_per_s2"),
        ("N/m", "N_per_m"),
        ("kg/s^2", "N_per_m"),
        ("mM", "mol_per_m3"),
        ("µF/cm²", "uF_per_cm2"),
        ("nS/mV", "nS_per_mV"),
    ],
)
def test_normalize_unit_aliases(raw, canonical):
    assert normalize_unit(raw) == canonical
    assert str(UnitEnum(raw)) == canonical


def test_unit_enum_attribute_aliases():
    assert str(getattr(UnitEnum, "kg/s^2")) == "N_per_m"
    assert str(getattr(UnitEnum, "mV/ms")) == "mV_per_ms"


def _spring(velocity_unit, stiffness_unit):
    """A spring-mass model whose two compound units are written by hand."""
    return Dynamics(
        name="SpringMass",
        state_variables={
            "x": {
                "description": "Displacement from equilibrium",
                "unit": "m",
                "equation": {"rhs": "v"},
                "initial_value": 2.0,
            },
            "v": {
                "description": "Velocity",
                "unit": velocity_unit,
                "equation": {"rhs": "-(k/m) * x"},
                "initial_value": 0.0,
            },
        },
        parameters={
            "k": {
                "description": "Spring stiffness",
                "unit": stiffness_unit,
                "value": 0.0001,
            },
            "m": {
                "description": "Mass",
                "unit": "kg",
                "value": 1.0,
            },
        },
    )


def test_dynamics_accepts_human_readable_spring_units():
    """Slash notation reaches the record under the curated name for the same unit."""
    spring = _spring(velocity_unit="m/s", stiffness_unit="kg/s^2")

    assert str(spring.state_variables["v"].unit) == "m_per_s"
    assert str(spring.parameters["k"].unit) == "N_per_m"
    assert unit_to_symbol(spring.parameters["k"].unit) == "N/m"


def test_dynamics_records_an_uncurated_unit_as_written():
    """`m/ms` and `kg/ms²` are real units TVBO has not curated.

    Both used to be rewritten to the curated unit a thousand (and a million) times
    away — `m/s` and `N/m` — which loads and runs and is wrong. Recording them as
    written keeps the declaration honest and leaves the checker able to say it
    cannot settle them.
    """
    spring = _spring(velocity_unit="m/ms", stiffness_unit="kg/ms^2")

    assert str(spring.state_variables["v"].unit) == "m/ms"
    assert str(spring.parameters["k"].unit) == "kg/ms^2"
