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
        ("kg/ms^2", "N_per_m"),
        ("m/ms", "m_per_s"),
        ("µF/cm²", "uF_per_cm2"),
        ("nS/mV", "nS_per_mV"),
    ],
)
def test_normalize_unit_aliases(raw, canonical):
    assert normalize_unit(raw) == canonical
    assert str(UnitEnum(raw)) == canonical


def test_unit_enum_attribute_aliases():
    assert str(getattr(UnitEnum, "kg/ms^2")) == "N_per_m"
    assert str(getattr(UnitEnum, "m/ms")) == "m_per_s"


def test_dynamics_accepts_human_readable_spring_units():
    spring = Dynamics(
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
                "unit": "m/ms",
                "equation": {"rhs": "-(k/m) * x"},
                "initial_value": 0.0,
            },
        },
        parameters={
            "k": {
                "description": "Spring stiffness",
                "unit": "kg/ms^2",
                "value": 0.0001,
            },
            "m": {
                "description": "Mass",
                "unit": "kg",
                "value": 1.0,
            },
        },
    )

    assert str(spring.state_variables["v"].unit) == "m_per_s"
    assert str(spring.parameters["k"].unit) == "N_per_m"
    assert unit_to_symbol(spring.parameters["k"].unit) == "N/m"
