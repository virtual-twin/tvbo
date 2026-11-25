"""Descriptions reach the generated ModelingToolkit code as Julia string literals, so the characters Julia gives a meaning inside a string arrive escaped."""

import re

from tvbo import SimulationExperiment

DESCRIPTION = 'Rate of the "conductance" state, \\dot{x} in $mV/ms$'


def test_a_description_is_one_well_formed_julia_string():
    """A double quote would end the literal and a dollar sign would interpolate, so both arrive escaped, as does the backslash, in the description of a parameter, a state variable and a derived variable alike."""
    dynamics = {
        "name": "Escapes",
        "parameters": {"a": {"value": 1.0, "description": DESCRIPTION}},
        "state_variables": {"x": {"equation": {"rhs": "-a*x + y"}, "initial_value": 0.0, "description": DESCRIPTION}},
        "derived_variables": {"y": {"equation": {"rhs": "2*a"}, "description": DESCRIPTION}},
    }
    code = SimulationExperiment(dynamics=dynamics, integration={"duration": 1.0, "step_size": 0.1}).render_code(
        format="modelingtoolkit"
    )
    literals = re.findall(r'description="((?:[^"\\$]|\\.)*)"', code)
    assert literals.count('Rate of the \\"conductance\\" state, \\\\dot{x} in \\$mV/ms\\$') == 3, literals
