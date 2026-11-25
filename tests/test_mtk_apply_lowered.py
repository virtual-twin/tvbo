"""The ModelingToolkit adapter turns MTK's first-order lowering back into a Dynamics.

``ModelingToolkitAdapter._apply_lowered`` writes each lowered equation onto its state variable and adds a state variable for each auxiliary MTK introduces (``x_t`` for ``dx/dt``). It runs only after Julia lowered a model, so these tests hand it a lowering of the shape ``_lower_sympy`` returns and need no Julia. ``lower`` wraps the result back into the kind of system it was given; an experiment keeps its other settings and names the lowered dynamics as its ``model``.
"""

import sympy as sp

from tvbo import Dynamics, SimulationExperiment
from tvbo.adapters.modelingtoolkit import ModelingToolkitAdapter

OSCILLATOR = Dynamics.from_string(
    """
name: Oscillator
parameters:
  w: {value: 2.0}
state_variables:
  x:
    initial_value: 1.0
    derivative_initial_value: 0.5
    equation_order: 2
    equation: {lhs: "Derivative(x, t, 2)", rhs: "-w**2*x"}
"""
)


def _lowered():
    x, x_t, w = sp.symbols("x x_t w")
    return {
        "equations": {"x": sp.Eq(sp.Symbol("Dx"), x_t), "x_t": sp.Eq(sp.Symbol("Dx_t"), -(w**2) * x)},
        "unknowns": ["x", "x_t"],
        "parameters": ["w"],
    }


def test_a_lowered_second_order_model_is_a_first_order_dynamics_with_its_auxiliary():
    lowered = ModelingToolkitAdapter(OSCILLATOR)._apply_lowered(_lowered())
    assert lowered.name == "Oscillator_FirstOrder"
    assert set(lowered.state_variables) == {"x", "x_t"}
    assert {sv.equation_order for sv in lowered.state_variables.values()} == {1}
    aux = lowered.state_variables["x_t"]
    assert aux.name == "x_t" and aux.equation.rhs == "-w**2*x" and aux.initial_value == 0.5


def test_lowering_leaves_the_original_dynamics_unchanged():
    ModelingToolkitAdapter(OSCILLATOR)._apply_lowered(_lowered())
    assert set(OSCILLATOR.state_variables) == {"x"} and OSCILLATOR.state_variables["x"].equation_order == 2


def test_lowering_an_experiment_returns_the_experiment_with_its_dynamics_lowered(monkeypatch):
    monkeypatch.setattr(ModelingToolkitAdapter, "_lower_sympy", lambda self, **kwargs: _lowered())
    experiment = SimulationExperiment(dynamics=OSCILLATOR, label="oscillator run")
    lowered = ModelingToolkitAdapter(experiment).lower()
    assert isinstance(lowered, SimulationExperiment)
    assert set(lowered.dynamics.state_variables) == {"x", "x_t"} and lowered.model == "Oscillator_FirstOrder"
    assert lowered.label == "oscillator run" and experiment.dynamics is OSCILLATOR and experiment.model == "Oscillator"
