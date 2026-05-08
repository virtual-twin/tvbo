from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Event, Exploration, ExplorationAxis


dynamics = DynamicalSystem(
    state_variables={
        "x": {"equation": {"rhs": "a*x + perturbation"}, "initial_value": 0.0}
    },
    parameters={"a": {"value": 1.0}},
)
experiment = SimulationExperiment(dynamics=dynamics)
experiment.explorations = Exploration(
    name="a_sweep",
    space=ExplorationAxis(parameter="a", explored_values=[1.0, -1.0]),
)
experiment.integration.duration = 4.0
experiment.integration.step_size = 0.01
experiment.events["perturbation"] = Event(
    name="perturbation",
    event_type="stimulus",
    parameters={
        "onset": {"value": 1.0},
        "width": {"value": 0.1},
        "amplitude": {"value": 1.0},
    },
    equation={"rhs": "Piecewise((amplitude, (t >= onset) & (t < onset + width)), (0.0, True))"},
)

original_render_equation = experiment.dynamics.render_equation


def traced_render_equation(obj, *args, **kwargs):
    name = getattr(obj, "name", None)
    equation = getattr(getattr(obj, "equation", None), "rhs", None)
    if equation and "perturbation" in str(equation):
        print("TRACE", name, kwargs)
    return original_render_equation(obj, *args, **kwargs)


experiment.dynamics.render_equation = traced_render_equation
code = experiment.render_code("tvboptim")
for line in code.splitlines():
    if "dx_dt =" in line:
        print(line)
