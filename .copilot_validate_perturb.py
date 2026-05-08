from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Event, Exploration, ExplorationAxis


def make_experiment():
    dynamics = DynamicalSystem(
        state_variables={
            "x": {
                "equation": {"rhs": "a*x + perturbation"},
                "initial_value": 0.0,
            }
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
        duration=0.1,
        parameters={
            "onset": {"value": 1.0},
            "width": {"value": 0.1},
            "amplitude": {"value": 1.0},
        },
        equation={
            "rhs": "Piecewise((amplitude, (t >= onset) & (t < onset + width)), (0.0, True))"
        },
    )
    return experiment


experiment = make_experiment()
code = experiment.render_code("tvboptim")
for line in code.splitlines():
    if "dx_dt =" in line or "signal =" in line:
        print(line)

result = experiment.run("tvboptim")
print("OK", type(result).__name__)
print("exploration dims", result.explorations["a_sweep"].data.shape)
