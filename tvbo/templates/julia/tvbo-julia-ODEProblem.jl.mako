<%page args="model, duration"/>
## ODEProblem setup (deterministic)
using OrdinaryDiffEqTsit5

# Initial conditions (scalar state vector)
u0 = [
        % for sv in model.state_variables.values():
        ${sv.initial_value}, # Initial value for ${sv.name}
        % endfor
    ]

# Define time span
tspan = (0.0, ${duration}) # Adjust time span as needed

prob = ODEProblem(${model.name}!, u0, tspan, p)
