<%page args="mc, duration"/>
## ODEProblem setup (deterministic). ``mc`` = build_model_context(model).
using OrdinaryDiffEqTsit5

# Initial conditions (flat state vector; multi-mode SVs are length-n_modes blocks)
u0 = [
    % for val in mc['u0']:
        ${val},
    % endfor
    ]

# Define time span
tspan = (0.0, ${duration}) # Adjust time span as needed

prob = ODEProblem(${mc['func_name']}!, u0, tspan, p)
