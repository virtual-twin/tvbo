<%page args="model, duration"/>
<%
n_modes = getattr(model, 'number_of_modes', 1) or 1
%>
## ODEProblem setup (deterministic)
using OrdinaryDiffEqTsit5

# Initial conditions (flat state vector; multi-mode SVs are length-n_modes blocks)
u0 = [
        % for sv in model.state_variables.values():
        % if n_modes > 1:
        % for k in range(n_modes):
        ${sv.initial_value}, # ${sv.name} (mode ${k})
        % endfor
        % else:
        ${sv.initial_value}, # Initial value for ${sv.name}
        % endif
        % endfor
    ]

# Define time span
tspan = (0.0, ${duration}) # Adjust time span as needed

prob = ODEProblem(${model.name}!, u0, tspan, p)
