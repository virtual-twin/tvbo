## -*- coding: utf-8 -*-
## SDEProblem setup (additive diagonal noise)
<%page args="model, duration"/>
using DifferentialEquations
<%include file="/tvbo-julia-model.jl.mako" args="model=model" />

# Initial conditions (scalar state vector)
u0 = [
        % for sv in model.state_variables.values():
        ${sv.initial_value}, # Initial value for ${sv.name}
        % endfor
    ]

# Define time span
tspan = (0.0, ${duration})

# Construct per-state sigma vector directly from state variable noise definitions
sigma_vec = [
    % for sv in model.state_variables.values():
    % if getattr(sv, 'noise', None) and getattr(getattr(sv, 'noise', None), 'intensity', None) and getattr(getattr(sv.noise, 'intensity', None), 'value', None) is not None:
    ${float(sv.noise.intensity.value)}, # ${sv.name}
    % else:
    0.0, # ${sv.name}
    % endif
    % endfor
]

# Drift and diffusion
f! = ${model.metadata.name}!
function g!(du, u, p, t)
    @inbounds for i in eachindex(u)
        du[i] = sigma_vec[i]
    end
    return nothing
end

prob = SDEProblem(f!, g!, u0, tspan, p)
