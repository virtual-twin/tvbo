## -*- coding: utf-8 -*-
## SDEProblem setup (additive diagonal noise). ``mc`` = build_model_context(model);
## ``model`` is still needed for the per-state noise intensities below.
<%page args="model, mc, duration"/>
using StochasticDiffEq
<%include file="/tvbo-julia-model.jl.mako" args="mc=mc" />

# Initial conditions (flat state vector)
u0 = [
    % for val in mc['u0']:
        ${val},
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
f! = ${model.name}!
function g!(du, u, p, t)
    @inbounds for i in eachindex(u)
        du[i] = sigma_vec[i]
    end
    return nothing
end

prob = SDEProblem(f!, g!, u0, tspan, p)
