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

## Per-state sigma through the shared reader, so every declared spelling
## (`parameters.sigma`, `intensity`, `parameters.nsig`) means here what it means
## on the other backends.
<%! from tvbo.utils import noise_sigma %>\
# Per-state noise standard deviation (diagonal diffusion)
sigma_vec = [
    % for sv in model.state_variables.values():
    ${float(noise_sigma(getattr(sv, 'noise', None)) or 0.0)}, # ${sv.name}
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
