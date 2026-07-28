## -*- coding: utf-8 -*-
<%!
from tvbo.adapters.julia_model import build_model_context
%>
<%
if 'experiment' in context.keys():
    model = context['experiment'].dynamics
    dt = context['experiment'].integration.step_size
    duration = context['experiment'].integration.duration
else:
    model = context['model']

if 'duration' not in context.keys():
    duration=1000
if 'dt' not in context.keys():
    dt = 0.01
plot = context.get('plot', False)
fout = context.get('fout', False)

# All metadata→Julia translation is prepared here; the includes only emit syntax.
mc = build_model_context(model)
%>

## ODE vs SDE: any state variable with a positive noise amplitude makes it stochastic.
## Read through the shared `noise_sigma`, so a recipe spelling its amplitude as
## `parameters.sigma` is not silently integrated as a deterministic ODE here while
## every other backend simulates it with noise.
<%
from tvbo.utils import noise_sigma


def has_noise(model):
    # Prefer live state_variables (may include user-added noise) over metadata snapshot
    return any(
        (noise_sigma(getattr(sv, 'noise', None)) or 0.0) > 0
        for sv in getattr(model, 'state_variables', {}).values()
    )
%>
% if has_noise(model):
<%include file="/tvbo-julia-SDEProblem.jl.mako" args="model=model, mc=mc, duration=duration" />
% else:
<%include file="/tvbo-julia-model.jl.mako" args="mc=mc" />
<%include file="/tvbo-julia-ODEProblem.jl.mako" args="mc=mc, duration=duration" />
% endif

# Solve
% if has_noise(model):
sol = solve(prob, EulerHeun(); dt=${dt}, saveat=${dt})
% else:
sol = solve(prob, Tsit5(); saveat=${dt})
% endif

%if plot:
# Plot the solution
using Plots
plot(
    sol,
    linewidth = 5,
    title = "Solution to ${model.name} ODE",
    xaxis = "Time (t)",
    yaxis = "u(t) (units)",
    label = "Simulation"
)
%endif

%if fout:

%endif
