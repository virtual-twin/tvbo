## -*- coding: utf-8 -*-
<%!
from tvbo.adapters.julia_model import build_model_context, julia_solve
%>
<%
experiment = context.get('experiment', None)
model = experiment.dynamics if experiment is not None else context['model']
dt = context.get('dt', experiment.integration.step_size if experiment is not None else 0.01)
duration = context.get('duration', experiment.integration.duration if experiment is not None else 1000)
plot = context.get('plot', False)
fout = context.get('fout', False)

# All metadata→Julia translation is prepared here; the includes only emit syntax.
mc = build_model_context(model)
solve = julia_solve(model, experiment.integration if experiment is not None else None, dt)
%>
% if solve['stochastic']:
<%include file="/tvbo-julia-SDEProblem.jl.mako" args="model=model, mc=mc, duration=duration" />
% else:
<%include file="/tvbo-julia-model.jl.mako" args="mc=mc" />
<%include file="/tvbo-julia-ODEProblem.jl.mako" args="mc=mc, duration=duration, package=solve['package']" />
% endif

# Solve
sol = solve(prob, ${solve['solver']}(); ${solve['kwargs']})

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
