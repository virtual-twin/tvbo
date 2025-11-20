## -*- coding: utf-8 -*-
<%
if 'experiment' in context.keys():
    model = context['experiment'].model
    dt = context['experiment'].metadata.integration.step_size
    duration = context['experiment'].metadata.integration.duration
else:
    model = context['model']

if 'duration' not in context.keys():
    duration=1000
if 'dt' not in context.keys():
    dt = 0.01
if 'plot' not in context.keys():
    # Provide a safe default so `%if plot:` blocks do not raise NameError
    plot = False
%>

## Decide problem type (ODE vs SDE) based on presence of any state variable noise intensity > 0
<%
def has_noise(model):
    # Prefer live state_variables (may include user-added noise) over metadata snapshot
    for sv in getattr(model, 'state_variables', {}).values():
        n = getattr(sv, 'noise', None)
        if n and getattr(getattr(n, 'intensity', None), 'value', None):
            try:
                if float(n.intensity.value) > 0:
                    return True
            except Exception:
                pass
    return False
%>
% if has_noise(model):
<%include file="/tvbo-julia-SDEProblem.jl.mako" args="model=model, duration=duration" />
% else:
<%include file="/tvbo-julia-model.jl.mako" args="model=model" />
<%include file="/tvbo-julia-ODEProblem.jl.mako" args="model=model, duration=duration" />
% endif

# Solve either deterministic ODE or stochastic SDE.
sol = if prob isa ODEProblem
    solve(prob, Tsit5(); saveat=${dt})
elseif prob isa SDEProblem
    solve(prob, EulerHeun(); dt=${dt}, saveat=${dt})
else
    solve(prob; saveat=${dt})
end

%if plot:
# Plot the solution
using Plots
plot(
    sol,
    linewidth = 5,
    title = "Solution to ${model.metadata.name} ODE",
    xaxis = "Time (t)",
    yaxis = "u(t) (units)",
    label = "Simulation"
)
%endif

%if fout:

%endif
