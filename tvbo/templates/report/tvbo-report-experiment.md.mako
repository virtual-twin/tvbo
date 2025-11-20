<%
from sympy import latex, Eq, symbols, sympify, Symbol, Function
from tvbo.export import report

# Short-hands
exp = experiment
integ = getattr(exp, 'integration', None)
net = getattr(exp, 'network', None) or getattr(exp, 'connectivity', None)
cpl = getattr(exp, 'coupling', None)
mons = getattr(exp, 'monitors', []) or []
# Normalize monitors to a list of monitor objects if a dict was provided
if isinstance(mons, dict):
  mons = list(mons.values())
stim = getattr(exp, 'stimulation', None)
reqs = getattr(exp, 'requirements', None) or []
env  = getattr(exp, 'environment', None)

# Safe helpers

def _p(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

%>

# Simulation Experiment ${exp.id if hasattr(exp, 'id') else ''}
${exp.label or ''}

% if getattr(exp, 'description', None):
${exp.description}
% endif

## Summary

- Duration: ${_p(integ, 'duration', 'N/A')}
- Time step (dt): ${_p(integ, 'step_size', 'N/A')}
- Steps: ${_p(integ, 'steps', 'N/A')}
- Transient time: ${_p(integ, 'transient_time', 0)}
- Delayed coupling: ${_p(cpl, 'delayed', False)}
- Delayed integration: ${_p(integ, 'delayed', False)}

% if net is not None:
- Regions: ${_p(net, 'number_of_regions', 'N/A')}
- Nodes: ${_p(net, 'number_of_nodes', 'N/A')}
% if getattr(net, 'parcellation', None) and getattr(net.parcellation, 'atlas', None):
- Atlas: ${_p(net.parcellation.atlas, 'name', 'N/A')}
% endif
% if getattr(net, 'conduction_speed', None):
- Conduction speed: ${getattr(net.conduction_speed, 'value', 'N/A')} ${getattr(net.conduction_speed, 'unit', '')}
% endif
% endif

% if env or reqs:
## Software
% if env:
- Environment: ${getattr(env, 'label', '')} ${getattr(env, 'version', '')} (${getattr(env, 'platform', '')})
% endif
% if reqs:
- Requirements:
% for r in (reqs if isinstance(reqs, (list, tuple)) else [reqs]):
  - ${getattr(r, 'name', getattr(r, 'label', ''))} ${getattr(r, 'version', '')} ${' | '.join(getattr(r, 'modules', []) or [])}
% endfor
% endif
% endif

## Integration

| Key | Value |
|-----|-------|
| Method | ${_p(integ, 'method', 'N/A')} |
| Step size | ${_p(integ, 'step_size', 'N/A')} |
| Duration | ${_p(integ, 'duration', 'N/A')} |
| Steps | ${_p(integ, 'steps', 'N/A')} |
| Transient | ${_p(integ, 'transient_time', 0)} |
| Delayed | ${_p(integ, 'delayed', False)} |

% if getattr(integ, 'noise', None):
### Noise

- Type: ${getattr(integ.noise, 'noise_type', getattr(integ.noise, 'type', 'gaussian'))}
- Additive: ${getattr(integ.noise, 'additive', True)}; Correlated: ${getattr(integ.noise, 'correlated', False)}
- Seed: ${getattr(integ.noise, 'seed', '')}
% if getattr(integ, 'state_wise_sigma', None):
- State-wise sigma: ${list(getattr(integ, 'state_wise_sigma', []) or [])}
% endif
% if getattr(integ.noise, 'parameters', None):

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
% for p in integ.noise.parameters.values():
| ${p.name} | ${p.value} | ${p.unit if getattr(p, 'unit', None) else ''} | ${p.description or ''} |
% endfor
% endif
% endif

% if cpl is not None:
## Coupling

- Name: ${_p(cpl, 'name', '') or _p(cpl, 'label', '')}
- Delayed: ${_p(cpl, 'delayed', False)}
% if getattr(cpl, 'pre_expression', None) or getattr(cpl, 'post_expression', None):
- Pre: $${getattr(cpl.pre_expression, 'rhs', getattr(cpl, 'pre_expression', ''))}$$
- Post: $${getattr(cpl.post_expression, 'rhs', getattr(cpl, 'post_expression', ''))}$$
% endif

% if getattr(cpl, 'parameters', None):
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
% for p in cpl.parameters.values():
| ${p.name} | ${p.value} | ${p.unit if getattr(p, 'unit', None) else ''} | ${p.description or ''} |
% endfor
% endif
% endif

% if mons:
## Monitors

| Name | Period | Modality | Variable |
|------|--------|----------|----------|
% for m in mons:
| ${getattr(m, 'name', '') or getattr(m, 'label', '')} | ${getattr(m, 'period', '')} | ${getattr(m, 'imaging_modality', '')} | ${getattr(getattr(m, 'select_variable', None), 'name', '')} |
% endfor
% endif

% if stim is not None:
## Stimulation

% if getattr(stim, 'equation', None):
$${getattr(stim.equation, 'rhs', '')}$$
% endif
% if getattr(stim, 'parameters', None):

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
% for p in stim.parameters.values():
| ${p.name} | ${p.value} | ${p.unit if getattr(p, 'unit', None) else ''} | ${p.description or ''} |
% endfor
% endif
% endif

## Local Dynamics

% if experiment.local_dynamics:
<%!
  # Note: include the existing model template to avoid redundancy
%>
<%include file="tvbo-report-model.md.mako"/>
% else:
_No local dynamics specified._
% endif

% if getattr(exp, 'references', None):
## References
${"\n\n".join([report.get_citation(n) for n in exp.references])}
% endif
