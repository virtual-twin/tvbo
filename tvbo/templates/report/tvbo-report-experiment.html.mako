<%
from tvbo.export import report
exp = experiment
model = getattr(exp, 'local_dynamics', None)
integ = getattr(exp, 'integration', None)
net = getattr(exp, 'network', None) or getattr(exp, 'connectivity', None)
cpl = getattr(exp, 'coupling', None)
mons = getattr(exp, 'monitors', []) or []
stim = getattr(exp, 'stimulation', None)
reqs = getattr(exp, 'requirements', None) or []
env  = getattr(exp, 'environment', None)

def _p(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

# Normalize monitors to a list if provided as dict
if isinstance(mons, dict):
  mons = list(mons.values())
%>
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Simulation Experiment ${exp.id if hasattr(exp, 'id') else ''}</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
    h1,h2,h3 { color: #333; }
    table { width: 95%; border-collapse: collapse; margin: 12px 0; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #f4f4f4; }
    ul { margin: 0.5em 0 0.5em 1.25em; }
  </style>
  <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <h1>Simulation Experiment ${exp.id if hasattr(exp, 'id') else ''}</h1>
  % if getattr(exp, 'label', None):
  <h2>${exp.label}</h2>
  % endif
  % if getattr(exp, 'description', None):
  <p>${exp.description}</p>
  % endif

  <h2>Summary</h2>
  <ul>
    <li>Duration: ${_p(integ, 'duration', 'N/A')}</li>
    <li>Time step (dt): ${_p(integ, 'step_size', 'N/A')}</li>
    <li>Steps: ${_p(integ, 'steps', 'N/A')}</li>
    <li>Transient time: ${_p(integ, 'transient_time', 0)}</li>
    <li>Delayed coupling: ${_p(cpl, 'delayed', False)}</li>
    <li>Delayed integration: ${_p(integ, 'delayed', False)}</li>
    % if net is not None:
    <li>Regions: ${_p(net, 'number_of_regions', 'N/A')}</li>
    <li>Nodes: ${_p(net, 'number_of_nodes', 'N/A')}</li>
      % if getattr(net, 'parcellation', None) and getattr(net.parcellation, 'atlas', None):
      <li>Atlas: ${_p(net.parcellation.atlas, 'name', 'N/A')}</li>
      % endif
      % if getattr(net, 'conduction_speed', None):
      <li>Conduction speed: ${getattr(net.conduction_speed, 'value', 'N/A')} ${getattr(net.conduction_speed, 'unit', '')}</li>
      % endif
    % endif
  </ul>

  % if env or reqs:
  <h2>Software</h2>
  % if env:
  <p>Environment: ${getattr(env, 'label', '')} ${getattr(env, 'version', '')} (${getattr(env, 'platform', '')})</p>
  % endif
  % if reqs:
  <p>Requirements:</p>
  <ul>
  % for r in (reqs if isinstance(reqs, (list, tuple)) else [reqs]):
    <li>${getattr(r, 'name', getattr(r, 'label', ''))} ${getattr(r, 'version', '')} ${' | '.join(getattr(r, 'modules', []) or [])}</li>
  % endfor
  </ul>
  % endif
  % endif

  <h2>Integration</h2>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    <tr><td>Method</td><td>${_p(integ, 'method', 'N/A')}</td></tr>
    <tr><td>Step size</td><td>${_p(integ, 'step_size', 'N/A')}</td></tr>
    <tr><td>Duration</td><td>${_p(integ, 'duration', 'N/A')}</td></tr>
    <tr><td>Steps</td><td>${_p(integ, 'steps', 'N/A')}</td></tr>
    <tr><td>Transient</td><td>${_p(integ, 'transient_time', 0)}</td></tr>
    <tr><td>Delayed</td><td>${_p(integ, 'delayed', False)}</td></tr>
  </table>

  % if getattr(integ, 'noise', None):
  <h3>Noise</h3>
  <ul>
    <li>Type: ${getattr(integ.noise, 'noise_type', getattr(integ.noise, 'type', 'gaussian'))}</li>
    <li>Additive: ${getattr(integ.noise, 'additive', True)}; Correlated: ${getattr(integ.noise, 'correlated', False)}</li>
    <li>Seed: ${getattr(integ.noise, 'seed', '')}</li>
    % if getattr(integ, 'state_wise_sigma', None):
    <li>State-wise sigma: ${list(getattr(integ, 'state_wise_sigma', []) or [])}</li>
    % endif
  </ul>
  % if getattr(integ.noise, 'parameters', None):
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Description</th></tr>
    % for p in integ.noise.parameters.values():
    <tr><td>${p.name}</td><td>${p.value}</td><td>${p.unit if getattr(p, 'unit', None) else ''}</td><td>${p.description or ''}</td></tr>
    % endfor
  </table>
  % endif
  % endif

  % if cpl is not None:
  <h2>Coupling</h2>
  <p>Name: ${_p(cpl, 'name', '') or _p(cpl, 'label', '')} &mdash; Delayed: ${_p(cpl, 'delayed', False)}</p>
  % if getattr(cpl, 'pre_expression', None) or getattr(cpl, 'post_expression', None):
  <p>Pre: \(${getattr(cpl.pre_expression, 'rhs', getattr(cpl, 'pre_expression', ''))}\)<br/>
     Post: \(${getattr(cpl.post_expression, 'rhs', getattr(cpl, 'post_expression', ''))}\)</p>
  % endif
  % if getattr(cpl, 'parameters', None):
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Description</th></tr>
    % for p in cpl.parameters.values():
    <tr><td>${p.name}</td><td>${p.value}</td><td>${p.unit if getattr(p, 'unit', None) else ''}</td><td>${p.description or ''}</td></tr>
    % endfor
  </table>
  % endif
  % endif

  % if mons:
  <h2>Monitors</h2>
  <table>
    <tr><th>Name</th><th>Period</th><th>Modality</th><th>Variable</th></tr>
    % for m in mons:
    <tr><td>${getattr(m, 'name', '') or getattr(m, 'label', '')}</td><td>${getattr(m, 'period', '')}</td><td>${getattr(m, 'imaging_modality', '')}</td><td>${getattr(getattr(m, 'select_variable', None), 'name', '')}</td></tr>
    % endfor
  </table>
  % endif

  % if stim is not None:
  <h2>Stimulation</h2>
  % if getattr(stim, 'equation', None):
  <p>\(${getattr(stim.equation, 'rhs', '')}\)</p>
  % endif
  % if getattr(stim, 'parameters', None):
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Description</th></tr>
    % for p in stim.parameters.values():
    <tr><td>${p.name}</td><td>${p.value}</td><td>${p.unit if getattr(p, 'unit', None) else ''}</td><td>${p.description or ''}</td></tr>
    % endfor
  </table>
  % endif
  % endif

  <h2>Local Dynamics</h2>
  % if model is not None:
  <%include file="tvbo-report-model.html.mako" args="model=model"/>
  % else:
  <p><em>No local dynamics specified.</em></p>
  % endif

  % if getattr(exp, 'references', None):
  <h2>References</h2>
  % for n in exp.references:
    <p>${report.get_citation(n)}</p>
  % endfor
  % endif
</body>
</html>
