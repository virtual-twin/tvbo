<%doc>
Publication-Ready Simulation Experiment Report Template
========================================================

Generates a methods section suitable for direct inclusion in scientific publications.
All equations are rendered in LaTeX, parameters are tabulated, and the text follows
standard neuroscience modeling conventions.

Context Variables:
- experiment: SimulationExperiment instance

Output:
- Markdown with LaTeX equations suitable for journal methods sections
</%doc>
<%
from sympy import latex, Eq, symbols, sympify, Symbol, Function, IndexedBase, Idx
from tvbo.export import report
from tvbo.parse.expression import parse_eq

# Short-hands
exp = experiment
model = getattr(exp, 'local_dynamics', None)
integ = getattr(exp, 'integration', None)
net = getattr(exp, 'network', None) or getattr(exp, 'connectivity', None)
cpl = getattr(exp, 'coupling', None)
mons = getattr(exp, 'monitors', []) or []
if isinstance(mons, dict):
    mons = list(mons.values())
stim = getattr(exp, 'stimulation', None)
obs_dict = getattr(exp, 'observations', None) or {}
if hasattr(obs_dict, 'values'):
    observations = list(obs_dict.values())
else:
    observations = list(obs_dict) if obs_dict else []
funcs = getattr(exp, 'functions', None) or {}
if hasattr(funcs, 'values'):
    functions = list(funcs.values())
else:
    functions = list(funcs) if funcs else []
opts = getattr(exp, 'optimization', None) or []
if hasattr(opts, 'values'):
    optimizations = list(opts.values())
else:
    optimizations = list(opts) if opts else []

# Safe attribute access
def _p(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

def safe_latex(expr_str, local_symbols=None, local_funcs=None):
    """Safely convert expression string to LaTeX."""
    if not expr_str:
        return ''
    try:
        local_symbols = local_symbols or []
        local_funcs = local_funcs or all_func_names
        parsed = parse_eq(expr_str, parameters=local_symbols, functions=local_funcs)
        return latex(parsed, mul_symbol='dot')
    except Exception:
        return str(expr_str)

def eq_latex(lhs, rhs, local_symbols=None):
    """Create LaTeX equation from lhs = rhs."""
    if not rhs:
        return ''
    try:
        local_symbols = local_symbols or []
        lhs_sym = Symbol(lhs) if isinstance(lhs, str) else lhs
        rhs_parsed = parse_eq(rhs, parameters=local_symbols)
        return latex(Eq(lhs_sym, rhs_parsed), mul_symbol='dot')
    except Exception:
        return f"{lhs} = {rhs}"

def deriv_latex(var_name, rhs, local_symbols=None):
    """Create LaTeX for derivative equation."""
    if not rhs:
        return ''
    try:
        local_symbols = local_symbols or []
        rhs_parsed = parse_eq(rhs, parameters=local_symbols)
        return f"\\frac{{d{var_name}}}{{dt}} = {latex(rhs_parsed, mul_symbol='dot')}"
    except Exception:
        return f"d{var_name}/dt = {rhs}"

# Collect all parameter names for symbol resolution
all_param_names = []
if model and hasattr(model, 'parameters') and model.parameters:
    all_param_names.extend([p.name for p in model.parameters.values()])
if model and hasattr(model, 'state_variables') and model.state_variables:
    all_param_names.extend([s.name for s in model.state_variables.values()])
if model and hasattr(model, 'derived_variables') and model.derived_variables:
    all_param_names.extend([d.name for d in model.derived_variables.values()])
if cpl and hasattr(cpl, 'parameters') and cpl.parameters:
    all_param_names.extend([p.name for p in cpl.parameters.values()])

# Collect function names for recognition
all_func_names = [_p(f, 'name', '') for f in functions if _p(f, 'name', '')]
all_obs_names = [_p(o, 'name', '') for o in observations if _p(o, 'name', '')]
%>
# ${exp.label or 'Simulation Experiment'}

% if getattr(exp, 'description', None):
${exp.description}

% endif
---

% if model:
<%
model_name = _p(model, 'name', 'Neural Mass Model')
model_desc = _p(model, 'description', '')
%>

## 1. Neural Mass Model: ${model_name}

% if model_desc:
${model_desc.split('.')[0]}.
% endif
% if model.state_variables:
The model comprises **${len(model.state_variables)} state variables** representing neural population activity.
% endif

% if model.derived_variables:

### 1.1 Auxiliary Variables

% for name, dvar in model.derived_variables.items():
<%
dvar_eq = getattr(dvar, 'equation', None)
dvar_rhs = getattr(dvar_eq, 'rhs', '') if dvar_eq else ''
%>
$$${eq_latex(name, dvar_rhs, all_param_names)}$$
% endfor
% endif

### 1.2 State Equations

% for name, svar in model.state_variables.items():
<%
svar_eq = getattr(svar, 'equation', None)
svar_rhs = getattr(svar_eq, 'rhs', '') if svar_eq else ''
%>
$$${deriv_latex(name, svar_rhs, all_param_names)}$$
% endfor

### 1.3 Parameters

| Parameter | Value | Unit | Description |
|:----------|------:|:-----|:------------|
% for name, param in model.parameters.items():
| $${latex(Symbol(name))}$ | ${_p(param, 'value', '—')} | ${_p(param, 'unit', '—') or '—'} | ${_p(param, 'description', '') or ''} |
% endfor

% if any(_p(p, 'free', False) for p in model.parameters.values()):
**Free parameters** (optimized): ${', '.join(['$' + latex(Symbol(name)) + '$' for name, p in model.parameters.items() if _p(p, 'free', False)])}
% endif
% endif

% if cpl:
<%
cpl_name = _p(cpl, 'name', 'Coupling')
cpl_delayed = _p(cpl, 'delayed', False)
incoming = _p(cpl, 'incoming_states', [])
if isinstance(incoming, str):
    incoming = [incoming]
incoming = list(incoming) if incoming else []
cpl_params = list(cpl.parameters.values()) if hasattr(cpl, 'parameters') and cpl.parameters else []
cpl_param_names = [p.name for p in cpl_params]
%>

## 2. Network Coupling: ${cpl_name}

% if incoming:
Coupling function receiving states ${', '.join(['$' + latex(Symbol(s)) + '$' for s in incoming])} from connected regions\
% if cpl_delayed:
 with conduction delays\
% endif
.
% endif

% if hasattr(cpl, 'pre_expression') and cpl.pre_expression:
<%
pre_rhs = getattr(cpl.pre_expression, 'rhs', str(cpl.pre_expression))
%>
**Pre-synaptic transformation:**
$$c_{\text{pre}} = ${safe_latex(pre_rhs, cpl_param_names + incoming)}$$
% endif

% if hasattr(cpl, 'post_expression') and cpl.post_expression:
<%
post_rhs = getattr(cpl.post_expression, 'rhs', str(cpl.post_expression))
%>
**Post-synaptic transformation:**
$$c_{\text{post}} = ${safe_latex(post_rhs, cpl_param_names + ['gx'])}$$
% endif

% if cpl_params:
### Coupling Parameters

| Parameter | Value | Description |
|:----------|------:|:------------|
% for param in cpl_params:
| $${latex(Symbol(param.name))}$ | ${_p(param, 'value', '—')} | ${_p(param, 'description', '') or ''} |
% endfor
% endif
% endif

% if net:
<%
n_regions = _p(net, 'number_of_regions', None)
cond_speed = _p(net, 'conduction_speed', None)
norm = _p(net, 'normalization', None)
net_label = _p(net, 'label', '')
%>
## 3. Brain Network${': ' + net_label if net_label else ''}

% if n_regions:
- **Regions:** ${n_regions}
% endif
% if cond_speed:
- **Conduction velocity:** ${_p(cond_speed, 'value', '')} ${_p(cond_speed, 'unit', 'mm/ms')}
% endif
% if norm and hasattr(norm, 'rhs'):
- **Normalization:** $W_{\text{norm}} = ${safe_latex(norm.rhs, ['W', 'W_max', 'W_min'])}$
% endif
% endif

% if integ:
<%
method = _p(integ, 'method', 'Euler')
dt = _p(integ, 'step_size', 1.0)
duration = _p(integ, 'duration', None)
transient = _p(integ, 'transient_time', 0)
noise = _p(integ, 'noise', None)
%>
## 4. Numerical Integration

- **Method:** ${method}
- **Time step:** $\Delta t = ${dt}$ ms
% if duration:
- **Duration:** ${duration} ms
% endif
% if transient:
- **Transient:** ${transient} ms (discarded)
% endif

% if noise:
<%
noise_additive = _p(noise, 'additive', True)
noise_params = _p(noise, 'parameters', {})
if hasattr(noise_params, 'values'):
    noise_params = list(noise_params.values())
else:
    noise_params = list(noise_params) if noise_params else []
%>
### Stochastic Noise

${'Additive' if noise_additive else 'Multiplicative'} Gaussian noise: $d\mathbf{x} = f(\mathbf{x}, t)\,dt + \sigma\,d\mathbf{W}_t$

% if noise_params:
| Parameter | Value |
|:----------|------:|
% for param in noise_params:
| $${latex(Symbol(param.name))}$ | ${_p(param, 'value', '—')} |
% endfor
% endif
% endif
% endif

<%
# Filter functions with equations only
funcs_with_eq = [f for f in functions if _p(_p(f, 'equation', None), 'rhs', '')]
%>
% if funcs_with_eq:
## 5. Analysis Functions

% for func in funcs_with_eq:
<%
func_name = _p(func, 'name', 'function')
func_desc = _p(func, 'description', '')
func_eq = _p(func, 'equation', None)
func_rhs = getattr(func_eq, 'rhs', '') if func_eq else ''
func_args = _p(func, 'arguments', [])
if hasattr(func_args, 'values'):
    func_args = list(func_args.values())
elif not isinstance(func_args, list):
    func_args = list(func_args) if func_args else []
arg_names = [_p(a, 'name', '') for a in func_args if _p(a, 'name', '')]
%>
**${func_name}**${'  —  ' + func_desc if func_desc else ''}

$$${func_name}(${', '.join(arg_names)}) = ${safe_latex(func_rhs, arg_names)}$$

% endfor
% endif

% if observations:
## 6. Observables

| Observable | Description | Source | Pipeline |
|:-----------|:------------|:-------|:---------|
% for obs in observations:
<%
obs_name = _p(obs, 'name', 'observation')
obs_label = _p(obs, 'label', obs_name)
obs_desc = _p(obs, 'description', '')
if obs_desc and len(obs_desc) > 60:
    obs_desc = obs_desc[:57] + '...'
obs_source = _p(obs, 'source', None)
obs_src_obs = _p(obs, 'source_observation', None)
pipeline = _p(obs, 'pipeline', [])
if hasattr(pipeline, '__iter__') and not isinstance(pipeline, str):
    pipeline = list(pipeline)
else:
    pipeline = []
src_str = ''
if obs_source:
    src_str = '$' + str(_p(obs_source, 'name', obs_source)) + '$'
elif obs_src_obs:
    src_str = str(_p(obs_src_obs, 'name', obs_src_obs))
pipe_str = ' → '.join([_p(s, 'name', '?') for s in pipeline]) if pipeline else '—'
%>
| **${obs_name}** | ${obs_desc} | ${src_str} | ${pipe_str} |
% endfor
% endif

% if optimizations:
## 7. Optimization

% for opt in optimizations:
<%
opt_name = _p(opt, 'name', 'optimization')
opt_label = _p(opt, 'label', opt_name)
opt_desc = _p(opt, 'description', '')
free_params = _p(opt, 'free_parameters', [])
if hasattr(free_params, '__iter__') and not isinstance(free_params, str):
    free_params = list(free_params)
else:
    free_params = [free_params] if free_params else []
loss_obj = _p(opt, 'loss', None)
loss_eq = _p(loss_obj, 'equation', None) if loss_obj else None
loss_rhs = getattr(loss_eq, 'rhs', '') if loss_eq else getattr(loss_obj, 'rhs', '') if loss_obj else ''
algorithm = _p(opt, 'algorithm', None)
lr = _p(opt, 'learning_rate', None)
max_iter = _p(opt, 'max_iterations', None)
%>
### ${opt_label}

% if opt_desc:
${opt_desc}

% endif
% if free_params:
**Free parameters:** ${', '.join(['$' + latex(Symbol(str(p))) + '$' for p in free_params])}
% endif

% if loss_rhs:
**Loss function:**
$$\mathcal{L} = ${safe_latex(loss_rhs, [str(p) for p in free_params], all_func_names + all_obs_names)}$$
% endif

% if algorithm or lr or max_iter:
| Setting | Value |
|:--------|------:|
% if algorithm:
| Algorithm | ${algorithm} |
% endif
% if lr:
| Learning rate | ${lr} |
% endif
% if max_iter:
| Max iterations | ${max_iter} |
% endif
% endif
% endfor
% endif

% if stim:
<%
stim_eq = _p(stim, 'equation', None)
stim_rhs = getattr(stim_eq, 'rhs', '') if stim_eq else ''
stim_params = _p(stim, 'parameters', {})
if hasattr(stim_params, 'values'):
    stim_params = list(stim_params.values())
else:
    stim_params = list(stim_params) if stim_params else []
%>
## Stimulation Protocol

% if stim_rhs:
$$I_{\text{stim}}(t) = ${safe_latex(stim_rhs, [p.name for p in stim_params])}$$
% endif

% if stim_params:
| Parameter | Value | Unit | Description |
|:----------|------:|:-----|:------------|
% for param in stim_params:
| $${latex(Symbol(param.name))}$ | ${_p(param, 'value', '—')} | ${_p(param, 'unit', '—') or '—'} | ${_p(param, 'description', '') or ''} |
% endfor
% endif
% endif

<%
# Collect references
refs = []
if model and hasattr(model, 'ontology') and model.ontology:
    model_refs = getattr(model.ontology, 'has_reference', None)
    if model_refs:
        refs.extend(list(model_refs))
if hasattr(exp, 'references') and exp.references:
    refs.extend(list(exp.references))
ref_names = [getattr(r, 'name', None) for r in refs if getattr(r, 'name', None)]
%>
% if ref_names:
---

## References

${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
