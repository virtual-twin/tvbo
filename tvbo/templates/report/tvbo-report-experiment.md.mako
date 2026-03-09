<%doc>
Publication-Ready Simulation Experiment Report Template
========================================================

Generates a comprehensive methods section suitable for direct inclusion in
scientific publications. All equations are rendered in LaTeX, parameters are
tabulated, and the text follows standard neuroscience modeling conventions.

Sections (auto-numbered, conditional):
1. Brain Network (connectivity, coupling functions)
2. Local Dynamics (state equations, derived variables, parameters)
3. Numerical Integration (solver, noise)
4. Observations (primary, derived, pipelines)
5. Functions (experiment-level analysis functions)
6. Algorithms (iterative tuning: FIC, EIB, ...)
7. Optimization (gradient-based fitting: loss, stages)
8. Explorations (parameter sweeps)
+ Stimulation, Execution, References (if present)

Context Variables:
- experiment: SimulationExperiment instance
- derivative_notation: 'dot' or 'd' (default: 'dot')
</%doc>
<%
from sympy import latex, Eq, Symbol
from tvbo.export import report
from tvbo.parse.expression import parse_eq

# ── Short-hands ──
exp = experiment
model = getattr(exp, 'dynamics', None)
integ = getattr(exp, 'integration', None)
net = getattr(exp, 'network', None) or getattr(exp, 'connectivity', None)
stim = getattr(exp, 'stimulation', None)

# Coupling: experiment-level or from network
cpl = getattr(exp, 'coupling', None)
net_couplings = {}
if net and hasattr(net, 'coupling') and net.coupling:
    if isinstance(net.coupling, dict):
        net_couplings = net.coupling
    elif isinstance(net.coupling, list):
        net_couplings = {getattr(c, 'name', f'coupling_{i}'): c for i, c in enumerate(net.coupling)}

def _to_dict(obj):
    """Normalize None / list / dict into a dict keyed by .name."""
    if not obj:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {getattr(item, 'name', f'item_{i}'): item for i, item in enumerate(obj)}
    return {}

observations = _to_dict(getattr(exp, 'observations', None))
derived_observations = _to_dict(getattr(exp, 'derived_observations', None))
functions = _to_dict(getattr(exp, 'functions', None))
optimizations = _to_dict(getattr(exp, 'optimization', None))
explorations = _to_dict(getattr(exp, 'explorations', None))
algorithms = _to_dict(getattr(exp, 'algorithms', None))
execution = getattr(exp, 'execution', None)
references = getattr(exp, 'references', []) or []

derivative_notation = context.get('derivative_notation', 'dot')

# ── Auto-numbering ──
_sec = [0]
def sec():
    _sec[0] += 1
    return _sec[0]

# ── Safe attribute access ──
def _p(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

# ── Collect all known symbols for equation parsing ──
all_param_names = []
if model:
    for coll in ('parameters', 'state_variables', 'derived_variables', 'derived_parameters', 'coupling_terms'):
        obj = getattr(model, coll, None)
        if obj:
            if isinstance(obj, dict):
                all_param_names.extend(obj.keys())
            elif hasattr(obj, 'values'):
                all_param_names.extend([x.name for x in obj.values()])

# Add coupling parameter names
for cpl_obj in ([cpl] if cpl else []) + list(net_couplings.values()):
    if cpl_obj:
        cpl_p = getattr(cpl_obj, 'parameters', None)
        if isinstance(cpl_p, dict):
            all_param_names.extend(cpl_p.keys())
        elif hasattr(cpl_p, 'values') and cpl_p:
            all_param_names.extend([_p(p, 'name', '') for p in cpl_p.values()])

# Global coupling strength
if net:
    gcs = _p(net, 'global_coupling_strength', None)
    if gcs:
        all_param_names.append(_p(gcs, 'name', 'G'))

# Collect function names (experiment-level + model-level)
all_func_names = [_p(f, 'name', '') for f in functions.values() if _p(f, 'name', '')]
if model and hasattr(model, 'functions') and model.functions:
    all_func_names.extend([f.name for f in model.functions.values()])

# ── LaTeX helpers ──
def safe_latex(expr_str, extra_symbols=None):
    """Safely convert expression string to LaTeX."""
    if not expr_str:
        return ''
    try:
        syms = list(all_param_names) + (extra_symbols or [])
        parsed = parse_eq(str(expr_str), parameters=syms, functions=all_func_names)
        return latex(parsed, mul_symbol='dot')
    except Exception:
        return str(expr_str)

def eq_latex(lhs, rhs, extra_symbols=None):
    """Create LaTeX equation lhs = rhs."""
    if not rhs:
        return ''
    try:
        syms = list(all_param_names) + (extra_symbols or [])
        lhs_sym = Symbol(lhs) if isinstance(lhs, str) else lhs
        rhs_parsed = parse_eq(str(rhs), parameters=syms, functions=all_func_names)
        return latex(Eq(lhs_sym, rhs_parsed), mul_symbol='dot')
    except Exception:
        return f"{lhs} = {rhs}"

def deriv_latex(var_name, rhs, extra_symbols=None):
    """Create LaTeX for a derivative equation."""
    if not rhs:
        return ''
    try:
        syms = list(all_param_names) + (extra_symbols or [])
        rhs_parsed = parse_eq(str(rhs), parameters=syms, functions=all_func_names)
        var_latex = latex(Symbol(var_name))
        if derivative_notation == 'dot':
            return f"\\dot{{{var_latex}}} = {latex(rhs_parsed, mul_symbol='dot')}"
        return f"\\frac{{d{var_latex}}}{{dt}} = {latex(rhs_parsed, mul_symbol='dot')}"
    except Exception:
        return f"d{var_name}/dt = {rhs}"

def _get_func_args(func):
    """Extract argument name list from a Function object."""
    func_args = getattr(func, 'arguments', [])
    if hasattr(func_args, 'values'):
        func_args = list(func_args.values())
    elif not isinstance(func_args, list):
        func_args = list(func_args) if func_args else []
    return [a.name if hasattr(a, 'name') else str(a) for a in func_args]

def _pipeline_str(pipeline):
    """Render a pipeline as arrow-separated function names."""
    if not pipeline:
        return '—'
    if not isinstance(pipeline, list):
        pipeline = [pipeline]
    names = []
    for step in pipeline:
        fn = _p(step, 'function', None) or _p(_p(step, 'callable', None), 'name', None) or _p(_p(step, 'class_call', None), 'name', None)
        names.append(str(fn) if fn else '?')
    return ' → '.join(names)
%>
# ${exp.label or 'Simulation Experiment'}

% if getattr(exp, 'description', None):
${exp.description}

% endif
---

<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: BRAIN NETWORK
# ═══════════════════════════════════════════════════════════════════
%>\
% if net:
<%
s = sec()
n_regions = _p(net, 'number_of_nodes', None) or _p(net, 'number_of_regions', None)
cond_speed = _p(net, 'conduction_speed', None)
gcs = _p(net, 'global_coupling_strength', None)
norm = _p(net, 'normalization', None)
net_label = _p(net, 'label', '')
net_desc = _p(net, 'description', '')
parcellation = _p(net, 'parcellation', None)
structural = _p(net, 'structural_measures', []) or []
observational = _p(net, 'observational_measures', []) or []
%>\

${'##'} ${s}. Brain Network${ ': ' + net_label if net_label else ''}

% if net_desc:
${net_desc.strip()}

% endif
% if parcellation:
<%
parc_label = _p(parcellation, 'label', '') or _p(parcellation, 'name', '')
parc_atlas = _p(parcellation, 'atlas', '')
%>\
% if parc_label:
- **Parcellation:** ${parc_label}${ ' (' + parc_atlas.name + ')' if parc_atlas else ''}
% endif
% endif
% if n_regions:
- **Regions:** ${n_regions}
% endif
% if cond_speed:
- **Conduction velocity:** ${_p(cond_speed, 'value', '?')} ${_p(cond_speed, 'unit', 'mm/ms') or 'mm/ms'}
% endif
% if gcs:
- **Global coupling strength:** $${latex(Symbol(_p(gcs, 'name', 'G')))} = ${_p(gcs, 'value', '?')}$
% endif
% if norm and hasattr(norm, 'rhs') and norm.rhs:
- **Normalization:** $W_{\text{norm}} = ${safe_latex(str(norm.rhs), ['W', 'W_max', 'W_min', 'max', 'min'])}$
% endif
% if structural:
- **Structural measures:** ${', '.join(structural) if isinstance(structural, list) else structural}
% endif
% if observational:
- **Observational measures:** ${', '.join(observational) if isinstance(observational, list) else observational}
% endif

<%
# Coupling functions (from network or experiment-level)
cpl_list = list(net_couplings.items()) if net_couplings else ([(cpl.name, cpl)] if cpl and hasattr(cpl, 'name') else [])
%>\
% for cpl_name, cpl_obj in cpl_list:
<%
cpl_label = _p(cpl_obj, 'label', cpl_name)
cpl_desc = _p(cpl_obj, 'description', '')
cpl_delayed = _p(cpl_obj, 'delayed', False)

incoming = _p(cpl_obj, 'incoming_states', [])
if isinstance(incoming, str):
    incoming = [incoming]
incoming = list(incoming) if incoming else []

cpl_params_obj = _p(cpl_obj, 'parameters', {})
if isinstance(cpl_params_obj, dict):
    cpl_items = list(cpl_params_obj.items())
elif hasattr(cpl_params_obj, 'items'):
    cpl_items = list(cpl_params_obj.items())
else:
    cpl_items = [(getattr(p, 'name', '?'), p) for p in (list(cpl_params_obj) if cpl_params_obj else [])]
cpl_pnames = [n for n, _ in cpl_items]
%>\

${'###'} Coupling: ${cpl_label or cpl_name}

% if cpl_desc:
${cpl_desc.strip()}

% endif
% if incoming:
Receives ${ ', '.join(['$' + latex(Symbol(s)) + '$' for s in incoming])} from connected regions\
% if cpl_delayed:
 with conduction delays\
% endif
.

% endif
% if hasattr(cpl_obj, 'pre_expression') and cpl_obj.pre_expression:
<%
pre_rhs = getattr(cpl_obj.pre_expression, 'rhs', str(cpl_obj.pre_expression))
%>
**Pre-synaptic:** $c_{\text{pre}} = ${safe_latex(str(pre_rhs), cpl_pnames + incoming)}$

% endif
% if hasattr(cpl_obj, 'post_expression') and cpl_obj.post_expression:
<%
post_rhs = getattr(cpl_obj.post_expression, 'rhs', str(cpl_obj.post_expression))
%>
**Post-synaptic:** $c_{\text{post}} = ${safe_latex(str(post_rhs), cpl_pnames + ['gx'])}$

% endif
% if cpl_items:

| Parameter | Value | Description |
|:----------|------:|:------------|
% for pname, param in cpl_items:
<% pval = _p(param, 'value', '—'); pdesc = _p(param, 'description', '') or ''; pfree = _p(param, 'free', False); phet = _p(param, 'heterogeneous', False); tags = []; tags.append('free') if pfree else None; tags.append('heterogeneous') if phet else None; tag_str = f" *({', '.join(tags)})*" if tags else '' %>\
| $${latex(Symbol(pname))}$ | ${pval} | ${pdesc}${tag_str} |
% endfor
% endif

% endfor
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: LOCAL DYNAMICS
# ═══════════════════════════════════════════════════════════════════
%>\
% if model:
<%
s = sec()
model_name = _p(model, 'name', 'Neural Mass Model')
model_label = _p(model, 'label', '') or model_name
model_desc = _p(model, 'description', '')
svars = getattr(model, 'state_variables', {}) or {}
dvars = getattr(model, 'derived_variables', {}) or {}
dparams = getattr(model, 'derived_parameters', {}) or {}
mfuncs = getattr(model, 'functions', {}) or {}
params = getattr(model, 'parameters', {}) or {}
output = getattr(model, 'output', []) or []
%>\

${'##'} ${s}. Local Dynamics: ${model_label}

% if model_desc:
${model_desc.strip().split('.')[0]}.
% endif
% if svars:
The model comprises **${len(svars)} state variables**\
% if dvars:
 and **${len(dvars)} derived variable${'s' if len(dvars) > 1 else ''}**\
% endif
.
% endif

${'###'} ${s}.1 State Equations

% for name, svar in svars.items():
<%
svar_eq = getattr(svar, 'equation', None)
svar_rhs = getattr(svar_eq, 'rhs', '') if svar_eq else ''
%>
$$${deriv_latex(name, svar_rhs)}$$
% endfor

% if dvars or mfuncs:
where

% for name, dvar in dvars.items():
<%
dvar_eq = getattr(dvar, 'equation', None)
dvar_rhs = getattr(dvar_eq, 'rhs', '') if dvar_eq else ''
dvar_desc = _p(dvar, 'description', '')
%>\
$$${eq_latex(name, dvar_rhs)}$$
% if dvar_desc:

*${dvar_desc}*
% endif

% endfor

% for fname, func in mfuncs.items():
<%
func_eq = getattr(func, 'equation', None)
func_rhs = getattr(func_eq, 'rhs', '') if func_eq else ''
func_desc = _p(func, 'definition', '') or _p(func, 'description', '')
arg_names = _get_func_args(func)
%>\
% if func_rhs:
$$${fname}(${', '.join(arg_names)}) = ${safe_latex(func_rhs, arg_names)}$$
% if func_desc:

*${func_desc}*
% endif

% endif
% endfor
% endif

${'###'} ${s}.2 Parameters

| Parameter | Value | Unit | Description |
|:----------|------:|:-----|:------------|
% for name, param in params.items():
| $${latex(Symbol(name))}$ | ${_p(param, 'value', '—')} | ${_p(param, 'unit', '—') or '—'} | ${_p(param, 'description', '') or ''} |
% endfor
% if dparams:
% for name, dp in dparams.items():
<% dp_eq = getattr(dp, 'equation', None); dp_rhs = getattr(dp_eq, 'rhs', '') if dp_eq else ''; dp_desc = _p(dp, 'description', '') or 'Derived' %>\
| $${latex(Symbol(name))}$ | $${safe_latex(dp_rhs)}$ | — | ${dp_desc} |
% endfor
% endif

% if any(_p(p, 'free', False) for p in params.values()):

**Free parameters** (optimized): ${', '.join(['$' + latex(Symbol(name)) + '$' for name, p in params.items() if _p(p, 'free', False)])}
% endif

% if output:
<%
if isinstance(output, dict):
    out_names = list(output.keys())
elif isinstance(output, list):
    out_names = [str(o) for o in output]
else:
    out_names = [str(output)]
%>
**Output:** ${', '.join(['$' + latex(Symbol(n)) + '$' for n in out_names])}
% endif
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: NUMERICAL INTEGRATION
# ═══════════════════════════════════════════════════════════════════
%>\
% if integ:
<%
s = sec()
method = _p(integ, 'method', 'Euler')
dt = _p(integ, 'step_size', 1.0)
duration = _p(integ, 'duration', None)
transient = _p(integ, 'transient_time', 0)
noise = _p(integ, 'noise', None)
%>\

${'##'} ${s}. Numerical Integration

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
noise_nsig = _p(noise, 'nsig', None)
noise_params = _p(noise, 'parameters', {})
if hasattr(noise_params, 'values'):
    noise_param_list = list(noise_params.values())
elif isinstance(noise_params, list):
    noise_param_list = noise_params
else:
    noise_param_list = [noise_params] if noise_params else []
%>
${'Additive' if noise_additive else 'Multiplicative'} Gaussian noise: $d\mathbf{x} = f(\mathbf{x},t)\,dt + \sigma\,d\mathbf{W}_t$

% if noise_nsig is not None:
$\sigma = ${noise_nsig}$
% elif noise_param_list:
| Parameter | Value |
|:----------|------:|
% for np_item in noise_param_list:
| $${latex(Symbol(_p(np_item, 'name', 'σ')))}$ | ${_p(np_item, 'value', '—')} |
% endfor
% endif
% endif
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════
has_obs = bool(observations) or bool(derived_observations)
%>\
% if has_obs:
<%
s = sec()
%>\

${'##'} ${s}. Observations

% if observations:
${'###'} ${s}.1 Primary Observations

| Name | Source | Period | Pipeline |
|:-----|:-------|-------:|:---------|
% for oname, obs in observations.items():
<% obs_label = _p(obs, 'label', oname); obs_source = _p(obs, 'source', None); obs_period = _p(obs, 'period', None); obs_agg = _p(obs, 'aggregation', None); pipeline = _p(obs, 'pipeline', []); period_str = f"{obs_period} ms" if obs_period else (str(obs_agg) if obs_agg else '—'); src_str = '$' + str(obs_source) + '$' if obs_source else '—' %>\
| **${obs_label}** | ${src_str} | ${period_str} | ${_pipeline_str(pipeline)} |
% endfor
% endif

% if derived_observations:
${'###'} ${s}.2 Derived Observations

| Name | Source Observations | Pipeline |
|:-----|:-------------------|:---------|
% for doname, dobs in derived_observations.items():
<% dobs_label = _p(dobs, 'label', doname); src_obs = _p(dobs, 'source_observations', []); src_obs = [src_obs] if isinstance(src_obs, str) else (list(src_obs) if src_obs else []); d_pipeline = _p(dobs, 'pipeline', []) %>\
| **${dobs_label}** | ${', '.join(src_obs)} | ${_pipeline_str(d_pipeline)} |
% endfor
% endif

<%
# Detailed pipeline descriptions
all_obs = list(observations.items()) + list(derived_observations.items())
obs_with_pipeline = [(oname, obs) for oname, obs in all_obs if _p(obs, 'pipeline', None)]
subsec = 3 if derived_observations else 2
%>
% if obs_with_pipeline:
${'###'} ${s}.${subsec} Processing Pipelines

% for oname, obs in obs_with_pipeline:
<%
obs_label = _p(obs, 'label', oname)
pipeline = _p(obs, 'pipeline', [])
if not isinstance(pipeline, list):
    pipeline = [pipeline]
%>\
**${obs_label}:**

| Step | Function | Output | Arguments |
|-----:|:---------|:-------|:----------|
% for i, step in enumerate(pipeline):
<%
fn = _p(step, 'function', None) or _p(_p(step, 'callable', None), 'name', None) or _p(_p(step, 'class_call', None), 'name', None)
step_out = _p(step, 'output', '—')
step_args = _p(step, 'arguments', {})
if hasattr(step_args, 'items'):
    arg_strs = [f"{k}={_p(v, 'value', '?')}" for k, v in step_args.items()]
elif hasattr(step_args, 'values'):
    arg_strs = [f"{_p(a, 'name', '?')}={_p(a, 'value', '?')}" for a in step_args.values()]
elif isinstance(step_args, list):
    arg_strs = [f"{_p(a, 'name', '?')}={_p(a, 'value', '?')}" for a in step_args]
else:
    arg_strs = []
%>\
| ${i+1} | `${fn or '?'}` | ${step_out} | ${', '.join(arg_strs) if arg_strs else '—'} |
% endfor

% endfor
% endif
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: EXPERIMENT-LEVEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
funcs_with_eq = {k: f for k, f in functions.items() if _p(_p(f, 'equation', None), 'rhs', '')}
%>\
% if funcs_with_eq:
<%
s = sec()
%>\

${'##'} ${s}. Functions

% for fname, func in funcs_with_eq.items():
<%
func_desc = _p(func, 'description', '')
if func_desc:
    func_desc = func_desc.strip().split('\n')[0]
func_eq = _p(func, 'equation', None)
func_rhs = getattr(func_eq, 'rhs', '') if func_eq else ''
arg_names = _get_func_args(func)
# Equation-level parameters
eq_params = getattr(func_eq, 'parameters', None) if func_eq else None
if hasattr(eq_params, 'values'):
    eq_params = list(eq_params.values())
elif isinstance(eq_params, dict):
    eq_params = list(eq_params.values())
elif not isinstance(eq_params, list) if eq_params else True:
    eq_params = []
%>
**${fname}**${'  —  ' + func_desc if func_desc else ''}

$$${fname}(${', '.join(arg_names)}) = ${safe_latex(func_rhs, arg_names)}$$

% if eq_params:
| Parameter | Value | Description |
|:----------|------:|:------------|
% for ep in eq_params:
| $${latex(Symbol(_p(ep, 'name', '?')))}$ | ${_p(ep, 'value', '—')} | ${_p(ep, 'description', '') or ''} |
% endfor

% endif
% endfor
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: ALGORITHMS (iterative tuning)
# ═══════════════════════════════════════════════════════════════════
%>\
% if algorithms:
<%
s = sec()
%>\

${'##'} ${s}. Algorithms

% for algo_idx, (algo_name, algo) in enumerate(algorithms.items()):
<%
algo_label = _p(algo, 'name', algo_name)
algo_desc = _p(algo, 'description', '')
algo_type = _p(algo, 'type', '')
n_iter = _p(algo, 'n_iterations', None)
sim_period = _p(algo, 'simulation_period', None)
lr = _p(algo, 'learning_rate', None)
lr_warmup = _p(algo, 'learning_rate_warmup', False)
lr_schedule = _p(algo, 'learning_rate_schedule', None)
apply_every = _p(algo, 'apply_every', 1)

objective = _p(algo, 'objective', None)

update_rules = _p(algo, 'update_rules', {})
if hasattr(update_rules, 'items'):
    ur_items = list(update_rules.items())
elif hasattr(update_rules, 'values'):
    ur_items = [(r.name, r) for r in update_rules.values()]
elif isinstance(update_rules, list):
    ur_items = [(_p(r, 'name', f'rule_{i}'), r) for i, r in enumerate(update_rules)]
else:
    ur_items = []

hypers = _p(algo, 'hyperparameters', {})
if hasattr(hypers, 'items'):
    hp_items = list(hypers.items())
elif hasattr(hypers, 'values'):
    hp_items = [(_p(h, 'name', '?'), h) for h in hypers.values()]
elif isinstance(hypers, list):
    hp_items = [(_p(h, 'name', '?'), h) for h in hypers]
else:
    hp_items = []

depends = _p(algo, 'depends_on', [])
if isinstance(depends, str):
    depends = [depends]
depends = list(depends) if depends else []

includes = _p(algo, 'includes', [])
if hasattr(includes, 'values'):
    includes = list(includes.values())
elif not isinstance(includes, list):
    includes = list(includes) if includes else []

algo_obs = _p(algo, 'observations', [])
if isinstance(algo_obs, str):
    algo_obs = [algo_obs]
algo_obs = list(algo_obs) if algo_obs else []
%>\

${'###'} ${s}.${algo_idx + 1} ${algo_label}

% if algo_desc:
${algo_desc.strip()}

% endif
% if objective:
<%
obj_desc = _p(objective, 'description', '') or _p(objective, 'label', '')
obj_type = _p(objective, 'type', '')
obj_target_var = _p(objective, 'target_variable', None)
obj_target_val = _p(objective, 'target_value', None)
obj_target_data = _p(objective, 'target_data', None)
obj_metric = _p(objective, 'metric', None)
obj_metric_rhs = getattr(obj_metric, 'rhs', '') if obj_metric else ''
%>
**Objective:** ${obj_desc or obj_type}\
% if obj_target_var:
 — target $${latex(Symbol(str(obj_target_var)))}$\
% if obj_target_val is not None:
 $= ${obj_target_val}$\
% endif
% endif
% if obj_target_data:
 (data: ${obj_target_data})\
% endif

% if obj_metric_rhs:
$$\text{metric} = ${safe_latex(obj_metric_rhs)}$$
% endif

% endif
% if depends:
**Depends on:** ${', '.join([str(d) for d in depends])}

% endif
% if includes:
**Includes:** ${', '.join([str(_p(inc, 'algorithm', '?')) for inc in includes])}

% endif
% if algo_obs:
**Observations:** ${', '.join([str(o) for o in algo_obs])}

% endif
% if ur_items:
**Update Rules:**

% for rule_name, rule in ur_items:
<%
rule_eq = _p(rule, 'equation', None)
rule_rhs = getattr(rule_eq, 'rhs', '') if rule_eq else ''
rule_desc = _p(rule, 'description', '')
target_param = _p(rule, 'target_parameter', None)
tp_name = _p(target_param, 'name', '?') if target_param else '?'
bounds = _p(rule, 'bounds', None)
requires = _p(rule, 'requires', [])
if isinstance(requires, str):
    requires = [requires]
requires = list(requires) if requires else []
warmup = _p(rule, 'warmup', None)
# Collect all hyperparameter names for parsing update rule equations
hp_names = [n for n, _ in hp_items]
%>
${'####'} ${rule_name}${'  —  ' + rule_desc if rule_desc else ''}

$$${latex(Symbol(tp_name))} \leftarrow ${safe_latex(rule_rhs, hp_names + [str(r) for r in requires])}$$

% if bounds:
Bounds: $[${_p(bounds, 'lo', '-\\infty')},\\ ${_p(bounds, 'hi', '\\infty')}]$\
% if warmup:
, with warm-up\
% endif

% endif
% if requires:
Requires: ${', '.join([str(r) for r in requires])}

% endif
% endfor
% endif

| Setting | Value |
|:--------|------:|
% if n_iter:
| Iterations | ${n_iter} |
% endif
% if sim_period:
| Simulation period | ${sim_period} ms |
% endif
% if lr:
| Learning rate | ${lr} |
% endif
% if lr_schedule:
| LR schedule | ${lr_schedule} |
% endif
% if apply_every and apply_every != 1:
| Apply every | ${apply_every} steps |
% endif
% for hp_name, hp in hp_items:
| ${hp_name} | ${_p(hp, 'value', '—')} |
% endfor

% endfor
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: OPTIMIZATION (gradient-based fitting)
# ═══════════════════════════════════════════════════════════════════
%>\
% if optimizations:
<%
s = sec()
%>\

${'##'} ${s}. Optimization

% for opt_idx, (opt_name, opt) in enumerate(optimizations.items()):
<%
opt_label = _p(opt, 'label', opt_name)
opt_desc = _p(opt, 'description', '')
depends = _p(opt, 'depends_on', None)

# Loss function
loss_obj = _p(opt, 'loss', None)
loss_fn = _p(loss_obj, 'function', None) if loss_obj else None
loss_callable = _p(loss_obj, 'callable', None) if loss_obj else None
loss_callable_name = _p(loss_callable, 'name', None) if loss_callable else None
loss_name = str(loss_fn or loss_callable_name or '')
loss_args = _p(loss_obj, 'arguments', {}) if loss_obj else {}
if hasattr(loss_args, 'items'):
    loss_arg_items = list(loss_args.items())
elif hasattr(loss_args, 'values'):
    loss_arg_items = [(_p(a, 'name', '?'), a) for a in loss_args.values()]
elif isinstance(loss_args, list):
    loss_arg_items = [(_p(a, 'name', '?'), a) for a in loss_args]
else:
    loss_arg_items = []

# Stages
stages = _p(opt, 'stages', {})
if hasattr(stages, 'values'):
    stage_items = list(stages.items()) if isinstance(stages, dict) else [(s.name, s) for s in stages.values()]
elif isinstance(stages, list):
    stage_items = [(_p(st, 'name', f'stage_{i}'), st) for i, st in enumerate(stages)]
else:
    stage_items = []

# Top-level parameters (used when no stages)
free_params = _p(opt, 'free_parameters', [])
if isinstance(free_params, str):
    free_params = [free_params]
free_params = list(free_params) if free_params else []
algorithm = _p(opt, 'algorithm', None)
opt_lr = _p(opt, 'learning_rate', None)
max_iter = _p(opt, 'max_iterations', None)
opt_integ = _p(opt, 'integration', None)
%>\

${'###'} ${s}.${opt_idx + 1} ${opt_label}

% if opt_desc:
${opt_desc.strip()}

% endif
% if depends:
**Depends on:** ${depends}

% endif
% if loss_name:
**Loss function:** `${loss_name}`
% if loss_arg_items:

| Argument | Value | Description |
|:---------|:------|:------------|
% for arg_name, arg in loss_arg_items:
| ${arg_name} | ${_p(arg, 'value', '—')} | ${_p(arg, 'description', '') or ''} |
% endfor
% endif

% endif
% if free_params and not stage_items:
**Free parameters:** ${', '.join(['$' + latex(Symbol(str(p))) + '$' for p in free_params])}

% endif
% if (algorithm or opt_lr or max_iter or opt_integ) and not stage_items:

| Setting | Value |
|:--------|------:|
% if algorithm:
| Algorithm | ${algorithm} |
% endif
% if opt_lr:
| Learning rate | ${opt_lr} |
% endif
% if max_iter:
| Max iterations | ${max_iter} |
% endif
% if opt_integ:
| Duration | ${_p(opt_integ, 'duration', '—')} ms |
% endif
% endif

% if stage_items:
${'####'} Stages

% for st_idx, (st_name, stage) in enumerate(stage_items):
<%
s_label = _p(stage, 'label', st_name)
s_desc = _p(stage, 'description', '')
s_free = _p(stage, 'free_parameters', [])
if isinstance(s_free, str):
    s_free = [s_free]
s_free = list(s_free) if s_free else []
s_freeze = _p(stage, 'freeze_parameters', [])
s_freeze = list(s_freeze) if s_freeze else []
s_algo = _p(stage, 'algorithm', 'adam')
s_lr = _p(stage, 'learning_rate', None)
s_max_iter = _p(stage, 'max_iterations', None)
s_warmup = _p(stage, 'warmup_from', None)
s_hypers = _p(stage, 'hyperparameters', {})
if hasattr(s_hypers, 'items'):
    s_hp_items = list(s_hypers.items())
elif hasattr(s_hypers, 'values'):
    s_hp_items = [(_p(h, 'name', '?'), h) for h in s_hypers.values()]
elif isinstance(s_hypers, list):
    s_hp_items = [(_p(h, 'name', '?'), h) for h in s_hypers]
else:
    s_hp_items = []
%>
**Stage ${st_idx + 1}: ${s_label}**${'  —  ' + s_desc if s_desc else ''}

% if s_free:
Free parameters: ${', '.join(['$' + latex(Symbol(str(p))) + '$' for p in s_free])}
% endif
% if s_freeze:
Frozen: ${', '.join(['$' + latex(Symbol(str(p))) + '$' for p in s_freeze])}
% endif

| Setting | Value |
|:--------|------:|
| Algorithm | ${s_algo} |
% if s_lr:
| Learning rate | ${s_lr} |
% endif
% if s_max_iter:
| Max iterations | ${s_max_iter} |
% endif
% if s_warmup:
| Warm-up from | ${s_warmup} |
% endif
% for hp_name, hp in s_hp_items:
| ${hp_name} | ${_p(hp, 'value', '—')} |
% endfor

% endfor
% endif
% endfor
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# SECTION: EXPLORATIONS (parameter sweeps)
# ═══════════════════════════════════════════════════════════════════
%>\
% if explorations:
<%
s = sec()
%>\

${'##'} ${s}. Parameter Explorations

% for expl_idx, (expl_name, expl) in enumerate(explorations.items()):
<%
expl_label = _p(expl, 'label', expl_name)
expl_desc = _p(expl, 'description', '')
expl_mode = _p(expl, 'mode', 'product')
expl_n_par = _p(expl, 'n_parallel', 1)
expl_params = _p(expl, 'parameters', {})
if isinstance(expl_params, dict):
    ep_items = list(expl_params.items())
elif hasattr(expl_params, 'items'):
    ep_items = list(expl_params.items())
else:
    ep_items = [(_p(p, 'name', '?'), p) for p in (list(expl_params) if expl_params else [])]
observable = _p(expl, 'observable', None)
%>\

${'###'} ${s}.${expl_idx + 1} ${expl_label}

% if expl_desc:
${expl_desc.strip()}

% endif
**Mode:** ${expl_mode}

| Parameter | Values / Range | Steps |
|:----------|:---------------|------:|
% for ep_name, ep in ep_items:
<%
domain = _p(ep, 'domain', None)
explored = _p(ep, 'explored_values', None)
if explored:
    vals = [str(v) for v in explored]
    if len(vals) <= 8:
        range_str = '{' + ', '.join(vals) + '}'
    else:
        range_str = f"[{vals[0]}, ..., {vals[-1]}]"
    n = len(vals)
elif domain:
    lo = _p(domain, 'lo', '?')
    hi = _p(domain, 'hi', '?')
    n = _p(domain, 'n', '?')
    log = _p(domain, 'log_scale', False)
    range_str = f"[{lo}, {hi}]" + (' (log)' if log else '')
else:
    range_str = '—'
    n = '—'
%>\
| $${latex(Symbol(ep_name))}$ | ${range_str} | ${n} |
% endfor

% if observable:
<%
obs_fn = _p(observable, 'function', None) or _p(_p(observable, 'callable', None), 'name', None)
%>
**Observable:** `${obs_fn or '?'}`
% endif

% endfor
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# STIMULATION
# ═══════════════════════════════════════════════════════════════════
%>\
% if stim:
<%
s = sec()
stim_eq = _p(stim, 'equation', None)
stim_rhs = getattr(stim_eq, 'rhs', '') if stim_eq else ''
stim_params = _p(stim, 'parameters', {})
if hasattr(stim_params, 'values'):
    stim_param_list = list(stim_params.values())
elif isinstance(stim_params, list):
    stim_param_list = stim_params
else:
    stim_param_list = [stim_params] if stim_params else []
%>\

${'##'} ${s}. Stimulation Protocol

% if stim_rhs:
$$I_{\text{stim}}(t) = ${safe_latex(stim_rhs, [_p(p, 'name', '') for p in stim_param_list])}$$
% endif

% if stim_param_list:
| Parameter | Value | Unit | Description |
|:----------|------:|:-----|:------------|
% for param in stim_param_list:
| $${latex(Symbol(_p(param, 'name', '?')))}$ | ${_p(param, 'value', '—')} | ${_p(param, 'unit', '—') or '—'} | ${_p(param, 'description', '') or ''} |
% endfor
% endif
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# EXECUTION CONFIG
# ═══════════════════════════════════════════════════════════════════
%>\
% if execution:
<%
s = sec()
exec_precision = _p(execution, 'precision', None)
exec_accel = _p(execution, 'accelerator', None)
exec_seed = _p(execution, 'random_seed', None)
exec_workers = _p(execution, 'n_workers', None)
has_exec_info = exec_precision or (exec_accel and exec_accel != 'cpu') or exec_seed is not None or exec_workers
%>\
% if has_exec_info:

${'##'} ${s}. Execution

| Setting | Value |
|:--------|------:|
% if exec_workers:
| Workers | ${exec_workers} |
% endif
% if exec_precision:
| Precision | ${exec_precision} |
% endif
% if exec_accel and exec_accel != 'cpu':
| Accelerator | ${exec_accel} |
% endif
% if exec_seed is not None:
| Random seed | ${exec_seed} |
% endif
% endif
% endif
<%
# ═══════════════════════════════════════════════════════════════════
# REFERENCES
# ═══════════════════════════════════════════════════════════════════
all_refs = []
if model and hasattr(model, 'ontology') and model.ontology:
    model_refs = getattr(model.ontology, 'has_reference', None)
    if model_refs:
        all_refs.extend(list(model_refs))
ref_names = [getattr(r, 'name', None) for r in all_refs if getattr(r, 'name', None)]
%>\
% if ref_names or references:

---

${'##'} References

% if ref_names:
${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
% if references:
% for ref in (references if isinstance(references, list) else [references]):
- ${ref}
% endfor
% endif
% endif
