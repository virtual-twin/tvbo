<%doc>
Base dfun Template - Generates dynamics functions in jax/numpy format.

Signatures:
- 'standard': dfun(current_state, t, cX, _p) - for network simulation
- 'scipy': ModelName(current_state, t, param=val, ...) - for scipy.odeint
</%doc>
<%!
import textwrap
from tvbo.templates.base.utils import get_coupling_terms, get_func_name, get_func_args, np_module, needs_scipy_special, referenced_parameters
%>

## The special-function import is gated on the equations actually using it — emitting it
## unconditionally is what left `jsp` unused in every JAX module. `model=None` cannot be
## checked, so it keeps the old unconditional form.
<%def name="imports(model=None, fmt='jax')">
<% _scipy = needs_scipy_special(model, fmt) if model is not None else True %>\
% if fmt == 'jax':
import jax.numpy as jnp
% if _scipy:
import jax.scipy as jsp
% endif
% else:
import numpy as np
% if _scipy:
import scipy.special
% endif
% endif
</%def>

<%def name="params(model, fmt, source='_p')">
<%
render = lambda obj: model.render_equation(obj, format=fmt)
pnames = referenced_parameters(model, [p.name for p in model.parameters.values()])
%>\
% if source == '_p' and pnames:
# Parameters
% for p in pnames:
${p} = ${source}.${p}
% endfor
% endif
% for dp in (model.derived_parameters or {}).values():
${dp.name} = ${render(dp)}
% endfor
</%def>

<%def name="coupling(model, var='cX')">
<%
_, global_terms, has_local = get_coupling_terms(model)
%>\
% if global_terms or has_local:
# Coupling
% for i, ct in enumerate(global_terms):
${ct} = ${var}[${i}]
% endfor
% if has_local:
local_coupling = 0
% endif
% endif
</%def>

<%def name="state(model)">
# State variables
% for i, sv in enumerate(model.state_variables):
${sv} = current_state[${i}]
% endfor
</%def>

<%def name="funcs(model, fmt)">\
<%
render = lambda obj: model.render_equation(obj, format=fmt)
%>\
% if model.functions:

# Functions
% for f in (model.functions or {}).values():
def ${f.name}(${', '.join(get_func_args(f))}):
    return ${render(f)}
% endfor
% endif
</%def>

<%def name="derived(model, fmt)">\
<%
render = lambda obj: model.render_equation(obj, format=fmt)
%>\
% if model.derived_variables:

# Derived variables
% for dv in (model.derived_variables or {}).values():
${dv.name} = ${render(dv)}
% endfor
% endif
</%def>

<%def name="derivs(model, fmt, stim=None, return_aux=False)">
<%
render = lambda obj: model.render_equation(obj, format=fmt)
svars = list(model.state_variables.values())
dvars = list((model.derived_variables or {}).keys())
np = np_module(fmt)
%>\
# Derivatives
% for sv in svars:
d${sv.name}_dt = ${render(sv)}${(' + ' + stim) if stim and getattr(sv, 'stimulation_variable', False) else ''}
% endfor

## broadcast_arrays so state-independent derivatives (e.g. `duA_dt = cA`, a bare
## constant/param → scalar shape) stack cleanly with state-/coupling-dependent
## ones (broadcast to the node/mode shape). No-op when all shapes already match.
derivatives = ${np}.array(${np}.broadcast_arrays(${', '.join('d' + sv.name + '_dt' for sv in svars)}))
% if return_aux and dvars:
auxiliaries = ${np}.array([${', '.join(dvars)}])

return derivatives, auxiliaries
% else:

return derivatives
% endif
</%def>

<%def name="body(model, fmt, source='_p', with_coupling=True, stim=None, return_aux=False)">
${params(model, fmt, source)}\
% if with_coupling:
${coupling(model)}\
% endif

${state(model)}\
${funcs(model, fmt)}\
${derived(model, fmt)}
${derivs(model, fmt, stim, return_aux)}\
</%def>

<%def name="full_dfun(model, fmt='jax', signature='standard', func_name=None, return_aux=False)">
<%
_, global_terms, has_local = get_coupling_terms(model)
name = get_func_name(model, func_name)
%>\
% if signature == 'standard':
def dfun(current_state, t, cX, _p):
${textwrap.indent(capture(body, model, fmt, '_p', True, None, return_aux).strip(), '    ')}
% elif signature == 'scipy':
def ${name}(
    current_state,
    t,
% for p in model.parameters.values():
    ${p.name}=${p.value if p.value is not None else 0.0},
% endfor
% for ct in global_terms:
    ${ct}=0.0,
% endfor
    stimulus=False,
):
    stim_t = stimulus(t) if stimulus else 0.0
% if has_local:
    local_coupling = 0.0
% endif

${textwrap.indent(capture(body, model, fmt, 'kwargs', False, 'stim_t', return_aux).strip(), '    ')}
% endif
</%def>
