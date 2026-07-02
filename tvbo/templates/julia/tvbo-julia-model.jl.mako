## -*- coding: utf-8 -*-
<%page args="model"/>
<%!
from tvbo.codegen import render_expression
import re

# Julia special functions that require `using SpecialFunctions`
_SPECIAL_FUNCS = {'erf', 'erfc', 'erfi', 'erfcx', 'lgamma',
                  'digamma', 'beta', 'lbeta', 'besselj', 'bessely'}
%>
<%
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_terms.keys()) if model.coupling_terms else []
dv_names = list(model.derived_variables.keys()) if model.derived_variables else []
dp_names = list(model.derived_parameters.keys()) if model.derived_parameters else []
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
func_names = {str(fname): str(fname) for fname in (getattr(model, 'functions', None) or {}).keys()}
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols, user_functions=func_names)
n_sv = len(sv_names)
# Mode axis: multi-mode models (ReducedSet*, StefanescuJirsa*) carry each state
# variable as a length-n_modes block in the flat state vector, so the dfun operates
# on per-mode vectors (mode_dot/mode_sum + broadcasting) and writes vector slices.
n_modes = getattr(model, 'number_of_modes', 1) or 1

# Detect if any equation uses special functions (erfc, erf, etc.)
_all_rhs = []
for sv in model.state_variables.values():
    _all_rhs.append(str(sv.equation.rhs))
for dv in (model.derived_variables or {}).values():
    _all_rhs.append(str(dv.equation.rhs))
for dp in (model.derived_parameters or {}).values():
    _all_rhs.append(str(dp.equation.rhs))
_all_rhs_str = ' '.join(_all_rhs)
_needs_special = any(re.search(rf'\b{fn}\s*\(', _all_rhs_str) for fn in _SPECIAL_FUNCS)
# Piecewise branches are evaluated eagerly; the Julia printer routes domain-restricted
# powers in them through NaNMath (NaN instead of DomainError, matching numpy/JAX).
_needs_nanmath = 'Piecewise' in _all_rhs_str

# Rename function argument 'x' if any state variable is named 'x'
_arg_x = '_x' if 'x' in sv_names else 'x'
%>
% if _needs_special:
using SpecialFunctions
% endif
% if _needs_nanmath:
import NaNMath
% endif

<%
_all_pnames = param_names + ct_names
_pnames_str = ", ".join(_all_pnames) + ("," if len(_all_pnames) == 1 else "")
%>
function ${model.name}!(dx, ${_arg_x}, p, t = 0)

    (;${_pnames_str}) = p

% if n_modes > 1:
    ## Each state variable is a length-n_modes block (the mode axis).
    % for i, name in enumerate(sv_names):
    ${name} = @view ${_arg_x}[${i * n_modes + 1}:${(i + 1) * n_modes}]
    % endfor
% elif n_sv == 1:
    ${sv_names[0]} = ${_arg_x}[1]
% else:
    ${", ".join(sv_names)} = ${_arg_x}
% endif

    ## Model function definitions (e.g. Sigm)
    % for fname, fdef in (model.functions or {}).items():
<%
    fargs = [str(name) for name in fdef.arguments]  # arguments keyed by name
    fbody = juliacode(fdef.equation.rhs)
%>\
    ${fname}(${", ".join(fargs)}) = ${fbody}
    % endfor

    ## Derived parameters
    % for dp in (model.derived_parameters or {}).values():
    ${dp.name} = ${juliacode(dp.equation.rhs)}
    % endfor

    ## Derived variables (with conditional / ifelse support)
    % for dv in (model.derived_variables or {}).values():
    % if getattr(dv, 'conditional', False) and getattr(dv, 'cases', None):
<%
    cases = list(dv.cases)
    parts = []
    for case in cases:
        cond_str = str(case.condition).strip()
        eq_rhs = juliacode(case.equation.rhs)
        if cond_str.lower() == 'true':
            parts.append(eq_rhs)
        else:
            parts.append((cond_str, eq_rhs))
    def build_ifelse(parts):
        if len(parts) == 1:
            return parts[0] if isinstance(parts[0], str) else parts[0][1]
        cond, val = parts[0]
        return f"ifelse({cond}, {val}, {build_ifelse(parts[1:])})"
    ifelse_expr = build_ifelse(parts)
%>\
    ${dv.name} = ${ifelse_expr}
    % else:
    ${dv.name} = ${juliacode(dv.equation.rhs)}
    % endif
    % endfor

    ## State variable derivatives
    % for i, sv in enumerate(model.state_variables.values()):
    % if n_modes > 1:
    dx[${i * n_modes + 1}:${(i + 1) * n_modes}] .= ${juliacode(sv.equation.rhs)}
    % else:
    dx[${i+1}] = ${juliacode(sv.equation.rhs)}
    % endif
    % endfor
    dx
end

<%
_pval_parts = [f"{p.name} = {p.value}" for p in model.parameters.values()] + [f"{ct} = 0.0" for ct in ct_names]
_pval_str = ", ".join(_pval_parts) + ("," if len(_pval_parts) == 1 else "")
%>
# Parameter values
p = (${_pval_str})
