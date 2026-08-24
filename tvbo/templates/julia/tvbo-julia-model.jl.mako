## -*- coding: utf-8 -*-
## Slim emitter: all metadata→Julia translation lives in
## tvbo.adapters.julia_model.build_model_context (passed in as ``mc``).
<%page args="mc"/>
% if mc['needs_special']:
using SpecialFunctions
% endif
% if mc['needs_nanmath']:
import NaNMath
% endif
% if mc.get('network_mode'):
## Network-coupled RHS with n_nodes blocks per state var: long-range coupling is a W·s matvec once per step, and the per-node scalar RHS runs in a loop that reuses the single-node equation emission verbatim rather than broadcasting.
const W_NET = ${mc['w_const']}

function ${mc['func_name']}!(dx, ${mc['arg_x']}, p, t = 0)

    (;${mc['destructure']}) = p
    N = ${mc['n_nodes']}

    ## Model function definitions (e.g. Sigm)
% for fname, fargs, fbody in mc['functions']:
    ${fname}(${", ".join(fargs)}) = ${fbody}
% endfor

    ## Derived parameters (node-independent — computed once)
% for name, rhs in mc['derived_params']:
    ${name} = ${rhs}
% endfor

    ## Long-range coupling matvecs (c = W·s, evaluated once per step)
% for line in mc['coupling_pre']:
    ${line}
% endfor

    @inbounds for i in 1:N
        ## Per-node state
% for line in mc['unpack']:
        ${line}
% endfor
        ## Per-node heterogeneous parameters (e.g. FIC-tuned J_i)
% for line in mc.get('pernode_gather', []):
        ${line}
% endfor
        ## Per-node coupling (gather / local)
% for line in mc['coupling_body']:
        ${line}
% endfor
        ## Derived variables
% for name, rhs in mc['derived_vars']:
        ${name} = ${rhs}
% endfor
        ## State variable derivatives
% for lhs, rhs in mc['dfun']:
        ${lhs} ${rhs}
% endfor
    end
    dx
end
% else:
function ${mc['func_name']}!(dx, ${mc['arg_x']}, p, t = 0)

    (;${mc['destructure']}) = p

% for line in mc['unpack']:
    ${line}
% endfor

    ## Model function definitions (e.g. Sigm)
% for fname, fargs, fbody in mc['functions']:
    ${fname}(${", ".join(fargs)}) = ${fbody}
% endfor

    ## Derived parameters
% for name, rhs in mc['derived_params']:
    ${name} = ${rhs}
% endfor

    ## Derived variables (conditional cases already folded to ifelse)
% for name, rhs in mc['derived_vars']:
    ${name} = ${rhs}
% endfor

    ## State variable derivatives
% for lhs, rhs in mc['dfun']:
    ${lhs} ${rhs}
% endfor
    dx
end
% endif

# Parameter values
p = (${mc['param_values']})
