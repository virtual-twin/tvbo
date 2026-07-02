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

# Parameter values
p = (${mc['param_values']})
