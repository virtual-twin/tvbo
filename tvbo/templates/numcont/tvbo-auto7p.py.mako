<%
import re
from tvbo.codegen.code import render_equation as render_eq
from tvbo.classes.equation import _clash1
model = context['model']
params = model.parameters.values()

# Collect all symbol names so the parser recognizes them as Symbols
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_terms.keys()) if model.coupling_terms else []
dv_names = list(model.derived_variables.keys()) if model.derived_variables else []
dp_names = list(model.derived_parameters.keys()) if model.derived_parameters else []
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names

# For single-node bifurcation analysis, all coupling inputs must be zeroed
# out (they are otherwise undeclared Fortran identifiers). Use coupling_inputs
# (canonical) and fall back to coupling_terms (deprecated) for old models.
ci_names = list(model.coupling_inputs.keys()) if model.coupling_inputs else []
coupling_zero = list({*ci_names, *ct_names})

replace = {
    p.name: (p.name + 'low' if p.name[0].islower() and p.name in [n.name.lower() for n in params if n.name != p.name] else p.name)
    for p in params
}
%>
SUBROUTINE FUNC(NDIM, U, ICP, PAR, IJAC, F, DFDU, DFDP)

    IMPLICIT NONE

    INTEGER NDIM, IJAC, ICP(*)
    DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
    DOUBLE PRECISION ${",".join([sv.name for sv in model.state_variables.values()])}
    DOUBLE PRECISION ${", ".join([f"{replace[p.name]}" for p in model.parameters.values()])}
% if model.derived_parameters:
    DOUBLE PRECISION ${", ".join([f"{dp.name}" for dp in model.derived_parameters.values()])}
% endif
% if model.derived_variables:
    DOUBLE PRECISION ${", ".join([f"{k}" for k in model.derived_variables.keys()])}
% endif

    % for i, p in enumerate(model.parameters.values()):
    ${replace[p.name]} = PAR(${i+1 if i+1 <= 10 else i+3})
    % endfor

% if model.derived_parameters:
    % for dp in model.derived_parameters.values():
    ${dp.name} = ${render_eq(dp.equation, format='fortran', replace=replace, parameters=all_symbols)}
    % endfor
% endif

    % for i, sv in enumerate(model.state_variables.values()):
    ${sv.name} = U(${i+1})
    % endfor

% if model.derived_variables:
    % for k,v in model.derived_variables.items():
    ${k} = ${render_eq(v.equation, user_functions={f:f for f in model.functions.keys()}, format='fortran', replace=replace, parameters=all_symbols, remove=coupling_zero)}
    % endfor
% endif

    % for i, sv in enumerate(model.state_variables.values()):
    F(${i+1}) = ${render_eq(sv.equation, user_functions={f:f for f in model.functions.keys()}, format='fortran', replace=replace, parameters=all_symbols, remove=coupling_zero)}
    % endfor

END SUBROUTINE FUNC

!----------------------------------------------------------------------
!----------------------------------------------------------------------


SUBROUTINE STPNT
END SUBROUTINE STPNT

SUBROUTINE BCND
END SUBROUTINE BCND

SUBROUTINE ICND
END SUBROUTINE ICND

SUBROUTINE FOPT
END SUBROUTINE FOPT

SUBROUTINE PVLS
END SUBROUTINE PVLS

