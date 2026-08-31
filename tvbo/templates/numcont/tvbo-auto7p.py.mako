<%
model = context['model']

sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_inputs.keys()) if model.coupling_inputs else []
dv_names = list(model.in_dependency_order('derived_variables').keys()) if model.derived_variables else []
dp_names = list(model.in_dependency_order('derived_parameters').keys()) if model.derived_parameters else []

# Single-node continuation zeroes every coupling input: otherwise undeclared in Fortran.
coupling_zero = list(ct_names)

# Fortran is case-insensitive and FUNC's own arguments are in scope, so a symbol colliding with either is renamed for the whole emission.
reserved = {"ndim", "u", "icp", "par", "ijac", "f", "dfdu", "dfdp"}
emitted = sv_names + param_names + dv_names + dp_names
lowered = [name.lower() for name in emitted]


def rename(name):
    """*name* as Fortran may declare it: suffixed when it collides with FUNC's own arguments, or when another emitted symbol differs from it only in case (`A` keeps the name, `a` becomes `alow`)."""
    if name.lower() in reserved:
        return name + "_par"
    if name[0].islower() and lowered.count(name.lower()) > 1:
        return name + "low"
    return name


replace = {name: rename(name) for name in emitted}

# Fortran has no closures, so model functions such as Sigm are inlined into every right-hand side.
render_eq = lambda obj: model.render_equation(obj, format='fortran', inline_functions=True, replace=replace, remove=coupling_zero)
%>
SUBROUTINE FUNC(NDIM, U, ICP, PAR, IJAC, F, DFDU, DFDP)

    IMPLICIT NONE

    INTEGER NDIM, IJAC, ICP(*)
    DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
    DOUBLE PRECISION ${",".join([replace[sv.name] for sv in model.state_variables.values()])}
    DOUBLE PRECISION ${", ".join([f"{replace[p.name]}" for p in model.parameters.values()])}
% if model.derived_parameters:
    DOUBLE PRECISION ${", ".join([replace[dp.name] for dp in model.in_dependency_order('derived_parameters').values()])}
% endif
% if model.derived_variables:
    DOUBLE PRECISION ${", ".join([replace[k] for k in model.in_dependency_order('derived_variables').keys()])}
% endif

    % for i, p in enumerate(model.parameters.values()):
    ${replace[p.name]} = PAR(${i+1 if i+1 <= 10 else i+3})
    % endfor

% if model.derived_parameters:
    % for dp in model.in_dependency_order('derived_parameters').values():
    ${replace[dp.name]} = ${render_eq(dp)}
    % endfor
% endif

    % for i, sv in enumerate(model.state_variables.values()):
    ${replace[sv.name]} = U(${i+1})
    % endfor

% if model.derived_variables:
    % for k,v in model.in_dependency_order('derived_variables').items():
    ${replace[k]} = ${render_eq(v)}
    % endfor
% endif

    % for i, sv in enumerate(model.state_variables.values()):
    F(${i+1}) = ${render_eq(sv)}
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

