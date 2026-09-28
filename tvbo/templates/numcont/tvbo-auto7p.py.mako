<%
model = context['model']
replace = context['replace']
coupling_zero = context['coupling_zero']

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

