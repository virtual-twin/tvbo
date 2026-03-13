## -*- coding: utf-8 -*-
<%!
from tvbo.codegen.code import render_equation as render_eq
from tvbo.classes.equation import _clash1
%>
<%
# Collect all symbol names so the parser recognizes them as Symbols
sv_names = list(model.metadata.state_variables.keys())
param_names = list(model.metadata.parameters.keys())
ct_names = list(model.metadata.coupling_terms.keys()) if model.metadata.coupling_terms else []
dv_names = list(model.metadata.derived_variables.keys()) if model.metadata.derived_variables else []
dp_names = list(model.metadata.derived_parameters.keys()) if model.metadata.derived_parameters else []
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
%>
import numpy as np
from numcont import ContinuationSystem as cs

class ${model.metadata.name}BifModel(cs.ContSystem):

    def __init__(self, fortran_file, data_path, N=1):
        super().__init__()

        self.SetParameterNames(${", ".join([f"'{p.name}'" for p in model.metadata.parameters.values()])})
        self.SetVariableNames(${", ".join([f"'{v.name}'" for v in model.metadata.state_variables.values()])})

        self.AutoFortranFile = fortran_file
        self.AutoDataPath = data_path

        # Parameters
        % for p in model.metadata.parameters.values():
        self.${p.name} = ${p.value}
        % endfor

        self.SetN(N)

    def SetN(self, N):
        self.N = N
        self.x0 = np.zeros((${len(model.metadata.state_variables)}, self.N))

    def dfun(self, t, x):
        dx_dt = self.dx_dt

        % for p in model.metadata.parameters.values():
        ${p.name} = self.${p.name}
        % endfor

        % for i, ivar in enumerate(model.metadata.state_variables):
        ${ivar} = x[${i}]
        % endfor

    % if model.metadata.derived_parameters:
        % for dp in model.metadata.derived_parameters.values():
        ${dp.name} = ${render_eq(dp.equation, format='numpy', parameters=all_symbols)}
        % endfor
    % endif

    % if model.metadata.functions:
        % for f in model.metadata.functions.values():
        def ${f.name}(${', '.join([arg.name if hasattr(arg, 'name') else str(arg) for arg in (f.arguments.values() if hasattr(f.arguments, 'values') else f.arguments)])}):
            return ${render_eq(f.equation, format='numpy', parameters=all_symbols)}
        % endfor
    % endif

    % if model.metadata.derived_variables:
        % for k,v in model.metadata.derived_variables.items():
        ${k} = ${render_eq(v.equation, user_functions={f:f for f in model.metadata.functions.keys()}, format='numpy', parameters=all_symbols)}
        % endfor
    % endif

    % for i, sv in enumerate(model.metadata.state_variables.values()):
        dx_dt[${i}] = ${render_eq(sv.equation, user_functions={f:f for f in model.metadata.functions.keys()}, format='numpy', parameters=all_symbols, remove=['local_coupling']+
        [f.name for f in model.metadata.coupling_terms.values()])}
    % endfor

        return dx_dt
