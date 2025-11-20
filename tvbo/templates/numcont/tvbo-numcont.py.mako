## -*- coding: utf-8 -*-
<%!
from tvbo.export.code import render_equation as render_eq
from tvbo.knowledge.simulation.equations import _clash1
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
        ${dp.name} = ${render_eq(dp.equation, format='numpy')}
        % endfor
    % endif

    % if model.metadata.functions:
        % for f in model.metadata.functions.values():
        def ${f.name}(${", ".join([arg.name for arg in f.arguments.values()])}):
            return ${render_eq(f.equation, format='numpy')}
        % endfor
    % endif

    % if model.metadata.derived_variables:
        % for k,v in model.metadata.derived_variables.items():
        ${k} = ${render_eq(v.equation, user_functions={f:f for f in model.metadata.functions.keys()}, format='numpy')}
        % endfor
    % endif

    % for i, sv in enumerate(model.metadata.state_variables.values()):
        dx_dt[${i}] = ${render_eq(sv.equation, user_functions={f:f for f in model.metadata.functions.keys()}, format='numpy', remove=['local_coupling']+
        [f.name for f in model.metadata.coupling_terms.values()])}
    % endfor

        return dx_dt
