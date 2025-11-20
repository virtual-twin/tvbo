## -*- coding: utf-8 -*-
##

class ${model_name}(Model):

% for k, v in parameters.items():
    % if v['range'] is not None and len(v['range']) == 3:
        <% lo, hi, step = v['range'] %>
        ${k} = NArray(
            label=r":math:`${v['symbol']}`",
            default=np.array([${v['default']}]),
            domain=Range(lo=${lo}, hi=${hi}, step=${step}),
            doc="""${v['definition'][:200] if v['definition'] else ''}"""
        )
    % else:
        ${k} = NArray(
            label=r":math:`${v['symbol']}`",
            default=np.array([${v['default']}]),
            doc="""${v['definition'][:200] if v['definition'] else ''}"""
        )
    % endif
% endfor

## State variables
        tvbo_model="${model_name}"
        _nvar = ${len(state_variables.keys())}
        state_variables = ${list(state_variables.keys())}
        state_variable_range = Final(
                label="State Variable ranges [lo, hi]",
                default={
% for k, v in state_variables.items():
                    "${k}": np.array([${f"{str(v['range'][0]).replace('lo=', '')} ,{str(v['range'][1]).replace('hi=', '')}" if v['range'][0] else '-1e9, 1e9'}]),
% endfor
                },
                doc="""Expected ranges of the state variables for initial condition generation and phase plane setup."""
        )

## SV-Boundaries
% if any(v['boundaries'][0] is not None or v['boundaries'][1] is not None for v in state_variables.values()):
        state_variable_boundaries = Final(
                label="State Variable boundaries [lo, hi]",
                default={
% for k, v in state_variables.items():
    % if v['boundaries'] and (v['boundaries'][0] is not None or v['boundaries'][1] is not None):
                    "${k}": np.array([${str(v['boundaries'][0]).replace('lo=', '')}, ${str(v['boundaries'][1]).replace('hi=', '')}]),
        % else:
                    "${k}": np.array([-inf, inf]),
    % endif
% endfor
                },
        )
% endif

        cvar = np.array(${cvars}, dtype=np.int32)
                ## self.parameter_names = parameter_names
                ## self.local_parameter_names = []


        variables_of_interest = List(
                of=str,
                label="Variables watched by Monitors",
                choices=${tuple(set(tuple(state_variables.keys())+tuple(vois)))},
                default=${tuple(vois)},
                doc="""default state variables to be monitored"""
        )

        parameter_names = List(
                of=str,
                label="List of parameters for this model",
                default=${list(parameters.keys())},
        )
        local_parameter_names = ${spatial_parameter_names}

        state_variable_dfuns = Final(
                label="Drift functions",
                default={
% for sv, eq in state_variable_dfuns.items():
                        "${sv}": "${eq}",
% endfor
                }
        )
        coupling_terms = Final(
                label="Coupling terms",
                default=${coupling_terms}
        )
% if non_integrated_variables is not None:
        non_integrated_variables = ${non_integrated_variables}
% endif
        _R = None
        _stimulus = 0.0
        use_numba = False

## Split SVs into integrated and non-integrated
<%
if non_integrated_variables == None:
                integrated_state_variables = state_variables
else:
                integrated_state_variables = [var for var in state_variables if var not in non_integrated_variables]
%>
% if len(parameters)<32:
########## Numba Dfun ##########
        @guvectorize([(f64[:],)*${len(parameters)+3}], '(n),(m)' + ',()'*${len(parameters)} + '->(n)', nopython=True)
        def _numba_dfun(
                state_variables,
                coupling,
% for par,v in parameters.items():
                ${par},
% endfor
                derivative
                ):
                "Gufunc for ${model_name} model equations"
                pi = np.pi
                exp = np.exp
                local_coupling = 0

% for par,v in parameters.items():
                ${par} = ${par}[0]
% endfor
## Assign state variables
% for svar in integrated_state_variables:
                ${svar} = state_variables[${list(state_variables.keys()).index(svar)}]
% endfor
## Coupling Terms
% for cterm in coupling_terms:
                ${cterm} = coupling[${coupling_terms.index(cterm)}]
% endfor

## Assign functions
% for var, term in ninvar_dfuns.items():
    %if var not in integrated_state_variables:
                ${var} = ${term}
    %endif
% endfor

## Statevariables
% for svar in integrated_state_variables:
                derivative[${loop.index}] = ${state_variable_dfuns[svar]}
% endfor
% endif

########## Basic Dfun ##########
        def dfun(self, state_variables, coupling, local_coupling=0):
% if len(import_statements) > 0:
        % for import_statement in import_statements:
                ${import_statement}
        % endfor
% endif
% if len(parameters)<32:
                if self.use_numba:
                        deriv = self._numba_dfun(
                                state_variables.reshape(state_variables.shape[:-1]).T,
                                coupling.reshape(coupling.shape[:-1]).T,
% for par,v in parameters.items():
                                self.${par},
% endfor
                                )
                        deriv = deriv.T[..., np.newaxis]
                        return deriv
% endif
## Assign state variables
                # shape (n_sv, n_modes)
% for svar in integrated_state_variables:
                ${svar} = state_variables[${list(state_variables.keys()).index(svar)}, :]
% endfor

## Assign parameters
% for par,v in parameters.items():
                ${par} = self.${par}
% endfor

                pi = np.pi
                exp = np.exp
## Coupling Terms
% for cterm in coupling_terms:
                ${cterm} = coupling[${coupling_terms.index(cterm)}, :]
% endfor

## Assign functions
                # compute internal states for dfuns
% for var, term in ninvar_dfuns.items():
    %if var not in integrated_state_variables:
                ${var} = ${term}
    %endif
% endfor

                derivative = np.empty_like(state_variables)
                # compute derivatives
% for svar in integrated_state_variables:
                derivative[${loop.index}] = ${state_variable_dfuns[svar]}
% endfor
                return derivative
