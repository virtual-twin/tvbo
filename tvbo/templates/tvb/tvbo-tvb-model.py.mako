## -*- coding: utf-8 -*-
##
<%namespace name="fn" file="/base/function-def.mako"/>
<%
import numpy as np
from tvbo.classes.equation import _clash1
if 'experiment' in context.keys():
    model = context['experiment'].dynamics
    standalone = False
else:
    model = context['model']
    standalone = True
render = lambda obj: model.render_equation(obj, format='numpy')

# In TVB, local_coupling is a separate dfun argument, not part of the coupling array.
# Use the ontology to identify which coupling inputs are global (part of coupling array)
# vs local (passed as separate dfun argument). This correctly handles models like SupHopf
# that have named local coupling terms (e.g. lc_0) beyond just 'local_coupling'.
from tvbo.ontology import owl as _onto
try:
    _global_names = set(_onto.get_model_coupling_terms(model.name, only_global=True).keys())
    global_coupling_inputs = {k: v for k, v in model.coupling_inputs.items() if k in _global_names}
except Exception:
    # Model not in ontology — filter by naming convention:
    # 'local_coupling' and 'lc_*' prefixed terms are local coupling
    def _is_local(name):
        return name == 'local_coupling' or name.startswith('lc_')
    global_coupling_inputs = {k: v for k, v in model.coupling_inputs.items() if not _is_local(k)}
local_coupling_inputs = {k: v for k, v in model.coupling_inputs.items() if k not in global_coupling_inputs}
has_local_coupling = bool(local_coupling_inputs)
%>
% if standalone:
# Auto-generated standalone model file
import numpy as np
from tvb.basic.neotraits.api import Attr, Final, List, NArray, Range
from tvb.simulator.models.base import Model

% endif
class ${model.name}(Model):

    % for p in model.parameters.values():
    ${p.name} = NArray(
        label=r":math:`${p.symbol or p.name}`",
        default=np.array([${p.value}]),
        % if p.domain:
        domain=Range(lo=${p.domain.lo}, hi=${p.domain.hi}, step=${p.domain.step if p.domain.step is not None else 1.0}),
        % endif
        % if p.description:
        doc="""${p.description}"""
        % elif p.definition:
        doc="""${p.definition[:200].replace('\n', '')}"""
        % endif
    )
    % endfor

    _nvar = ${len(model.state_variables)}
    state_variables = ${list(model.state_variables.keys())}
########## StateVariable Ranges and Boundaries ##########
<%
# Define 1e9 constant for readability
INFINITY = '1e9'
NEGINFINITY = '-1e9'

def format_range_or_boundary(sv, attr, default=(NEGINFINITY, INFINITY)):
    if getattr(sv, attr):
        lo = str(getattr(sv, attr).lo) if getattr(sv, attr).lo is not None else default[0]
        hi = str(getattr(sv, attr).hi) if getattr(sv, attr).hi is not None else default[1]
    else:
        lo, hi = default

    return f'"{sv.name}": np.array([{lo}, {hi}])'
%>

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            ${",\n\t\t\t".join([format_range_or_boundary(sv, 'domain') for sv in model.state_variables.values()])}
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup."""
    )

% if any(sv.boundaries is not None for sv in model.state_variables.values()):
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            ${",\n\t\t\t".join([format_range_or_boundary(sv, 'boundaries') for sv in model.state_variables.values()])}
        },
        doc="""State variable boundaries for phase plane setup."""
    )
% endif
########## Variables Of Interest ##########
    <%
    output_keys = tuple(model.output) if isinstance(model.output, list) else tuple(model.output.keys()) if model.output else ()
    choices = tuple(model.state_variables.keys()) + output_keys

    variables_of_interest = tuple(sv.name for sv in model.state_variables.values() if sv.variable_of_interest) + output_keys
    %>

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=${choices},
        default=${tuple(variables_of_interest) if variables_of_interest else tuple(model.state_variables.keys())},
        doc="""Default state variables to be monitored."""
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=${list(model.parameters.keys())},
    )

    local_parameter_names = ${[p.name for p in model.parameters.values() if isinstance(p, np.ndarray)]}

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            ${"\n\t\t\t".join([f'"{sv.name}": "{render(sv)}",' for sv in model.state_variables.values()])}
        },
    )

    cvar = np.array(${[i for i, sv in enumerate(model.state_variables.values()) if sv.coupling_variable]}, dtype=np.int32)

    coupling_terms = Final(
        label="Coupling terms",
        default=${[p.name for p in global_coupling_inputs.values()]}
    )

    _R = None
    _stimulus = 0.0
    use_numba = False

########## DerivedParameter ##########
% if model.derived_parameters:
    def update_derived_parameters(self):

    % for p in model.parameters:
        ${p} = self.${p}
    % endfor

    % for dp in model.derived_parameters.values():
        self.${dp.name} = ${dp.name} = ${render(dp)}
    % endfor
% endif

########## OutputTransforms ##########
% if model.output:
    def _build_observer(self):
        template = ("def observe(state):\n"
                    "    {svars} = state\n"
                    % if isinstance(model.output, list):
                        % for var_name in model.output:
<%                          dv = model.derived_variables.get(var_name) %>\
                    "    ${var_name} = ${render(dv) if dv else var_name}\n"
                        % endfor
                    % else:
                        % for ot in model.output.values():
                    "    ${ot.name} = ${render(ot)}\n"
                        % endfor
                    % endif
                    "    return numpy.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars=svars,
            voi_names=','.join(self.variables_of_interest)
        )
        namespace = {'numpy': np, 'np': np, 'pi':np.pi, 'sin':np.sin}
        namespace.update(self.__dict__)
        self.log.debug('building observer with code:\n%s', code)
        exec(code, namespace)
        self.observe = namespace['observe']
        self.observe.code = code
% endif

########## Basic Dfun ##########
    def dfun(self, state_variables, coupling, local_coupling=0):

        # shape (n_sv, n_modes)
% for sv in model.state_variables:
        ${sv} = state_variables[${list(model.state_variables.keys()).index(sv)}, :]
% endfor

% for p in model.parameters:
        ${p} = self.${p}
% endfor

% if model.functions:
        # Functions
% for f in model.functions.values():
<%
    fndef = capture(fn.function_def, f, format='numpy', render_func=render).strip()
    indented_fndef = '\n'.join('        ' + line if line.strip() else '' for line in fndef.split('\n'))
%>
${indented_fndef}
% endfor
% endif

% for p in model.derived_parameters:
        ${p} = self.${p}
% endfor

## Coupling Terms
        # Coupling Terms (from coupling array)
% for cterm in global_coupling_inputs:
        ${cterm} = coupling[${list(global_coupling_inputs).index(cterm)}, :]
% endfor

## Derived Variables
        # Derived Variables
% for k,v in model.derived_variables.items():
        self.${k} = ${k} = ${render(v)}
% endfor

## State-Equations
        # Time Derivatives
        return np.array([
    % for sv in model.state_variables.values():
            ${render(sv)}, # ${sv.name}
    % endfor
        ])
