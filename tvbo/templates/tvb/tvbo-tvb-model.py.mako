## -*- coding: utf-8 -*-
##
<%
import numpy as np
from tvbo.knowledge.simulation.equations import _clash1
if 'experiment' in context.keys():
    model = context['experiment'].local_dynamics.metadata
else:
    model = context['model'].metadata
render = lambda obj: model.render_equation(obj, format='python')
%>
class ${model.name}(Model):

    % for p in model.parameters.values():
    ${p.name} = NArray(
        label=r":math:`${p.symbol or p.name}`",
        default=np.array([${p.value}]),
        % if p.domain:
        domain=Range(lo=${p.domain.lo}, hi=${p.domain.hi}, step=${p.domain.step}),
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
    choices = tuple(model.state_variables.keys()) + (tuple(model.output_transforms.keys()) if model.output_transforms else ())

    variables_of_interest = tuple(sv.name for sv in model.state_variables.values() if sv.variable_of_interest) + (tuple(model.output_transforms.keys()) if model.output_transforms else ())
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

    ## TODO: Was this hack necessary for anything?
    ## % if any([sv.coupling_variable for sv in model.state_variables.values()]):
    ## cvar = np.array(${[i for i, sv in enumerate(model.state_variables.values()) if sv.coupling_variable]*len(model.coupling_terms)}, dtype=np.int32)
    cvar = np.array(${list(range(len(model.state_variables)))}, dtype=np.int32)

    coupling_terms = Final(
        label="Coupling terms",
        default=${[p.name for p in model.coupling_terms.values()]}
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
% if model.output_transforms:
    def _build_observer(self):
        template = ("def observe(state):\n"
                    "    {svars} = state\n"
                    % for ot in model.output_transforms.values():
                    "    ${ot.name} = ${render(ot)}\n"
                    % endfor
                    "    return numpy.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars=svars,
            voi_names=','.join(self.variables_of_interest)
        )
        namespace = {'numpy': np, 'pi':np.pi, 'sin':np.sin}
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
        def ${f.name}(${", ".join([arg.name for arg in f.arguments.values()])}):
            return ${render(f)}
% endfor
% endif

% for p in model.derived_parameters:
        ${p} = self.${p}
% endfor

## Coupling Terms
        # Coupling Terms
% for cterm in model.coupling_terms:
        ${cterm} = coupling[${list(model.coupling_terms).index(cterm)}, :]
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
