## -*- coding: utf-8 -*-
##
<%namespace name="fn" file="/base/function-def.mako"/>
<%
import numpy as np
from tvbo.codegen.templater import time_dependent_equations
from tvbo.templates.base.utils import docstring_text, get_func_name, is_local_coupling, referenced_parameters
from tvbo.templates.tvboptim.utils import get_mode_layout, get_recorded_variable_names, state_only_dependencies, state_only_derived_var_names
if 'experiment' in context.keys():
    model = context['experiment'].dynamics
    standalone = False
else:
    model = context['model']
    standalone = True
render = lambda obj: model.render_equation(obj, format='numpy')
## Opt-in (model.cse): hoist repeated subexpressions — e.g. repeated muV/sigmaV/T_V
## calls in transfer functions — so the interpreted TVB dfun evaluates each once.
## When off, render_func_cse is None and function_def emits the flat form.
render_cse = (lambda obj: model.render_equation_cse(obj, format='numpy')) if getattr(model, 'cse', False) else None

## In TVB, local_coupling is a separate dfun argument, not part of the coupling array.
global_coupling_inputs = {k: v for k, v in model.coupling_inputs.items() if not is_local_coupling(k, v)}
local_coupling_inputs = {k: v for k, v in model.coupling_inputs.items() if k not in global_coupling_inputs}
has_local_coupling = bool(local_coupling_inputs)

# TVB's dfun takes no time argument, so a `t` term would emit an unbound name.
_time_dependent = time_dependent_equations(model)
if _time_dependent:
    raise ValueError(
        f"{model.name!r} is non-autonomous: the derivative(s) of "
        f"{', '.join(_time_dependent)} depend on time. TVB's Model.dfun takes no time "
        f"argument, so the TVB backend cannot express it — render to 'jax', 'tvboptim' "
        f"or 'julia', which pass t, or move the time dependence into a stimulus."
    )
%>
% if standalone:
# Auto-generated standalone model file
import numpy as np
import scipy.special
from tvb.basic.neotraits.api import Attr, Final, List, NArray, Range
from tvb.simulator.models.base import Model

% endif
class ${get_func_name(model)}(Model):

    % for p in model.parameters.values():
    % if isinstance(p.value, (list, tuple)):
    # Array-valued model constant (e.g. mode-coupling matrix); a fixed class
    # attribute rather than a per-region NArray.
    ${p.name} = np.array(${list(p.value)})
    % else:
    ${p.name} = NArray(
        label=r":math:`${p.symbol or p.name}`",
        default=np.array([${p.value}]),
        % if p.domain:
        domain=Range(lo=${p.domain.lo}, hi=${p.domain.hi}, step=${p.domain.step if p.domain.step is not None else 1.0}),
        % endif
        % if p.description:
        doc="""${docstring_text(p.description)}"""
        % elif p.definition:
        doc="""${docstring_text(p.definition[:200].replace('\n', ''))}"""
        % endif
    )
    % endif
    % endfor

    _nvar = ${len(model.state_variables)}
    state_variables = ${list(model.state_variables.keys())}
% if getattr(model, 'number_of_modes', None) and model.number_of_modes != 1:
    number_of_modes = ${model.number_of_modes}
% endif
########## StateVariable Ranges and Boundaries ##########
<%
import math
from tvbo.adapters.tvb import tvb_state_variable_ranges, tvb_state_variable_boundaries

def _lit(v):
    """Render a float as a TVB-code literal, preserving infinities as np.inf."""
    return ('np.inf' if v > 0 else '-np.inf') if math.isinf(v) else repr(float(v))

def _entry(name, bounds):
    """Render one ``"name": np.array([lo, hi])`` dict entry."""
    lo, hi = bounds
    return f'"{name}": np.array([{_lit(lo)}, {_lit(hi)}])'

# Resolution (which source feeds the range vs the clamp) lives in the adapter;
# the template only renders the resulting {name: (lo, hi)} mappings.
sv_ranges = tvb_state_variable_ranges(model)
sv_boundaries = tvb_state_variable_boundaries(model)
%>

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            ${",\n\t\t\t".join([_entry(n, b) for n, b in sv_ranges.items()])}
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup."""
    )

% if sv_boundaries:
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            ${",\n\t\t\t".join([_entry(n, b) for n, b in sv_boundaries.items()])}
        },
        doc="""State variable boundaries for phase plane setup."""
    )
% endif
########## Variables Of Interest ##########
    <%
    # The channels tvboptim presents (`get_output_channels`): the states `record` keeps, in state order, then the derived variables tvboptim records, each once.
    output_keys = tuple(get_recorded_variable_names(model, None if standalone else context['experiment'])[1])
    recorded_states = tuple(sv.name for sv in model.state_variables.values() if getattr(sv, 'record', True))
    choices = tuple(model.state_variables.keys()) + output_keys
    variables_of_interest = recorded_states + output_keys
    # As tvboptim records them: an output the state alone determines is computed from the recorded state on a single-mode model, any other is kept from the step's first dfun evaluation, which TVB's observer, seeing only the state, cannot repeat.
    recomputable = state_only_derived_var_names(model) if output_keys and get_mode_layout(model)[0] == 1 else ()
    stepped_outputs = tuple(name for name in output_keys if name not in recomputable)
    # The observer binds the state-only outputs and the derived variables they read, in dependency order with the model's functions inlined, as tvboptim's realignment renders them.
    observed_derived = state_only_dependencies(model, output_keys) if recomputable else ()
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
% if stepped_outputs:
    _first_evaluation = None
% endif

########## DerivedParameter ##########
% if model.derived_parameters:
    def update_derived_parameters(self):

    % for p in model.parameters:
        ${p} = self.${p}
    % endfor

    % for dp in model.in_dependency_order('derived_parameters').values():
        self.${dp.name} = ${dp.name} = ${render(dp)}
    % endfor
% endif

########## OutputTransforms ##########
% if output_keys:
    def _build_observer(self):
        template = ("def observe(state):\n"
                    "    {svars} = state\n"
                    % if stepped_outputs:
                    "    ${', '.join(stepped_outputs)}, = model._first_evaluation\n"
                    "    model._first_evaluation = None\n"
                    % endif
                    % for name in observed_derived:
                    "    ${name} = ${model.render_equation(model.derived_variables[name], format='numpy', inline_functions=True)}\n"
                    % endfor
                    "    return np.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars=svars,
            voi_names=','.join(self.variables_of_interest)
        )
        # the namespace dfun runs in, plus this model's parameters
        namespace = {**globals(), **self.__dict__}
% if stepped_outputs:
        namespace['model'] = self
% endif
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

% for p in referenced_parameters(model):
        ${p} = self.${p}
% endfor

% if model.functions:
        # Functions
% for f in model.functions.values():
<%
    fndef = capture(fn.function_def, f, format='numpy', render_func=render, render_func_cse=render_cse).strip()
    indented_fndef = '\n'.join('        ' + line if line.strip() else '' for line in fndef.split('\n'))
%>
${indented_fndef}
% endfor
% endif

% for p in model.in_dependency_order('derived_parameters'):
        ${p} = self.${p}
% endfor

## Coupling Terms
        # Coupling Terms (from coupling array)
% for cterm in global_coupling_inputs:
        ${cterm} = coupling[${list(global_coupling_inputs).index(cterm)}, :]
% endfor

## Derived Variables
        # Derived Variables
% for k,v in model.in_dependency_order('derived_variables').items():
        self.${k} = ${k} = ${render(v)}
% endfor
% if stepped_outputs:
        if self._first_evaluation is None:
            self._first_evaluation = (${', '.join(stepped_outputs)},)
% endif

## State-Equations
        # Time Derivatives
        # Broadcast every derivative to a common per-node shape so constant
        # (state-independent) derivatives — e.g. ``duA/dt = cA`` — stack into a
        # homogeneous array alongside node-shaped ones, matching TVB's own
        # ``numpy.full_like`` handling of constant drifts.
        _derivs = np.broadcast_arrays(
    % for sv in model.state_variables.values():
            np.asarray(${render(sv)}), # ${sv.name}
    % endfor
        )
        return np.array(_derivs)
