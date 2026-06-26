## -*- coding: utf-8 -*-
##
## RateML-style Python/Numba Model Template
## =========================================
##
## Generates TVB-compatible Python model with Numba gufunc for performance.
## Uses TVBO SimulationExperiment/Dynamics metadata directly.
##
<%doc>
Context Variables:
- model: Dynamics instance (required)
- experiment: SimulationExperiment (optional, for coupling info)

Output:
- TVB-compatible Model class with Numba-accelerated dfun
</%doc>
<%
from tvbo.templates.rateml.utils import (
    python_code, has_boundaries, get_initial_value,
    get_domain_str, get_boundary_str, get_range_str
)

# Model name: capitalize and add 'T' suffix (RateML convention)
model_name = model.name.replace(' ', '').replace('-', '') + 'T'

# State variables
state_vars = list(model.state_variables.items())
n_states = len(state_vars)

# Parameters (constants in RateML terminology)
params = list(model.parameters.items()) if model.parameters else []

# Derived variables
derived_vars = list(model.derived_variables.items()) if model.derived_variables else []

# Coupling terms
coupling_terms = list(model.coupling_terms.keys()) if model.coupling_terms else ['c_pop0']

# Check for state variable boundaries
svboundaries = has_boundaries(model)

# Variables of interest (exposures)
exposures = [name for name, sv in state_vars if getattr(sv, 'record', True)]
%>
from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range


class ${model_name}(ModelNumbaDfun):
    """
    ${model.name} model generated from TVBO specification.

    ${model.description or ''}
    """

    % for p_name, param in params:
    ${p_name} = NArray(
        label=":math:`${p_name}`",
        default=numpy.array([${param.value if param.value is not None else 1.0}]),
        % if get_range_str(param):
        domain=Range(${get_range_str(param)}),
        % endif
        doc="""${(param.description or param.definition or '')[:100]}"""
    )
    % endfor

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            % for sv_name, sv in state_vars:
            "${sv_name}": numpy.array([${get_domain_str(sv)}])${',' if not loop.last else ''}
            % endfor
        },
        doc="""state variables"""
    )

    % if svboundaries:
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            % for sv_name, sv in state_vars:
            % if get_boundary_str(sv):
            "${sv_name}": numpy.array([${get_boundary_str(sv)}])${',' if not loop.last else ''}
            % endif
            % endfor
        },
    )
    % endif

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=(${', '.join(f"'{e}'" for e in exposures)}),
        default=(${', '.join(f"'{e}'" for e in exposures)}),
        doc="Variables to monitor"
    )

    state_variables = [${', '.join(f"'{name}'" for name, _ in state_vars)}]

    _nvar = ${n_states}
    cvar = numpy.array([${', '.join(str(i) for i in range(n_states))}], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_${model_name}(vw_, c_, \
% for p_name, _ in params:
self.${p_name}, \
% endfor
local_coupling)

        return deriv.T[..., numpy.newaxis]


## Numba signature: (n) for state, (m) for coupling, then scalar params, output (n)
@guvectorize([(float64[:], float64[:], \
% for _ in range(len(params) + 1):
float64, \
% endfor
float64[:])], '(n),(m)' + ',()'*${len(params) + 1} + '->(n)', nopython=True)
def _numba_dfun_${model_name}(vw, coupling, \
% for p_name, _ in params:
${p_name}, \
% endfor
local_coupling, dx):
    "Gufunc for ${model_name} model equations."

    # Long-range coupling
    % for i, ct in enumerate(coupling_terms):
    ${ct} = coupling[${i}]
    % endfor

    # Unpack state variables
    % for i, (sv_name, sv) in enumerate(state_vars):
    ${sv_name} = vw[${i}]
    % endfor

    % if derived_vars:
    # Derived variables
    % for dv_name, dv in derived_vars:
    ${dv_name} = ${python_code(dv.equation)}
    % endfor
    % endif

    # Time derivatives
    % for i, (sv_name, sv) in enumerate(state_vars):
    dx[${i}] = ${python_code(sv.equation)}
    % endfor
