## -*- coding: utf-8 -*-
##
##
## TheVirtualBrain-Scientific Package. This package holds all simulators, and
## analysers necessary to run brain-simulations. You can use it stand alone or
## in conjunction with TheVirtualBrain-Framework Package. See content of the
## documentation-folder for more details. See also http://www.thevirtualbrain.org
##
## (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
##
## This program is free software: you can redistribute it and/or modify it under the
## terms of the GNU General Public License as published by the Free Software Foundation,
## either version 3 of the License, or (at your option) any later version.
## This program is distributed in the hope that it will be useful, but WITHOUT ANY
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
## PARTICULAR PURPOSE.  See the GNU General Public License for more details.
## You should have received a copy of the GNU General Public License along with this
## program.  If not, see <http://www.gnu.org/licenses/>.
##
##
##   CITATION:
## When using The Virtual Brain for scientific publications, please cite it as explained here:
## https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
##
##

def dfun(self, state_variables, coupling, local_coupling=0):
% for par,v in parameters.items():
    ${par} = self.${par}
% endfor

## % for par in spatial_parameter_names:
##     ${par} = parmat[${loop.index}]
## % endfor

    pi = numpy.pi
    exp = numpy.exp

    # unpack coupling terms and states as in dfuns
    ${','.join(coupling_terms)} = coupling

<%
if non_integrated_variables == None:
    integrated_state_variables = state_variables
else:
    integrated_state_variables = [var for var in state_variables if var not in non_integrated_variables]
%>
    ${','.join(integrated_state_variables)} = state_variables

    # compute internal states for dfuns
% for var, term in non_integrated_variables.items():
    %if var not in integrated_state_variables:
    ${var} = ${term}
    %endif
% endfor

    derivative = numpy.empty_like(integrated_state_variables)
    # compute derivatives
% for svar in integrated_state_variables:
    derivative[${loop.index}] = ${state_variable_dfuns[svar]};
% endfor

    return derivative
