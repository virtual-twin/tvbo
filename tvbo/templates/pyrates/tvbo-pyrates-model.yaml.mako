## -*- coding: utf-8 -*-
##
## PyRates Model Template (OperatorTemplate only)
## ================================================
## Generates PyRates OperatorTemplate from TVBO Dynamics model.
## This template defines ONLY the dynamics (equations and variables).
## Use tvbo-pyrates-network.yaml.mako for Node/Circuit topology.
## Use tvbo-pyrates-experiment.yaml.mako for complete runnable experiments.
##
## Note: Custom functions (e.g., Sigm) are automatically inlined into equations
## since PyRates doesn't support user-defined functions in YAML templates.
## We use format='sympy' since PyRates parses equations with SymPy internally.
##
## This template can be used standalone or included via <%namespace>.
##

<%def name="render_operator(m, op_name=None)">
<%
    from tvbo.codegen.pyrates import operator_template
    _op = operator_template(m, op_name)
    _op_name, description = _op['op_name'], _op['description']
    equations, variables = _op['equations'], _op['variables']
%>\
${_op_name}:
  base: OperatorTemplate
  description: "${description}"
% if len(equations) == 1:
  equations: "${equations[0]}"
% else:
  equations:
% for eq in equations:
    - "${eq}"
% endfor
% endif
  variables:
% for var_name, var_spec in variables.items():
    ${var_name}: ${var_spec}
% endfor
</%def>\
##
## Standalone rendering when used directly (not via namespace)
##
% if 'model' in context.keys() and context['model'] is not None:
<%
model = context['model']
name = model.name or "tvbo_model"
op_name = f"{name}_op"
%>\
# PyRates OperatorTemplate: ${name}
# Generated from TVBO Dynamics model

${render_operator(model, op_name)}
% endif
