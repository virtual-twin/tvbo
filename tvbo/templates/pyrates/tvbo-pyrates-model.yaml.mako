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
    # Replace reserved names that conflict with SymPy/PyRates built-ins:
    # - 'I' conflicts with PyRates built-in 'I' (imaginary unit/current)
    # - 'gamma' conflicts with SymPy's gamma function
    # - 'beta' conflicts with SymPy's beta function (Euler beta)
    # - 'zeta' conflicts with SymPy's zeta function (Riemann zeta)
    # - 'lambda' is a Python keyword
    # - 'E' conflicts with SymPy's E (Euler's number)
    # - 'N' conflicts with SymPy's N (numerical evaluation)
    # - 'S' conflicts with SymPy's S (sympify shorthand)
    # - 'O' conflicts with SymPy's O (big-O notation)
    repl = {
        "I": "Ipar",
        "gamma": "gamma_par",
        "beta": "beta_par",
        "zeta": "zeta_par",
        "lambda": "lambda_par",
        "E": "E_par",
        "N": "N_par",
        "S": "S_par",
        "O": "O_par",
    }

    # Get model name
    name = m.name or "tvbo_model"
    _op_name = op_name or f"{name}_op"

    # Collect equations and variables
    equations = []
    variables = {}

    # Add derived variables (algebraic equations)
    for k, dv in m.derived_variables.items():
        # Use sympy format (bare function names) and inline any custom functions
        equations.append(f"{k} = {m.render_equation(dv, format='sympy', inline_functions=True, replace=repl)}")
        variables[k] = "variable"

    # Add state variable equations (differential equations)
    for k, sv in (m.state_variables or {}).items():
        # Use sympy format and inline any custom functions
        equations.append(f"{k}' = {m.render_equation(sv, format='sympy', inline_functions=True, replace=repl)}")
        iv = sv.initial_value
        variables[k] = f"variable({iv})"

    # Add output transforms (algebraic equations)
    # Note: PyRates only allows ONE output per operator, so we mark these as 'variable'
    for k, ot in (m.output or {}).items():
        equations.append(f"{k} = {m.render_equation(ot, format='sympy', inline_functions=True, replace=repl)}")
        variables[k] = "variable"

    # Add parameters as constants
    for param_name, param in (m.parameters or {}).items():
        if param_name in repl:
            param_name = repl[param_name]

        val = param.value
        variables[param_name] = float(val)

    # Add derived parameters as equations
    for dp_name, dp in (m.derived_parameters or {}).items():
        eq_str = m.render_equation(dp, format='sympy', inline_functions=True, replace=repl)
        equations.append(f"{dp_name} = {eq_str}")
        variables[dp_name] = "variable"

    # Add coupling terms as inputs
    for ct_name in (m.coupling_terms or {}).keys():
        variables[ct_name] = "input"

    description = m.description or f"TVBO model: {name}"
%>\
${_op_name}:
  base: OperatorTemplate
  description: "${description.replace('"', "'")}"
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
% if isinstance(var_spec, float):
    ${var_name}: ${var_spec}
% else:
    ${var_name}: ${var_spec}
% endif
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
