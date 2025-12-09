## -*- coding: utf-8 -*-
##
## PyRates Model Template (OperatorTemplate only)
## ================================================
## Generates PyRates OperatorTemplate from TVBO Dynamics model.
## This template defines ONLY the dynamics (equations and variables).
## Use tvbo-pyrates-network.yaml.mako for Node/Circuit topology.
## Use tvbo-pyrates-experiment.yaml.mako for complete runnable experiments.
##
<%
model = context['model']

def convert_rhs(rhs):
    """Convert TVBO equation RHS to PyRates syntax."""
    if not rhs:
        return rhs
    return str(rhs).replace("numpy.", "").replace("np.", "").replace("math.", "")

# Get model name
name = model.name or "tvbo_model"
op_name = f"{name}_op"

# Collect equations and variables
equations = []
variables = {}

# Add derived variables (algebraic equations)
for var_name, dv in (model.derived_variables or {}).items():
    if dv.equation and dv.equation.rhs:
        rhs = convert_rhs(str(dv.equation.rhs))
        equations.append(f"{var_name} = {rhs}")
        variables[var_name] = "variable"

# Add state variable equations (differential equations)
for var_name, sv in (model.state_variables or {}).items():
    if sv.equation and sv.equation.rhs:
        rhs = convert_rhs(str(sv.equation.rhs))
        equations.append(f"{var_name}' = {rhs}")
        iv = sv.initial_value if sv.initial_value is not None else 0.0
        variables[var_name] = f"variable({iv})"

# Add output transforms (algebraic equations)
# Note: PyRates only allows ONE output per operator, so we mark these as 'variable'
# They can still be recorded via the outputs dict in run()
for var_name, ot in (model.output or {}).items():
    if ot.equation and ot.equation.rhs:
        rhs = convert_rhs(str(ot.equation.rhs))
        equations.append(f"{var_name} = {rhs}")
        variables[var_name] = "variable"

# Add parameters as constants
for param_name, param in (model.parameters or {}).items():
    val = param.value if param.value is not None else 0.0
    variables[param_name] = float(val)

# Add coupling terms as inputs
for ct_name in (model.coupling_terms or {}).keys():
    variables[ct_name] = "input"

description = model.description or f"TVBO model: {name}"
%>\
# PyRates OperatorTemplate: ${name}
# Generated from TVBO Dynamics model

${op_name}:
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
