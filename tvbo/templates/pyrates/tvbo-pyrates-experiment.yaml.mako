## -*- coding: utf-8 -*-
##
## PyRates Experiment Template (Complete Runnable YAML)
## =====================================================
## Combines Model (OperatorTemplate) + Network (Node/Circuit) into a
## complete, self-contained PyRates YAML file ready for simulation.
##
## This is the template to use when exporting for actual PyRates execution.
##
<%
# Can be used with either:
# - A single Dynamics model (creates single-node experiment)
# - A Network object (creates multi-node experiment)

# Check what we received
if 'network' in context.keys() and context['network'] is not None:
    network = context['network']
    is_network = True
else:
    model = context.get('model')
    is_network = False

def convert_rhs(rhs):
    """Convert TVBO equation RHS to PyRates syntax."""
    if not rhs:
        return rhs
    return str(rhs).replace("numpy.", "").replace("np.", "").replace("math.", "")

def render_operator(m, op_name):
    """Render a single OperatorTemplate from a Dynamics model."""
    equations = []
    variables = {}

    # Derived variables
    for var_name, dv in (m.derived_variables or {}).items():
        if dv.equation and dv.equation.rhs:
            rhs = convert_rhs(str(dv.equation.rhs))
            equations.append(f"{var_name} = {rhs}")
            variables[var_name] = "variable"

    # State variables
    for var_name, sv in (m.state_variables or {}).items():
        if sv.equation and sv.equation.rhs:
            rhs = convert_rhs(str(sv.equation.rhs))
            equations.append(f"{var_name}' = {rhs}")
            iv = sv.initial_value if sv.initial_value is not None else 0.0
            variables[var_name] = f"variable({iv})"

    # Output transforms
    for var_name, ot in (m.output_transforms or {}).items():
        if ot.equation and ot.equation.rhs:
            rhs = convert_rhs(str(ot.equation.rhs))
            equations.append(f"{var_name} = {rhs}")
            variables[var_name] = "output"

    # Parameters
    for param_name, param in (m.parameters or {}).items():
        val = param.value if param.value is not None else 0.0
        variables[param_name] = float(val)

    # Coupling terms as inputs
    for ct_name in (m.coupling_terms or {}).keys():
        variables[ct_name] = "input"

    description = m.description or f"TVBO model: {m.name or op_name.replace('_op', '')}"

    return {
        'name': op_name,
        'description': description,
        'equations': equations,
        'variables': variables,
    }

# Collect all operators and nodes
operators = {}
nodes = {}
edges = []

if is_network:
    circuit_name = getattr(network, 'name', None) or "tvbo_circuit"

    # Collect unique models and their operators
    for node_id in network.graph.nodes:
        node_data = network.graph.nodes[node_id]
        if 'model' in node_data:
            m = node_data['model']
            model_name = m.name or f"model_{node_id}"
            op_name = f"{model_name}_op"
            if op_name not in operators:
                operators[op_name] = render_operator(m, op_name)
            nodes[node_id] = model_name

    # Collect edges
    for edge in network.graph.edges(data=True):
        if len(edge) == 3:
            src, tgt, data = edge
        else:
            src, tgt, key, data = edge

        src_model = nodes.get(src, "unknown")
        tgt_model = nodes.get(tgt, "unknown")
        weight = data.get('weight', 1.0)
        delay = data.get('delay', 0.0)

        # Get connection variables
        src_node_data = network.graph.nodes[src]
        tgt_node_data = network.graph.nodes[tgt]
        src_m = src_node_data.get('model')
        tgt_m = tgt_node_data.get('model')
        src_var = list(src_m.state_variables.keys())[0] if src_m and src_m.state_variables else "x"
        tgt_var = list(tgt_m.coupling_terms.keys())[0] if tgt_m and tgt_m.coupling_terms else src_var

        edges.append({
            'src': src,
            'tgt': tgt,
            'src_op': f"{src_model}_op",
            'tgt_op': f"{tgt_model}_op",
            'src_var': src_var,
            'tgt_var': tgt_var,
            'weight': weight,
            'delay': delay,
        })
else:
    # Single model case
    name = model.name or "tvbo_model"
    op_name = f"{name}_op"
    circuit_name = f"{name}_circuit"
    operators[op_name] = render_operator(model, op_name)
    nodes['p'] = name
%>\
${"%" + "YAML 1.2"}
---
# PyRates Experiment Template
# Generated from TVBO ${"Network" if is_network else "Dynamics"}
# This file is self-contained and ready for PyRates simulation.

#############################################
# OPERATORS (Model Dynamics)
#############################################
% for op_name, op in operators.items():

${op_name}:
  base: OperatorTemplate
  description: "${op['description'].replace('"', "'")}"
% if len(op['equations']) == 1:
  equations: "${op['equations'][0]}"
% else:
  equations:
% for eq in op['equations']:
    - "${eq}"
% endfor
% endif
  variables:
% for var_name, var_spec in op['variables'].items():
% if isinstance(var_spec, float):
    ${var_name}: ${var_spec}
% else:
    ${var_name}: ${var_spec}
% endif
% endfor
% endfor

#############################################
# NODES (Network Topology)
#############################################
% if is_network:
% for node_id, model_name in nodes.items():

${model_name}:
  base: NodeTemplate
  operators:
    - ${model_name}_op
% endfor
% else:

${name}:
  base: NodeTemplate
  operators:
    - ${op_name}
% endif

#############################################
# CIRCUIT (Complete Network)
#############################################

${circuit_name}:
  base: CircuitTemplate
  nodes:
% for node_id, model_name in nodes.items():
    ${node_id}: ${model_name}
% endfor
% if edges:
  edges:
% for e in edges:
    - [${e['src']}/${e['src_op']}/${e['src_var']}, ${e['tgt']}/${e['tgt_op']}/${e['tgt_var']}, null, {weight: ${e['weight']}, delay: ${e['delay']}}]
% endfor
% endif
