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
import numpy as np

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
    for var_name, ot in (m.output or {}).items():
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

def get_node_labels(network, n_nodes):
    """Get node labels from Network."""
    if hasattr(network, 'node_labels') and network.node_labels:
        return list(network.node_labels)
    elif hasattr(network, 'region_labels') and network.region_labels:
        return list(network.region_labels)
    else:
        return [f"node_{i}" for i in range(n_nodes)]

# Collect all operators and nodes
operators = {}
nodes = {}
edges = []

if is_network:
    circuit_name = getattr(network, 'label', None) or getattr(network, 'name', None) or "tvbo_circuit"

    # Check if network has a .graph attribute (NetworkX graph-based)
    if hasattr(network, 'graph') and network.graph is not None:
        # NetworkX-based network (deprecated _Network style)
        for node_id in network.graph.nodes:
            node_data = network.graph.nodes[node_id]
            if 'model' in node_data:
                m = node_data['model']
                model_name = m.name or f"model_{node_id}"
                op_name = f"{model_name}_op"
                if op_name not in operators:
                    operators[op_name] = render_operator(m, op_name)
                nodes[node_id] = model_name

        # Collect edges from graph
        for edge in network.graph.edges(data=True):
            if len(edge) == 3:
                src, tgt, data = edge
            else:
                src, tgt, key, data = edge

            src_model = nodes.get(src, "unknown")
            tgt_model = nodes.get(tgt, "unknown")
            weight = data.get('weight', 1.0)
            delay = data.get('delay', 0.0)

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
        # Base Network/Connectome class (datamodel-based)
        # Get model from context or use a default operator name
        base_model = context.get('model')
        weights = network.weights_matrix if hasattr(network, 'weights_matrix') else None
        n_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
        if weights is not None:
            n_nodes = weights.shape[0]
        node_labels = get_node_labels(network, n_nodes)

        if base_model:
            model_name = base_model.name or "tvbo_model"
            op_name = f"{model_name}_op"
            operators[op_name] = render_operator(base_model, op_name)
        else:
            model_name = "node"
            op_name = "node_op"

        # Create nodes
        for i, label in enumerate(node_labels):
            safe_label = str(label).replace(" ", "_").replace("-", "_")
            nodes[safe_label] = model_name

        # Create edges from weights matrix
        if weights is not None:
            src_var = "x"
            tgt_var = "c_in"
            if base_model:
                if base_model.state_variables:
                    src_var = list(base_model.state_variables.keys())[0]
                if base_model.coupling_terms:
                    tgt_var = list(base_model.coupling_terms.keys())[0]

            for i in range(n_nodes):
                for j in range(n_nodes):
                    w = float(weights[i, j])
                    if w != 0.0:
                        src_label = str(node_labels[i]).replace(" ", "_").replace("-", "_")
                        tgt_label = str(node_labels[j]).replace(" ", "_").replace("-", "_")
                        edges.append({
                            'src': src_label,
                            'tgt': tgt_label,
                            'src_op': op_name,
                            'tgt_op': op_name,
                            'src_var': src_var,
                            'tgt_var': tgt_var,
                            'weight': w,
                            'delay': np.float64(0.0),
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
<%
# Get unique model names to avoid duplicate NodeTemplate definitions
unique_models = set(nodes.values())
%>
% if is_network:
% for model_name in sorted(unique_models):

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
