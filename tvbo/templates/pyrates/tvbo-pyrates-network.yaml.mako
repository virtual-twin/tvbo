## -*- coding: utf-8 -*-
##
## PyRates Network Template (NodeTemplate and CircuitTemplate)
## ============================================================
## Generates PyRates Node and Circuit topology from TVBO Network.
## This template defines ONLY the network structure (nodes, edges, connectivity).
## Model operators are referenced but not defined here.
## Use tvbo-pyrates-model.yaml.mako for dynamics/equations.
## Use tvbo-pyrates-experiment.yaml.mako for complete runnable experiments.
##
<%
import numpy as np

# Can be used with either:
# - A single Dynamics model (creates single-node circuit)
# - A Network object (creates multi-node circuit with edges)

# Check what we received
if 'network' in context.keys() and context['network'] is not None:
    network = context['network']
    is_network = True
else:
    model = context.get('model')
    is_network = False

def get_weights_matrix(network):
    """Get weights matrix from Network."""
    if hasattr(network, 'weights_matrix'):
        return network.weights_matrix
    elif hasattr(network, 'weights') and network.weights is not None:
        if hasattr(network.weights, 'data'):
            return np.array(network.weights.data)
        elif isinstance(network.weights, np.ndarray):
            return network.weights
    return None

def get_node_labels(network, n_nodes):
    """Get node labels from Network."""
    if hasattr(network, 'node_labels') and network.node_labels:
        return list(network.node_labels)
    elif hasattr(network, 'region_labels') and network.region_labels:
        return list(network.region_labels)
    else:
        return [f"node_{i}" for i in range(n_nodes)]

# Build node and edge information
node_models = {}
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
                node_models[node_id] = model_name

        for edge in network.graph.edges(data=True):
            if len(edge) == 3:
                src, tgt, data = edge
            else:
                src, tgt, key, data = edge
            edges.append((src, tgt, data))
    else:
        # Base Network/Connectome class (datamodel-based)
        base_model = context.get('model')
        weights = get_weights_matrix(network)
        n_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
        if weights is not None:
            n_nodes = weights.shape[0]
        node_labels = get_node_labels(network, n_nodes)

        model_name = base_model.name if base_model else "node"

        # Create nodes
        for i, label in enumerate(node_labels):
            safe_label = str(label).replace(" ", "_").replace("-", "_")
            node_models[safe_label] = model_name

        # Create edges from weights matrix
        if weights is not None:
            for i in range(n_nodes):
                for j in range(n_nodes):
                    w = float(weights[i, j])
                    if w != 0.0:
                        src_label = str(node_labels[i]).replace(" ", "_").replace("-", "_")
                        tgt_label = str(node_labels[j]).replace(" ", "_").replace("-", "_")
                        edges.append((src_label, tgt_label, {'weight': w, 'delay': 0.0}))
else:
    # Single model case
    name = model.name or "tvbo_model"
    op_name = f"{name}_op"
    circuit_name = f"{name}_circuit"
%>\
# PyRates Network Template
# Generated from TVBO ${"Network" if is_network else "Dynamics"}

% if is_network:
## Node Templates (reference operators defined in model files)
% for node_id, model_name in node_models.items():
${model_name}:
  base: NodeTemplate
  operators:
    - ${model_name}_op

% endfor

## Circuit Template (network topology)
${circuit_name}:
  base: CircuitTemplate
  nodes:
% for node_id, model_name in node_models.items():
    ${node_id}: ${model_name}
% endfor
% if edges:
  edges:
% for src, tgt, data in edges:
<%
src_model = node_models.get(src, "unknown")
tgt_model = node_models.get(tgt, "unknown")
weight = data.get('weight', 1.0)
delay = data.get('delay', 0.0)
# Default variable names (x for source, c_in for target coupling)
src_var = "x"
tgt_var = "c_in"
# Try to get actual variable names from graph if available
if hasattr(network, 'graph') and network.graph is not None:
    src_node_data = network.graph.nodes.get(src, {})
    tgt_node_data = network.graph.nodes.get(tgt, {})
    src_m = src_node_data.get('model')
    tgt_m = tgt_node_data.get('model')
    if src_m and src_m.state_variables:
        src_var = list(src_m.state_variables.keys())[0]
    if tgt_m and tgt_m.coupling_inputs:
        tgt_var = list(tgt_m.coupling_inputs.keys())[0]
elif context.get('model'):
    base_m = context.get('model')
    if base_m.state_variables:
        src_var = list(base_m.state_variables.keys())[0]
    if base_m.coupling_inputs:
        tgt_var = list(base_m.coupling_inputs.keys())[0]
%>\
    - [${src}/${src_model}_op/${src_var}, ${tgt}/${tgt_model}_op/${tgt_var}, null, {weight: ${weight}, delay: ${delay}}]
% endfor
% endif

% else:
## Single Node Template
${name}:
  base: NodeTemplate
  operators:
    - ${op_name}

## Single Node Circuit
${circuit_name}:
  base: CircuitTemplate
  nodes:
    p: ${name}

% endif
