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

def get_node_models(network):
    """Extract unique models from network nodes."""
    models = {}
    for node_id in network.graph.nodes:
        node_data = network.graph.nodes[node_id]
        if 'model' in node_data:
            m = node_data['model']
            model_name = m.name or f"model_{node_id}"
            models[node_id] = model_name
    return models

def get_edges(network):
    """Extract edges with coupling info."""
    edges = []
    is_multi = hasattr(network.graph, 'edges')
    for edge in network.graph.edges(data=True):
        if len(edge) == 3:
            src, tgt, data = edge
        else:
            src, tgt, key, data = edge
        edges.append((src, tgt, data))
    return edges

if is_network:
    circuit_name = getattr(network, 'name', None) or "tvbo_circuit"
    node_models = get_node_models(network)
    edges = get_edges(network)
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
# Get first state variable as source, first coupling term as target
src_node_data = network.graph.nodes[src]
tgt_node_data = network.graph.nodes[tgt]
src_m = src_node_data.get('model')
tgt_m = tgt_node_data.get('model')
src_var = list(src_m.state_variables.keys())[0] if src_m and src_m.state_variables else "x"
tgt_var = list(tgt_m.coupling_terms.keys())[0] if tgt_m and tgt_m.coupling_terms else src_var
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
