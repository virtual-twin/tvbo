## -*- coding: utf-8 -*-
##
## PyRates Experiment Template (Complete Runnable YAML)
## =====================================================
## Combines Model (OperatorTemplate) + Network (Node/Circuit) into a
## complete, self-contained PyRates YAML file ready for simulation.
##
## This is the template to use when exporting for actual PyRates execution.
##
<%namespace name="pyrates_model" file="tvbo-pyrates-model.yaml.mako"/>
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

def get_node_labels(network, n_nodes):
    """Get node labels from Network."""
    if hasattr(network, 'node_labels') and network.node_labels:
        return list(network.node_labels)
    elif hasattr(network, 'region_labels') and network.region_labels:
        return list(network.region_labels)
    else:
        return [f"node_{i}" for i in range(n_nodes)]

# Collect all operators (dynamics models) and nodes
operators = {}  # op_name -> Dynamics model (not dict anymore)
nodes = {}
edges = []

if is_network:
    circuit_name = getattr(network, 'label', None) or getattr(network, 'name', None) or "tvbo_circuit"

    # Check if network has schema-based nodes with individual dynamics
    dynamics_library = context.get('dynamics') or {}

    # Get default dynamics (first in library) for nodes without explicit dynamics
    default_dyn = None
    if dynamics_library:
        default_dyn = next(iter(dynamics_library.values()))

    if hasattr(network, 'nodes') and network.nodes:
        # Schema-based Network with nodes list
        node_dynamics_map = {}  # node_id -> Dynamics object

        for node in network.nodes:
            node_id = node.id
            node_label = getattr(node, 'label', None) or f"node_{node_id}"
            safe_label = str(node_label).replace(" ", "_").replace("-", "_")

            # Get dynamics for this node
            dyn = None
            if hasattr(node, 'dynamics') and node.dynamics:
                if isinstance(node.dynamics, str):
                    # Reference to dynamics library
                    dyn = dynamics_library.get(node.dynamics)
                else:
                    # Inline dynamics object
                    dyn = node.dynamics

            # Fall back to default dynamics if none specified
            if dyn is None:
                dyn = default_dyn

            if dyn:
                model_name = getattr(dyn, 'name', None) or f"model_{node_id}"
                op_name = f"{model_name}_op"
                if op_name not in operators:
                    operators[op_name] = dyn  # Store the model, not a dict
                nodes[safe_label] = model_name
                node_dynamics_map[node_id] = dyn
            else:
                nodes[safe_label] = "node"
                node_dynamics_map[node_id] = None

        def get_edge_param(edge, name, default=0.0):
            """Edge weight/delay/distance as a float, via the shared reader."""
            from tvbo.utils import edge_param
            val = edge_param(edge, name)
            return float(val) if val is not None else default

        # Collect edges from network.edges
        if hasattr(network, 'edges') and network.edges:
            for edge in network.edges:
                src_id = edge.source
                tgt_id = edge.target
                # Get weight, delay, distance from edge.parameters
                weight = get_edge_param(edge, 'weight', 1.0)
                delay = get_edge_param(edge, 'delay', 0.0)
                distance = get_edge_param(edge, 'distance', 0.0)

                # If delay not set but distance is, compute delay from distance and conduction speed
                if delay == 0.0 and distance > 0.0:
                    cs = getattr(network, 'conduction_speed', None)
                    if cs and hasattr(cs, 'value') and cs.value:
                        delay = distance / cs.value

                # Find node labels
                src_label = None
                tgt_label = None
                for n in network.nodes:
                    if n.id == src_id:
                        src_label = getattr(n, 'label', None) or f"node_{src_id}"
                    if n.id == tgt_id:
                        tgt_label = getattr(n, 'label', None) or f"node_{tgt_id}"

                src_label = str(src_label).replace(" ", "_").replace("-", "_")
                tgt_label = str(tgt_label).replace(" ", "_").replace("-", "_")

                src_model = nodes.get(src_label, "node")
                tgt_model = nodes.get(tgt_label, "node")

                src_dyn = node_dynamics_map.get(src_id)
                tgt_dyn = node_dynamics_map.get(tgt_id)
                # Prefer output variable for source, fall back to first state variable
                if src_dyn and src_dyn.output:
                    src_var = src_dyn.output[0] if isinstance(src_dyn.output, list) else list(src_dyn.output.keys())[0]
                elif src_dyn and src_dyn.state_variables:
                    src_var = list(src_dyn.state_variables.keys())[0]
                else:
                    src_var = "x"
                _tgt_ci = getattr(tgt_dyn, 'coupling_inputs', None) or {}
                tgt_var = list(_tgt_ci.keys())[0] if tgt_dyn and _tgt_ci else src_var

                edges.append({
                    'src': src_label,
                    'tgt': tgt_label,
                    'src_op': f"{src_model}_op",
                    'tgt_op': f"{tgt_model}_op",
                    'src_var': src_var,
                    'tgt_var': tgt_var,
                    'weight': weight,
                    'delay': delay,
                })

    # Check if network has a .graph attribute (NetworkX graph-based) - DEPRECATED
    elif hasattr(network, 'graph') and network.graph is not None:
        # NetworkX-based network (deprecated _Network style)
        for node_id in network.graph.nodes:
            node_data = network.graph.nodes[node_id]
            if 'model' in node_data:
                m = node_data['model']
                model_name = m.name or f"model_{node_id}"
                op_name = f"{model_name}_op"
                if op_name not in operators:
                    operators[op_name] = m  # Store the model
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
            # Prefer output variable for source, fall back to first state variable
            if src_m and src_m.output:
                src_var = src_m.output[0] if isinstance(src_m.output, list) else list(src_m.output.keys())[0]
            elif src_m and src_m.state_variables:
                src_var = list(src_m.state_variables.keys())[0]
            else:
                src_var = "x"
            _tgt_ci2 = getattr(tgt_m, 'coupling_inputs', None) or {}
            tgt_var = list(_tgt_ci2.keys())[0] if tgt_m and _tgt_ci2 else src_var

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
        weights = network.matrix("weight") if hasattr(network, 'matrix') else None
        n_nodes = getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)
        if weights is not None:
            n_nodes = weights.shape[0]
        node_labels = get_node_labels(network, n_nodes)

        if base_model:
            model_name = base_model.name or "tvbo_model"
            op_name = f"{model_name}_op"
            operators[op_name] = base_model  # Store the model
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
                _base_ci = getattr(base_model, 'coupling_inputs', None) or {}
                if _base_ci:
                    tgt_var = list(_base_ci.keys())[0]

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
    operators[op_name] = model  # Store the model
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
% for op_name, dyn_model in operators.items():

${pyrates_model.render_operator(dyn_model, op_name)}
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
