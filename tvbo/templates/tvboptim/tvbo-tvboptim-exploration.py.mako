# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Parameter Exploration Template
========================================

Generates parameter exploration (grid search) code for tvboptim.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Grid search configuration and execution functions
</%doc>
<%
# Get experiment info
model = experiment.dynamics
network = experiment.network
n_nodes = N_nodes = (getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)) if network else 1

# Get exploration specifications
exploration_dict = getattr(experiment, 'explorations', None) or {}
if isinstance(exploration_dict, dict):
    exploration_list = list(exploration_dict.values())
elif isinstance(exploration_dict, list):
    exploration_list = exploration_dict
else:
    exploration_list = [exploration_dict] if exploration_dict else []

# Parse explorations
explorations = []
for expl in exploration_list:
    exp_info = {
        'name': getattr(expl, 'name', 'exploration'),
        'label': getattr(expl, 'label', ''),
        'mode': getattr(expl, 'mode', 'product'),
        'n_parallel': getattr(expl, 'n_parallel', 8),
        'axes': [],
    }
    axes_list = getattr(expl, 'space', None) or []
    for axis in axes_list:
        domain = getattr(axis, 'domain', None)
        if domain:
            pname = str(getattr(axis, 'parameter', ''))
            # Strip dotted prefix (Class.param) → bare param name
            if '.' in pname:
                pname = pname.rsplit('.', 1)[1]
            exp_info['axes'].append({
                'name': pname,
                'lo': float(domain.lo) if domain.lo is not None else 0.0,
                'hi': float(domain.hi) if domain.hi is not None else 1.0,
                'n': int(domain.n) if hasattr(domain, 'n') and domain.n else 32,
            })
    observable = getattr(expl, 'observable', None)
    if observable:
        exp_info['observable'] = getattr(observable, 'name', str(observable))
    explorations.append(exp_info)
%>
"""Parameter exploration for tvboptim Network Dynamics."""

import copy
import jax.numpy as jnp

from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution

% for expl in explorations:
<%
    total_points = 1
    for ax in expl['axes']:
        total_points *= ax['n']
%>

def setup_${expl['name']}_grid(state):
    """Configure grid for ${expl['name']} exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    """
    grid_state = copy.deepcopy(state)
    % for ax in expl['axes']:
    grid_state.dynamics.${ax['name']} = GridAxis(low=${ax['lo']}, high=${ax['hi']}, n=${ax['n']})
    % endfor
    return Space(grid_state, mode="${expl['mode']}")


def run_${expl['name']}_exploration(state, observable_fn, n_pmap: int = ${expl['n_parallel']}):
    """Run ${expl['name']} parameter exploration."""
    grid = setup_${expl['name']}_grid(state)
    exec = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec.run()
    return grid, jnp.stack(results)

% endfor
