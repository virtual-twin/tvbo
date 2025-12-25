# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Parameter Exploration Template
========================================

Generates parameter exploration (grid search) code for tvboptim
from a TVBO experiment specification.

Context Variables:
- experiment: SimulationExperiment instance (required)
- explorations: List of Exploration specs (from experiment.explorations)

Output:
- Grid search configuration using Space and GridAxis
- Parallel execution setup
- Result collection and visualization
</%doc>
<%
import numpy as np
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1

# Get exploration specifications
exploration_list = getattr(experiment, 'explorations', None) or []
if not isinstance(exploration_list, list):
    exploration_list = [exploration_list]

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
    # Get parameters with domain ranges
    if hasattr(expl, 'parameters') and expl.parameters:
        for param in expl.parameters.values() if hasattr(expl.parameters, 'values') else expl.parameters:
            domain = getattr(param, 'domain', None)
            if domain:
                exp_info['axes'].append({
                    'name': param.name,
                    'lo': float(domain.lo) if domain.lo else 0.0,
                    'hi': float(domain.hi) if domain.hi else 1.0,
                    'n': int(domain.n) if hasattr(domain, 'n') and domain.n else 32,
                    'log_scale': bool(getattr(domain, 'log_scale', False)),
                })
    # Get observable reference
    observable = getattr(expl, 'observable', None)
    if observable:
        exp_info['observable'] = getattr(observable, 'name', str(observable))
    explorations.append(exp_info)
%>
"""Parameter exploration (grid search) for tvboptim Network Dynamics.

Auto-generated from TVBO experiment specification.

Explorations defined:
% for expl in explorations:
- ${expl['name']}: ${expl['label'] or 'Parameter grid search'}
  Axes: ${', '.join([a['name'] for a in expl['axes']])}
  Mode: ${expl['mode']}
% endfor
"""

import copy
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution, SequentialExecution
from tvboptim.utils import cache


# ============================================================================
# Grid Configuration
# ============================================================================

% for expl in explorations:
<%
    n_axes = len(expl['axes'])
    total_points = 1
    for ax in expl['axes']:
        total_points *= ax['n']
%>
def setup_${expl['name']}_grid(state):
    """Configure grid for ${expl['name']} exploration.

    Grid: ${' x '.join([f"{ax['name']}[{ax['n']}]" for ax in expl['axes']])} = ${total_points} points
    Mode: ${expl['mode']}

    Parameters
    ----------
    state : State
        Base simulation state from prepare()

    Returns
    -------
    Space
        Configured parameter space for exploration
    """
    grid_state = copy.deepcopy(state)

    # Configure grid axes
    % for ax in expl['axes']:
    grid_state.dynamics.${ax['name']} = GridAxis(
        low=${ax['lo']},
        high=${ax['hi']},
        n=${ax['n']},
    )
    % endfor

    return Space(grid_state, mode="${expl['mode']}")


def run_${expl['name']}_exploration(state, observable_fn, n_pmap: int = ${expl['n_parallel']}):
    """Run ${expl['name']} parameter exploration.

    Parameters
    ----------
    state : State
        Base simulation state
    observable_fn : callable
        Function (state) -> observable to compute at each grid point
    n_pmap : int
        Number of parallel evaluations (default: ${expl['n_parallel']})

    Returns
    -------
    tuple
        (grid, results) - Space object and array of observable values
    """
    grid = setup_${expl['name']}_grid(state)

    # Run parallel exploration
    exec = ParallelExecution(observable_fn, grid, n_pmap=n_pmap)
    results = exec.run()

    return grid, jnp.stack(results)


% if n_axes == 2:
def plot_${expl['name']}_landscape(grid, results, ax=None, cmap='cividis', **kwargs):
    """Plot 2D parameter landscape for ${expl['name']}.

    Parameters
    ----------
    grid : Space
        Parameter grid from exploration
    results : array
        Observable values at each grid point
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    **kwargs
        Additional arguments for imshow

    Returns
    -------
    matplotlib.image.AxesImage
        The image object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get parameter ranges
    pc = grid.collect()
    <%
        ax1, ax2 = expl['axes'][0], expl['axes'][1]
    %>
    x = pc.dynamics.${ax1['name']}
    y = pc.dynamics.${ax2['name']}

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Reshape results to 2D
    n1, n2 = ${ax1['n']}, ${ax2['n']}
    img_data = results.reshape(n1, n2).T

    im = ax.imshow(
        img_data,
        cmap=cmap,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='auto',
        **kwargs,
    )

    ax.set_xlabel('${ax1['name']}')
    ax.set_ylabel('${ax2['name']}')

    return im

% endif
% endfor

# ============================================================================
# Cached Exploration Functions
# ============================================================================

% for expl in explorations:
@cache("${expl['name']}", redo=False)
def cached_${expl['name']}_exploration(state, observable_fn, n_pmap: int = ${expl['n_parallel']}):
    """Cached version of ${expl['name']} exploration.

    Results are stored to disk and reused on subsequent calls.
    Set redo=True in @cache decorator to force recomputation.
    """
    _, results = run_${expl['name']}_exploration(state, observable_fn, n_pmap)
    return results

% endfor

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # This requires model, state, and observable_fn to be defined
    # from the simulation template

    print("Exploration configurations generated.")
    print("To run explorations, import this module and call:")
% for expl in explorations:
    print(f"  - run_${expl['name']}_exploration(state, observable_fn)")
% endfor
