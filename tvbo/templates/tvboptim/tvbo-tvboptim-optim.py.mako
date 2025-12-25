# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Optimization Template
===============================

Generates optimization/fitting code for tvboptim.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Parameter marking functions
- Optimizer configuration
- Optimization execution
</%doc>
<%
# Get experiment info
model = experiment.local_dynamics
network = experiment.network
coupling = experiment.coupling
n_nodes = network.number_of_regions if network else 1

# Get optimization specifications
optim_list = getattr(experiment, 'optimization', None) or []
if not isinstance(optim_list, list):
    optim_list = [optim_list]

# Extract optimizable parameters
optim_params = []
for name, param in model.parameters.items():
    if getattr(param, 'free', False) or (hasattr(param, 'domain') and param.domain):
        optim_params.append(param)

coupling_optim_params = []
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        if getattr(param, 'free', False) or (hasattr(param, 'domain') and param.domain):
            coupling_optim_params.append(param)

# Extract optimizer settings
optimizer_name = 'adamaxw'
learning_rate = 0.001
max_steps = 100

for opt in optim_list:
    if hasattr(opt, 'algorithm') and opt.algorithm:
        optimizer_name = str(opt.algorithm)
    if hasattr(opt, 'learning_rate') and opt.learning_rate:
        learning_rate = float(opt.learning_rate)
    if hasattr(opt, 'max_iterations') and opt.max_iterations:
        max_steps = int(opt.max_iterations)
%>
"""Optimization configuration for tvboptim."""

import copy
import optax
import jax.numpy as jnp

from tvboptim.types import Parameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback


def mark_parameters_optimizable(state, n_nodes: int = ${n_nodes}):
    """Mark parameters as optimizable and set their shapes."""
    init_state = copy.deepcopy(state)

    % for param in optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
%>
    init_state.dynamics.${param_name} = Parameter(init_state.dynamics.${param_name})
    % if is_heterogeneous:
    init_state.dynamics.${param_name}.shape = (n_nodes,)
    % endif
    % endfor

    % for param in coupling_optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False)
%>
    init_state.coupling.${param_name} = Parameter(init_state.coupling.${param_name})
    % if is_heterogeneous:
    init_state.coupling.${param_name}.shape = (n_nodes,)
    % endif
    % endfor

    return init_state


def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    print_every: int = 10,
    has_aux: bool = True,
):
    """Create configured optimizer."""
    optimizers = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "adamax": optax.adamax,
        "adamaxw": optax.adamaxw,
        "sgd": optax.sgd,
    }
    opt_fn = optimizers.get(optimizer, optax.adamaxw)
    callback = MultiCallback([DefaultPrintCallback(every=print_every)])
    return OptaxOptimizer(loss_fn, opt_fn(learning_rate), callback=callback, has_aux=has_aux)


def run_optimization(
    init_state,
    loss_fn,
    max_steps: int = ${max_steps},
    learning_rate: float = ${learning_rate},
    optimizer: str = "${optimizer_name}",
    **kwargs,
):
    """Run gradient-based optimization."""
    opt = create_optimizer(loss_fn, optimizer=optimizer, learning_rate=learning_rate, **kwargs)
    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps)
    return fitted_params, fitting_data
