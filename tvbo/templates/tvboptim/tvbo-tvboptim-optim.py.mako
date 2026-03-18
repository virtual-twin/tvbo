# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Optimization Template
===============================

Generates optimization/fitting code for tvboptim from metadata.

Uses schema ifabsent defaults:
- algorithm: 'adam'
- learning_rate: 0.001
- max_iterations: 100

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Parameter marking functions
- Optimizer configuration
- Optimization execution
</%doc>
<%
# Get experiment info
model = experiment.dynamics
network = experiment.network
coupling = experiment.coupling
n_nodes = N_nodes = (getattr(network, 'number_of_nodes', None) or getattr(network, 'number_of_regions', 1)) if network else 1

# Get optimization specifications
optim_raw = getattr(experiment, 'optimizations', None) or []
if isinstance(optim_raw, dict):
    optim_list = list(optim_raw.values())
elif isinstance(optim_raw, list):
    optim_list = optim_raw
else:
    optim_list = [optim_raw] if optim_raw else []

has_optimization = len(optim_list) > 0

# Extract optimizable parameters from dynamics
optim_params = []
for name, param in model.parameters.items():
    if getattr(param, 'free', False):
        optim_params.append(param)

# Extract optimizable parameters from coupling
coupling_optim_params = []
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        if getattr(param, 'free', False):
            coupling_optim_params.append(param)

# Extract optimizer settings - uses schema ifabsent defaults
optimizer_name = 'adam'
learning_rate = 0.001
max_steps = 100

for opt in optim_list:
    if hasattr(opt, 'algorithm') and opt.algorithm:
        optimizer_name = str(opt.algorithm)
    if hasattr(opt, 'learning_rate') and opt.learning_rate is not None:
        learning_rate = float(opt.learning_rate)
    if hasattr(opt, 'max_iterations') and opt.max_iterations is not None:
        max_steps = int(opt.max_iterations)
%>
"""
Optimization Configuration for tvboptim
========================================

Auto-generated from TVBO optimization metadata.

Optimizer: ${optimizer_name}
Learning rate: ${learning_rate}
Max iterations: ${max_steps}

Optimizable dynamics parameters: ${len(optim_params)}
% for param in optim_params:
- ${param.name}${ ' (heterogeneous)' if getattr(param, 'heterogeneous', False) else ''}
% endfor
% if coupling_optim_params:

Optimizable coupling parameters: ${len(coupling_optim_params)}
% for param in coupling_optim_params:
- ${param.name}${ ' (heterogeneous)' if getattr(param, 'heterogeneous', False) else ''}
% endfor
% endif
"""

import copy
import optax
import jax.numpy as jnp

from tvboptim.types import Parameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback


def mark_parameters_optimizable(state, n_nodes: int = ${n_nodes}):
    """Mark parameters as optimizable and set their shapes.

    Parameters
    ----------
    state : Bunch
        Network state from prepare()
    n_nodes : int
        Number of nodes for heterogeneous parameters

    Returns
    -------
    Bunch
        Deep copy of state with Parameter wrappers for optimizable params
    """
    init_state = copy.deepcopy(state)

% for param in optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
%>
    # ${param_name}${ ' (heterogeneous, shape=(n_nodes,))' if is_heterogeneous else ''}
    init_state.dynamics.${param_name} = Parameter(init_state.dynamics.${param_name})
% if is_heterogeneous:
    init_state.dynamics.${param_name}.shape = (n_nodes,)
% endif
% endfor
% if coupling_optim_params:

    # Coupling parameters (accessed via state.coupling[key].params)
% for param in coupling_optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False)
%>
    # ${param_name}${ ' (heterogeneous)' if is_heterogeneous else ''}
    # Note: Coupling params require different access pattern
    # init_state.coupling['coupling_key'].params.${param_name} = Parameter(...)
% endfor
% endif

    return init_state


def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    print_every: int = 10,
    has_aux: bool = True,
):
    """Create configured optimizer.

    Parameters
    ----------
    loss_fn : callable
        Loss function: state -> (loss_value, aux_data)
    optimizer : str
        Optimizer name: adam, adamw, adamax, adamaxw, sgd
    learning_rate : float
        Learning rate
    print_every : int
        Callback print frequency
    has_aux : bool
        Whether loss_fn returns auxiliary data

    Returns
    -------
    OptaxOptimizer
        Configured optimizer instance
    """
    optimizers = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "adamax": optax.adamax,
        "adamaxw": optax.adamaxw,
        "sgd": optax.sgd,
    }
    opt_fn = optimizers.get(optimizer.lower(), optax.adam)
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
    """Run gradient-based optimization.

    Parameters
    ----------
    init_state : Bunch
        State with Parameter-wrapped optimizable params
    loss_fn : callable
        Loss function: state -> (loss_value, aux_data)
    max_steps : int
        Maximum optimization iterations
    learning_rate : float
        Learning rate
    optimizer : str
        Optimizer name

    Returns
    -------
    tuple
        (fitted_params, fitting_data)
    """
    opt = create_optimizer(
        loss_fn,
        optimizer=optimizer,
        learning_rate=learning_rate,
        **kwargs
    )
    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps)
    return fitted_params, fitting_data
