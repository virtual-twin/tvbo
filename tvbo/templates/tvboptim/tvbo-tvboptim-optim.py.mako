# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Optimization Template
===============================

Generates optimization/fitting code for tvboptim
from a TVBO experiment specification.

Context Variables:
- experiment: SimulationExperiment instance (required)
- optimization: List of Optimization specs (from experiment.optimization)

Output:
- Parameter setup with shapes and bounds
- Optimizer configuration
- Optimization loop
- Callbacks
</%doc>
<%
import numpy as np
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1
dt = experiment.integration.step_size if experiment.integration else 1.0

# Get optimization specifications (new schema field)
optim_list = getattr(experiment, 'optimization', None) or []
if not isinstance(optim_list, list):
    optim_list = [optim_list]

# Extract optimizable parameters (those with free=True or domain)
optim_params = []
for name, param in model.parameters.items():
    is_free = getattr(param, 'free', False)
    has_domain = hasattr(param, 'domain') and param.domain is not None
    if is_free or has_domain:
        optim_params.append(param)

# Also check coupling parameters
coupling = experiment.coupling
coupling_optim_params = []
if coupling and hasattr(coupling, 'parameters'):
    for name, param in coupling.parameters.items():
        is_free = getattr(param, 'free', False)
        has_domain = hasattr(param, 'domain') and param.domain is not None
        if is_free or has_domain:
            coupling_optim_params.append(param)

# Extract optimizer settings from first Optimization spec
optimizer_name = 'adamaxw'
learning_rate = 0.001
max_steps = 100
has_aux = True

for opt in optim_list:
    if hasattr(opt, 'algorithm') and opt.algorithm:
        optimizer_name = str(opt.algorithm)
    if hasattr(opt, 'learning_rate') and opt.learning_rate:
        learning_rate = float(opt.learning_rate)
    if hasattr(opt, 'max_iterations') and opt.max_iterations:
        max_steps = int(opt.max_iterations)
    if hasattr(opt, 'hyperparameters') and opt.hyperparameters:
        for hp in opt.hyperparameters.values():
            if hp.name == 'has_aux':
                has_aux = bool(hp.value)
    # Check free_parameters references
    if hasattr(opt, 'free_parameters') and opt.free_parameters:
        for pref in opt.free_parameters:
            pname = pref if isinstance(pref, str) else getattr(pref, 'name', None)
            if pname and pname in model.parameters:
                p = model.parameters[pname]
                if p not in optim_params:
                    optim_params.append(p)
%>
"""Optimization configuration for tvboptim.

Auto-generated from TVBO experiment specification.
"""

import copy
import jax
import jax.numpy as jnp
import optax

from tvboptim.types import Parameter, Space
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import (
    MultiCallback,
    DefaultPrintCallback,
    SavingCallback,
    AbstractCallback,
)


# ============================================================================
# Optimizable Parameters Configuration
# ============================================================================

def mark_parameters_optimizable(state, n_nodes: int = ${n_nodes}):
    """Mark parameters as optimizable and set their shapes.

    This function prepares the simulation state for gradient-based optimization
    by wrapping parameters with Parameter() and setting heterogeneous shapes.

    Parameters
    ----------
    state : State
        Initial simulation state from prepare()
    n_nodes : int
        Number of network nodes (for heterogeneous parameters)

    Returns
    -------
    State
        Modified state with optimizable parameters
    """
    init_state = copy.deepcopy(state)

    # Dynamics parameters
% for param in optim_params:
<%
    param_name = param.name
    is_heterogeneous = getattr(param, 'heterogeneous', False) or getattr(param, 'shape', None)
    has_bounds = hasattr(param, 'domain') and param.domain
    lo = param.domain.lo if has_bounds else None
    hi = param.domain.hi if has_bounds else None
%>
    # ${param_name}: ${'heterogeneous' if is_heterogeneous else 'homogeneous'}
    init_state.dynamics.${param_name} = Parameter(init_state.dynamics.${param_name})
    % if is_heterogeneous:
    init_state.dynamics.${param_name}.shape = (n_nodes,)
    % endif
    % if has_bounds:
    # Bounds: [${lo}, ${hi}]
    % endif

% endfor
% if not optim_params:
    # No optimizable parameters specified in dynamics
    # Example: mark a and b as heterogeneous optimizable
    # init_state.dynamics.a = Parameter(init_state.dynamics.a)
    # init_state.dynamics.a.shape = (n_nodes,)
    # init_state.dynamics.b = Parameter(init_state.dynamics.b)
    # init_state.dynamics.b.shape = (n_nodes,)
    pass
% endif

    # Coupling parameters
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
% if not coupling_optim_params:
    # No optimizable coupling parameters specified
    pass
% endif

    return init_state


# ============================================================================
# Callbacks
# ============================================================================

class PlotCallback(AbstractCallback):
    """Callback to plot optimization progress.

    Shows brain maps or parameter distributions during optimization.
    """
    def __init__(self, every: int = 10, plot_fn=None) -> None:
        super().__init__(every)
        self.plot_fn = plot_fn

    def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
        """Execute callback at specified intervals."""
        if self.plot_fn is not None:
            self.plot_fn(i, diff_state, aux_data, loss_value)
        return False, diff_state, static_state


class ConvergenceCallback(AbstractCallback):
    """Callback to check for convergence and early stopping."""
    def __init__(
        self,
        every: int = 1,
        loss_threshold: float = 0.01,
        patience: int = 10,
    ) -> None:
        super().__init__(every)
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0

    def do(self, i, diff_state, static_state, fitting_data, aux_data, loss_value, grads):
        """Check convergence criteria."""
        # Check threshold
        if loss_value < self.loss_threshold:
            print(f"Converged: loss {loss_value:.6f} < threshold {self.loss_threshold}")
            return True, diff_state, static_state

        # Check improvement (patience-based early stopping)
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping: no improvement for {self.patience} steps")
                return True, diff_state, static_state

        return False, diff_state, static_state


# ============================================================================
# Optimizer Setup
# ============================================================================

def create_optimizer(
    loss_fn,
    optimizer: str = "${optimizer_name}",
    learning_rate: float = ${learning_rate},
    max_steps: int = ${max_steps},
    print_every: int = 10,
    save_path: str = None,
    has_aux: bool = True,
):
    """Create configured optimizer.

    Parameters
    ----------
    loss_fn : callable
        Loss function (state) -> (loss, aux_data)
    optimizer : str
        Optimizer name: "adam", "adamw", "adamax", "adamaxw", "sgd"
    learning_rate : float
        Learning rate
    max_steps : int
        Maximum optimization steps
    print_every : int
        Print progress every N steps
    save_path : str, optional
        Path to save optimization history
    has_aux : bool
        Whether loss function returns auxiliary data

    Returns
    -------
    OptaxOptimizer
        Configured optimizer instance
    """
    # Select optimizer
    optimizers = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "adamax": optax.adamax,
        "adamaxw": optax.adamaxw,
        "sgd": optax.sgd,
    }
    opt_fn = optimizers.get(optimizer, optax.adamaxw)

    # Setup callbacks
    callbacks = [DefaultPrintCallback(every=print_every)]
    if save_path:
        callbacks.append(SavingCallback(path=save_path))

    callback = MultiCallback(callbacks)

    # Create optimizer
    return OptaxOptimizer(
        loss_fn,
        opt_fn(learning_rate),
        callback=callback,
        has_aux=has_aux,
    )


# ============================================================================
# Run Optimization
# ============================================================================

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
    init_state : State
        Initial state with optimizable parameters marked
    loss_fn : callable
        Loss function
    max_steps : int
        Maximum optimization steps
    learning_rate : float
        Learning rate
    optimizer : str
        Optimizer name

    Returns
    -------
    tuple
        (fitted_params, fitting_data) - optimized state and history
    """
    opt = create_optimizer(
        loss_fn,
        optimizer=optimizer,
        learning_rate=learning_rate,
        max_steps=max_steps,
        **kwargs,
    )

    fitted_params, fitting_data = opt.run(init_state, max_steps=max_steps)

    return fitted_params, fitting_data


# ============================================================================
# Grid Exploration (Pre-optimization)
# ============================================================================

def explore_parameter_space(
    observation_fn,
    state,
    param_name: str,
    param_values,
    n_pmap: int = 8,
):
    """Explore parameter space in parallel using grid search.

    Useful for understanding the loss landscape before optimization.

    Parameters
    ----------
    observation_fn : callable
        Observation function (state) -> observable
    state : State
        Base simulation state
    param_name : str
        Parameter name to explore (dot notation: "dynamics.a")
    param_values : array-like
        Values to explore
    n_pmap : int
        Number of parallel evaluations

    Returns
    -------
    jnp.ndarray
        Observation values for each parameter value
    """
    from tvboptim.types import GridAxis, Space
    from tvboptim.execution import ParallelExecution

    # Create grid state
    grid_state = copy.deepcopy(state)

    # Set parameter as GridAxis
    parts = param_name.split('.')
    obj = grid_state
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(
        obj,
        parts[-1],
        GridAxis(low=min(param_values), high=max(param_values), n=len(param_values)),
    )

    # Create space and execute
    grid = Space(grid_state, mode="product")
    exec = ParallelExecution(observation_fn, grid, n_pmap=n_pmap)

    return exec.run()


def explore_2d_parameter_space(
    observation_fn,
    state,
    param1_name: str,
    param1_values,
    param2_name: str,
    param2_values,
    n_pmap: int = 8,
):
    """Explore 2D parameter space.

    Parameters
    ----------
    observation_fn : callable
        Observation function
    state : State
        Base state
    param1_name, param2_name : str
        Parameter names
    param1_values, param2_values : array-like
        Values to explore
    n_pmap : int
        Parallel evaluations

    Returns
    -------
    jnp.ndarray
        2D array of observations [len(param1), len(param2)]
    """
    from tvboptim.types import GridAxis, Space
    from tvboptim.execution import ParallelExecution

    grid_state = copy.deepcopy(state)

    # Set both parameters as GridAxis
    for param_name, values in [(param1_name, param1_values), (param2_name, param2_values)]:
        parts = param_name.split('.')
        obj = grid_state
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], GridAxis(low=min(values), high=max(values), n=len(values)))

    grid = Space(grid_state, mode="product")
    exec = ParallelExecution(observation_fn, grid, n_pmap=n_pmap)

    results = exec.run()
    return jnp.array(results).reshape(len(param1_values), len(param2_values))
