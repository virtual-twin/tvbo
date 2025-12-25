# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Loss Function Template
================================

Generates loss function code for tvboptim optimization
from a TVBO experiment specification.

Context Variables:
- experiment: SimulationExperiment instance (required)
- optimization: List of Optimization specs (from experiment.optimization)

Output:
- Loss function definitions
- Observation functions (spectrum, peak_freq, FC, etc.)
- Metric functions (correlation, RMSE, etc.)
</%doc>
<%
import numpy as np
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1
dt = experiment.integration.step_size if experiment.integration else 1.0

# Get optimization specifications (new schema)
optim_list = getattr(experiment, 'optimization', None) or []
if not isinstance(optim_list, list):
    optim_list = [optim_list]

# Extract loss function from optimization spec
loss_equation = None
loss_label = 'loss'
for opt in optim_list:
    if hasattr(opt, 'loss') and opt.loss:
        loss_eq = opt.loss
        loss_equation = getattr(loss_eq, 'rhs', None) or getattr(loss_eq, 'righthandside', None)
        loss_label = getattr(loss_eq, 'label', 'loss')
        break

# Get observations for deriving observables
observations = getattr(experiment, 'observations', None) or []
if hasattr(observations, 'values'):
    observations = list(observations.values())

# Check for spectral analysis
has_spectral = any('spectrum' in str(getattr(obs, 'name', '')).lower() or 'psd' in str(getattr(obs, 'name', '')).lower()
                   for obs in observations)
has_fc = any('fc' in str(getattr(obs, 'name', '')).lower() or 'connectivity' in str(getattr(obs, 'name', '')).lower()
             for obs in observations)

jaxcode = lambda expr: render_expression(expr, format='jax') if expr else 'None'
%>
"""Loss function definitions for tvboptim optimization.

Auto-generated from TVBO experiment specification.
"""

import jax
import jax.numpy as jnp
import jax.scipy.signal


# ============================================================================
# Observable Functions
# ============================================================================

def spectrum(
    model_fn,
    state,
    dt: float = ${dt},
    downsample: int = 10,
    fs: float = None,
):
    """Compute power spectral density for each region using Welch's method.

    Parameters
    ----------
    model_fn : callable
        Compiled model function from prepare()
    state : State
        Simulation state (parameters + initial conditions)
    dt : float
        Integration timestep in ms
    downsample : int
        Downsampling factor before spectral analysis
    fs : float, optional
        Sampling frequency (computed from dt/downsample if not provided)

    Returns
    -------
    tuple
        (frequencies, power_spectra) where power_spectra has shape [n_nodes, n_freqs]
    """
    # Run simulation
    result = model_fn(state)

    # Compute sampling frequency
    if fs is None:
        fs = 1000.0 / (dt * downsample)

    # Downsample and compute spectrum
    data = result.data[::downsample, 0, :]  # [time, nodes]
    f, Pxx = jax.scipy.signal.welch(data.T, fs=fs)

    return f, Pxx


def avg_spectrum(model_fn, state, **kwargs):
    """Compute average power spectrum across all regions.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state

    Returns
    -------
    tuple
        (frequencies, average_spectrum)
    """
    f, Pxx = spectrum(model_fn, state, **kwargs)
    avg_S = jnp.mean(Pxx, axis=0)
    return f, avg_S


def peak_frequency(model_fn, state, **kwargs):
    """Extract peak frequency from average spectrum.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state

    Returns
    -------
    float
        Mean peak frequency across all regions
    """
    f, S = avg_spectrum(model_fn, state, **kwargs)
    idx = jnp.argmax(S)
    return f[idx]


def regional_peak_frequencies(model_fn, state, **kwargs):
    """Extract peak frequency for each region.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state

    Returns
    -------
    jnp.ndarray
        Peak frequencies [n_nodes]
    """
    f, Pxx = spectrum(model_fn, state, **kwargs)
    peak_indices = jnp.argmax(Pxx, axis=1)
    return f[peak_indices]


% if has_fc:
def functional_connectivity(
    model_fn,
    state,
    skip_t: int = 20,
    bold_monitor=None,
):
    """Compute functional connectivity from simulation.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state
    skip_t : int
        Time points to skip (transient removal)
    bold_monitor : Bold, optional
        BOLD monitor for transformation

    Returns
    -------
    jnp.ndarray
        FC matrix [n_nodes, n_nodes]
    """
    from tvboptim.observations.observation import compute_fc

    result = model_fn(state)

    if bold_monitor is not None:
        bold_result = bold_monitor(result)
        return compute_fc(bold_result, skip_t=skip_t)

    return compute_fc(result, skip_t=skip_t)
% endif


# ============================================================================
# Metric Functions
# ============================================================================

def correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Pearson correlation between two arrays.

    Parameters
    ----------
    x, y : jnp.ndarray
        Arrays to correlate

    Returns
    -------
    float
        Correlation coefficient
    """
    return jnp.corrcoef(x.ravel(), y.ravel())[0, 1]


def mean_correlation(X: jnp.ndarray, Y: jnp.ndarray) -> float:
    """Mean correlation across rows of X and Y.

    Parameters
    ----------
    X, Y : jnp.ndarray
        Arrays with shape [n_samples, n_features]

    Returns
    -------
    float
        Mean correlation across samples
    """
    correlations = jax.vmap(lambda x, y: jnp.corrcoef(x, y)[0, 1])(X, Y)
    return correlations.mean()


def rmse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Root mean squared error between two arrays.

    Parameters
    ----------
    x, y : jnp.ndarray
        Arrays to compare

    Returns
    -------
    float
        RMSE value
    """
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def mse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Mean squared error between two arrays."""
    return jnp.mean((x - y) ** 2)


# ============================================================================
# Loss Functions
# ============================================================================

% if loss_equation:
def loss(model_fn, state, target_data):
    """${loss_label}

    Auto-generated from optimization.loss specification.
    Equation: ${loss_equation}

    Parameters
    ----------
    model_fn : callable
        Compiled model function from prepare()
    state : State
        Simulation state with parameters
    target_data : jnp.ndarray
        Target observations (e.g., target PSDs)

    Returns
    -------
    tuple
        (loss_value, aux_data) for optimization with has_aux=True
    """
    # Compute predicted spectrum
    f, predicted_psd = spectrum(model_fn, state)

    # Target PSDs
    target_psd = target_data

    # Loss equation: ${loss_equation}
    # Correlation between each predicted and target PSD
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted_psd, target_psd
    )
    loss_value = 1.0 - correlations.mean()

    return loss_value, predicted_psd

% endif
def loss_spectral_correlation(
    model_fn,
    state,
    target_psds: jnp.ndarray,
):
    """Loss based on correlation between predicted and target PSDs.

    This is the loss used in JR peak frequency optimization:
    loss = 1 - mean_correlation(predicted_psd, target_psd)

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state with parameters
    target_psds : jnp.ndarray
        Target power spectral densities [n_nodes, n_freqs]

    Returns
    -------
    tuple
        (loss_value, predicted_psds) for aux data
    """
    # Compute predicted spectrum
    f, predicted_psd = spectrum(model_fn, state)

    # Correlation between each predicted and target PSD
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted_psd, target_psds
    )

    # Loss = 1 - mean correlation
    loss_value = 1.0 - correlations.mean()

    return loss_value, predicted_psd


% if has_fc:
def loss_fc_correlation(
    model_fn,
    state,
    target_fc: jnp.ndarray,
    bold_monitor=None,
    skip_t: int = 20,
):
    """Loss based on correlation between simulated and target FC.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state
    target_fc : jnp.ndarray
        Target FC matrix [n_nodes, n_nodes]
    bold_monitor : Bold, optional
        BOLD monitor for transformation
    skip_t : int
        Transient removal time points

    Returns
    -------
    float
        Loss value (1 - FC correlation)
    """
    from tvboptim.observations.observation import fc_corr

    fc_sim = functional_connectivity(model_fn, state, skip_t, bold_monitor)

    # Extract upper triangle for comparison
    corr = fc_corr(fc_sim, target_fc)

    return 1.0 - corr


def loss_fc_rmse(
    model_fn,
    state,
    target_fc: jnp.ndarray,
    bold_monitor=None,
    skip_t: int = 20,
):
    """Loss based on RMSE between simulated and target FC.

    Parameters
    ----------
    model_fn : callable
        Compiled model function
    state : State
        Simulation state
    target_fc : jnp.ndarray
        Target FC matrix
    bold_monitor : Bold, optional
        BOLD monitor
    skip_t : int
        Transient removal

    Returns
    -------
    float
        RMSE between upper triangles of FC matrices
    """
    from tvboptim.observations.observation import rmse

    fc_sim = functional_connectivity(model_fn, state, skip_t, bold_monitor)

    return rmse(fc_sim, target_fc)
% endif
% endif


# ============================================================================
# Gradient-Compatible Loss Wrapper
# ============================================================================

def make_loss_fn(model_fn, target_data, loss_type: str = "spectral_correlation"):
    """Create a loss function closure for optimization.

    Parameters
    ----------
    model_fn : callable
        Compiled model function from prepare()
    target_data : jnp.ndarray
        Target data (PSDs, FC matrix, etc.)
    loss_type : str
        Type of loss: "spectral_correlation", "fc_correlation", "fc_rmse"

    Returns
    -------
    callable
        Loss function that takes state and returns (loss_value, aux_data)
    """
    if loss_type == "spectral_correlation":
        def loss_fn(state):
            return loss_spectral_correlation(model_fn, state, target_data)
        return loss_fn
% if has_fc:
    elif loss_type == "fc_correlation":
        def loss_fn(state):
            loss = loss_fc_correlation(model_fn, state, target_data)
            return loss, None
        return loss_fn
    elif loss_type == "fc_rmse":
        def loss_fn(state):
            loss = loss_fc_rmse(model_fn, state, target_data)
            return loss, None
        return loss_fn
% endif
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
