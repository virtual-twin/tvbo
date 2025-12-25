# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Loss Function Template
================================

Generates loss function code for tvboptim optimization.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Loss function definitions
- Metric functions (correlation, RMSE, etc.)
</%doc>
<%
# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1
dt = experiment.integration.step_size if experiment.integration else 1.0

# Get observations
observations = getattr(experiment, 'observations', None) or []
if hasattr(observations, 'values'):
    observations = list(observations.values())

has_fc = any('fc' in str(getattr(obs, 'name', '')).lower() for obs in observations)
%>
"""Loss function definitions for tvboptim optimization."""

import jax
import jax.numpy as jnp
import jax.scipy.signal


# =============================================================================
# Observable Functions
# =============================================================================

def spectrum(model_fn, state, dt: float = ${dt}, downsample: int = 10, fs: float = None):
    """Compute power spectral density using Welch's method."""
    result = model_fn(state)
    if fs is None:
        fs = 1000.0 / (dt * downsample)
    data = result.data[::downsample, 0, :]
    f, Pxx = jax.scipy.signal.welch(data.T, fs=fs)
    return f, Pxx


def avg_spectrum(model_fn, state, **kwargs):
    """Compute average power spectrum across all regions."""
    f, Pxx = spectrum(model_fn, state, **kwargs)
    return f, jnp.mean(Pxx, axis=0)


def peak_frequency(model_fn, state, **kwargs):
    """Extract peak frequency from average spectrum."""
    f, S = avg_spectrum(model_fn, state, **kwargs)
    return f[jnp.argmax(S)]


# =============================================================================
# Metric Functions
# =============================================================================

def correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Pearson correlation between two arrays."""
    return jnp.corrcoef(x.ravel(), y.ravel())[0, 1]


def rmse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Root mean squared error between two arrays."""
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def mse(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Mean squared error between two arrays."""
    return jnp.mean((x - y) ** 2)


# =============================================================================
# Loss Functions
# =============================================================================

def loss_spectral_correlation(model_fn, state, target_psds: jnp.ndarray):
    """Loss based on correlation between predicted and target PSDs."""
    f, predicted_psd = spectrum(model_fn, state)
    correlations = jax.vmap(lambda pred, targ: jnp.corrcoef(pred, targ)[0, 1])(
        predicted_psd, target_psds
    )
    loss_value = 1.0 - correlations.mean()
    return loss_value, predicted_psd


% if has_fc:
def functional_connectivity(model_fn, state, skip_t: int = 20):
    """Compute functional connectivity from simulation."""
    from tvboptim.observations.observation import compute_fc
    result = model_fn(state)
    return compute_fc(result, skip_t=skip_t)


def loss_fc_correlation(model_fn, state, target_fc: jnp.ndarray, skip_t: int = 20):
    """Loss based on correlation between simulated and target FC."""
    from tvboptim.observations.observation import fc_corr
    fc_sim = functional_connectivity(model_fn, state, skip_t)
    return 1.0 - fc_corr(fc_sim, target_fc)
% endif


def make_loss_fn(model_fn, target_data, loss_type: str = "spectral_correlation"):
    """Create a loss function closure for optimization."""
    if loss_type == "spectral_correlation":
        return lambda state: loss_spectral_correlation(model_fn, state, target_data)
% if has_fc:
    elif loss_type == "fc_correlation":
        return lambda state: (loss_fc_correlation(model_fn, state, target_data), None)
% endif
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
