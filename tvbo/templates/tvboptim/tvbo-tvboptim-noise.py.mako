# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Noise Template
========================

Generates noise configuration for tvboptim.experimental.network_dynamics
from a TVBO integration/noise specification.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- integration: Integration instance with noise attribute

Output:
- Python code for noise model instantiation (AdditiveNoise, MultiplicativeNoise)
</%doc>
<%
import numpy as np

# Get noise configuration from context
if 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.local_dynamics
    noise_sigma = np.asarray(experiment.noise_sigma_array) if hasattr(experiment, 'noise_sigma_array') else None
else:
    integration = context.get('integration', None)
    model = context.get('model', None)
    noise_sigma = context.get('noise_sigma', None)

# Check if stochastic
has_noise = integration is not None and integration.noise is not None

if has_noise:
    noise_config = integration.noise
    # Extract noise parameters
    noise_type = getattr(noise_config, 'type', 'additive').lower() if hasattr(noise_config, 'type') else 'additive'

    # Get sigma values
    if noise_sigma is not None:
        sigma_values = noise_sigma.flatten().tolist()
    elif hasattr(noise_config, 'sigma'):
        sigma_val = noise_config.sigma
        if hasattr(sigma_val, '__iter__'):
            sigma_values = list(sigma_val)
        else:
            sigma_values = [float(sigma_val)]
    elif hasattr(noise_config, 'nsig'):
        # TVB-style nsig: nsig = 0.5 * sigma^2
        nsig = noise_config.nsig
        if hasattr(nsig, '__iter__'):
            sigma_values = [np.sqrt(2 * float(n)) for n in nsig]
        else:
            sigma_values = [np.sqrt(2 * float(nsig))]
    else:
        sigma_values = [0.1]  # Default

    # Get which states receive noise
    apply_to = getattr(noise_config, 'apply_to', None)
    if apply_to is None and model is not None:
        # Default: all state variables
        apply_to = list(model.state_variables.keys())
    elif apply_to is None:
        apply_to = None  # Will apply to all
else:
    noise_type = 'additive'
    sigma_values = [0.0]
    apply_to = None
%>
"""Noise configuration for tvboptim Network Dynamics.

Auto-generated from TVBO integration specification.
Noise type: ${noise_type}
Sigma: ${sigma_values}
"""

import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.noise import AdditiveNoise, MultiplicativeNoise


def get_noise(key=None, **kwargs):
    """Get configured noise instance.

    Parameters
    ----------
    key : jax.random.PRNGKey, optional
        Random key for noise generation. If None, uses default.
    **kwargs : dict
        Override default parameters (sigma, apply_to)

    Returns
    -------
    noise : AbstractNoise or None
        Configured noise model, or None if deterministic
    """
    if key is None:
        key = jax.random.PRNGKey(0)

% if has_noise and sigma_values and any(s != 0 for s in sigma_values):
    # Noise parameters
    sigma = kwargs.get('sigma', ${sigma_values[0] if len(sigma_values) == 1 else sigma_values})
    % if apply_to:
    apply_to = kwargs.get('apply_to', ${repr(apply_to)})
    % else:
    apply_to = kwargs.get('apply_to', None)
    % endif

    % if noise_type == 'multiplicative':
    return MultiplicativeNoise(
        sigma=sigma,
        apply_to=apply_to,
        key=key,
        **{k: v for k, v in kwargs.items() if k not in ('sigma', 'apply_to')}
    )
    % else:
    return AdditiveNoise(
        sigma=sigma,
        apply_to=apply_to,
        key=key,
    )
    % endif
% else:
    # No noise (deterministic)
    return None
% endif


def get_noise_params():
    """Get noise parameters as a dict.

    Returns
    -------
    dict
        Noise parameters
    """
    return {
        'type': '${noise_type}',
        'sigma': ${sigma_values},
        'apply_to': ${repr(apply_to)},
    }


# Convenience: Export default noise instance
noise = get_noise()
