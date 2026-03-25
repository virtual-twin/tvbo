# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Noise Template
========================

Generates noise configuration for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- integration: Integration instance with noise attribute

Output:
- Noise getter function
</%doc>
<%
import numpy as np

# Get from context
if 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.dynamics
    noise_sigma = np.asarray(experiment.noise_sigma_array) if hasattr(experiment, 'noise_sigma_array') else None
else:
    integration = context.get('integration', None)
    model = context.get('model', None)
    noise_sigma = context.get('noise_sigma', None)

# Parse noise config
has_noise = integration is not None and integration.noise is not None
noise_type = 'additive'
sigma_values = [0.0]
apply_to = None

if has_noise:
    noise_config = integration.noise
    noise_type = getattr(noise_config, 'type', 'additive').lower() if hasattr(noise_config, 'type') else 'additive'

    if noise_sigma is not None:
        sigma_values = noise_sigma.flatten().tolist()
    elif hasattr(noise_config, 'sigma'):
        sigma_val = noise_config.sigma
        sigma_values = list(sigma_val) if hasattr(sigma_val, '__iter__') else [float(sigma_val)]
    elif hasattr(noise_config, 'nsig'):
        nsig = noise_config.nsig
        if hasattr(nsig, '__iter__'):
            sigma_values = [np.sqrt(2 * float(n)) for n in nsig]
        else:
            sigma_values = [np.sqrt(2 * float(nsig))]
    else:
        sigma_values = [0.1]

    apply_to = getattr(noise_config, 'apply_to', None)
    if apply_to is None and model is not None:
        apply_to = list(model.state_variables.keys())
%>

from tvboptim.experimental.network_dynamics.noise import AdditiveNoise, MultiplicativeNoise


def get_noise(key=None, sigma=None, apply_to=None, **kwargs):
    """Get configured noise instance."""
    import jax
    if key is None:
        key = jax.random.PRNGKey(0)

% if has_noise and sigma_values and any(s != 0 for s in sigma_values):
    if sigma is None:
        sigma = ${sigma_values[0] if len(sigma_values) == 1 else sigma_values}
    if apply_to is None:
        apply_to = ${repr(apply_to)}

    % if noise_type == 'multiplicative':
    return MultiplicativeNoise(sigma=sigma, apply_to=apply_to, key=key, **kwargs)
    % else:
    return AdditiveNoise(sigma=sigma, apply_to=apply_to, key=key)
    % endif
% else:
    return None
% endif


noise = get_noise()
