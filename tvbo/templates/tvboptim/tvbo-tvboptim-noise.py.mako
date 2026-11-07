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
from tvbo.utils import noise_sigma

# Every amplitude is read by tvbo.utils.noise_sigma: per state variable through the experiment, else the integration's own.
if 'experiment' in context.keys():
    model = experiment.dynamics
    noise_config = experiment.run_noise
    sigma_values = [float(s) for s in experiment.noise_sigma_array]
else:
    model = context.get('model', None)
    noise_config = getattr(context.get('integration', None), 'noise', None)
    sigma_values = [noise_sigma(noise_config) or 0.0]

has_noise = noise_config is not None
noise_type = 'additive'
apply_to = None

if has_noise:
    noise_type = getattr(noise_config, 'type', 'additive').lower() if hasattr(noise_config, 'type') else 'additive'
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
