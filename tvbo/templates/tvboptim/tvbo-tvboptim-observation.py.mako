# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Observation/Monitor Template
======================================

Generates observation functions for tvboptim.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Observation functions (spectrum, FC, etc.)
</%doc>
<%
# Get experiment info
model = experiment.local_dynamics
state_names = list(model.state_variables.keys()) if model else ['x']
dt = experiment.integration.step_size if experiment.integration else 0.1

# Get observations
observations = getattr(experiment, 'observations', None) or {}
if hasattr(observations, 'values'):
    observations = dict(observations.items()) if hasattr(observations, 'items') else {}
elif hasattr(observations, '__iter__') and not isinstance(observations, dict):
    observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations)}

# Parse observations
obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': getattr(obs, 'label', ''),
        'description': getattr(obs, 'description', ''),
        'source': None,
        'source_observation': None,
        'equation': None,
    }
    if hasattr(obs, 'source') and obs.source:
        obs_info['source'] = getattr(obs.source, 'name', str(obs.source))
    if hasattr(obs, 'source_observation') and obs.source_observation:
        src_obs = obs.source_observation
        obs_info['source_observation'] = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)
    if hasattr(obs, 'equation') and obs.equation:
        obs_info['equation'] = getattr(obs.equation, 'rhs', None)
    obs_list.append(obs_info)
%>
"""Observation functions for tvboptim Network Dynamics."""

import jax
import jax.numpy as jnp

from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# Module-level model function (set externally)
model = None

% for obs in obs_list:
<%
    obs_name = obs['name']
    obs_source = obs['source']
    obs_src_obs = obs['source_observation']
    obs_eq = obs['equation']
%>

def ${obs_name}(state):
    """${obs['label'] or obs['name']}

    ${obs['description'] or 'Auto-generated observation function.'}
    """
    % if obs_src_obs:
    ## Derived observation - calls another observation function
    % if 'argmax' in str(obs_eq) or 'peak' in obs_name.lower():
    f, S = ${obs_src_obs}(state)
    return f[jnp.argmax(S)]
    % elif 'mean' in str(obs_eq):
    f, Pxx = ${obs_src_obs}(state)
    return f, jnp.mean(Pxx, axis=0)
    % else:
    return ${obs_src_obs}(state)
    % endif
    % else:
    ## Root observation - runs model and extracts state variable
    result = model(state)
    % if obs_source:
    state_idx = ${state_names.index(obs_source) if obs_source in state_names else 0}
    % else:
    state_idx = 0
    % endif
    % if 'welch' in str(obs_eq) or 'spectrum' in obs_name.lower():
    f, Pxx = jax.scipy.signal.welch(result.data[::10, state_idx, :].T, fs=100.0)
    return f, Pxx
    % else:
    return result.data[:, state_idx, :]
    % endif
    % endif

% endfor
