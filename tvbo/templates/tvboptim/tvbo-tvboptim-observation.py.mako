# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Observation/Monitor Template
======================================

Generates observation and monitor configurations for tvboptim
from a TVBO experiment specification.

Context Variables:
- experiment: SimulationExperiment instance (required)
- observations: List of Observation specs (from experiment.observations)

Output:
- Python code for:
  - Observation functions (spectrum, peak_freq, etc.)
  - BOLD monitor (if specified)
  - FC computation functions
  - Target observation functions
</%doc>
<%
import numpy as np
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
dt = experiment.integration.step_size if experiment.integration else 0.1

# Get observations
observations = getattr(experiment, 'observations', None) or []
if hasattr(observations, 'values'):
    observations = list(observations.values())

# Parse observation info
obs_list = []
for obs in observations:
    obs_info = {
        'name': getattr(obs, 'name', f'obs_{len(obs_list)}'),
        'label': getattr(obs, 'label', ''),
        'description': getattr(obs, 'description', ''),
        'equation': None,
        'source': None,
        'source_observation': None,
        'params': {},
    }
    # Get equation
    if hasattr(obs, 'equation') and obs.equation:
        eq = obs.equation
        obs_info['equation'] = getattr(eq, 'rhs', None) or getattr(eq, 'righthandside', None)
    # Get source state variable
    if hasattr(obs, 'source') and obs.source:
        obs_info['source'] = getattr(obs.source, 'name', str(obs.source))
    # Get source observation (for chained observations)
    if hasattr(obs, 'source_observation') and obs.source_observation:
        src_obs = obs.source_observation
        obs_info['source_observation'] = getattr(src_obs, 'name', str(src_obs))
    # Get parameters
    if hasattr(obs, 'parameters') and obs.parameters:
        for pname, param in obs.parameters.items():
            obs_info['params'][pname] = getattr(param, 'value', None)
    obs_list.append(obs_info)

# Check for BOLD/imaging observations
has_bold = any(getattr(obs, 'imaging_modality', None) == 'BOLD' for obs in observations)

# State variable names for VOI
state_names = list(model.state_variables.keys()) if model else ['x']
voi_name = state_names[0]

jaxcode = lambda expr: render_expression(expr, format='jax') if expr else 'None'
%>
"""Observation and monitor configurations for tvboptim Network Dynamics.

Auto-generated from TVBO experiment specification.

Observations defined:
% for obs in obs_list:
- ${obs['name']}: ${obs['label'] or obs['description'] or 'Observation function'}
% endfor
"""

import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

% if has_bold:
from tvboptim.observations.tvb_monitors.bold import Bold, LotkaVolterraHRFKernel
% endif


# ============================================================================
# Observation Functions (from TVBO Specification)
# ============================================================================

% for obs in obs_list:
<%
    obs_name = obs['name']
    obs_eq = obs['equation']
    obs_source = obs['source']
    obs_src_obs = obs['source_observation']
    obs_params = obs['params']
%>
def ${obs_name}(model_fn_or_result, state=None):
    """${obs['label'] or obs['name']}

    ${obs['description'] or 'Auto-generated observation function.'}

    % if obs_params:
    Parameters (from specification):
    % for pname, pval in obs_params.items():
    - ${pname}: ${pval}
    % endfor
    % endif

    Parameters
    ----------
    model_fn_or_result : callable or NativeSolution
        Either model function (state -> result) or pre-computed result
    state : State, optional
        Simulation state (required if model_fn_or_result is callable)

    Returns
    -------
    Observation value
    """
    # Get simulation result
    if callable(model_fn_or_result):
        result = model_fn_or_result(state)
    else:
        result = model_fn_or_result

    % for pname, pval in obs_params.items():
    ${pname} = ${pval}
    % endfor

    % if obs_source:
    # Extract source state variable: ${obs_source}
    state_idx = ${state_names.index(obs_source) if obs_source in state_names else 0}
    raw_data = result.data[:, state_idx, :]
    % elif obs_src_obs:
    # Chain from source observation: ${obs_src_obs}
    # NOTE: Call ${obs_src_obs}() first and pass result here
    raw_data = model_fn_or_result  # Expects pre-computed observation
    % else:
    raw_data = result.data[:, 0, :]  # Default to first state variable
    % endif

    % if obs_eq:
    # Equation: ${obs_eq}
    % if 'welch' in str(obs_eq):
    # Spectrum computation using Welch's method
    fs = ${obs_params.get('fs', 100)}
    subsample = int(1000 / (${dt} * fs))  # Subsample to target fs
    f, Pxx = jax.scipy.signal.welch(raw_data[::subsample].T, fs=fs)
    return f, Pxx
    % elif 'argmax' in str(obs_eq):
    # Peak extraction
    f, S = raw_data  # Expects (frequencies, spectrum) tuple
    return f[jnp.argmax(S)]
    % elif 'mean' in str(obs_eq) and 'axis' in str(obs_eq):
    # Mean along axis
    return jnp.mean(raw_data, axis=0)
    % elif 'cauchy' in str(obs_eq).lower():
    # Cauchy/Lorentzian distribution for target PSDs
    # This is typically used for generating target data
    def cauchy_pdf(x, x0, gamma):
        return 1 / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))
    # Return the function for external use
    return cauchy_pdf
    % else:
    # Generic equation - attempt to evaluate
    # ${obs_eq}
    return raw_data
    % endif
    % else:
    return raw_data
    % endif


% endfor

# ============================================================================
# Convenience Functions
# ============================================================================

def get_observation_chain(model_fn, state):
    """Execute all observations in sequence.

    Returns a dictionary with all observation results.
    """
    result = model_fn(state)
    observations = {}

    % for obs in obs_list:
    % if not obs['source_observation']:
    observations['${obs['name']}'] = ${obs['name']}(result)
    % endif
    % endfor

    # Chained observations
    % for obs in obs_list:
    % if obs['source_observation']:
    observations['${obs['name']}'] = ${obs['name']}(observations['${obs['source_observation']}'])
    % endif
    % endfor

    return observations


# ============================================================================
# Monitor Configurations
# ============================================================================

% if has_bold:
def get_bold_monitor(
    period: float = ${bold_period},
    downsample_period: float = ${downsample_period},
    voi: int = ${voi_index},
    history: NativeSolution = None,
    **kwargs
) -> Bold:
    """Create BOLD signal monitor.

    Parameters
    ----------
    period : float
        BOLD sampling period in ms (default: ${bold_period} ms = 1 TR)
    downsample_period : float
        Intermediate downsampling period in ms (default: ${downsample_period} ms)
    voi : int
        Index of variable of interest to monitor (default: ${voi_index} = ${voi_name})
    history : NativeSolution, optional
        Previous simulation result for warm-starting the monitor
    **kwargs : dict
        Additional parameters passed to Bold constructor

    Returns
    -------
    Bold
        Configured BOLD monitor instance
    """
    return Bold(
        period=period,
        downsample_period=downsample_period,
        voi=voi,
        history=history,
        **kwargs
    )
% endif

% if has_temporal_average:
def get_temporal_average_monitor(
    period: float = ${downsample_period * 10},
    **kwargs
) -> TemporalAverage:
    """Create temporal averaging monitor.

    Parameters
    ----------
    period : float
        Averaging period in ms
    **kwargs : dict
        Additional parameters

    Returns
    -------
    TemporalAverage
        Configured temporal average monitor
    """
    return TemporalAverage(period=period, **kwargs)
% endif


# ============================================================================
# Observation Functions
# ============================================================================

def compute_functional_connectivity(
    result: NativeSolution,
    voi: int = ${voi_index},
    skip_t: int = 0,
    bold_monitor: Bold = None,
) -> jnp.ndarray:
    """Compute functional connectivity matrix from simulation result.

    Parameters
    ----------
    result : NativeSolution
        Simulation result from tvboptim network dynamics
    voi : int
        Variable of interest index (default: ${voi_index})
    skip_t : int
        Number of initial time points to skip for transient removal
    bold_monitor : Bold, optional
        If provided, applies BOLD transformation before computing FC

    Returns
    -------
    jnp.ndarray
        Functional connectivity matrix [n_nodes, n_nodes]
    """
    if bold_monitor is not None:
        # Apply BOLD transformation first
        bold_result = bold_monitor(result)
        return compute_fc(bold_result, s_var=0, skip_t=skip_t)
    else:
        return compute_fc(result, s_var=voi, skip_t=skip_t)


def fc_correlation(fc_sim: jnp.ndarray, fc_target: jnp.ndarray) -> float:
    """Compute correlation between simulated and target FC matrices.

    Parameters
    ----------
    fc_sim : jnp.ndarray
        Simulated functional connectivity matrix
    fc_target : jnp.ndarray
        Target (empirical) functional connectivity matrix

    Returns
    -------
    float
        Pearson correlation coefficient between upper triangular elements
    """
    return fc_corr(fc_sim, fc_target)


def fc_rmse(fc_sim: jnp.ndarray, fc_target: jnp.ndarray) -> float:
    """Compute RMSE between simulated and target FC matrices.

    Parameters
    ----------
    fc_sim : jnp.ndarray
        Simulated functional connectivity matrix
    fc_target : jnp.ndarray
        Target (empirical) functional connectivity matrix

    Returns
    -------
    float
        Root mean squared error between matrices
    """
    return rmse(fc_sim, fc_target)


% if observations:
# ============================================================================
# Custom Observations from TVBO Specification
# ============================================================================

<%
from tvbo.export.code import render_expression
jaxcode = lambda expr: render_expression(expr, format='jax')
%>

% for obs in observations:
<%
obs_name = obs.name if hasattr(obs, 'name') else f'observation_{loop.index}'
obs_fn = obs.function if hasattr(obs, 'function') else None
%>
def ${obs_name}(result, params=None):
    """${obs.description if hasattr(obs, 'description') else 'Custom observation function.'}

    Auto-generated from TVBO observation specification.

    Parameters
    ----------
    result : NativeSolution
        Simulation result
    params : dict, optional
        Additional parameters

    Returns
    -------
    Observation value
    """
    % if obs_fn:
    return ${jaxcode(obs_fn)}
    % else:
    # Placeholder - implement observation logic
    return result.data
    % endif

% endfor
% endif


# ============================================================================
# Loss Functions for Optimization
# ============================================================================

def create_fc_loss(
    fc_target: jnp.ndarray,
    bold_monitor: Bold = None,
    skip_t: int = 20,
    metric: str = 'rmse',
):
    """Create a loss function for FC fitting.

    Parameters
    ----------
    fc_target : jnp.ndarray
        Target functional connectivity matrix
    bold_monitor : Bold, optional
        BOLD monitor for signal transformation
    skip_t : int
        Time points to skip for transient
    metric : str
        Loss metric: 'rmse' or 'correlation'

    Returns
    -------
    Callable
        Loss function (state) -> float
    """
    def loss_fn(model_fn, state):
        # Run simulation
        result = model_fn(state)

        # Compute FC
        fc_sim = compute_functional_connectivity(
            result,
            bold_monitor=bold_monitor,
            skip_t=skip_t
        )

        # Compute loss
        if metric == 'rmse':
            return fc_rmse(fc_sim, fc_target)
        else:  # correlation
            return 1.0 - fc_correlation(fc_sim, fc_target)

    return loss_fn
