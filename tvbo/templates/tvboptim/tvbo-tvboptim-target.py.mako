# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Target Observation Template
=====================================

Generates target observation/fitting target code for tvboptim optimization
from a TVBO experiment specification.

Context Variables:
- experiment: SimulationExperiment instance (required)
- fitting: ModelFitting instance (optional, from experiment.modelfitting)

Output:
- Target data generation functions
- Target distributions (Cauchy, etc.)
- Empirical data loading
</%doc>
<%
import numpy as np
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1

# Get fitting specifications
fitting_list = experiment.modelfitting if hasattr(experiment, 'modelfitting') and experiment.modelfitting else []
if not isinstance(fitting_list, list):
    fitting_list = [fitting_list]

# Extract targets from fitting specifications
targets = []
for fit in fitting_list:
    if hasattr(fit, 'targets') and fit.targets:
        targets.extend(fit.targets if isinstance(fit.targets, list) else [fit.targets])

# Get observations
observations = experiment.observations if hasattr(experiment, 'observations') and experiment.observations else []
if not isinstance(observations, list):
    observations = [observations] if observations else []
%>
"""Target observation definitions for tvboptim optimization.

Auto-generated from TVBO experiment specification.
"""

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Target Distribution Functions
# ============================================================================

def cauchy_pdf(x: jnp.ndarray, x0: float, gamma: float = 1.0) -> jnp.ndarray:
    """Cauchy (Lorentzian) distribution for target spectra.

    Useful for defining target PSDs with characteristic peak frequencies.

    Parameters
    ----------
    x : jnp.ndarray
        Frequency values
    x0 : float
        Peak location (center frequency)
    gamma : float
        Half-width at half-maximum (default: 1.0)

    Returns
    -------
    jnp.ndarray
        Cauchy distribution values
    """
    return 1.0 / (np.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))


def gaussian_pdf(x: jnp.ndarray, mu: float, sigma: float = 1.0) -> jnp.ndarray:
    """Gaussian distribution for target spectra.

    Parameters
    ----------
    x : jnp.ndarray
        Values
    mu : float
        Mean (peak location)
    sigma : float
        Standard deviation (width)

    Returns
    -------
    jnp.ndarray
        Gaussian distribution values
    """
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * np.pi))


# ============================================================================
# Target Data Generation
# ============================================================================

% for i, target in enumerate(targets):
<%
target_name = target.label.replace(' ', '_').lower() if hasattr(target, 'label') and target.label else f'target_{i}'
target_symbol = target.symbol if hasattr(target, 'symbol') and target.symbol else 'target'
has_equation = hasattr(target, 'equation') and target.equation
%>
def generate_${target_name}(
    % if has_equation:
    ${', '.join([f'{p.name}: float = {p.value}' for p in target.parameters.values()])}
    % else:
    n_nodes: int = ${n_nodes},
    % endif
) -> jnp.ndarray:
    """Generate ${target.label if hasattr(target, 'label') else 'target'} data.

    % if hasattr(target, 'definition') and target.definition:
    ${target.definition}
    % endif

    Returns
    -------
    jnp.ndarray
        Target data array
    """
    % if has_equation:
    ${target_symbol} = ${render_expression(target.equation.rhs, format='jax')}
    return ${target_symbol}
    % else:
    # Default: return placeholder (replace with actual target data)
    return jnp.zeros((n_nodes,))
    % endif


% endfor
% if not targets:
def generate_target_peak_frequencies(
    lengths: jnp.ndarray,
    reference_idx: int = 0,
    f_min: float = 7.0,
    f_max: float = 11.0,
) -> jnp.ndarray:
    """Generate target peak frequencies from distance to reference region.

    Implements distance-based frequency gradient (e.g., MEG alpha gradient).

    Parameters
    ----------
    lengths : jnp.ndarray
        Tract length matrix [n_nodes, n_nodes]
    reference_idx : int
        Index of reference region (e.g., visual cortex)
    f_min : float
        Minimum frequency (at maximum distance)
    f_max : float
        Maximum frequency (at reference region)

    Returns
    -------
    jnp.ndarray
        Target peak frequencies for each node [n_nodes]
    """
    # Distance from reference region
    dist_from_ref = lengths[reference_idx, :]

    # Linear mapping from distance to frequency
    min_dist = dist_from_ref.min()
    max_dist = dist_from_ref.max()
    delta_f = (f_max - f_min) / (max_dist - min_dist + 1e-8)

    peak_freqs = f_max - delta_f * (dist_from_ref - min_dist)
    return peak_freqs


def generate_target_spectra(
    frequencies: jnp.ndarray,
    peak_freqs: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """Generate target PSDs from peak frequencies using Cauchy distribution.

    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequency axis [n_freqs]
    peak_freqs : jnp.ndarray
        Target peak frequency for each node [n_nodes]
    gamma : float
        Width parameter for Cauchy distribution

    Returns
    -------
    jnp.ndarray
        Target PSDs [n_nodes, n_freqs]
    """
    target_psds = jax.vmap(lambda fp: cauchy_pdf(frequencies, fp, gamma))(peak_freqs)
    return target_psds


def load_empirical_fc(
    path: str = None,
    name: str = "dk_average",
) -> jnp.ndarray:
    """Load empirical functional connectivity matrix.

    Parameters
    ----------
    path : str, optional
        Path to FC matrix file
    name : str
        Named dataset to load (if path not provided)

    Returns
    -------
    jnp.ndarray
        Empirical FC matrix [n_nodes, n_nodes]
    """
    if path is not None:
        import numpy as np
        return jnp.array(np.load(path))

    # Try tvboptim data loader
    try:
        from tvboptim.data import load_functional_connectivity
        return load_functional_connectivity(name=name)
    except ImportError:
        raise ValueError("Provide path or install tvboptim for named datasets")
% endif


# ============================================================================
# Observation Pipeline Functions
# ============================================================================

% for obs in observations:
<%
obs_name = obs.name.replace(' ', '_').lower() if hasattr(obs, 'name') and obs.name else 'observation'
pipeline = obs.pipeline if hasattr(obs, 'pipeline') and obs.pipeline else []
%>
% if pipeline:
def ${obs_name}_pipeline(result):
    """Observation pipeline: ${obs.description if hasattr(obs, 'description') else obs_name}

    Pipeline stages:
    % for stage in pipeline:
    - ${stage.name if hasattr(stage, 'name') else 'transform'}
    % endfor
    """
    x = result
    % for stage in pipeline:
    % if hasattr(stage, 'callable') and stage.callable:
    from ${stage.callable.module} import ${stage.callable.qualname.split('.')[-1]}
    x = ${stage.callable.qualname.split('.')[-1]}(x)
    % elif hasattr(stage, 'equation') and stage.equation:
    x = ${render_expression(stage.equation.rhs, format='jax')}
    % endif
    % endfor
    return x
% endif

% endfor
