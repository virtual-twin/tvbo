# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Target Observation Template
=====================================

Generates target data generation functions for tvboptim optimization.

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Target distribution functions (Cauchy, Gaussian)
- Target data generation functions
</%doc>
<%
import numpy as np

# Get experiment info
model = experiment.local_dynamics
network = experiment.network
n_nodes = network.number_of_regions if network else 1
%>
"""Target observation definitions for tvboptim optimization."""

import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Target Distribution Functions
# =============================================================================

def cauchy_pdf(x: jnp.ndarray, x0: float, gamma: float = 1.0) -> jnp.ndarray:
    """Cauchy (Lorentzian) distribution for target spectra."""
    return 1.0 / (np.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))


def gaussian_pdf(x: jnp.ndarray, mu: float, sigma: float = 1.0) -> jnp.ndarray:
    """Gaussian distribution for target spectra."""
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * np.pi))


# =============================================================================
# Target Data Generation
# =============================================================================

def generate_target_peak_frequencies(
    lengths: jnp.ndarray,
    reference_idx: int = 0,
    f_min: float = 7.0,
    f_max: float = 11.0,
) -> jnp.ndarray:
    """Generate target peak frequencies from distance to reference region."""
    dist_from_ref = lengths[reference_idx, :]
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
    """Generate target PSDs from peak frequencies using Cauchy distribution."""
    target_psds = jax.vmap(lambda fp: cauchy_pdf(frequencies, fp, gamma))(peak_freqs)
    return target_psds


def load_empirical_fc(path: str = None, name: str = "dk_average") -> jnp.ndarray:
    """Load empirical functional connectivity matrix."""
    if path is not None:
        return jnp.array(np.load(path))
    try:
        from tvboptim.data import load_functional_connectivity
        return load_functional_connectivity(name=name)
    except ImportError:
        raise ValueError("Provide path or install tvboptim for named datasets")
