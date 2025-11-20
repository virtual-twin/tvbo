# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

def g(dt, nt, n_svar, n_nodes, n_modes, seed=0, sigma_vec=None, sigma=0.0, state=None):
    """Standard Gaussian white noise using xi ~ N(0,1).

    Returns (nt, n_svar, n_nodes, n_modes): sqrt(dt) * sigma * xi.

    - sigma_vec: optional per-state sigma (length n_svar).
    - sigma: scalar fallback when sigma_vec is None.
    - state: optional current state placeholder for future correlative noise.
    """
    key = jax.random.PRNGKey(int(seed))
    xi = jax.random.normal(key, (nt, n_svar, n_nodes, n_modes))

    if sigma_vec is not None:
        sigma_b = jnp.asarray(sigma_vec)[None, ..., None, None]
    else:
        sigma_b = jnp.asarray(sigma)

    noise = jnp.sqrt(dt) * sigma_b * xi
    return noise

## single common g() function only (no g_step)
