"""The in-loop sliding-window FC of the FIC/EIB tuning loop is an O(n^2)-per-step running co-moment, and it equals the FC recomputed over the whole window.

The generated tuning step folds each new BOLD sample in with ``add``, drops the sample leaving the window with ``evict``, and reads the Pearson matrix with ``emit``; ``resync`` rebuilds the accumulator from the window every ``resync_every`` steps and ``resync_masked`` does the same from a fixed-size ring with a traced window length. These tests drive the lowered ``windowed_fc`` recipe exactly as the template emits it and compare every step against a from-scratch correlation of the current window.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tvbo.codegen.reducers import resolve_streaming_reducer
from tvbo.codegen.streaming_reducers import lookup_streaming_reducer


def _lowered():
    spec = lookup_streaming_reducer("tvboptim", "tvboptim.observations.observation", "compute_fc")
    if spec is None:
        pytest.skip("windowed_fc streaming reducer disabled (TVBO_STREAMING_REDUCERS=0)")
    return spec, resolve_streaming_reducer(spec, "jax")


def _block(assignments, state, **inputs):
    """Run one lowered assignment block the way the template emits it: sequential ``lhs = rhs`` in one namespace."""
    namespace = {"jnp": jnp, "jax": jax, **state, **inputs}
    for lhs, rhs in assignments:
        namespace[lhs] = eval(rhs, namespace)
    return namespace


def _window_fc(window):
    fc = np.corrcoef(np.asarray(window).T)
    np.fill_diagonal(fc, 0.0)
    return fc


def test_sliding_window_fc_matches_recomputation_every_step():
    spec, lowered = _lowered()
    rng = np.random.default_rng(0)
    n_nodes, window, n_steps = 12, 150, 450
    series = jnp.asarray(0.3 + rng.normal(size=(window + n_steps, n_nodes)) @ rng.normal(size=(n_nodes, n_nodes)) * 0.05)

    ns = _block(lowered["resync"], {}, x=series[:window])
    state = {name: ns[name] for name in spec.state}
    for step in range(n_steps):
        state = {k: _block(lowered["evict"], state, v=series[step])[k] for k in spec.state}
        state = {k: _block(lowered["add"], state, v=series[window + step])[k] for k in spec.state}
        emitted = eval(lowered["emit"], {"jnp": jnp, **state})
        np.testing.assert_allclose(emitted, _window_fc(series[step + 1 : window + step + 1]), atol=1e-10)
    assert int(state["count"]) == window


def test_masked_ring_resync_equals_window_resync():
    spec, lowered = _lowered()
    rng = np.random.default_rng(1)
    ring = jnp.asarray(rng.normal(size=(400, 9)))
    for window in (20, 150, 400):
        mask = (jnp.arange(ring.shape[0]) >= ring.shape[0] - window).astype(ring.dtype).reshape(-1, 1)
        masked = _block(lowered["resync_masked"], {}, x=ring, m=mask, L=window)
        plain = _block(lowered["resync"], {}, x=ring[-window:])
        np.testing.assert_allclose(np.ravel(masked["mean"]), np.ravel(plain["mean"]), atol=1e-14)
        np.testing.assert_allclose(masked["comoment"], plain["comoment"], atol=1e-11)
        emitted = eval(lowered["emit"], {"jnp": jnp, "comoment": masked["comoment"]})
        np.testing.assert_allclose(emitted, _window_fc(ring[-window:]), atol=1e-12)
