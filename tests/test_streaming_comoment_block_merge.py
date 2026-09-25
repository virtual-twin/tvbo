"""The streamed cumulative-FC reducer folds a block at once and still equals the correlation of every accepted sample.

The reducer rendered from ``render_comoment_reduction`` merges each integration block's mean and centred co-moment into its running state (pairwise Welford) instead of folding sample by sample. These tests feed it the blocks a post-tuning evaluation hands it, with the leading ``skip`` samples ending mid-block, and compare its Pearson matrix against ``np.corrcoef`` over the kept samples and against the per-sample Welford fold it replaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo.codegen.reducers import resolve_streaming_reducer
from tvbo.codegen.streaming_reducers import lookup_streaming_reducer

OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")


def _reducer(skip_t):
    spec = lookup_streaming_reducer("tvboptim", "tvboptim.observations.observation", "compute_fc")
    if spec is None:
        pytest.skip("windowed_fc streaming reducer disabled (TVBO_STREAMING_REDUCERS=0)")
    lowered = resolve_streaming_reducer(spec, "jax")
    red = {"states": list(spec.state), "add": lowered["add"], "emit": lowered["emit"], "skip_t": skip_t}
    source = OBS_TEMPLATE.get_def("render_comoment_reduction").render(red=red, name="corr", s_idx=1, dt=1.0)
    namespace = {"jnp": jnp, "jax": jax}
    exec(compile(source, "<comoment_reduction>", "exec"), namespace)
    return namespace["_reduction_corr"], spec, lowered


def _per_sample_fold(lowered, spec, samples):
    """The sample-by-sample Welford fold of the recipe's ``add``, the form the block merge replaces."""
    state = {"count": jnp.array(0), "mean": jnp.zeros(samples.shape[1]), "comoment": jnp.zeros((samples.shape[1],) * 2)}
    for v in samples:
        namespace = {"jnp": jnp, **state, "v": v}
        for lhs, rhs in lowered["add"]:
            namespace[lhs] = eval(rhs, namespace)
        state = {k: namespace[k] for k in spec.state}
    return eval(lowered["emit"], {"jnp": jnp, **state})


@pytest.mark.parametrize("block_len", [7, 50, 720])
@pytest.mark.parametrize("skip", [0, 13, 60])
def test_block_merge_equals_the_correlation_of_the_kept_samples(block_len, skip):
    make, spec, lowered = _reducer(skip_t=5)
    rng = np.random.default_rng(block_len + skip)
    n_nodes = 9
    n_blocks = max(4, -(-(skip + 5 + 30) // block_len))
    mixing = rng.normal(size=(n_nodes, n_nodes)) * 0.3
    signal = 2.0 + rng.normal(size=(n_blocks * block_len, n_nodes)) @ mixing
    trajectory = jnp.asarray(np.stack([np.zeros_like(signal), signal, np.ones_like(signal)], axis=1))

    init, update, finalize = make(skip=skip)
    acc = init(trajectory[0], trajectory.shape[0])
    for k in range(n_blocks):
        acc = jax.jit(update)(acc, trajectory[k * block_len : (k + 1) * block_len])
    streamed = np.asarray(finalize(acc))

    kept = signal[skip + 5 :]
    expected = np.corrcoef(kept.T)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(streamed, expected, atol=1e-12)
    np.testing.assert_allclose(streamed, _per_sample_fold(lowered, spec, jnp.asarray(kept)), atol=1e-12)
    assert int(acc[0]) == kept.shape[0]
