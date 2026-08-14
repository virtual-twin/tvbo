"""Cumulative streaming mean/std/variance reducer (``aggregation`` + ``reduce: streaming``).

An observation declared ``aggregation: mean`` (or ``std`` / ``variance``) AND ``reduce: streaming`` — with no HRF/BOLD pipeline — is folded into the integrator carry as an ``(init, update, finalize)`` running-moment accumulator, so the source trajectory is never materialised. These tests pin:

* the resolver synthesizes the reducer from ``aggregation`` alone (one sum accumulator for
  ``mean``; a second sum-of-squares for ``std`` / ``variance``), tags it with no ``kind`` so it reuses the recurrence emitter, and stays truthy as the bare streaming predicate;
* the emitted reducer is byte-identical (to f64 rounding) to the host ``jnp.mean`` /
  ``jnp.std`` / ``jnp.var`` (ddof=0) of the materialised trajectory — the values the post-scan ``aggregation`` path computes — both as one block and across ANY block decomposition (the ``prepare(reduce=...)`` grid path feeds blocks, not one trajectory);
* the recurrence factory accepts-and-ignores ``warm_history`` / ``progress`` so the post-
  tuning eval shares ONE call site with the BOLD convolution reducer;
* the streaming post-eval plan folds the stat stream (defaulting the block to 1000 when no
  TR-aligned convolution reducer is present).

tvbo emits the reducer as code — no backend ships a cumulative mean/std reducer.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo.datamodel.schema import Observation
from tvbo.templates.tvboptim.utils import resolve_reduction, streaming_post_eval_plan

_OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")


class _Exp:
    """Minimal experiment stub — the stat-stream resolver reads nothing off it, but the post-eval plan walks ``observations``."""

    def __init__(self, observations=None):
        self.observations = observations or {}
        self._source_file = None


def _stat_observation(aggregation, source="x", reduce="streaming"):
    """A cumulative reduction over a recorded per-node source variable."""
    return Observation(name="obs", source=[source], aggregation=aggregation, reduce=reduce)


# ── Resolver ────────────────────────────────────────────────────────────────────────────


def test_reduce_absent_keeps_the_post_scan_path():
    obs = _stat_observation("mean", reduce=None)
    assert resolve_reduction(obs) is None


def test_mean_stream_synthesizes_one_accumulator():
    red = resolve_reduction(_stat_observation("mean"))
    assert "kind" not in red  # routes to the recurrence emitter
    assert red["source"] == "x"
    assert red["statistic"] == "mean"
    assert red["windowed"] is False
    assert red["skip_inclusive"] is True
    assert [s["name"] for s in red["states"]] == ["_s_sum"]
    assert red["states"][0]["is_accumulator"] is True
    assert {str(s) for s in red["states"][0]["update"].free_symbols} == {"_s_sum", "x"}
    assert {str(s) for s in red["output"].free_symbols} == {"_s_sum", "count"}


def test_std_stream_synthesizes_two_accumulators():
    red = resolve_reduction(_stat_observation("std"))
    assert [s["name"] for s in red["states"]] == ["_s_sum", "_s_sq"]
    assert all(s["is_accumulator"] for s in red["states"])
    # std reads sqrt(E[x^2] - E[x]^2): the finalize is a sqrt over the two accumulators.
    assert "sqrt" in str(red["output"]).lower()


def test_variance_stream_output_has_no_sqrt():
    red = resolve_reduction(_stat_observation("variance"))
    assert [s["name"] for s in red["states"]] == ["_s_sum", "_s_sq"]
    assert "sqrt" not in str(red["output"]).lower()


def test_bare_predicate_without_experiment_is_truthy():
    """The exploration path calls ``resolve_reduction(obs)`` with no experiment, as the "is this a streaming reducer?" predicate, so the full dict must resolve regardless."""
    red = resolve_reduction(_stat_observation("mean"))
    assert red is not None and "kind" not in red


# ── Emitted reducer: byte-identity vs the host jnp.mean / jnp.std / jnp.var ─────────────────


def _emit_reducer(red, name="obs"):
    """Render the reduction via the dispatcher and exec it (proves stat streams route to the recurrence branch)."""
    src = _OBS_TEMPLATE.get_def("render_reduction").render(red=red, name=name, s_idx=0, dt=1.0)
    ns = {"jnp": jnp, "jax": jax}
    exec(compile(src, "<reducer>", "exec"), ns)
    return ns[f"_reduction_{name}"]


def _trajectory(seed, T=512, n_states=4, n=6):
    """A recorded [T, n_states, n] trajectory; a nonzero offset exercises a real mean."""
    key = jax.random.PRNGKey(seed)
    return 1.0 + 0.3 * jax.random.normal(key, (T, n_states, n))


@pytest.mark.parametrize(
    "aggregation, reference",
    [
        ("mean", lambda c: jnp.mean(c, axis=0)),
        ("std", lambda c: jnp.std(c, axis=0)),  # ddof=0
        ("variance", lambda c: jnp.var(c, axis=0)),  # ddof=0
    ],
)
def test_reducer_matches_host_aggregation(aggregation, reference):
    red = resolve_reduction(_stat_observation(aggregation))
    factory = _emit_reducer(red)
    data = _trajectory(seed=1)
    col = data[:, 0, :]  # source column -> [T, n]

    init, update, finalize = factory(s_var=0, dt=1.0)
    acc = init(data[0], data.shape[0])
    got = finalize(update(acc, data))

    ref = reference(col)
    assert got.shape == ref.shape
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9, atol=1e-12)


def test_mean_includes_the_first_sample():
    """Regression: a cumulative mean must fold the sample AT skip=0, not the step after (the `skip_inclusive` gate). Dropping sample 0 would bias the mean and miscount."""
    red = resolve_reduction(_stat_observation("mean"))
    factory = _emit_reducer(red)
    data = _trajectory(seed=7, T=64)

    init, update, finalize = factory(s_var=0, dt=1.0)
    got = finalize(update(init(data[0], data.shape[0]), data))
    # Byte-identical count: mean over ALL T samples, not T-1.
    np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.mean(data[:, 0, :], axis=0)), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("aggregation", ["mean", "std", "variance"])
@pytest.mark.parametrize("block_size", [64, 128, 171, 256])
def test_reducer_is_block_decomposition_invariant(aggregation, block_size):
    """The grid path feeds blocks, not one trajectory. Sequential summation folds samples in the same order regardless of block boundaries, so the value is bit-exact across any decomposition."""
    red = resolve_reduction(_stat_observation(aggregation))
    factory = _emit_reducer(red)
    data = _trajectory(seed=2, T=512)

    init, update, finalize = factory(s_var=0, dt=1.0)
    single = finalize(update(init(data[0], data.shape[0]), data))

    acc = init(data[0], data.shape[0])
    for s in range(0, data.shape[0], block_size):
        acc = update(acc, data[s : s + block_size])
    multi = finalize(acc)
    assert float(jnp.max(jnp.abs(multi - single))) == 0.0


def test_factory_accepts_warm_history_and_progress_kwargs():
    """Edit 3: the recurrence factory accepts-and-ignores the BOLD-only warm_history / progress kwargs so the post-tuning eval has ONE reducer call site."""
    red = resolve_reduction(_stat_observation("mean"))
    factory = _emit_reducer(red)
    data = _trajectory(seed=3, T=128)

    init, update, finalize = factory(s_var=0, dt=1.0, warm_history=None, progress=True)
    got = finalize(update(init(data[0], data.shape[0]), data))
    np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.mean(data[:, 0, :], axis=0)), rtol=1e-9, atol=1e-12)


# ── Differentiability: the streamed statistic must be a usable fit target ───────────────────


@pytest.mark.parametrize(
    "aggregation, host",
    [
        ("mean", lambda c: jnp.mean(c, axis=0)),
        ("std", lambda c: jnp.std(c, axis=0)),  # ddof=0
    ],
)
def test_grad_flows_through_the_reducer_and_matches_host(aggregation, host):
    """A streamed mean/std observation must be differentiable so it can be a fit target.

    ``jax.grad`` of a loss over the folded reducer must be finite and byte-identical (to f64) to the gradient through the materialised host ``jnp.mean``/``jnp.std(ddof=0)``.
    Guards the autodiff hazards: the integer ``count``/``_gstep`` accumulators are constant w.r.t. the parameter, the ``jnp.where`` skip gate stays differentiable, and ``std``'s ``sqrt(var)`` has a finite gradient for var>0.
    """
    factory = _emit_reducer(resolve_reduction(_stat_observation(aggregation)))
    init, update, finalize = factory(s_var=0, dt=1.0)
    base = _trajectory(seed=5, T=200)

    def _traj(theta):
        return 1.0 + theta * base  # source column 0 depends differentiably on theta

    def _stream_loss(theta):
        data = _traj(theta)
        return jnp.sum(finalize(update(init(data[0], data.shape[0]), data)) ** 2)

    def _host_loss(theta):
        return jnp.sum(host(_traj(theta)[:, 0, :]) ** 2)

    theta = 1.3
    g_stream = jax.grad(_stream_loss)(theta)
    g_host = jax.grad(_host_loss)(theta)
    assert jnp.isfinite(g_stream)
    np.testing.assert_allclose(float(g_stream), float(g_host), rtol=1e-6, atol=1e-9)

    # The grid path folds blocks, not one trajectory: the gradient must survive that too.
    def _blocked_loss(theta, bs=64):
        data = _traj(theta)
        acc = init(data[0], data.shape[0])
        for s in range(0, data.shape[0], bs):
            acc = update(acc, data[s : s + bs])
        return jnp.sum(finalize(acc) ** 2)

    g_block = jax.grad(_blocked_loss)(theta)
    assert jnp.isfinite(g_block)
    np.testing.assert_allclose(float(g_block), float(g_host), rtol=1e-6, atol=1e-9)


# ── Streaming post-eval plan ───────────────────────────────────────────────────────────────


def test_stat_stream_is_folded_in_the_post_eval_plan():
    obs = _stat_observation("mean")
    study = _Exp(observations={"obs": obs})
    plan = streaming_post_eval_plan(study)
    assert plan["names"] == ["obs"]
    # No convolution reducer to TR-align to -> the block defaults to 1000 (not None).
    assert plan["period_in_steps"] == 1000


# ── Cumulative co-moment FC reducer (compute_fc pipeline + reduce: streaming) ────────────────

import tvboptim.observations.observation as _tvboptim_obs


def _fc_observation(skip_t=20, source="x_e_pre"):
    """A compute_fc (node-node correlation) pipeline opted into streaming."""
    return Observation(
        name="inp_corr",
        source=[source],
        reduce="streaming",
        pipeline=[
            {
                "callable": {"name": "compute_fc", "module": "tvboptim.observations.observation"},
                "arguments": [{"name": "timeseries", "value": source}, {"name": "skip_t", "value": skip_t}],
            }
        ],
    )


def _emit_fc_reducer(red, s_idx=4, name="inp_corr"):
    src = _OBS_TEMPLATE.get_def("render_reduction").render(red=red, name=name, s_idx=s_idx, dt=1.0)
    ns = {"jnp": jnp, "jax": jax}
    exec(compile(src, "<fc-reducer>", "exec"), ns)
    return ns[f"_reduction_{name}"]


def test_fc_stream_routes_to_the_comoment_reducer():
    red = resolve_reduction(_fc_observation(skip_t=20))
    assert red["kind"] == "comoment"  # NOT the BOLD convolution path
    assert red["source"] == "x_e_pre"
    assert red["skip_t"] == 20
    assert red["windowed"] is False
    assert red["states"] == ["count", "mean", "comoment"]


def test_fc_reducer_matches_compute_fc():
    """The folded FC must be byte-identical (to f64) to the materialised ``compute_fc(source, skip_t=20)`` — a zero-diagonal Pearson correlation over the post-skip window — held as an O(n^2) co-moment with no trajectory."""
    red = resolve_reduction(_fc_observation(skip_t=20))
    factory = _emit_fc_reducer(red, s_idx=4)
    data = _trajectory(seed=13, T=400, n_states=5, n=8)  # x_e_pre at column 4

    init, update, finalize = factory(s_var=4, dt=1.0, skip=0)  # factory adds skip_t=20
    got = finalize(update(init(data[0], data.shape[0]), data))
    ref = _tvboptim_obs.compute_fc(timeseries=data[:, 4, :], skip_t=20)
    assert got.shape == ref.shape == (8, 8)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("block_size", [37, 64, 128, 199])
def test_fc_reducer_is_block_decomposition_invariant(block_size):
    """The grid path folds blocks; the cumulative co-moment must be bit-exact across any block boundary (Welford add is order-fixed for a fixed sample order)."""
    red = resolve_reduction(_fc_observation(skip_t=20))
    factory = _emit_fc_reducer(red, s_idx=4)
    data = _trajectory(seed=17, T=400, n_states=5, n=8)

    init, update, finalize = factory(s_var=4, dt=1.0, skip=0)
    single = finalize(update(init(data[0], data.shape[0]), data))
    acc = init(data[0], data.shape[0])
    for s in range(0, data.shape[0], block_size):
        acc = update(acc, data[s : s + block_size])
    assert float(jnp.max(jnp.abs(finalize(acc) - single))) == 0.0


def test_fc_reducer_gradient_matches_compute_fc():
    """The FC reducer must be differentiable so a streamed FC can be a fit target. Grad of a loss over the folded FC is finite and matches the gradient through the materialised compute_fc to f64. A shared, theta-scaled drive injects real (tunable) correlation structure so the gradient is nonzero and meaningful (a uniform scale leaves correlation — hence the gradient — exactly zero)."""
    red = resolve_reduction(_fc_observation(skip_t=20))
    factory = _emit_fc_reducer(red, s_idx=4)
    init, update, finalize = factory(s_var=4, dt=1.0, skip=0)

    T, n_states, n = 300, 5, 6
    base = _trajectory(seed=21, T=T, n_states=n_states, n=n)
    common = jax.random.normal(jax.random.PRNGKey(22), (T, 1))

    def _traj(theta):
        return base.at[:, 4, :].add(theta * common)  # shared drive -> tunable FC

    def _stream_loss(theta):
        d = _traj(theta)
        return jnp.nansum(finalize(update(init(d[0], d.shape[0]), d)) ** 2)

    def _host_loss(theta):
        d = _traj(theta)
        return jnp.nansum(_tvboptim_obs.compute_fc(timeseries=d[:, 4, :], skip_t=20) ** 2)

    theta = 1.2
    g_stream = jax.grad(_stream_loss)(theta)
    g_host = jax.grad(_host_loss)(theta)
    assert jnp.isfinite(g_stream) and float(jnp.abs(g_stream)) > 0.0
    np.testing.assert_allclose(float(g_stream), float(g_host), rtol=1e-6, atol=1e-8)
