"""Streaming stride reducer (``Observation.reduce: streaming`` over a pure decimation).

The simplest streamable pipeline keeps every ``k``-th sample of its source and drops the
rest (``SubSampling(period=TR)``). Materialised it costs a whole trajectory to produce a
``1/k`` slice of it; folded into the integrator carry it costs only the slice. Because the
reducer keeps exactly the samples the pipeline keeps, the streamed value is bit-identical
rather than merely close. These tests pin:

* the resolver recognises a stride pipeline whether the stride is declared as a ``period``
  (with the integration ``step_size``) or as an explicit ``step``, is opt-in, and leaves a
  non-stride pipeline to the BOLD path;
* the emitted reducer equals a from-scratch ``data[ds-1::ds]`` exactly, under ANY
  stride-aligned block decomposition (the ``prepare(reduce=...)`` path feeds blocks);
* ``streaming_post_eval_plan`` aligns its block size to the stride, so no write slot ever
  straddles a block boundary.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo.datamodel.schema import Argument, FunctionCall, Integrator, Observation
from tvbo.templates.tvboptim.utils import (
    reduction_dims,
    resolve_reduction,
    streaming_post_eval_plan,
)

_OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")


class _Exp:
    """Minimal experiment stub carrying the integration grid the resolver reads."""

    def __init__(self, step_size=0.0005):
        self.integration = Integrator(step_size=step_size)
        self.functions = {}
        self._source_file = None


def _stride_observation(period=0.72, step=None, reduce="streaming", name="SubSampling"):
    arguments = {}
    if period is not None:
        arguments["period"] = Argument(name="period", value=period)
    if step is not None:
        arguments["step"] = Argument(name="step", value=step)
    obs = Observation(
        name="bold_frames",
        source=["BOLD"],
        reduce=reduce,
        pipeline=[FunctionCall(callable={"name": name, "module": "tvbo.observations"},
                               arguments=arguments)],
    )
    return obs, _Exp()


# ── Resolver ────────────────────────────────────────────────────────────────────────────


def test_reduce_absent_keeps_the_post_scan_path():
    obs, exp = _stride_observation(reduce=None)
    assert resolve_reduction(obs, exp) is None


def test_period_and_step_size_give_the_stride():
    obs, exp = _stride_observation(period=0.72)
    red = resolve_reduction(obs, exp)
    assert red["kind"] == "stride"
    assert red["source"] == "BOLD"
    assert red["ds_steps"] == 1440          # 0.72 s / 0.0005 s


def test_explicit_step_is_honoured_without_a_period():
    obs, exp = _stride_observation(period=None, step=64)
    assert resolve_reduction(obs, exp)["ds_steps"] == 64


def test_predicate_call_without_experiment_is_side_effect_free_and_truthy():
    obs, _ = _stride_observation()
    red = resolve_reduction(obs)   # the "is this streaming?" predicate — no experiment
    assert red is not None and red["kind"] == "stride"


def test_non_stride_pipeline_falls_through_to_the_bold_path():
    # An unrecognised pipeline must NOT be silently treated as a stride: it falls through,
    # and the BOLD resolver's error names what a streaming pipeline needs.
    obs, exp = _stride_observation(name="compute_metastability")
    with pytest.raises(ValueError, match="HRF kernel generator"):
        resolve_reduction(obs, exp)


def test_post_eval_plan_aligns_the_block_to_the_stride():
    obs, exp = _stride_observation(period=0.72)
    exp.observations = {"bold_frames": obs}
    plan = streaming_post_eval_plan(exp)
    assert plan["names"] == ["bold_frames"]
    assert plan["period_in_steps"] == 1440


# ── Declared axis names ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind,dims", [
    ("convolution", ("time", "node")),   # HRF-Volterra BOLD: one kept sample per TR, per node
    ("stride", ("time", "node")),        # decimation: every k-th sample, per node
    ("comoment", ("node", "node_j")),    # cumulative co-moment FC: a node-by-node matrix
    ("recurrence", ("node",)),           # a folded statistic: one value per node
])
def test_each_reduction_kind_declares_its_axes(kind, dims):
    """Axis names come from the reduction's KIND, not from the array's shape.

    Shape cannot distinguish an ``(n_freq, n_node)`` spectrum from an ``(n_node, n_node)``
    matrix, so the reducer that fixes the shape is what names the axes. This pins the map
    every consumer downstream relies on.
    """
    assert reduction_dims({"kind": kind}) == dims


@pytest.mark.parametrize("red", [None, {}, {"kind": "something_new"}])
def test_an_undeclared_reduction_yields_no_axis_names(red):
    """No declaration means no label — never a guessed one.

    The container then falls back to positional ``<name>_d<i>``, which is an honest gap; a
    wrong label would be silently wrong everywhere it is selected by name.
    """
    assert reduction_dims(red) == ()


def test_the_plan_carries_the_declared_dims_for_each_streamed_observation():
    obs, exp = _stride_observation(period=0.72)
    exp.observations = {"bold_frames": obs}
    assert streaming_post_eval_plan(exp)["dims"] == {"bold_frames": ("time", "node")}


def test_a_plan_with_nothing_streaming_still_carries_a_dims_mapping():
    # The experiment template reads plan['dims'] unconditionally, so the key must always
    # exist — a missing key would be a codegen-time AttributeError, not a quiet default.
    _, exp = _stride_observation(reduce=None)
    exp.observations = {}
    assert streaming_post_eval_plan(exp)["dims"] == {}


# ── Emitted reducer: identity vs a from-scratch stride ──────────────────────────────────


def _emit_reducer(ds_steps):
    src = _OBS_TEMPLATE.get_def("render_stride_reduction").render(
        red={"kind": "stride", "source": "BOLD", "ds_steps": ds_steps},
        name="bold_frames", s_idx=0, dt=1.0,
    )
    ns = {"jnp": jnp, "jax": jax}
    exec(compile(src, "<reducer>", "exec"), ns)
    return ns["_reduction_bold_frames"]


@pytest.mark.parametrize("ds", [1, 4, 25])
def test_reducer_matches_a_from_scratch_stride(ds):
    rng = np.random.default_rng(0)
    n_steps, n_node = 300, 7
    data = jnp.asarray(rng.standard_normal((n_steps, 1, n_node)))

    init, update, finalize = _emit_reducer(ds)()
    acc = update(init(data[0], n_steps), data)
    got = np.asarray(finalize(acc))

    expected = np.asarray(data[:, 0, :])[ds - 1::ds]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("block", [25, 50, 100])
def test_reducer_is_block_decomposition_invariant(block):
    rng = np.random.default_rng(1)
    ds, n_steps, n_node = 25, 300, 4
    data = jnp.asarray(rng.standard_normal((n_steps, 1, n_node)))

    init, update, finalize = _emit_reducer(ds)()
    acc = init(data[0], n_steps)
    for start in range(0, n_steps, block):
        acc = update(acc, data[start:start + block])
    got = np.asarray(finalize(acc))

    assert np.array_equal(got, np.asarray(data[:, 0, :])[ds - 1::ds])


@pytest.mark.parametrize("n_steps", [305, 312, 324])
def test_reducer_handles_a_partial_tail_block(n_steps):
    """A run length that is not a multiple of the block leaves a short tail.

    tvboptim's ``_blocked_scan`` runs ``n_steps // block_size`` full blocks and then ONE
    tail call of the remainder, so the reducer must neither write past its buffer nor
    leave the last slot unwritten — a stale zero row would look like a real sample. The
    buffer is sized ``len(range(ds-1, n_steps, ds))``, which equals the number of full
    blocks when ``block_size == ds``; this pins that identity for tails that do and do not
    themselves contain a sample.
    """
    rng = np.random.default_rng(2)
    ds, n_node = 25, 3
    data = jnp.asarray(rng.standard_normal((n_steps, 1, n_node)))

    init, update, finalize = _emit_reducer(ds)()
    acc = init(data[0], n_steps)
    n_blocks = n_steps // ds
    for b in range(n_blocks):
        acc = update(acc, data[b * ds:(b + 1) * ds])
    if n_steps % ds:
        acc = update(acc, data[n_blocks * ds:])
    got = np.asarray(finalize(acc))

    expected = np.asarray(data[:, 0, :])[ds - 1::ds]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


# ── The sweep path's transient ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "skip, ds, n_steps, kept",
    [
        (0, 25, 300, 12),        # no transient: every strided sample
        (100, 25, 300, 8),       # skip on a stride boundary
        (124, 25, 300, 8),       # the sample AT skip survives
        (125, 25, 300, 7),       # skip past it, one fewer
        (200_000, 1440, 1_928_000, 1200),   # the Pang2023 resting sweep
    ],
)
def test_skip_drops_exactly_the_transient_samples(skip, ds, n_steps, kept):
    """`skip` is honoured, not silently ignored.

    A sweep folds transient and main run into ONE window and asks the reducer to drop the
    transient; the single-run path integrates them separately. When `skip` was accepted
    and discarded, a swept cell came back with `skip/ds` extra leading samples — 1,338
    BOLD frames where the same experiment run alone gives 1,200 — and nothing raised.
    """
    init, update, finalize = _emit_reducer(ds)(skip=skip)
    data = jnp.zeros((n_steps, 1, 2))
    acc = init(data[0], n_steps)
    # One update per block, block == stride, which is how the sweep drives it.
    for b in range(n_steps // ds):
        acc = update(acc, data[b * ds:(b + 1) * ds])
    got = np.asarray(finalize(acc))

    assert got.shape[0] == kept


def test_skip_keeps_the_post_transient_samples_themselves():
    """The samples that survive are the post-transient ones, in order."""
    rng = np.random.default_rng(3)
    ds, skip, n_steps = 25, 100, 300
    data = jnp.asarray(rng.standard_normal((n_steps, 1, 3)))

    init, update, finalize = _emit_reducer(ds)(skip=skip)
    acc = update(init(data[0], n_steps), data)
    got = np.asarray(finalize(acc))

    every = np.asarray(data[:, 0, :])[ds - 1::ds]
    positions = np.arange(ds - 1, n_steps, ds)
    assert np.array_equal(got, every[positions >= skip])
