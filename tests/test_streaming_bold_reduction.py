"""Streaming HRF-Volterra BOLD reducer (``Observation.reduce: streaming``).

An HRF-Volterra ``bold`` pipeline marked ``reduce: streaming`` is lifted to an ``(init, update, finalize)`` block reducer that folds the neural trajectory into a downsampled-history ring, evaluates the HRF ``'valid'`` convolution ONLY at the TR boundaries via ``strided_convolve`` (no FFT buffer), and writes the Volterra-scaled BOLD samples into a preallocated buffer — so the full trajectory is never held. These tests pin:

* the resolver lifts the kernel / decimation stride / TR stride / Volterra scaling from the
  declared pipeline, is opt-in (absent ``reduce`` keeps the post-scan path), and requires a pure ``subsample`` decimation — a ``temporal_average`` window is rejected (even ``period_samples=1`` shifts by one sample, reproducing tvboptim TemporalAverage);
* the emitted reducer is byte-identical (to f64 rounding, ``strided_convolve`` ~1e-12 vs the
  FFT) to a from-scratch SubSampling BOLD — the decimation ``streaming_hrf_bold`` requires — both cold (zero ring) and warm-started, and identical across ANY period-aligned block decomposition (the ``prepare(reduce=...)`` grid path feeds blocks, not one trajectory).

tvbo emits the reducer as code — it never calls tvboptim's ``streaming_hrf_bold`` primitive.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tvbo.datamodel.schema import (
    Argument,
    Equation,
    Function,
    FunctionCall,
    Observation,
    Parameter,
    Range,
)
from tvbo.templates.tvboptim.utils import resolve_reduction, streaming_post_eval_plan

from .reducer_harness import OBS_TEMPLATE, reducer_namespace


class _Exp:
    """Minimal experiment stub exposing the ``functions`` the resolver reads for defaults."""

    def __init__(self, functions):
        self.functions = functions
        self._source_file = None


def _bold_observation(ds_step=1, tr_stride=180, k_1=5.6, V_0=0.02, reduce="streaming", decimation="subsample"):
    """An HRF-Volterra ``bold`` pipeline (kernel -> decimation -> volterra -> subsample-at-TR).

    ``decimation`` selects the pipeline's downsampling step: ``subsample`` (backend-independent pure stride, the only stream-safe form) or ``temporal_average`` (averaging window; rejected).
    """
    functions = {
        "hrf_kernel": Function(name="hrf_kernel", time_range=Range(lo=0, hi=20000.0, n=5000)),
        "subsample": Function(
            name="subsample",
            equation=Equation(rhs="subsample(data, step - 1, step)"),
            arguments={"step": Argument(name="step", value=ds_step)},
        ),
        "temporal_average": Function(
            name="temporal_average",
            arguments={"period_samples": Argument(name="period_samples", value=ds_step)},
        ),
        "subsample_bold": Function(name="subsample_bold", arguments={"s": Argument(name="s", value=tr_stride)}),
        "volterra_transform": Function(
            name="volterra_transform",
            equation=Equation(
                rhs="k_1 * V_0 * (data - 1.0)",
                parameters={
                    "k_1": Parameter(name="k_1", value=k_1),
                    "V_0": Parameter(name="V_0", value=V_0),
                },
            ),
        ),
    }
    obs = Observation(
        name="bold",
        source=["S_e"],
        period=720.0,
        reduce=reduce,
        pipeline=[
            # Steps reference their Function by name (as a loaded YAML pipeline does); the definitions (time_range, argument/equation defaults) live in experiment.functions.
            FunctionCall(function="hrf_kernel"),
            FunctionCall(function=decimation, output="downsampled_data"),
            FunctionCall(function="volterra_transform"),
            FunctionCall(function="subsample_bold"),
        ],
    )
    return obs, _Exp(functions)


# ── Resolver ────────────────────────────────────────────────────────────────────────────


def test_reduce_absent_keeps_the_post_scan_path():
    obs, exp = _bold_observation(reduce=None)
    assert resolve_reduction(obs, exp) is None


def test_streaming_lifts_the_pipeline_constants():
    obs, exp = _bold_observation(k_1=5.6, V_0=0.02, tr_stride=180)
    red = resolve_reduction(obs, exp)
    assert red["kind"] == "convolution"
    assert red["source"] == "S_e"
    assert red["kernel_call"] == "hrf_kernel()"
    assert red["ds_steps"] == 1
    assert red["tr_stride"] == 180
    assert red["k_1"] == 5.6
    assert red["V_0"] == 0.02


def test_predicate_call_without_experiment_is_side_effect_free_and_truthy():
    obs, _ = _bold_observation()
    red = resolve_reduction(obs)  # the "is this streaming?" predicate — no experiment
    assert red is not None and red["kind"] == "convolution"


def test_temporal_average_decimation_is_rejected():
    # temporal_average is not stream-safe even at period_samples=1 (it shifts by one sample, reproducing tvboptim TemporalAverage); streaming requires a pure subsample decimation.
    obs, exp = _bold_observation(decimation="temporal_average")
    with pytest.raises(ValueError, match="subsample decimation"):
        resolve_reduction(obs, exp)


def test_post_eval_plan_names_and_deliverables():
    obs, exp = _bold_observation()
    # A tiny experiment-like object carrying just the observations the plan walks.
    fc = Observation(
        name="fc",
        source=["bold"],
        pipeline=[FunctionCall(name="fc", function="compute_fc")],
    )

    class _StudyExp(_Exp):
        pass

    study = _StudyExp(exp.functions)
    study.observations = {"bold": obs, "fc": fc}
    plan = streaming_post_eval_plan(study)
    assert plan["names"] == ["bold"]
    assert "fc" in plan["deliverables"]
    assert plan["period_in_steps"] == 180  # ds_steps * tr_stride


# ── Emitted reducer: byte-identity vs a from-scratch SubSampling BOLD ──────────────────────


def _bold_red(ds, tr, K):
    """The resolved constants `render_convolution_reduction` emits from, as `resolve_reduction` hands them over.

    One shape rather than seven copies: the cases differ only in the decimation stride, the TR and the kernel's length, and a constant that appears in every copy is a constant no case is actually varying.
    """
    return {
        "kind": "convolution",
        "source": "S_e",
        "kernel_call": "hrf_kernel()",
        "ds_steps": ds,
        "tr_stride": tr,
        "k_1": 5.6,
        "V_0": 0.02,
        "warmup_steps": K * ds,  # the kernel's span in raw steps, as resolve_reduction reads it off `time_range`
    }


def _emit_reducer(red, kernel):
    """Render the convolution reducer def and exec it with a stub ``hrf_kernel``."""
    src = OBS_TEMPLATE.get_def("render_convolution_reduction").render(red=red, name="bold", s_idx=0, dt=1.0)
    ns = reducer_namespace(hrf_kernel=lambda: kernel)
    exec(compile(src, "<reducer>", "exec"), ns)
    return ns["_reduction_bold"]


def _reference_bold(data_col, kernel, ds, tr, k_1, V_0, skip=0):
    """From-scratch SubSampling BOLD, the decimation ``streaming_hrf_bold`` requires.

    Cold ring; 'valid' HRF convolution of concat(ring, downsampled data); Volterra scaling; sample at every TR boundary; drop the whole BOLD samples that fall inside ``skip``. The settle is part of ``data_col``, so the kernel's warm-up is real signal rather than a separately-supplied history.
    """
    K, n = kernel.shape[0], data_col.shape[-1]
    ds_data = data_col[ds - 1 :: ds]
    sig = jnp.concatenate([jnp.zeros((K, n)), ds_data], axis=0)
    conv = jax.vmap(lambda x: jsp.signal.fftconvolve(x, kernel, "valid"), in_axes=1, out_axes=1)(sig)
    return (V_0 * k_1 * (conv - 1.0))[tr::tr][skip // (ds * tr) :]


@pytest.mark.parametrize("skip", [0, 36, 180])  # 0 -> no settle; both are whole BOLD periods
def test_reducer_matches_subsampling_reference(skip):
    tr, K, T, n = 18, 50, 360, 6
    red = _bold_red(1, tr, K)
    kernel = jax.random.normal(jax.random.PRNGKey(0), (K,))
    factory = _emit_reducer(red, kernel)

    data = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (T, 4, n))  # [T, n_states, n]
    data_col = data[:, 0, :]

    init, update, finalize = factory(s_var=0, dt=1.0, skip=skip)
    acc = init(data[0], T)
    acc = update(acc, data)
    got = finalize(acc)

    ref = _reference_bold(data_col, kernel, ds=1, tr=tr, k_1=5.6, V_0=0.02, skip=skip)
    assert got.shape == ref.shape
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-11  # strided_convolve ~1e-12 vs FFT


def test_skip_drops_only_the_settles_samples():
    """`skip` reports the same samples an unskipped fold does, minus the settle's.

    This is the property the single-scan design rests on: the settle is folded (so the HRF ring warms on real signal) and only the settle's own output samples are withheld, so a run with a settle and the same run without one agree sample for sample past t=0.
    """
    tr, K, T, n, skip = 18, 50, 360, 6, 90
    red = _bold_red(1, tr, K)
    kernel = jax.random.normal(jax.random.PRNGKey(7), (K,))
    factory = _emit_reducer(red, kernel)
    data = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(8), (T, 4, n))

    def fold(**kw):
        init, update, finalize = factory(s_var=0, dt=1.0, **kw)
        return finalize(update(init(data[0], T), data))

    whole, cut = fold(), fold(skip=skip)
    assert cut.shape[0] == whole.shape[0] - skip // tr
    assert float(jnp.max(jnp.abs(cut - whole[skip // tr :]))) == 0.0


@pytest.mark.parametrize("block_size", [18, 36, 90, 180])  # all multiples of period_in_steps (18)
def test_reducer_is_block_decomposition_invariant(block_size):
    tr, K, T, n = 18, 50, 360, 6
    red = _bold_red(1, tr, K)
    kernel = jax.random.normal(jax.random.PRNGKey(3), (K,))
    factory = _emit_reducer(red, kernel)
    data = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(4), (T, 4, n))

    init, update, finalize = factory(s_var=0, dt=1.0)
    acc = init(data[0], T)
    single = finalize(update(acc, data))

    acc = init(data[0], T)
    for s in range(0, T, block_size):
        acc = update(acc, data[s : s + block_size])
    multi = finalize(acc)
    # The block scaffolding is exact: no rounding accumulates across block boundaries.
    assert float(jnp.max(jnp.abs(multi - single))) == 0.0


# ── Two scans: the settle warms the ring the window no longer contains ────────────


@pytest.mark.parametrize("ds", [1, 3, 5])
def test_a_handed_settle_reproduces_the_in_band_fold(ds):
    """Folding the settle in-band and handing it over as warm-up are the same answer, exactly.

    Under one scan the reducer saw the settle and its ring warmed on real signal. Under two the window it folds opens at t=0 and that history is somewhere else, so the ring has to be given it. What must hold is not "close": the decimated tail IS the signal the one-scan fold convolved against, so reconstructing the ring from it reproduces that fold bit for bit — anything less means the rows are misaligned, and a misalignment of one decimated sample is invisible in a tolerance and wrong in every reported frame.
    """
    tr, K, n = 6, 40, 5
    settle_steps, measured_steps = 300, 360
    red = _bold_red(ds, tr, K)
    kernel = jax.random.normal(jax.random.PRNGKey(0), (K,))
    factory = _emit_reducer(red, kernel)
    whole = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (settle_steps + measured_steps, 4, n))
    settle, measured = whole[:settle_steps], whole[settle_steps:]

    init, update, finalize = factory(s_var=0, dt=1.0, skip=settle_steps)
    in_band = finalize(update(init(whole[0], whole.shape[0]), whole))

    init, update, finalize = factory(s_var=0, dt=1.0, skip=0, settle=settle)
    two_scan = finalize(update(init(measured[0], measured_steps), measured))

    assert two_scan.shape == in_band.shape
    assert float(jnp.max(jnp.abs(two_scan - in_band))) == 0.0


def test_without_the_settle_the_first_samples_are_wrong_by_the_kernel():
    """The guard earns its place: cold is not almost-right, and it is wrong exactly where the kernel reaches.

    A run that omits the warm-up returns the same shape, finite throughout and correctly stamped — which is why this was silent. The error is confined to the kernel's support and is large inside it, so a test that only checked shape or finiteness would pass on the broken answer.
    """
    tr, K, ds, n = 6, 40, 3, 5
    settle_steps, measured_steps = 300, 360
    red = _bold_red(ds, tr, K)
    kernel = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (K,)))
    factory = _emit_reducer(red, kernel)
    whole = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (settle_steps + measured_steps, 4, n))
    settle, measured = whole[:settle_steps], whole[settle_steps:]

    init, update, finalize = factory(s_var=0, dt=1.0, skip=0, settle=settle)
    warm = np.asarray(finalize(update(init(measured[0], measured_steps), measured)))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    cold = np.asarray(finalize(update(init(measured[0], measured_steps), measured)))

    support = -(-K // tr)  # BOLD samples the kernel's support spans
    assert np.abs(cold[:support] - warm[:support]).max() > 0.1 * np.abs(warm[:support]).max()
    np.testing.assert_allclose(cold[support:], warm[support:], atol=1e-12)


def test_a_settle_shorter_than_the_kernel_warms_what_it_covers():
    """A settle too short to fill the ring is not refused, and it is not treated as absent either.

    The rows it does cover are real signal; only the span no settle reached stays zero. That is the honest answer — better than cold everywhere, and it degrades continuously as the settle grows rather than switching on at some threshold.
    """
    tr, K, ds, n = 6, 40, 3, 5
    measured_steps = 360
    red = _bold_red(ds, tr, K)
    kernel = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (K,)))
    factory = _emit_reducer(red, kernel)
    full_settle = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), ((K - 1) * ds + 1, 4, n))
    measured = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (measured_steps, 4, n))

    def _fold(**kw):
        init, update, finalize = factory(s_var=0, dt=1.0, skip=0, **kw)
        return np.asarray(finalize(update(init(measured[0], measured_steps), measured)))

    full, short, cold = _fold(settle=full_settle), _fold(settle=full_settle[-30:]), _fold()
    assert np.abs(short - full).max() < np.abs(cold - full).max()
    assert np.abs(short - cold).max() > 0.0


def test_the_declared_warmup_is_what_a_caller_must_keep():
    """The reducer states its own warm-up support, so a caller trims the settle by a number it is given rather than one it derives.

    What the caller relies on is sufficiency: keep this many steps and the answer is the one an untrimmed settle gives. It is deliberately not the tightest such number — the leading `tr_stride` ring rows only ever feed convolution outputs that fall before the first reported sample — because a caller trimming one settle for several reducers wants a bound it can take the max of, not one that is exact for each. Non-vacuity is asserted separately: a materially shorter settle does change the answer, so the bound is about the kernel and not merely large.
    """
    tr, K, ds, n = 6, 40, 3, 5
    measured_steps = 360
    red = _bold_red(ds, tr, K)
    kernel = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (K,)))
    src = OBS_TEMPLATE.get_def("render_convolution_reduction").render(red=red, name="bold", s_idx=0, dt=1.0)
    ns = reducer_namespace(hrf_kernel=lambda: kernel)
    exec(compile(src, "<reducer>", "exec"), ns)
    warmup = int(ns["_warmup_bold"])
    assert warmup == red["warmup_steps"], "the emitted literal is the resolved support, not a shape read back off the kernel"
    assert warmup >= (K - 1) * ds + 1, "and it covers every ring row the convolution can reach"

    factory = ns["_reduction_bold"]
    settle = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (2000, 4, n))
    measured = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (measured_steps, 4, n))

    def _fold(s):
        init, update, finalize = factory(s_var=0, dt=1.0, skip=0, settle=s)
        return np.asarray(finalize(update(init(measured[0], measured_steps), measured)))

    whole = _fold(settle)
    np.testing.assert_array_equal(_fold(settle[-warmup:]), whole)
    assert not np.array_equal(_fold(settle[-(warmup // 2) :]), whole)


def test_supplying_the_settle_twice_is_refused():
    """`skip` and `settle` are the two ways of saying where the settle is, and they are exclusive.

    Given both, the same signal warms the ring and is folded again, so every reported sample carries the settle twice — the same shape, finite, correctly stamped, and wrong. Refused rather than silently preferred one way or the other, because which one a caller meant is not recoverable.
    """
    tr, K, ds, n = 6, 40, 3, 5
    kernel = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (K,)))
    factory = _emit_reducer(_bold_red(ds, tr, K), kernel)
    settle = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (300, 4, n))
    init, _, _ = factory(s_var=0, dt=1.0, skip=300, settle=settle)
    with pytest.raises(ValueError, match="Pass one"):
        init(settle[0], 360)


def test_the_emission_grid_names_the_step_each_sample_actually_covers():
    """`emission_times` is what a caller stamps a streamed value with, so it has to name the step that moves it.

    The streaming path returns bare arrays — there is no `ts` on them the way there is on a materialised `ObservationResult` — so the grid comes from the reduction. Checked by perturbation rather than against the emitting expression, which would only restate it: a sample covers the steps up to its own timestamp and none after, so the step AT `t[m]` must move sample m and the very next step must not. A phase error of one whole period — stamping the first sample at the window's opening step rather than at the end of the period it integrates — is invisible to every other assertion here.
    """
    from tvbo.templates.tvboptim.utils import emission_times

    tr, K, ds, n, dt = 6, 40, 3, 4, 1.0
    red = _bold_red(ds, tr, K)
    period = ds * tr
    n_steps = period * 5
    kernel = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (K,)))
    factory = _emit_reducer(red, kernel)
    base = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_steps, 4, n))

    def _fold(data):
        init, update, finalize = factory(s_var=0, dt=dt, skip=0)
        return np.asarray(finalize(update(init(data[0], n_steps), data)))

    reference = _fold(base)
    times = emission_times(red, reference.shape[0], dt)
    assert times is not None and len(times) == reference.shape[0]

    def _moved(step):
        got = _fold(base.at[step, 0, :].add(10.0))
        return {m for m in range(reference.shape[0]) if np.abs(got[m] - reference[m]).max() > 1e-9}

    for m in (0, 1, reference.shape[0] - 1):
        at = int(round(times[m] / dt)) - 1  # the timestamp names the last step of the period, 1-based
        assert m in _moved(at), f"sample {m} is stamped {times[m]:g} but the step at that time does not move it"
        if at + 1 < n_steps:
            assert m not in _moved(at + 1), f"sample {m} is stamped {times[m]:g} but a later step still moves it"


def test_a_reduction_that_folds_time_away_has_no_grid():
    """`emission_times` returns None rather than a plausible-looking axis for a reduction whose output has no time.

    A co-moment FC is a node-by-node matrix and a recurrence is one value per node; handing either a time axis would be inventing one, and the caller needs to be able to tell the difference rather than stamp whatever it is given.
    """
    from tvbo.templates.tvboptim.utils import emission_times

    for kind in ("comoment", "recurrence", "wave"):
        assert emission_times({"kind": kind}, 4, 1.0) is None, kind


@pytest.mark.parametrize(
    "red, period, has_time",
    [
        ({"kind": "convolution", "ds_steps": 40, "tr_stride": 180}, 7200, True),
        ({"kind": "stride", "ds_steps": 25}, 25, True),
        ({"kind": "monitor", "period_steps": 250}, 250, True),
        ({"kind": "wave", "period_steps": 5}, 5, False),
        ({"kind": "recurrence"}, None, False),
        ({"kind": "comoment"}, None, False),
    ],
)
def test_having_a_period_and_reporting_one_are_different_questions(red, period, has_time):
    """A reduction's block alignment and its time axis are asked separately, because one kind answers them differently.

    A `wave` detector decimates on a period and then folds those samples into per-group scalars: it needs a block its slot boundaries align to, and it has no time axis. Deriving the first from the second dropped its block silently back to the 1000-step default — silently, because nothing fails, the run just materialises the trajectory the streaming path exists to avoid, and Koller exp_41 came back at 286 GiB.
    """
    from tvbo.templates.tvboptim.utils import emission_period_steps, emission_times

    assert emission_period_steps(red) == period
    assert (emission_times(red, 3, 1.0) is not None) is has_time
