"""Streaming HRF-Volterra BOLD reducer (``Observation.reduce: streaming``).

An HRF-Volterra ``bold`` pipeline marked ``reduce: streaming`` is lifted to an
``(init, update, finalize)`` block reducer that folds the neural trajectory into a
downsampled-history ring, evaluates the HRF ``'valid'`` convolution ONLY at the TR
boundaries via ``strided_convolve`` (no FFT buffer), and writes the Volterra-scaled BOLD
samples into a preallocated buffer — so the full trajectory is never held. These tests pin:

* the resolver lifts the kernel / decimation stride / TR stride / Volterra scaling from the
  declared pipeline, is opt-in (absent ``reduce`` keeps the post-scan path), and requires a
  pure ``subsample`` decimation — a ``temporal_average`` window is rejected (even
  ``period_samples=1`` shifts by one sample, reproducing tvboptim TemporalAverage);
* the emitted reducer is byte-identical (to f64 rounding, ``strided_convolve`` ~1e-12 vs the
  FFT) to a from-scratch SubSampling BOLD — the decimation ``streaming_hrf_bold`` requires —
  both cold (zero ring) and warm-started, and identical across ANY period-aligned block
  decomposition (the ``prepare(reduce=...)`` grid path feeds blocks, not one trajectory).

tvbo emits the reducer as code — it never calls tvboptim's ``streaming_hrf_bold`` primitive.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest
from mako.template import Template

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

_OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")


class _Exp:
    """Minimal experiment stub exposing the ``functions`` the resolver reads for defaults."""

    def __init__(self, functions):
        self.functions = functions
        self._source_file = None


def _bold_observation(ds_step=1, tr_stride=180, k_1=5.6, V_0=0.02, reduce="streaming",
                      decimation="subsample"):
    """An HRF-Volterra ``bold`` pipeline (kernel -> decimation -> volterra -> subsample-at-TR).

    ``decimation`` selects the pipeline's downsampling step: ``subsample`` (backend-independent
    pure stride, the only stream-safe form) or ``temporal_average`` (averaging window; rejected).
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
        "subsample_bold": Function(
            name="subsample_bold", arguments={"s": Argument(name="s", value=tr_stride)}
        ),
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
            # Steps reference their Function by name (as a loaded YAML pipeline does); the
            # definitions (time_range, argument/equation defaults) live in experiment.functions.
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
    # temporal_average is not stream-safe even at period_samples=1 (it shifts by one sample,
    # reproducing tvboptim TemporalAverage); streaming requires a pure subsample decimation.
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


def _emit_reducer(red, kernel):
    """Render the convolution reducer def and exec it with a stub ``hrf_kernel``."""
    src = _OBS_TEMPLATE.get_def("render_convolution_reduction").render(
        red=red, name="bold", s_idx=0, dt=1.0
    )
    ns = {"jnp": jnp, "jax": jax, "hrf_kernel": lambda: kernel}
    exec(compile(src, "<reducer>", "exec"), ns)
    return ns["_reduction_bold"]


def _reference_bold(data_col, warm, kernel, ds, tr, k_1, V_0):
    """From-scratch SubSampling BOLD (the decimation ``streaming_hrf_bold`` requires):
    ring = downsampled transient fitted to one kernel length; 'valid' HRF convolution of
    concat(ring, downsampled data); Volterra scaling; sample at every TR boundary."""
    K, n = kernel.shape[0], data_col.shape[-1]
    ds_data = data_col[ds - 1 :: ds]
    if warm is None:
        ring = jnp.zeros((K, n))
    else:
        wd = warm[ds - 1 :: ds]
        ring = (
            jnp.vstack([jnp.zeros((K - wd.shape[0], n), wd.dtype), wd]) if wd.shape[0] < K
            else wd[-K:]
        )
    sig = jnp.concatenate([ring, ds_data], axis=0)
    conv = jax.vmap(lambda x: jsp.signal.fftconvolve(x, kernel, "valid"), in_axes=1, out_axes=1)(sig)
    return (V_0 * k_1 * (conv - 1.0))[tr::tr]


@pytest.mark.parametrize("warm_len", [0, 30, 80])  # 0 -> cold; <K and >K exercise the pad/trim
def test_reducer_matches_subsampling_reference(warm_len):
    tr, K, T, n = 18, 50, 360, 6
    red = {"kind": "convolution", "source": "S_e", "kernel_call": "hrf_kernel()",
           "ds_steps": 1, "tr_stride": tr, "k_1": 5.6, "V_0": 0.02}
    kernel = jax.random.normal(jax.random.PRNGKey(0), (K,))
    factory = _emit_reducer(red, kernel)

    data = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (T, 4, n))  # [T, n_states, n]
    data_col = data[:, 0, :]
    warm = None if warm_len == 0 else 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (warm_len, n))

    init, update, finalize = factory(s_var=0, dt=1.0, warm_history=warm)
    acc = init(data[0], T)
    acc = update(acc, data)
    got = finalize(acc)

    ref = _reference_bold(data_col, warm, kernel, ds=1, tr=tr, k_1=5.6, V_0=0.02)
    assert got.shape == ref.shape
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-11  # strided_convolve ~1e-12 vs FFT


@pytest.mark.parametrize("block_size", [18, 36, 90, 180])  # all multiples of period_in_steps (18)
def test_reducer_is_block_decomposition_invariant(block_size):
    tr, K, T, n = 18, 50, 360, 6
    red = {"kind": "convolution", "source": "S_e", "kernel_call": "hrf_kernel()",
           "ds_steps": 1, "tr_stride": tr, "k_1": 5.6, "V_0": 0.02}
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
