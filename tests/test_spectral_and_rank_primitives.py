"""The spectral, ranking and linear-algebra primitives a study's statistics are declared with, each checked against the library a reader would reach for.

``welch``, ``hann`` and ``rfftfreq`` give a power spectrum; ``rankdata`` gives the ranks a Spearman correlation is a Pearson correlation of; ``pinv`` solves a least-squares design; ``interp`` reads a sampled curve between its samples. Each is rendered for jax and numpy, executed in float64, and compared with scipy or numpy.
"""

import numpy as np
import pytest
import scipy.signal
import scipy.stats

from tvbo.codegen import render_expression
from tvbo.parse.expression import parse_eq

FORMATS = ["jax", "numpy"]


def _run(rhs, fmt, **values):
    """Render *rhs* for *fmt* and evaluate it on *values* in float64, in one namespace as tvbo executes generated code, so a lambda in the expression sees the values."""
    code = render_expression(parse_eq(rhs, parameters=list(values)), format=fmt, parameters=list(values))
    if fmt == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        return np.asarray(eval(code, {"jnp": jnp, **{k: jnp.asarray(v) for k, v in values.items()}}))
    return np.asarray(eval(code, {"np": np, **values}))


@pytest.fixture(scope="module")
def signal():
    """Two channels of noisy oscillation, time on the leading axis."""
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100.0
    return np.stack([np.sin(2 * np.pi * 7 * t), np.sin(2 * np.pi * 13 * t + 1)], axis=1) + 0.3 * rng.standard_normal((3000, 2))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    ("nperseg", "noverlap", "nfft", "detrend"), [(256, 128, 1024, 1), (256, 0, 256, 0), (200, 50, 511, 1)]
)
def test_welch_matches_scipy(fmt, signal, nperseg, noverlap, nfft, detrend):
    got = _run(f"welch(x, hann({nperseg}), fs, {noverlap}, {nfft}, {detrend})", fmt, x=signal, fs=100.0)
    _, want = scipy.signal.welch(
        signal,
        fs=100.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant" if detrend else False,
        axis=0,
    )
    assert got.shape == want.shape == (nfft // 2 + 1, 2)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("fmt", FORMATS)
def test_welch_of_one_channel_keeps_one_axis(fmt, signal):
    got = _run("welch(x, hann(256), 100.0, 128, 512, 1)", fmt, x=signal[:, 0])
    np.testing.assert_allclose(
        got, scipy.signal.welch(signal[:, 0], fs=100.0, nperseg=256, nfft=512)[1], rtol=1e-10, atol=1e-14
    )


def test_welch_refuses_a_detrend_it_cannot_emit():
    with pytest.raises(ValueError, match="detrend must be 0"):
        render_expression(parse_eq("welch(x, hann(8), 1.0, 4, 8, 2)", parameters=["x"]), format="numpy", parameters=["x"])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", [1, 8, 255, 512])
def test_hann_is_scipys_periodic_window(fmt, n):
    np.testing.assert_allclose(_run(f"hann({n})", fmt), scipy.signal.get_window("hann", n), atol=1e-15)


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_integral_float_length_is_accepted_as_the_integer(fmt, signal):
    """A length derived from a period and a step arrives as `512.0`; it is the same window, overlap and FFT length as `512`."""
    np.testing.assert_array_equal(_run("hann(8.0)", fmt), _run("hann(8)", fmt))
    np.testing.assert_array_equal(_run("rfftfreq(16.0, 0.1)", fmt), _run("rfftfreq(16, 0.1)", fmt))
    np.testing.assert_array_equal(
        _run("welch(x, hann(64.0), 100.0, 32.0, 128.0, 1)", fmt, x=signal),
        _run("welch(x, hann(64), 100.0, 32, 128, 1)", fmt, x=signal),
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    "rhs,where",
    [
        ("hann(8.5)", r"hann\(n\): n must be an integral length"),
        ("rfftfreq(7.2, 0.1)", r"rfftfreq\(n, d\): n must be"),
        ("welch(x, hann(8), 1.0, 4.5, 8, 0)", "noverlap must be an integral length"),
        ("welch(x, hann(8), 1.0, 4, 8.1, 0)", "nfft must be an integral length"),
    ],
)
def test_a_non_integral_length_is_refused_naming_the_argument(fmt, rhs, where, signal):
    """Never rounded in silence: the error names the primitive and the argument it cannot turn into a shape."""
    with pytest.raises(ValueError, match=where):
        _run(rhs, fmt, x=signal)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", [511, 1024])
def test_rfftfreq_is_numpys_and_welchs_frequency_axis(fmt, n):
    got = _run(f"rfftfreq({n}, 1 / fs)", fmt, fs=100.0)
    np.testing.assert_allclose(got, np.fft.rfftfreq(n, 1 / 100.0), rtol=1e-15)
    np.testing.assert_allclose(got, scipy.signal.welch(np.ones(n), fs=100.0, nperseg=n)[0], rtol=1e-15)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_rankdata_averages_ties_like_scipy(fmt, axis):
    x = np.array([[3.0, 1.0, 3.0, 2.0], [0.5, 0.5, 0.5, 9.0], [4.0, -1.0, 2.0, 2.0]])
    np.testing.assert_array_equal(_run(f"rankdata(x, {axis})", fmt, x=x), scipy.stats.rankdata(x, method="average", axis=axis))


@pytest.mark.parametrize("fmt", FORMATS)
def test_spearman_composes_as_pearson_of_ranks(fmt):
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(40), rng.standard_normal(40)
    b[:10] = b[0]
    got = _run("pearson(rankdata(a, 0), rankdata(b, 0))", fmt, a=a, b=b)
    assert abs(float(got) - scipy.stats.spearmanr(a, b).statistic) < 1e-12


@pytest.mark.parametrize("fmt", FORMATS)
def test_pinv_cuts_at_the_declared_tolerance(fmt):
    """A singular value between the backends' default cutoffs is kept or dropped by ``rtol`` alone."""
    u, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((6, 3)))
    M = u * np.array([1.0, 1e-3, 1e-13])
    for rtol in (1e-15, 1e-10):
        got = _run(f"pinv(M, {rtol})", fmt, M=M)
        np.testing.assert_allclose(got, np.linalg.pinv(M, rtol=rtol), rtol=1e-6, atol=1e-9)
    assert np.abs(_run("pinv(M, 1e-15)", fmt, M=M)).max() > 1e12 > np.abs(_run("pinv(M, 1e-10)", fmt, M=M)).max()


@pytest.mark.parametrize("fmt", FORMATS)
def test_interp_reads_between_samples_and_holds_the_ends(fmt):
    xp, fp = np.array([0.0, 1.0, 3.0]), np.array([10.0, 20.0, 0.0])
    x = np.array([-1.0, 0.5, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(_run("interp(x, xp, fp)", fmt, x=x, xp=xp, fp=fp), np.interp(x, xp, fp))
