"""`strided_convolve` array-op: a subsampled HRF convolution with no FFT buffer.

`strided_convolve(X, k, s)` computes the ``'valid'`` convolution of ``X`` (leading
time axis) with kernel ``k``, evaluated ONLY at the ``[s::s]`` output indices — the
samples that survive TR subsampling. It fuses ``fftconvolve(X, k, 'valid')`` and
``subsample data[s::s]`` into one windowed matmul, so the BOLD forward model never
materialises the full-length FFT (the transient-memory peak of the online fit).

The numeric contract, asserted below:
  * BYTE-IDENTICAL to a direct full ``'valid'`` convolution then ``[s::s]`` — same
    arithmetic, discarded outputs skipped.
  * Equal to the FFT path to FFT roundoff (~1e-12 float64) — a different algorithm,
    never bit-for-bit.
"""

import numpy as np
import pytest

from tvbo.parse.expression import parse_eq
from tvbo.codegen.code import JaxPrinter, NumPyPrinter


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    L, n_new, N = 64, 48, 5
    x = rng.standard_normal((L + n_new, N))
    k = rng.standard_normal((L,))
    return x, k, L


def test_renders_for_numpy_and_jax():
    expr = parse_eq("strided_convolve(X, k, s)")
    jax_src = JaxPrinter(module="jnp").doprint(expr)
    np_src = NumPyPrinter(module="np").doprint(expr)
    # Fuses convolve+subsample into a single tensordot — no fftconvolve, no FFT buffer.
    assert "tensordot" in jax_src and "fftconvolve" not in jax_src
    assert "tensordot" in np_src
    assert jax_src.startswith("jnp.") and np_src.startswith("np.")


def test_byte_identical_to_windowed_matmul(signal):
    """strided_convolve == an explicit windowed matmul over the retained indices.

    The point of strided_convolve is that it skips outputs a later subsample would
    discard WITHOUT changing the arithmetic of the ones it keeps. Reconstructing the
    kept outputs as the same reversed-kernel · window dot products must give a
    bit-identical result — the exact equivalence the FFT path cannot offer.
    """
    x, k, L = signal
    s = 16
    src = NumPyPrinter(module="np").doprint(parse_eq("strided_convolve(X, k, s)"))
    got = eval(src, {"np": np, "X": x, "k": k, "s": s})

    n_valid = x.shape[0] - L + 1
    kept = np.arange(s, n_valid, s)
    windows = np.stack([x[j:j + L] for j in kept], axis=0)  # (n_kept, L, N)
    ref = np.tensordot(k[::-1], windows, axes=([0], [1]))

    assert got.shape == ref.shape
    assert np.array_equal(got, ref), "strided must equal the windowed matmul bit-for-bit"


def test_matches_fftconvolve_within_roundoff(signal):
    """strided_convolve equals scipy fftconvolve('valid')[s::s] to FFT roundoff."""
    scipy_signal = pytest.importorskip("scipy.signal")
    x, k, L = signal
    s = 16
    src = NumPyPrinter(module="np").doprint(parse_eq("strided_convolve(X, k, s)"))
    got = eval(src, {"np": np, "X": x, "k": k, "s": s})

    fft = np.stack(
        [scipy_signal.fftconvolve(x[:, n], k, mode="valid") for n in range(x.shape[1])],
        axis=1,
    )[s::s]

    assert np.allclose(got, fft, rtol=1e-9, atol=1e-9)


def test_preserves_trailing_axes(signal):
    """A middle singleton axis (time, 1, node) rides through the reduction."""
    x, k, L = signal
    s = 16
    x3 = x[:, None, :]  # (T, 1, N) — the BOLD prepend_history shape
    src = NumPyPrinter(module="np").doprint(parse_eq("strided_convolve(X, k, s)"))
    got = eval(src, {"np": np, "X": x3, "k": k, "s": s})
    ref2d = eval(src, {"np": np, "X": x, "k": k, "s": s})
    assert got.shape[1] == 1 and got.shape[2] == x.shape[1]
    assert np.allclose(np.squeeze(got, axis=1), ref2d)
