"""The curated functions in ``tvbo/database/functions/``, each named by ``iri`` and checked against the library a reader would reach for.

A study's statistic is declared by naming one of these, so the definition lives once in the database and every study that names it computes the same quantity. Each is loaded through ``tvbo:function/<name>``, rendered for jax and numpy, executed in float64, and compared with scipy or numpy.
"""

import numpy as np
import pytest
import scipy.signal
import scipy.spatial.distance
import scipy.special
import scipy.stats

from tvbo import Function

FORMATS = ["jax", "numpy"]


def _curated(name, fmt):
    """The callable of ``tvbo:function/<name>`` for *fmt*, in float64."""
    if fmt == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    return Function.from_string(f"iri: tvbo:function/{name}").to_callable(format=fmt)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def test_every_curated_function_resolves_by_iri():
    """Each file in the folder is addressable as ``tvbo:function/<name>`` and carries an equation."""
    from tvbo.data.registry import database_dir

    names = sorted(path.stem for path in database_dir("Function").glob("*.yaml"))
    assert names == ["FCCorrelation", "SetCosine", "SetMean", "SoftPeakFrequency", "WelchPSD"]
    for name in names:
        function = Function.from_string(f"iri: tvbo:function/{name}")
        assert function.name == name and function.equation.rhs


@pytest.mark.parametrize("fmt", FORMATS)
def test_fc_correlation_is_pearson_over_the_off_diagonal_upper_triangle(fmt, rng):
    a, b = rng.standard_normal((300, 20)), rng.standard_normal((300, 20))
    b[:, :8] += a[:, :8]
    fc1, fc2 = np.corrcoef(a.T), np.corrcoef(b.T)
    upper = np.triu_indices(20, 1)
    got = float(_curated("FCCorrelation", fmt)(fc1, fc2))
    assert abs(got - scipy.stats.pearsonr(fc1[upper], fc2[upper]).statistic) < 1e-12


@pytest.mark.parametrize("fmt", FORMATS)
def test_welch_psd_is_scipys_default_welch(fmt, rng):
    t = np.arange(4000) / 100.0
    x = np.stack([np.sin(2 * np.pi * 9 * t), np.cos(2 * np.pi * 21 * t)], axis=1) + 0.4 * rng.standard_normal((4000, 2))
    got = np.asarray(_curated("WelchPSD", fmt)(x, 100.0, 512, 256, 2048))
    np.testing.assert_allclose(
        got, scipy.signal.welch(x, fs=100.0, nperseg=512, noverlap=256, nfft=2048, axis=0)[1], rtol=1e-10, atol=1e-14
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("beta", [1.0, 150.0])
def test_soft_peak_frequency_is_a_softmax_weighted_mean_frequency(fmt, beta, rng):
    freqs = np.fft.rfftfreq(256, 1 / 100.0)
    psd = np.exp(-(((freqs - 17.0) / 2.0) ** 2)) + 0.05 * rng.random(freqs.size)
    psd[:3] += 4.0
    got = float(_curated("SoftPeakFrequency", fmt)(psd, freqs, beta, 3))
    want = float(scipy.special.softmax(beta * psd[3:] / psd[3:].max()) @ freqs[3:])
    assert abs(got - want) < 1e-10
    if beta == 150.0:
        assert abs(got - 17.0) < 0.5, "a sharp softmax lands on the peak the leading bins would otherwise outweigh"


@pytest.mark.parametrize("fmt", FORMATS)
def test_set_mean_is_numpys_weighted_average(fmt, rng):
    values = rng.standard_normal(30)
    for membership in (rng.integers(0, 2, 30).astype(float), rng.random(30)):
        got = float(_curated("SetMean", fmt)(values, membership))
        assert abs(got - np.average(values, weights=membership)) < 1e-12


@pytest.mark.parametrize("fmt", FORMATS)
def test_set_cosine_is_scipys_cosine_similarity(fmt, rng):
    values = rng.standard_normal(30)
    for membership in (rng.integers(0, 2, 30).astype(float), rng.random(30)):
        got = float(_curated("SetCosine", fmt)(values, membership))
        assert abs(got - (1 - scipy.spatial.distance.cosine(values, membership))) < 1e-12
        assert abs(float(_curated("SetCosine", fmt)(7.0 * values, membership)) - got) < 1e-12
