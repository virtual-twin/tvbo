"""One reader for the noise amplitude (``tvbo.utils.noise_sigma``).

``Noise`` admits three spellings of the same physical quantity — ``parameters.sigma``
(σ), ``parameters.nsig`` (dispersion D = σ²/2) and the legacy ``intensity`` slot — and
they used to be read by five separate implementations that disagreed. These tests pin
the shared reader's contract and assert that every backend goes through it, so a recipe
cannot mean a different amplitude on different backends.
"""

import math

import pytest

from tvbo import Dynamics
from tvbo.utils import noise_sigma


def _noise(**kw):
    from tvbo.datamodel.schema import Noise

    return Noise(**kw)


# ── the reader's contract ────────────────────────────────────────────


def test_absent_noise_is_none_not_zero():
    """``None`` distinguishes "no noise declared" from an explicit zero amplitude."""
    assert noise_sigma(None) is None
    assert noise_sigma(_noise(additive=True)) is None      # a block declaring no amplitude
    assert noise_sigma(_noise(parameters={"sigma": {"value": 0.0}})) == 0.0


def test_sigma_is_taken_verbatim():
    assert noise_sigma(_noise(parameters={"sigma": {"value": 0.0316}})) == pytest.approx(0.0316)


def test_nsig_is_a_dispersion():
    """TVB stores D = σ²/2, so σ = sqrt(2 D) — never the raw number."""
    assert noise_sigma(_noise(parameters={"nsig": {"value": 0.5}})) == pytest.approx(1.0)


def test_sigma_wins_over_nsig():
    n = _noise(parameters={"sigma": {"value": 0.2}, "nsig": {"value": 0.5}})
    assert noise_sigma(n) == pytest.approx(0.2)


def test_intensity_is_ambiguous_so_the_caller_declares_the_convention():
    """The whole point of the parameter: the two readings differ by sqrt(2v)/v."""
    n = _noise(intensity={"name": "D", "value": 0.5})
    assert noise_sigma(n, intensity_means="dispersion") == pytest.approx(math.sqrt(1.0))
    assert noise_sigma(n, intensity_means="sigma") == pytest.approx(0.5)


# ── every backend reads through it ───────────────────────────────────

_SV = "state_variables:\n  x: {equation: {rhs: '-x'}, initial_value: 0.1, noise: %s}\n"


def _dyn(noise_yaml):
    return Dynamics.from_string("name: N\n" + _SV % noise_yaml)


@pytest.mark.parametrize(
    "noise_yaml, expected",
    [
        ("{additive: true, parameters: {sigma: {value: 0.0316}}}", 0.0316),
        ("{additive: true, parameters: {nsig: {value: 0.5}}}", 1.0),
        ("{additive: true, intensity: {name: D, value: 0.5}}", math.sqrt(1.0)),
    ],
)
def test_tvboptim_adapter_reads_the_declared_amplitude(noise_yaml, expected):
    """The declared σ reaches tvboptim — it used to be replaced by a hard-coded 0.01."""
    pytest.importorskip("tvboptim")
    from tvbo.adapters.tvboptim import _extract_noise

    assert float(_extract_noise(_dyn(noise_yaml)).params.sigma) == pytest.approx(expected)


def test_tvboptim_adapter_agrees_with_the_codegen_template():
    """The in-process adapter and the experiment template must read one recipe alike."""
    pytest.importorskip("tvboptim")
    from tvbo.adapters.tvboptim import _extract_noise

    for noise_yaml in (
        "{additive: true, parameters: {sigma: {value: 0.00283}}}",
        "{additive: true, parameters: {nsig: {value: 0.004}}}",
        "{additive: true, intensity: {name: D, value: 0.004}}",
    ):
        dyn = _dyn(noise_yaml)
        sv_noise = dyn.state_variables["x"].noise
        template_reading = noise_sigma(sv_noise, intensity_means="dispersion") or 0.0
        assert float(_extract_noise(dyn).params.sigma) == pytest.approx(template_reading)


def test_noise_block_without_an_amplitude_is_not_a_noise_target():
    """No fabricated default: an amplitude nobody wrote is an amplitude nobody meant."""
    pytest.importorskip("tvboptim")
    from tvbo.adapters.tvboptim import _extract_noise

    assert _extract_noise(_dyn("{additive: true}")) is None


def test_base_adapter_reads_intensity_as_sigma():
    """The point-neuron spelling (`intensity: {name: sigma_ext}`) is σ, not a dispersion."""
    from tvbo.adapters.base import BaseAdapter

    dyn = _dyn("{intensity: {name: sigma_ext, value: 1.0, unit: mV}}")
    assert BaseAdapter.get_noise_sigmas(dyn) == [1.0]
    assert BaseAdapter.is_stochastic_dynamics({"N": dyn}) is True
    assert BaseAdapter.is_stochastic_dynamics({"N": _dyn("{additive: true}")}) is False


def test_brian2_membrane_noise_keeps_intensity_as_sigma_with_its_unit():
    from tvbo.adapters.brian2 import _membrane_noise_sigma

    dyn = _dyn("{intensity: {name: sigma_ext, value: 1.0, unit: mV}}")
    assert _membrane_noise_sigma(dyn.state_variables["x"]) == (1.0, "mV")
    assert _membrane_noise_sigma(_dyn("{additive: true}").state_variables["x"]) is None
