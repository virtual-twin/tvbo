"""The CUDA kernel draws the noise the experiment declares, scaled as tvboptim scales its additive noise.

The kernel added ``nsig * curand_normal(...)`` with ``nsig`` declared nowhere: it compiled only for a model that happened to carry a parameter of that name, read it as the spread of each step's increment with no ``sqrt(dt)``, and never saw the declared amplitude. It now reads each state variable's amplitude through `noise_sigma_array`, the reader every backend shares, so ``sigma`` and the dispersion ``nsig`` mean what they mean elsewhere, and adds ``sigma * sqrt(dt) * N(0, 1)`` after the Euler step: the increment tvboptim's solver adds for an `AdditiveNoise`, ``diffusion * sqrt(dt) * dW``. The check evaluates the kernel's factor numerically, so it needs no GPU.
"""

from __future__ import annotations

import copy
import math
import re

import pytest

from tvbo import SimulationExperiment

from .test_initial_state_distribution import ONE_VARIABLE

DT = ONE_VARIABLE["integration"]["step_size"]


def _kernel(noise=None, own_noise=None):
    """The CUDA kernel of G240's one-variable recipe on one node, with *noise* on the integration and *own_noise* on ``x``."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    if noise:
        spec["integration"]["noise"] = noise
    if own_noise:
        spec["dynamics"]["state_variables"]["x"]["noise"] = own_noise
    return SimulationExperiment(**spec).render_code("cuda")


def _unit_normal_factor(source):
    """The factor ``x``'s update multiplies a unit normal by, evaluated at the recipe's step."""
    match = re.search(r"x \+= (.+?) \* curand_normal\(&crndst\)", source)
    assert match, "the kernel draws no noise for x"
    expression = re.sub(r"(\d)f\b", r"\1", match[1]).replace("sqrt_dt", "sqrt(dt)")
    return eval(expression, {"sqrt": math.sqrt, "dt": DT})


def _tvboptim_factor(sigma):
    """What tvboptim's solver multiplies a unit normal by in one step of an `AdditiveNoise` with *sigma*."""
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

    noise = AdditiveNoise(sigma=sigma)
    return float(noise.diffusion(0.0, None, noise.params)) * math.sqrt(DT)


@pytest.mark.parametrize(
    "noise, sigma",
    [({"parameters": {"sigma": {"value": 0.05}}}, 0.05), ({"parameters": {"nsig": {"value": 0.02}}}, math.sqrt(0.04))],
    ids=["sigma", "nsig"],
)
def test_the_kernel_scales_its_draw_as_tvboptim_does(noise, sigma):
    """The declared amplitude, in either spelling, reaches the kernel as tvboptim's ``sigma * sqrt(dt)``."""
    assert _unit_normal_factor(_kernel(noise)) == pytest.approx(_tvboptim_factor(sigma), rel=1e-12)


def test_a_state_variable_s_own_noise_is_drawn():
    """Noise declared on ``x`` alone, with none on the integration, is drawn as well."""
    assert _unit_normal_factor(_kernel(own_noise={"parameters": {"sigma": {"value": 0.03}}})) == pytest.approx(
        _tvboptim_factor(0.03), rel=1e-12
    )


def test_a_deterministic_run_draws_nothing():
    """Without declared noise the kernel neither seeds a generator nor draws."""
    assert "curand_normal" not in _kernel()
