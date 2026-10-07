"""The DifferentialEquations.jl backend integrates by the declared method over the declared window.

A single decay ``x' = -x`` from 1 has a closed-form discrete trajectory for each fixed-step method: Euler multiplies by ``1 - h`` per step, classical RK4 by ``1 - h + h²/2 - h³/6 + h⁴/24``. The settle is integrated ahead of the measured window in the same run, so the first measured sample is ``settle / h + 1`` steps in, as on every other backend.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("juliacall")

from tvbo.classes.experiment import SimulationExperiment

H, SETTLE, DURATION = 0.1, 0.5, 1.0
GROWTH = {"Euler": 1 - H, "RungeKutta4thOrder": 1 - H + H**2 / 2 - H**3 / 6 + H**4 / 24}

_SPEC = """
id: 1
dynamics:
  name: Decay
  state_variables:
    x: {equation: {rhs: "-x"}, initial_value: 1.0}
integration: {method: %(method)s, step_size: %(h)s, duration: %(duration)s, transient_time: %(settle)s}
"""


@pytest.mark.julia
@pytest.mark.backend_julia_diffeq
@pytest.mark.xdist_group("julia")
@pytest.mark.parametrize("method", sorted(GROWTH))
def test_the_declared_method_steps_the_declared_window(tmp_path, method):
    spec = tmp_path / "decay.yaml"
    spec.write_text(_SPEC % {"method": method, "h": H, "duration": DURATION, "settle": SETTLE})

    sim = SimulationExperiment.from_file(str(spec)).run("julia").integration
    measured = sim.data.sel(variable="x").squeeze()
    settle = sim.transient.data.time.values

    steps = np.arange(1, round(DURATION / H) + 1) + round(SETTLE / H)
    np.testing.assert_allclose(measured["time"].values, np.arange(1, len(steps) + 1) * H)
    assert settle[0] == pytest.approx(-SETTLE) and settle[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(measured.values, GROWTH[method] ** steps, rtol=1e-12, atol=0)
