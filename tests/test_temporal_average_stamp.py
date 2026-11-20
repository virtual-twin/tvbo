"""A temporal average is stamped by its backend's convention, however it is declared.

Every window averages the states after steps ``(k - 1) istep + 1 .. k istep``. tvbo stamps its mean, on tvboptim and on jax, with the label tvboptim's built-in ``TemporalAverage`` gives the same window: the mean time of the samples it averages. tvb's own monitor stamps it at the centre of the interval those steps cover, ``(k istep - istep / 2) dt``, half a step earlier, as ``test_tvb_monitor_phase`` pins it. The probe of ``test_delay_history_from_opening`` reads either in closed form: ``y(t) = t a W x(0)`` is linear, so a window's mean is ``y`` at the mean of its samples' times.
"""

from __future__ import annotations

import numpy as np
import pytest

from .test_delay_history_from_opening import DT, DURATION, RTOL, A, W
from .test_subsample_phase import PERIOD, _observed_y
from .test_tvb_delayed_clock import _undelayed

OBSERVATIONS = {
    "curated": {"iri": "tvbo:observation_models/temporal_average_tvb", "period": PERIOD},
    "inline": {
        "source": ["y"],
        "period": PERIOD,
        "pipeline": [
            {
                "name": "temporal_average",
                "output": "averaged",
                "equation": {
                    "rhs": "window_mean(X, window_size)",
                    "parameters": {"window_size": {"equation": {"rhs": "iround(period / dt)"}}},
                },
                "arguments": {"window_size": {"description": "Steps per window, declared on the equation parameter."}},
                "apply_on_dimension": "time",
            }
        ],
    },
}
"""The curated average (tvb's generated monitor, tvboptim's ``TVBTemporalAverage``, jax's pipeline) and a declared ``window_mean`` pipeline with no class reference, its ``window_size`` declared as the curated average declares it."""


def _tvb_interval_centres():
    """TVB's convention: each window's stamp at the centre of the interval its steps cover, and the time its mean reads ``y`` at, half a step later."""
    centres = PERIOD * np.arange(1, int(DURATION / PERIOD) + 1) - PERIOD / 2
    return centres, centres + DT / 2


def _tvboptim_window_labels():
    """The tvboptim convention: the labels tvboptim's built-in ``TemporalAverage`` gives windows of `PERIOD` over the run's post-step samples, each the mean time of the samples it averages and so also where its mean reads ``y``."""
    pytest.importorskip("tvboptim")
    from tvboptim.experimental.network_dynamics.result import NativeSolution
    from tvboptim.observations.tvb_monitors.downsampling import TemporalAverage

    steps = int(round(DURATION / DT))
    raw = NativeSolution(ts=DT * np.arange(1, steps + 1), ys=np.zeros((steps, 1, 1)), dt=DT)
    labels = np.asarray(TemporalAverage(period=PERIOD)(raw).ts, dtype=float)
    return labels, labels


BACKENDS = [
    pytest.param("tvb", _tvb_interval_centres, marks=pytest.mark.backend_tvb, id="tvb"),
    pytest.param("tvboptim", _tvboptim_window_labels, marks=pytest.mark.backend_tvboptim, id="tvboptim"),
    pytest.param("jax", _tvboptim_window_labels, marks=pytest.mark.backend_jax, id="jax"),
]
"""Each backend with the convention its temporal average is stamped by."""


@pytest.mark.parametrize(("backend", "convention"), BACKENDS)
@pytest.mark.parametrize("declaration", list(OBSERVATIONS))
def test_a_temporal_average_is_stamped_by_its_backend_s_convention(declaration, backend, convention):
    """Each 3 ms window carries its backend's convention's stamp, and holds the mean of ``y`` over its three steps, ``y`` at the time that convention reads it."""
    stamps, y, opening = _observed_y(_undelayed(), OBSERVATIONS[declaration], backend)
    expected, reads_at = convention()
    np.testing.assert_allclose(stamps, expected, rtol=0, atol=1e-9)
    closed_form = reads_at[:, None] * A * (W @ opening)[None, :]
    np.testing.assert_allclose(y, closed_form, rtol=RTOL, atol=RTOL)
