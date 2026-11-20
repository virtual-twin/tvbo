"""A declared subsample takes its samples at whole periods on every backend and however it is computed.

A sample row is the state after its step, so the subsample stamped ``k period`` holds the state ``k period`` into the window: where TVB's periodic monitors, tvboptim's ``SubSampling`` (which a curated ``SubSample`` becomes on tvboptim) and tvbo's streamed stride reducer take it. A pipeline that declares ``subsample(X, step)`` was computed on tvboptim, when not streamed, as ``X[::step]``, the first state of each period (``dt``, ``dt + period``, ...), and stamped at the period's end, so every sample was ``period - dt`` older than its stamp and disagreed with the same declaration on tvb. The probe of ``test_delay_history_from_opening`` reads it in closed form: ``y(t) = t a W x(0)``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import xarray as xr

from tvbo import SimulationExperiment
from tvbo.data.types import SimulationResult

from .test_delay_history_from_opening import DT, DURATION, RTOL, A, W
from .test_tvb_delayed_clock import _settled, _undelayed

PERIOD = 3 * DT
"""A period of three steps, which does not divide the 4 steps the delayed probe's history counts."""

INLINE = {
    "source": ["y"],
    "period": PERIOD,
    "pipeline": [
        {
            "name": "subsample",
            "output": "subsampled",
            "equation": {"rhs": "subsample(X, step)", "parameters": {"step": {"equation": {"rhs": "iround(period / dt)"}}}},
            "arguments": {"step": {"description": "Decimation factor, declared on the equation parameter."}},
            "apply_on_dimension": "time",
        }
    ],
}
"""``y`` decimated by a declared pipeline, with no class reference for a backend to substitute its own monitor for. ``step`` is declared as the curated ``SubSample`` declares it: tvboptim binds an equation's constants from ``equation.parameters``, and jax supplies a sampling argument by its name."""

NAME = "Observed"
"""The observation's name, capitalised: tvb keys an observation by its generated monitor class (``m.title``), which capitalises the first letter, where tvboptim and jax key it by the declared name."""

CURATED = {"iri": "tvbo:observation_models/subsample", "period": PERIOD}
"""The curated ``SubSample``: tvb runs its generated monitor, tvboptim its own ``SubSampling``, jax the curated pipeline ``subsample(X, step)``."""

OPENINGS = {"undelayed": _undelayed, "delayed": lambda: _settled(0.0)}
BACKENDS = [
    pytest.param("tvb", marks=pytest.mark.backend_tvb),
    pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
]


def _stamps_and_y(observation):
    """*observation*'s stamps and its ``y`` as ``(time, node)``, however the backend wraps it.

    tvb and jax return a result whose ``data`` is a labelled array; tvboptim returns a labelled array (streamed) or a solution carrying ``ts``, ``ys`` and ``variable_names`` (materialised). An unlabelled variable axis must hold ``y`` alone.
    """
    if not isinstance(observation, (xr.DataArray, SimulationResult)):
        ys = np.asarray(observation.ys)
        names = list(getattr(observation, "variable_names", None) or [])
        if ys.ndim == 3:
            ys = ys[:, names.index("y"), :] if "y" in names else ys.squeeze(1)
        return np.asarray(observation.ts, dtype=float), ys
    data = observation if isinstance(observation, xr.DataArray) else observation.data
    if "mode" in data.dims:
        data = data.isel(mode=0)
    if "variable" in data.dims:
        data = data.sel(variable="y") if "y" in data["variable"].values else data.squeeze("variable")
    times = data["time"].values
    return times, data.transpose("time", ...).values.reshape(len(times), -1)


def _observed_y(spec, observation, backend):
    """*observation* of *spec*, declared as `NAME`, run on *backend*: its stamps, its ``y`` as ``(time, node)`` and each node's opening ``x``."""
    pytest.importorskip(backend)
    spec = copy.deepcopy(spec)
    spec["observations"] = {NAME: copy.deepcopy(observation)}
    stamps, y = _stamps_and_y(SimulationExperiment(**spec).run(backend).integration.observations[NAME])
    return stamps, y, np.array([node["state"]["x"]["value"] for node in spec["network"]["nodes"]])


def _assert_samples_whole_periods(spec, observation, backend):
    """*observation* of *spec* run on *backend* is stamped ``period, 2 period, ...`` and holds the closed form ``y(t) = t a W x(0)`` at each stamp."""
    stamps, y, opening = _observed_y(spec, observation, backend)
    expected = PERIOD * np.arange(1, int(DURATION / PERIOD) + 1)
    np.testing.assert_allclose(stamps, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y, expected[:, None] * A * (W @ opening)[None, :], rtol=RTOL, atol=RTOL)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("opening", list(OPENINGS))
def test_a_declared_subsample_samples_whole_periods(opening, backend):
    """The same pipeline, materialised, holds the state ``k period`` into the window at the stamp ``k period`` on tvb and on tvboptim, the first sample one period in rather than one step."""
    _assert_samples_whole_periods(OPENINGS[opening](), INLINE, backend)


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize(
    "observation",
    [{**INLINE, "reduce": "streaming"}, CURATED],
    ids=["streamed", "curated"],
)
def test_the_streamed_and_the_curated_subsample_sample_whole_periods(observation):
    """The pipeline streamed through the stride reducer, and the curated ``SubSample`` that tvboptim runs as its own ``SubSampling``, already sampled on that grid; they pin what the materialised pipeline is brought onto."""
    _assert_samples_whole_periods(_undelayed(), observation, "tvboptim")


@pytest.mark.backend_jax
@pytest.mark.parametrize("backend", ["jax"])
@pytest.mark.parametrize("opening", ["undelayed"])
def test_jax_s_curated_subsample_samples_whole_periods(opening, backend):
    """The jax backend computes the curated ``SubSample`` through its pipeline, so the pipeline's grid is its grid: the state ``k period`` into the window at ``k period``, not ``period - dt`` earlier."""
    _assert_samples_whole_periods(OPENINGS[opening](), CURATED, backend)
