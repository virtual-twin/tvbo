# -*- coding: utf-8 -*-
#
# Module: observation_sampling.py
#
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
"""Backend-agnostic resolver for observation-monitor sampling step counts.

Observation models (e.g. ``BOLD_TVB``) are declared once as a backend-neutral
YAML pipeline. The *number of samples* an observation emits, however, depends on the integration time-step ``dt`` chosen at run time, not on any value that can
be frozen into the YAML. Freezing a step count into the pipeline (as
``subsample_to_period.stepsize = 180``) only holds when the input already sits on a particular stock grid; a backend that applies that literal to the raw
integration grid produces the wrong sample count.

This module is the single source of truth that every Python backend (tvboptim, jax, tvb) uses to turn ``(declarative observation, integration dt)`` into the
integer step counts that drive downsampling. Backends legitimately diverge only in *how* those counts are applied (circular-buffer convolution vs. functional
window-mean+subsample vs. TVB monitor step-mod loop); the counts themselves must be identical.

It is intentionally free of heavy dependencies (no jax/tvb/juliacall, no
``tvbo.classes``) so both the tvboptim runtime module and the export adapters can import it cheaply and without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "ObservationSampling",
    "resolve_observation_sampling",
    "tvb_iround",
]


def tvb_iround(value: float) -> int:
    """Round half-down to an integer, matching TVB monitor step counting.

    This is the canonical rounding shared by every backend so that boundary ratios resolve to the same step count everywhere.
    """
    rounded = round(value) - 0.5
    return int(rounded) + (rounded > 0)


# Backwards/dual name: the tvboptim runtime historically called this
# ``_tvb_iround``. Keep an alias so imports of either name resolve here.
_tvb_iround = tvb_iround


def _to_numeric(value: Any) -> Any:
    """Coerce a string to int/float when possible, otherwise return as-is."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value) if "." not in value else float(value)
        except ValueError:
            return value
    return value


def _iter_parameter_values(parameters: Any):
    """Yield ``(name, value)`` pairs from a schema Parameter collection."""
    if not parameters:
        return
    if hasattr(parameters, "items"):
        items = parameters.items()
    elif isinstance(parameters, (list, tuple)):
        items = ((getattr(p, "name", None), p) for p in parameters)
    else:
        items = []
    for key, parameter in items:
        name = getattr(parameter, "name", None) or key
        value = getattr(parameter, "value", parameter)
        if name is not None and value is not None:
            yield str(name), _to_numeric(value)


def _parameter_value(parameters: Any, name: str, default: Any = None) -> Any:
    for parameter_name, value in _iter_parameter_values(parameters):
        if parameter_name == name:
            return value
    return default


def _pipeline_argument(pipeline: Any, name: str) -> Any:
    """Return the named pipeline argument object (arguments are keyed by name)."""
    for step in pipeline or []:
        args = getattr(step, "arguments", None) or {}
        if hasattr(args, "__contains__") and name in args:
            return args[name]
    return None


def _time_argument_ms(argument: Any, default: float) -> float:
    """Resolve a time-valued schema argument to milliseconds."""
    if argument is None or getattr(argument, "value", None) is None:
        return default
    value = float(_to_numeric(argument.value))
    unit = str(getattr(argument, "unit", "") or "").lower()
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1000.0
    return value


def _obs_period(observation: Any) -> Optional[float]:
    """The output sampling period in ms (``TR``): explicit ``period`` or ``TR``."""
    period = getattr(observation, "period", None)
    if period is None:
        period = _parameter_value(getattr(observation, "parameters", None), "TR")
    if period is None:
        return None
    return float(_to_numeric(period))


def _obs_downsample_period(observation: Any, default: float) -> float:
    """The interim (stock-grid) period in ms, from the pipeline ``stock_dt`` arg."""
    stock_dt = _pipeline_argument(getattr(observation, "pipeline", None), "stock_dt")
    return _time_argument_ms(stock_dt, default)


@dataclass(frozen=True)
class ObservationSampling:
    """Resolved sampling step counts for one observation at a given ``dt``.

    Attributes:
        period: Output sampling period in ms (``TR``).
        downsample_period: Interim/stock-grid period in ms.
        interim_istep: Integration steps per interim (stock) sample.
        output_istep: Integration steps per output sample (``period / dt``).
        output_interim_count: Interim samples per output sample
            (``output_istep // interim_istep``).
    """

    period: Optional[float]
    downsample_period: float
    interim_istep: int
    output_istep: int
    output_interim_count: int


def resolve_observation_sampling(
    observation: Any,
    dt: float,
    *,
    default_downsample_period: float = 4.0,
) -> ObservationSampling:
    """Resolve an observation's sampling step counts from the integration ``dt``.

    This is the single supervenient resolver: it computes the integer step counts from the declarative observation plus the run-time ``dt``. Every
    Python backend routes through it so the emitted sample count is identical.

    Args:
        observation: A schema observation (must expose ``pipeline`` and either
            ``period`` or a ``TR`` parameter for output-sampled monitors).
        dt: Integration time-step in ms.
        default_downsample_period: Fallback interim period (ms) when the pipeline
            declares no ``stock_dt`` argument.

    Returns:
        An :class:`ObservationSampling` with the resolved step counts. When the
        observation declares no output period, ``output_istep`` /
        ``output_interim_count`` are 0.
    """
    dt = float(dt)
    period = _obs_period(observation)
    downsample_period = _obs_downsample_period(observation, default_downsample_period)

    interim_istep = tvb_iround(downsample_period / dt)
    output_istep = tvb_iround(period / dt) if period is not None else 0
    output_interim_count = output_istep // interim_istep if interim_istep else 0

    return ObservationSampling(
        period=period,
        downsample_period=downsample_period,
        interim_istep=interim_istep,
        output_istep=output_istep,
        output_interim_count=output_interim_count,
    )
