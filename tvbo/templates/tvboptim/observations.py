"""TVB-compatible observation monitors for generated tvboptim code."""

import importlib.util
import math

import equinox as eqx
import numpy as np

from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations.tvb_monitors.downsampling import AbstractMonitor, _slice_variable_names


def _tvb_iround(value: float) -> int:
    rounded = round(value) - 0.5
    return int(rounded) + (rounded > 0)


class TVBTemporalAverage(AbstractMonitor):
    """Temporal average with TVB monitor window indexing."""

    period: float = eqx.field(static=True)

    def __init__(self, voi=None, period=4.0):
        self.voi = self._normalize_voi(voi)
        self.period = period

    def __call__(self, sol):
        dt = self._resolve_dt(sol)
        step_count = _tvb_iround(self.period / dt)
        ys = np.asarray(sol.ys)[:, self.voi, ...]
        window_count = ys.shape[0] // step_count
        windowed = ys[: window_count * step_count].reshape(window_count, step_count, *ys.shape[1:])
        averaged = np.mean(windowed, axis=1)

        t0 = float(np.asarray(sol.ts)[0]) - dt
        step_numbers = np.arange(1, window_count + 1) * step_count
        times = t0 + (step_numbers - step_count / 2.0) * dt
        return NativeSolution(
            ts=times,
            ys=averaged,
            dt=self.period,
            variable_names=_slice_variable_names(sol, self.voi),
        )


class TVBBold(AbstractMonitor):
    """FirstOrderVolterra BOLD monitor with TVB stock-buffer semantics."""

    period: float = eqx.field(static=True)
    downsample_period: float = eqx.field(static=True)
    hrf_length: float = eqx.field(static=True)
    tau_s: float = eqx.field(static=True)
    tau_f: float = eqx.field(static=True)
    k_1: float = eqx.field(static=True)
    V_0: float = eqx.field(static=True)
    scaling: float = eqx.field(static=True)
    hrf_equation: str = eqx.field(static=True)
    hrf_parameters: object = eqx.field(static=True)
    history: object = eqx.field(static=True)

    def __init__(
        self,
        k_1=5.6,
        V_0=0.02,
        period=1000.0,
        downsample_period=4.0,
        hrf_length=20000.0,
        tau_s=0.8,
        tau_f=0.4,
        scaling=1.0 / 3.0,
        hrf_equation="1/3 * exp(-0.5*(t / tau_s)) * sin(sqrt(1/tau_f - 1/(4*tau_s**2)) * t) / sqrt(1/tau_f - 1/(4*tau_s**2))",
        hrf_parameters=None,
        voi=None,
        history=None,
    ):
        self.voi = self._normalize_voi(voi)
        self.k_1 = k_1
        self.V_0 = V_0
        self.period = period
        self.downsample_period = downsample_period
        self.hrf_length = hrf_length
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.scaling = scaling
        self.hrf_equation = hrf_equation
        self.hrf_parameters = hrf_parameters or {}
        self.history = history

    def _hrf(self):
        stock_sample_rate = 2.0**-2
        stock_steps = int(math.ceil(stock_sample_rate * self.hrf_length))
        stock_time_max = self.hrf_length / 1000.0
        stock_time_step = stock_time_max / stock_steps
        stock_time = np.arange(0.0, stock_time_max, stock_time_step)
        namespace = {
            "t": stock_time,
            "var": stock_time,
            "tau_s": self.tau_s,
            "tau_f": self.tau_f,
            "k_1": self.k_1,
            "V_0": self.V_0,
            "scaling": self.scaling,
            **dict(self.hrf_parameters),
        }
        if importlib.util.find_spec("numexpr") is not None:
            import numexpr

            kernel_values = numexpr.evaluate(self.hrf_equation, local_dict=namespace)
        else:
            kernel_values = eval(self.hrf_equation, np.__dict__, namespace)
        return kernel_values[::-1][np.newaxis, :]

    def __call__(self, sol):
        dt = self._resolve_dt(sol)
        interim_step_count = _tvb_iround(self.downsample_period / dt)
        output_step_count = _tvb_iround(self.period / dt)
        output_interim_count = output_step_count // interim_step_count

        interim = TVBTemporalAverage(voi=self.voi, period=self.downsample_period)(sol).ys
        squeeze_mode = interim.ndim == 3
        if squeeze_mode:
            interim = interim[..., np.newaxis]
        hrf = self._hrf()
        stock = np.zeros((hrf.shape[1], *interim.shape[1:]))

        outputs = []
        times = []
        t0 = float(np.asarray(sol.ts)[0]) - dt
        for interim_index, interim_sample in enumerate(interim, start=1):
            stock[(interim_index % stock.shape[0]) - 1, :] = interim_sample
            if interim_index % output_interim_count == 0:
                roll_index = (interim_index % stock.shape[0]) - 1
                rolled_hrf = np.roll(hrf, roll_index, axis=1)
                stock_for_dot = stock.transpose((1, 2, 0, 3))
                bold = (np.dot(rolled_hrf, stock_for_dot) - 1.0) * self.k_1 * self.V_0
                output = bold.reshape(stock.shape[1:])
                outputs.append(output[..., 0] if squeeze_mode else output)
                times.append(t0 + interim_index * interim_step_count * dt)

        return NativeSolution(
            ts=np.asarray(times),
            ys=np.asarray(outputs),
            dt=self.period,
            variable_names=_slice_variable_names(sol, self.voi),
        )