"""A streamed BOLD convolves its decimated signal with an HRF kernel sampled on that same grid, or refuses to run.

The reducer takes its decimation from the observation's ``downsample_period`` (or the pipeline's ``subsample`` step) and convolves sample by sample, so a kernel sampled on another spacing is the response stretched in time with nothing raised. The case that produced it: a study declaring its own 1 ms ``hrf_kernel`` under ``tvbo:BOLD_HRF_strided``, whose curated helpers merge non-destructively while its 4 ms ``downsample_period`` still set the decimation, streamed a fourfold-stretched response that correlated with the same run's pipeline BOLD at r 0.15.
"""

from types import SimpleNamespace

import pytest

from tvbo.datamodel.schema import Argument, Equation, Function, FunctionCall, Observation, Parameter, Range
from tvbo.templates.tvboptim.utils import kernel_support_steps, kernel_time_range, resolve_reduction


def _experiment(kernel_range, dt, downsample_period=None, subsample_step=1):
    """A streamed HRF-Volterra BOLD observation and a stub experiment integrating at *dt* ms."""
    functions = {
        "hrf_kernel": Function(
            name="hrf_kernel", time_range=kernel_range, arguments={"duration": Argument(name="duration", value=20000.0)}
        ),
        "subsample": Function(
            name="subsample",
            equation=Equation(rhs="subsample(data, step - 1, step)"),
            arguments={"step": Argument(name="step", value=subsample_step)},
        ),
        "subsample_bold": Function(name="subsample_bold", arguments={"s": Argument(name="s", value=500)}),
        "volterra_transform": Function(
            name="volterra_transform",
            equation=Equation(
                rhs="k_1 * V_0 * (data - 1.0)",
                parameters={"k_1": Parameter(name="k_1", value=5.6), "V_0": Parameter(name="V_0", value=0.02)},
            ),
        ),
    }
    obs = Observation(
        name="bold",
        source=["x"],
        period=2000.0,
        downsample_period=downsample_period,
        reduce="streaming",
        pipeline=[
            FunctionCall(function="hrf_kernel"),
            FunctionCall(function="subsample", output="downsampled_data"),
            FunctionCall(function="volterra_transform"),
            FunctionCall(function="subsample_bold"),
        ],
    )
    return obs, SimpleNamespace(functions=functions, integration=SimpleNamespace(step_size=dt), _source_file=None)


ONE_MS = Range(lo=0, hi="duration", n=20000)
FOUR_MS = Range(lo=0, hi=20000.0, n=5000)


def test_a_one_ms_kernel_under_a_four_ms_decimation_is_refused():
    obs, exp = _experiment(ONE_MS, dt=1.0, downsample_period=4.0)
    with pytest.raises(
        ValueError,
        match=r"every 4 ms \(its downsample_period\).*sampled every 1\.0000\d* ms.*factor of 4.*downsample_period: 1",
    ):
        resolve_reduction(obs, exp)


def test_the_kernels_own_grid_streams_at_any_finer_step():
    obs, exp = _experiment(ONE_MS, dt=0.1, downsample_period=1.0)
    assert resolve_reduction(obs, exp)["ds_steps"] == 10


def test_the_curated_four_ms_kernel_matches_its_own_grid():
    obs, exp = _experiment(FOUR_MS, dt=1.0, downsample_period=4.0)
    assert resolve_reduction(obs, exp)["ds_steps"] == 4


def test_a_subsample_stride_is_checked_the_same_way():
    obs, exp = _experiment(ONE_MS, dt=1.0, subsample_step=4)
    with pytest.raises(ValueError, match=r"its subsample step"):
        resolve_reduction(obs, exp)


def _generator(lo=0, hi=20000.0, n=None, step=None, **arguments):
    """A kernel generator as the resolver reads one; a plain namespace, because the schema's ``Range.n`` takes only an integer."""
    return SimpleNamespace(
        time_range=SimpleNamespace(lo=lo, hi=hi, n=n, step=step),
        arguments={k: SimpleNamespace(value=v) for k, v in arguments.items()},
    )


def test_the_three_ways_a_kernel_samples_itself():
    def spacing(generator):
        return kernel_time_range("k", None, {"k": generator})[2]

    assert spacing(_generator(hi="duration", n="duration / 4 + 1", duration=20000.0)) == pytest.approx(4.0)
    assert spacing(_generator(step="2 * h", h=0.5)) == pytest.approx(1.0)
    assert spacing(_generator(dt=0.5)) == pytest.approx(0.5)
    assert spacing(_generator(n="m")) is None


def test_a_grid_that_does_not_resolve_is_refused_by_name():
    obs, exp = _experiment(ONE_MS, dt=1.0, downsample_period=1.0)
    exp.functions = {**exp.functions, "hrf_kernel": _generator(n="m")}
    with pytest.raises(ValueError, match=r"time_range\.n = m.*does not resolve"):
        resolve_reduction(obs, exp)


def test_the_support_a_kernel_consumes_is_unchanged():
    fns = {
        "hrf_kernel": Function(
            name="hrf_kernel", time_range=ONE_MS, arguments={"duration": Argument(name="duration", value=20000.0)}
        )
    }
    assert kernel_support_steps("hrf_kernel", None, fns, 0.1) == 200000
    assert kernel_support_steps("not_a_kernel", None, fns, 0.1) == 0
