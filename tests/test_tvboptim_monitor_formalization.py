"""Every concrete tvboptim monitor, recreated from a declarative tvbo Observation, is bitwise equal to the tvboptim class: `ys` and `ts`, eager, under jit and vmapped inside a loss, cold and warm-started (`dev/Observation-Formalization.md`).

What the tvboptim class takes as constructor arguments (`period`, `downsample_period`, `dt_bw` as `step_size`, …) the spec declares, and the observation's `source` is what tvboptim calls `voi`; only a warm-start `history` is handed to the generated class at run time. TVB's `SubSample` is tvboptim's `SubSampling`, so one spec pins both, and TVB's `GlobalAverage` is that sampling followed by the mean over nodes.
"""

import re
import types

import numpy as np
import pytest

pytest.importorskip("tvboptim")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations.tvb_monitors import (
    BalloonWindkesselBold,
    DoubleExponentialHRFKernel,
    FirstOrderVolterraHRFKernel,
    GammaHRFKernel,
    HRFBold,
    MixtureOfGammasHRFKernel,
    SubSampling,
    TemporalAverage,
)

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment
from tvbo.templates.tvboptim.utils import monitor_class_name

pytestmark = pytest.mark.xfail(
    reason="phase 2 of dev/Observation-Formalization.md: curated specs, schema slots and codegen not landed yet", strict=False
)

N_NODES = 3
CONCRETE = re.compile(
    r"\b(SubSampling|TemporalAverage|HRFBold|BalloonWindkesselBold|streaming_hrf_bold|\w+HRFKernel|compute_fc|fc_corr|compute_fcd|welford_cov)\b"
)


def _solution(duration, dt, n_var=1, seed=0, scale=1.0):
    """A synthetic trajectory on tvboptim's own clock, ``ts = t0 + (k + 1) dt`` with ``t0 = 0``."""
    n = int(round(duration / dt))
    ys = jnp.asarray(np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, n_var, N_NODES))) * scale
    return NativeSolution(ts=0.0 + (jnp.arange(n) + 1) * dt, ys=ys, dt=dt, variable_names=tuple(f"v{i}" for i in range(n_var)))


def _observation(spec, declared):
    """The curated spec *spec* observing the model's first state variable, tvboptim's ``voi=0``, with *declared* set on it: an Observation slot by its name, anything else as the value of the spec's parameter of that name."""
    observation = Observation.from_db(spec)
    observation.source = ["S"]
    for name, value in declared.items():
        if hasattr(observation, name):
            setattr(observation, name, value)
        else:
            observation.parameters[name].value = value
    return observation


def _experiment(spec, dt, duration, **declared):
    """A two-node experiment measuring *duration* at step *dt*, observed only through the curated spec *spec* with *declared* set on it."""
    network = Network.from_matrix(
        weights=np.array([[0.0, 0.2], [0.1, 0.0]]), lengths=np.zeros((2, 2)), labels=["left", "right"]
    )
    network.coupling["Linear"] = Coupling.from_db("Linear")
    return SimulationExperiment(
        dynamics=Dynamics.from_db("ReducedWongWang"),
        network=network,
        integration={"method": "Heun", "duration": duration, "step_size": dt},
        observations=[_observation(spec, declared)],
    )


def _generated(spec, sol, history=None, **declared):
    """The monitor tvbo generates for *spec* with *declared* set on it, for an experiment that measures exactly *sol* (a longer trajectory would carry its excess as the settle), warm-started on *history* when one is given."""
    module = types.ModuleType(f"generated_{spec}")
    code = _experiment(spec, sol.dt, sol.ts.shape[0] * sol.dt, **declared).render_code("tvboptim")
    exec(compile(code, f"<generated {spec}>", "exec"), module.__dict__)
    monitor = getattr(module, monitor_class_name(spec))
    return monitor() if history is None else monitor(history=history)


def _assert_bitwise(got, expected):
    for field in ("ys", "ts"):
        a, b = np.asarray(getattr(got, field)), np.asarray(getattr(expected, field))
        assert a.shape == b.shape, f"{field}: shape {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"{field}: dtype {a.dtype} vs {b.dtype}"
        assert np.array_equal(a, b), f"{field}: {int(np.sum(a != b))}/{a.size} differ, max|d|={np.abs(a - b).max():.2e}"
    assert got.dt == expected.dt
    assert tuple(got.variable_names or ()) == tuple(expected.variable_names or ())


def _call(monitor, sol, jit):
    return jax.jit(lambda s: monitor(s))(sol) if jit else monitor(sol)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("dt", "period"), [(4.0, 4.0), (4.0, 16.0), (0.1, 4.0), (0.1, 1.0)])
def test_sub_sampling(dt, period, jit):
    sol = _solution(400.0, dt)
    _assert_bitwise(
        _call(_generated("SubSample", sol, period=period), sol, jit),
        _call(SubSampling(voi=0, period=period), sol, jit),
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("dt", "period"), [(4.0, 4.0), (4.0, 16.0), (0.1, 4.0), (0.1, 1.0), (0.05, 2.0)])
def test_temporal_average_including_its_truncated_window_starts(dt, period, jit):
    sol = _solution(400.0, dt)
    _assert_bitwise(
        _call(_generated("TemporalAverage_tvboptim", sol, period=period), sol, jit),
        _call(TemporalAverage(voi=0, period=period), sol, jit),
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("dt", "period"), [(4.0, 4.0), (0.1, 4.0), (0.5, 2.0)])
def test_global_average_is_the_sampling_averaged_over_nodes(dt, period, jit):
    sol = _solution(400.0, dt)
    sampled = _call(SubSampling(voi=0, period=period), sol, jit)
    expected = NativeSolution(
        ts=sampled.ts, ys=jnp.mean(sampled.ys, axis=2, keepdims=True), dt=period, variable_names=sampled.variable_names
    )
    _assert_bitwise(_call(_generated("GlobalAverage", sol, period=period), sol, jit), expected)


KERNELS = {
    "FirstOrderVolterra": FirstOrderVolterraHRFKernel(),
    "Gamma": GammaHRFKernel(),
    "DoubleExponential": DoubleExponentialHRFKernel(),
    "MixtureOfGammas": MixtureOfGammasHRFKernel(),
}


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "history"])
@pytest.mark.parametrize("kernel", list(KERNELS))
@pytest.mark.parametrize("dt", [4.0, 0.1])
def test_hrf_bold(dt, kernel, warm, jit):
    sol = _solution(60_000.0, dt)
    history = _solution(30_000.0, dt, seed=1) if warm else None
    reference = HRFBold(period=1000.0, downsample_period=4.0, voi=0, kernel=KERNELS[kernel], history=history)
    generated = _generated(f"HRFBold_{kernel}", sol, history, period=1000.0, downsample_period=4.0)
    _assert_bitwise(_call(generated, sol, jit), _call(reference, sol, jit))


def test_hrf_bold_inside_a_vmapped_loss():
    """A parameter grid over the input's gain, the way a sweep calls a monitor inside its loss."""
    dt = 4.0
    sol, history = _solution(60_000.0, dt), _solution(30_000.0, dt, seed=1)
    gains = jnp.linspace(0.5, 1.5, 5)

    def swept(monitor):
        return jax.jit(
            jax.vmap(
                lambda g: monitor(NativeSolution(ts=sol.ts, ys=sol.ys * g, dt=sol.dt, variable_names=sol.variable_names)).ys
            )
        )(gains)

    got = swept(_generated("HRFBold_FirstOrderVolterra", sol, history, period=1000.0, downsample_period=4.0))
    expected = swept(HRFBold(period=1000.0, downsample_period=4.0, voi=0, history=history))
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_hrf_bold_gradient_through_a_loss():
    """The gradient a fit takes through the monitor, with respect to the input's gain."""
    dt = 4.0
    sol, history = _solution(60_000.0, dt), _solution(30_000.0, dt, seed=1)

    def gradient(monitor):
        return jax.jit(
            jax.grad(
                lambda g: jnp.sum(
                    monitor(NativeSolution(ts=sol.ts, ys=sol.ys * g, dt=sol.dt, variable_names=sol.variable_names)).ys ** 2
                )
            )
        )(1.1)

    got = gradient(_generated("HRFBold_FirstOrderVolterra", sol, history, period=1000.0, downsample_period=4.0))
    expected = gradient(HRFBold(period=1000.0, downsample_period=4.0, voi=0, history=history))
    assert np.array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("dt", "dt_bw"), [(1.0, 1.0), (2.0, 1.0), (0.5, 1.0)], ids=["same", "repeat", "stride"])
def test_balloon_windkessel_bold(dt, dt_bw, jit):
    sol = _solution(20_000.0, dt, scale=5.0)
    reference = BalloonWindkesselBold(period=2000.0, dt_bw=dt_bw, voi=0)
    generated = _generated("BalloonWindkesselBold", sol, period=2000.0, step_size=dt_bw)
    _assert_bitwise(_call(generated, sol, jit), _call(reference, sol, jit))


CARRYING = {
    "BOLD_HRF_strided": (4.0, 720.0, {"reduce": "streaming"}),
    "BalloonWindkesselBold": (1.0, 2000.0, {}),
}
"""Streaming monitors with memory, a kernel's ring and an observer's states, as `(dt, period, declared)`."""


def _ys(output):
    return np.asarray(output.ys if hasattr(output, "ys") else output)


def _windows(sol, length):
    """*sol* cut into consecutive *length*-step windows, as a tuning loop hands them to its monitor one at a time."""
    return [
        NativeSolution(ts=sol.ts[i : i + length], ys=sol.ys[i : i + length], dt=sol.dt, variable_names=sol.variable_names)
        for i in range(0, sol.ys.shape[0] - length + 1, length)
    ]


@pytest.mark.parametrize("spec", list(CARRYING))
def test_a_streaming_monitor_carried_across_windows_reports_what_one_fold_over_them_reports(spec):
    dt, period, declared = CARRYING[spec]
    sol = _solution(30 * period, dt, scale=5.0)
    monitor = _generated(spec, sol, **declared)
    whole = _ys(monitor(sol))
    windows = _windows(sol, int(round(period / dt)))
    carried, pieces = monitor.open_warmup(N_NODES), []
    for window in windows:
        pieces.append(_ys(carried(window)))
        carried = carried.carry_warmup(window)
    assert np.array_equal(np.concatenate(pieces), whole)
    assert not np.array_equal(np.concatenate([_ys(monitor(window)) for window in windows]), whole), (
        "each window opening cold must differ"
    )


@pytest.mark.parametrize("spec", list(CARRYING))
def test_a_streaming_monitor_carried_through_a_scan_reports_what_one_jitted_fold_reports(spec):
    dt, period, declared = CARRYING[spec]
    sol = _solution(30 * period, dt, scale=5.0)
    monitor = _generated(spec, sol, **declared)
    length = int(round(period / dt))

    def step(carried, ys):
        window = NativeSolution(ts=(jnp.arange(length) + 1) * dt, ys=ys, dt=dt, variable_names=sol.variable_names)
        return carried.carry_warmup(window), _call(carried, window, jit=False)

    stacked = sol.ys.reshape(-1, length, *sol.ys.shape[1:])
    _, pieces = jax.jit(lambda m, xs: jax.lax.scan(step, m, xs))(monitor.open_warmup(N_NODES), stacked)
    got = np.concatenate(list(_ys(pieces)))
    assert np.array_equal(got, _ys(_call(monitor, sol, jit=True)))


@pytest.mark.parametrize(("spec", "extra"), [("BOLD_HRF_strided", 0), ("BOLD_HRF_strided", 7), ("BalloonWindkesselBold", 0)])
def test_a_settle_run_as_its_own_scan_warms_a_streaming_monitor_as_an_in_band_settle_does(spec, extra):
    dt, period, declared = CARRYING[spec]
    settle = _solution(30 * period + extra * dt, dt, seed=1, scale=5.0)
    measured = _solution(10 * period, dt, seed=2, scale=5.0)
    joined = NativeSolution(
        ts=jnp.concatenate([settle.ts, settle.ts[-1] + measured.ts]),
        ys=jnp.concatenate([settle.ys, measured.ys]),
        dt=dt,
        variable_names=measured.variable_names,
    )
    monitor = _generated(spec, measured, **declared)
    assert np.array_equal(_ys(monitor.carry_warmup(settle)(measured)), _ys(monitor(joined)))


@pytest.mark.parametrize(
    "spec",
    [
        "SubSample",
        "TemporalAverage_tvboptim",
        "GlobalAverage",
        "HRFBold_FirstOrderVolterra",
        "HRFBold_Gamma",
        "HRFBold_DoubleExponential",
        "HRFBold_MixtureOfGammas",
        "BalloonWindkesselBold",
    ],
)
def test_generated_code_uses_only_abstract_tvboptim_classes(spec):
    code = _experiment(spec, 1.0, 1000.0).render_code("tvboptim")
    assert f"class {monitor_class_name(spec)}(AbstractMonitor)" in code, "the monitor is not generated from its spec"
    imports = [
        line.strip() for line in code.splitlines() if line.lstrip().startswith(("import ", "from ")) and "tvboptim" in line
    ]
    offending = [line for line in imports if CONCRETE.search(line)]
    assert not offending, offending
