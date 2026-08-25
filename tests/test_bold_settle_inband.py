"""A kernel observation reports the measured window, warmed by the settle and by nothing else.

`transient_time` marks the head of ONE integration, so an HRF pipeline sees the settle as ordinary signal: its convolution warms on real data instead of on a separately handed-over history. The contract that makes that safe has three parts, and this module pins each without running a simulation, so the assertions are about the observation and not about an integrator:

* the reported series covers the measured window — one sample per TR, no more and no fewer, and is stamped on the measured window's own clock;
* it depends on the settle ONLY through the kernel's own support, so lengthening the settle or changing it further back leaves every reported sample untouched;
* it does depend on that support, so the first property is a real invariance and not an observation that ignores its warm-up.

The third is what keeps the second honest: a pipeline that dropped the settle entirely would satisfy "changing the settle changes nothing" trivially, and would be wrong.

A tuning loop hands the same observation one short window per iteration, and no single one of them carries a settle. The last property covers that case: the observation carries its own warm-up between calls, so the samples it reports across a run of windows are the samples one uninterrupted window would have reported.
"""

import types

import pytest

pytest.importorskip("tvboptim")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from tvbo import SimulationExperiment, database_path
from tvbo.export.formats import _render_tvboptim

DT = 4.0
TR_MS = 1000.0
KERNEL_MS = 20000.0  # the recipe's HRF support
N_NODES = 5


class _Trajectory:
    """The minimum a monitor reads: a time axis on the measurement clock and (time, state, node) data."""

    def __init__(self, data, ts):
        self.data = self.ys = data
        self.ts = self.time = ts
        self.dt = DT
        self.variable_names = ["S"]


def _bold_monitor(transient_ms, duration_ms):
    """The `Bold` class the codegen emits for a recipe with this settle and this measured window."""
    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "RWW_BOLD_FC_Optimization.yaml"))
    exp.explorations = None
    exp.integration.step_size = DT
    exp.integration.transient_time = transient_ms
    exp.integration.duration = duration_ms
    module = types.ModuleType("generated")
    exec(compile(_render_tvboptim(exp), "<generated>", "exec"), module.__dict__)
    return module.Bold(voi=0, dt=DT)


def _window(settle_steps, measured_steps, *, seed, settle_head=None):
    """A synthetic window. `settle_head` replaces everything before the last kernel support."""
    n = settle_steps + measured_steps
    data = 0.3 + 0.05 * jax.random.normal(jax.random.key(seed), (n, 1, N_NODES))
    if settle_head is not None:
        keep = int(round(KERNEL_MS / DT))
        head = 0.3 + 0.05 * jax.random.normal(jax.random.key(settle_head), (max(0, settle_steps - keep), 1, N_NODES))
        data = data.at[: head.shape[0]].set(head)
    ts = jnp.arange(1, n + 1) * DT - settle_steps * DT
    return _Trajectory(data, ts)


@pytest.mark.parametrize("transient_ms,duration_ms", [(40000.0, 4000.0), (60000.0, 8000.0), (20000.0, 4000.0)])
def test_reports_one_sample_per_tr_of_the_measured_window(transient_ms, duration_ms):
    """The settle is measured, not reported: its length never changes how much comes back."""
    monitor = _bold_monitor(transient_ms, duration_ms)
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    out = np.asarray(monitor(_window(settle_steps, measured_steps, seed=0)).data)

    assert out.shape[0] == int(duration_ms // TR_MS), (
        f"{out.shape[0]} samples for a {duration_ms:g}ms window at TR {TR_MS:g}ms"
    )
    assert np.isfinite(out).all()


def _responding_outputs(monitor, settle_steps, measured_steps, measured_step):
    """Which reported samples a unit delta at `measured_step` (1-based) moves."""
    n = settle_steps + measured_steps
    ts = jnp.arange(1, n + 1) * DT - settle_steps * DT
    base = jnp.zeros((n, 1, N_NODES))
    quiet = np.asarray(monitor(_Trajectory(base, ts)).data)
    struck = np.asarray(monitor(_Trajectory(base.at[settle_steps + measured_step - 1].set(1.0), ts)).data)
    moved = np.abs(struck - quiet).max(axis=tuple(range(1, quiet.ndim))) > 1e-12
    return np.nonzero(moved)[0]


@pytest.mark.parametrize("transient_ms,duration_ms", [(40000.0, 4000.0), (60000.0, 8000.0), (8000.0, 4000.0)])
def test_the_report_is_stamped_on_the_measured_window(transient_ms, duration_ms):
    """The whole settle comes off the time axis, however much of it the kernel kept as signal.

    The kernel deliberately leaves its own support in front of `t = 0` so the convolution warms on real data — but what the observation *reports* is the measured window, so that is the clock it reports on. Cutting the axis by what the data was cut by instead would stamp a measured sample with a settle timestamp.

    The expected values are NOT read off the emitting expression — see `test_each_sample_is_stamped_where_it_actually_sits`, which locates the samples by perturbation and is what makes this parametrisation meaningful rather than circular.
    """
    monitor = _bold_monitor(transient_ms, duration_ms)
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    ts = np.asarray(monitor(_window(settle_steps, measured_steps, seed=1)).ts)

    expected = (np.arange(int(duration_ms // TR_MS)) + 1) * TR_MS
    assert ts.shape == expected.shape
    np.testing.assert_allclose(ts, expected)
    assert ts[0] > 0, "the first reported sample is stamped inside the settle"


def test_each_sample_is_stamped_where_it_actually_sits():
    """A reported sample's timestamp must name the last measured step that can still move it.

    This is the only assertion here that does not take the emitting code's word for the time axis. A sample reports a whole TR, so it covers the steps up to its own timestamp and none after: perturbing the step AT `ts[m]` must move sample `m`, and perturbing the very next step must not. A phase error of one period — stamping the first sample at the window's opening step rather than at the end of the period it integrates — shows up here and nowhere else, which is how it survived the rest of this module.
    """
    transient_ms, duration_ms = 40000.0, 8000.0
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    monitor = _bold_monitor(transient_ms, duration_ms)
    ts = np.asarray(monitor(_window(settle_steps, measured_steps, seed=0)).ts)
    tr_steps = int(round(TR_MS / DT))

    for m in (0, 1, len(ts) - 1):
        at = int(round(ts[m] / DT))
        assert m in _responding_outputs(monitor, settle_steps, measured_steps, at), (
            f"sample {m} is stamped {ts[m]:g}ms but the step at that time does not move it"
        )
        if at + 1 <= measured_steps:
            assert m not in _responding_outputs(monitor, settle_steps, measured_steps, at + 1), (
                f"sample {m} is stamped {ts[m]:g}ms but a later step still moves it"
            )


def test_the_settle_reaches_the_report_only_through_the_kernels_support():
    """Two runs sharing their measurement and their last kernel support report the same samples.

    This is what "the settle warms it in-band" has to mean. Everything further back is warm-up for the warm-up, and a pipeline that let it through would make a reported sample depend on how long the settle happened to be.
    """
    transient_ms, duration_ms = 60000.0, 4000.0
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    monitor = _bold_monitor(transient_ms, duration_ms)

    same_tail = np.asarray(monitor(_window(settle_steps, measured_steps, seed=0)).data)
    other_head = np.asarray(monitor(_window(settle_steps, measured_steps, seed=0, settle_head=99)).data)

    assert same_tail.shape == other_head.shape
    # Not bit-identical, and cannot be: the convolution is an FFT over the whole window, so signal outside the kernel's support still reaches the report's last bits. The bound is float64 round-off; `test_the_kernels_support_does_reach_the_report` shows what a real dependence looks like.
    assert float(np.max(np.abs(same_tail - other_head))) < 1e-12 * float(np.max(np.abs(same_tail)))


def test_the_kernels_support_does_reach_the_report():
    """The invariance above is real only because the support itself is used; disturb it and the report moves."""
    transient_ms, duration_ms = 60000.0, 4000.0
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    monitor = _bold_monitor(transient_ms, duration_ms)

    base = _window(settle_steps, measured_steps, seed=0)
    disturbed = _Trajectory(base.data.at[settle_steps - 1].add(1.0), base.ts)

    out_base = np.asarray(monitor(base).data)
    out_disturbed = np.asarray(monitor(disturbed).data)

    assert float(np.max(np.abs(out_base - out_disturbed))) > 1e-6, "the last settle step left no trace on the report"


def test_a_settle_shorter_than_the_kernel_opens_on_zeros():
    """A settle too short to fill the kernel is padded, not refused: the shortfall opens on zeros and the window still reports in full."""
    transient_ms, duration_ms = 8000.0, 4000.0  # 8 s of settle against a 20 s kernel
    settle_steps = int(round(transient_ms / DT))
    measured_steps = int(round(duration_ms / DT))
    out = np.asarray(_bold_monitor(transient_ms, duration_ms)(_window(settle_steps, measured_steps, seed=3)).data)

    assert out.shape[0] == int(duration_ms // TR_MS)
    assert np.isfinite(out).all()


def test_successive_windows_report_what_one_window_would_have():
    """A tuning loop's windows, warmed from one another, convolve as if they had never been cut apart."""
    n_windows = 12
    settle_steps = int(round(KERNEL_MS / DT))
    tr_steps = int(round(TR_MS / DT))
    stream = _window(settle_steps, n_windows * tr_steps, seed=0)

    whole = np.asarray(_bold_monitor(KERNEL_MS, n_windows * TR_MS)(stream).data)

    monitor = _bold_monitor(KERNEL_MS, n_windows * TR_MS).open_warmup(N_NODES)
    monitor = monitor.carry_warmup(_Trajectory(stream.data[:settle_steps], stream.ts[:settle_steps]))
    pieces = []
    for i in range(n_windows):
        lo, hi = settle_steps + i * tr_steps, settle_steps + (i + 1) * tr_steps
        piece = _Trajectory(stream.data[lo:hi], stream.ts[lo:hi])
        pieces.append(np.asarray(monitor(piece).data))
        monitor = monitor.carry_warmup(piece)
    stitched = np.concatenate(pieces, axis=0)

    assert stitched.shape == whole.shape
    assert float(np.max(np.abs(whole - stitched))) < 1e-12 * float(np.max(np.abs(whole)))


def test_a_window_that_carries_no_warmup_opens_on_zeros():
    """Without a carried warm-up the kernel opens on zeros, so the loop's first window differs from the stitched one."""
    tr_steps = int(round(TR_MS / DT))
    piece = _window(0, tr_steps, seed=0)
    monitor = _bold_monitor(KERNEL_MS, 12 * TR_MS)

    assert monitor._warmup is None
    cold = np.asarray(monitor(piece).data)
    warm = np.asarray(monitor.open_warmup(N_NODES).carry_warmup(_window(0, 4 * tr_steps, seed=7))(piece).data)

    assert cold.shape == warm.shape == (1, 1, N_NODES)
    assert float(np.max(np.abs(cold - warm))) > 1e-6, "the carried warm-up left no trace on the report"
