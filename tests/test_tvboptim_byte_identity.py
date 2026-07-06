"""Byte-identity interop: tvbo-native (YAML + exp.run) vs tvboptim-native (hand-written).

Each workflow is run two ways and asserted byte-identical. The comparison runs
eager (``jax.disable_jit``) so it tests codegen faithfulness — that tvbo emits the
same operations as a hand-written tvboptim workflow — without XLA's FMA-fusion
nondeterminism (which under JIT differs by ~0.5 ULP and is amplified by stiff
transients). Durations are tiny so eager stays fast.
"""

import pytest

pytest.importorskip("tvboptim")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from tvbo import SimulationExperiment, database_path

EXPERIMENTS_DIR = database_path / "experiments"

# Eager comparison is exact for the trajectory; observations that go through an
# FFT/convolution accumulate at machine precision, so allow a tight float-eps band.
ATOL = 1e-12


@pytest.fixture
def eager():
    """Run the test body with JIT disabled (isolates codegen from XLA fusion)."""
    with jax.disable_jit():
        yield


def assert_identical(name, a, b, atol=ATOL):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    d = float(np.abs(a - b).max())
    assert d <= atol, f"{name}: max|Δ|={d:.3e} exceeds {atol:.0e}"


def _load_sim(name, t1, transient):
    """Load an experiment, shrink durations, drop target-only observations."""
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / f"{name}.yaml"))
    exp.integration.duration = t1
    exp.integration.transient_time = transient
    # Target/empirical observations are only needed for loss (optimization).
    # Drop any that source network observational data so simulation mode runs
    # without a companion BIDS measure — then transitively drop observations that
    # depend on a dropped one.
    if exp.observations:
        state_vars = set(exp.dynamics.state_variables or {})

        def _unsatisfiable(o):
            for s in (str(s) for s in (getattr(o, "source", None) or [])):
                if "network.observations" in s:
                    return True
                # bare name pointing at an observation that has been dropped
                if "." not in s and s not in exp.observations and s not in state_vars:
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for n, o in list(exp.observations.items()):
                if _unsatisfiable(o):
                    del exp.observations[n]
                    changed = True
    exp.configure()
    return exp


# =============================================================================
# RWW — Reduced Wong-Wang, BOLD/FC
# =============================================================================
def test_rww_trajectory_and_bold(eager):
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
    from tvboptim.experimental.network_dynamics.coupling import FastLinearCoupling
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.observations.tvb_monitors.bold import HRFBold

    T1, TRANSIENT, DT = 2000.0, 8000.0, 4.0

    # --- tvbo-native ---
    exp = _load_sim("RWW_BOLD_FC_Optimization", T1, TRANSIENT)
    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    bold_tvbo = np.asarray(r.observations.bold.data)

    # --- tvboptim-native (hand-written, matching the YAML's parameters) ---
    net = Network(
        dynamics=ReducedWongWang(w=0.5, I_o=0.32, INITIAL_STATE=(0.3,)),
        coupling={"instant": FastLinearCoupling(local_states=["S"], G=0.5)},
        graph=DenseGraph(W, region_labels=labels),
        noise=AdditiveNoise(sigma=0.00283, apply_to=["S"], key=jax.random.key(0)),
    )
    mi, si = prepare(net, Heun(), t0=0.0, t1=TRANSIENT, dt=DT)
    ri = mi(si)
    net.update_history(ri)
    mm, sm = prepare(net, Heun(), t0=0.0, t1=T1, dt=DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)
    bold_ref = np.asarray(HRFBold(period=1000.0, downsample_period=4.0, voi=0, history=ri)(rm).data)

    assert_identical("RWW trajectory", sim_tvbo[:, 0:1, :], sim_ref[:, 0:1, :])
    assert_identical("RWW bold", bold_tvbo, bold_ref)


# =============================================================================
# JR — Jansen-Rit, delayed sigmoidal coupling
# =============================================================================
def test_jr_trajectory(eager):
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.coupling import DelayedSigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

    T1, TRANSIENT, DT = 300.0, 600.0, 1.0

    # --- tvbo-native ---
    exp = _load_sim("JR_MEG_FrequencyGradient_Optimization", T1, TRANSIENT)
    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    delays = np.asarray(exp.network.calculate_delays())
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    psd_tvbo = np.asarray(r.observations.simulated_psd.psd)

    # --- tvboptim-native (YAML overrides a, b, mu; integration noise on all states, seed 42) ---
    states = ["y0", "y1", "y2", "y3", "y4", "y5"]
    net = Network(
        dynamics=JansenRit(a=0.065, b=0.065, mu=0.15),
        coupling={"delayed": DelayedSigmoidalJansenRit(incoming_states=["y1", "y2"], G=15.0)},
        graph=DenseDelayGraph(W, delays, region_labels=labels),
        noise=AdditiveNoise(sigma=0.0001, apply_to=states, key=jax.random.key(42)),
    )
    mi, si = prepare(net, Heun(), t0=0.0, t1=TRANSIENT, dt=DT)
    ri = mi(si)
    net.update_history(ri)
    mm, sm = prepare(net, Heun(), t0=0.0, t1=T1, dt=DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("JR trajectory", sim_tvbo[:, :n, :], sim_ref[:, :n, :])

    # PSD observation: subsample y0 by 10 (-> 100 Hz), Welch
    from jax.scipy.signal import welch

    sub = sim_ref[::10, 0, :]
    _, psd_ref = welch(sub.T, fs=100.0, nperseg=256)
    pt = psd_tvbo[:, 0, :] if psd_tvbo.ndim == 3 else psd_tvbo  # drop singleton voi axis
    assert_identical("JR psd", pt, psd_ref)


# =============================================================================
# Exploration / optimization — smoke tests
#
# The per-point observable (sim -> bold -> fc -> rmse) is proven byte-identical
# eager by the trajectory/observation tests above; the sweep and the optimizer
# are tvboptim's own Space/GridAxis/ParallelExecution/OptaxOptimizer. Full-sweep
# byte-identity can't be asserted directly: eager pmap (under jax.disable_jit) is
# pathologically slow, and JITed sweeps drift ~1e-2 for stiff grid points via XLA
# FMA fusion. So these confirm the generated exploration/optimization code runs
# and produces finite, correctly-shaped results.
# =============================================================================
def _load_landscape(t1, transient):
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "RWW_BOLD_FC_Optimization.yaml"))
    exp.integration.duration = t1
    exp.integration.transient_time = transient
    for axis in exp.explorations["parameter_landscape"].space:
        axis.domain.n = 2
    # a short run has too few BOLD TRs for skip_t=20; keep FC non-degenerate
    # (arguments are keyed by name).
    for step in exp.observations["fc"].pipeline:
        args = getattr(step, "arguments", None) or {}
        if "skip_t" in args:
            args["skip_t"].value = 0
    return exp


def test_rww_exploration_runs():
    exp = _load_landscape(2000.0, 500.0)
    exp.configure()
    r = exp.run("tvboptim", mode="exploration", n_w=2, n_G=2, n_pmap=1)
    grid = np.asarray(r.explorations.parameter_landscape.results)
    assert grid.size == 4, f"expected 2x2 grid, got {grid.shape}"
    assert np.all(np.isfinite(grid)), "exploration produced non-finite values"


def test_rww_optimization_runs():
    exp = _load_landscape(2000.0, 500.0)
    for opt in (exp.optimizations or {}).values():
        for stage in (getattr(opt, "stages", None) or []):
            if getattr(stage, "max_iterations", None):
                stage.max_iterations = 2
    exp.configure()
    r = exp.run("tvboptim", mode="optimization")
    assert getattr(r, "optimizations", None) is not None, "no optimization results"


# =============================================================================
# TBPTT — truncated backprop-through-time via the backend-neutral
# integration.differentiation config -> tvboptim Heun(grad_horizon=...).
# The forward pass is invariant to grad_horizon (tvboptim-tested), so we assert
# both that the generated code carries the truncation and that the forward
# trajectory is byte-identical to a hand-written Heun(grad_horizon=W) reference.
# =============================================================================
def test_tbptt_differentiation(eager):
    from tvbo.datamodel.schema import Differentiation
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.coupling import DelayedSigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

    T1, TRANSIENT, DT = 300.0, 600.0, 1.0
    WINDOW_MS = 100.0
    W = int(round(WINDOW_MS / DT))  # grad_horizon in steps

    exp = _load_sim("JR_MEG_FrequencyGradient_Optimization", T1, TRANSIENT)
    exp.integration.differentiation = Differentiation(truncation_window=WINDOW_MS, mode="reverse")
    exp.configure()

    # the backend-neutral config must lower to the native truncated-BPTT solver
    code = exp.render_code("tvboptim")
    assert f"grad_horizon={W}" in code, "differentiation did not lower to grad_horizon"

    Wm = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    delays = np.asarray(exp.network.calculate_delays())
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)

    states = ["y0", "y1", "y2", "y3", "y4", "y5"]
    net = Network(
        dynamics=JansenRit(a=0.065, b=0.065, mu=0.15),
        coupling={"delayed": DelayedSigmoidalJansenRit(incoming_states=["y1", "y2"], G=15.0)},
        graph=DenseDelayGraph(Wm, delays, region_labels=labels),
        noise=AdditiveNoise(sigma=0.0001, apply_to=states, key=jax.random.key(42)),
    )
    mi, si = prepare(net, Heun(grad_horizon=W), t0=0.0, t1=TRANSIENT, dt=DT)
    ri = mi(si)
    net.update_history(ri)
    mm, sm = prepare(net, Heun(grad_horizon=W), t0=0.0, t1=T1, dt=DT)
    sim_ref = np.asarray(mm(sm).data)

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("TBPTT forward (grad_horizon)", sim_tvbo[:, :n, :], sim_ref[:, :n, :])


# =============================================================================
# TBPTT analysis observations (Lyapunov, AD gradient) — the run_motivation
# diagnostics declared as tvbo `analysis` metadata.
#
# These run under JIT, not the `eager` fixture: pure-eager reverse-mode / JVP over
# the integration scan dispatches millions of primitives per step and is
# intractable. Under JIT the generated analysis code and the hand-written
# reference still agree to float precision, because both build the analysis solver
# with the identical plain Heun (the truncation window is an optimization knob, not
# part of the diagnostic — it does not touch the forward pass).
# =============================================================================
def _tbptt_reference_network(weights, labels, transient, dt, G=8.0):
    """Warmed-up Jansen-Rit reference network matching the TBPTT YAML."""
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.coupling import SigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

    net = Network(
        dynamics=JansenRit(a=0.1, b=0.05, mu=0.08),
        coupling={"instant": SigmoidalJansenRit(incoming_states=("y1", "y2"), G=G)},
        graph=DenseGraph(weights, region_labels=labels),
        noise=AdditiveNoise(sigma=0.0316, apply_to="y4", key=jax.random.key(0)),
    )
    warm_solve, warm_cfg = prepare(net, Heun(), t1=transient, dt=dt)
    warm_res = warm_solve(warm_cfg)
    net.update_history(warm_res)
    return net, warm_res


def _only_observation(exp, keep):
    """Drop every observation except ``keep`` (isolates one analysis diagnostic)."""
    for name in list(exp.observations):
        if name != keep:
            del exp.observations[name]


def test_tbptt_lyapunov():
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.analysis.lyapunov import _lyapunov_spectrum_jvp

    T1, TRANSIENT, DT, SEG = 300.0, 300.0, 1.0, 200.0

    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
    exp.integration.duration = T1
    exp.integration.transient_time = TRANSIENT
    exp.observations["lyapunov"].analysis.parameters["segment_time"].value = SEG
    _only_observation(exp, "lyapunov")
    exp.configure()

    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    r = exp.run("tvboptim", mode="simulation")
    lyap_tvbo = np.asarray(r.observations.lyapunov)

    net, _ = _tbptt_reference_network(W, labels, TRANSIENT, DT)
    le_solve, le_cfg = prepare(net, Heun(), t0=0.0, t1=SEG, dt=DT)
    lyap_ref = np.asarray(_lyapunov_spectrum_jvp(le_solve, le_cfg, t=SEG, n=5, k=1))

    assert_identical("TBPTT lyapunov", lyap_tvbo, lyap_ref)


def test_tbptt_ad_gradient():
    import equinox as eqx
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse
    from tvboptim.observations.tvb_monitors.bold import HRFBold

    # Long enough for a non-degenerate FC (>= a few BOLD samples after skip); the
    # workflow's skip_t=25 is scaled down like duration for a fast interop check.
    T1, TRANSIENT, DT, SKIP = 7200.0, 300.0, 1.0, 2

    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
    exp.integration.duration = T1
    exp.integration.transient_time = TRANSIENT
    exp.observations["fc"].pipeline[0].arguments["skip_t"].value = SKIP  # shrink BOLD skip
    del exp.observations["lyapunov"]  # isolate the gradient diagnostic
    exp.configure()

    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    r = exp.run("tvboptim", mode="simulation")
    grad_tvbo = np.asarray(r.observations.ad_gradient)
    loss_tvbo = np.asarray(r.observations.loss)
    fc_target = jnp.asarray(np.asarray(r.observations.empirical_fc))

    # tvboptim-native reference: run_motivation's AD leg (full, untruncated grad).
    net, warm_res = _tbptt_reference_network(W, labels, TRANSIENT, DT)
    solve_fc, cfg = prepare(net, Heun(), t1=T1, dt=DT)
    bold = HRFBold(period=720.0, voi=0, history=warm_res)
    G_lens = lambda c: c.coupling["instant"].G  # noqa: E731

    def fc_rmse_loss(cfg_):
        return rmse(compute_fc(bold(solve_fc(cfg_)), skip_t=SKIP), fc_target)

    loss_ref, grad_ref = jax.value_and_grad(lambda g: fc_rmse_loss(eqx.tree_at(G_lens, cfg, g)))(8.0)

    assert_identical("TBPTT AD loss", loss_tvbo, np.asarray(loss_ref), atol=1e-10)
    assert_identical("TBPTT AD gradient", grad_tvbo, np.asarray(grad_ref), atol=1e-10)


def test_tbptt_fd_gradient():
    import equinox as eqx
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse
    from tvboptim.observations.tvb_monitors.bold import HRFBold

    T1, TRANSIENT, DT, SKIP = 7200.0, 300.0, 1.0, 2
    DELTA, SEEDS, SEED_BASE, G0 = 0.3, 8, 0, 8.0

    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
    exp.integration.duration = T1
    exp.integration.transient_time = TRANSIENT
    exp.observations["fc"].pipeline[0].arguments["skip_t"].value = SKIP
    del exp.observations["lyapunov"]
    del exp.observations["ad_gradient"]
    exp.configure()

    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    r = exp.run("tvboptim", mode="simulation")
    fd_tvbo = np.asarray(r.observations.fd_gradient)
    fc_target = jnp.asarray(np.asarray(r.observations.empirical_fc))

    # tvboptim-native reference: seed-averaged central difference, common random numbers.
    net, warm_res = _tbptt_reference_network(W, labels, TRANSIENT, DT)
    solve_fc, cfg = prepare(net, Heun(), t1=T1, dt=DT)
    bold = HRFBold(period=720.0, voi=0, history=warm_res)
    G_lens = lambda c: c.coupling["instant"].G  # noqa: E731

    def fc_rmse_loss(cfg_):
        return rmse(compute_fc(bold(solve_fc(cfg_)), skip_t=SKIP), fc_target)

    def central_fd(key):
        c = eqx.tree_at(lambda c: c.noise.key, cfg, key)
        fp = fc_rmse_loss(eqx.tree_at(G_lens, c, G0 + DELTA))
        fm = fc_rmse_loss(eqx.tree_at(G_lens, c, G0 - DELTA))
        return (fp - fm) / (2.0 * DELTA)

    fd_ref = jnp.mean(jax.lax.map(central_fd, jax.random.split(jax.random.key(SEED_BASE), SEEDS)))

    assert_identical("TBPTT FD gradient", fd_tvbo, np.asarray(fd_ref), atol=1e-10)


def test_tbptt_optimization():
    """Fit G to empirical FC, both descent legs, tvbo vs hand-written run_optim.

    `integration.differentiation` drives the descent: reverse + truncation_window ->
    `Heun(grad_horizon=W)` + optimizer mode 'rev' (TBPTT); forward + no window ->
    plain `Heun()` + mode 'fwd' (exact untruncated full AD). Runs under JIT.
    """
    import optax
    import jax.numpy as jnp
    from tvbo.datamodel.schema import Differentiation
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse
    from tvboptim.observations.tvb_monitors.bold import HRFBold
    from tvboptim.optim.optax import OptaxOptimizer
    from tvboptim.types import Parameter

    T1, TRANSIENT, DT, SKIP = 7200.0, 300.0, 1.0, 2
    LR, MAX_STEPS, G0, W = 1.0, 4, 8.0, 100

    def _leg(diff, ref_solver, mode):
        exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
        exp.integration.duration = T1
        exp.integration.transient_time = TRANSIENT
        exp.integration.differentiation = diff
        exp.observations["fc"].pipeline[0].arguments["skip_t"].value = SKIP
        exp.optimizations["fit_G"].stages[0].max_iterations = MAX_STEPS
        for k in ("lyapunov", "ad_gradient", "fd_gradient"):
            del exp.observations[k]
        exp.configure()

        Wm = np.asarray(exp.network.weights)
        labels = [n.label for n in exp.network.nodes]
        r = exp.run("tvboptim", mode="optimization")
        g_tvbo = np.asarray(r.optimizations.fit_G.state.coupling.c_sigmJR.G)
        fc_target = jnp.asarray(np.asarray(r.optimizations.fit_G.simulation.observations.empirical_fc))

        # Warmup forward pass is solver-independent (grad_horizon only changes the
        # backward pass), so the plain-Heun reference warmup reproduces the history.
        net, warm_res = _tbptt_reference_network(Wm, labels, TRANSIENT, DT)
        solve, cfg = prepare(net, ref_solver, t1=T1, dt=DT)
        bold = HRFBold(period=720.0, voi=0, history=warm_res)

        def loss(state):
            return rmse(compute_fc(bold(solve(state)), skip_t=SKIP), fc_target)

        cfg.coupling["instant"].G = Parameter(jnp.asarray(G0))
        opt = OptaxOptimizer(loss, optax.adam(LR), has_aux=False)
        fitted, _ = opt.run(cfg, max_steps=MAX_STEPS, mode=mode)
        return g_tvbo, np.asarray(fitted.coupling["instant"].G)

    g_t, g_r = _leg(Differentiation(truncation_window=100.0, mode="reverse"), Heun(grad_horizon=W), "rev")
    assert_identical("TBPTT optim (rev, grad_horizon=TBPTT)", g_t, g_r, atol=1e-9)

    g_t, g_r = _leg(Differentiation(mode="forward"), Heun(), "fwd")
    assert_identical("TBPTT optim (fwd, full AD)", g_t, g_r, atol=1e-9)


def test_tbptt_gsweep_runs():
    """G-sweep exploration smoke test. The per-point observable (FC-RMSE via
    compute_all_observations) is byte-identical to the diagnostics; the full-sweep
    grid can't be asserted byte-identical (eager pmap times out, JIT drifts per
    stiff grid point), so this just checks the sweep runs and stays finite."""
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
    exp.integration.duration = 7200.0
    exp.integration.transient_time = 300.0
    exp.observations["fc"].pipeline[0].arguments["skip_t"].value = 2
    exp.configure()
    r = exp.run("tvboptim", mode="exploration", n_G=3, n_pmap=1)
    grid = np.asarray(r.explorations.G_sweep.results)
    assert grid.size == 3, f"expected 3-point G sweep, got {grid.shape}"
    assert np.all(np.isfinite(grid)), "G sweep produced non-finite values"


def test_tbptt_motivation_sweep_records_diagnostics():
    """`record:` exploration that sweeps the diagnostics themselves (loss + AD/FD
    gradient + Lyapunov) per G — the declarative run_motivation probe. Smoke test:
    each recorded diagnostic comes back as a finite length-n_G array, and the two
    signature phenomena hold (the exact AD gradient blows up somewhere the FD ground
    truth stays bounded; the Lyapunov exponent crosses zero into chaos)."""
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "TBPTT_JansenRit_FC_Optimization.yaml"))
    exp.integration.duration = 3600.0
    exp.integration.transient_time = 300.0
    exp.observations["fc"].pipeline[0].arguments["skip_t"].value = 2
    exp.configure()
    r = exp.run("tvboptim", mode="exploration", n_G=5, n_pmap=1)
    obs = r.explorations.motivation_sweep.observations
    diag = {k: np.asarray(obs[k]).ravel() for k in ("loss", "ad_gradient", "fd_gradient", "lyapunov")}
    for k, v in diag.items():
        assert v.shape == (5,), f"{k}: expected 5-point sweep, got {v.shape}"
        assert np.all(np.isfinite(v)), f"{k}: non-finite over the sweep"
    # AD gradient blows up in the chaotic band; the FD ground truth stays bounded.
    assert np.max(np.abs(diag["ad_gradient"])) > 100 * np.max(np.abs(diag["fd_gradient"]))
    # Lyapunov exponent spans the chaos onset (negative at low G, positive in chaos).
    assert diag["lyapunov"].min() < 0 < diag["lyapunov"].max()


# =============================================================================
# EI — custom two-population Reduced Wong-Wang with dual-output coupling
# =============================================================================
def test_ei_trajectory(eager):
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
    from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
    from tvboptim.experimental.network_dynamics.core.bunch import Bunch
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.observations.tvb_monitors.bold import HRFBold

    T1, TRANSIENT, DT = 720.0, 720.0, 4.0

    # --- tvbo-native ---
    exp = _load_sim("EI_Tuning_FIC_EIB_Optimization", T1, TRANSIENT)
    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    n_nodes = W.shape[0]
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    bold_tvbo = np.asarray(r.observations.bold.data)

    # --- tvboptim-native: hand-written custom two-population RWW-EIB ---
    class RWWeib(AbstractDynamics):
        STATE_NAMES = ("S_e", "S_i")
        INITIAL_STATE = (0.001, 0.001)
        AUXILIARY_NAMES = ("H_e", "H_i")
        DEFAULT_PARAMS = Bunch(
            a_e=310.0, b_e=125.0, d_e=0.160, gamma_e=0.641 / 1000, tau_e=100.0, w_p=1.4, W_e=1.0,
            a_i=615.0, b_i=177.0, d_i=0.087, gamma_i=1.0 / 1000, tau_i=10.0, W_i=0.7,
            J_N=0.15, J_i=1.0, I_o=0.382, I_ext=0.0, lamda=1.0,
        )
        COUPLING_INPUTS = {"coupling": 2}

        def dynamics(self, t, state, params, coupling, external):
            S_e, S_i = state[0], state[1]
            c_lre = params.J_N * coupling.coupling[0]
            c_ffi = params.J_N * coupling.coupling[1]
            J_N_S_e = params.J_N * S_e
            x_e_pre = params.w_p * J_N_S_e - params.J_i * S_i + params.W_e * params.I_o + c_lre + params.I_ext
            x_i_pre = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * c_ffi
            x_e = params.a_e * x_e_pre - params.b_e
            x_i = params.a_i * x_i_pre - params.b_i
            H_e = x_e / (1.0 - jnp.exp(-params.d_e * x_e))
            H_i = x_i / (1.0 - jnp.exp(-params.d_i * x_i))
            dS_e = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e
            dS_i = -(S_i / params.tau_i) + H_i * params.gamma_i
            return jnp.array([dS_e, dS_i]), jnp.array([H_e, H_i])

    class EIBcoup(InstantaneousCoupling):
        N_OUTPUT_STATES = 2
        DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)

        def __init__(self, **kwargs):
            super().__init__(incoming_states=["S_e"], **kwargs)

        def pre(self, incoming_states, local_states, params):
            S_e = incoming_states[0]
            return jnp.stack([S_e * params.wLRE, S_e * params.wFFI], axis=0)

        def post(self, summed_inputs, local_states, params):
            return summed_inputs

    ones_mat = jnp.full((n_nodes, n_nodes), 1.0)
    net = Network(
        dynamics=RWWeib(J_i=jnp.full((n_nodes,), 1.0)),
        coupling={"coupling": EIBcoup(wLRE=ones_mat, wFFI=ones_mat)},
        graph=DenseGraph(W, region_labels=labels),
        noise=AdditiveNoise(sigma=0.01, apply_to=["S_e"], key=jax.random.key(0)),
    )
    solver = BoundedSolver(Heun(), low=jnp.array([0.0, 0.0])[:, None], high=jnp.array([1.0, 1.0])[:, None])
    mi, si = prepare(net, solver, t0=0.0, t1=TRANSIENT, dt=DT)
    ri = mi(si)
    net.update_history(ri)
    mm, sm = prepare(net, solver, t0=0.0, t1=T1, dt=DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)
    bold_ref = np.asarray(HRFBold(period=720.0, downsample_period=4.0, voi=0, history=ri)(rm).data)

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("EI trajectory", sim_tvbo[:, :n, :], sim_ref[:, :n, :])
    assert_identical("EI bold", bold_tvbo, bold_ref)


# =============================================================================
# Bayesian inference — Stimulation_with_Bayesian_Inference (numpyro NUTS).
# The forward model is byte-identical to a hand-built tvboptim reference; since the
# observed data, likelihood, priors, and NUTS setup all match, the posterior is too.
# =============================================================================
def _bayesian_reference_forward():
    """tvboptim-native forward at the true (declared) params -> recorded_ts."""
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
    from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.external_input import PulseInput

    net = Network(
        dynamics=Generic2dOscillator(a=-1.5, b=-15.0, c=0.0, d=0.015, e=3.0, f=1.0,
                                      tau=4.0, I=0.1, VARIABLES_OF_INTEREST=("V",)),
        coupling={"instant": LinearCoupling(incoming_states="V", G=0.0)},
        graph=DenseGraph(jnp.zeros((1, 1))),
        external_input={"stimulus": PulseInput(onset=10.0, duration=1.0, amplitude=0.4)},
    )
    sf, cfg = prepare(net, Heun(), t0=0.0, t1=150.0, dt=0.2)
    return np.asarray(sf(cfg).ys[::15, 0, 0]).ravel()


def test_bayesian_forward_byte_identity(eager):
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Stimulation_Bayesian_Inference.yaml"))
    exp.configure()
    rec_tvbo = np.asarray(exp.run("tvboptim", mode="simulation").observations.recorded_ts.data).ravel()
    rec_ref = _bayesian_reference_forward()
    n = min(len(rec_tvbo), len(rec_ref))
    assert_identical("Bayesian forward recorded_ts", rec_tvbo[:n], rec_ref[:n])


def test_bayesian_inference_recovers_and_steers():
    """MCMC smoke (reduced samples): both params recovered near truth, the amplitude/I
    degeneracy ridge (negative correlation), and prior-steering across scenarios."""
    import jax
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Stimulation_Bayesian_Inference.yaml"))
    for inf in exp.inferences.values():
        inf.num_warmup, inf.num_samples = 200, 400
    exp.configure()
    rec = np.asarray(exp.run("tvboptim", mode="simulation").observations.recorded_ts.data).ravel()
    observed = rec + 0.1 * np.asarray(jax.random.normal(jax.random.key(42), (rec.shape[0],)))
    r = exp.run("tvboptim", mode="inference", recorded_ts=observed)

    post = {k: (np.asarray(r.inferences[k].posterior["stimulus.amplitude"]),
                np.asarray(r.inferences[k].posterior["Generic2dOscillator.I"])) for k in r.inferences}
    a_A, i_A = post["scenario_A"]
    assert abs(a_A.mean() - 0.4) < 0.15, f"amplitude not recovered: {a_A.mean():.3f}"
    assert abs(i_A.mean() - 0.1) < 0.15, f"I not recovered: {i_A.mean():.3f}"
    assert np.corrcoef(a_A, i_A)[0, 1] < -0.5, "expected amplitude/I degeneracy ridge"
    # prior-steering: the tight-excitability prior (C) clearly narrows the I posterior
    # vs the wide prior (A) — the robust steering signal at reduced sample counts.
    assert post["scenario_C"][1].std() < post["scenario_A"][1].std(), "prior-steering (I) not observed"


# =============================================================================
# Hopf Pareto — SupHopf, NSGA-II multi-objective (FC vs frequency gradient)
# + per-Pareto-seed parallel gradient refinement. Replicates tvboptim
# Hopf_Pareto_ParallelOpt. The GA/refine run under ParallelExecution (pmap), so
# those are asserted under JIT (eager pmap is intractable); the trajectory and
# frequency observations are eager like the other workflows.
# =============================================================================
import copy as _copy


def _load_hopf(t1, num_gen=2, pop=4, refine_iter=2):
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Hopf_Pareto_ParallelOpt.yaml"))
    exp.integration.duration = float(t1)
    exp.integration.transient_time = float(t1)
    _ga_params = exp.explorations["ga_presearch"].parameters
    _ga_params["num_generations"].value = num_gen
    _ga_params["population_size"].value = pop
    exp.optimizations["refine"].max_iterations = refine_iter
    exp.configure()
    return exp


def _hopf_ref(exp, t1):
    """Hand-written tvboptim reference matching the Hopf YAML (shared warm-up)."""
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.dynamics.tvb import SupHopf
    from tvboptim.experimental.network_dynamics.coupling import FastLinearCoupling
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    import jax.numpy as jnp

    W = np.asarray(exp.network.weights)
    omega = np.asarray(list(exp.dynamics.parameters["omega"].value))
    net = Network(
        dynamics=SupHopf(a=0.01, omega=jnp.asarray(omega), INITIAL_STATE=(0.1, 0.0)),
        coupling={"instant": FastLinearCoupling(local_states=["x"], G=0.025)},
        graph=DenseGraph(W),
        noise=AdditiveNoise(sigma=0.01, apply_to=["x", "y"], key=jax.random.key(0)),
    )
    mi, si = prepare(net, Heun(), t1=float(t1), dt=1.0)
    ri = mi(si)
    net.update_history(ri)
    mm, sm = prepare(net, Heun(), t1=float(t1), dt=1.0)
    return net, ri, mm, sm, omega


def test_hopf_trajectory_and_frequency(eager):
    exp = _load_hopf(2000.0)
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    psd_tvbo = np.asarray(getattr(r.observations.psd, "psd", getattr(r.observations.psd, "data")))
    peaks_tvbo = np.asarray(getattr(r.observations.fitted_peaks, "data", r.observations.fitted_peaks))

    _net, _ri, mm, sm, _omega = _hopf_ref(exp, 2000.0)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)
    x_sub = sim_ref[::10, 0, :]
    _, psd_ref = jax.scipy.signal.welch(x_sub.T, fs=100.0, nperseg=512, nfft=2048)
    psd_ref = np.asarray(psd_ref)
    f = np.linspace(0, 50.0, psd_ref.shape[1])
    pt, ft = psd_ref[:, 3:], f[3:]
    w = jax.nn.softmax(150.0 * (pt / (np.max(pt, axis=1, keepdims=True) + 1e-12)), axis=1)
    peaks_ref = np.asarray(jax.numpy.sum(w * ft[None, :], axis=1))

    assert_identical("Hopf trajectory", sim_tvbo, sim_ref)
    assert_identical("Hopf psd", np.squeeze(psd_tvbo), np.squeeze(psd_ref))
    assert_identical("Hopf fitted_peaks", np.squeeze(peaks_tvbo), np.squeeze(peaks_ref))


def test_hopf_bold_and_fc():
    from tvboptim.observations.tvb_monitors.bold import HRFBold
    from tvboptim.observations.observation import compute_fc

    T = 30000.0
    exp = _load_hopf(T)
    r = exp.run("tvboptim", mode="simulation")
    bold_tvbo = np.asarray(getattr(r.observations.bold, "data", r.observations.bold))
    fc_tvbo = np.asarray(getattr(r.observations.fc, "data", r.observations.fc))

    _net, ri, mm, sm, _omega = _hopf_ref(exp, T)
    rm = mm(sm)
    bm = HRFBold(period=1000.0, downsample_period=1.0, voi=0, history=ri)(rm)
    assert_identical("Hopf bold", np.squeeze(bold_tvbo), np.squeeze(np.asarray(bm.data)))
    assert_identical("Hopf fc", np.squeeze(fc_tvbo), np.squeeze(np.asarray(compute_fc(bm, skip_t=20))))


def test_hopf_ga_pareto_and_refine():
    """Stage 0 (NSGA-II front) + Stage 1 (per-seed refine) byte-identical to the reference."""
    import jax.numpy as jnp
    import optax
    from tvboptim.observations.tvb_monitors.bold import HRFBold
    from tvboptim.observations.observation import compute_fc, fc_corr
    from tvboptim.types import DataAxis, Parameter, BoundedParameter
    from tvboptim.types.spaces import Space
    from tvboptim.execution import ParallelExecution
    from tvboptim.optim.optax import OptaxOptimizer
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    N, T = 4, 30000.0
    exp = _load_hopf(T, num_gen=2, pop=4, refine_iter=2)
    res = exp.run("tvboptim", mode="all")
    ga = res.explorations.ga_presearch
    refine = res.optimizations.refine
    ga_X, ga_F = np.atleast_2d(np.asarray(ga.pareto_X)), np.atleast_2d(np.asarray(ga.pareto_F))
    emp = np.asarray(getattr(res.observations.empirical_fc, "data", res.observations.empirical_fc))
    pft = np.asarray([p for p in exp.functions["freq_grad_corr_fn"].arguments["target"].value])

    net, ri, mB, base_state = _hopf_ref(exp, T)[0:4]
    base_state = _copy.deepcopy(base_state)
    bold_mon = HRFBold(period=1000.0, downsample_period=1.0, voi=0, history=ri)
    nn = np.asarray(exp.network.weights).shape[0]

    def _metrics(s):
        rr = mB(s)
        x_sub = rr.data[::10, 0, :]
        _, psd = jax.scipy.signal.welch(x_sub.T, fs=100.0, nperseg=512, nfft=2048)
        f = jnp.linspace(0, 50.0, psd.shape[1])
        pt, ft = psd[:, 3:], f[3:]
        w = jax.nn.softmax(150.0 * (pt / (jnp.max(pt, axis=1, keepdims=True) + 1e-12)), axis=1)
        peaks = jnp.sum(w * ft[None, :], axis=1)
        fcv = fc_corr(compute_fc(bold_mon(rr), skip_t=20), jnp.asarray(emp))
        fgc = jnp.corrcoef(peaks, jnp.asarray(pft))[0, 1]
        return jnp.mean(jnp.abs(peaks - 9.0)), 1.0 - fcv, fcv, fgc

    def _ga_eval(s):
        mae, omf, _, _ = _metrics(s)
        return {"psd_mae": mae, "one_minus_fc": omf}

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=3, n_obj=2, xl=np.array([0.001, -0.1, -2.0]), xu=np.array([0.15, 0.01, -0.5]))

        def _evaluate(self, X, out, *a, **k):
            b = _copy.deepcopy(base_state)
            b.coupling.instant.G = DataAxis(jnp.asarray(X[:, 0]))
            b.dynamics.a = DataAxis(jnp.asarray(X[:, 1]))
            b.noise.sigma = DataAxis(jnp.asarray(10.0 ** X[:, 2]))
            rs = list(ParallelExecution(_ga_eval, Space(b, mode="zip"), n_pmap=N).run())
            F = np.array([[float(np.asarray(r["psd_mae"])), float(np.asarray(r["one_minus_fc"]))] for r in rs])
            out["F"] = np.nan_to_num(F, nan=1e6, posinf=1e6, neginf=1e6)

    _res = pymoo_minimize(_P(), NSGA2(pop_size=4), ("n_gen", 2), seed=42, verbose=False)
    assert_identical("Hopf GA pareto_X", ga_X, np.atleast_2d(np.asarray(_res.X)))
    assert_identical("Hopf GA pareto_F", ga_F, np.atleast_2d(np.asarray(_res.F)))

    # Stage 1: refine each Pareto seed (mirrors optimize_from_seed).
    opt = OptaxOptimizer(lambda s: (lambda m: -m[2] - m[3])(_metrics(s)), optax.adam(0.001))

    def _refine_seed(s):
        s.coupling.instant.G = Parameter(jnp.asarray(s.coupling.instant.G))
        s.dynamics.a = Parameter(jnp.asarray(s.dynamics.a) * jnp.ones(nn))
        s.dynamics.omega = BoundedParameter(jnp.asarray(s.dynamics.omega), 0.0, jnp.inf)
        fin, _ = opt.run(s, max_steps=2, chunk_size=2)
        return {"G": jnp.asarray(fin.coupling.instant.G), "a": jnp.asarray(fin.dynamics.a),
                "om": jnp.asarray(fin.dynamics.omega)}

    seed = _copy.deepcopy(base_state)
    seed.coupling.instant.G = DataAxis(jnp.asarray(ga_X[:, 0]))
    seed.dynamics.a = DataAxis(jnp.asarray(ga_X[:, 1]))
    seed.noise.sigma = DataAxis(jnp.asarray(10.0 ** ga_X[:, 2]))
    rows = list(ParallelExecution(_refine_seed, Space(seed, mode="zip"), n_pmap=N).run())
    for tr, rr in zip(refine.seeds, rows):
        assert_identical("Hopf refine G", np.asarray(tr.G_final), np.asarray(rr["G"]), atol=1e-10)
        assert_identical("Hopf refine a", np.asarray(tr.a_final), np.asarray(rr["a"]), atol=1e-10)
        assert_identical("Hopf refine omega", np.asarray(tr.omega_final), np.asarray(rr["om"]), atol=1e-10)


# =============================================================================
# EI_Tuning — the gradient optimization must RUN (not just the trajectory).
# Guards a wrapping crash class the byte-identity trajectory tests don't cover:
# an aggregated observable (mean_activity — time axis collapsed to a per-node
# value) must reach the loss as a plain array, not a re-wrapped NativeSolution
# (`mean_activity + target` would raise "unsupported operand ... 'NativeSolution'").
# Fixed two ways, complementary: the observation template returns raw for a
# collapsed aggregation (producer, root cause), and the loss-builder unwraps
# `.data` (consumer). This test passes on either fix and pins the class.
# =============================================================================
def test_ei_optimization_runs():
    exp = SimulationExperiment.from_file(
        str(EXPERIMENTS_DIR / "EI_Tuning_FIC_EIB_Optimization.yaml")
    )
    exp.integration.duration = 2000.0
    exp.integration.transient_time = 2000.0
    for _o in exp.optimizations.values():
        if hasattr(_o, "max_iterations"):
            _o.max_iterations = 2
        for _st in (getattr(_o, "stages", None) or []):
            _st.max_iterations = 2
    exp.configure()

    # The crash was an exception at loss evaluation, so completing is the guard.
    r = exp.run("tvboptim", mode="optimization")

    # Additionally assert the objective is a finite scalar (not NaN / not an object).
    def _final_loss(res):
        opts = getattr(res, "optimizations", None)
        for _v in (opts.values() if opts else []):
            for _attr in ("loss", "final_loss", "loss_history"):
                _val = getattr(_v, _attr, None)
                if _val is None:
                    continue
                try:
                    arr = np.asarray(_val, dtype=float).ravel()
                except (TypeError, ValueError):
                    continue  # e.g. a history dict — not a numeric loss
                if arr.size:
                    return float(arr[-1])
        return None

    lv = _final_loss(r)
    assert lv is None or np.isfinite(lv), f"EI optimization loss not finite: {lv}"
