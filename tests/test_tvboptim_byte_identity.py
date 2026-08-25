"""Byte-identity interop: tvbo-native (YAML + exp.run) vs tvboptim-native (hand-written).

Each workflow is run two ways and asserted byte-identical. The comparison runs eager (``jax.disable_jit``) so it tests codegen faithfulness — that tvbo emits the same operations as a hand-written tvboptim workflow — without XLA's FMA-fusion nondeterminism (which under JIT differs by ~0.5 ULP and is amplified by stiff transients). Durations are tiny so eager stays fast.

Two families depart from that eager default. Exploration and optimization workflows are smoke-tested rather than compared point by point: the per-point observable (sim -> bold -> fc -> rmse, via ``compute_all_observations``) is already proven identical by the trajectory and observation tests, and the sweep and the optimizer are tvboptim's own Space/GridAxis/ParallelExecution/OptaxOptimizer. A whole sweep cannot be compared directly — eager pmap is pathologically slow and a JITed sweep drifts ~1e-2 at stiff grid points through XLA FMA fusion — so those tests assert only that the generated code runs and returns finite, correctly-shaped results.

The TBPTT analysis diagnostics (Lyapunov, AD gradient) likewise run under JIT, because eager reverse-mode or JVP over the integration scan dispatches millions of primitives per step and is intractable. They still agree to float precision: the generated analysis code and the hand-written reference build the analysis solver with the identical plain Heun, the truncation window being an optimization knob that never touches the forward pass.
"""

import types

import pytest

pytest.importorskip("tvboptim")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from tvbo import SimulationExperiment, database_path
from tvbo.utils import as_list

EXPERIMENTS_DIR = database_path / "experiments"

ATOL = 1e-12
"""Default comparison tolerance: eager comparison is exact for the trajectory, but observations that go through an FFT or a convolution accumulate at machine precision, so a tight float-eps band is allowed."""


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


def _one_scan(net, solver, t1, transient, dt):
    """One scan whose head is the settle — the window the generated code integrates.

    Returns the un-run callable, its config, and the number of steps the settle occupies, so each caller drops it where the observable it is checking drops it.
    """
    from tvboptim.experimental.network_dynamics import prepare

    model_fn, cfg = prepare(net, solver, t0=-transient, t1=t1, dt=dt)
    return model_fn, cfg, int(round(transient / dt))


def _emitted_bold(exp, voi=0):
    """The BOLD observable tvbo emits for this recipe, for the hand-written reference to reuse.

    A tvbo recipe declares BOLD as a pipeline — decimate, convolve the HRF over one kernel support of settle, Volterra, sample at TR — and not as a tvboptim monitor. `HRFBold` is a second convention for the same arithmetic: it decimates its warm-up separately from its data, which differs from decimating the one signal at the seam. Rebuilding the window with it would make these tests assert the distance between two conventions rather than what they exist for — the gradient, the descent and the search around the loss — so they share the observable and hand-write everything else. What the window itself has to be is pinned exactly, on synthetic input, in `tests/test_bold_settle_inband.py`.
    """
    module = types.ModuleType("generated_observables")
    exec(compile(exp.render_code("tvboptim"), "<generated>", "exec"), module.__dict__)
    return module.Bold(voi=voi, dt=exp.integration.step_size)


def _load_sim(name, t1, transient):
    """Load an experiment, shrink durations, drop target-only observations.

    Target and empirical observations are only needed for the optimization loss, so any that source network observational data are dropped and simulation mode then runs without a companion BIDS measure. Observations that depend on a dropped one are dropped transitively.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / f"{name}.yaml"))
    exp.integration.duration = t1
    exp.integration.transient_time = transient
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


# RWW — Reduced Wong-Wang, BOLD/FC
def test_rww_trajectory_and_bold(eager):
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
    from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

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
        coupling={"instant": LinearCoupling(incoming_states=["S"], G=0.5)},
        graph=DenseGraph(W, region_labels=labels),
        noise=AdditiveNoise(sigma=0.00283, apply_to=["S"], key=jax.random.key(0)),
    )
    # S is declared on [0, 1] with clamping, and one scan runs the settle through the same solver.
    solver = BoundedSolver(Heun(), low=jnp.array([0.0])[:, None], high=jnp.array([1.0])[:, None])
    mm, sm, n_settle = _one_scan(net, solver, T1, TRANSIENT, DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)[n_settle:]

    assert_identical("RWW trajectory", sim_tvbo[:, 0:1, :], sim_ref[:, 0:1, :])
    assert_identical("RWW bold", bold_tvbo, np.asarray(_emitted_bold(exp)(rm).data))


def test_jr_trajectory(eager):
    """Jansen-Rit with delayed sigmoidal coupling: trajectory and Welch PSD.

    The hand-written reference mirrors the YAML: it overrides a, b and mu, applies integration noise to every state, and seeds with 0 — the resolved ``execution.random_seed`` default when the YAML sets none.
    """
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.coupling import DelayedSigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.experimental.network_dynamics.solvers import Heun

    T1, TRANSIENT, DT = 300.0, 600.0, 1.0

    # --- tvbo-native ---
    exp = _load_sim("JR_MEG_FrequencyGradient_Optimization", T1, TRANSIENT)
    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    delays = np.asarray(exp.network.calculate_delays())
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    psd_tvbo = np.asarray(r.observations.simulated_psd.psd)

    # --- tvboptim-native ---
    states = ["y0", "y1", "y2", "y3", "y4", "y5"]
    net = Network(
        dynamics=JansenRit(a=0.065, b=0.065, mu=0.15),
        coupling={"delayed": DelayedSigmoidalJansenRit(incoming_states=["y1", "y2"], G=15.0)},
        graph=DenseDelayGraph(W, delays, region_labels=labels),
        noise=AdditiveNoise(sigma=0.0001, apply_to=states, key=jax.random.key(0)),
    )
    mm, sm, n_settle = _one_scan(net, Heun(), T1, TRANSIENT, DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)[n_settle:]

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("JR trajectory", sim_tvbo[:, :n, :], sim_ref[:, :n, :])

    # PSD observation: subsample y0 by 10 (-> 100 Hz), Welch
    from jax.scipy.signal import welch

    sub = sim_ref[::10, 0, :]
    _, psd_ref = welch(sub.T, fs=100.0, nperseg=256)
    pt = psd_tvbo[:, 0, :] if psd_tvbo.ndim == 3 else psd_tvbo  # drop singleton voi axis
    assert_identical("JR psd", pt, psd_ref)


# Exploration / optimization — smoke tests
def _load_landscape(t1, transient):
    """Load the RWW landscape experiment shrunk to a 2x2 grid with a non-degenerate FC.

    A short run has too few BOLD TRs for the workflow's ``skip_t=20``, so every pipeline step that takes a ``skip_t`` argument is reset to 0.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "RWW_BOLD_FC_Optimization.yaml"))
    exp.integration.duration = t1
    exp.integration.transient_time = transient
    for axis in exp.explorations["parameter_landscape"].space.values():
        axis.domain.n = 2
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


_MINI_EXPERIMENT = """
id: 7
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  parameters:
    conduction_speed: {name: conduction_speed, value: 3.0, unit: mm_per_ms}
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5}
    - {source: 1, target: 0, weight: 0.5}
  coupling:
    KuramotoCoupling:
      label: KuramotoCoupling
      parameters:
        a: {value: 0.01}
        N: {value: 1.0}
      pre_expression: {rhs: "sin(theta_j - theta_i)"}
      post_expression: {rhs: "a * gx / N"}
      incoming_states: [theta]
      local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 200.0
  step_size: 1.0
  transient_time: 50.0
explorations:
  sweep:
    name: sweep
    mode: product
    space:
      - parameter: conduction_speed
        explored_values: [2.0, 4.0]
"""
"""A small, self-contained experiment with a connectome and a sweep: a 2-node Kuramoto network, deterministic (RK4, no noise), no empirical observations."""


def test_experiment_result_roundtrips_via_sidecar(tmp_path):
    """A saved ExperimentResult is self-describing and reproducible.

    Runs a full experiment (2-node connectome + sweep), saves the unified ``<stem>.h5`` + ``<stem>.yaml`` sidecar, then reloads *only the sidecar spec* (connectome frozen alongside as an HDF5 companion), reruns, and re-saves. The two result Datasets must be identical and the two YAML sidecars byte-identical — the sidecar loses no config and replays exactly, and the HDF5 opens with a plain ``xarray.open_dataset``. The file name is BIDS-clean (``exp-<id>``).
    """
    import filecmp
    from pathlib import Path

    import xarray as xr

    spec = tmp_path / "mini.yaml"
    spec.write_text(_MINI_EXPERIMENT)

    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    written = exp.run("tvboptim").save(str(tmp_path / "A"))
    h5_a = next(p for p in written if p.endswith(".h5"))
    yaml_a = next(p for p in written if p.endswith(".yaml"))

    # BIDS-clean name: exp-<id> prefix, no spaces.
    name = Path(h5_a).name
    assert name.startswith("exp-") and " " not in name, name
    # Opens with a plain xarray call — no TVBO-specific reader.
    assert xr.open_dataset(h5_a).data_vars, "result HDF5 has no data variables"

    # Reload the sidecar spec alone (connectome frozen alongside), replicate, re-save.
    exp2 = SimulationExperiment.from_file(yaml_a)
    exp2.configure()
    written2 = exp2.run("tvboptim").save(str(tmp_path / "B"))
    h5_b = next(p for p in written2 if p.endswith(".h5"))
    yaml_b = next(p for p in written2 if p.endswith(".yaml"))

    # Same BIDS stem, data identical, provenance sidecar byte-identical.
    assert Path(h5_b).name == name
    xr.testing.assert_identical(xr.open_dataset(h5_a), xr.open_dataset(h5_b))
    assert filecmp.cmp(yaml_a, yaml_b, shallow=False), "sidecar YAML not byte-identical"


_MINI_DELAYED_EXPERIMENT = """
id: 8
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  parameters:
    conduction_speed: {name: conduction_speed, value: 3.0, unit: mm_per_ms}
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - source: 0
      target: 1
      parameters:
        weight: {value: 0.5}
        distance: {value: 30.0}
    - source: 1
      target: 0
      parameters:
        weight: {value: 0.5}
        distance: {value: 30.0}
  coupling:
    KuramotoCoupling:
      label: KuramotoCoupling
      delayed: true
      parameters:
        a: {value: 0.01}
        N: {value: 1.0}
      pre_expression: {rhs: "sin(theta_j - theta_i)"}
      post_expression: {rhs: "a * gx / N"}
      incoming_states: [theta]
      local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 200.0
  step_size: 1.0
  transient_time: 50.0
explorations:
  sweep:
    name: sweep
    mode: product
    space:
      - parameter: network.conduction_speed
        explored_values: [2.0, 4.0]
"""
"""A 2-node delayed Kuramoto network with a ``network.conduction_speed`` exploration axis. Unlike ``_MINI_EXPERIMENT`` (zero tract lengths, hence inert), the edges carry a distance, so delay = distance / conduction_speed is non-zero and the swept speed genuinely changes the dynamics. Exercises the DenseLengthGraph and live ``graph.speed`` codegen path for a network-scope axis."""


def test_network_conduction_speed_axis_sweeps(tmp_path):
    """A `network.conduction_speed` axis actually varies the delays.

    Guards two failure modes the delay-graph codegen must avoid: (1) the axis must be classified network-scope even when given as ``explored_values`` — otherwise it is swept as a non-existent dynamics parameter and every cell comes out identical (a silently-inert sweep); (2) sweeping the ``DenseLengthGraph.speed`` leaf under vmap must match running each speed on its own (no cross-contamination).
    """
    import copy

    import yaml

    spec = yaml.safe_load(_MINI_DELAYED_EXPERIMENT)

    def run(vals):
        d = copy.deepcopy(spec)
        d["explorations"]["sweep"]["space"][0]["explored_values"] = vals
        p = tmp_path / f"delayed_{len(vals)}_{vals[0]}.yaml"
        p.write_text(yaml.safe_dump(d))
        exp = SimulationExperiment.from_file(str(p))
        exp.configure()
        return np.asarray(exp.run("tvboptim").explorations.sweep.results)

    grid = run([2.0, 4.0])
    assert grid.shape[0] == 2, f"expected 2 cells, got {grid.shape}"
    assert np.all(np.isfinite(grid)), "sweep produced non-finite values"
    # Different conduction speeds => different delays => different dynamics.
    assert not np.allclose(grid[0], grid[1]), "conduction_speed sweep is inert (is_network lost?)"
    # Each swept cell equals running that single speed alone.
    assert_identical("speed=2.0 cell", grid[0], run([2.0])[0])
    assert_identical("speed=4.0 cell", grid[1], run([4.0])[0])


def test_rww_optimization_runs():
    exp = _load_landscape(2000.0, 500.0)
    for opt in (exp.optimizations or {}).values():
        for stage in getattr(opt, "stages", None) or []:
            if getattr(stage, "max_iterations", None):
                stage.max_iterations = 2
    exp.configure()
    r = exp.run("tvboptim", mode="optimization")
    assert getattr(r, "optimizations", None) is not None, "no optimization results"


def test_tbptt_differentiation(eager):
    """Truncated backprop-through-time lowers to a native ``Heun(grad_horizon=...)``.

    The backend-neutral ``integration.differentiation`` config must lower to tvboptim's truncated solver. The forward pass is invariant to ``grad_horizon`` (tvboptim-tested), so the test asserts both that the generated code carries the truncation and that the forward trajectory is byte-identical to a hand-written ``Heun(grad_horizon=W)`` reference.
    """
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.coupling import DelayedSigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.experimental.network_dynamics.solvers import Heun

    from tvbo.datamodel.schema import Differentiation

    T1, TRANSIENT, DT = 300.0, 600.0, 1.0
    WINDOW_MS = 100.0
    W = int(round(WINDOW_MS / DT))  # grad_horizon in steps

    exp = _load_sim("JR_MEG_FrequencyGradient_Optimization", T1, TRANSIENT)
    exp.integration.differentiation = Differentiation(truncation_window=WINDOW_MS, mode="reverse")
    exp.configure()

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
        noise=AdditiveNoise(sigma=0.0001, apply_to=states, key=jax.random.key(0)),
    )
    mm, sm, n_settle = _one_scan(net, Heun(grad_horizon=W), T1, TRANSIENT, DT)
    sim_ref = np.asarray(mm(sm).data)[n_settle:]

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("TBPTT forward (grad_horizon)", sim_tvbo[:, :n, :], sim_ref[:, :n, :])


# TBPTT analysis observations (Lyapunov, AD gradient) — the run_motivation diagnostics declared as tvbo `analysis` metadata
def _tbptt_reference_network(weights, labels, G=8.0):
    """Jansen-Rit reference network matching the TBPTT YAML; its settle is the head of the one scan its callers open."""
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.coupling import SigmoidalJansenRit
    from tvboptim.experimental.network_dynamics.dynamics.tvb import JansenRit
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

    net = Network(
        dynamics=JansenRit(a=0.1, b=0.05, mu=0.08),
        coupling={"instant": SigmoidalJansenRit(incoming_states=("y1", "y2"), G=G)},
        graph=DenseGraph(weights, region_labels=labels),
        noise=AdditiveNoise(sigma=0.0316, apply_to="y4", key=jax.random.key(0)),
    )
    return net


def _only_observation(exp, keep):
    """Drop every observation except ``keep`` (isolates one analysis diagnostic)."""
    for name in list(exp.observations):
        if name != keep:
            del exp.observations[name]


def test_tbptt_lyapunov():
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.analysis.lyapunov import _lyapunov_spectrum_jvp
    from tvboptim.experimental.network_dynamics.solvers import Heun

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

    net = _tbptt_reference_network(W, labels)
    le_solve, le_cfg = prepare(net, Heun(), t0=0.0, t1=SEG, dt=DT)
    lyap_ref = np.asarray(_lyapunov_spectrum_jvp(le_solve, le_cfg, t=SEG, n=5, k=1))

    assert_identical("TBPTT lyapunov", lyap_tvbo, lyap_ref)


def test_tbptt_ad_gradient():
    import equinox as eqx
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse

    # Long enough for a non-degenerate FC (>= a few BOLD samples after skip); the workflow's skip_t=25 is scaled down like duration for a fast interop check.
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
    net = _tbptt_reference_network(W, labels)
    solve_fc, cfg, _ = _one_scan(net, Heun(), T1, TRANSIENT, DT)
    bold = _emitted_bold(exp)
    G_lens = lambda c: c.coupling["instant"].G  # noqa: E731

    def fc_rmse_loss(cfg_):
        return rmse(compute_fc(bold(solve_fc(cfg_)), skip_t=SKIP), fc_target)

    loss_ref, grad_ref = jax.value_and_grad(lambda g: fc_rmse_loss(eqx.tree_at(G_lens, cfg, g)))(8.0)

    assert_identical("TBPTT AD loss", loss_tvbo, np.asarray(loss_ref), atol=1e-10)
    assert_identical("TBPTT AD gradient", grad_tvbo, np.asarray(grad_ref), atol=1e-10)


def test_tbptt_fd_gradient():
    import equinox as eqx
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse

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
    net = _tbptt_reference_network(W, labels)
    solve_fc, cfg, _ = _one_scan(net, Heun(), T1, TRANSIENT, DT)
    bold = _emitted_bold(exp)
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

    `integration.differentiation` drives the descent: reverse + truncation_window -> `Heun(grad_horizon=W)` + optimizer mode 'rev' (TBPTT); forward + no window -> plain `Heun()` + mode 'fwd' (exact untruncated full AD). Runs under JIT.

    ``G0`` is where the descent begins: it must match the YAML ``fit_G`` free-parameter ``initial_value`` (5.0), not the base or warm-up coupling G (8.0, used only to build the reference history in ``_tbptt_reference_network``).
    """
    import jax.numpy as jnp
    import optax
    from tvboptim.experimental.network_dynamics.solvers import Heun
    from tvboptim.observations.observation import compute_fc, rmse
    from tvboptim.optim.optax import OptaxOptimizer
    from tvboptim.types import Parameter

    from tvbo.datamodel.schema import Differentiation

    T1, TRANSIENT, DT, SKIP = 7200.0, 300.0, 1.0, 2
    LR, MAX_STEPS, G0, W = 1.0, 4, 5.0, 100

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

        net = _tbptt_reference_network(Wm, labels)
        solve, cfg, _ = _one_scan(net, ref_solver, T1, TRANSIENT, DT)
        bold = _emitted_bold(exp)

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
    """G-sweep exploration smoke test: the sweep runs and every grid point stays finite."""
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
    """`record:` exploration that sweeps the diagnostics themselves (loss + AD/FD gradient + Lyapunov) per G — the declarative run_motivation probe.

    Smoke test: each recorded diagnostic comes back as a finite length-n_G array, and the two signature phenomena hold (the exact AD gradient blows up somewhere the FD ground truth stays bounded; the Lyapunov exponent crosses zero into chaos).
    """
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


# EI — custom two-population Reduced Wong-Wang with dual-output coupling
def test_ei_trajectory(eager):
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.core.bunch import Bunch
    from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
    from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

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
            a_e=310.0,
            b_e=125.0,
            d_e=0.160,
            gamma_e=0.641 / 1000,
            tau_e=100.0,
            w_p=1.4,
            W_e=1.0,
            a_i=615.0,
            b_i=177.0,
            d_i=0.087,
            gamma_i=1.0 / 1000,
            tau_i=10.0,
            W_i=0.7,
            J_N=0.15,
            J_i=1.0,
            I_o=0.382,
            I_ext=0.0,
            lamda=1.0,
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
        EDGE_PARAMS = ("wLRE", "wFFI")
        """wLRE and wFFI are per-edge (n_nodes, n_nodes) weight matrices; the tvboptim 0.4.0 contract requires graph-shaped params to be declared so the base class aligns them to the message axes before ``pre()``."""

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
    mm, sm, n_settle = _one_scan(net, solver, T1, TRANSIENT, DT)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)[n_settle:]

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("EI trajectory", sim_tvbo[:, :n, :], sim_ref[:, :n, :])
    assert_identical("EI bold", bold_tvbo, np.asarray(_emitted_bold(exp)(rm).data))


# Bayesian inference — Stimulation_Bayesian_Inference (numpyro NUTS)
def _bayesian_reference_forward():
    """tvboptim-native forward at the true (declared) params -> recorded_ts."""
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
    from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
    from tvboptim.experimental.network_dynamics.external_input import PulseInput
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun

    net = Network(
        dynamics=Generic2dOscillator(
            a=-1.5, b=-15.0, c=0.0, d=0.015, e=3.0, f=1.0, tau=4.0, I=0.1, VARIABLES_OF_INTEREST=("V",)
        ),
        coupling={"instant": LinearCoupling(incoming_states="V", G=0.0)},
        graph=DenseGraph(jnp.zeros((1, 1))),
        external_input={"stimulus": PulseInput(onset=10.0, duration=1.0, amplitude=0.4)},
    )
    sf, cfg = prepare(net, Heun(), t0=0.0, t1=150.0, dt=0.2)
    return np.asarray(sf(cfg).ys[::15, 0, 0]).ravel()


def test_bayesian_forward_byte_identity(eager):
    """The Bayesian workflow's forward model matches a hand-built tvboptim reference.

    Byte-identity of the forward model is what makes the posterior comparable: the observed data, likelihood, priors and NUTS setup all match, so an identical forward pass implies an identical posterior and the inference test only has to smoke-check recovery and steering.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Stimulation_Bayesian_Inference.yaml"))
    exp.configure()
    rec_tvbo = np.asarray(exp.run("tvboptim", mode="simulation").observations.recorded_ts.data).ravel()
    rec_ref = _bayesian_reference_forward()
    n = min(len(rec_tvbo), len(rec_ref))
    assert_identical("Bayesian forward recorded_ts", rec_tvbo[:n], rec_ref[:n])


def test_bayesian_inference_recovers_and_steers():
    """MCMC smoke (reduced samples): both params recovered near truth, the amplitude/I degeneracy ridge (negative correlation), and prior-steering across scenarios."""
    import jax

    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Stimulation_Bayesian_Inference.yaml"))
    for inf in exp.inferences.values():
        inf.num_warmup, inf.num_samples = 200, 400
    exp.configure()
    rec = np.asarray(exp.run("tvboptim", mode="simulation").observations.recorded_ts.data).ravel()
    observed = rec + 0.1 * np.asarray(jax.random.normal(jax.random.key(42), (rec.shape[0],)))
    r = exp.run("tvboptim", mode="inference", recorded_ts=observed)

    post = {
        k: (
            np.asarray(r.inferences[k].posterior["stimulus.amplitude"]),
            np.asarray(r.inferences[k].posterior["Generic2dOscillator.I"]),
        )
        for k in r.inferences
    }
    a_A, i_A = post["scenario_A"]
    assert abs(a_A.mean() - 0.4) < 0.15, f"amplitude not recovered: {a_A.mean():.3f}"
    assert abs(i_A.mean() - 0.1) < 0.15, f"I not recovered: {i_A.mean():.3f}"
    assert np.corrcoef(a_A, i_A)[0, 1] < -0.5, "expected amplitude/I degeneracy ridge"
    # prior-steering: the tight-excitability prior (C) clearly narrows the I posterior vs the wide prior (A) — the robust steering signal at reduced sample counts.
    assert post["scenario_C"][1].std() < post["scenario_A"][1].std(), "prior-steering (I) not observed"


# Hopf Pareto — SupHopf, NSGA-II multi-objective (FC vs frequency gradient) plus per-Pareto-seed parallel gradient refinement; replicates tvboptim Hopf_Pareto_ParallelOpt
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
    """Hand-written tvboptim reference matching the Hopf YAML, settling inside its one scan."""
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network
    from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
    from tvboptim.experimental.network_dynamics.dynamics.tvb import SupHopf
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
    from tvboptim.experimental.network_dynamics.solvers import Heun

    W = np.asarray(exp.network.weights)
    omega = np.asarray(list(exp.dynamics.parameters["omega"].value))
    net = Network(
        dynamics=SupHopf(a=0.01, omega=jnp.asarray(omega), INITIAL_STATE=(0.1, 0.0)),
        coupling={"instant": LinearCoupling(incoming_states=["x"], G=0.025)},
        graph=DenseGraph(W),
        noise=AdditiveNoise(sigma=0.01, apply_to=["x", "y"], key=jax.random.key(0)),
    )
    mm, sm, n_settle = _one_scan(net, Heun(), float(t1), float(t1), 1.0)
    return net, n_settle, mm, sm, omega


def test_hopf_trajectory_and_frequency(eager):
    exp = _load_hopf(2000.0)
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)
    psd_tvbo = np.asarray(getattr(r.observations.psd, "psd", r.observations.psd.data))
    peaks_tvbo = np.asarray(getattr(r.observations.fitted_peaks, "data", r.observations.fitted_peaks))

    _net, n_settle, mm, sm, _omega = _hopf_ref(exp, 2000.0)
    rm = mm(sm)
    sim_ref = np.asarray(rm.data)[n_settle:]
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
    from tvboptim.observations.observation import compute_fc

    T = 30000.0
    exp = _load_hopf(T)
    r = exp.run("tvboptim", mode="simulation")
    bold_tvbo = np.asarray(getattr(r.observations.bold, "data", r.observations.bold))
    fc_tvbo = np.asarray(getattr(r.observations.fc, "data", r.observations.fc))

    _net, _n_settle, mm, sm, _omega = _hopf_ref(exp, T)
    rm = mm(sm)
    bm = _emitted_bold(exp)(rm)
    assert_identical("Hopf bold", np.squeeze(bold_tvbo), np.squeeze(np.asarray(bm.data)))
    assert_identical("Hopf fc", np.squeeze(fc_tvbo), np.squeeze(np.asarray(compute_fc(bm, skip_t=20))))


def test_hopf_ga_pareto_and_refine():
    """Stage 0 (NSGA-II front) + Stage 1 (per-seed refine) byte-identical to the reference.

    The GA and the refinement both run under ``ParallelExecution`` (pmap), so this comparison is made under JIT — eager pmap is intractable — while the Hopf trajectory and frequency observations stay eager like the other workflows.
    """
    import jax.numpy as jnp
    import optax
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from tvboptim.execution import ParallelExecution
    from tvboptim.observations.observation import compute_fc, fc_corr
    from tvboptim.optim.optax import OptaxOptimizer
    from tvboptim.types import BoundedParameter, DataAxis, Parameter
    from tvboptim.types.spaces import Space

    N, T = 4, 30000.0
    exp = _load_hopf(T, num_gen=2, pop=4, refine_iter=2)
    res = exp.run("tvboptim", mode="all")
    ga = res.explorations.ga_presearch
    refine = res.optimizations.refine
    ga_X, ga_F = np.atleast_2d(np.asarray(ga.pareto_X)), np.atleast_2d(np.asarray(ga.pareto_F))
    emp = np.asarray(getattr(res.observations.empirical_fc, "data", res.observations.empirical_fc))
    pft = np.asarray([p for p in exp.functions["freq_grad_corr_fn"].arguments["target"].value])

    net, n_settle, mB, base_state = _hopf_ref(exp, T)[0:4]
    base_state = _copy.deepcopy(base_state)
    bold_mon = _emitted_bold(exp)
    nn = np.asarray(exp.network.weights).shape[0]

    def _metrics(s):
        rr = mB(s)
        x_sub = rr.data[n_settle:][::10, 0, :]
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

    # The GA decision space is the exploration's ``space`` — a keyed dict. Derive the axis order, bounds, and transforms from it (keyed by parameter name) instead of hardcoding column positions, so this reference stays byte-identical to the codegen regardless of how the axes are ordered. The only piece not derivable from the space is how each swept parameter attaches to the tvboptim state, so that mapping lives in an explicit name-keyed table.
    _AXIS_SETTERS = {
        "LinCoupling.G": lambda b, v: setattr(b.coupling.instant, "G", v),
        "SupHopf.a": lambda b, v: setattr(b.dynamics, "a", v),
        "noise.sigma": lambda b, v: setattr(b.noise, "sigma", v),
    }

    def _untransform(x, transform):
        """Invert an exploration-axis transform (the GA searches in transformed space)."""
        if transform in (None, "", "none", "identity"):
            return x
        if transform == "log10":
            return 10.0**x
        raise ValueError(f"unhandled axis transform: {transform!r}")

    ga_axes = as_list(exp.explorations["ga_presearch"].space)
    ga_xl = np.array([ax.domain.lo for ax in ga_axes])
    ga_xu = np.array([ax.domain.hi for ax in ga_axes])

    def _apply_axes(state, X):
        """Assign each decision column to its parameter by name, not by position."""
        for col, ax in enumerate(ga_axes):
            val = DataAxis(jnp.asarray(_untransform(X[:, col], ax.transform)))
            _AXIS_SETTERS[ax.parameter](state, val)

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=len(ga_axes), n_obj=2, xl=ga_xl, xu=ga_xu)

        def _evaluate(self, X, out, *a, **k):
            b = _copy.deepcopy(base_state)
            _apply_axes(b, X)
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
        return {
            "G": jnp.asarray(fin.coupling.instant.G),
            "a": jnp.asarray(fin.dynamics.a),
            "om": jnp.asarray(fin.dynamics.omega),
        }

    seed = _copy.deepcopy(base_state)
    _apply_axes(seed, ga_X)
    rows = list(ParallelExecution(_refine_seed, Space(seed, mode="zip"), n_pmap=N).run())
    for tr, rr in zip(refine.seeds, rows, strict=True):
        assert_identical("Hopf refine G", np.asarray(tr.G_final), np.asarray(rr["G"]), atol=1e-10)
        assert_identical("Hopf refine a", np.asarray(tr.a_final), np.asarray(rr["a"]), atol=1e-10)
        assert_identical("Hopf refine omega", np.asarray(tr.omega_final), np.asarray(rr["om"]), atol=1e-10)


def test_ei_optimization_runs():
    """EI_Tuning: the gradient optimization must run, not just the trajectory.

    Guards a wrapping crash class the byte-identity trajectory tests do not cover: an aggregated observable (``mean_activity`` — time axis collapsed to a per-node value) must reach the loss as a plain array, not a re-wrapped ``NativeSolution``, or ``mean_activity + target`` raises "unsupported operand ... 'NativeSolution'". Two complementary places satisfy that: the observation template returns raw for a collapsed aggregation (producer, root cause) and the loss-builder unwraps ``.data`` (consumer). The test passes on either, so it pins the failure class rather than one mechanism.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "EI_Tuning_FIC_EIB_Optimization.yaml"))
    exp.integration.duration = 2000.0
    exp.integration.transient_time = 2000.0
    for _o in exp.optimizations.values():
        if hasattr(_o, "max_iterations"):
            _o.max_iterations = 2
        for _st in getattr(_o, "stages", None) or []:
            _st.max_iterations = 2
    exp.configure()

    # The crash was an exception at loss evaluation, so completing is the guard.
    r = exp.run("tvboptim", mode="optimization")

    # Additionally assert the objective is a finite scalar (not NaN / not an object).
    def _final_loss(res):
        opts = getattr(res, "optimizations", None)
        for _v in opts.values() if opts else []:
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


# Delays Shape Synchronization: length-graph delayed coupling (DenseLengthGraph)
def test_delay_speed_trajectory(eager):
    """Delayed Kuramoto on a DenseLengthGraph: tvbo codegen == native tvboptim.

    Guards the length-graph delay path. Because the connectome carries tract lengths, tvbo lowers the delayed coupling onto a ``DenseLengthGraph`` (delays = lengths / conduction_speed, so the conduction speed is a live graph leaf). The generated trajectory must be byte-identical to a hand-written native tvboptim delayed-Kuramoto run over the same length graph.
    """
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.coupling import DelayedKuramotoCoupling
    from tvboptim.experimental.network_dynamics.dynamics.tvb import Kuramoto
    from tvboptim.experimental.network_dynamics.graph import DenseLengthGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun

    T1 = 200.0

    # --- tvbo-native ---
    exp = _load_sim("Delay_Speed_Synchronization", T1, 0.0)
    W = np.asarray(exp.network.weights_matrix)  # already weight / max(weight)
    L = np.asarray(exp.network.lengths_matrix)
    labels = [n.label for n in exp.network.nodes]
    omega = float(exp.dynamics.parameters["omega"].value)
    G = float(exp.network.coupling["DelayedKuramotoCoupling"].parameters["G"].value)
    cs = exp.network.conduction_speed
    speed = float(getattr(cs, "value", cs))
    theta_tvbo = np.asarray(exp.run("tvboptim", mode="simulation").integration.data)

    # --- tvboptim-native reference over the same DenseLengthGraph ---
    Wj, Lj = jnp.asarray(W), jnp.asarray(L)
    graph = DenseLengthGraph(
        Wj,
        Lj,
        speed=speed,
        region_labels=labels,
        max_delay_bound=float(jnp.max(Lj)) / speed,
    )
    net = Network(
        dynamics=Kuramoto(omega=omega),
        coupling={"delayed": DelayedKuramotoCoupling(incoming_states="theta", local_states="theta", G=G)},
        graph=graph,
        noise=None,
    )
    mf, st = prepare(net, Heun(), t0=0.0, t1=T1, dt=0.5)
    theta_ref = np.asarray(mf(st).ys)

    n = min(theta_tvbo.shape[0], theta_ref.shape[0])
    assert_identical("delay-speed Kuramoto trajectory", theta_tvbo[:n], theta_ref[:n])


def test_delay_speed_sweep_runs():
    """The conduction_speed sweep drives the DenseLengthGraph `speed` leaf.

    Confirms the generated exploration code runs and that varying the conduction speed varies the order parameter, so the sweep is not silently inert.
    """
    exp = SimulationExperiment.from_file(str(EXPERIMENTS_DIR / "Delay_Speed_Synchronization.yaml"))
    exp.integration.duration = 200.0
    exp.integration.transient_time = 0.0
    exp.explorations["speed_sweep"].space["network.conduction_speed"].domain.n = 4
    exp.configure()
    grid = np.asarray(exp.run("tvboptim", mode="exploration").explorations.speed_sweep.results)
    assert np.all(np.isfinite(grid)), "order-parameter sweep produced non-finite values"
    assert not np.allclose(grid.ravel(), grid.ravel()[0]), "conduction_speed sweep is inert"


_DELAY_ONLY_EXPERIMENT = """
id: 9
dynamics:
  name: Kuramoto
  parameters: {omega: {value: 0.1, unit: rad_per_ms}}
  coupling_inputs: {c: {name: c}}
  state_variables:
    theta: {unit: rad, equation: {rhs: "omega + c"}, coupling_variable: true, initial_value: 0.1}
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  nodes: [{id: 0, label: r0}, {id: 1, label: r1}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.5}, delay: {value: 12.0}}}
    - {source: 1, target: 0, parameters: {weight: {value: 0.5}, delay: {value: 12.0}}}
  coupling:
    KuramotoCoupling:
      delayed: true
      parameters: {a: {value: 0.05}, N: {value: 1.0}}
      pre_expression: {rhs: "sin(theta_j - theta_i)"}
      post_expression: {rhs: "a * gx / N"}
      incoming_states: [theta]
      local_states: [theta]
integration: {method: Heun, duration: 100.0, step_size: 0.5, transient_time: 0.0}
"""
"""A 2-node network whose edges carry explicit ``delay`` attributes and no tract lengths. This is the DenseDelayGraph branch of graph selection: with no lengths to derive delays from a conduction speed, tvbo uses the per-edge delays directly."""


def test_delay_only_graph_trajectory(eager, tmp_path):
    """Delay-only network (explicit edge delays, no lengths): DenseDelayGraph path.

    Complements the length-graph tests: when the network carries only per-edge ``delay`` attributes, tvbo must select a ``DenseDelayGraph`` over those delays rather than a length graph. The generated trajectory must be byte-identical to a native tvboptim ``DenseDelayGraph`` run with the same delays.
    """
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import Network, prepare
    from tvboptim.experimental.network_dynamics.coupling import DelayedKuramotoCoupling
    from tvboptim.experimental.network_dynamics.dynamics.tvb import Kuramoto
    from tvboptim.experimental.network_dynamics.graph.base import DenseDelayGraph
    from tvboptim.experimental.network_dynamics.solvers import Heun

    p = tmp_path / "delay_only.yaml"
    p.write_text(_DELAY_ONLY_EXPERIMENT)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    assert float(np.asarray(exp.network.lengths_matrix).max()) == 0.0, "fixture must have no lengths"
    theta_tvbo = np.asarray(exp.run("tvboptim", mode="simulation").integration.data)[:, 0, :]

    W = jnp.asarray(np.asarray(exp.network.weights_matrix))
    D = jnp.nan_to_num(jnp.asarray(np.asarray(exp.network.calculate_delays())))
    a = float(exp.network.coupling["KuramotoCoupling"].parameters["a"].value)
    omega = float(exp.dynamics.parameters["omega"].value)
    net = Network(
        dynamics=Kuramoto(omega=omega),
        coupling={"delayed": DelayedKuramotoCoupling(incoming_states="theta", local_states="theta", G=a)},
        graph=DenseDelayGraph(W, D, region_labels=["r0", "r1"]),
        noise=None,
    )
    mf, st = prepare(net, Heun(), t0=0.0, t1=100.0, dt=0.5)
    theta_ref = np.asarray(mf(st).ys)[:, 0, :]

    n = min(theta_tvbo.shape[0], theta_ref.shape[0])
    assert_identical("delay-only Kuramoto trajectory", theta_tvbo[:n], theta_ref[:n])
