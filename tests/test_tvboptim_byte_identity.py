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
    # Target/empirical observations are only needed for loss (optimization);
    # drop any that source network observational data so simulation mode runs
    # without a companion BIDS measure.
    if exp.observations:
        for obs_name in [
            n for n, o in exp.observations.items()
            if any("network.observations" in str(s) for s in (getattr(o, "source", None) or []))
        ]:
            del exp.observations[obs_name]
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

    T1, TRANSIENT, DT = 720.0, 720.0, 4.0

    # --- tvbo-native ---
    exp = _load_sim("EI_Tuning_FIC_EIB_Optimization", T1, TRANSIENT)
    if exp.observations:  # trajectory-only test: drop all observations
        for k in list(exp.observations):
            del exp.observations[k]
    W = np.asarray(exp.network.weights)
    labels = [n.label for n in exp.network.nodes]
    n_nodes = W.shape[0]
    r = exp.run("tvboptim", mode="simulation")
    sim_tvbo = np.asarray(r.integration.data)

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

    n = min(sim_tvbo.shape[1], sim_ref.shape[1])
    assert_identical("EI trajectory", sim_tvbo[:, :n, :], sim_ref[:, :n, :])
