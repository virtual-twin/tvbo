"""
Test optimization stages: Compare TVBO-generated vs original tvboptim.

This test compares the global and regional optimization stages step-by-step.
"""
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import copy
import optax

# tvboptim imports
from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity, load_functional_connectivity
from tvboptim.observations.tvb_monitors.bold import Bold as TvboptimBold
from tvboptim.observations.observation import compute_fc, fc_corr, rmse
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingCallback


def test_global_optimization():
    """Test global optimization stage."""
    print("\n" + "="*70)
    print("GLOBAL OPTIMIZATION COMPARISON")
    print("="*70)

    # Load data
    weights, lengths, region_labels = load_structural_connectivity(name="dk_average")
    weights = weights / np.max(weights)
    n_nodes = weights.shape[0]
    fc_target = load_functional_connectivity(name="dk_average")

    # =========================================================================
    # ORIGINAL TVBOPTIM SETUP
    # =========================================================================
    print("\n--- Original tvboptim setup ---")

    graph = DenseGraph(weights, region_labels=region_labels)
    dynamics = ReducedWongWang(w=0.5, I_o=0.32, INITIAL_STATE=(0.3,))
    coupling = FastLinearCoupling(local_states=["S"], G=0.5)
    noise = AdditiveNoise(sigma=0.00283, apply_to="S", key=jax.random.key(0))

    network = Network(
        dynamics=dynamics,
        coupling={'instant': coupling},
        graph=graph,
        noise=noise
    )

    t1 = 120_000
    dt = 4.0
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    # Transient
    result_init = model(state)
    network.update_history(result_init)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    # BOLD monitor
    bold_monitor = TvboptimBold(
        period=1000.0,
        downsample_period=4.0,
        voi=0,
        history=result_init
    )

    # Loss function - ORIGINAL
    def loss_original(s):
        result = model(s)
        bold = bold_monitor(result)
        fc = compute_fc(bold, skip_t=20)
        return rmse(fc, fc_target)

    # Mark parameters - ORIGINAL (from RWW.qmd)
    state_orig = copy.deepcopy(state)
    state_orig.coupling.instant.G = Parameter(state_orig.coupling.instant.G)
    state_orig.dynamics.w = Parameter(state_orig.dynamics.w)

    print(f"Initial w: {state_orig.dynamics.w.value}")
    print(f"Initial G: {state_orig.coupling.instant.G.value}")
    print(f"Initial loss: {loss_original(state_orig):.6f}")

    # Run optimization - ORIGINAL (5 steps for quick test)
    cb_orig = MultiCallback([
        DefaultPrintCallback(every=1),
        SavingCallback(key="state", save_fun=lambda *args: args[1])
    ])
    opt_orig = OptaxOptimizer(loss_original, optax.adam(0.01, b2=0.9999), callback=cb_orig)
    fitted_orig, history_orig = opt_orig.run(state_orig, max_steps=5)

    print(f"\nOriginal fitted w: {fitted_orig.dynamics.w.value}")
    print(f"Original fitted G: {fitted_orig.coupling.instant.G.value}")
    print(f"Original final loss: {loss_original(fitted_orig):.6f}")

    # =========================================================================
    # TVBO-GENERATED SETUP
    # =========================================================================
    print("\n--- TVBO-generated setup ---")

    from tvbo import SimulationExperiment, Network as TvboNetwork

    exp = SimulationExperiment.from_file(
        '/Users/leonmartin_bih/tools/tvbo/database/experiments/RWW_BOLD_FC_Optimization.yaml'
    )
    exp.network = TvboNetwork.from_matrix(
        weights * np.max(weights),  # Undo normalization, exp does its own
        lengths, region_labels, normalization=exp.network.normalization
    )

    # Get rendered code module
    code = exp.render_code('tvboptim')

    # Execute the code to get functions
    exec_globals = {}
    exec(code, exec_globals)

    # Get functions from generated code
    create_network = exec_globals['create_network']
    run_simulation = exec_globals['run_simulation']
    make_loss_fn = exec_globals['make_loss_fn']
    mark_parameters_global_optimization = exec_globals['mark_parameters_global_optimization']
    create_optimizer = exec_globals['create_optimizer']

    # Create network via generated code
    network_tvbo = create_network(jnp.array(weights * np.max(weights)), region_labels=region_labels)

    # Run simulation
    sim_result = run_simulation(network_tvbo, t1=120000.0, dt=4.0, t_transient=120000.0)
    model_tvbo = sim_result.model_fn
    state_tvbo = sim_result.state
    transient_tvbo = sim_result.transient

    # Create loss function
    loss_tvbo = make_loss_fn(model_tvbo, fc_target, result_transient=transient_tvbo)

    print(f"Initial w: {state_tvbo.dynamics.w}")
    print(f"Initial G: {state_tvbo.coupling.c_instant.G}")
    print(f"Initial loss: {loss_tvbo(state_tvbo):.6f}")

    # Mark parameters - TVBO
    state_tvbo_marked = mark_parameters_global_optimization(state_tvbo, n_nodes=n_nodes)

    print(f"Marked w type: {type(state_tvbo_marked.dynamics.w)}")
    print(f"Marked G type: {type(state_tvbo_marked.coupling.c_instant.G)}")

    # Run optimization - TVBO (5 steps)
    opt_tvbo = create_optimizer(loss_tvbo, optimizer="adam", learning_rate=0.01, b2=0.9999)
    fitted_tvbo, history_tvbo = opt_tvbo.run(state_tvbo_marked, max_steps=5)

    print(f"\nTVBO fitted w: {fitted_tvbo.dynamics.w.value}")
    print(f"TVBO fitted G: {fitted_tvbo.coupling.c_instant.G.value}")
    print(f"TVBO final loss: {loss_tvbo(fitted_tvbo):.6f}")

    # =========================================================================
    # COMPARE
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Compare trajectories
    orig_w_traj = [s.dynamics.w.value for s in history_orig["state"].save]
    orig_G_traj = [s.coupling.instant.G.value for s in history_orig["state"].save]
    tvbo_w_traj = [s.dynamics.w.value for s in history_tvbo["state"].save]
    tvbo_G_traj = [s.coupling.c_instant.G.value for s in history_tvbo["state"].save]

    print("\nOptimization trajectory comparison:")
    print("Step | Orig w    | TVBO w    | Orig G    | TVBO G")
    print("-" * 55)
    for i in range(min(len(orig_w_traj), len(tvbo_w_traj))):
        print(f"{i:4d} | {orig_w_traj[i]:.6f} | {tvbo_w_traj[i]:.6f} | {orig_G_traj[i]:.6f} | {tvbo_G_traj[i]:.6f}")

    # Check if trajectories match
    w_match = np.allclose(orig_w_traj, tvbo_w_traj, rtol=1e-5)
    G_match = np.allclose(orig_G_traj, tvbo_G_traj, rtol=1e-5)

    print(f"\nw trajectories match: {w_match}")
    print(f"G trajectories match: {G_match}")

    return {
        'orig_w': orig_w_traj,
        'orig_G': orig_G_traj,
        'tvbo_w': tvbo_w_traj,
        'tvbo_G': tvbo_G_traj,
        'w_match': w_match,
        'G_match': G_match,
    }


def test_regional_optimization():
    """Test regional optimization stage."""
    print("\n" + "="*70)
    print("REGIONAL OPTIMIZATION COMPARISON")
    print("="*70)

    # Load data
    weights, lengths, region_labels = load_structural_connectivity(name="dk_average")
    weights_norm = weights / np.max(weights)
    n_nodes = weights.shape[0]
    fc_target = load_functional_connectivity(name="dk_average")

    # =========================================================================
    # ORIGINAL TVBOPTIM SETUP
    # =========================================================================
    print("\n--- Original tvboptim setup ---")

    graph = DenseGraph(weights_norm, region_labels=region_labels)
    dynamics = ReducedWongWang(w=0.5, I_o=0.32, INITIAL_STATE=(0.3,))
    coupling = FastLinearCoupling(local_states=["S"], G=0.5)
    noise = AdditiveNoise(sigma=0.00283, apply_to="S", key=jax.random.key(0))

    network = Network(
        dynamics=dynamics,
        coupling={'instant': coupling},
        graph=graph,
        noise=noise
    )

    t1 = 120_000
    dt = 4.0
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    result_init = model(state)
    network.update_history(result_init)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    bold_monitor = TvboptimBold(
        period=1000.0,
        downsample_period=4.0,
        voi=0,
        history=result_init
    )

    def loss_original(s):
        result = model(s)
        bold = bold_monitor(result)
        fc = compute_fc(bold, skip_t=20)
        return rmse(fc, fc_target)

    # Setup for regional optimization - ORIGINAL (from RWW.qmd)
    state_orig = copy.deepcopy(state)

    # Make w regional
    state_orig.dynamics.w = Parameter(state_orig.dynamics.w)
    state_orig.dynamics.w.shape = (n_nodes,)

    # Make I_o regional
    state_orig.dynamics.I_o = Parameter(state_orig.dynamics.I_o)
    state_orig.dynamics.I_o.shape = (n_nodes,)

    # Keep G fixed (not wrapped in Parameter)
    # state_orig.coupling.instant.G stays as float

    print(f"Initial w shape: {state_orig.dynamics.w.shape}")
    print(f"Initial I_o shape: {state_orig.dynamics.I_o.shape}")
    print(f"Initial loss: {loss_original(state_orig):.6f}")

    # Run optimization - ORIGINAL (3 steps)
    cb_orig = MultiCallback([
        DefaultPrintCallback(every=1),
        SavingCallback(key="state", save_fun=lambda *args: args[1])
    ])
    opt_orig = OptaxOptimizer(loss_original, optax.adam(0.004, b2=0.999), callback=cb_orig)
    fitted_orig, history_orig = opt_orig.run(state_orig, max_steps=3)

    print(f"\nOriginal fitted w mean: {np.mean(fitted_orig.dynamics.w.value):.6f}")
    print(f"Original fitted I_o mean: {np.mean(fitted_orig.dynamics.I_o.value):.6f}")
    print(f"Original final loss: {loss_original(fitted_orig):.6f}")

    # =========================================================================
    # TVBO-GENERATED SETUP
    # =========================================================================
    print("\n--- TVBO-generated setup ---")

    from tvbo import SimulationExperiment, Network as TvboNetwork

    exp = SimulationExperiment.from_file(
        '/Users/leonmartin_bih/tools/tvbo/database/experiments/RWW_BOLD_FC_Optimization.yaml'
    )
    exp.network = TvboNetwork.from_matrix(
        weights, lengths, region_labels, normalization=exp.network.normalization
    )

    code = exp.render_code('tvboptim')
    exec_globals = {}
    exec(code, exec_globals)

    create_network = exec_globals['create_network']
    run_simulation = exec_globals['run_simulation']
    make_loss_fn = exec_globals['make_loss_fn']
    mark_parameters_regional_optimization = exec_globals['mark_parameters_regional_optimization']
    create_optimizer = exec_globals['create_optimizer']

    network_tvbo = create_network(jnp.array(weights), region_labels=region_labels)
    sim_result = run_simulation(network_tvbo, t1=120000.0, dt=4.0, t_transient=120000.0)
    model_tvbo = sim_result.model_fn
    state_tvbo = sim_result.state
    transient_tvbo = sim_result.transient

    loss_tvbo = make_loss_fn(model_tvbo, fc_target, result_transient=transient_tvbo)

    print(f"Initial w: {state_tvbo.dynamics.w}")
    print(f"Initial I_o: {state_tvbo.dynamics.I_o}")
    print(f"Initial loss: {loss_tvbo(state_tvbo):.6f}")

    # Mark parameters - TVBO regional
    state_tvbo_marked = mark_parameters_regional_optimization(state_tvbo, n_nodes=n_nodes)

    print(f"Marked w type: {type(state_tvbo_marked.dynamics.w)}")
    print(f"Marked w shape: {state_tvbo_marked.dynamics.w.shape}")
    print(f"Marked I_o type: {type(state_tvbo_marked.dynamics.I_o)}")
    print(f"Marked I_o shape: {state_tvbo_marked.dynamics.I_o.shape}")

    # Run optimization - TVBO (3 steps)
    opt_tvbo = create_optimizer(loss_tvbo, optimizer="adam", learning_rate=0.004, b2=0.999)
    fitted_tvbo, history_tvbo = opt_tvbo.run(state_tvbo_marked, max_steps=3)

    print(f"\nTVBO fitted w mean: {np.mean(fitted_tvbo.dynamics.w.value):.6f}")
    print(f"TVBO fitted I_o mean: {np.mean(fitted_tvbo.dynamics.I_o.value):.6f}")
    print(f"TVBO final loss: {loss_tvbo(fitted_tvbo):.6f}")

    # =========================================================================
    # COMPARE
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    orig_w_final = np.array(fitted_orig.dynamics.w.value).flatten()
    tvbo_w_final = np.array(fitted_tvbo.dynamics.w.value).flatten()
    orig_Io_final = np.array(fitted_orig.dynamics.I_o.value).flatten()
    tvbo_Io_final = np.array(fitted_tvbo.dynamics.I_o.value).flatten()

    w_corr = np.corrcoef(orig_w_final, tvbo_w_final)[0, 1]
    Io_corr = np.corrcoef(orig_Io_final, tvbo_Io_final)[0, 1]

    print(f"\nFitted w correlation: {w_corr:.6f}")
    print(f"Fitted I_o correlation: {Io_corr:.6f}")
    print(f"w max diff: {np.abs(orig_w_final - tvbo_w_final).max():.6e}")
    print(f"I_o max diff: {np.abs(orig_Io_final - tvbo_Io_final).max():.6e}")

    return {
        'w_corr': w_corr,
        'Io_corr': Io_corr,
    }


if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZATION STAGES COMPARISON TEST")
    print("="*70)

    global_results = test_global_optimization()
    regional_results = test_regional_optimization()

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Global optimization w match: {global_results['w_match']}")
    print(f"Global optimization G match: {global_results['G_match']}")
    print(f"Regional optimization w correlation: {regional_results['w_corr']:.6f}")
    print(f"Regional optimization I_o correlation: {regional_results['Io_corr']:.6f}")
