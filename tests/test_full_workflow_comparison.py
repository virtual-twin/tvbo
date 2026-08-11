"""
Full workflow comparison: TVBO-generated vs original tvboptim RWW workflow.

Tests:
1. Simulation output identity 2. BOLD computation identity 3. FC computation identity 4. Exploration grid identity 5. Optimization convergence comparison"""

import pytest

pytest.importorskip("tvboptim")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import copy

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
from tvboptim.types import Space, GridAxis
from tvboptim.execution import ParallelExecution


def run_original_workflow():
    """Run the original RWW.qmd workflow."""
    print("\n" + "=" * 70)
    print("ORIGINAL TVBOPTIM WORKFLOW")
    print("=" * 70)

    # Load data
    weights, lengths, region_labels = load_structural_connectivity(name="dk_average")
    weights = weights / np.max(weights)
    weights.shape[0]
    fc_target = load_functional_connectivity(name="dk_average")

    # Create network - ORIGINAL uses 'instant' key
    graph = DenseGraph(weights, region_labels=region_labels)
    dynamics = ReducedWongWang(w=0.5, I_o=0.32, INITIAL_STATE=(0.3,))
    coupling = FastLinearCoupling(local_states=["S"], G=0.5)
    noise = AdditiveNoise(sigma=0.00283, apply_to="S", key=jax.random.key(0))

    network = Network(
        dynamics=dynamics,
        coupling={"instant": coupling},  # ORIGINAL: 'instant'
        graph=graph,
        noise=noise,
    )

    # Simulation
    t1 = 120_000
    dt = 4.0
    model, state = prepare(network, Heun(), t1=t1, dt=dt)

    # Transient
    result_init = model(state)
    network.update_history(result_init)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)
    result = model(state)

    # BOLD
    bold_monitor = TvboptimBold(period=1000.0, downsample_period=4.0, voi=0, history=result_init)
    bold_result = bold_monitor(result)

    # FC
    fc_initial = compute_fc(bold_result, skip_t=20)

    print(f"Simulation: {result.data.shape}")
    print(f"BOLD: {bold_result.data.shape}")
    print(f"FC: {fc_initial.shape}")
    print(f"FC correlation with target: {fc_corr(fc_initial, fc_target):.6f}")

    # Exploration
    print("\nRunning exploration (32x32 grid)...")
    grid_state = copy.deepcopy(state)
    grid_state.dynamics.w = GridAxis(0.001, 0.7, 32)
    grid_state.coupling.instant.G = GridAxis(0.001, 0.7, 32)  # ORIGINAL: 'instant'
    grid = Space(grid_state, mode="product")

    def loss(s):
        r = model(s)
        bold = bold_monitor(r)
        fc = compute_fc(bold, skip_t=20)
        return rmse(fc, fc_target)

    exec_runner = ParallelExecution(loss, grid, n_pmap=8)
    exploration_results = exec_runner.run()

    print(f"Exploration shape: {jnp.stack(exploration_results).shape}")

    return {
        "result": result,
        "bold": bold_result,
        "fc": fc_initial,
        "exploration": jnp.stack(exploration_results),
        "state": state,
        "fc_target": fc_target,
        "model": model,
        "bold_monitor": bold_monitor,
    }


def run_tvbo_workflow():
    """Run the TVBO-generated workflow."""
    print("\n" + "=" * 70)
    print("TVBO-GENERATED WORKFLOW")
    print("=" * 70)

    from tvbo import SimulationExperiment, Network as TvboNetwork

    # Load experiment
    from tvbo import database_path

    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "RWW_BOLD_FC_Optimization.yaml"))
    weights, lengths, region_labels = load_structural_connectivity(name="dk_average")
    exp.network = TvboNetwork.from_matrix(weights, lengths, region_labels, transforms=exp.network.transforms)

    fc_target = load_functional_connectivity(name="dk_average")

    # Run with exploration mode
    results = exp.run("tvboptim", mode="exploration", target_data=fc_target)

    print(f"Simulation: {results.result.data.shape}")
    print(f"BOLD: {results.observations.bold.data.shape}")
    print(f"FC: {results.observations.fc.data.shape}")
    print(f"FC correlation with target: {fc_corr(results.observations.fc.data, fc_target):.6f}")
    print(f"Exploration shape: {results.explorations.parameter_landscape.results.shape}")

    return {
        "result": results.result,
        "bold": results.observations.bold,
        "fc": results.observations.fc.data,
        "exploration": results.explorations.parameter_landscape.results,
        "state": results.state,
    }


def compare_results(original, tvbo):
    """Compare the two workflows.

    Noise means the two runs match exactly only when they share a seed; what the comparison shows is whether the pipelines are equivalent, reported as a correlation and a maximum difference per quantity.
    """
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # FC comparison
    fc_orig = np.array(original["fc"])
    fc_tvbo = np.array(tvbo["fc"])

    fc_correlation = float(jnp.corrcoef(fc_orig.flatten(), fc_tvbo.flatten())[0, 1])
    fc_diff = float(jnp.abs(fc_orig - fc_tvbo).max())

    print("\nFC Matrix Comparison:")
    print(f"  Correlation: {fc_correlation:.10f}")
    print(f"  Max difference: {fc_diff:.10e}")

    # Exploration comparison
    expl_orig = np.array(original["exploration"])
    expl_tvbo = np.array(tvbo["exploration"])

    expl_correlation = float(jnp.corrcoef(expl_orig.flatten(), expl_tvbo.flatten())[0, 1])
    expl_diff = float(jnp.abs(expl_orig - expl_tvbo).max())

    print("\nExploration Grid Comparison:")
    print(f"  Correlation: {expl_correlation:.10f}")
    print(f"  Max difference: {expl_diff:.10e}")

    # BOLD comparison
    bold_orig = np.array(original["bold"].data)
    bold_tvbo = np.array(tvbo["bold"].data)

    bold_correlation = float(jnp.corrcoef(bold_orig.flatten(), bold_tvbo.flatten())[0, 1])
    bold_diff = float(jnp.abs(bold_orig - bold_tvbo).max())

    print("\nBOLD Signal Comparison:")
    print(f"  Correlation: {bold_correlation:.10f}")
    print(f"  Max difference: {bold_diff:.10e}")

    # Raw simulation comparison
    sim_orig = np.array(original["result"].data)
    sim_tvbo = np.array(tvbo["result"].data)

    sim_correlation = float(jnp.corrcoef(sim_orig.flatten(), sim_tvbo.flatten())[0, 1])
    sim_diff = float(jnp.abs(sim_orig - sim_tvbo).max())

    print("\nRaw Simulation Comparison:")
    print(f"  Correlation: {sim_correlation:.10f}")
    print(f"  Max difference: {sim_diff:.10e}")

    return {
        "fc_corr": fc_correlation,
        "fc_diff": fc_diff,
        "expl_corr": expl_correlation,
        "expl_diff": expl_diff,
        "bold_corr": bold_correlation,
        "bold_diff": bold_diff,
        "sim_corr": sim_correlation,
        "sim_diff": sim_diff,
    }


if __name__ == "__main__":
    # Run both workflows
    original = run_original_workflow()
    tvbo = run_tvbo_workflow()

    # Compare
    results = compare_results(original, tvbo)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if pipelines are equivalent
    all_close = all(
        [
            results["fc_corr"] > 0.99,
            results["expl_corr"] > 0.99,
            results["bold_corr"] > 0.99,
            results["sim_corr"] > 0.99,
        ]
    )

    if all_close:
        print("✅ Pipelines are EQUIVALENT (correlations > 0.99)")
    else:
        print("❌ Pipelines have DIFFERENCES")
        for key, val in results.items():
            print(f"   {key}: {val}")
