#!/usr/bin/env python3
"""
Comprehensive comparison between schema-driven BOLD and TVB native implementation.
Tests that our corrected HRF (time-reversed) produces the same results as TVB.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from linkml_runtime.loaders import yaml_loader
from tvbo.datamodel.tvbo_datamodel import Observation
from tvbo.data.types import TimeSeries
from mako.template import Template
from mako.lookup import TemplateLookup

from tvb.simulator import simulator, models, monitors, coupling, integrators
from tvb.datatypes import connectivity, equations


def setup_tvb_simulator():
    """Setup TVB Simulator with single-node connectivity and BOLD monitor."""
    print("Setting up TVB Simulator...")

    # Single-node connectivity
    conn = connectivity.Connectivity(
        weights=np.array([[0.0]]),
        tract_lengths=np.array([[0.0]]),
        number_of_regions=1,
        region_labels=np.array(["Region_1"]),
        centres=np.array([[0.0, 0.0, 0.0]]),
    )
    conn.speed = np.array([4.0])

    # Linear model (no dynamics, pure passthrough)
    model = models.Linear(gamma=np.array([0.0]))

    # No coupling
    coupl = coupling.Linear(a=np.array([0.0]))

    # Integrator at 4ms
    integrator = integrators.HeunDeterministic(dt=4.0)

    # BOLD monitor with FirstOrderVolterra HRF
    bold_monitor = monitors.Bold()
    bold_monitor.period = 720.0  # TR in ms
    bold_monitor.hrf_length = 30000.0  # 30s HRF
    bold_monitor.hrf_kernel = equations.FirstOrderVolterra()
    bold_monitor.hrf_kernel.parameters = {
        "tau_s": 0.8,
        "tau_f": 0.4,
        "k_1": 5.6,
        "V_0": 0.02,
    }

    # Spike initial condition at t=0
    initial_conditions = np.zeros((1, model.nvar, 1, model.number_of_modes))
    initial_conditions[0, 0, 0, 0] = 1.0

    # Raw monitor to get neural activity
    raw_monitor = monitors.Raw()

    # Create simulator
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupl,
        integrator=integrator,
        monitors=[bold_monitor, raw_monitor],
        initial_conditions=initial_conditions,
    )
    sim.configure()

    return sim, bold_monitor


def load_schema_model():
    """Load schema-driven BOLD model."""
    print("Loading schema-driven model...")

    observation = yaml_loader.load(
        "/Users/leonmartin_bih/tools/tvbo/database/observation_models/bold_tvb.yaml",
        target_class=Observation,
    )

    template_dir = "/Users/leonmartin_bih/tools/tvbo/tvbo/templates/autodiff"
    lookup = TemplateLookup(directories=[template_dir])
    template = Template(
        filename=f"{template_dir}/jax-observation.py.mako", lookup=lookup
    )

    code = template.render(observation=observation, dt=4)
    namespace = {}
    exec(code, namespace)

    return namespace, observation


def run_comparison():
    """Run full comparison between TVB and schema implementations."""
    print("=" * 80)
    print("SCHEMA vs TVB BOLD IMPLEMENTATION COMPARISON")
    print("=" * 80)

    # Setup both implementations
    tvb_sim, tvb_monitor = setup_tvb_simulator()
    schema_namespace, observation = load_schema_model()

    # Run TVB simulation
    print("\nRunning TVB simulation (30 seconds)...")
    simulation_length = 30.0 * 1000.0  # 30s in ms

    tvb_bold_data = []
    tvb_bold_times = []
    tvb_raw_data = []
    tvb_raw_times = []

    for outputs in tvb_sim(simulation_length=simulation_length):
        # BOLD output (first monitor)
        if outputs[0] is not None:
            time, data = outputs[0]
            tvb_bold_times.append(time)
            tvb_bold_data.append(data)

        # Raw output (second monitor)
        if outputs[1] is not None:
            time, data = outputs[1]
            tvb_raw_times.append(time)
            tvb_raw_data.append(data)

    tvb_bold_signal = np.concatenate(tvb_bold_data, axis=0) if tvb_bold_data else None
    tvb_bold_times = np.array(tvb_bold_times) / 1000.0  # Convert to seconds
    tvb_raw_signal = np.concatenate(tvb_raw_data, axis=0)
    tvb_raw_times = np.array(tvb_raw_times)

    print(f"TVB simulation complete:")
    print(f"  Raw output: {len(tvb_raw_signal)} time points @ 4ms")
    print(
        f"  BOLD output: {len(tvb_bold_signal)} time points @ {(tvb_bold_times[1]-tvb_bold_times[0])*1000:.0f}ms"
    )

    # Apply schema model to same raw input
    print("\nApplying schema-driven BOLD model...")

    # TVB raw output shape: (time, svars, nodes, modes)
    # Ensure it has 4 dimensions
    if tvb_raw_signal.ndim == 3:
        # Add mode dimension
        tvb_raw_signal = tvb_raw_signal[..., np.newaxis]

    print(f"  Input shape: {tvb_raw_signal.shape}")

    schema_input = TimeSeries(
        data=tvb_raw_signal,
        time=tvb_raw_times,
        sample_period=4.0,
        units={"time": "ms", "state": None, "region": None, "mode": None},
    )

    schema_bold = schema_namespace["observe_BOLD_TVB"](schema_input)

    print(
        f"Schema BOLD output: {len(schema_bold.data)} time points @ {schema_bold.sample_period*1000:.0f}ms"
    )

    # Get HRF kernels for comparison
    print("\nComparing HRF kernels...")
    tvb_hrf = tvb_monitor.hemodynamic_response_function.flatten()
    tvb_hrf_times = tvb_monitor._stock_time

    schema_input_s = schema_input.convert_units("time", "s")
    schema_hrf = schema_namespace["HemodynamicResponseFunctionTVB"](schema_input_s)
    schema_hrf_data = np.array(schema_hrf.data).flatten()

    print(f"\nTVB HRF:")
    print(f"  Length: {len(tvb_hrf)} points")
    print(
        f"  Peak index: {tvb_hrf.argmax()} ({100*tvb_hrf.argmax()/len(tvb_hrf):.1f}% of duration)"
    )
    print(f"  Peak value: {tvb_hrf.max():.6f}")
    print(f"  Integral: {tvb_hrf.sum():.2f}")

    print(f"\nSchema HRF:")
    print(f"  Length: {len(schema_hrf_data)} points")
    print(
        f"  Peak index: {schema_hrf_data.argmax()} ({100*schema_hrf_data.argmax()/len(schema_hrf_data):.1f}% of duration)"
    )
    print(f"  Peak value: {schema_hrf_data.max():.6f}")
    print(f"  Integral: {schema_hrf_data.sum():.2f}")

    # Check HRF match
    peak_diff = np.abs(tvb_hrf.argmax() - schema_hrf_data.argmax())
    value_match = np.isclose(tvb_hrf.max(), schema_hrf_data.max(), rtol=0.01)
    integral_match = np.isclose(tvb_hrf.sum(), schema_hrf_data.sum(), rtol=0.01)

    print(f"\nHRF Comparison:")
    print(f"  Peak position difference: {peak_diff} points")
    print(f"  Peak values match: {'✓' if value_match else '✗'}")
    print(f"  Integrals match: {'✓' if integral_match else '✗'}")

    hrf_matches = (peak_diff < 10) and value_match and integral_match

    # Compare BOLD signals
    print("\nComparing BOLD signals...")

    # Need to align time points since they might differ slightly
    schema_bold_data = schema_bold.data[:, 0, 0, 0]
    tvb_bold_data_aligned = tvb_bold_signal[: len(schema_bold_data), 0, 0]

    # Get first 10 time points for detailed comparison
    n_compare = min(10, len(schema_bold_data), len(tvb_bold_data_aligned))

    print(f"\nFirst {n_compare} BOLD values:")
    print("  Time [s]  |  TVB BOLD  |  Schema BOLD  |  Diff")
    print("  " + "-" * 50)
    for i in range(n_compare):
        t = schema_bold.time[i]
        tvb_val = tvb_bold_data_aligned[i]
        schema_val = schema_bold_data[i]
        diff = schema_val - tvb_val
        print(f"  {t:8.2f}  |  {tvb_val:9.6f}  |  {schema_val:9.6f}  |  {diff:+.6f}")

    # Calculate statistics
    mse = np.mean(
        (schema_bold_data[:n_compare] - tvb_bold_data_aligned[:n_compare]) ** 2
    )
    max_diff = np.max(
        np.abs(schema_bold_data[:n_compare] - tvb_bold_data_aligned[:n_compare])
    )
    correlation = np.corrcoef(
        schema_bold_data[:n_compare], tvb_bold_data_aligned[:n_compare]
    )[0, 1]

    print(f"\nBOLD Signal Statistics:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Correlation: {correlation:.6f}")

    bold_matches = (mse < 1e-6) and (correlation > 0.99)

    # Create visualization
    print("\nGenerating comparison plots...")
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # HRF comparison
    axs[0, 0].plot(tvb_hrf_times, tvb_hrf, "o-", label="TVB", alpha=0.7, markersize=3)
    axs[0, 0].plot(
        schema_hrf.time, schema_hrf_data, "s-", label="Schema", alpha=0.7, markersize=3
    )
    axs[0, 0].set_title("HRF Kernel Comparison")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("HRF Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # HRF overlay (zoomed)
    axs[0, 1].plot(tvb_hrf_times, tvb_hrf, "o-", label="TVB", alpha=0.7, markersize=3)
    axs[0, 1].plot(
        schema_hrf.time, schema_hrf_data, "s-", label="Schema", alpha=0.7, markersize=3
    )
    axs[0, 1].set_xlim(20, 30)  # Zoom to peak region
    axs[0, 1].set_title("HRF Peak Region (20-30s)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("HRF Amplitude")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # BOLD time series comparison (first 10s)
    mask = schema_bold.time <= 10.0
    axs[0, 2].plot(
        tvb_bold_times[tvb_bold_times <= 10],
        tvb_bold_signal[tvb_bold_times <= 10, 0, 0],
        "o-",
        label="TVB",
        alpha=0.7,
        markersize=6,
    )
    axs[0, 2].plot(
        schema_bold.time[mask],
        schema_bold_data[mask],
        "s-",
        label="Schema",
        alpha=0.7,
        markersize=6,
    )
    axs[0, 2].set_title("BOLD Response (first 10s)")
    axs[0, 2].set_xlabel("Time [s]")
    axs[0, 2].set_ylabel("BOLD Signal")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # BOLD full time series
    axs[1, 0].plot(
        tvb_bold_times,
        tvb_bold_signal[:, 0, 0],
        "o-",
        label="TVB",
        alpha=0.7,
        markersize=4,
    )
    axs[1, 0].plot(
        schema_bold.time,
        schema_bold_data,
        "s-",
        label="Schema",
        alpha=0.7,
        markersize=4,
    )
    axs[1, 0].set_title("Full BOLD Response")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("BOLD Signal")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Difference plot
    diff = (
        schema_bold_data[: len(tvb_bold_data_aligned)]
        - tvb_bold_data_aligned[: len(schema_bold_data)]
    )
    axs[1, 1].plot(schema_bold.time[: len(diff)], diff, "r-", linewidth=2)
    axs[1, 1].set_title(f"Difference (Schema - TVB)\nMSE={mse:.2e}")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Difference")
    axs[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axs[1, 1].grid(True, alpha=0.3)

    # Scatter plot
    axs[1, 2].scatter(
        tvb_bold_data_aligned[: len(schema_bold_data)], schema_bold_data, alpha=0.5
    )
    min_val = min(tvb_bold_data_aligned.min(), schema_bold_data.min())
    max_val = max(tvb_bold_data_aligned.max(), schema_bold_data.max())
    axs[1, 2].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect match")
    axs[1, 2].set_title(f"BOLD Correlation\nR={correlation:.6f}")
    axs[1, 2].set_xlabel("TVB BOLD")
    axs[1, 2].set_ylabel("Schema BOLD")
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    plt.suptitle(
        "Schema-Driven vs TVB Native BOLD Implementation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "/Users/leonmartin_bih/tools/tvbo/tests/tvb_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Plot saved to: tests/tvb_comparison.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if hrf_matches and bold_matches:
        print("\n✓✓✓ SUCCESS! Schema implementation matches TVB exactly!")
        print("  ✓ HRF kernels match (time-reversed format)")
        print("  ✓ BOLD signals match (correlation > 0.99)")
        print("\nThe double reversal issue has been FIXED!")
        return True
    elif hrf_matches:
        print("\n✓ HRF kernels match")
        print("✗ BOLD signals differ")
        print(f"  MSE: {mse:.8f}")
        print(f"  Correlation: {correlation:.6f}")
        return False
    else:
        print("\n✗ HRF kernels do not match")
        print("✗ Implementation differs from TVB")
        return False


if __name__ == "__main__":
    success = run_comparison()
    sys.exit(0 if success else 1)
