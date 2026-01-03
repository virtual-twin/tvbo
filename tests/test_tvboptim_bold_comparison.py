"""
Compare TVBO-generated BOLD pipeline with original tvboptim Bold monitor.

This test identifies the exact differences between the two implementations.
"""
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import jax.scipy.signal

# tvboptim imports
from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity
from tvboptim.observations.tvb_monitors.bold import Bold as TvboptimBold
from tvboptim.observations.observation import compute_fc as tvboptim_compute_fc


def test_bold_comparison():
    """Compare step-by-step the BOLD pipeline outputs."""

    # =========================================================================
    # Setup - Common to both
    # =========================================================================
    weights, lengths, region_labels = load_structural_connectivity(name="dk_average")
    weights = weights / np.max(weights)
    n_nodes = weights.shape[0]

    # Create network
    graph = DenseGraph(weights, region_labels=region_labels)
    dynamics = ReducedWongWang(w=0.5, I_o=0.32, INITIAL_STATE=(0.3,))
    coupling = FastLinearCoupling(local_states=["S"], G=0.5)
    noise = AdditiveNoise(sigma=0.00283, apply_to="S", key=jax.random.key(42))

    network = Network(
        dynamics=dynamics,
        coupling={'instant': coupling},
        graph=graph,
        noise=noise
    )

    # Simulation parameters
    t1 = 120_000
    dt = 4.0

    # Run transient
    model, state = prepare(network, Heun(), t1=t1, dt=dt)
    result_init = model(state)

    # Update network and run main simulation
    network.update_history(result_init)
    model, state = prepare(network, Heun(), t1=t1, dt=dt)
    result = model(state)

    print("\n" + "="*70)
    print("STEP 1: Raw simulation result")
    print("="*70)
    print(f"result.data.shape = {result.data.shape}")
    print(f"result.time.shape = {result.time.shape}")
    print(f"result.dt = {result.dt}")

    # =========================================================================
    # TVBOPTIM Bold Monitor (Original)
    # =========================================================================
    print("\n" + "="*70)
    print("TVBOPTIM Bold Monitor (Original)")
    print("="*70)

    # Original parameters from RWW.qmd
    bold_monitor_original = TvboptimBold(
        period=1000.0,           # BOLD sampling period
        downsample_period=4.0,   # Intermediate downsampling
        voi=0,                   # Variable of interest
        history=result_init      # Uses transient for warm start
    )

    bold_result_original = bold_monitor_original(result)
    print(f"bold_result.data.shape = {bold_result_original.data.shape}")
    print(f"bold_result.time[:5] = {bold_result_original.time[:5]}")
    print(f"bold_result.dt = {bold_result_original.dt}")

    # Compute FC with original
    fc_original = tvboptim_compute_fc(bold_result_original, skip_t=20)
    print(f"fc_original.shape = {fc_original.shape}")

    # =========================================================================
    # TVBO-Generated Pipeline (what RWW_tvboptim.qmd does)
    # =========================================================================
    print("\n" + "="*70)
    print("TVBO-Generated Pipeline")
    print("="*70)

    # Step-by-step as in generated code
    voi = 0

    # 1. HRF kernel - DIFFERENCE: Generated code uses different formula!
    def tvbo_hrf_kernel(duration=20000.0, tau_s=0.8, tau_f=0.4):
        """TVBO generated HRF kernel"""
        t = jnp.linspace(0, duration, 5000)
        # NOTE: This is the generated formula - CHECK IT!
        return (1/3)*jnp.exp(-0.0005*t/tau_s)*jnp.sin((1/1000)*t*jnp.sqrt(tau_f**(-1.0) - (1/4)/tau_s**2))/jnp.sqrt(tau_f**(-1.0) - (1/4)/tau_s**2)

    def tvboptim_hrf_kernel(duration=20000.0, tau_s=0.8, tau_f=0.4, scaling=1/3):
        """Original tvboptim HRF kernel (LotkaVolterraHRFKernel)"""
        t = jnp.linspace(0, duration, 5000)
        t_seconds = t / 1000.0  # Convert to seconds!
        omega = jnp.sqrt(1.0 / tau_f - 1.0 / (4.0 * tau_s**2))
        return scaling * jnp.exp(-0.5 * (t_seconds / tau_s)) * jnp.sin(omega * t_seconds) / omega

    hrf_tvbo = tvbo_hrf_kernel()
    hrf_original = tvboptim_hrf_kernel()

    print(f"\nHRF Kernel comparison:")
    print(f"  hrf_tvbo.shape = {hrf_tvbo.shape}")
    print(f"  hrf_original.shape = {hrf_original.shape}")
    print(f"  hrf_tvbo[:5] = {hrf_tvbo[:5]}")
    print(f"  hrf_original[:5] = {hrf_original[:5]}")
    print(f"  HRF max difference: {jnp.abs(hrf_tvbo - hrf_original).max()}")
    print(f"  HRFs are identical: {jnp.allclose(hrf_tvbo, hrf_original, rtol=1e-5)}")

    # 2. Slice data for state variable
    _data = result.data[:, voi:voi+1, :]  # 3D: (time, 1, nodes)
    _history = result_init.data[-5000:, voi:voi+1, :]
    print(f"\n_data.shape = {_data.shape}")
    print(f"_history.shape = {_history.shape}")

    # 3. Temporal average (period_samples=1 means NO averaging!)
    def temporal_average(data, period_samples=1):
        """TVBO generated temporal_average"""
        return jnp.mean(data.reshape(-1, period_samples, *data.shape[1:]), axis=1)

    _downsampled_data = temporal_average(period_samples=1, data=_data)
    _downsampled_history = temporal_average(period_samples=1, data=_history)
    print(f"\n_downsampled_data.shape = {_downsampled_data.shape}")
    print(f"  (with period_samples=1, this is NO downsampling!)")

    # 4. Prepend history
    def prepend_history(data, history, kernel_samples=5000):
        """TVBO generated prepend_history"""
        return jnp.concatenate([history[-kernel_samples:], data], axis=0)

    _prepend_history = prepend_history(
        kernel_samples=5000, data=_downsampled_data, history=_downsampled_history
    )
    print(f"\n_prepend_history.shape = {_prepend_history.shape}")

    # 5. Convolution
    _hrf_kernel = hrf_tvbo
    _bold_convolve = jax.vmap(
        lambda y: jax.vmap(
            lambda x: jax.scipy.signal.fftconvolve(x, _hrf_kernel, 'valid'),
            in_axes=1, out_axes=1
        )(y),
        in_axes=1, out_axes=1
    )(_prepend_history)
    print(f"\n_bold_convolve.shape = {_bold_convolve.shape}")

    # 6. Volterra transform - DIFFERENCE: Generated uses different formula!
    def volterra_transform(data, k_1=5.6, V_0=0.02):
        """TVBO generated volterra_transform"""
        return V_0*k_1*(-1.0 + data)

    def tvboptim_volterra(bold, k_1=5.6, V_0=0.02):
        """Original tvboptim: k_1 * V_0 * (bold - 1.0)"""
        return k_1 * V_0 * (bold - 1.0)

    _volterra_tvbo = volterra_transform(data=_bold_convolve)
    _volterra_original = tvboptim_volterra(_bold_convolve)
    print(f"\nVolterra transform comparison:")
    print(f"  TVBO formula:     V_0*k_1*(-1.0 + data)  = V_0*k_1*(data - 1)")
    print(f"  Original formula: k_1*V_0*(bold - 1.0)")
    print(f"  These are IDENTICAL (just reordered)")
    print(f"  Volterra max diff: {jnp.abs(_volterra_tvbo - _volterra_original).max()}")

    # 7. Subsample - DIFFERENCE: step=250 vs period=1000/downsample_period=4 = 250
    def subsample_bold(data, step=250, n_samples=120):
        """TVBO generated subsample_bold"""
        return data[::step][:n_samples]

    _subsample_bold = subsample_bold(step=250, n_samples=120, data=_volterra_tvbo)
    print(f"\n_subsample_bold.shape = {_subsample_bold.shape}")

    # =========================================================================
    # CRITICAL ISSUE: FC computation
    # =========================================================================
    print("\n" + "="*70)
    print("FC COMPUTATION COMPARISON")
    print("="*70)

    # TVBO generated uses skip_t=20
    fc_tvbo = tvboptim_compute_fc(_subsample_bold, skip_t=20)
    print(f"\nfc_tvbo.shape = {fc_tvbo.shape}")
    print(f"fc_original.shape = {fc_original.shape}")

    print(f"\nFC correlation: {jnp.corrcoef(fc_tvbo.flatten(), fc_original.flatten())[0,1]}")
    print(f"FC max difference: {jnp.abs(fc_tvbo - fc_original).max()}")

    # =========================================================================
    # ROOT CAUSE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ROOT CAUSE ANALYSIS")
    print("="*70)

    # Check the actual BOLD from tvboptim
    bold_data_original = bold_result_original.data
    print(f"\nOriginal BOLD shape: {bold_data_original.shape}")
    print(f"Generated BOLD shape: {_subsample_bold.shape}")

    # The generated code keeps state dimension, original might not
    if bold_data_original.ndim == 3:
        print(f"Original BOLD dim: 3D (time, states, nodes)")
    if _subsample_bold.ndim == 3:
        print(f"Generated BOLD dim: 3D (time, states, nodes)")

    # Check if BOLD signals match
    if bold_data_original.shape == _subsample_bold.shape:
        bold_corr = jnp.corrcoef(
            bold_data_original.flatten(),
            _subsample_bold.flatten()
        )[0, 1]
        print(f"\nBOLD signals correlation: {bold_corr}")
        print(f"BOLD max difference: {jnp.abs(bold_data_original - _subsample_bold).max()}")
    else:
        # Try to match dimensions
        if bold_data_original.ndim == 3 and _subsample_bold.ndim == 3:
            # Match the minimum samples
            n_samples = min(bold_data_original.shape[0], _subsample_bold.shape[0])
            bold_corr = jnp.corrcoef(
                bold_data_original[:n_samples, 0, :].flatten(),
                _subsample_bold[:n_samples, 0, :].flatten()
            )[0, 1]
            print(f"\nBOLD signals correlation (aligned): {bold_corr}")

    return {
        'hrf_identical': bool(jnp.allclose(hrf_tvbo, hrf_original, rtol=1e-5)),
        'fc_correlation': float(jnp.corrcoef(fc_tvbo.flatten(), fc_original.flatten())[0,1]),
    }


if __name__ == "__main__":
    results = test_bold_comparison()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"HRF kernels identical: {results['hrf_identical']}")
    print(f"FC correlation: {results['fc_correlation']:.6f}")
