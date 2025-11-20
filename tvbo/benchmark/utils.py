import numpy as np
from tvbo import SimulationExperiment


def compare_timeseries(
    exp: SimulationExperiment, ts1, ts2, atol: float = 1e-10
) -> None:
    """
    Compare state variables between two time series using multiple measures.

    Args:
        exp (SimulationExperiment): Experiment object containing metadata about state variables.
        ts1 (object): First time series (e.g., downsampled data).
        ts2 (object): Second time series (e.g., TVB data).
        atol (float): Absolute tolerance for broader identity check (default: 1e-10).

    Returns:
        None: Prints comparison results for each state variable.
    """
    for sv in exp.model.state_variables.keys():
        data1 = ts1.get_state_variable(sv).data.squeeze()
        data2 = ts2.get_state_variable(sv).data.squeeze()

        # Compute measures
        correlation = np.corrcoef(data1, data2)[0, 1]
        mse = np.mean((data1 - data2) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.max(data1) - np.min(data1))
        max_diff = np.max(np.abs(data1 - data2))
        mean_diff = np.mean(np.abs(data1 - data2))
        abs_identical = np.array_equal(data1, data2)
        broad_identical = np.allclose(data1, data2, atol=atol)

        # Print results
        print(f"{sv}:")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Normalized RMSE (NRMSE): {nrmse:.6f}")
        print(f"  Absolute identity: {abs_identical}")
        print(f"  Broader identity (within atol={atol}): {broad_identical}")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")

        # Additional information for mismatches
        if not abs_identical and not broad_identical:
            if data1.shape != data2.shape:
                print(f"  Shape mismatch: {data1.shape} vs {data2.shape}")
