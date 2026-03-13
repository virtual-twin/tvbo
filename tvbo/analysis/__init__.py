"""Analysis subpackage.

Houses analysis result container classes (e.g., BifurcationResult) and related
APIs that are logically distinct from plotting utilities or simulation drivers.
"""
import numpy as np

from .bifurcation import BifurcationResult, PyRatesBifurcationResult  # re-export

__all__ = ["BifurcationResult", "PyRatesBifurcationResult", "compare_timeseries"]


def compare_timeseries(exp: "SimulationExperiment", ts1: "Any", ts2: "Any", atol: float = 1e-10):
    """Compare state variables between two time series using multiple measures.

    Args:
        exp: Experiment object containing metadata about state variables.
        ts1: First time series.
        ts2: Second time series.
        atol: Absolute tolerance for broader identity check.
    """
    for sv in exp.model.state_variables.keys():
        data1 = ts1.get_state_variable(sv).data.squeeze()
        data2 = ts2.get_state_variable(sv).data.squeeze()

        correlation = np.corrcoef(data1, data2)[0, 1]
        mse = np.mean((data1 - data2) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.max(data1) - np.min(data1))
        max_diff = np.max(np.abs(data1 - data2))
        abs_identical = np.array_equal(data1, data2)
        broad_identical = np.allclose(data1, data2, atol=atol)

        print(f"{sv}:")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Normalized RMSE (NRMSE): {nrmse:.6f}")
        print(f"  Absolute identity: {abs_identical}")
        print(f"  Broader identity (within atol={atol}): {broad_identical}")
        print(f"  Max difference: {max_diff:.6e}")
