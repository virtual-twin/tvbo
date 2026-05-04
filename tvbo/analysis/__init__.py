"""Analysis subpackage.

Houses analysis result container classes (e.g., BifurcationResult) and related
APIs that are logically distinct from plotting utilities or simulation drivers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tvbo.export.experiment import SimulationExperiment

from .bifurcation import BifurcationResult  # re-export

__all__ = [
    "BifurcationResult",
    "compare_timeseries",
    "per_window_fc",
    "ttest_correlation_strength",
]


def per_window_fc(tv, xv, window=1e3):
    """Calculate per-window functional connectivity.

    Parameters
    ----------
    tv : ndarray
        Time vector.
    xv : ndarray
        Data vector.
    window : float, optional
        Time window for calculation. Default is 1e3.

    Returns
    -------
    ndarray
        Correlation coefficients for each window.
    """
    cs = []
    for i in range(int(tv[-1] / window)):
        cs.append(np.corrcoef(xv[(tv > (i * 1e3)) * (tv < (1e3 * (i + 1)))].T))
    return np.array(cs)


def ttest_correlation_strength(cs):
    """Perform a t-test on the strength of the correlation.

    Parameters
    ----------
    cs : ndarray
        Correlation coefficients.

    Returns
    -------
    ndarray
        P-values of the t-test for each correlation coefficient.
    """
    from scipy import stats

    cs_z = np.arctanh(cs)
    for i in range(cs.shape[1]):
        cs_z[:, i, i] = 0.0
    _, p = stats.ttest_1samp(cs, 0.0)
    return p


def compare_timeseries(exp: SimulationExperiment, ts1: Any, ts2: Any, atol: float = 1e-10):
    """Compare state variables between two time series using multiple measures.

    Args:
        exp: Experiment object containing metadata about state variables.
        ts1: First time series.
        ts2: Second time series.
        atol: Absolute tolerance for broader identity check.
    """
    for sv in exp.dynamics.state_variables.keys():
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
