"""Test basic TimeSeries utilities: get_state_variable and calculate_frequency."""
import numpy as np
from tvbo.data.types import TimeSeries


def test_timeseries_get_state_variable_and_frequency():
    T = 200
    time = np.linspace(0, 199, T)  # ms
    # Create two sine waves with different frequencies
    f1 = 10  # Hz equivalent (simplified)
    f2 = 20
    data_var1 = np.sin(2 * np.pi * f1 * time / 1000)
    data_var2 = np.sin(2 * np.pi * f2 * time / 1000)
    data = np.stack([data_var1, data_var2], axis=1)  # (T, 2)
    data4d = data[:, :, None, None]  # (T, SV, Region=1, Mode=1)
    ts = TimeSeries(
        time=time,
        data=data4d,
        sample_period=1.0,
        labels_dimensions={"State Variable": ["V1", "V2"], "Region": ["R0"]},
    )

    v1_ts = ts.get_state_variable("V1")
    assert v1_ts.data.shape[1] == 1, "Filtered TimeSeries should have one state variable"

    freq_est = ts.calculate_frequency(state_variable="V1", region=0, mode=0)
    assert freq_est >= 0.0, "Dominant frequency should be non-negative"
