#
# Module: timeseries.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Time-series, EEG, and power-spectrum plotting for SimulationResult."""

import matplotlib.pyplot as plt
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────

def _prepare(data):
    """Squeeze singleton dims and average multi-node data → (time, variable)."""
    import xarray as xr

    time = data.coords['time'].values if 'time' in data.coords else np.arange(data.shape[0])
    if 'variable' in data.coords:
        var_names = list(np.atleast_1d(data.coords['variable'].values))
    else:
        var_names = None
    arr = data
    if 'node' in arr.dims:
        arr = arr.mean(dim='node')
    return time, var_names, arr


def _group_by_unit(var_names, units):
    """Group variable names by their unit string."""
    groups = {}
    for name in var_names:
        u = str(units.get(name, '')) or ''
        groups.setdefault(u, []).append(name)
    return groups


def _vals(arr, vname, n_time):
    return np.asarray(arr.sel(variable=vname)).reshape(n_time, -1)


# ── public ───────────────────────────────────────────────────────────

def plot_timeseries(result, ax=None, **kwargs):
    """Default timeseries line plot with per-unit subplots.

    Parameters
    ----------
    result : SimulationResult
        Must have ``.data`` (xr.DataArray) and ``._units`` dict.
    ax : matplotlib.axes.Axes, optional
        Target axes (single-panel only; ignored for multi-unit).
    **kwargs
        Forwarded to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    time, var_names, arr = _prepare(result.data)
    units = getattr(result, '_units', {})

    # Single or no variable
    if var_names is None or len(var_names) <= 1:
        created = ax is None
        if created:
            fig, ax = plt.subplots()
        label = var_names[0] if var_names else None
        ax.plot(time, np.asarray(arr).reshape(len(time), -1), label=label, **kwargs)
        ax.set_xlabel('time [ms]')
        if label:
            u = str(units.get(label, ''))
            ax.set_ylabel(f'{label} [{u}]' if u else label)
        if created:
            plt.close()
            return fig
        return None

    unit_groups = _group_by_unit(var_names, units)
    n_panels = len(unit_groups)

    if n_panels <= 1:
        created = ax is None
        if created:
            fig, ax = plt.subplots()
        for vname in var_names:
            ax.plot(time, _vals(arr, vname, len(time)), label=vname, **kwargs)
        ax.set_xlabel('time [ms]')
        unit_str = next(iter(unit_groups))
        ax.set_ylabel(unit_str if unit_str else ', '.join(var_names))
        ax.legend(fontsize='smaller')
        if created:
            plt.close()
            return fig
        return None

    # Multiple units → stacked subplots, shared x-axis
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(8, 2.5 * n_panels))
    if n_panels == 1:
        axes = [axes]
    for ax_i, (unit_str, vnames) in zip(axes, unit_groups.items()):
        for vname in vnames:
            ax_i.plot(time, _vals(arr, vname, len(time)), label=vname, **kwargs)
        ax_i.set_ylabel(unit_str if unit_str else ', '.join(vnames))
        ax_i.legend(fontsize='smaller')
    axes[-1].set_xlabel('time [ms]')
    fig.tight_layout()
    plt.close()
    return fig


def plot_eeg(result, VOI=None, mode=0, spacing=None, normalize=False,
             channel_labels=True, ax=None, linewidth=0.5, **kwargs):
    """EEG-like stacked-channel plot.

    Parameters
    ----------
    result : SimulationResult
        Must have ``.data`` (xr.DataArray with node dim).
    VOI : str, optional
        Variable of interest.  Defaults to first variable.
    mode : int
        Mode index.
    spacing : float, optional
        Vertical spacing; auto-computed from data if None.
    normalize : bool
        Z-score each channel before plotting.
    channel_labels : bool
        Show region labels on y-axis.
    ax : matplotlib.axes.Axes, optional
    linewidth : float
    """
    data = result.data
    var_names = list(np.atleast_1d(data.coords['variable'].values)) if 'variable' in data.coords else []

    if VOI is None:
        VOI = 'V' if 'V' in var_names else (var_names[0] if var_names else None)
    if VOI is not None and 'variable' in data.dims:
        data = data.sel(variable=VOI)

    # (time, node[, mode])
    arr = np.asarray(data)
    if arr.ndim >= 3:
        arr = arr[..., mode]   # drop mode
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    t = data.coords['time'].values if 'time' in data.coords else np.arange(arr.shape[0])

    if normalize:
        mu = arr.mean(axis=0, keepdims=True)
        sigma = arr.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        arr = (arr - mu) / sigma

    n_regions = arr.shape[1]

    if spacing is None:
        stds = np.std(arr, axis=0)
        base = float(np.median(stds))
        if not np.isfinite(base) or base == 0:
            base = float(np.median(np.max(np.abs(arr), axis=0)))
        if not np.isfinite(base) or base == 0:
            base = 1.0
        spacing = 2.5 * base

    node_labels = list(data.coords['node'].values) if 'node' in data.coords else [str(i) for i in range(n_regions)]

    created = ax is None
    if created:
        height = min(20.0, max(4.0, 0.22 * n_regions))
        fig, ax = plt.subplots(figsize=(10, height))
        max_len = max((len(str(l)) for l in node_labels), default=1)
        fig.subplots_adjust(left=min(0.5, max(0.1, 0.006 * max_len)))

    offsets = np.arange(n_regions) * spacing
    for i in range(n_regions):
        ax.plot(t, arr[:, i] + offsets[i], linewidth=linewidth, **kwargs)

    ax.set_xlabel('time [ms]')
    if channel_labels:
        ax.set_yticks(offsets)
        ax.set_yticklabels([str(l) for l in node_labels])
        ax.tick_params(axis='y', labelsize=8)
    else:
        ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    ax.set_title('EEG-like channels' + (f' — {VOI}' if VOI else ''))
    ax.grid(False)

    if created:
        plt.close()
        return fig
    return None


def plot_power_spectrum(result, VOI=None, ROI='mean', mode=0,
                        bands=None, ax=None, label='simulation', **kwargs):
    """FFT power spectrum with frequency-band annotations.

    Parameters
    ----------
    result : SimulationResult
    VOI : str, optional
        Variable of interest.
    ROI : str or int
        'mean' or region index.
    mode : int
    bands : dict, optional
    ax : matplotlib.axes.Axes, optional
    """
    from scipy.fft import fft, fftfreq
    from matplotlib import colormaps

    data = result.data
    var_names = list(np.atleast_1d(data.coords['variable'].values)) if 'variable' in data.coords else []

    if VOI is not None and 'variable' in data.dims:
        data = data.sel(variable=VOI)

    arr = np.asarray(data)
    time = data.coords['time'].values if 'time' in data.coords else np.arange(arr.shape[0])
    dt = float(time[1] - time[0]) / 1000  # ms → s

    n_samples = arr.shape[0]
    freq = fftfreq(n_samples, d=dt)[:n_samples // 2]
    power = np.abs(fft(arr, axis=0)[:n_samples // 2]) ** 2
    power /= power.sum(axis=0, keepdims=True)

    # Reduce to (freq, variable)
    if power.ndim >= 3:
        power = power[..., mode] if power.ndim == 3 else power[:, :, :, mode]
    if power.ndim >= 3:
        power = power.mean(axis=2) if ROI == 'mean' else power[:, :, ROI]

    created = ax is None
    if created:
        fig, ax = plt.subplots()

    if power.ndim == 1:
        ax.plot(freq, power, linewidth=1, label=VOI or label, **kwargs)
    else:
        for i in range(power.shape[1]):
            lbl = var_names[i] if i < len(var_names) else f'var {i}'
            ax.plot(freq, power[:, i], linewidth=1, label=lbl, **kwargs)
    ax.legend()
    ax.set_xlim([1, 150])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Power Spectrum')

    if bands is None:
        bands = {
            r'$\delta$': (1, 4), r'$\theta$': (4, 8), r'$\alpha$': (8, 12),
            r'$\beta$': (12, 30), r'$\gamma$': (30, 100),
        }
    colors = colormaps['viridis'](np.linspace(0, 1, len(bands)))
    ylim = ax.get_ylim()
    for i, (band, (lo, hi)) in enumerate(bands.items()):
        mid = 10 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) / 2)
        ax.axvspan(lo, hi, color=colors[i], alpha=0.1)
        ax.axvline(x=hi, color=colors[i], linestyle='--')
        ax.text(mid, ylim[1] * 0.8, band, ha='center', va='top',
                fontsize=12, fontweight='bold')

    if created:
        plt.close()
        return fig
    return None
