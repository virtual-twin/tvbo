#
# Module: analysis.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Plotting helpers for analysing simulation output.

This module provides Matplotlib-based plotting utilities for visualising
derived signal analyses, such as power spectra with shaded canonical EEG
frequency bands.
"""

from typing import Any, Optional

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np


# TODO: bands, colors params. not used, remove?
def plot_power_spectrum(
    frequency: np.ndarray,
    power: np.ndarray,
    bands: Optional[dict] = None,
    colors: Any = None,
    ax: Any = None,
    label: str = "simulation",
):
    """Plot a log-log power spectrum with shaded canonical EEG frequency bands.

    Args:
        frequency: 1-D array of frequencies (Hz).
        power: 1-D array of spectral powers, aligned with *frequency*.
        bands: Optional override of the band dict (defaults to δ/θ/α/β/γ).
        colors: Optional override of the per-band background colors.
        ax: Existing `matplotlib.axes.Axes` to draw into; required.
        label: Legend label for the spectrum trace.
    """
    ax.plot(
        frequency,
        power,
        c="k",
        linewidth=1,
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum")

    bands = {
        r"$\delta$": (1, 4),
        r"$\theta$": (4, 8),
        r"$\alpha$": (8, 12),
        r"$\beta$": (12, 30),
        r"$\gamma$": (30, 100),
    }
    colors = colormaps["viridis"](np.linspace(0, 1, len(bands)))

    # Adding vertical lines and labels for different frequency bands
    for i, (band, (start, end)) in enumerate(bands.items()):
        mid_point = 10 ** (np.log10(start) + (np.log10(end) - np.log10(start)) / 2)  # Logarithmic midpoint
        ax.axvspan(start, end, color=colors[i], alpha=0.2)  # Colored background with alpha transparency
        ax.axvline(x=end, color=colors[i], linestyle="--")  # End of the band
        ax.text(
            mid_point,
            plt.ylim()[1] * 0.8,
            band,
            horizontalalignment="center",
            verticalalignment="top",
            color="k",
            fontsize=12,
            fontweight="bold",
        )
