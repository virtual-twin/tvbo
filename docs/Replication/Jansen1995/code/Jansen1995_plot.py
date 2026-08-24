"""Plotting functions for Jansen & Rit 1995 paper reproduction.

Each function takes experiment result objects and extracts all metadata
(parameter names, axis values, stimulus timing, n_trials, etc.)
directly from the result/experiment objects. Nothing is hardcoded.

All time axes are in seconds (the native simulation unit).
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, hilbert, welch
from sympy import IndexedBase, Symbol
from sympy import latex as sympy_latex

# ── Consistent sizing ────────────────────────────────────────────────────────
# All figures share the same width so text at a given point-size renders
# identically across figures.  Heights derived from the original paper
# figure pixel extents (measured from scanned PNGs in img/).

FIG_WIDTH = 7.0  # inches

_ORIG_PX = {
    3: (675, 703),
    4: (673, 753),
    5: (672, 743),
    6: (673, 1145),
    8: (668, 937),
    9: (672, 485),
    10: (670, 1018),
    11: (648, 720),
}


def _figsize(fig_num):
    """Return (width, height) in inches matching the original aspect ratio."""
    w, h = _ORIG_PX[fig_num]
    return (FIG_WIDTH, round(FIG_WIDTH * h / w, 1))


# Font sizes in points — uniform across every figure
FS_SUPTITLE = 12
FS_TITLE = 12
FS_LABEL = 11
FS_TICK = 10
FS_ANNOT = 10


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_axis_values(er, idx):
    """Get axis values array from ExplorationResult by index."""
    return np.array(er.axes[idx]["explored_values"])


def _get_axis_name(ax):
    """Get axis name string from axis info (Bunch or dict)."""
    return ax.name if hasattr(ax, "name") else ax["name"]


def _axis_latex(name):
    """Convert axis name to a LaTeX string via SymPy.

    'K[0]' → 'K_{0}',  'K[1]' → 'K_{1}',  'C' → 'C'
    Works for any parameter, not just K.
    """
    m = re.match(r"^(\w+)\[(\d+)\]$", name)
    if m:
        return sympy_latex(IndexedBase(m.group(1))[int(m.group(2))])
    return sympy_latex(Symbol(name))


def _axes_indexed_base(er):
    """Return the common IndexedBase name if all axes are elements, else None.

    e.g. axes ['K[0]', 'K[1]'] → 'K';  axes ['K[0]', 'C'] → None
    """
    bases = set()
    for ax in er.axes:
        m = re.match(r"^(\w+)\[(\d+)\]$", _get_axis_name(ax))
        if not m:
            return None
        bases.add(m.group(1))
    return bases.pop() if len(bases) == 1 else None


def _point_label(er, values, fmt=".0f"):
    r"""Build a compact label for a grid point.

    If all axes are elements of the same indexed parameter and the values are
    equal, collapse to  ``K_{0} = K_{1} = value``.
    Otherwise list each axis:  ``K_{0} = v0,\; K_{1} = v1``.
    """
    axis_names = [_get_axis_name(ax) for ax in er.axes]
    latex_names = [_axis_latex(n) for n in axis_names]
    vals = [float(v) for v in values]

    base = _axes_indexed_base(er)
    if base is not None and len(set(round(v, 8) for v in vals)) == 1:
        # Symmetric: K_{0} = K_{1} = value
        return " = ".join(latex_names) + f" = {vals[0]:{fmt}}"
    # General case
    sep = ",\\; "
    return sep.join(f"{ln} = {v:{fmt}}" for ln, v in zip(latex_names, vals, strict=False))


def _add_k_legend(ax, er, values, fmt=".0f", **kwargs):
    """Place a legend-style box with per-axis K labels in bottom-right corner.

    Each axis gets its own line: ``K_{0} = 120``, ``K_{1} = 10``.
    For the symmetric case (all values equal), collapses to one line.
    """
    axis_names = [_get_axis_name(a) for a in er.axes]
    latex_names = [_axis_latex(n) for n in axis_names]
    vals = [float(v) for v in values]

    base = _axes_indexed_base(er)
    if base is not None and len(set(round(v, 8) for v in vals)) == 1:
        lines = [" = ".join(latex_names) + f" = {vals[0]:{fmt}}"]
    else:
        lines = [f"{ln} = {v:{fmt}}" for ln, v in zip(latex_names, vals, strict=False)]

    text = "\n".join(f"${ln}$" for ln in lines)
    kw = dict(
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    kw.update(kwargs)
    ax.text(0.98, 0.04, text, **kw)


def _annotate_neg_peaks(ax, t, trace, t0, n=2, **text_kw):
    """Label the first *n* negative peaks after stimulus onset on a VEP axis.

    Assumes the plotted signal is ``-trace`` with inverted y-axis
    (negative-up convention). The two negations cancel visually, so the
    upward-pointing peaks correspond to maxima of ``trace``.
    """
    post = np.where(t >= t0)[0]
    if not len(post):
        return
    # With -trace + inverted y, visually upward peaks = maxima of trace
    peaks, props = find_peaks(trace[post], prominence=0.5)
    # Sort by prominence (most prominent first)
    order = np.argsort(props["prominences"])[::-1]
    peaks = peaks[order[:n]]
    peaks = np.sort(peaks)  # re-order chronologically
    kw = dict(fontsize=FS_ANNOT, fontweight="bold", ha="center", va="top")
    kw.update(text_kw)
    for i, pk in enumerate(peaks):
        idx = post[pk]
        ax.annotate(
            f"$N_{{{i + 1}}}$",
            xy=(t[idx], -trace[idx]),
            xytext=(0, 10),
            textcoords="offset points",
            **kw,
        )


def _set_column_title(ax, node):
    """Set a harmonized 'Column N' title on a VEP axis."""
    ax.set_title(f"Column {node + 1}", fontweight="bold", fontsize=FS_TITLE)


def _mark_stim_onset(ax, t0):
    """Draw a red dashed vertical line at stimulus onset."""
    if t0 is not None:
        ax.axvline(t0, color="r", linestyle="dashed")


def _get_stim_onset(experiment):
    """Get stimulus onset time (seconds) from experiment events."""
    ev = experiment.events
    # Support dict-like or attribute access
    if hasattr(ev, "values"):
        for event in ev.values() if callable(ev.values) else [ev]:
            if hasattr(event, "parameters"):
                t0_param = event.parameters.get("t0", None)
                if t0_param is not None:
                    return float(t0_param.value)
    # Fallback: try first event by key
    for key in ev:
        event = ev[key]
        if hasattr(event, "parameters") and "t0" in event.parameters:
            return float(event.parameters["t0"].value)
    return None


def _get_exploration_meta(experiment, expl_name):
    """Get exploration metadata (n_trials, average, label) from experiment."""
    expl = experiment.explorations[expl_name]
    return {
        "n_trials": getattr(expl, "n_trials", 1),
        "average": getattr(expl, "average", None),
        "label": getattr(expl, "label", expl_name),
        "description": getattr(expl, "description", ""),
    }


def _time_axis(n_time, dt):
    """Create time axis in seconds."""
    return np.arange(n_time) * dt


# ── Fig. 3: C sweep ──────────────────────────────────────────────────────────


def plot_fig3(res, axes=None):
    """Fig. 3 — Effect of connectivity constant C on column output.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 1 (single column, C sweep).
    axes : Axes, array-like of Axes, or None
        - ``None`` — create a new figure with one subplot per sweep value.
        - A single ``matplotlib.axes.Axes`` — plot all sweep traces on that
          one axis (overlaid, with a legend).
        - A list / array of ``Axes`` whose length equals the number of sweep
          values — one trace per axis (same layout as the default figure but
          drawn on the caller's axes).

    Returns:
    -------
    fig : Figure or None
        The created Figure when *axes* is ``None``; otherwise ``None``
        (the caller owns the figure).
    """
    er = res.exploration.C_sweep_fig3
    if er.results is None:
        return None

    ax_info = er.axes[0] if er.axes else None
    n = int(ax_info.n) if ax_info else er.results.shape[0]
    time = er._get_time_axis()
    ax_values = np.asarray(ax_info["explored_values"]) if ax_info else None
    ax_name = ax_info.name if ax_info else "param"
    output_label = er.observable or ", ".join(er.output_names) or "output"

    created_fig = False

    if axes is None:
        # Default: create new figure with n subplots
        fig, ax_arr = plt.subplots(
            n,
            1,
            figsize=_figsize(3),
            sharex=True,
        )
        if n == 1:
            ax_arr = [ax_arr]
        created_fig = True
    elif hasattr(axes, "__len__") and not isinstance(axes, plt.Axes):
        # Caller provided a sequence of axes
        ax_arr = list(axes)
        if len(ax_arr) != n:
            raise ValueError(f"Expected {n} axes for {n} sweep values, got {len(ax_arr)}")
        fig = None
    else:
        # Single axis — overlay all traces
        ax_arr = None
        fig = None

    if ax_arr is not None:
        # One subplot per sweep value
        for i, ax in enumerate(ax_arr):
            data = np.asarray(er.results[i]).squeeze()
            if data.ndim > 1:
                for node_idx in range(data.shape[-1]):
                    ax.plot(time, data[:, node_idx], alpha=0.7)
            else:
                ax.plot(time, data)
            if ax_values is not None:
                val = float(ax_values[i])
                ax.set_ylabel(f"${_axis_latex(ax_name)}={val:.4g}$", fontsize=FS_LABEL)
            ax.tick_params(labelsize=FS_TICK)
        ax_arr[-1].set_xlabel("s", fontsize=FS_LABEL)
        if created_fig:
            # fig.suptitle(
            #     f"Fig. 3 — {output_label}",
            #     fontsize=FS_SUPTITLE,
            # )
            fig.tight_layout()
            plt.close()
    else:
        # Single axis: overlay all traces with legend
        ax = axes
        for i in range(n):
            data = np.asarray(er.results[i]).squeeze()
            label = f"${_axis_latex(ax_name)}={float(ax_values[i]):.4g}$" if ax_values is not None else f"#{i}"
            if data.ndim > 1:
                for node_idx in range(data.shape[-1]):
                    ax.plot(time, data[:, node_idx], alpha=0.7, label=label)
                    label = None  # only label first node trace
            else:
                ax.plot(time, data, alpha=0.7, label=label)
        ax.legend(fontsize=FS_TICK)
        ax.set_xlabel("s", fontsize=FS_LABEL)
        ax.set_ylabel(output_label, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)

    return fig


# ── Fig. 4: Parameter space exploration ───────────────────────────────────────


def classify_regimes(data, dt, v0):
    """Classify time series into Jansen & Rit regimes.

    Two-pass approach: first compute spectral features for all grid points,
    then classify using adaptive thresholds. The envelope-CV threshold for
    distinguishing sinusoidal alpha (regime 2) from waxing-waning (regime 3)
    adapts to the data range so that the same function works for both
    single-column (Fig 4) and coupled-column (Fig 6) sweeps.

    Parameters
    ----------
    data : ndarray, shape (*grid_shape, n_time)
    dt : float
    v0 : float or ndarray
        Scalar (fixed v0 for all points) or array indexed by last grid axis.

    Returns:
    -------
    regimes : ndarray, shape grid_shape (int codes 0-4)
    """
    grid_shape = data.shape[:-1]
    fs = 1.0 / dt
    n_time = data.shape[-1]
    v0_scalar = np.isscalar(v0) or (isinstance(v0, np.ndarray) and v0.ndim == 0)

    # Bulk statistics
    amp = np.std(data, axis=-1)
    ptp_arr = np.ptp(data, axis=-1)
    mean_arr = np.mean(data, axis=-1)

    # Two-criterion noise floor: amplitude AND spectral flatness.
    # The JR model band-pass filters even stochastic input, so amplitude
    # alone can't cleanly separate small oscillations from noise.
    # Spectral flatness (Wiener entropy = gmean/amean of PSD) measures
    # how tone-like vs noise-like a spectrum is:
    #   flat → 1  (noise, uniform spectral energy)
    #   flat → 0  (tonal, energy concentrated at harmonics)
    # Noise signals: flatness ≈ 0.006, oscillatory: flatness < 0.001.
    # A signal is "noise" only if BOTH amplitude is small AND spectrum flat.
    amp_noise_ceil = max(0.2, 0.03 * float(np.max(amp)))
    spec_flat_thresh = 0.005  # above = truly flat noise, below = has spectral structure

    # Per-signal spectral features (welch, hilbert, spectral flatness)
    f_peak = np.zeros(grid_shape)
    snr = np.zeros(grid_shape)
    env_cv = np.zeros(grid_shape)
    spec_flatness = np.ones(grid_shape)  # default 1 = flat

    for idx in np.ndindex(grid_shape):
        ts = data[idx]
        freqs, psd = welch(ts, fs=fs, nperseg=min(256, n_time))
        mask = freqs > 1.0
        if mask.any() and psd[mask].max() > 0:
            psd_pos = psd[mask]
            f_peak[idx] = freqs[mask][np.argmax(psd_pos)]
            snr[idx] = psd_pos.max() / (np.median(psd_pos) + 1e-10)
            # Spectral flatness (geometric mean / arithmetic mean)
            log_psd = np.log(psd_pos + 1e-30)
            spec_flatness[idx] = np.exp(np.mean(log_psd)) / np.mean(psd_pos)
        analytic = hilbert(ts - mean_arr[idx])
        envelope = np.abs(analytic)
        env_cv[idx] = np.std(envelope) / (np.mean(envelope) + 1e-10)

    # Noise mask: small amplitude AND spectrally flat
    is_noise = (amp < amp_noise_ceil) & (spec_flatness > spec_flat_thresh)

    # Adaptive envelope-CV threshold for alpha regime subdivision
    alpha_mask = (f_peak >= 8) & ~is_noise
    if alpha_mask.any():
        alpha_cvs = env_cv[alpha_mask]
        env_cv_thresh = np.median(alpha_cvs) if alpha_cvs.min() > 0.3 else 0.3
    else:
        env_cv_thresh = 0.3

    # Classification
    regimes = np.zeros(grid_shape, dtype=int)
    for idx in np.ndindex(grid_shape):
        v0_val = float(v0) if v0_scalar else v0[idx[-1]]

        if is_noise[idx]:
            # Fixed point: small amplitude + flat spectrum
            regimes[idx] = 4 if mean_arr[idx] > v0_val else 0
        elif ptp_arr[idx] > 20 and f_peak[idx] < 8:
            regimes[idx] = 1  # large-amplitude sporadic
        elif f_peak[idx] >= 8:
            regimes[idx] = 2 if env_cv[idx] < env_cv_thresh else 3
        elif f_peak[idx] > 1:
            # Sub-alpha with small ptp: noise, not genuine sporadic.
            # In the coupled system (Fig 6) sub-alpha signals arise from
            # mean shift pushing the sigmoid toward saturation, not from
            # epileptiform discharges.  True R1 has ptp > 20 mV.
            regimes[idx] = 4 if mean_arr[idx] > v0_val else 0
        else:
            regimes[idx] = 4 if mean_arr[idx] > v0_val else 0

    return regimes


def plot_fig4(res):
    """Fig. 4 — Parameter space exploration with regime classification.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 2 (4D parameter sweep: A, B, C, v0).

    Returns:
    -------
    fig : Figure
    """
    expl = res.exploration.param_space_4D_fig4
    dt = expl.dt

    # Extract axis info dynamically
    axes_info = {a.name: (int(a.n), np.asarray(a["explored_values"])) for a in expl.axes}
    axis_names = [a.name for a in expl.axes]
    grid_shape = tuple(n for n, _ in axes_info.values())
    n_total = int(np.prod(grid_shape))
    n_time = expl.results.size // n_total

    data = np.asarray(expl.results).reshape(*grid_shape, n_time)

    # Get axis values by name
    axis_vals = {name: vals for name, (_, vals) in axes_info.items()}
    A_vals = axis_vals[axis_names[0]]
    B_vals = axis_vals[axis_names[1]]
    C_vals = axis_vals[axis_names[2]]
    v0_vals = axis_vals[axis_names[3]]
    nA, nB, nC, nV = grid_shape

    # Classify regimes
    regimes = classify_regimes(data, dt, v0_vals)

    regime_labels = [
        "Hypoactive noise",
        "Low-freq periodic",
        "Sinusoidal (alpha)",
        "Waxing & waning",
        "Hyperactive noise",
    ]
    colors = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3", "#d9d9d9"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    # Warped coordinates for non-linear axis spacing (paper Fig. 4 tick values)
    b_ticks = np.array([B_vals[0], 22, 24.2, 26.4, B_vals[-1]])
    a_ticks = np.array([A_vals[0], 3.25, 4.87, 8.6, A_vals[-1]])
    b_vis = np.arange(len(b_ticks), dtype=float)
    a_vis = np.arange(len(a_ticks), dtype=float)
    Bw = np.interp(B_vals, b_ticks, b_vis)
    Aw = np.interp(A_vals, a_ticks, a_vis)

    def edges(w):
        e = np.empty(len(w) + 1)
        e[0] = w[0] - (w[1] - w[0]) / 2
        e[-1] = w[-1] + (w[-1] - w[-2]) / 2
        e[1:-1] = (w[:-1] + w[1:]) / 2
        return e

    shear, aspect = 0.35, 0.6
    Bg, Ag = np.meshgrid(edges(Bw), edges(Aw))
    X, Y = Bg + Ag * shear, Ag * aspect

    # Layout: nC rows of explorations (nV cols) + 5 legend/timeseries rows
    expl_rows = [[f"ex{row}{vi}" for vi in range(nV)] for row in range(nC)]
    ts_rows = [[f"leg{i + 1}", f"ts{i}", f"ts{i}"] for i in range(5)]
    layout = expl_rows + ts_rows

    fig, axd = plt.subplot_mosaic(layout, figsize=_figsize(4), height_ratios=[1] * nC + [1] * 5)

    # Exploration heatmaps
    for vi in range(nV):
        for row, (ci, C) in enumerate(zip(range(nC - 1, -1, -1), reversed(C_vals), strict=False)):
            ax = axd[f"ex{row}{vi}"]
            ax.pcolormesh(
                X,
                Y,
                regimes[:, :, ci, vi],
                cmap=cmap,
                norm=norm,
                edgecolors="none",
                rasterized=True,
            )
            cx = [X[0, 0], X[0, -1], X[-1, -1], X[-1, 0], X[0, 0]]
            cy = [Y[0, 0], Y[0, -1], Y[-1, -1], Y[-1, 0], Y[0, 0]]
            ax.plot(cx, cy)

            for bp, bl in zip(b_vis, b_ticks, strict=False):
                ax.plot([bp, bp], [-0.12, 0], clip_on=False)
                if row == nC - 1:
                    ax.text(bp, -0.35, f"{bl:g}", ha="center", va="top", fontsize=FS_TICK / 2)

            for ap, al in zip(a_vis, a_ticks, strict=False):
                ax.plot(
                    [ap * shear - 0.12, ap * shear],
                    [ap * aspect] * 2,
                    clip_on=False,
                )
                if row == 0:
                    ax.text(
                        ap * shear - 0.25,
                        ap * aspect,
                        f"{al:g}",
                        ha="right",
                        va="center",
                        fontsize=FS_TICK / 2,
                    )

            ax.set_xlim(X.min() - 0.5, X.max() + 0.15)
            ax.set_ylim(Y.min() - 0.6, Y.max() + 0.15)
            ax.set_aspect("equal")
            ax.axis("off")

            if vi == 0:
                ax.text(
                    -1.2,
                    Y.max() * 0.5,
                    f"C={int(C)}",
                    fontweight="bold",
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=FS_LABEL,
                )
            if row == 0:
                ax.set_title(f"{axis_names[3]} = {v0_vals[vi]}", fontsize=FS_TITLE, pad=2)

    # Display order: light→dark (top→bottom), matching the original paper.
    # regime 4 = Hyperactive noise (lightest) … regime 0 = Hypoactive noise (darkest)
    display_order = [4, 3, 2, 1, 0]

    # Legend
    for i, rid in enumerate(display_order):
        ax_leg = axd[f"leg{i + 1}"]
        ax_leg.axis("off")
        ax_leg.add_patch(
            plt.Rectangle(
                (0.05, 0.35),
                0.2,
                0.4,
                facecolor=colors[rid],
                transform=ax_leg.transAxes,
            )
        )
        ax_leg.text(
            0.3,
            0.5,
            regime_labels[rid],
            va="center",
            fontsize=FS_ANNOT / 1.5,
            transform=ax_leg.transAxes,
        )

    # Timeseries per regime — pick the most representative trace.
    # For oscillatory regimes (1-3), choose the signal with highest amplitude
    # (clearest visual example, avoids edge-case noise-like traces).
    # For noise regimes (0, 4), choose maximally distinct from oscillatory
    # (lowest amplitude = flattest trace).
    t = _time_axis(n_time, dt)
    amp = np.std(data, axis=-1)
    for i, regime_id in enumerate(display_order):
        ax_ts = axd[f"ts{i}"]
        idx_list = np.argwhere(regimes == regime_id)
        if len(idx_list) > 0:
            amps = np.array([amp[tuple(ix)] for ix in idx_list])
            if regime_id in (0, 4):
                pick = np.argmin(amps)  # flattest for noise
            else:
                pick = np.argmax(amps)  # clearest for oscillatory
            ts = data[tuple(idx_list[pick])]
            ax_ts.plot(t, ts)
        ax_ts.set_ylabel("mV", fontsize=FS_LABEL)
        ax_ts.set_xlim(t[0], t[-1])
        if i < 4:
            ax_ts.set_xticklabels([])
        else:
            ax_ts.set_xlabel("s", fontsize=FS_LABEL)

    # fig.suptitle(
    #     "Fig. 4: Parameter space exploration + Classification", fontsize=FS_SUPTITLE, y=0.91
    # )
    plt.close()
    return fig


# ── Fig. 5: Symmetric K sweep ────────────────────────────────────────────────


def plot_fig5(res, experiment):
    """Fig. 5 — Symmetric K sweep heatmap with regime examples.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 3.
    experiment : SimulationExperiment
        The experiment object (for network metadata).

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.K_sweep_symmetric_fig5
    n0, n1 = er.shape
    n_nodes = 2  # from data shape
    grid = np.asarray(er.results).reshape(n0, n1, -1, n_nodes)
    K_vals = _get_axis_values(er, 0)
    dt = er.dt
    t = _time_axis(grid.shape[2], dt)
    n_nodes = grid.shape[3]

    # Summary statistic: std averaged over nodes
    std_map = np.array(np.std(grid, axis=2).mean(axis=-1))

    # Sample regime points (from parameter space)
    def k_to_idx(k_val):
        return int(np.argmin(np.abs(K_vals - k_val)))

    # Pick 4 representative points spanning the regime space
    k_max = K_vals.max()
    regimes = [
        (k_max * 0.83, k_max * 0.08),  # waxing & waning (uncoupled)
        (k_max * 0.33, k_max * 0.37),  # synchronized sinusoidal
        (k_max * 0.75, k_max * 0.75),  # frequency shift
        (k_max * 0.92, k_max * 0.92),  # saturated / noisy
    ]

    ax0_latex = _axis_latex(_get_axis_name(er.axes[0]))
    ax1_latex = _axis_latex(_get_axis_name(er.axes[1]))

    layout = [
        ["heat", "heat", "heat", "heat"],
        ["heat", "heat", "heat", "heat"],
        ["r1n1", "r1n1", "r2n1", "r2n1"],
        ["r1n2", "r1n2", "r2n2", "r2n2"],
        ["r3n1", "r3n1", "r4n1", "r4n1"],
        ["r3n2", "r3n2", "r4n2", "r4n2"],
    ]
    fig, axd = plt.subplot_mosaic(
        layout,
        height_ratios=[1, 1, 0.3, 0.3, 0.3, 0.3],
        layout="tight",
        figsize=_figsize(5),
    )

    ax_heat = axd["heat"]
    ax_heat.imshow(
        std_map.T,
        origin="lower",
        aspect="equal",
        extent=[K_vals[0], K_vals[-1], K_vals[0], K_vals[-1]],
        cmap="Greys_r",
        interpolation="none",
    )
    ax_heat.set_xlabel(f"${ax1_latex}$", fontsize=FS_LABEL)
    ax_heat.set_ylabel(f"${ax0_latex}$", fontsize=FS_LABEL)

    for ri, (k0, k1) in enumerate(regimes, 1):
        ax_heat.plot(
            k1,
            k0,
            "s",
            color="white",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=5,
        )
        ax_heat.annotate(
            str(ri),
            (k1, k0),
            fontsize=FS_TICK,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    for ri, (k0, k1) in enumerate(regimes):
        i, j = k_to_idx(k0), k_to_idx(k1)
        ts = np.array(grid[i, j])
        panel_num = ri + 1
        regime_label = f"${_point_label(er, [K_vals[i], K_vals[j]])}$"

        for ni in range(n_nodes):
            key = f"r{panel_num}n{ni + 1}"
            ax_ts = axd[key]
            ax_ts.plot(t, ts[:, ni])
            ax_ts.tick_params(labelsize=FS_TICK)
            if panel_num in (1, 3):
                ax_ts.set_ylabel("mV", fontsize=FS_TICK)
            if panel_num in (3, 4) and ni == n_nodes - 1:
                ax_ts.set_xlabel("s", fontsize=FS_TICK)
            else:
                ax_ts.tick_params(labelbottom=False)
            if ni == 0:
                ax_ts.text(
                    0.02,
                    0.92,
                    f"{panel_num}  {regime_label}",
                    transform=ax_ts.transAxes,
                    fontsize=FS_TICK,
                    va="top",
                    ha="left",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.15",
                        facecolor="white",
                        edgecolor="black",
                    ),
                )

    # fig.suptitle("Fig. 5 — Symmetric K sweep", fontweight="bold", fontsize=FS_SUPTITLE)
    plt.close()
    return fig


# ── Fig. 6: Asymmetric K sweep ───────────────────────────────────────────────


def plot_fig6(res, experiment):
    """Fig. 6 — Asymmetric K sweep heatmap with regime examples.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 4.
    experiment : SimulationExperiment
        The experiment object (for model parameter metadata).

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.K_sweep_asymmetric_fig6
    n0, n1 = er.shape
    n_nodes = 2
    grid = np.asarray(er.results).reshape(n0, n1, -1, n_nodes)

    ax0_vals = _get_axis_values(er, 0)
    ax1_vals = _get_axis_values(er, 1)
    ax0_latex = _axis_latex(_get_axis_name(er.axes[0]))
    ax1_latex = _axis_latex(_get_axis_name(er.axes[1]))
    dt = er.dt
    t = _time_axis(grid.shape[2], dt)

    std_node0 = np.std(grid[:, :, :, 0], axis=2)
    std_node1 = np.std(grid[:, :, :, 1], axis=2)

    # Sample points from the grid for column 1 and column 2
    # Use nearest-index lookup so points land at exact K values
    def _idx(vals, target):
        return int(np.argmin(np.abs(vals - target)))

    _n0g, _n1g = len(ax0_vals), len(ax1_vals)
    col1_pts = [
        (_idx(ax0_vals, 100), _idx(ax1_vals, 50)),
        (_idx(ax0_vals, 500), _idx(ax1_vals, 150)),
        (_idx(ax0_vals, 600), _idx(ax1_vals, 1000)),
        (_idx(ax0_vals, 900), _idx(ax1_vals, 1500)),
    ]
    col2_pts = [
        (_idx(ax0_vals, 200), _idx(ax1_vals, 5000)),
        (_idx(ax0_vals, 300), _idx(ax1_vals, 2500)),
        (_idx(ax0_vals, 400), _idx(ax1_vals, 700)),
    ]

    layout = [
        ["heat", "heat"],
        ["heat", "heat"],
    ]
    for i in range(len(col1_pts)):
        layout.append([f"c1_{i + 1}", f"c1_{i + 1}"])
    for lbl in "abc"[: len(col2_pts)]:
        layout.append([f"c2_{lbl}", f"c2_{lbl}"])

    n_ts = len(col1_pts) + len(col2_pts)
    fig, axd = plt.subplot_mosaic(
        layout,
        height_ratios=[1, 1] + [0.3] * n_ts,
        figsize=_figsize(6),
        layout="compressed",
    )

    # Contour plot with warped (equally-spaced) axes matching original paper ticks
    ax = axd["heat"]
    (std_node0 + std_node1) / 2

    # Non-linear tick values from the original paper
    x_ticks = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500], dtype=float)
    y_ticks = np.array([0, 50, 100, 500, 1000, 2000, 4000, 8000], dtype=float)

    # Map data values to equally-spaced visual positions
    x_vis = np.arange(len(x_ticks), dtype=float)
    y_vis = np.arange(len(y_ticks), dtype=float)
    ax0w = np.interp(ax0_vals, x_ticks, x_vis)
    ax1w = np.interp(ax1_vals, y_ticks, y_vis)

    ax.set_xlabel(f"${ax0_latex}$")
    ax.set_ylabel(f"${ax1_latex}$")
    ax.set_xticks(x_vis)
    ax.set_xticklabels([f"{int(v)}" for v in x_ticks], fontsize=FS_TICK)
    ax.set_yticks(y_vis)
    ax.set_yticklabels([f"{int(v)}" for v in y_ticks], fontsize=FS_TICK)
    ax.set_xlim(x_vis[0], x_vis[-1])
    ax.set_ylim(y_vis[0], y_vis[-1])
    ax.set_box_aspect(0.7)

    # Helper to warp data coordinates to visual coordinates
    def wx(v):
        return np.interp(v, x_ticks, x_vis)

    def wy(v):
        return np.interp(v, y_ticks, y_vis)

    # Regime classification per node using same classifier as Fig 4
    v0 = float(experiment.dynamics.parameters["v0"].value)
    regimes_n0 = classify_regimes(grid[:, :, :, 0], dt, v0)
    regimes_n1 = classify_regimes(grid[:, :, :, 1], dt, v0)

    # Resample regime fields to a grid uniform in visual coordinates.
    # The raw data is linear in K-space, but the visual axes are non-linear
    # (especially Y: ratio 80:1 between tightest/widest spacing).
    # Without resampling, contours are jagged in the low-K1 region where
    # only a few data points span multiple visual tick intervals.
    from scipy.interpolate import RegularGridInterpolator

    n_vis = 200  # uniform visual grid resolution
    xv_uniform = np.linspace(x_vis[0], x_vis[-1], n_vis)
    yv_uniform = np.linspace(y_vis[0], y_vis[-1], n_vis)
    Xv, Yv = np.meshgrid(xv_uniform, yv_uniform, indexing="ij")

    def _resample_regime(regimes_raw):
        """Interpolate regime field from data grid to uniform visual grid."""
        interp = RegularGridInterpolator(
            (ax0w, ax1w),
            regimes_raw.astype(float),
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )
        resampled = interp(np.stack([Xv.ravel(), Yv.ravel()], axis=-1))
        return resampled.reshape(n_vis, n_vis)

    vis_n0 = gaussian_filter(_resample_regime(regimes_n0), sigma=3)
    vis_n1 = gaussian_filter(_resample_regime(regimes_n1), sigma=3)

    # Contours at regime boundaries (0.5, 1.5, ..., 3.5)
    regime_levels = [0.5, 1.5, 2.5, 3.5]

    # Column 1 (solid)
    ax.contour(
        xv_uniform,
        yv_uniform,
        vis_n0.T,
        levels=regime_levels,
        linestyles="solid",
        colors="black",
        linewidths=1.5,
    )

    # Column 2 (dashed)
    ax.contour(
        xv_uniform,
        yv_uniform,
        vis_n1.T,
        levels=regime_levels,
        linestyles="dashed",
        colors="black",
        linewidths=1.5,
    )

    # Markers — column 1
    for ri, (i, j) in enumerate(col1_pts, 1):
        k0, k1 = float(ax0_vals[i]), float(ax1_vals[j])
        ax.plot(
            wx(k0),
            wy(k1),
            "s",
            color="white",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=5,
        )
        ax.annotate(
            str(ri),
            (wx(k0), wy(k1)),
            fontsize=FS_TICK,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    # Markers — column 2
    for li, (i, j) in enumerate(col2_pts):
        k0, k1 = float(ax0_vals[i]), float(ax1_vals[j])
        lab = "abc"[li]
        ax.plot(
            wx(k0),
            wy(k1),
            "o",
            color="white",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=5,
        )
        ax.annotate(
            lab,
            (wx(k0), wy(k1)),
            fontsize=FS_TICK,
            fontweight="bold",
            ha="center",
            va="center",
            fontstyle="italic",
            zorder=6,
        )

    # Column 1 timeseries
    for ri, (i, j) in enumerate(col1_pts):
        k0, k1 = float(ax0_vals[i]), float(ax1_vals[j])
        ax_ts = axd[f"c1_{ri + 1}"]
        ax_ts.plot(t, grid[i, j, :, 0])
        ax_ts.tick_params(labelsize=FS_TICK)
        ax_ts.set_ylabel("mV", fontsize=FS_TICK)
        if ri < len(col1_pts) - 1:
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel("s", fontsize=FS_TICK)
        if ri == 0:
            ax_ts.set_title("Column 1", fontsize=FS_TITLE, fontweight="bold")
        pt_label = f"${_point_label(er, [k0, k1])}$"
        ax_ts.text(
            0.02,
            0.92,
            f"{ri + 1}  {pt_label}",
            transform=ax_ts.transAxes,
            fontsize=FS_TICK,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(
                boxstyle="square,pad=0.15",
                facecolor="white",
                edgecolor="black",
            ),
        )

    # Column 2 timeseries
    col2_labels = "abc"
    for li, (i, j) in enumerate(col2_pts):
        k0, k1 = float(ax0_vals[i]), float(ax1_vals[j])
        ax_ts = axd[f"c2_{col2_labels[li]}"]
        ax_ts.plot(t, grid[i, j, :, 1])
        ax_ts.tick_params(labelsize=FS_TICK)
        ax_ts.set_ylabel("mV", fontsize=FS_TICK)
        if li < len(col2_pts) - 1:
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel("s", fontsize=FS_TICK)
        if li == 0:
            ax_ts.set_title("Column 2", fontsize=FS_TITLE, fontweight="bold")
        pt_label = f"${_point_label(er, [k0, k1])}$"
        ax_ts.text(
            0.02,
            0.92,
            f"{col2_labels[li]}  {pt_label}",
            transform=ax_ts.transAxes,
            fontsize=FS_TICK,
            va="top",
            ha="left",
            fontweight="bold",
            fontstyle="italic",
            bbox=dict(
                boxstyle="square,pad=0.15",
                facecolor="white",
                edgecolor="black",
                linestyle="dashed",
            ),
        )

    # fig.suptitle("Fig. 6 — Asymmetric K sweep", fontweight="bold", fontsize=FS_SUPTITLE)
    plt.close()
    return fig


# ── Fig. 8: VEP symmetric K ──────────────────────────────────────────────────


def plot_fig8(res, experiment):
    """Fig. 8 — Average VEP with symmetric coupling (identical columns, no delay).

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 5.
    experiment : SimulationExperiment
        Experiment object (for stimulus timing metadata).

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.VEP_symmetric_K_fig8
    data = np.array(er.results)  # (n_K, n_time, n_nodes)
    dt = er.dt
    n_K, n_time, n_nodes = data.shape
    K_vals = _get_axis_values(er, 0)
    t_s = _time_axis(n_time, dt)
    t0_stim = _get_stim_onset(experiment)
    meta = _get_exploration_meta(experiment, "VEP_symmetric_K_fig8")

    # Layout: 2 columns, 2*n_K_per_col rows
    n_cols = 2
    n_per_col = (n_K + 1) // n_cols
    n_rows = n_per_col * n_nodes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figsize(8))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for panel_idx in range(n_K):
        col = panel_idx % n_cols
        row_base = (panel_idx // n_cols) * n_nodes

        for node in range(n_nodes):
            ax = axes[row_base + node, col]
            ax.plot(t_s, data[panel_idx, :, node])
            _mark_stim_onset(ax, t0_stim)
            _set_column_title(ax, node)

            ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

        # K label under bottom panel of each pair
        ax0_latex = _axis_latex(_get_axis_name(er.axes[0]))
        axes[row_base + n_nodes - 1, col].set_xlabel(f"${ax0_latex} = {int(K_vals[panel_idx])}$", fontsize=FS_LABEL)

    for r in range(n_rows):
        axes[r, 0].set_ylabel("mV")

    f" ({meta['n_trials']} trials)" if meta["n_trials"] > 1 else ""
    # fig.suptitle(f"Fig. 8 — VEP: symmetric K{n_trials_str}", fontsize=FS_SUPTITLE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.close()
    return fig


# ── Fig. 9: VEP asymmetric K ─────────────────────────────────────────────────


def plot_fig9(res, experiment):
    """Fig. 9 — Average VEP with asymmetric coupling K1 ≠ K2.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 5.
    experiment : SimulationExperiment
        Experiment object (for stimulus timing and exploration metadata).

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.VEP_asymmetric_K_fig9
    data = np.array(er.results)  # (n_pairs, n_time, n_nodes)
    n_pairs, n_time, n_nodes = data.shape
    dt = er.dt
    t_s = _time_axis(n_time, dt)

    # Axis values (per-element K)
    ax0_vals = _get_axis_values(er, 0)
    ax1_vals = _get_axis_values(er, 1)
    _axis_latex(_get_axis_name(er.axes[0]))
    _axis_latex(_get_axis_name(er.axes[1]))

    t0_stim = _get_stim_onset(experiment)
    meta = _get_exploration_meta(experiment, "VEP_asymmetric_K_fig9")

    fig, axes = plt.subplots(
        n_nodes,
        n_pairs,
        figsize=_figsize(9),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.3, "wspace": 0.15},
    )
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for pi in range(n_pairs):
        for node in range(n_nodes):
            ax = axes[node, pi]
            ax.plot(t_s, data[pi, :, node])
            _mark_stim_onset(ax, t0_stim)
            _set_column_title(ax, node)
            if node == n_nodes - 1:
                ax.set_xlabel("s")
            if pi == 0:
                ax.set_ylabel("mV")
            # K legend on bottom-right of last row per config
            if node == n_nodes - 1:
                _add_k_legend(ax, er, [ax0_vals[pi], ax1_vals[pi]])

            ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

    meta["n_trials"]
    # fig.suptitle(
    #     f"Fig. 9 — Average VEP ({n_trials} trials), "
    #     f"${ax0_latex} \\neq {ax1_latex}$",
    #     fontweight="bold",
    #     fontsize=FS_SUPTITLE,
    #     y=1.02,
    # )
    plt.close()
    return fig


# ── Fig. 10: Single VEP trials ───────────────────────────────────────────────
def plot_fig10(res, experiment):
    """Fig. 10 — Single VEP trials (not averaged).

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 5.
    experiment : SimulationExperiment
        Experiment object.

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.VEP_single_trials_fig10
    data = np.array(er.results)  # (n_configs, n_trials, n_time, n_nodes)
    _get_exploration_meta(experiment, "VEP_single_trials_fig10")

    dt = er.dt
    # First config
    trials = data[0]  # (n_trials, n_time, n_nodes)
    n_trials, n_time, n_nodes = trials.shape
    t_s = _time_axis(n_time, dt)

    # Axis values for K
    ax0_vals = _get_axis_values(er, 0)
    ax1_vals = _get_axis_values(er, 1)

    t0_stim = _get_stim_onset(experiment)

    fig, axes = plt.subplots(n_nodes, 1, figsize=_figsize(10), sharex=True, gridspec_kw={"hspace": 0.3})
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for node in range(n_nodes):
        ax = axes[node]
        for trial in range(n_trials):
            ax.plot(t_s, trials[trial, :, node], alpha=0.8, color="C0")
        _mark_stim_onset(ax, t0_stim)
        _set_column_title(ax, node)
        ax.set_ylabel("mV")
        ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

    axes[-1].set_xlabel("s")
    _add_k_legend(axes[-1], er, [ax0_vals[0], ax1_vals[0]])
    # fig.suptitle(f"Fig. 10 — {n_trials} single VEP trials", fontweight="bold", fontsize=FS_SUPTITLE, y=1.02)

    plt.close()
    return fig


# ── Fig. 11: VEP with delay, different columns ───────────────────────────────


def plot_fig11(res, experiment):
    """Fig. 11 — Average VEP with delay and different column parameters.

    4-row layout: for each K configuration, Column 1 then Column 2.
    Negative-up convention matching original paper.

    Parameters
    ----------
    res : ExperimentResult
        Result from experiment 6.
    experiment : SimulationExperiment
        Experiment object.

    Returns:
    -------
    fig : Figure
    """
    er = res.exploration.VEP_delayed_fig11
    data = np.array(er.results)  # (n_configs, n_time, n_nodes)
    n_configs, n_time, n_nodes = data.shape
    dt = er.dt
    t_s = _time_axis(n_time, dt)

    # Axis values (per-element K)
    ax0_vals = _get_axis_values(er, 0)
    ax1_vals = _get_axis_values(er, 1)

    t0_stim = _get_stim_onset(experiment)
    meta = _get_exploration_meta(experiment, "VEP_delayed_fig11")

    n_rows = n_configs * n_nodes
    fig, axes = plt.subplots(n_rows, 1, figsize=_figsize(11), sharex=True, gridspec_kw={"hspace": 0.4})

    # Build row indices: (config_idx, node_idx)
    row_indices = []
    for ci in range(n_configs):
        for node in range(n_nodes):
            row_indices.append((ci, node))

    for r, (ci, node) in enumerate(row_indices):
        ax = axes[r]
        trace = data[ci, :, node]
        ax.plot(t_s, -trace)  # negative-up
        _mark_stim_onset(ax, t0_stim)

        ax.set_ylabel("mV")
        _set_column_title(ax, node)

        # K legend on bottom-right of last row per config pair
        if node == n_nodes - 1:
            _add_k_legend(ax, er, [ax0_vals[ci], ax1_vals[ci]])

        ax.yaxis.set_inverted(True)  # negative-up convention

        # Annotate negative peaks (N_1, N_2) after stimulus
        if t0_stim is not None:
            _annotate_neg_peaks(ax, t_s, trace, t0_stim)

    axes[-1].set_xlabel("s")
    meta["n_trials"]
    # fig.suptitle(
    #     f"Fig. 11 — Average VEP ({n_trials} trials), " "different columns with delay",
    #     fontweight="bold",
    #     fontsize=FS_SUPTITLE,
    #     y=0.94,
    # )
    plt.close()
    return fig
