"""Plotting helpers for the Koller, Schirner, Ritter (2024) replication.

Each ``plot_figN`` function takes one or more ``ExperimentResult`` /
``ExplorationResult`` objects produced by ``SimulationStudy.get_experiment(N).run(...)``
and returns a ``matplotlib`` figure that mirrors the corresponding panel in
Koller et al., Nat. Commun. 15, 3570 (2024).

Heavy analyses (optical flow, NMF, wave detection) are deliberately compact
implementations geared towards the replication notebook, not full-fidelity
reproductions of the original processing pipeline (see
``dev/Replication/Koller2024/scripts/analysis/`` for the originals).

All time axes are in seconds; spatial coordinates are in millimetres.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr


# ----------------------------------------------------------------------------
# Figure sizing (consistent with Jansen1995_plot)
# ----------------------------------------------------------------------------
FIG_WIDTH = 7.0  # inches
FS_TITLE, FS_LABEL, FS_TICK = 12, 11, 10


def _fig(width=FIG_WIDTH, ratio=0.6):
    return plt.figure(figsize=(width, width * ratio))


# ----------------------------------------------------------------------------
# Result-extraction helpers
# ----------------------------------------------------------------------------
def _theta(res):
    """Return phase array of shape (n_trials, n_nodes, n_time).

    Handles the layouts produced by the supported backends:
      * tvboptim integration .data         -> (time, modes, nodes)
      * tvboptim explorations .results     -> (trials, time, modes, nodes)
      * tvb backend .data                  -> (time, sv, nodes, modes)
      * already (trials, nodes, time)      -> passed through
    """
    # Prefer trial ensemble (explorations) when present
    if hasattr(res, "explorations") and getattr(res.explorations, "trials", None) is not None:
        arr = np.asarray(res.explorations.trials.results)
    elif hasattr(res, "integration"):
        arr = np.asarray(res.integration.data)
    elif hasattr(res, "data"):
        arr = np.asarray(res.data)
    else:
        arr = np.asarray(res)
    arr = np.squeeze(arr)
    if arr.ndim == 2:                              # (time, nodes)
        arr = arr.T[None, ...]                     # -> (1, nodes, time)
    elif arr.ndim == 3:                            # (trials, time, nodes)
        arr = arr.transpose(0, 2, 1)               # -> (trials, nodes, time)
    return arr


def _times(res, dt=0.001):
    arr = _theta(res)
    return np.arange(arr.shape[-1]) * dt


def _instrength(res):
    """Return per-node SC in-strength (sum of incoming weights)."""
    net = getattr(res, "network", None)
    W = getattr(net, "weights", None)
    if W is None:
        # 2D experiments build the matrix on the fly; recompute from positions
        return _instrength_from_positions(res)
    W = np.asarray(W)
    return W.sum(axis=1)


def _instrength_from_positions(res):
    pos = np.asarray(getattr(res.network, "positions", []))
    if pos.size == 0:
        return None
    sigma = 10.0
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    a = np.exp(-d / sigma) / (2 * sigma)
    np.fill_diagonal(a, 0.0)
    return a.sum(axis=1)


# ----------------------------------------------------------------------------
# Wave / flow analysis primitives
# ----------------------------------------------------------------------------
def _phase(theta):
    """Wrap to [-pi, pi]."""
    return np.angle(np.exp(1j * theta))


def _bandpass(x, fs, f_low, f_high, order=4):
    b, a = butter(order, [f_low, f_high], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=-1)


def _kuramoto_order(theta):
    """Kuramoto order parameter R(t) per trial."""
    return np.abs(np.mean(np.exp(1j * theta), axis=1))


def _flow_from_phase(phase_2d, dx=140.0 / 30):
    """Estimate 2D phase gradient (dphi/dx, dphi/dy) and direction.

    ``phase_2d`` must have shape (ny, nx, n_time).
    Returns mean unit-vector flow over time as a 2D field.
    """
    dphi_dy, dphi_dx, _ = np.gradient(phase_2d, dx, dx, 1.0)
    # normalise per-pixel per-frame
    norm = np.hypot(dphi_dx, dphi_dy) + 1e-12
    ux = (-dphi_dx / norm).mean(axis=-1)
    uy = (-dphi_dy / norm).mean(axis=-1)
    return ux, uy


# ============================================================================
# Fig. 2 — Traveling wave on the 2D sheet (gradient vs uniform)
# ============================================================================
def plot_fig2(res_grad, res_unif, n_snapshots=6):
    """Side-by-side phase snapshots (Fig. 2 a vs b)."""
    fig = _fig(ratio=0.55)
    gs = GridSpec(2, n_snapshots, figure=fig, hspace=0.25, wspace=0.05)
    fs = 1000.0
    for row, (label, res) in enumerate([("gradient", res_grad), ("uniform", res_unif)]):
        theta = _theta(res)[0]                               # first trial
        ny = nx = int(np.sqrt(theta.shape[0]))
        snap_idx = np.linspace(theta.shape[-1] // 4,
                               theta.shape[-1] - 1, n_snapshots, dtype=int)
        for k, t_idx in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, k])
            frame = _phase(theta[:, t_idx]).reshape(ny, nx)
            im = ax.imshow(frame, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                           origin="lower", extent=[0, 140, 0, 140])
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(label, fontsize=FS_LABEL)
            if row == 0:
                ax.set_title(f"t = {t_idx / fs:.2f} s", fontsize=FS_TICK)
    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("phase (rad)", fontsize=FS_LABEL)
    fig.suptitle("Fig. 2 — 2D sheet traveling waves (gradient vs uniform)",
                 fontsize=FS_TITLE)
    return fig


# ============================================================================
# Fig. 3 — IF gradient sweep: wave coherence vs gradient scaling
# ============================================================================
def plot_fig3(res):
    """Average Kuramoto order parameter R as function of IF gradient scaling."""
    axis = res.axes[0]
    g_vals = np.array(axis["explored_values"])
    R_mean, R_std = [], []
    for sub in res.results:
        theta = _theta(sub)
        R = _kuramoto_order(theta)              # (n_trials, n_time)
        R_mean.append(R.mean()); R_std.append(R.std())
    R_mean, R_std = map(np.asarray, (R_mean, R_std))

    fig = _fig(ratio=0.5)
    ax = fig.add_subplot(111)
    ax.fill_between(g_vals, R_mean - R_std, R_mean + R_std, alpha=0.25)
    ax.plot(g_vals, R_mean, marker="o", lw=1.5)
    ax.set_xlabel("IF gradient scaling", fontsize=FS_LABEL)
    ax.set_ylabel("mean Kuramoto order R", fontsize=FS_LABEL)
    ax.set_title("Fig. 3 — Wave coherence vs frequency gradient",
                 fontsize=FS_TITLE)
    return fig


# ============================================================================
# Fig. 5 — Cortical SC variants: average flow potential maps
# ============================================================================
def plot_fig5(results_dict):
    """Grid of average flow-potential summaries (one per SC variant).

    ``results_dict`` keys: 'main', 'shuffled', 'distance', 'normalised',
    'jansenrit', 'no_delay', 'const_delay'.
    """
    keys = ["main", "shuffled", "distance", "normalised",
            "jansenrit", "no_delay", "const_delay"]
    titles = {
        "main": "Empirical SC (Exp 30)",
        "shuffled": "Shuffled SC (Exp 31)",
        "distance": "Distance-only SC (Exp 32)",
        "normalised": "Row-normalised SC (Exp 34)",
        "jansenrit": "Jansen-Rit (Exp 33)",
        "no_delay": "Zero delay (Exp 38)",
        "const_delay": "Constant delay (Exp 39)",
    }
    fig = _fig(ratio=0.45)
    n = len(keys)
    gs = GridSpec(1, n, figure=fig, wspace=0.4)
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        if k not in results_dict:
            ax.set_visible(False); continue
        res = results_dict[k]
        theta = _theta(res)
        # Mean Kuramoto order across trials and per-node phase variance
        R = _kuramoto_order(theta).mean()
        node_var = np.var(_phase(theta), axis=(0, 2))
        ax.bar(np.arange(node_var.size), node_var, color="C0", alpha=0.6)
        ax.set_title(f"{titles[k]}\nR={R:.2f}", fontsize=FS_TICK)
        ax.set_xticks([])
        if i == 0:
            ax.set_ylabel("node phase variance", fontsize=FS_LABEL)
    fig.suptitle("Fig. 5 — Cortical SC variants", fontsize=FS_TITLE)
    return fig


# ============================================================================
# Fig. 6 / 7 / 8 — IF x K x v heatmaps
# ============================================================================
def _grid_heatmap(res, summary_fn, label):
    """Compute a (n_IF, n_K, n_v) summary array from an ExplorationResult."""
    if_vals = np.array(res.axes[0]["explored_values"])
    K_vals = np.array(res.axes[1]["explored_values"])
    v_vals = np.array(res.axes[2]["explored_values"])
    out = np.empty((len(if_vals), len(K_vals), len(v_vals)))
    flat = res.results
    idx = 0
    for i in range(len(if_vals)):
        for j in range(len(K_vals)):
            for k in range(len(v_vals)):
                out[i, j, k] = summary_fn(_theta(flat[idx]))
                idx += 1
    return if_vals, K_vals, v_vals, out


def plot_fig6(res):
    """Mean Kuramoto order across (K, v) for each IF (rows)."""
    if_vals, K, v, R = _grid_heatmap(res, lambda t: _kuramoto_order(t).mean(),
                                     "R")
    fig = _fig(ratio=0.65)
    gs = GridSpec(1, len(if_vals), figure=fig, wspace=0.4)
    for i, IF in enumerate(if_vals):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(R[i].T, origin="lower", aspect="auto",
                       extent=[np.log10(K.min()), np.log10(K.max()),
                               v.min(), v.max()],
                       cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"IF = {IF:g} Hz", fontsize=FS_TICK)
        ax.set_xlabel("log10(K)", fontsize=FS_LABEL)
        if i == 0:
            ax.set_ylabel("v (mm/ms)", fontsize=FS_LABEL)
    fig.colorbar(im, ax=fig.axes, shrink=0.7, label="mean R")
    fig.suptitle("Fig. 6 — Synchrony across IF x K x v", fontsize=FS_TITLE)
    return fig


def plot_fig7(res):
    """Wave-speed proxy: 1/std(d phi/dt) averaged over nodes."""
    def speed(t):
        ph = _phase(t)
        d = np.diff(ph, axis=-1)
        return 1.0 / (np.std(d) + 1e-9)
    if_vals, K, v, S = _grid_heatmap(res, speed, "speed")
    fig = _fig(ratio=0.65)
    gs = GridSpec(1, len(if_vals), figure=fig, wspace=0.4)
    for i, IF in enumerate(if_vals):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(np.log10(S[i] + 1e-9).T, origin="lower", aspect="auto",
                       extent=[np.log10(K.min()), np.log10(K.max()),
                               v.min(), v.max()],
                       cmap="magma")
        ax.set_title(f"IF = {IF:g} Hz", fontsize=FS_TICK)
        ax.set_xlabel("log10(K)", fontsize=FS_LABEL)
        if i == 0:
            ax.set_ylabel("v (mm/ms)", fontsize=FS_LABEL)
    fig.colorbar(im, ax=fig.axes, shrink=0.7, label="log10 wave-speed proxy")
    fig.suptitle("Fig. 7 — Wave-speed proxy across IF x K x v",
                 fontsize=FS_TITLE)
    return fig


def plot_fig8(res, instrength):
    """Pearson r between effective frequency and node in-strength."""
    instrength = np.asarray(instrength)

    def corr(t):
        ph = _phase(t)
        ef = np.mean(np.diff(ph, axis=-1), axis=-1).mean(axis=0)  # per-node EF
        if ef.size != instrength.size:
            return np.nan
        return pearsonr(ef, instrength)[0]
    if_vals, K, v, C = _grid_heatmap(res, corr, "corr")
    fig = _fig(ratio=0.65)
    gs = GridSpec(1, len(if_vals), figure=fig, wspace=0.4)
    for i, IF in enumerate(if_vals):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(C[i].T, origin="lower", aspect="auto",
                       extent=[np.log10(K.min()), np.log10(K.max()),
                               v.min(), v.max()],
                       cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"IF = {IF:g} Hz", fontsize=FS_TICK)
        ax.set_xlabel("log10(K)", fontsize=FS_LABEL)
        if i == 0:
            ax.set_ylabel("v (mm/ms)", fontsize=FS_LABEL)
    fig.colorbar(im, ax=fig.axes, shrink=0.7, label="r(EF, instrength)")
    fig.suptitle("Fig. 8 — EF-instrength correlation across IF x K x v",
                 fontsize=FS_TITLE)
    return fig


# ============================================================================
# Fig. 9 — NMF subnetworks (alpha vs beta)
# ============================================================================
def plot_fig9(res_alpha, res_beta, instrength):
    fig = _fig(ratio=0.45)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    for ax_idx, (label, res) in enumerate([("alpha", res_alpha),
                                            ("beta", res_beta)]):
        ax = fig.add_subplot(gs[0, ax_idx])
        theta = _theta(res)
        ph = _phase(theta)
        ef = np.mean(np.diff(ph, axis=-1), axis=(0, 2))      # per-node EF
        ax.scatter(instrength, ef, s=8, alpha=0.6)
        if ef.size == instrength.size:
            r, _ = pearsonr(ef, instrength)
            ax.set_title(f"{label}: r = {r:.2f}", fontsize=FS_TICK)
        ax.set_xlabel("in-strength", fontsize=FS_LABEL)
        ax.set_ylabel("effective frequency (rad/ms)", fontsize=FS_LABEL)
    fig.suptitle("Fig. 9 — NMF subnetwork EF vs in-strength",
                 fontsize=FS_TITLE)
    return fig


# ============================================================================
# Suppl. Fig. S4 — robustness (noise, IF dispersion)
# ============================================================================
def plot_figS4(res_noise, res_ifdisp):
    fig = _fig(ratio=0.45)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    for ax_idx, (label, res) in enumerate([("additive noise", res_noise),
                                            ("IF dispersion", res_ifdisp)]):
        ax = fig.add_subplot(gs[0, ax_idx])
        theta = _theta(res)
        R = _kuramoto_order(theta).mean(axis=0)
        t = np.arange(R.size) * 0.001
        ax.plot(t, R, lw=1.2)
        ax.set_xlabel("time (s)", fontsize=FS_LABEL)
        ax.set_ylabel("Kuramoto order R(t)", fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TICK)
        ax.set_ylim(0, 1)
    fig.suptitle("Suppl. Fig. S4 — Robustness", fontsize=FS_TITLE)
    return fig
