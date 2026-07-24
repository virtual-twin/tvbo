#!/usr/bin/env python3
"""Compose side-by-side A/B images: A = the original paper figure on a light-grey
panel; B = the TVBO reproduction on white. Writes ``output/figures/ab_fig{N}.png``.

Reusable as-is across replications. Point ``_REPRO`` at each paper figure's
reproduction image (rendered from the study's ``figures:`` block by ``tvbo figure render``)
and drop the paper originals into ``original_study/img/fig{N}.png``. A/B composition is a
REPORT concern — this helper is called from the report's internal build (the ``ab()`` helper),
NOT from a plotting script; the composite embeds the © original, so it is local-only and
git-ignored, never in the public report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent           # <study>/  (code/figures/ -> <study>/)
FIGS = ROOT / "output" / "figures"  # generated figures live under the gitignored output/
FIGS.mkdir(parents=True, exist_ok=True)
ORIG = ROOT / "original_study" / "img"

# paper figure number -> its reproduction PNG under output/figures/ (edit for your study).
_REPRO = {
    1: FIGS / "fig01_topology.png",
    # 5: FIGS / "tvbo_fig5.png",
    # 8: FIGS / "tvbo_fig8.png",
}


def compose_ab(repro=None):
    repro = repro or _REPRO
    for n, rp in repro.items():
        op = ORIG / f"fig{n}.png"
        if not (op.exists() and rp.exists()):
            print(f"skip fig{n} (missing {'orig' if not op.exists() else 'repro'})")
            continue
        orig, rep = plt.imread(op), plt.imread(rp)
        ha, wa = orig.shape[:2]; hb, wb = rep.shape[:2]
        H = 1000.0
        wa2, wb2 = wa * H / ha, wb * H / hb
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=((wa2 + wb2) / 100.0 + 1.6, H / 100.0 + 0.9),
            gridspec_kw={"width_ratios": [wa2, wb2], "wspace": 0.34})
        for ax, img, title in [(a1, orig, "A  Original"), (a2, rep, "B  TVBO reproduction")]:
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.03, wspace=0.34)
        fig.canvas.draw()
        for ax, fc in [(a1, "#ececec"), (a2, "#ffffff")]:
            bb = ax.get_position()
            px, ptop, pbot = 0.014, 0.085, 0.02
            fig.add_artist(Rectangle(
                (bb.x0 - px, bb.y0 - pbot), bb.width + 2 * px, bb.height + ptop + pbot,
                transform=fig.transFigure, facecolor=fc, edgecolor="#9e9e9e",
                linewidth=1.3, zorder=-1))
        fig.patch.set_facecolor("white")
        fig.savefig(FIGS / f"ab_fig{n}.png", dpi=120, facecolor="white")
        plt.close(fig)
        print(f"ab_fig{n}.png")


if __name__ == "__main__":
    compose_ab()
