"""Measure how a rendered figure differs in LAYOUT from a reference image.

A replication's figure is meant to land on the published one's layout: same aspect, same panel grid, panels in the same places at the same relative sizes. Judging that by eye does not scale and does not produce a number you can put in a report, so this module reduces both images to their panel geometry and reports the differences.

Panels are found by recursive XY-cut — the classic document-layout decomposition:
project the ink onto each axis, split at runs of blank, recurse. It needs no knowledge of either figure's provenance, which is the point: the reference is a bitmap from a PDF and ours comes from a mosaic spec, and they are compared on equal terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Box:
    """A content block in fractional page coordinates (0-1, origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        """Width as a fraction of the page."""
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        """Height as a fraction of the page."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Fraction of the page the block covers."""
        return self.w * self.h

    def iou(self, other: Box) -> float:
        """Intersection over union — 1.0 when the two blocks coincide exactly."""
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


def _ink(path: Path, threshold: float = 0.98) -> np.ndarray:
    """Boolean mask of non-background pixels, alpha-aware and background-agnostic."""
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))
    arr = np.asarray(img, dtype=float)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.max() > 1.0:
        arr = arr / 255.0
    alpha = arr[..., 3] if arr.shape[-1] == 4 else None
    rgb = arr[..., :3] if arr.shape[-1] >= 3 else arr[..., :1]
    # Compare against the modal corner colour rather than assuming white: a figure saved on a transparent or tinted canvas is otherwise all ink.
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = np.median(corners, axis=0)
    mask = np.abs(rgb - bg).max(axis=-1) > (1.0 - threshold)
    if alpha is not None:
        mask &= alpha > 0.1
    return mask


def _runs(profile: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Index spans of consecutive True, separated by gaps of at least *min_gap*."""
    idx = np.flatnonzero(profile)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > min_gap)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), (ends + 1).tolist(), strict=True))


def content_blocks(
    mask: np.ndarray,
    *,
    min_gap_frac: float = 0.012,
    min_size_frac: float = 0.03,
    depth: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Recursive XY-cut of an ink mask into content blocks, in pixel coordinates.

    Args:
        mask: Boolean ink mask, ``(rows, cols)``.
        min_gap_frac: Blank run, as a fraction of the page's smaller side, that counts
            as a gutter. Below this, adjacent panels merge into one block.
        min_size_frac: Blocks smaller than this fraction of the page in BOTH axes are
            dropped as decorations (tick labels, a stray legend).
        depth: Maximum alternating cut depth.

    Returns:
        ``(x0, y0, x1, y1)`` per block, in reading order.
    """
    h, w = mask.shape
    min_gap = max(2, int(round(min_gap_frac * min(h, w))))
    min_px = max(4, int(round(min_size_frac * min(h, w))))

    def cut(y0, y1, x0, x1, level, axis):
        sub = mask[y0:y1, x0:x1]
        if not sub.any():
            return []
        # Trim to the sub-block's own ink before deciding anything about it.
        rows, cols = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
        y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
        x0, x1 = x0 + cols[0], x0 + cols[-1] + 1
        sub = mask[y0:y1, x0:x1]

        if level >= depth:
            return [(x0, y0, x1, y1)]
        profile = sub.any(axis=1) if axis == 0 else sub.any(axis=0)
        spans = _runs(profile, min_gap)
        if len(spans) <= 1:
            # No cut on this axis; try the other one, then stop.
            if level + 1 >= depth:
                return [(x0, y0, x1, y1)]
            return cut(y0, y1, x0, x1, level + 1, 1 - axis)
        out = []
        for a, b in spans:
            if axis == 0:
                out += cut(y0 + a, y0 + b, x0, x1, level + 1, 1)
            else:
                out += cut(y0, y1, x0 + a, x0 + b, level + 1, 0)
        return out

    blocks = cut(0, h, 0, w, 0, 0)
    blocks = [b for b in blocks if (b[2] - b[0]) >= min_px and (b[3] - b[1]) >= min_px]
    return sorted(blocks, key=lambda b: (round(b[1] / max(1, min_gap)), b[0]))


def page_boxes(path: Path, **kwargs) -> tuple[list[Box], tuple[int, int]]:
    """Panel boxes of one image in fractional page coordinates, plus its pixel size."""
    mask = _ink(Path(path))
    h, w = mask.shape
    boxes = [Box(x0 / w, y0 / h, x1 / w, y1 / h) for x0, y0, x1, y1 in content_blocks(mask, **kwargs)]
    return boxes, (w, h)


def match_boxes(ours: list[Box], theirs: list[Box]) -> list[tuple[Box | None, Box | None]]:
    """Pair our panels with theirs by best overlap, leaving unmatched panels as ``None``.

    Greedy on IoU rather than an assignment solve: when the layouts already broadly agree the two are identical, and when they do not, a greedy pairing degrades into obvious ``None`` rows instead of an inscrutable global optimum.
    """
    pairs, used = [], set()
    for a in ours:
        best, best_iou = None, 0.0
        for j, b in enumerate(theirs):
            if j in used:
                continue
            score = a.iou(b)
            if score > best_iou:
                best, best_iou = j, score
        if best is None:
            pairs.append((a, None))
        else:
            used.add(best)
            pairs.append((a, theirs[best]))
    pairs += [(None, b) for j, b in enumerate(theirs) if j not in used]
    return pairs


def compare(ours: Path, theirs: Path, **kwargs) -> dict:
    """Layout comparison of two figure images.

    Returns:
        A dict with the page geometry of each, the matched panel pairs and their
        offsets in percent of page, and summary statistics (mean/max offset, IoU).
    """
    a_boxes, a_size = page_boxes(Path(ours), **kwargs)
    b_boxes, b_size = page_boxes(Path(theirs), **kwargs)
    pairs = match_boxes(a_boxes, b_boxes)

    rows = []
    for i, (a, b) in enumerate(pairs, start=1):
        if a is None or b is None:
            rows.append({"panel": i, "ours": a, "theirs": b, "iou": 0.0, "dx": None, "dy": None, "dw": None, "dh": None})
            continue
        rows.append(
            {
                "panel": i,
                "ours": a,
                "theirs": b,
                "iou": a.iou(b),
                "dx": 100.0 * (a.x0 - b.x0),
                "dy": 100.0 * (a.y0 - b.y0),
                "dw": 100.0 * (a.w - b.w),
                "dh": 100.0 * (a.h - b.h),
            }
        )

    matched = [r for r in rows if r["iou"] > 0]
    offsets = [max(abs(r["dx"]), abs(r["dy"])) for r in matched] or [float("nan")]
    return {
        "ours": {"path": str(ours), "size": a_size, "aspect": a_size[0] / a_size[1], "n_panels": len(a_boxes)},
        "theirs": {"path": str(theirs), "size": b_size, "aspect": b_size[0] / b_size[1], "n_panels": len(b_boxes)},
        "aspect_ratio_error": abs(a_size[0] / a_size[1] - b_size[0] / b_size[1]),
        "rows": rows,
        "n_matched": len(matched),
        "n_unmatched": len(rows) - len(matched),
        "mean_iou": float(np.mean([r["iou"] for r in matched])) if matched else 0.0,
        "mean_offset": float(np.mean(offsets)),
        "max_offset": float(np.max(offsets)),
    }


def report_table(result: dict) -> str:
    """The per-panel comparison as a markdown table, built through `md_table`."""
    from tvbo.utils.report import md_table

    rows = []
    for r in result["rows"]:
        if r["ours"] is None:
            rows.append([r["panel"], "—", "missing in ours", "", "", "", ""])
        elif r["theirs"] is None:
            rows.append([r["panel"], "extra in ours", "—", "", "", "", ""])
        else:
            a = r["ours"]
            rows.append(
                [
                    r["panel"],
                    f"{a.x0:.3f}, {a.y0:.3f}",
                    f"{a.w:.3f} x {a.h:.3f}",
                    f"{r['dx']:+.1f}",
                    f"{r['dy']:+.1f}",
                    f"{r['dw']:+.1f}",
                    f"{r['iou']:.3f}",
                ]
            )
    return md_table(
        ["Panel", "Our origin", "Our size", "dx %", "dy %", "dw %", "IoU"],
        rows,
        aligns=["r", "l", "l", "r", "r", "r", "r"],
    )


class Pane(NamedTuple):
    """One side of an A/B row.

    Args:
        images: The image to draw — a path, several paths stacked vertically (a paper that
            splits one quantity over separate scans), or None.
        title: Heading above the pane.
        fallback: Drawn in place of a missing image, so the pane still holds its slot in the
            layout instead of the row silently collapsing to one side.
    """

    images: Path | str | Sequence[Path | str] | None = None
    title: str = ""
    fallback: str = ""


_PLACEHOLDER_ASPECT = 1.4


def _as_rgb(array: np.ndarray) -> np.ndarray:
    """A greyscale scan as RGB — `imshow` would otherwise false-colour it through a colormap."""
    return np.repeat(array[..., None], 3, axis=-1) if array.ndim == 2 else array[..., :3]


def _pane_image(images) -> np.ndarray | None:
    """The pane's image: one scan, or several padded to a common width and stacked."""
    import matplotlib.image as mpimg

    if images is None:
        return None
    paths = [images] if isinstance(images, (str, Path)) else list(images)
    arrays = [_as_rgb(np.asarray(mpimg.imread(str(p)), dtype=float)) for p in paths if Path(p).is_file()]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    width = max(a.shape[1] for a in arrays)
    fill = max(a.max() for a in arrays)  # pad with the images' own white
    return np.concatenate([np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)), constant_values=fill) for a in arrays])


def image_row(panes: Sequence[Pane], width: float = 6.7, fontsize: float = 8):
    """A one-row figure holding *panes* at a COMMON height, widths following their aspect.

    Equal heights with aspect-proportional widths is what makes an A/B honest: neither side is stretched to match the other, and the row fills *width* inches exactly, so the pair lands on a report's text block without letterboxing. Returns ``(fig, axes)`` so a caller can annotate before saving.

    Built on `matplotlib.figure.Figure` rather than `pyplot`, so calling this from a notebook (a Quarto report is one) neither switches the global backend nor leaks a figure into pyplot's registry.

    ``fontsize=0`` drops the labels rather than drawing them at zero size, which FreeType rejects outright; it is how a caller measures the row's pure image geometry.
    """
    from matplotlib.figure import Figure

    arrays = [_pane_image(p.images) for p in panes]
    ratios = [a.shape[1] / a.shape[0] if a is not None else _PLACEHOLDER_ASPECT for a in arrays]
    fig = Figure(figsize=(width, width / sum(ratios)))
    axes = fig.subplots(1, len(panes), squeeze=False, gridspec_kw={"width_ratios": ratios})[0]
    for ax, array, pane in zip(axes, arrays, panes, strict=True):
        if array is not None:
            ax.imshow(array)
        elif fontsize:
            ax.text(0.5, 0.5, pane.fallback, ha="center", va="center", fontsize=fontsize)
        if fontsize:
            ax.set_title(pane.title, fontsize=fontsize)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0.02)
    return fig, axes


def _flatten_alpha(path) -> None:
    """Rewrite a raster without its alpha channel, composited on white.

    Matplotlib writes RGBA even when nothing is transparent, and a uniformly opaque alpha channel becomes a soft mask in the embedding PDF that some viewers mishandle, painting the page and every page after it black.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode not in ("RGBA", "LA", "PA"):
            return
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        dpi = im.info.get("dpi")
    flat.save(path, **({"dpi": dpi} if dpi else {}))


def side_by_side(panes: Sequence[Pane], outfile: Path, width: float = 6.7, fontsize: float = 6, dpi: int = 300) -> Path:
    """Write *panes* as one row at a common height — the A/B composite a report embeds.

    A replication report sets the published figure beside its reproduction. Composing that pair at render time, rather than shipping a rendered composite, is what keeps a copyrighted original out of every artifact but the one the caller names here.
    """
    fig, _ = image_row(panes, width, fontsize)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi)
    _flatten_alpha(outfile)
    return outfile


def overlay(result: dict, outfile: Path, titles: tuple[str, str] = ("ours", "reference")) -> Path:
    """Write a side-by-side of both images with their detected panels outlined.

    The numbers say how far off the layout is; this says *which* panel drifted.
    """
    from matplotlib.patches import Rectangle

    sides = ("ours", "theirs")
    panes = [
        Pane(result[s]["path"], f"{t} — {result[s]['size'][0]}x{result[s]['size'][1]}px, {result[s]['n_panels']} panels")
        for s, t in zip(sides, titles, strict=True)
    ]
    fig, axes = image_row(panes, width=14, fontsize=10)
    for ax, side, colour in zip(axes, sides, ("#d62728", "#1f77b4"), strict=True):
        w, h = result[side]["size"]
        for r in result["rows"]:
            box = r[side]
            if box is None:
                continue
            ax.add_patch(Rectangle((box.x0 * w, box.y0 * h), box.w * w, box.h * h, fill=False, lw=1.2, ec=colour))
            ax.text(box.x0 * w, box.y0 * h - 4, str(r["panel"]), color=colour, fontsize=8)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=110)
    _flatten_alpha(outfile)
    return outfile
