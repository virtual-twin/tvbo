"""Tests for `tvbo figure compare` — layout A/B against a reference image.

The verb exists so "the figure is the wrong shape" becomes a number instead of an
impression, which only works if the decomposition is trustworthy: it must find the panels
a reader would point at, pair them with the reference's, and not invent structure where
there is none.
"""

import numpy as np
import pytest

from tvbo.utils.figure_compare import (
    Box,
    Pane,
    compare,
    content_blocks,
    match_boxes,
    page_boxes,
    report_table,
    side_by_side,
)


def _mask(h, w, rects):
    """An ink mask with a filled rectangle per (y0, y1, x0, x1) in fractional coords."""
    m = np.zeros((h, w), dtype=bool)
    for y0, y1, x0, x1 in rects:
        m[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    return m


def _png(path, rects, size=(400, 600)):
    """Write a white page with black rectangles — a stand-in for a rendered figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    h, w = size
    img = np.ones((h, w, 3), dtype=float)
    img[_mask(h, w, rects)] = 0.0
    mpimg.imsave(str(path), img)
    return path


# ── Box geometry ────────────────────────────────────────────────────────────────────────


def test_iou_is_one_for_a_coincident_box():
    b = Box(0.1, 0.1, 0.4, 0.5)
    assert b.iou(b) == pytest.approx(1.0)


def test_iou_is_zero_for_disjoint_boxes():
    assert Box(0.0, 0.0, 0.2, 0.2).iou(Box(0.5, 0.5, 0.9, 0.9)) == 0.0


def test_iou_of_a_half_overlap():
    """Two unit-area boxes sharing half their extent: 0.5 / 1.5."""
    assert Box(0.0, 0.0, 0.4, 0.4).iou(Box(0.2, 0.0, 0.6, 0.4)) == pytest.approx(1 / 3)


# ── Decomposition ───────────────────────────────────────────────────────────────────────


def test_a_two_by_two_grid_is_found_in_reading_order():
    rects = [(0.05, 0.40, 0.05, 0.45), (0.05, 0.40, 0.55, 0.95), (0.60, 0.95, 0.05, 0.45), (0.60, 0.95, 0.55, 0.95)]
    blocks = content_blocks(_mask(400, 600, rects))
    assert len(blocks) == 4
    tops = [b[1] for b in blocks]
    assert tops[0] < tops[2] and tops[1] < tops[3]  # top row precedes bottom
    assert blocks[0][0] < blocks[1][0]  # left precedes right


def test_one_solid_block_stays_one_block():
    assert len(content_blocks(_mask(400, 600, [(0.1, 0.9, 0.1, 0.9)]))) == 1


def test_a_blank_page_yields_nothing():
    assert content_blocks(np.zeros((400, 600), dtype=bool)) == []


def test_panels_closer_than_the_gutter_are_not_split():
    """A published raster whose panels touch reads as one block — by design, not by bug."""
    touching = [(0.1, 0.9, 0.10, 0.49), (0.1, 0.9, 0.50, 0.90)]
    assert len(content_blocks(_mask(400, 600, touching), min_gap_frac=0.05)) == 1


def test_decorations_below_the_size_floor_are_dropped():
    rects = [(0.1, 0.9, 0.1, 0.9), (0.95, 0.96, 0.95, 0.96)]
    assert len(content_blocks(_mask(400, 600, rects))) == 1


# ── Matching ────────────────────────────────────────────────────────────────────────────


def test_boxes_pair_with_their_best_overlap():
    ours = [Box(0.0, 0.0, 0.4, 0.4), Box(0.6, 0.6, 1.0, 1.0)]
    theirs = [Box(0.62, 0.62, 1.0, 1.0), Box(0.02, 0.02, 0.4, 0.4)]
    pairs = match_boxes(ours, theirs)
    assert pairs[0][1] is theirs[1] and pairs[1][1] is theirs[0]


def test_an_unmatched_panel_on_either_side_is_reported_not_dropped():
    pairs = match_boxes([Box(0.0, 0.0, 0.2, 0.2)], [Box(0.8, 0.8, 1.0, 1.0)])
    assert (Box(0.0, 0.0, 0.2, 0.2), None) in pairs
    assert (None, Box(0.8, 0.8, 1.0, 1.0)) in pairs


# ── End to end ──────────────────────────────────────────────────────────────────────────


def test_a_figure_compared_against_itself_is_a_perfect_match(tmp_path):
    rects = [(0.05, 0.40, 0.05, 0.45), (0.60, 0.95, 0.55, 0.95)]
    p = _png(tmp_path / "fig.png", rects)
    result = compare(p, p)
    assert result["n_unmatched"] == 0
    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["max_offset"] == pytest.approx(0.0)
    assert result["aspect_ratio_error"] == pytest.approx(0.0)


def test_a_shifted_panel_shows_up_as_an_offset(tmp_path):
    a = _png(tmp_path / "a.png", [(0.10, 0.40, 0.10, 0.40)])
    b = _png(tmp_path / "b.png", [(0.20, 0.50, 0.10, 0.40)])
    rows = compare(a, b)["rows"]
    assert rows[0]["dy"] == pytest.approx(-10.0, abs=1.0)
    assert rows[0]["dx"] == pytest.approx(0.0, abs=1.0)


def test_a_different_page_shape_is_reported_independently_of_the_panels(tmp_path):
    """Aspect is the number to read first — it must not depend on the decomposition."""
    a = _png(tmp_path / "a.png", [(0.1, 0.9, 0.1, 0.9)], size=(400, 600))
    b = _png(tmp_path / "b.png", [(0.1, 0.9, 0.1, 0.9)], size=(600, 600))
    result = compare(a, b)
    assert result["ours"]["aspect"] == pytest.approx(1.5)
    assert result["theirs"]["aspect"] == pytest.approx(1.0)
    assert result["aspect_ratio_error"] == pytest.approx(0.5)


def test_the_panel_counts_of_both_sides_are_reported(tmp_path):
    a = _png(tmp_path / "a.png", [(0.05, 0.40, 0.05, 0.45), (0.60, 0.95, 0.05, 0.45)])
    b = _png(tmp_path / "b.png", [(0.05, 0.95, 0.05, 0.45)])
    result = compare(a, b)
    assert (result["ours"]["n_panels"], result["theirs"]["n_panels"]) == (2, 1)


def test_the_markdown_table_names_every_panel(tmp_path):
    a = _png(tmp_path / "a.png", [(0.05, 0.40, 0.05, 0.45), (0.60, 0.95, 0.05, 0.45)])
    table = report_table(compare(a, a))
    assert table.count("\n") >= 3  # header, rule, one row per panel
    assert "IoU" in table


# ── A/B composite ───────────────────────────────────────────────────────────────────────


def _aspect(path):
    import matplotlib.image as mpimg

    a = mpimg.imread(str(path))
    return a.shape[1] / a.shape[0]


def test_the_composite_is_as_wide_as_both_images_at_one_height(tmp_path):
    """Equal heights, aspect-proportional widths: the row's aspect is the sum of theirs."""
    a = _png(tmp_path / "a.png", [(0.1, 0.9, 0.1, 0.9)], size=(400, 600))  # 1.5
    b = _png(tmp_path / "b.png", [(0.1, 0.9, 0.1, 0.9)], size=(600, 600))  # 1.0
    out = side_by_side([Pane(a, "left"), Pane(b, "right")], tmp_path / "ab.png", fontsize=0)
    assert _aspect(out) == pytest.approx(1.5 + 1.0, rel=0.05)


def test_neither_side_is_stretched_to_match_the_other(tmp_path):
    """A tall figure beside a wide one must not be squared off — that would fake the A/B."""
    tall = _png(tmp_path / "tall.png", [(0.05, 0.95, 0.05, 0.95)], size=(900, 300))
    wide = _png(tmp_path / "wide.png", [(0.05, 0.95, 0.05, 0.95)], size=(300, 900))
    out = side_by_side([Pane(tall), Pane(wide)], tmp_path / "ab.png", fontsize=0)
    assert _aspect(out) == pytest.approx(1 / 3 + 3.0, rel=0.05)


def test_the_composite_is_written_where_it_was_asked_for(tmp_path):
    a = _png(tmp_path / "a.png", [(0.1, 0.9, 0.1, 0.9)])
    out = side_by_side([Pane(a), Pane(a)], tmp_path / "nested" / "ab.png")
    assert out.is_file() and out.parent.name == "nested"


def test_page_boxes_are_fractional_and_within_the_page(tmp_path):
    p = _png(tmp_path / "fig.png", [(0.10, 0.40, 0.10, 0.40)])
    boxes, (w, h) = page_boxes(p)
    assert (w, h) == (600, 400)
    for b in boxes:
        assert 0.0 <= b.x0 < b.x1 <= 1.0 and 0.0 <= b.y0 < b.y1 <= 1.0
