# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""A panel that draws no decorations takes the margin its neighbours reserve for theirs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tvbo.adapters.bsplot import _trim_to_content, expand_bare_axes


def _figure():
    """Two cells side by side: one carrying an axis label, one blank, drawn so the engine has settled."""
    fig, axd = plt.subplot_mosaic("ab", figsize=(4, 2), layout="constrained")
    axd["a"].plot([0, 1], [0, 1])
    axd["a"].set_xlabel("x label")
    axd["b"].axis("off")
    fig.canvas.draw()
    return fig, axd


def test_a_bare_panel_grows_and_a_decorated_one_does_not():
    fig, axd = _figure()
    before = {k: axd[k].get_position().size.copy() for k in axd}
    moved = expand_bare_axes(fig, [axd["b"]])
    assert moved == [axd["b"]]
    assert (axd["b"].get_position().size > before["b"] + 0.01).any(), "the bare panel must take the reserved margin"
    assert (abs(axd["a"].get_position().size - before["a"]) < 1e-3).all(), "a panel that was not asked must not move"
    plt.close(fig)


def test_a_bare_panel_stops_at_its_neighbours_ink():
    """It can never overlap: the decorated panel's tick labels and axis label bound the growth."""
    fig, axd = _figure()
    ink = axd["a"].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    expand_bare_axes(fig, [axd["b"]])
    assert axd["b"].get_position().x0 >= ink.x1, "grew over the neighbour it should have stopped at"
    plt.close(fig)


def test_a_neighbour_reaching_into_the_cell_is_not_stepped_over():
    """A neighbour whose own ink already hangs into this cell bounds the growth at the cell edge."""
    fig, axd = plt.subplot_mosaic("ab", figsize=(4, 2), layout="constrained")
    axd["a"].axis("off")
    axd["b"].plot([0, 1], [0, 1])
    axd["b"].text(-1.5, 0.5, "a label reaching left", transform=axd["b"].transAxes, ha="left")
    fig.canvas.draw()
    expand_bare_axes(fig, [axd["a"]])
    assert axd["a"].get_position().x1 <= axd["b"].get_position().x0, "grew across the neighbour it overlaps"
    plt.close(fig)


def test_a_grown_panel_gives_back_the_space_beyond_every_label():
    """The margin a row-spanning panel is handed for a label it does not draw is returned once the labels settle."""
    fig, axd = _figure()
    fig.set_layout_engine("none")
    inverse = fig.transFigure.inverted()
    floor = axd["a"].get_tightbbox(fig.canvas.get_renderer()).transformed(inverse).y0
    axd["b"].set_position([0.55, floor - 0.2, 0.4, 0.95])  # as the engine deals it, reaching past the label below its neighbour
    _trim_to_content(fig, [axd["b"]])
    assert axd["b"].get_position().y0 >= floor - 1e-6, "kept the margin it hangs below every label to hold"
    plt.close(fig)


def test_the_trim_never_pulls_a_panel_off_its_own_cell():
    """A side is given back only towards the panel's own middle: trimming past it would move the panel, not fit it."""
    fig, axd = _figure()
    fig.set_layout_engine("none")
    inverse = fig.transFigure.inverted()
    ink = axd["a"].get_tightbbox(fig.canvas.get_renderer()).transformed(inverse)
    axd["b"].set_position([ink.x1 - 0.25, 0.1, 0.55, 0.8])  # reaching back over the neighbour, so a trim to its ink would move the panel rather than fit it
    _trim_to_content(fig, [axd["b"]])
    assert axd["b"].get_position().x1 > ink.x1, "trimmed to a neighbour that ends before the panel begins"
    plt.close(fig)


def test_two_bare_panels_cannot_both_claim_one_gap():
    """They are grown one at a time against where the previous one now ends."""
    fig, axd = plt.subplot_mosaic("bc", figsize=(4, 2), layout="constrained")
    for key in axd:
        axd[key].axis("off")
    expand_bare_axes(fig, [axd["b"], axd["c"]])
    assert axd["b"].get_position().x1 <= axd["c"].get_position().x0, "the two panels overlap"
    plt.close(fig)
