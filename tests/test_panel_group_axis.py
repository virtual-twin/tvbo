"""A categorical axis whose entries fall into named groups, declared not drawn.

A paper labels 47 task contrasts as seven families — one name per family, centred on its block, with a rule between blocks — not as 47 tick labels. The same shape recurs wherever a categorical axis has structure (ROIs by system, nodes by module, subjects by cohort), so it is an axis feature rather than something each bespoke composite panel redraws.

The convention worth pinning is what a `bound` IS. They are cumulative COUNTS, so the group sizes are readable straight off the declaration; an entry is drawn centred on its index, so the gap after count n sits at n - 0.5. Declaring plotted coordinates instead would put the half-cell shift in every recipe and get it wrong in some of them.
"""

from __future__ import annotations

import pytest

from tvbo.adapters.bsplot import _group_axis

BOUNDS = [3, 16, 19]
LABELS = ["Social", "Motor", "Gambling"]


def _spec(**kw):
    return {"ygroups": {"bounds": BOUNDS, **kw}}


def test_no_declaration_means_no_group_axis():
    assert _group_axis({}, "y") is None


def test_a_rule_is_drawn_between_blocks_and_not_at_the_axis_end():
    """The last bound is where the data stops; a rule there would double the frame."""
    g = _group_axis(_spec(), "y")
    assert g["rules"] == [2.5, 15.5]


def test_a_label_sits_at_the_centre_of_its_block():
    g = _group_axis(_spec(labels=LABELS), "y")
    assert [lbl["at"] for lbl in g["labels"]] == [1.0, 9.0, 17.0]
    assert [lbl["text"] for lbl in g["labels"]] == LABELS


def test_bounds_are_counts_so_a_group_of_three_spans_indices_zero_to_two():
    """Reading the declaration: 3 means three entries, and its gap lands after index 2."""
    g = _group_axis({"ygroups": {"bounds": [3, 6]}}, "y")
    assert g["rules"] == [2.5]
    assert _group_axis({"ygroups": {"bounds": [3, 6], "labels": ["a", "b"]}}, "y")["labels"][0]["at"] == 1.0


def test_the_half_cell_shift_is_overridable_for_an_edge_indexed_axis():
    g = _group_axis(_spec(edge_offset=0.0), "y")
    assert g["rules"] == [3.0, 16.0]


def test_a_bare_list_is_read_as_the_bounds():
    assert _group_axis({"ygroups": BOUNDS}, "y")["rules"] == [2.5, 15.5]


def test_labels_are_optional_so_the_rules_alone_can_repeat_across_cells():
    """The paper names the families once, on the leftmost matrix; the rest keep the rules."""
    g = _group_axis(_spec(), "y")
    assert g["labels"] == [] and g["rules"]


def test_a_declaration_without_bounds_says_what_is_missing():
    with pytest.raises(ValueError, match="`bounds:`"):
        _group_axis({"ygroups": {"labels": LABELS}}, "y")


def test_mismatched_bounds_and_labels_are_refused():
    """Silently zipping to the shorter list would drop a family with no sign of it."""
    with pytest.raises(ValueError, match="parallel"):
        _group_axis({"ygroups": {"bounds": BOUNDS, "labels": ["only one"]}}, "y")


@pytest.mark.parametrize("axis,ha,va", [("y", "right", "center"), ("x", "center", "top")])
def test_the_label_is_anchored_away_from_the_axis_it_annotates(axis, ha, va):
    g = _group_axis({f"{axis}groups": {"bounds": BOUNDS, "labels": LABELS}}, axis)
    assert g["kwargs"]["ha"] == ha and g["kwargs"]["va"] == va
    assert g["axis"] == axis


def test_both_axes_can_carry_groups_at_once():
    from pathlib import Path

    from tests.test_panel_kind_grid import _Panel
    from tvbo.adapters.bsplot import _resolve_drawable

    panel = _Panel(kind="heatmap", opts={"ygroups": {"bounds": BOUNDS}, "xgroups": {"bounds": [2, 4]}})
    got = _resolve_drawable(panel, "p", Path("."))
    assert [g["axis"] for g in got["groups"]] == ["x", "y"]


def test_a_panel_without_groups_carries_an_empty_list_not_none():
    """The template iterates it unconditionally."""
    from pathlib import Path

    from tests.test_panel_kind_grid import _Panel
    from tvbo.adapters.bsplot import _resolve_drawable

    assert _resolve_drawable(_Panel(kind="heatmap"), "p", Path("."))["groups"] == []
