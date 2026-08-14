"""`kind: grid` is what a paper's lettered panel usually is, and it needs no study code.

Almost every composite panel in a neuroscience figure is a grid of the same kind of cell labelled once per row and column — a row of mode surfaces under a single (a), a 7x4 Data/Reconstruction matrix whose task name is written once at the left. Declaring each cell as its own mosaic entry repeats those labels in every cell and renumbers panels the paper letters once, so studies wrote a bespoke composite drawer instead; this kind is that drawer, generalised.

A grid cell IS an inset — the same class, resolved by the same function — and the tests below pin that: the geometry the grid computes, the labels it writes once, and the fact that a cell draws exactly as the same kind draws in a mosaic slot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tvbo.adapters.bsplot import _grid_geometry, _resolve_drawable


class _Opt:
    def __init__(self, value):
        self.value = value


class _Cell:
    def __init__(self, kind=None, opts=None, layers=None, render=None):
        self.kind = kind
        self.opts = {k: _Opt(v) for k, v in (opts or {}).items()}
        self.layers = list(layers or [])
        self.render = render
        self.path = self.label = self.annotations = self.legend = None
        self.bounds = self.cell = self.cells = None


class _Panel(_Cell):
    def __init__(self, kind="grid", opts=None, layers=None, cell=None, cells=None):
        super().__init__(kind=kind, opts=opts, layers=layers)
        self.cell = cell
        self.cells = cells
        self.placeholder = self.number = self.number_loc = self.insets = None


class _Layer:
    """The minimum a layer resolver reads; the container path is never opened here."""

    def __init__(self, output="x"):
        self.used = type(
            "U", (), {"iri": f"file:{output}.h5", "output": output, "sel": None, "experiment": None, "analysis": None}
        )()
        self.encoding = self.mark = self.style = self.triangle = None
        self.transform = self.label = None


def _surface_cell(**opts):
    return _Cell(kind="surface", opts={"mesh": "m.npz", **opts})


# ------------------------------------------------------------------ geometry


def test_cells_tile_the_panel_top_left_to_bottom_right():
    boxes, _ = _grid_geometry({"nrows": 2, "ncols": 2, "wspace": 0.0, "hspace": 0.0}, 4)
    assert [b[:2] for b in boxes] == [[0.0, 0.5], [0.5, 0.5], [0.0, 0.0], [0.5, 0.0]]
    assert all(b[2:] == [0.5, 0.5] for b in boxes)


def test_spacing_is_taken_out_of_every_cell_not_the_panel():
    """A gap that shrank the grid instead of the cells would misalign it with its neighbours."""
    boxes, _ = _grid_geometry({"nrows": 1, "ncols": 2, "wspace": 0.1, "hspace": 0.0}, 2)
    assert boxes[0][0] == pytest.approx(0.05) and boxes[0][2] == pytest.approx(0.4)
    assert boxes[1][0] == pytest.approx(0.55)


def test_a_cell_past_the_declared_rows_is_dropped():
    boxes, _ = _grid_geometry({"nrows": 1, "ncols": 2}, 4)
    assert len(boxes) == 2


def test_label_strips_are_reserved_only_when_there_are_labels():
    plain, _ = _grid_geometry({"nrows": 1, "ncols": 1}, 1)
    labelled, _ = _grid_geometry({"nrows": 1, "ncols": 1, "row_labels": ["a"], "col_labels": ["b"]}, 1)
    assert plain[0][0] == pytest.approx(0.01) and plain[0][3] == pytest.approx(0.98)
    assert labelled[0][0] > plain[0][0] and labelled[0][3] < plain[0][3]


def test_each_row_and_column_is_labelled_once():
    _, labels = _grid_geometry({"nrows": 2, "ncols": 3, "row_labels": ["r0", "r1"], "col_labels": ["c0", "c1", "c2"]}, 6)
    assert [lbl["text"] for lbl in labels] == ["c0", "c1", "c2", "r0", "r1"]
    assert all(lbl["kwargs"]["va"] == "bottom" for lbl in labels[:3])


def test_surplus_labels_are_ignored_rather_than_drawn_off_the_grid():
    _, labels = _grid_geometry({"nrows": 1, "ncols": 2, "col_labels": ["a", "b", "c"]}, 2)
    assert [lbl["text"] for lbl in labels] == ["a", "b"]


def test_between_and_trailing_write_an_equation_through_the_gaps():
    _, labels = _grid_geometry({"nrows": 1, "ncols": 3, "between": [" ", "=", "+"], "trailing": "..."}, 3)
    assert [lbl["text"] for lbl in labels] == ["=", "+", "..."]


def test_a_bottom_strip_is_reserved_like_the_top_one():
    """Cells carrying tick labels or a shared axis label need the room taken out of them."""
    plain, _ = _grid_geometry({"nrows": 1, "ncols": 1, "hspace": 0.0}, 1)
    with_bottom, _ = _grid_geometry({"nrows": 1, "ncols": 1, "hspace": 0.0, "bottom": 0.16}, 1)
    assert with_bottom[0][1] == pytest.approx(0.16)
    assert with_bottom[0][3] == pytest.approx(plain[0][3] - 0.16)


def test_a_rotated_row_label_is_centred_rather_than_right_aligned():
    _, labels = _grid_geometry({"nrows": 1, "ncols": 1, "row_labels": ["Connectome"], "row_label_rotation": 90.0}, 1)
    assert labels[0]["kwargs"]["rotation"] == 90.0
    assert labels[0]["kwargs"]["ha"] == "center"


# ------------------------------------------------------------------ resolution


def test_layers_fill_the_grid_one_cell_each_through_the_shared_template():
    panel = _Panel(opts={"nrows": 1, "ncols": 2}, cell=_surface_cell(view="lateral"), layers=[_Layer("a"), _Layer("b")])
    got = _resolve_drawable(panel, "p", Path("."))
    assert len(got["insets"]) == 2
    assert all(c["kind"] == "surface" for c in got["insets"])
    assert all(c["ctx"]["opts"]["view"] == "lateral" for c in got["insets"])
    assert [c["layers"][0]["output"] for c in got["insets"]] == ["a", "b"]


def test_a_grid_draws_nothing_itself_so_its_layers_are_not_drawn_twice():
    panel = _Panel(opts={"ncols": 1}, cell=_surface_cell(), layers=[_Layer("a")])
    assert _resolve_drawable(panel, "p", Path("."))["layers"] == []


def test_declared_cells_may_differ_and_are_merged_over_the_template():
    """The common shape: a first cell showing the bare mesh the rest are maps on."""
    panel = _Panel(
        opts={"nrows": 1, "ncols": 2},
        cell=_surface_cell(view="lateral"),
        cells=[_Cell(kind="surface", opts={"color": "w"}), _Cell(kind="surface", layers=[_Layer("a")])],
    )
    cells = _resolve_drawable(panel, "p", Path("."))["insets"]
    assert cells[0]["ctx"]["opts"] == {"mesh": "m.npz", "view": "lateral", "color": "w"}
    assert not cells[0]["layers"] and cells[1]["layers"]


def test_a_row_scoped_opt_is_declared_once_for_the_row():
    panel = _Panel(
        opts={"nrows": 2, "ncols": 2, "row.view": ["lateral", "medial"]},
        cell=_surface_cell(),
        layers=[_Layer(str(i)) for i in range(4)],
    )
    views = [c["ctx"]["opts"]["view"] for c in _resolve_drawable(panel, "p", Path("."))["insets"]]
    assert views == ["lateral", "lateral", "medial", "medial"]


def test_a_column_scoped_opt_advances_across_instead_of_down():
    panel = _Panel(
        opts={"nrows": 2, "ncols": 2, "col.cmap": ["viridis", "seismic"]},
        cell=_surface_cell(),
        layers=[_Layer(str(i)) for i in range(4)],
    )
    maps = [c["ctx"]["opts"]["cmap"] for c in _resolve_drawable(panel, "p", Path("."))["insets"]]
    assert maps == ["viridis", "seismic", "viridis", "seismic"]


def test_cells_and_layers_together_are_refused():
    panel = _Panel(opts={"ncols": 1}, cell=_surface_cell(), cells=[_Cell(kind="surface")], layers=[_Layer("a")])
    with pytest.raises(ValueError, match="not both"):
        _resolve_drawable(panel, "p", Path("."))


def test_a_grid_with_nothing_to_draw_says_so():
    with pytest.raises(ValueError, match="nothing to draw"):
        _resolve_drawable(_Panel(opts={"ncols": 1}, cell=_surface_cell()), "p", Path("."))


def test_the_compact_form_needs_the_template_that_says_what_a_cell_is():
    with pytest.raises(ValueError, match="`cell:`"):
        _resolve_drawable(_Panel(opts={"ncols": 1}, layers=[_Layer("a")]), "p", Path("."))


def test_grid_cell_keys_are_unique_so_nesting_cannot_collide():
    inner = _Panel(opts={"ncols": 2}, cell=_surface_cell(), layers=[_Layer("a"), _Layer("b")])
    inner.bounds = [0.0, 0.0, 0.5, 0.5]
    outer = _Panel(opts={"ncols": 1}, cell=_surface_cell(), layers=[_Layer("c")])
    outer.insets = [inner]
    got = _resolve_drawable(outer, "p", Path("."))
    keys = [c["key"] for c in got["insets"]] + [c["key"] for c in got["insets"][-1]["insets"]]
    assert len(keys) == len(set(keys))


def test_a_grid_is_a_drawer_so_the_format_pass_leaves_its_cells_alone():
    """A grid of heatmaps sets deliberate ticks; re-deriving them would undo the panel."""
    panel = _Panel(opts={"ncols": 1}, cell=_surface_cell(), layers=[_Layer("a")])
    assert _resolve_drawable(panel, "p", Path("."))["drawer"] is True


def test_an_inset_without_bounds_is_refused_with_the_grid_named_as_the_exception():
    panel = _Panel(kind="cartesian")
    panel.insets = [_Cell(kind="cartesian")]
    with pytest.raises(ValueError, match="Only a `grid` cell may omit them"):
        _resolve_drawable(panel, "p", Path("."))


# ------------------------------------------------------------------ built-in kinds


@pytest.mark.parametrize("kind", ["colorbar", "legend"])
def test_a_builtin_kind_needs_no_render_and_no_code_modules(kind):
    """The point of these kinds: a shared scale or key with nothing registered by the study."""
    from tvbo.adapters.bsplot import CUSTOM_PANELS

    got = _resolve_drawable(_Panel(kind=kind, opts={}), "p", Path("."))
    assert got["render"] == kind and kind in CUSTOM_PANELS
    assert got["ctx"] is not None


def test_a_legend_names_its_entries_in_parallel_typed_lists():
    from tvbo.adapters.bsplot import legend_panel

    drawn = {}

    class _Ax:
        def axis(self, v):
            drawn["axis"] = v

        def legend(self, handles, labels, **kw):
            drawn["handles"], drawn["labels"], drawn["kw"] = handles, labels, kw

    legend_panel(None, _Ax(), {"opts": {"labels": ["long", "short"], "colors": ["#000000"], "linestyles": ["-", "--"]}})
    assert drawn["labels"] == ["long", "short"] and drawn["axis"] == "off"
    assert [h.get_linestyle() for h in drawn["handles"]] == ["-", "--"]
    assert drawn["handles"][1].get_color() == "k"  # shorter than labels -> default


def test_a_legend_with_no_labels_says_what_is_missing():
    from tvbo.adapters.bsplot import legend_panel

    with pytest.raises(ValueError, match="`labels:`"):
        legend_panel(None, None, {"opts": {"colors": ["k"]}})
