# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""A study states its look once, as a Theme, and every figure is drawn by it: the layering, the axis rule, and each slot's consumer."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.datamodel import schema as dm
from tvbo.plot import axis_rule, palette
from tvbo.utils.study_layout import study_path

HOUSE = {
    "ticks": "round",
    "max_ticks": 4,
    "axis_ends": "near",
    "end_tick_reach": 0.2,
    "spine_offset": 0.02,
    "spine_trim": True,
    "align_labels": True,
    "colorbar_ends": True,
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _line(lo=0.03, hi=9.7, scale=1.0):
    fig, ax = plt.subplots()
    x = np.linspace(lo, hi, 50)
    ax.plot(x, scale * x)
    return fig, ax


def _rule(figure) -> dict:
    return {k: v for k, v in bsplot.theme_spec(figure).items() if k in axis_rule.SLOTS}


# --------------------------------------------------------------------------- round ticks


@pytest.mark.parametrize("ends", axis_rule.ENDS)
@pytest.mark.parametrize(
    "extent", [(0.03, 9.7), (5.1, 14.9), (-0.0031, 0.0102), (0.6, 1.4), (12.0, 97.0), (0.2, 0.91), (-1.0, 1.0), (3e5, 7.2e6)]
)
def test_round_ticks_are_multiples_of_a_round_step_inside_the_limits_they_come_with(extent, ends):
    a, b = extent
    margin = 0.05 * (b - a)
    ticks, (low, high) = axis_rule.round_ticks(a, b, a - margin, b + margin, ends=ends)
    step = ticks[1] - ticks[0]
    mantissa = step / 10 ** np.floor(np.log10(step))
    assert any(np.isclose(mantissa, m) for m in (1, 2, 5, 10)), f"step {step} is not 1, 2 or 5 times a power of ten"
    assert np.allclose(ticks / step, np.round(ticks / step)), (
        "every tick is a multiple of the step, so its label can state it exactly"
    )
    assert low <= a and high >= b, "no rule may cut the data"
    assert low - 1e-9 * (b - a) <= ticks[0] and ticks[-1] <= high + 1e-9 * (b - a)
    assert len(ticks) >= 2


def test_axis_ends_states_which_of_round_ticks_end_ticks_and_no_empty_space_gives_way():
    a, b = 0.2, 0.91
    view = (a - 0.05 * (b - a), b + 0.05 * (b - a))
    _, at_data = axis_rule.round_ticks(a, b, *view, ends="data")
    on_ticks, at_ticks = axis_rule.round_ticks(a, b, *view, ends="ticks")
    near_ticks, near = axis_rule.round_ticks(a, b, *view, ends="near")
    assert at_data == view, "`data` never moves a limit"
    assert at_ticks == (on_ticks[0], on_ticks[-1]) == (0.0, 1.0), (
        "`ticks` ends on a tick both sides, at the cost of the empty space"
    )
    assert near[0] == near_ticks[0] == 0.2, "`near` takes the tick that is there for the taking"
    assert near[1] == view[1], "and leaves the far end at the data rather than widening the axis to reach 1.0"
    assert (b - a) / (at_ticks[1] - at_ticks[0]) < 0.75 < (b - a) / (near[1] - near[0]), (
        "which is the space `ticks` gives up and `near` keeps"
    )


def test_the_tick_budget_and_the_round_steps_are_the_rules_to_state():
    assert len(axis_rule.round_ticks(0, 60, ends="ticks", max_ticks=4)[0]) == 4
    assert len(axis_rule.round_ticks(0, 60, ends="ticks", max_ticks=7)[0]) == 7
    assert axis_rule.round_ticks(0, 75, ends="ticks")[0].tolist() == [0, 50, 100]
    assert axis_rule.round_ticks(0, 75, ends="ticks", steps=(1, 2, 2.5, 5))[0].tolist() == [0, 25, 50, 75], (
        "2.5 counts as round only where the theme admits it"
    )
    with pytest.raises(ValueError, match=r"\[1, 10\)"):
        axis_rule.round_ticks(0, 1, steps=(0.5, 1))


def test_a_view_no_step_fits_gets_one_tick_too_many_rather_than_two_unround_ones():
    ticks, limits = axis_rule.round_ticks(5.1, 14.9, ends="data")
    assert limits == (5.1, 14.9)
    assert ticks.tolist() == [6, 8, 10, 12, 14], (
        "five multiples of 2 fit and only one of 5 does; the data's own ends are not ticks"
    )


# --------------------------------------------------------------------------- the pass over a figure


def test_an_empty_rule_moves_nothing():
    fig, ax = _line()
    fig.canvas.draw()
    before = (
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_xticks().tolist(),
        ax.spines["left"].get_position(),
        ax.spines["left"].get_bounds(),
    )
    axis_rule.place(fig, {})
    axis_rule.frame(fig, {})
    fig.canvas.draw()
    assert before == (
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_xticks().tolist(),
        ax.spines["left"].get_position(),
        ax.spines["left"].get_bounds(),
    )


def test_whatever_a_panel_or_a_drawer_chose_stands():
    fig, axd = plt.subplot_mosaic("abc")
    x = np.linspace(0.03, 9.7, 50)
    axd["a"].plot(x, x)
    axd["a"].set_xticks([1, 4, 9])
    axd["a"].set_ylim(-3, 11)
    axd["b"].plot(x, x**2 + 1)
    axd["b"].set_yscale("log")
    axd["c"].imshow(np.zeros((8, 8)))
    before = {k: (a.get_xlim(), a.get_ylim()) for k, a in axd.items()}
    axis_rule.place(fig, {"ticks": "round", "axis_ends": "ticks"})
    assert axd["a"].get_xticks().tolist() == [1, 4, 9] and axd["a"].get_xlim() == before["a"][0], (
        "declared ticks keep their frame too"
    )
    assert axd["a"].get_ylim() == (-3, 11) and axd["a"].get_yticks().tolist() == [0, 5, 10], (
        "declared limits stay; the ticks inside them are round"
    )
    assert axd["b"].get_ylim() == before["b"][1] and axd["b"].get_yscale() == "log", "a log axis is not this rule's to place"
    assert axd["b"].get_xlim() == (0, 10), "while its linear axis is"
    assert (axd["c"].get_xlim(), axd["c"].get_ylim()) == before["c"], "an image's extent is its frame"
    assert axd["c"].get_ylim()[0] > axd["c"].get_ylim()[1], "and stays top-down"


def test_axes_on_a_shared_scale_are_placed_from_their_common_extent():
    fig, (p, q) = plt.subplots(1, 2)
    x = np.linspace(0, 10, 20)
    p.plot(x, x)
    q.plot(x, 3 * x)
    axis_rule.place(fig, {"ticks": "round", "axis_ends": "ticks"}, shared={"y": [[p, q]]})
    assert p.get_ylim() == q.get_ylim() == (0, 30)
    assert p.get_yticks().tolist() == q.get_yticks().tolist() == [0, 10, 20, 30]


def test_native_ticks_can_still_end_the_axis_on_a_tick():
    fig, ax = _line()
    axis_rule.place(fig, {"axis_ends": "ticks"})
    fig.canvas.draw()
    ticks = ax.get_xticks()
    assert ax.get_xlim() == (ticks[0], ticks[-1]) == (0, 10)
    assert not isinstance(ax.xaxis.get_major_locator(), matplotlib.ticker.FixedLocator), (
        "the backend's locator is still the one placing them"
    )


def test_declared_ticks_refuse_every_axis_nobody_gave_ticks_naming_them_all_at_once():
    fig, axd = plt.subplot_mosaic("ab")
    x = np.linspace(0, 1, 10)
    axd["a"].plot(x, x)
    axd["a"].set_xticks([0, 1])
    axd["b"].plot(x, x)
    inset = axd["b"].inset_axes([0.5, 0.5, 0.4, 0.4])
    inset.plot(x, x)
    inset.set_xticks([0, 1])
    with pytest.raises(ValueError) as refused:
        axis_rule.place(fig, {"ticks": "declared"}, names={id(a): k for k, a in axd.items()})
    message = str(refused.value)
    assert "panel a: y" in message and "panel b: x, y" in message and "panel b (inset): y" in message
    assert "panel a: x" not in message
    for ax in (axd["a"], axd["b"], inset):
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    axis_rule.place(fig, {"ticks": "declared"})


def test_a_boxed_frame_keeps_its_four_sides_whole():
    fig, (open_ax, box_ax) = plt.subplots(1, 2)
    for ax in (open_ax, box_ax):
        ax.plot([0.3, 9.7], [0.3, 9.7])
    for side in ("top", "right"):
        open_ax.spines[side].set_visible(False)
        box_ax.spines[side].set_visible(True)
    rule = {"ticks": "round", "spine_offset": 0.02, "spine_trim": True}
    axis_rule.place(fig, rule)
    axis_rule.frame(fig, rule)
    assert open_ax.spines["left"].get_position() == ("axes", -0.02)
    assert open_ax.spines["left"].get_bounds() == tuple(open_ax.get_yticks()[[0, -1]]) == (0.0, 10.0)
    assert box_ax.spines["left"].get_position() == ("outward", 0.0) and box_ax.spines["left"].get_bounds() is None


# --------------------------------------------------------------------------- the layers of a look


def test_a_figure_no_theme_speaks_about_follows_no_axis_rule():
    look = bsplot.theme_spec(dm.Figure(name="f"))
    assert _rule(dm.Figure(name="f")) == {}
    assert {k: v for k, v in look.items() if k not in palette.FIELDS} == palette.DEFAULT_GEOMETRY


def test_auto_format_true_asks_for_the_curated_house_rule_and_false_switches_every_rule_off():
    assert _rule(dm.Figure(name="f", auto_format=True)) == HOUSE
    study = dm.Theme(iri="tvbo:theme/bsplot")
    assert _rule(bsplot.adopt_theme(dm.Figure(name="f"), study)) == HOUSE, "unset follows the study"
    assert _rule(bsplot.adopt_theme(dm.Figure(name="f", auto_format=False, spine_offset=0.05), study)) == {}, (
        "and false leaves the axes as drawn, whoever stated the rule"
    )


def test_each_layer_states_only_what_it_changes_and_the_nearer_one_wins():
    study = dm.Theme(
        iri="tvbo:theme/bsplot",
        ticks="native",
        spines="open",
        font_size=7,
        tick_length=3.0,
        colormaps={"diverging": "coolwarm"},
        opts={"hatch.linewidth": 0.5},
    )
    figure = bsplot.adopt_theme(
        dm.Figure(
            name="f",
            theme=dm.Theme(axis_ends="ticks", tick_length=2.0, opts={"lines.dashed_pattern": [2, 2]}),
            spines="box",
            font_size=8,
        ),
        study,
    )
    look = bsplot.theme_spec(figure)
    assert look["ticks"] == "native", "the study's statement beats the curated theme it names"
    assert look["spine_offset"] == 0.02, "which still supplies everything the study left unsaid"
    assert look["axis_ends"] == "ticks" and look["tick_length"] == 2.0, "the figure's theme beats the study's"
    assert look["spines"] == "box" and look["font_size"] == 8, "and the figure's own slots beat both"
    assert look["colormaps"] == {**palette.DEFAULT["colormaps"], "diverging": "coolwarm"}, (
        "a theme adds one scale without unsetting the others"
    )
    assert look["opts"] == {"hatch.linewidth": 0.5, "lines.dashed_pattern": [2, 2]}
    assert _rule(bsplot.adopt_theme(dm.Figure(name="f", auto_format=True), dm.Theme(ticks="native")))["ticks"] == "native", (
        "a declared rule is never overruled by the house one"
    )


def test_every_theme_slot_has_a_consumer():
    """A slot the schema accepts and the adapter never reads is a declaration that silently does nothing."""
    read = bsplot.THEME_CONSUMERS | set(axis_rule.SLOTS)
    assert set(palette.GEOMETRY) == read, (
        f"unread: {sorted(set(palette.GEOMETRY) - read)}, unknown: {sorted(read - set(palette.GEOMETRY))}"
    )
    named = {p for ps in bsplot._THEME_RCPARAMS.values() for p in ps}
    named |= {p for choices in bsplot._THEME_RCPARAM_VALUES.values() for params in choices.values() for p in params}
    named |= {p for ps in bsplot._TYPE_SCALES.values() for p in ps} | set(bsplot._BODY_SIZED)
    assert not [p for p in named if p not in matplotlib.rcParams], "every setting the table names is one the backend has"


def test_an_enum_slot_reaches_the_backend_as_its_value():
    """Flattened like a record, an enum member came out as `{}` and the slot was dropped without a word."""
    theme = dm.Theme(
        tick_direction="in",
        line_cap="round",
        grid_style="dotted",
        legend_loc="upper right",
        title_loc="left",
        math_font="stix",
        tick_format="sci",
        spines="box",
        editable_text=True,
    )
    rc = bsplot.theme_rcparams(bsplot.theme_spec(dm.Figure(name="f", theme=theme)))
    assert rc["xtick.direction"] == rc["ytick.direction"] == "in"
    assert rc["lines.solid_capstyle"] == "round" and rc["grid.linestyle"] == "dotted"
    assert rc["legend.loc"] == "upper right" and rc["axes.titlelocation"] == "left" and rc["mathtext.fontset"] == "stix"
    assert rc["axes.formatter.limits"] == [-2, 3] and rc["axes.spines.top"] is True and rc["svg.fonttype"] == "none"
    with matplotlib.rc_context():
        matplotlib.rcParams.update(rc)


def test_type_sizes_are_multiples_of_one_body_size():
    sized = bsplot.type_scales({"font_size": 7, "label_scale": 1.15})
    assert sized["font.size"] == 1.0 and sized["axes.labelsize"] == 1.15 and sized["xtick.labelsize"] == 1.0
    assert set(sized) == {
        "font.size",
        "figure.titlesize",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    }
    assert bsplot.type_scales({"tick_label_scale": 0.9}) == {"xtick.labelsize": 0.9, "ytick.labelsize": 0.9}, (
        "without a body size only what is stated applies"
    )
    assert bsplot.type_scales({}) == {}


# --------------------------------------------------------------------------- end to end


def _study(tmp_path, theme: str = "", figure: str = ""):
    """A study on disk with one run container and one three-panel figure, its theme a spec fragment when given."""
    root = tmp_path / "Tiny"
    (root / "spec").mkdir(parents=True)
    (root / "code").mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "Tiny", "DatasetType": "study"}))
    results = study_path("results", root=root)
    results.mkdir(parents=True)
    t = np.linspace(0.03, 9.7, 200)
    xr.Dataset(
        {"v": ("time", 0.0102 * np.sin(t)), "w": ("time", t**2), "m": (("i", "j"), np.random.default_rng(0).random((8, 8)))},
        coords={"time": t},
    ).to_netcdf(results / "exp-1_result.h5", engine="h5netcdf")
    if theme:
        (root / "spec" / "desc-house_theme.yaml").write_text("tvbo_class: tvbo:Theme\n" + theme)
    (root / "Tiny.yaml").write_text(
        "label: Tiny\n"
        + ("theme: !include spec/desc-house_theme.yaml\n" if theme else "")
        + "figures:\n  - name: fig_one\n    layout: abc\n    width: 180\n    height: 55\n"
        + figure
        + "    panels:\n"
        "      a: {kind: cartesian, layers: [{used: {iri: exp-1, output: v}, encoding: {x: time, y: v}}]}\n"
        "      b: {kind: cartesian, yticks: [0, 50, 100], legend: {}, layers: [{used: {iri: exp-1, output: w}, encoding: {x: time, y: w}, label: w}]}\n"
        "      c: {kind: heatmap, colorbar: {decimals: 1}, layers: [{used: {iri: exp-1, output: m}, encoding: {x: j, y: i}}]}\n"
    )
    return root


def _render(root, tmp_path):
    study = tvbo.SimulationStudy.from_file(str(root / "Tiny.yaml"))
    figure = study.figures[0]
    drawn = bsplot.render(figure, base_dir=str(root), outfile=str(tmp_path / "fig.png"), script_path=str(tmp_path / "plot.py"))
    return figure, drawn, {a.get_label(): a for a in drawn.axes}


def test_a_study_with_no_theme_draws_its_axes_as_the_backend_places_them(tmp_path):
    figure, drawn, axes = _render(_study(tmp_path), tmp_path)
    code = (tmp_path / "plot.py").read_text()
    assert "_RULE = {}" in code and "format_fig" not in code
    assert not isinstance(axes["a"].yaxis.get_major_locator(), matplotlib.ticker.FixedLocator)
    assert axes["a"].get_xlim()[0] < 0.03 and axes["a"].spines["left"].get_bounds() is None
    assert axes["a"].spines["left"].get_position() == ("outward", 0.0)
    assert [t.get_text() for t in axes["<colorbar>"].get_yticklabels()][:2] == ["0.0", "0.2"], (
        "a bar's declared decimals hold with no rule in force"
    )


def test_a_study_states_its_rule_once_in_a_theme_file_and_every_figure_follows_it(tmp_path):
    theme = "iri: tvbo:theme/bsplot\nfont_size: 7\npanel_number_case: upper\npanel_number_format: '({})'\nlegend_loc: lower right\nformat: pdf\n"
    figure, drawn, axes = _render(_study(tmp_path, theme), tmp_path)
    assert axes["a"].get_xlim() == (0, 10) and axes["a"].get_xticks().tolist() == [0, 5, 10]
    assert axes["a"].spines["left"].get_position() == ("axes", -0.02) and axes["a"].spines["left"].get_bounds() == (
        -0.01,
        0.01,
    )
    assert axes["b"].get_yticks().tolist() == [0, 50, 100], "the ticks a panel declares stand over the rule"
    assert [t.get_text() for t in axes["<colorbar>"].get_yticklabels()] == ["0.0", "0.5", "1.0"]
    letters = [t.get_text() for a in ("a", "b", "c") for t in axes[a].texts]
    assert letters == ["(A)", "(B)", "(C)"]
    assert "**(A)**" in bsplot.compose_caption(figure), "the caption letters a panel the way the figure draws it"
    assert bsplot.output_format(figure) == "pdf"
    assert axes["b"].get_legend()._loc == matplotlib.legend.Legend.codes["lower right"], (
        "a legend its panel does not place sits where the theme says"
    )
    assert axes["a"].xaxis.label.get_size() == 7.0


def test_a_figure_overrides_its_studys_rule_without_restating_it(tmp_path):
    theme = "iri: tvbo:theme/bsplot\n"
    figure = "    spines: box\n    theme: {axis_ends: ticks, colorbar_ends: false, colorbar_outline: true}\n"
    _, drawn, axes = _render(_study(tmp_path, theme, figure), tmp_path)
    assert axes["a"].get_ylim() == (-0.02, 0.02), "both ends on a tick, as this one figure asks"
    assert axes["a"].spines["left"].get_bounds() is None and axes["a"].spines["top"].get_visible(), (
        "a boxed frame is neither stood off nor trimmed"
    )
    assert len(axes["<colorbar>"].get_yticks()) > 3, "and its bar keeps the backend's round ticks"


def test_a_panels_scientific_axis_writes_its_exponent_the_way_the_theme_does(tmp_path):
    figure = "    theme: {exponent_as_power: false}\n"
    root = _study(tmp_path, figure=figure)
    spec = root / "Tiny.yaml"
    spec.write_text(spec.read_text().replace("a: {kind: cartesian,", "a: {kind: cartesian, ytick_format: sci,"))
    _, drawn, axes = _render(root, tmp_path)
    assert axes["a"].yaxis.get_major_formatter().get_useMathText() is False, "1e-3, as the theme states, not the typeset power"
    spec.write_text(spec.read_text().replace("    theme: {exponent_as_power: false}\n", ""))
    _, drawn, axes = _render(root, tmp_path)
    assert axes["a"].yaxis.get_major_formatter().get_useMathText() is True


def test_declared_ticks_are_enforced_at_render_time(tmp_path):
    with pytest.raises(ValueError, match="panel a: x, y.*panel b: x"):
        _render(_study(tmp_path, "ticks: declared\n"), tmp_path)


def test_a_nested_study_takes_its_parents_theme_unless_it_states_its_own(tmp_path):
    parent = tvbo.SimulationStudy(
        label="paper",
        theme={"ticks": "round"},
        figures=[{"name": "own"}],
        studies=[
            {"label": "plain", "figures": [{"name": "inherits"}]},
            {"label": "styled", "theme": {"ticks": "declared"}, "figures": [{"name": "keeps"}]},
        ],
    )
    from tvbo.behaviour.study import hand_down_theme

    hand_down_theme(parent)
    rule = {f.name: _rule(f).get("ticks") for s in (parent, *parent.studies) for f in s.figures}
    assert rule == {"own": "round", "inherits": "round", "keeps": "declared"}
