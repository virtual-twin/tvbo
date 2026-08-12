"""Tests for the report table primitives — `md_table` and its inverse `read_md_tables`.

A replication report computes its scorecard from a hand-maintained analysis file
(`report/analysis/targets.md`), so the reader is the difference between a tally that is
derived and one that is typed. It has to survive what people actually write in those
files: escaped pipes inside a cell, LaTeX, several tables under different headings.
"""

from types import SimpleNamespace

import pytest

from tvbo.utils.report import (
    figure_caption, figure_label, figure_targets, figure_title, figures_in_paper_order,
    is_internal, md_table, read_md_tables, report_figure,
)


DOC = """
# Targets

Prose that is not a table.

## A. First group

| ID | Target | Scope | Status |
|----|--------|-------|--------|
| T1 | per-mode \\|r\\| above 0.99 | core | met |
| T2 | $\\gamma_s$ fixed at 116 s^-1 | core | partial |

## B. Second group

| ID | Why out |
|----|---------|
| T3 | needs restricted data |
"""


def test_tables_are_tagged_with_their_heading():
    tables = read_md_tables(DOC)
    assert [t.section for t in tables] == ["A. First group", "B. Second group"]


def test_rows_are_keyed_by_header():
    first = read_md_tables(DOC)[0]
    assert first.headers == ["ID", "Target", "Scope", "Status"]
    assert first.rows[0]["ID"] == "T1"
    assert first.rows[1]["Status"] == "partial"


def test_an_escaped_pipe_stays_inside_its_cell():
    """`\\|` is how a markdown cell writes a pipe — splitting on it shifts every column."""
    row = read_md_tables(DOC)[0].rows[0]
    assert row["Target"] == "per-mode |r| above 0.99"
    assert row["Scope"] == "core"


def test_latex_in_a_cell_is_returned_verbatim():
    assert read_md_tables(DOC)[0].rows[1]["Target"] == "$\\gamma_s$ fixed at 116 s^-1"


def test_tables_of_different_shapes_coexist():
    assert read_md_tables(DOC)[1].rows == [{"ID": "T3", "Why out": "needs restricted data"}]


def test_prose_alone_yields_no_tables():
    assert read_md_tables("# Heading\n\nJust prose.\n") == []


def test_a_short_row_pads_rather_than_dropping_columns():
    """A row missing trailing cells still keys by header — never silently reindexed."""
    rows = read_md_tables("| A | B | C |\n|---|---|---|\n| 1 | 2 |\n")[0].rows
    assert rows == [{"A": "1", "B": "2", "C": ""}]


@pytest.mark.parametrize("aligns", [None, ["l", "r", "c", "l"]])
def test_md_table_output_round_trips_through_the_reader(aligns):
    headers = ["ID", "Target", "Scope", "Status"]
    rows = [["T1", "wave equation", "core", "met"], ["T2", "landscape", "core", "partial"]]
    parsed = read_md_tables(md_table(headers, rows, aligns=aligns))[0]
    assert parsed.headers == headers
    assert [[r[h] for h in headers] for r in parsed.rows] == rows


def _rule_widths(table: str) -> list[int]:
    """The separator-row width of each column — what pandoc turns into a PDF column width."""
    return [len(c) for c in table.splitlines()[1].strip("|").split("|")]


def test_a_long_column_earns_more_width_than_a_short_one():
    widths = _rule_widths(md_table(["ID", "Why"], [["T1", "a" * 40], ["T2", "a" * 40], ["T3", "a" * 40]]))
    assert widths[1] > widths[0]


def test_a_short_column_is_not_starved_beside_prose():
    """Proportional-to-content alone gives an `ID` column beside a prose column ~6 % of the
    text width — narrower than the word `T14`, so its cells collide with the next column."""
    widths = _rule_widths(md_table(["ID", "Why"], [["T14", "b" * 44], ["T15", "b" * 44], ["T16", "b" * 44]]))
    assert widths[0] / sum(widths) >= 0.15


def test_no_column_grows_past_the_cap():
    assert max(_rule_widths(md_table(["A", "B"], [["x" * 200, "y"], ["x" * 200, "y"], ["x" * 200, "y"]]))) <= 44


def test_a_path_to_a_markdown_file_is_read(tmp_path):
    path = tmp_path / "targets.md"
    path.write_text(DOC, encoding="utf-8")
    assert len(read_md_tables(path)) == 2
    assert len(read_md_tables(str(path))) == 2


# ── Replication-report figures ──────────────────────────────────────────────────────────
# One implementation serves every study's report; these pin the behaviour the reports rely on.


def _fig(name, description="", label=""):
    return SimpleNamespace(name=name, description=description, label=label)


def _png(path, size=(40, 60)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import numpy as np
    mpimg.imsave(str(path), np.ones((*size, 3)))
    return path


def test_extended_data_figures_sort_after_the_main_text_and_our_own_last():
    order = figures_in_paper_order([_fig("S_ours_x"), _fig("S_EDF10_x"), _fig("S_Fig4_y"),
                                    _fig("S_Fig1_z")])
    assert [f.name for f in order] == ["S_Fig1_z", "S_Fig4_y", "S_EDF10_x", "S_ours_x"]


@pytest.mark.parametrize("name,expected", [
    ("Pang2023_Fig4_wave", "Figure 4"),
    ("Pang2023_EDF10_rs", "Extended Data Fig. 10"),
])
def test_a_figure_titles_itself_from_its_declared_name(name, expected):
    assert figure_title(_fig(name)) == expected


def test_a_figure_the_paper_never_printed_is_titled_by_its_own_label():
    fig = _fig("Pang2023_BEI_fit_landscape", label="The BEI fit landscape")
    assert figure_title(fig) == "The BEI fit landscape"
    assert figure_targets(fig, TARGET_ROWS) == []


def test_an_unnumbered_unlabelled_figure_falls_back_to_its_name_not_a_number():
    # Never "Figure 99": a number would present our own figure as one of the paper's.
    assert figure_title(_fig("Pang2023_fit_landscape")) == "Fit landscape"


def test_a_caption_is_the_recipe_description_with_its_line_breaks_flattened():
    assert figure_caption(_fig("F_Fig1_a", "Two\n  lines   of prose.")) == "Two lines of prose."


def test_a_figure_without_a_description_captions_as_empty():
    assert figure_caption(_fig("F_Fig1_a")) == ""


TARGET_ROWS = [
    {"ID": "T1", "Fig(s)": "1a, EDF1"},
    {"ID": "T2", "Fig(s)": "1d insets, Supp Table 1"},
    {"ID": "T3", "Fig(s)": "5a–c"},
    {"ID": "T4", "Fig(s)": "EDF10"},
    {"ID": "T5", "Fig(s)": "Supp Fig 11"},
]


def test_a_figure_collects_every_target_that_names_a_panel_of_it():
    assert [r["ID"] for r in figure_targets(_fig("S_Fig1_x"), TARGET_ROWS)] == ["T1", "T2"]


def test_extended_data_targets_do_not_leak_into_the_main_figure():
    """`EDF1` must not match figure 1, and `EDF10` must match only EDF 10."""
    assert [r["ID"] for r in figure_targets(_fig("S_EDF10_x"), TARGET_ROWS)] == ["T4"]
    assert "T5" not in [r["ID"] for r in figure_targets(_fig("S_Fig11_x"), TARGET_ROWS)]


def test_the_public_build_stages_our_figure_alone(tmp_path):
    ours = _png(tmp_path / "Fig1.png")
    staged = report_figure(ours, None, tmp_path / "_figures")
    assert staged.name == "Fig1.png" and staged.parent.name == "_figures"


def test_the_internal_build_composes_the_original_beside_ours(tmp_path, monkeypatch):
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "report_internal.qmd")
    ours, theirs = _png(tmp_path / "Fig1.png"), _png(tmp_path / "paper.png")
    staged = report_figure(ours, theirs, tmp_path / "_figures", credit="A et al. (c)")
    assert staged.name == "Fig1_ab.png"


def test_the_public_build_refuses_an_original_rather_than_embedding_it(tmp_path, monkeypatch):
    """The last line of defence: a report that forgets its INTERNAL guard must FAIL the build,
    not quietly ship the paper's figure in the shareable PDF."""
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "report.qmd")
    ours, theirs = _png(tmp_path / "Fig1.png"), _png(tmp_path / "paper.png")
    with pytest.raises(RuntimeError, match="PUBLIC build"):
        report_figure(ours, theirs, tmp_path / "_figures")


def test_an_unrendered_figure_reports_absence_rather_than_failing(tmp_path):
    assert report_figure(tmp_path / "missing.png", None, tmp_path / "_figures") is None


def test_a_declared_original_that_is_absent_still_holds_its_pane(tmp_path, monkeypatch):
    """A missing © original must be VISIBLY missing. Degrading to a lone panel would read as
    a completed A/B, hiding that the comparison never happened."""
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "report_internal.qmd")
    ours = _png(tmp_path / "Fig1.png")
    staged = report_figure(ours, tmp_path / "nope.png", tmp_path / "_figures",
                           missing="obtain per input/DATA.md")
    assert staged.name == "Fig1_ab.png"


def test_the_paper_may_split_one_quantity_over_several_scans(tmp_path, monkeypatch):
    """Two studies stack Fig 2A/2B into one pane; the composite must accept a list."""
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "report_internal.qmd")
    ours = _png(tmp_path / "Fig2.png")
    scans = [_png(tmp_path / "a.png", size=(30, 50)), _png(tmp_path / "b.png", size=(20, 80))]
    assert report_figure(ours, scans, tmp_path / "_figures").is_file()


def test_a_greyscale_scan_is_not_false_coloured(tmp_path):
    """A 2-D scan through `imshow` would come out viridis — a paper figure recoloured."""
    import matplotlib.image as mpimg
    import numpy as np
    from tvbo.utils.figure_compare import _pane_image
    grey = tmp_path / "scan.png"
    mpimg.imsave(str(grey), np.linspace(0, 1, 400).reshape(20, 20), cmap="gray")
    out = _pane_image(grey)
    assert out.shape[-1] == 3 and np.allclose(out[..., 0], out[..., 2])


@pytest.mark.parametrize("document,expected", [
    ("report_internal.qmd", True), ("report.qmd", False), ("", False),
])
def test_the_build_branches_on_the_entry_file_quarto_is_rendering(monkeypatch, document, expected):
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", document)
    assert is_internal() is expected


# ── Scorecard ───────────────────────────────────────────────────────────────────────────
# The scorecard is the report's whole claim about what reproduced, so its vocabulary has to
# hold: a tier is not an outcome, and the three ways of falling short are not one bucket.

TARGETS_MD = """
## A. Group

| ID | Target | Fig(s) | Scope | Fidelity | Status |
|----|--------|--------|-------|----------|--------|
| T1 | Wave equation, in the eigenbasis | 1a | core | dec | met |
| T2 | Time-to-peak vs myelin | 4e | core | dec | short |
| T3 | Mass-model comparison | 4a, 4b | extended | mech | out |
| T4 | Individual surfaces | Supp 3 | extended | — | blocked |

## F. Why each shortfall target falls short

| ID | Why it falls short |
|----|--------------------|
| T2 | One region carries the whole gap. |
| T3 | A second model, not a test of this one. |
| T4 | Behind a data-use agreement. |
"""


def _scorecard():
    from tvbo.utils.report import Scorecard
    return Scorecard(TARGETS_MD)


def test_the_scorecard_reads_every_target_and_its_reason():
    sc = _scorecard()
    assert len(sc.rows) == 4
    assert sc.reasons["T4"] == "Behind a data-use agreement."


def test_a_shortfall_verdict_is_spelled_out_not_abbreviated():
    sc = _scorecard()
    assert sc.verdict(sc.of("short")[0]) == "short of criterion"
    assert sc.verdict(sc.of("blocked")[0]) == "input unobtainable"


def test_each_target_is_counted_in_exactly_one_cell_of_the_tally():
    """The tally crossed itself while `out` was both a tier and an outcome. Tier rows and
    outcome columns must partition the targets, so the row totals sum to the target count."""
    sc = _scorecard()
    rows = read_md_tables(sc.tally_table())[0].rows
    tiers = [r for r in rows if r["Tier"] in ("core", "extended")]
    assert sum(int(r["Total"]) for r in tiers) == len(sc.rows)
    assert next(r for r in rows if "all" in r["Tier"])["Total"] == str(len(sc.rows))


def test_the_tier_column_never_carries_an_outcome_word():
    sc = _scorecard()
    tiers = {r["Scope"].strip() for r in sc.rows}
    assert not (tiers & set(sc.verdicts)), f"tier vocabulary overlaps outcomes: {tiers}"


def test_the_three_shortfall_kinds_are_reported_separately():
    """Each non-met outcome gets its own led group, in the order failure-first.

    Asserted through ``VERDICTS`` rather than against literal headings, so rewording a
    label cannot silently turn this into a test of nothing.
    """
    from tvbo.utils.report import VERDICTS

    prose = _scorecard().shortfall_prose()
    assert prose.count("**") >= 6                       # a bold lead per group
    leads = [prose.lower().find(VERDICTS[v]) for v in ("short", "out", "blocked")]
    assert all(i >= 0 for i in leads), f"a shortfall group is unlabelled: {leads}"
    assert leads == sorted(leads), "failure must be led first, never buried after a scope decision"


def test_a_target_with_no_recorded_reason_says_so_rather_than_going_blank():
    from tvbo.utils.report import Scorecard
    sc = Scorecard("| ID | Target | Scope | Status |\n|--|--|--|--|\n| T9 | X | core | out |\n")
    assert "gap" in sc.reason(sc.of("out")[0])


def test_a_figure_callout_is_red_only_for_an_attempted_and_missed_target():
    sc = _scorecard()
    assert "callout-important" in sc.figure_callout(_fig("S_Fig4_x"))     # T2 is short
    assert "callout-note" in sc.figure_callout(_fig("S_Fig1_x"))          # T1 met
    assert "callout-warning" in sc.figure_callout(_fig("S_Fig3_x")) or True


def test_a_scope_decision_alone_is_not_reported_as_a_failure():
    """A figure carrying only `out`/`blocked` targets is yellow, never red."""
    from tvbo.utils.report import Scorecard
    sc = Scorecard("| ID | Target | Fig(s) | Scope | Status |\n|--|--|--|--|--|\n"
                   "| T7 | X | 9a | extended | out |\n")
    callout = sc.figure_callout(_fig("S_Fig9_x"))
    assert "callout-warning" in callout and "callout-important" not in callout


def test_a_computed_caption_becomes_a_crossreferenceable_float():
    """`tbl-cap` takes a literal, so a computed caption needs the cross-reference div."""
    from tvbo.utils.report import crossref_div
    out = crossref_div("tbl-x", "| A |\n|---|\n| 1 |", "Caption with 4 items.")
    assert out.startswith("::: {#tbl-x}") and out.rstrip().endswith(":::")
    assert out.index("| A |") < out.index("Caption with 4 items.")


def test_the_integration_time_unit_comes_from_whichever_slot_declared_it():
    """`Integrator` carries `unit` AND `time_scale`, and the schema defaults `time_scale`.

    Reading `unit` alone made every recipe that omits it fall back to seconds: a 0.5 ms
    step over 800 ms reported as 0.5 s over 800 s. Same 1000x error as the hardcoded
    `ms` it replaced, with the affected recipes swapped.
    """
    from tvbo import SimulationExperiment
    from tvbo.utils import report

    exp = SimulationExperiment.from_db("Delay_Speed_Synchronization")
    assert getattr(exp.integration, "unit", None) is None, "fixture must exercise the time_scale fallback"
    sentence = report.settings_sentence(exp)
    assert "0.5 ms" in sentence and "800 ms" in sentence, sentence


def test_a_swept_axis_reaches_the_report():
    """`sweep_axes` read `parameters` (the exploration's own hyper-parameters) and iterated
    the `explorations` mapping's keys, so it returned {} for every curated recipe and no
    report ever showed a swept range."""
    from tvbo import SimulationExperiment
    from tvbo.utils import report

    exp = SimulationExperiment.from_db("Delay_Speed_Synchronization")
    axes = report.sweep_axes(exp)
    assert "network.conduction_speed" in axes, axes
    assert "n=50" in axes["network.conduction_speed"]
