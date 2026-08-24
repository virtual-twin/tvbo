"""`SimulationStudy.report` writes each shared thing once.

A per-experiment Methods section repeats the model for every experiment that uses it:
Jansen1995's seven experiments emitted 1209 lines and about thirty tables, of which the six state equations, the symbol table and the parameter table were the same text over and over. These tests pin the contract that removes that — one system per family, deltas for variants, one comparison table carrying only what differs — and the two properties a generated cross-referenced document breaks silently without: unique anchors, and no number left to drift.
"""

import re
from types import SimpleNamespace

import pytest

from tvbo import SimulationStudy

RECIPE = """
title: A study whose experiments share one model
label: Mini
key: Mini2020
experiments:
  - id: 1
    label: baseline
    description: Establishes the resting decay.
    references: ["Fig. 1"]
    observations: &observations
      rate:
        label: Mean deflection
        source: x
        aggregation: mean
        time_scale: s
        description: Averaged over the post-transient window.
      spectrum:
        label: Power spectrum
        source: x
        window_size: 100
        time_scale: s
    dynamics: &base
      name: Decay
      label: Linear decay
      state_variables:
        x:
          equation: {rhs: "-k * x"}
          unit: mV
          description: membrane deflection
      parameters:
        k: {value: 2.0, unit: 1/s, description: decay rate}
        drive: {value: 0.0, description: constant drive}
    integration: &integration
      method: RungeKutta4thOrder
      duration: 1.0
      step_size: 0.001
      transient_time: 0.1
      unit: s
  - id: 2
    label: repeat
    description: The same model again, to be collapsed onto the first.
    references: ["Fig. 2"]
    observations: *observations
    dynamics: *base
    integration: *integration
    network:
      number_of_nodes: 2
      coupling:
        c_net:
          label: Linear diffusive coupling
          delayed: false
          incoming_states: [x]
          local_states: [x]
          pre_expression: {rhs: "x_j - x_i"}
          post_expression: {rhs: "gx"}
  - id: 3
    label: driven
    description: Adds a constant drive.
    part: supplementary
    references: ["Fig. 3"]
    dynamics:
      <<: *base
      name: DecayDriven
      label: Linear decay with drive
      state_variables:
        x:
          equation: {rhs: "-k * x + drive"}
          unit: mV
          description: membrane deflection
      parameters:
        k: {value: 2.0, unit: 1/s, description: decay rate}
        drive: {value: 1.5, description: constant drive}
    integration:
      <<: *integration
      duration: 4.0
  - id: 4
    label: observed
    description: The base model with a haemodynamic readout bolted on.
    references: ["Fig. 4"]
    dynamics:
      <<: *base
      name: DecayObserved
      label: Linear decay with a readout
      state_variables:
        x:
          equation: {rhs: "-k * x"}
          unit: mV
          description: membrane deflection
        h:
          equation: {rhs: "x - h"}
          description: readout filter
      parameters:
        k: {value: 2.0, unit: 1/s, description: decay rate}
        drive: {value: 0.0, description: constant drive}
    integration: *integration
  - id: 5
    label: unrelated
    description: A different system that happens to share the readout variable.
    references: ["Fig. 5"]
    dynamics:
      name: Oscillator
      label: Harmonic oscillator with the same readout
      state_variables:
        theta:
          equation: {rhs: "omega"}
          description: phase
        h:
          equation: {rhs: "theta - h"}
          description: readout filter
      parameters:
        omega: {value: 1.0, unit: 1/s, description: angular frequency}
    integration: *integration
"""


_BLOCK = re.compile(r"\$\$(.+?)\$\$(\s*\{#(eq-[a-z0-9-]+)\})?", re.S)


def _unnumbered(text):
    r"""Display equations in *text* carrying neither a Quarto anchor nor a ``\\tag``.

    Matched by capturing the optional anchor, never by a negative lookahead: `$$.+?$$` followed by `(?!...)` backtracks *past the closing delimiter* to satisfy the lookahead, so it silently reports whatever makes the assertion pass. The first version of this check did exactly that and could not fail.
    """
    return [" ".join(m.group(1).split())[:60] for m in _BLOCK.finditer(text) if not m.group(3) and "\\tag{" not in m.group(1)]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    path = tmp_path_factory.mktemp("study") / "Mini.yaml"
    path.write_text(RECIPE)
    return SimulationStudy.from_file(str(path))


def test_a_shared_model_is_written_once(study):
    """Experiments 1 and 2 declare the same model, so it appears once, not twice."""
    report = study.report("qmd", part="all")
    assert report.count("Linear decay with drive") >= 1
    assert report.count(r"\dot{x}") == 2, "one base equation plus the variant's redefinition"
    assert report.count("| $k$ |") == 1, "the symbol table is not repeated per experiment"


def test_a_variant_contributes_only_its_delta(study):
    """The driven model redefines one equation; nothing else of it is reprinted."""
    report = study.report("qmd", part="all")
    assert "use**" not in report and "uses **Linear decay with drive**" in report
    assert "drive" in report.split("uses **Linear decay with drive**")[1]


def test_sharing_an_auxiliary_variable_does_not_merge_two_systems(study):
    """A shared readout variable is not evidence of a shared model.

    Membership is subset-or-superset of the family's first model, never bare overlap.
    Overlap merged Pang2023's wave field with its BEI mass model because both carry the four Balloon-Windkessel haemodynamic variables, and the report then presented a mass model the paper never published as a *variant* of the wave field.
    """
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    families = report.model_families(exps)
    assert len(families) == 2, [str(f.label) for f in families]
    decay, oscillator = families
    assert {e.id for e in decay.experiments} == {1, 2, 3, 4}
    assert {e.id for e in oscillator.experiments} == {5}


def test_a_superset_of_the_state_stays_in_the_family(study):
    """Adding a state variable extends a model; it does not make a new one."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    decay = report.model_families(exps)[0]
    labels = [str(report.slot(m.model, "label", "")) for m in decay.models]
    assert "Linear decay with a readout" in labels, labels


def test_every_emitted_table_is_captioned(study):
    """An uncaptioned table still steps LaTeX's table counter, so it shifts every number.

    Pang2023's events table was the one left bare, and it pushed the report's first captioned table to "Table 2" — the reader sees a document whose tables start at two, and the float they cannot see is the one that took the number.
    """
    import re

    report = study.report("qmd", part="all")
    lines = report.splitlines()
    tables = sum(1 for i, line in enumerate(lines) if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|")))
    captions = len(re.findall(r"^: .*\{#tbl-[a-z0-9-]+\}", report, re.M))
    assert tables == captions, f"{tables} tables but {captions} captions"


def test_every_anchor_is_unique(study):
    """A duplicate anchor makes every reference to it ambiguous, and Quarto is silent."""
    import re

    anchors = re.findall(r"\{#((?:eq|tbl)-[a-z0-9-]+)\}", study.report("qmd", part="all"))
    assert anchors, "the qmd format must emit anchors"
    assert len(anchors) == len(set(anchors)), f"duplicate anchors: {anchors}"


def test_part_selects_which_experiments_get_a_paragraph(study):
    """`part` moves an experiment's prose, and never hides it from the comparison table."""
    main, supplementary = study.report("qmd"), study.report("qmd", part="supplementary")
    assert "Adds a constant drive." not in main
    assert "Adds a constant drive." in supplementary
    assert "Establishes the resting decay." in main
    assert "| 3 |" in main, "a demoted experiment still has its row in the table"


def test_the_comparison_table_carries_only_what_varies(study):
    """Duration differs across the three; the solver and step size do not."""
    report = study.report("qmd", part="all")
    table = report.split("Experiments using")[0].rsplit("| Exp |", 1)[-1]
    assert "Duration" in table
    assert "Method" not in table and "Delta t" not in table


def test_time_carries_the_integrator_unit(study):
    """The integration block used to hardcode ms, reporting a 1 s run as 1 ms."""
    assert "1 s" in study.report("qmd") and "1 ms" not in study.report("qmd")


def test_markdown_numbers_equations_where_it_cannot_anchor(study):
    r"""Plain markdown has no anchor syntax, so numbering falls back to \\tag."""
    report = study.report("markdown", part="all")
    assert r"\tag{1}" in report and "{#eq-" not in report


def test_every_display_equation_carries_a_number(study):
    """One unnumbered equation is one the prose cannot cite, and nothing flags it.

    The coupling equation was rendered by the coupling's own template, which had no access to the report's numbering — so across ten studies every state equation was numbered and the eleven equations joining the nodes into a network were not.
    """
    for fmt in ("qmd", "markdown"):
        bare = _unnumbered(study.report(fmt, part="all"))
        assert not bare, f"{fmt}: unnumbered {bare}"


def test_the_coupling_equation_is_numbered_with_the_rest(study):
    """The bug this pins: the coupling rendered through its own template, which had no access to the report's numbering, so it emitted bare `$$`."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    assert _unnumbered(report.coupling_prose(exps)), "fixture must exercise the coupling path"
    assert not _unnumbered(report.coupling_prose(exps, report.Equations("semantic", "qmd")))


def test_a_declared_state_variable_always_renders_its_equation(study):
    """A state variable whose equation never renders leaves a hole nothing reports.

    Heterogeneous networks declare their models as plain datamodel objects, which lack the symbolic machinery; treating that as "no equations" gave Mongillo2008's twenty-nine experiments a Methods section with no mathematics at all.
    """
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    for family in report.model_families(exps):
        for entry in family.models:
            declared = set(dict(report.name_items(report.slot(entry.model, "state_variables", {}) or {})))
            rendered = {name for name, _ in report.model_equations(entry.model, "state")}
            assert declared <= rendered, f"{family.label}: {declared - rendered} never rendered"


def test_equations_none_leaves_them_bare(study):
    report = study.report("qmd", part="all", equations="none")
    assert "{#eq-" not in report and r"\tag{" not in report


def test_orientation_can_be_pinned(study):
    """`auto` keeps the table narrow; pinning it keeps the Methods' shape stable."""
    rows = study.report("qmd", part="all", orient="rows")
    columns = study.report("qmd", part="all", orient="columns")
    assert rows != columns


@pytest.mark.parametrize("bad", ["latex", "docx"])
def test_unknown_format_is_refused(study, bad):
    with pytest.raises(ValueError):
        study.report(bad)


def test_unknown_part_is_refused(study):
    with pytest.raises(ValueError):
        study.report("qmd", part="appendix")


def test_a_label_does_not_repeat_the_id_the_heading_carries(study):
    """Recipes open a label with the experiment's own number, giving "Experiment 30: Exp 30 …".

    Six of Schirner2023's ten read that way, and the dash the recipe used to attach the number went with it.
    """
    from tvbo.utils.report import experiment_title

    exp = SimpleNamespace(id=30, label="Exp 30 — FIC+EIB tuning (group-avg)")
    assert experiment_title(exp) == "FIC+EIB tuning (group-avg)"
    assert experiment_title(SimpleNamespace(id=3, label="Experiment 3: driven")) == "driven"
    assert experiment_title(SimpleNamespace(id=7, label="Isolated column")) == "Isolated column"
    assert experiment_title(SimpleNamespace(id=7, label="Exp 70 revisited")) == "Exp 70 revisited"


def test_an_identity_half_of_the_coupling_is_a_clause_not_an_equation(study):
    """`c_pre = local_states` and `c_post = gx` state nothing, and `local_states` is an alias token `symbolic()` substitutes — printed raw it typesets as a variable."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    prose = report.coupling_prose(exps, report.Equations("semantic", "qmd"))
    assert "no post-synaptic transformation" in prose
    assert "local_states" not in prose and "c_{\\text{post}}" not in prose


# ── An equation reaches the report by being rendered, never by being typed ──────────────


def test_a_hand_written_equation_is_reported(tmp_path):
    """Pang2023 hand-set the paper's PDE above a section explaining it is not integrated."""
    from tvbo.utils.report import unrendered_equations

    qmd = tmp_path / "report.qmd"
    qmd.write_text("# Methods\n\nThe model is\n\n$$\\dot{x} = -k x.$$\n\nand so on.\n")
    assert unrendered_equations(qmd) == [(5, "\\dot{x} = -k x.")]
    assert unrendered_equations(str(qmd)) == [(5, "\\dot{x} = -k x.")]


def test_a_generated_equation_is_not_reported(tmp_path):
    """Equations `study.report()` emits live inside an executable cell and are the point."""
    from tvbo.utils.report import unrendered_equations

    qmd = tmp_path / "report.qmd"
    qmd.write_text('# Methods\n\n```{python}\n#| echo: false\nprint(STUDY.report("qmd"))   # emits $$\\dot{x} = -k x$$\n```\n')
    assert unrendered_equations(qmd) == []


def test_the_reported_line_number_survives_a_stripped_cell(tmp_path):
    """The cell is replaced by its own newlines, so a later equation keeps its true line."""
    from tvbo.utils.report import unrendered_equations

    qmd = tmp_path / "report.qmd"
    qmd.write_text("```{python}\na = 1\nb = 2\n```\n\n$$E = mc^2$$\n")
    assert unrendered_equations(qmd) == [(6, "E = mc^2")]


# ── A grid too small to be a float is a sentence ────────────────────────────────────────


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([["`Q`", "stimulus"]], "Event `Q` (Type: stimulus)."),
        ([["50", "2000 ms"], ["51", "7000 ms"]], "Event 50 (Type: 2000 ms); Event 51 (Type: 7000 ms)."),
    ],
)
def test_a_grid_too_small_to_be_a_float_is_written_as_a_sentence(rows, expected):
    """A captioned float tells the reader to look something up; two numbers do not earn one.

    Pang2023 spent a numbered table on the fact that its model declares one event, and Schirner2023 spent one on two experiments differing only in duration.
    """
    from tvbo.utils.report import table_or_prose

    assert table_or_prose(["Event", "Type"], rows) == expected


def test_a_grid_large_enough_stays_a_table():
    from tvbo.utils.report import table_or_prose

    assert table_or_prose(["Event", "Type"], [["a", "1"], ["b", "2"], ["c", "3"]]).startswith("|")


def test_a_one_row_table_still_renders_as_a_table():
    """The shared primitive does not collapse on row count: 13 curated models are single-state.

    `state_variable_table`, `param_table` and the scorecard have no subject column for a sentence to name, and `read_md_tables` is documented as md_table's inverse.
    """
    from tvbo.utils.report import md_table, read_md_tables

    one_row = md_table(["Parameter", "Value"], [["sigma", "0.01"]])
    assert one_row.startswith("|")
    assert read_md_tables(one_row)[0].rows == [{"Parameter": "sigma", "Value": "0.01"}]


def test_a_grid_down_to_one_column_is_a_list_not_a_table():
    """A one-column float spends a number and a caption restating the heading above it.

    Column dropping gets there on its own: a coupling whose terms carry no value, unit or description leaves `| Term |` and nothing else.
    """
    from tvbo.utils.report import md_table

    assert md_table(["Term", "Value"], [["$c_1$", ""], ["$c_2$", ""]]) == "$c_1$, $c_2$"
    assert md_table(["A", "B"], [["", ""], ["", ""]]) == ""


def test_prose_keeps_its_caption_as_a_lead_in():
    """The observations caption carries the settings lifted out of the rows.

    Dropping it on the prose path took `time_scale = ms` — chosen by nobody, stated nowhere else — out of the report entirely.
    """
    from tvbo.utils.report import captioned

    out = captioned("Observation bold (Source: S).", "What each records. Throughout, time_scale = ms.", "obs", "qmd")
    assert out == "What each records. Throughout, time_scale = ms.\n\nObservation bold (Source: S).\n"
    assert "tbl-" not in out


def test_experiment_ids_are_listed_in_numeric_order():
    """Ordering ids as text gives 1, 2, 20, 21, 3 — Deco2014 listed ten exactly that way."""
    from tvbo.utils.report import _id_text

    assert _id_text([1, 2, 20, 21, 3, 30]) == "1, 2, 3, 20, 21, 30"


def test_a_pipeline_step_is_named_by_what_the_recipe_calls_it():
    """Reading `callable` first printed Deco2014's five named steps as `? → ? → fftconvolve → ? → ?`."""
    from tvbo.utils.report import pipeline_text

    steps = [
        SimpleNamespace(name="hemodynamic_response"),
        SimpleNamespace(name="convolve", callable=SimpleNamespace(name="fftconvolve")),
    ]
    assert pipeline_text(steps) == "hemodynamic_response → convolve"


# ── The observation table stays dense ───────────────────────────────────────────────────


def test_a_setting_every_observation_shares_is_stated_once(study):
    """A study declares its clock once, so `time_unit` was printed on all 34 rows of one study's table and all 29 of another — the same value, on every line."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    obs = report.observation_table(exps)
    assert "scale = s" in obs.shared
    assert "scale=s" not in obs.table


def test_an_observation_description_becomes_prose_not_a_column(study):
    """Descriptions are paragraphs; as a cell they widen the table for every row to serve a few."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    obs = report.observation_table(exps)
    assert "Averaged over the post-transient window." in obs.notes
    assert "Averaged over the post-transient window." not in obs.table
    assert "Description" not in obs.table.splitlines()[0]


def test_sampling_and_pipeline_are_one_column(study):
    """Each was under half full and both answer *how the raw state becomes the quantity*."""
    from tvbo.utils import report

    exps = [study.get_experiment(i) for i in study.experiment_ids()]
    header = report.observation_table(exps).table.splitlines()[0]
    assert "Reduction" in header
    assert "Pipeline" not in header and "Sampling" not in header


# ── The coupling's symbols live in the model's glossary ─────────────────────────────────


def _param(value, unit="", description=""):
    return SimpleNamespace(value=value, unit=unit, description=description)


def test_a_coupling_parameter_the_model_already_declares_is_not_listed_twice():
    """Jansen1995's coupling restates the model's sigmoid constants at the model's values.

    They were a second, uncaptioned table after the coupling block — three rows the reader had just read in the glossary.
    """
    from tvbo.utils.report import symbol_table

    model = SimpleNamespace(state_variables={}, derived_parameters={}, parameters={"e0": _param(2.5), "r": _param(0.56)})
    coupling = SimpleNamespace(parameters={"e0": _param(2.5), "K": _param(1.0, description="gain")})
    table = symbol_table(model, couplings=[coupling])
    assert table.count("$e_{0}$") == 1
    assert "| $K$ | coupling |" in table


def test_a_value_written_two_ways_is_one_setting():
    """`6` and `6.0` compared as text are two settings, and the report then invents a difference: Jansen1995's glossary listed $v_0$ twice, once as a symbol the coupling supposedly introduces."""
    from tvbo.utils.report import symbol_table

    model = SimpleNamespace(state_variables={}, derived_parameters={}, parameters={"v0": _param(6)})
    coupling = SimpleNamespace(parameters={"v0": _param(6.0)})
    assert symbol_table(model, couplings=[coupling]).count("$v_{0}$") == 1


def _edge(**kw):
    """An explicit edge stub: the attributes the readers ask for, absent ones as None."""
    return SimpleNamespace(parameters=None, weight=0.5, **{"delay": None, "distance": None, **kw})


def _delay_experiment(net):
    return SimpleNamespace(
        id=1,
        references=[],
        network=net,
        connectivity=None,
        part="main",
        integration=SimpleNamespace(unit="ms", method="Heun", step_size=None, duration=None, transient_time=None),
        dynamics=SimpleNamespace(parameters={}),
        explorations=None,
    )


def test_a_network_that_measures_tract_lengths_is_reported_by_its_speed():
    """The comparison table says what the backend integrates, and lengths win over edge delays.

    `graph_selection` lowers a network carrying both onto a length graph, whose delays are lengths / conduction_speed; reporting the edge delay there prints a number no cell ever runs with.
    """
    from tvbo.utils.report import experiment_facts

    speed = SimpleNamespace(value=3.0, unit="mm_per_ms")
    on_edges = _delay_experiment(SimpleNamespace(number_of_nodes=2, edges=[_edge(delay=2.0)], conduction_speed=speed))
    on_lengths = _delay_experiment(
        SimpleNamespace(number_of_nodes=2, edges=[_edge(delay=2.0, distance=30.0)], conduction_speed=speed)
    )
    assert experiment_facts(on_edges)["Delays"] == "2 ms"
    assert experiment_facts(on_lengths)["Delays"] == "3 mm/ms"
