"""`SimulationStudy.report` writes each shared thing once.

A per-experiment Methods section repeats the model for every experiment that uses it:
Jansen1995's seven experiments emitted 1209 lines and about thirty tables, of which the
six state equations, the symbol table and the parameter table were the same text over and
over. These tests pin the contract that removes that — one system per family, deltas for
variants, one comparison table carrying only what differs — and the two properties a
generated cross-referenced document breaks silently without: unique anchors, and no
number left to drift.
"""

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
    dynamics: *base
    integration: *integration
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
"""


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
    """Plain markdown has no anchor syntax, so numbering falls back to \\tag."""
    report = study.report("markdown", part="all")
    assert r"\tag{1}" in report and "{#eq-" not in report


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
