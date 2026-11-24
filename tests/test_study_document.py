"""A study's declared `document`: every block kind, every refusal, and the README rendered from the lifecycle study."""

import ast
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tvbo import SimulationStudy
from tvbo.plot import backends, lifecycle
from tvbo.utils import document

REPO = Path(__file__).resolve().parents[1]
LIFECYCLE = REPO / "docs" / "0-SimulationStudyLifecycle"

NESTED = """
citekey: Inner
title: A nested study
experiments:
  - id: 1
    label: one node
    # the curated model, unchanged
    dynamics: {iri: tvbo:models/Generic2dOscillator}
    integration: {method: Heun, step_size: 0.1, duration: 10}
figures:
  - name: trace
    label: "The trace"
"""

OUTER = """
citekey: Outer
title: A study holding one
studies:
  - !include inner/study.yaml
document:
  assets: assets
  sections:
    - title: Intro
      blocks:
        - {kind: text, text: "Hello."}
    - title: Folded
      collapsed: true
      blocks:
        - {kind: spec, study: Inner, select: experiments.1.dynamics}
"""


def _study(tmp_path, sections=None):
    """The outer study written under *tmp_path*, its document's sections replaced by *sections* when given."""
    (tmp_path / "inner").mkdir(exist_ok=True)
    (tmp_path / "inner" / "study.yaml").write_text(NESTED)
    outer = OUTER if sections is None else OUTER.split("  sections:")[0] + "  sections:\n" + sections
    (tmp_path / "study.yaml").write_text(outer)
    return SimulationStudy.from_file(str(tmp_path / "study.yaml"))


def _blocks(*lines):
    return "    - blocks:\n" + "".join(f"        - {line}\n" for line in lines)


def test_text_and_collapsed_section(tmp_path):
    text = _study(tmp_path).report("gfm")
    assert text.startswith("## Intro\n\nHello.")
    assert "<details>\n<summary><b>Folded</b></summary>\n\n```yaml\ndynamics:" in text


def test_spec_cuts_the_authors_lines_through_an_include(tmp_path):
    _study(tmp_path)
    cut = document.cut(tmp_path / "study.yaml", "studies.Inner.experiments.1")
    assert cut.splitlines()[0] == "id: 1"
    assert "# the curated model, unchanged" in cut
    assert document.cut(tmp_path / "study.yaml", "studies.Inner").startswith("citekey: Inner")


@pytest.mark.parametrize(
    "select, message", [("experiments.2", "declares '2'"), ("nope", "no key 'nope'"), ("title.x", "below a scalar")]
)
def test_spec_selector_that_matches_nothing_fails_by_name(tmp_path, select, message):
    _study(tmp_path)
    with pytest.raises(ValueError, match=message):
        document.cut(tmp_path / "inner" / "study.yaml", select)


def test_usage_is_cumulative_and_resolves(tmp_path):
    study = _study(
        tmp_path,
        _blocks(
            "{kind: usage, study: Inner, operation: load}",
            "{kind: usage, study: Inner, operation: generate, experiment: '1', backend: jax}",
            "{kind: usage, study: Inner, operation: share, experiment: '1'}",
        ),
    )
    snippets = [b.split("```python\n")[1] for b in study.report("gfm").split("```\n") if "```python" in b]
    assert snippets[0] == 'from tvbo import SimulationStudy\n\nstudy = SimulationStudy.from_file("inner/study.yaml")\n'
    assert snippets[1] == 'print(study.get_experiment(1).render_code("jax"))\n'
    assert snippets[2].startswith("from tvbo.platform import TVBOPlatform\n\nplatform = TVBOPlatform(")


def test_every_usage_operation_parses_and_names_real_attributes():
    from tvbo.classes.study import StudyResult
    from tvbo.platform import TVBOPlatform

    owners = {"study": SimulationStudy, "result": StudyResult, "platform": TVBOPlatform}
    assert list(document.USAGE) == ["load", "generate", "run", "figure", "report", "share"]
    shown = (set(), {})
    for operation in document.USAGE:
        code = document.usage(operation, path="s.yaml", experiment=1, backend="jax", figure="f", shown=shown)
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in owners:
                assert hasattr(owners[node.value.id], node.attr), f"{operation}: {node.value.id}.{node.attr}"
    assert 'get_experiment("baseline")' in document.usage("generate", path="s.yaml", experiment="baseline", backend="jax")


def test_usage_missing_a_slot_is_refused(tmp_path):
    study = _study(tmp_path, _blocks("{kind: usage, study: Inner, operation: generate, experiment: '1'}"))
    with pytest.raises(ValueError, match=r"a generate usage needs \['backend'\]"):
        study.report("gfm")


def test_symbol_source_keeps_decorators_and_dedents():
    code = "import x\n\nclass A:\n    @staticmethod\n    def f():\n        return 1\n"
    assert document.symbol_source(code, "A.f") == "@staticmethod\ndef f():\n    return 1"
    with pytest.raises(ValueError, match="defines no 'A.g'"):
        document.symbol_source(code, "A.g")
    with pytest.raises(ValueError, match="not Python"):
        document.symbol_source("function f() end", "f", "julia")


def test_generated_code_block(tmp_path):
    study = _study(
        tmp_path,
        _blocks(
            "{kind: generated_code, study: Inner, experiment: '1', backend: tvboptim, symbol: Generic2dOscillator.dynamics}"
        ),
    )
    text = study.report("gfm")
    assert text.startswith("```python\ndef dynamics(") and "dV_dt" in text


def test_slot_a_kind_does_not_read_is_refused(tmp_path):
    study = _study(tmp_path, _blocks("{kind: text, text: hi, select: experiments}"))
    with pytest.raises(ValueError, match=r"a text block reads \['text'\], not \['select'\]"):
        study.report("gfm")


def test_unknown_nested_study_is_refused(tmp_path):
    study = _study(tmp_path, _blocks("{kind: methods, study: Nobody}"))
    with pytest.raises(ValueError, match="no nested study is labelled 'Nobody'"):
        study.report("gfm")


def test_methods_block_is_the_nested_studys_methods(tmp_path):
    study = _study(tmp_path, _blocks("{kind: methods, study: Inner}"))
    inner = dict(study.nested_studies())["Inner"]
    assert study.report("gfm").strip() == inner.methods("markdown", level=3).strip()


def test_figure_and_tree_need_a_run_or_reused_assets(tmp_path):
    study = _study(tmp_path, _blocks("{kind: figure, study: Inner, figure: trace}", "{kind: tree, study: Inner, depth: 1}"))
    with pytest.raises(ValueError, match="a figure block shows a run"):
        study.report("gfm")
    with pytest.raises(ValueError, match=r"expected one figure-Inner-trace\.\* .* found \[\]"):
        study.report("gfm", reuse_assets=True)


def test_figure_and_tree_from_a_result_then_reused(tmp_path):
    study = _study(tmp_path, _blocks("{kind: figure, study: Inner, figure: trace}", "{kind: tree, study: Inner, depth: 1}"))
    root = tmp_path / "inner"
    (root / "trace.png").write_bytes(b"png")
    image = SimpleNamespace(name="trace", path=root / "trace.png")
    inner = SimpleNamespace(figure=lambda name: image, root=root, studies={})
    result = SimpleNamespace(studies={"Inner": inner})
    fresh = study.report("gfm", result=result)
    assert "![The trace](assets/figure-Inner-trace.png)" in fresh
    assert "inner/\n├── study.yaml\n└── trace.png" in fresh
    assert study.report("gfm", reuse_assets=True) == fresh


def test_catalogue(tmp_path):
    study = _study(tmp_path, _blocks("{kind: catalogue, of: [Dynamics], entries: ['tvbo:models/Generic2dOscillator']}"))
    text = study.report("gfm")
    assert (
        "| `Dynamics` |" in text and "Generic2dOscillator<br><sub><code>tvbo:models/Generic2dOscillator</code></sub>" in text
    )
    with pytest.raises(ValueError, match="no records of class 'Nothing'"):
        _study(tmp_path, _blocks("{kind: catalogue, of: [Nothing]}")).report("gfm")


def test_backends_block(tmp_path):
    assert _study(tmp_path, _blocks("{kind: backends, form: table}")).report("gfm").strip() == backends.markdown().strip()
    picture = _study(tmp_path, _blocks("{kind: backends}")).report("gfm")
    assert 'srcset="assets/backends-dark.svg"' in picture and 'src="assets/backends-light.svg"' in picture
    assert (tmp_path / "assets" / "backends-dark.svg").read_text() == backends.svg("tvbo:themes/dark")


def test_lifecycle_block_in_gfm_and_qmd(tmp_path):
    study = _study(tmp_path, _blocks("{kind: lifecycle, band: numbers, icons: fa}"))
    assert "<picture>" in study.report("gfm")
    assert study.report("qmd").startswith("![The Simulation Study Lifecycle")
    assert (tmp_path / "assets" / "lifecycle-light.svg").read_text() == lifecycle.wheel(band="numbers", icons="fa")


@pytest.mark.parametrize("theme", ["tvbo:themes/default", "tvbo:themes/dark"])
@pytest.mark.parametrize("band", ["icons", "numbers"])
@pytest.mark.parametrize("icons", ["line", "fa"])
def test_wheel_variants(theme, band, icons):
    svg = lifecycle.wheel(theme, band=band, icons=icons)
    assert svg.startswith("<svg") and 'role="img"' in svg
    assert all(f"stage-{key}" in svg for key in lifecycle.stages())
    assert "<image" not in svg and 'href="http' not in svg
    assert svg == lifecycle.wheel(theme, band=band, icons=icons)


def test_report_dispatch(tmp_path):
    study = _study(tmp_path)
    with pytest.raises(ValueError, match="takes no Methods options"):
        study.report("gfm", part="all")
    inner = dict(study.nested_studies())["Inner"]
    assert inner.report("markdown") == inner.methods("markdown")


def test_readme_is_the_lifecycle_studys_render(tmp_path):
    """README.md is what the lifecycle study renders now, and every drawn asset it links is current; run `make readme` when this fails."""
    docs = tmp_path / "docs"
    shutil.copytree(LIFECYCLE, docs / LIFECYCLE.name, ignore=shutil.ignore_patterns("*.py", "*.qmd"))
    (docs / "examples" / "first-simulation").mkdir(parents=True)
    shutil.copy(REPO / "docs" / "examples" / "first-simulation" / "study.yaml", docs / "examples" / "first-simulation")
    study = SimulationStudy.from_file(str(docs / LIFECYCLE.name / "study.yaml"))
    text = study.report("gfm", outputfile=tmp_path / "README.md", reuse_assets=True)
    assert text == (REPO / "README.md").read_text(encoding="utf-8")


def test_asset_url_links_markdown_and_quarto_links_relative(tmp_path):
    study = _study(tmp_path, _blocks("{kind: backends}"))
    study.document.asset_url = "https://example.org/assets"
    assert 'src="https://example.org/assets/backends-light.svg"' in study.report("gfm")
    assert study.report("qmd").strip() == "![What each simulation backend can do](assets/backends-light.svg)"


def test_a_block_limited_to_formats_renders_only_in_them(tmp_path):
    study = _study(tmp_path, _blocks("{kind: text, text: badges, formats: [gfm]}", "{kind: text, text: pitch}"))
    assert study.report("gfm") == "badges\n\npitch\n"
    assert study.report("qmd") == "pitch\n"


def test_a_stage_section_opens_with_the_stage_icon(tmp_path):
    sections = "    - {title: Run it, stage: run, collapsed: true, blocks: [{kind: text, text: go}]}\n"
    study = _study(tmp_path, sections)
    gfm = study.report("gfm")
    assert '<summary><picture><source media="(prefers-color-scheme: dark)" srcset="assets/stage-run-dark.svg">' in gfm
    assert (
        '<img src="assets/stage-run-light.svg" alt="" width="20" height="20" align="top"></picture> <b>Run it</b></summary>'
        in gfm
    )
    assert (tmp_path / "assets" / "stage-run-light.svg").read_text() == lifecycle.icon("run")
    assert study.report("qmd").startswith('<details>\n<summary><img src="assets/stage-run-light.svg"')
    with pytest.raises(ValueError, match="'walk' is not a lifecycle stage"):
        _study(tmp_path, sections.replace("stage: run", "stage: walk")).report("gfm")


def test_spec_excerpt_stops_before_the_comments_of_the_next_item(tmp_path):
    recipe = tmp_path / "s.yaml"
    recipe.write_text(
        "experiments:\n  - id: 1\n    label: a\n\n  # the second doubles the step\n  - id: 2\n    label: b\n# figures follow\nfigures: []\n"
    )
    assert document.cut(recipe, "experiments.1") == "id: 1\nlabel: a"
    assert document.cut(recipe, "experiments.2") == "id: 2\nlabel: b"


def test_usage_reloads_a_study_the_name_no_longer_holds_and_needs_its_run():
    shown = (set(), {})
    document.usage("load", path="a.yaml", shown=shown)
    document.usage("load", path="b.yaml", shown=shown)
    assert document.usage("generate", path="a.yaml", experiment=1, backend="jax", shown=shown).startswith(
        'study = SimulationStudy.from_file("a.yaml")'
    )
    with pytest.raises(ValueError, match="a report usage reads `result`; put a run usage of a.yaml before it"):
        document.usage("report", path="a.yaml", shown=shown)
    document.usage("run", path="b.yaml", shown=shown)
    with pytest.raises(ValueError, match="run usage of a.yaml"):
        document.usage("figure", path="a.yaml", figure="f", shown=shown)


@pytest.mark.parametrize("selector", ["one node", "1"])
def test_experiment_selected_as_tvbo_run_selects_it(tmp_path, selector):
    study = _study(tmp_path, _blocks(f"{{kind: usage, study: Inner, operation: share, experiment: {selector}}}"))
    assert "study.get_experiment(1)" in study.report("gfm")


def test_an_experiment_selector_that_matches_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="a usage block: No experiment\\(s\\) matching 'two' in study"):
        _study(tmp_path, _blocks("{kind: usage, study: Inner, operation: share, experiment: two}")).report("gfm")


def test_a_folded_section_keeps_its_heading_in_the_pdf_text(tmp_path):
    sections = "    - {title: Run it, stage: run, collapsed: true, blocks: [{kind: text, text: go}]}\n"
    assert _study(tmp_path, sections).report("pdf") == "## Run it\n\ngo\n"


def test_a_rerun_figure_replaces_the_one_before_and_reuse_names_it_alike(tmp_path):
    study = _study(tmp_path, _blocks("{kind: figure, study: Inner}"))
    root = tmp_path / "inner"
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "figure-Inner-trace.gif").write_bytes(b"old")
    (root / "trace.png").write_bytes(b"png")
    result = SimpleNamespace(
        studies={"Inner": SimpleNamespace(figure=lambda name: SimpleNamespace(name=name, path=root / "trace.png"), studies={})}
    )
    fresh = study.report("gfm", result=result)
    assert sorted(p.name for p in (tmp_path / "assets").iterdir()) == ["figure-Inner-trace.png"]
    assert fresh == study.report("gfm", reuse_assets=True) == "![The trace](assets/figure-Inner-trace.png)\n"


@pytest.mark.parametrize("still, shown", [("still: 2", "png"), ("frames: 4", "gif")])
def test_an_animated_figure_shows_the_still_it_declares(tmp_path, still, shown):
    """A figure block shows the still an animation declares, which `figure_still` names as the render loop does, and the movie otherwise."""
    from tvbo.run.study import figure_still

    _study(tmp_path, _blocks("{kind: figure, study: Inner}"))
    root = tmp_path / "inner"
    inner = root / "study.yaml"
    inner.write_text(
        inner.read_text().replace('label: "The trace"', f'label: "The trace"\n    animation: {{over: time, {still}}}')
    )
    study = SimulationStudy.from_file(str(tmp_path / "study.yaml"))
    for suffix in ("gif", "png"):
        (root / f"trace.{suffix}").write_bytes(suffix.encode())
    (figure,) = dict(study.walk_studies(include_self=False))["Inner"].figures
    assert figure_still(figure, root / "trace.gif") == (root / "trace.png" if shown == "png" else None)
    result = SimpleNamespace(
        studies={"Inner": SimpleNamespace(figure=lambda name: SimpleNamespace(name=name, path=root / "trace.gif"), studies={})}
    )
    assert study.report("gfm", result=result) == f"![The trace](assets/figure-Inner-trace.{shown})\n"
    assert (tmp_path / "assets" / f"figure-Inner-trace.{shown}").read_bytes() == shown.encode()


def test_reuse_writes_nothing_and_refuses_a_stale_drawing(tmp_path):
    study = _study(tmp_path, _blocks("{kind: backends}"))
    with pytest.raises(ValueError, match="backends-light.svg is missing or stale"):
        study.report("gfm", reuse_assets=True)
    assert not (tmp_path / "assets").exists()
    study.report("gfm")
    (tmp_path / "assets" / "backends-dark.svg").write_text("<svg/>")
    with pytest.raises(ValueError, match="backends-dark.svg is missing or stale"):
        study.report("gfm", reuse_assets=True)


def test_a_run_returns_every_nested_result_at_every_depth(tmp_path, monkeypatch):
    from tvbo.run import study as study_run

    monkeypatch.setattr(study_run, "run_tree", lambda *a, **k: None)
    (tmp_path / "mid").mkdir()
    (tmp_path / "mid" / "inner.yaml").write_text("citekey: Inner\ntitle: Inner\nexperiments: []\n")
    (tmp_path / "mid" / "study.yaml").write_text("citekey: Mid\ntitle: Mid\nstudies:\n  - !include inner.yaml\n")
    (tmp_path / "study.yaml").write_text("citekey: Outer\ntitle: Outer\nstudies:\n  - !include mid/study.yaml\n")
    result = SimulationStudy.from_file(str(tmp_path / "study.yaml")).run()
    assert list(result.studies["Mid"].studies) == ["Inner"]
