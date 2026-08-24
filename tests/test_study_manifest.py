"""The study-of-studies container: manifest resolution, verify gates, and composed captions.

Phase 0 of the native-manuscript design. A study-of-studies is a `SimulationStudy` with `members:` whose `results:` are the numbers the prose cites (computed from a container or authored from prior work), whose figures carry `Panel.description` so captions compose from the spec, and whose buildability `tvbo verify` checks. These pin that plumbing so a regression can't silently ship a wrong number, an orphan figure, or a caption that disagrees with its panels.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.data import study_manifest as I


def _write_container(root: Path, name: str, var: str, value) -> None:
    """Write a scalar analysis container where `resolve_dataref` looks for it.

    Through `analysis_container_path`, so the fixture cannot name the container differently from the code under test.
    """
    from tvbo.data.dataref import analysis_container_path

    path = analysis_container_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({var: xr.DataArray(np.array(value))}).to_netcdf(path, engine="h5netcdf")


def _write_digest(root: Path, name: str, analysis) -> None:
    """Record ``analysis``'s declaration digest in its container's sidecar, as a run would."""
    import yaml

    from tvbo.data.dataref import analysis_container_path, sidecar_path

    sidecar = sidecar_path(analysis_container_path(root, name))
    sidecar.write_text(yaml.safe_dump({"declaration_digest": I._analysis_fingerprint(analysis)}))


def _member_results(tmp_path: Path) -> Path:
    """Where the ``toy`` member keeps its results, asked of the layout rather than spelled out."""
    from tvbo.utils.study_layout import study_path

    return study_path("results", root=tmp_path / "members")


@pytest.fixture
def collection(tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text("title: Toy\nkey: toy\nsimulation_experiments: []\n", encoding="utf-8")
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: TVB-O\n"
        "citekey: tvbo_manuscript\n"
        "members:\n"
        "  - {recipe: members/toy.yaml, label: toy}\n"
        "  - {recipe: members/heavy.yaml, label: heavy, optional: true}\n"
        "results:\n"
        "  - {key: n_errors, used: {iri: 'tvbo:ana/toy/tally', output: n_errors}, format: '{:d}'}\n"
        "  - {key: parcels, value: '379', source: Glasser2016, description: HCP parcels}\n",
        encoding="utf-8",
    )
    return tvbo.SimulationStudy.from_file(str(spec))


def test_from_file_loads_members_results_archive(collection):
    inv = collection
    assert isinstance(inv, tvbo.SimulationStudy)
    assert [m.label for m in inv.members] == ["toy", "heavy"]
    assert {r.key for r in inv.results} == {"n_errors", "parcels"}


def test_optional_member_dropped_unless_requested(collection):
    light = [lbl for lbl, _ in collection.member_recipes(include_optional=False)]
    full = [lbl for lbl, _ in collection.member_recipes(include_optional=True)]
    assert light == ["toy"]
    assert full == ["toy", "heavy"]


def test_authored_value_resolves_without_a_container(collection, tmp_path):
    results, prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert results["parcels"] == "379"
    assert prov["parcels"] == {"computed": False, "value": "379", "source": "Glasser2016", "description": "HCP parcels"}
    assert any(p.startswith("n_errors:") for p in problems)  # no container yet


def test_computed_value_resolves_and_formats(collection, tmp_path):
    _write_container(_member_results(tmp_path), "tally", "n_errors", 17)
    results, prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert problems == []
    assert results["n_errors"] == "17"
    assert prov["n_errors"]["computed"] is True


def test_emit_manifest_writes_quarto_meta_shape(collection, tmp_path):
    _write_container(_member_results(tmp_path), "tally", "n_errors", 17)
    out, problems = I.emit_manifest(
        collection, tmp_path / "derivatives" / "tvbo", tmp_path / "_output" / "manuscript_results.yml"
    )
    assert problems == []
    import yaml

    payload = yaml.safe_load(out.read_text())
    assert payload["results"] == {"n_errors": "17", "parcels": "379"}


def test_verify_flags_missing_members_and_dead_keys(collection, tmp_path):
    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    problems = I.verify(collection, tmp_path)
    assert any("n_errors" in p for p in problems)  # container missing -> dead key
    _write_container(_member_results(tmp_path), "tally", "n_errors", 17)
    assert I.verify(collection, tmp_path) == []


def test_verify_coverage_is_bidirectional(collection, tmp_path):
    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    _write_container(_member_results(tmp_path), "tally", "n_errors", 17)
    problems = I.verify(collection, tmp_path, manuscript_keys={"n_errors", "ghost"})
    assert any("ghost" in p and "cited in the manuscript" in p for p in problems)
    assert any("parcels" in p and "never cited" in p for p in problems)


def test_verify_against_committed_manifest_is_container_free(collection, tmp_path):
    import yaml

    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    manifest = tmp_path / "manuscript_results.yml"
    manifest.write_text(yaml.safe_dump({"results": {"n_errors": "17", "parcels": "379"}}), encoding="utf-8")
    # NO container is written; offline verify would fail on n_errors, but a committed manifest skips resolution
    assert I.verify(collection, tmp_path, manuscript_keys={"n_errors", "parcels"}, manifest_path=manifest) == []


def test_verify_manifest_flags_binding_absent_from_manifest(collection, tmp_path):
    import yaml

    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    manifest = tmp_path / "manuscript_results.yml"
    manifest.write_text(yaml.safe_dump({"results": {"parcels": "379"}}), encoding="utf-8")  # n_errors not regenerated
    problems = I.verify(collection, tmp_path, manifest_path=manifest)
    assert any("n_errors" in p and "absent from the committed manifest" in p for p in problems)


# --------------------------------------------------------------------------- count


@pytest.fixture
def count_investigation(tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text(
        "title: Toy\nkey: toy\n"
        "figures:\n"
        "  - {name: f1, layout: a, panels: {a: {panel_key: a, kind: image}}}\n"
        "  - {name: f2, layout: a, panels: {a: {panel_key: a, kind: image}}}\n"
        "  - {name: f3, layout: a, panels: {a: {panel_key: a, kind: image}}}\n",
        encoding="utf-8",
    )
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: TVB-O\ncitekey: tvbo_manuscript\n"
        "members:\n  - {recipe: members/toy.yaml, label: toy}\n"
        "results:\n"
        "  - {key: n_members, count: members}\n"
        "  - {key: n_toy_figs, count: 'toy.figures'}\n",
        encoding="utf-8",
    )
    return tvbo.SimulationStudy.from_file(str(spec))


def test_count_tallies_member_and_investigation_collections(count_investigation, tmp_path):
    results, prov, problems = I.resolve_results(count_investigation, tmp_path / "output")
    assert problems == []
    assert results["n_members"] == "1"  # bare collection on the collection
    assert results["n_toy_figs"] == "3"  # counted from the loaded member spec
    assert prov["n_toy_figs"] == {"computed": True, "count": "toy.figures"}


def test_count_unknown_collection_is_a_build_problem(tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text("title: Toy\nkey: toy\n", encoding="utf-8")
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: TVB-O\ncitekey: tvbo_manuscript\n"
        "members:\n  - {recipe: members/toy.yaml, label: toy}\n"
        "results:\n  - {key: oops, count: 'toy.bogus'}\n",
        encoding="utf-8",
    )
    inv = tvbo.SimulationStudy.from_file(str(spec))
    _, _, problems = I.resolve_results(inv, tmp_path / "output")
    assert any("oops" in p and "bogus" in p for p in problems)  # typo fails the build, not tallies to 0


def test_count_value_used_are_mutually_exclusive(tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text("title: Toy\nkey: toy\n", encoding="utf-8")
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: TVB-O\ncitekey: tvbo_manuscript\n"
        "members:\n  - {recipe: members/toy.yaml, label: toy}\n"
        "results:\n  - {key: bad, count: members, value: '5'}\n",
        encoding="utf-8",
    )
    inv = tvbo.SimulationStudy.from_file(str(spec))
    _, _, problems = I.resolve_results(inv, tmp_path / "output")
    assert any("bad" in p and "mutually exclusive" in p for p in problems)


# --------------------------------------------------------------------------- captions


@pytest.fixture
def figure(tmp_path):
    spec = tmp_path / "fig.yaml"
    spec.write_text(
        "title: Fig test\ncitekey: figtest\n"
        "figures:\n"
        "  - name: fig-1\n"
        "    layout: ab\n"
        "    description: Overview.\n"
        "    panels:\n"
        "      b:\n"
        "        panel_key: b\n"
        "        kind: heatmap\n"
        "        label: FC\n"
        "        description: reproduces the gradient\n"
        "        layers: [{used: {analysis: fc}, encoding: {x: region, y: region}}]\n"
        "      a:\n"
        "        panel_key: a\n"
        "        kind: cartesian\n"
        "        label: Time series\n"
        "        layers: [{used: {experiment: sweep}, mark: line, encoding: {x: time, y: rate}}]\n",
        encoding="utf-8",
    )
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def test_caption_walks_layout_order_not_declaration_order(figure):
    caption = bsplot.compose_caption(figure)
    assert caption.index("(a)") < caption.index("(b)")  # layout ab, though b declared first
    assert caption.startswith("Overview.")  # authored lead first


def test_caption_derives_structure_and_keeps_authored_interpretation(figure):
    caption = bsplot.compose_caption(figure)
    assert "line of rate vs time from experiment sweep" in caption
    assert "region as a matrix from analysis fc" in caption
    assert "reproduces the gradient" in caption  # Panel.description survives


def test_write_caption_emits_a_partial(figure, tmp_path):
    path = bsplot.write_caption(figure, tmp_path / "figures")
    assert path.name == "fig-1.caption.qmd"
    assert "**(a)**" in path.read_text()


@pytest.fixture
def grid_figure(tmp_path):
    """A grid whose cells draw two sources, one of them from two camera angles."""
    spec = tmp_path / "grid.yaml"
    spec.write_text(
        "title: Grid test\ncitekey: gridtest\n"
        "figures:\n"
        "  - name: fig-g\n"
        "    layout: a\n"
        "    panels:\n"
        "      a:\n"
        "        panel_key: a\n"
        "        kind: grid\n"
        "        cell: {kind: surface}\n"
        "        opts: {ncols: {name: ncols, value: 2}}\n"
        "        layers:\n"
        "          - {used: {analysis: lag_data, output: lag_data}}\n"
        "          - {used: {analysis: lag_data, output: lag_data}}\n"
        "          - {used: {analysis: lag_wave, output: lag_wave}}\n"
        "          - {used: {analysis: lag_wave, output: lag_wave}}\n",
        encoding="utf-8",
    )
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def test_a_grid_names_each_source_once_not_once_per_cell(grid_figure):
    """Eight cells showing one analysis had put its name in the caption eight times.

    The same map drawn laterally and medially is ONE binding seen twice, and a caption that repeats it per cell buries the authored sentence behind identical phrases.
    """
    caption = bsplot.compose_caption(grid_figure)
    assert caption.count("analysis lag_data") == 1
    assert caption.count("analysis lag_wave") == 1
    assert "surface from analysis lag_data; surface from analysis lag_wave" in caption


def test_an_output_that_repeats_its_analysis_name_is_not_printed_twice(grid_figure):
    """``analysis lag_data (lag_data)`` says one thing twice."""
    assert "(lag_data)" not in bsplot.compose_caption(grid_figure)


@pytest.fixture
def multi_output_figure(tmp_path):
    """One analysis drawn three ways in a panel — a density with its mean and a reference."""
    spec = tmp_path / "multi.yaml"
    spec.write_text(
        "title: Multi test\ncitekey: multitest\n"
        "figures:\n"
        "  - name: fig-m\n"
        "    layout: a\n"
        "    panels:\n"
        "      a:\n"
        "        panel_key: a\n"
        "        kind: cartesian\n"
        "        layers:\n"
        "          - {used: {analysis: dist, output: density}, mark: area,\n"
        "             encoding: {x: value, y: density}}\n"
        "          - {used: {analysis: dist, output: mean}, mark: rule,\n"
        "             encoding: {x: value}}\n"
        "          - {used: {analysis: dist, output: reference}, mark: rule,\n"
        "             encoding: {x: value}}\n",
        encoding="utf-8",
    )
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def test_layers_sharing_one_analysis_name_it_once(multi_output_figure):
    """Three outputs of one container is one source, not three.

    Naming it per layer pushed the authored caption behind the same phrase repeated, and a reader cannot tell from it that the three lines describe a single result.
    """
    caption = bsplot.compose_caption(multi_output_figure)
    assert caption.count("analysis dist") == 1
    assert "area of density vs value, rule at mean, rule at reference from analysis dist" in caption


def test_a_rule_is_described_by_the_value_it_stands_at(multi_output_figure):
    """``rule of value`` names the axis; what a reader needs is WHICH value."""
    assert "rule of value" not in bsplot.compose_caption(multi_output_figure)


# ------------------------------------------------- gaps the code review surfaced


def test_a_member_container_resolves_from_the_members_own_layout(collection, tmp_path):
    """A ``used:`` binding into a MEMBER, which the schema documents as supported.

    The IRI's ``<study>`` segment names the member; the member's own directory then answers where its results live. Nothing searches, so a member's container is found in exactly one place — its own.
    """
    _write_container(_member_results(tmp_path), "tally", "n_errors", 7)
    results, _prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert not problems, problems
    assert results["n_errors"] == "7"


def test_a_container_in_the_owning_studys_root_is_not_used_for_a_member_binding(collection, tmp_path):
    """The failure the search order used to hide: a same-named container in the wrong study.

    Both studies can declare an analysis called ``tally``. Resolving by identity means the member's binding reads the member's number, never whichever root happened to be searched first.
    """
    _write_container(tmp_path / "derivatives" / "tvbo", "tally", "n_errors", 1)
    _write_container(_member_results(tmp_path), "tally", "n_errors", 99)
    results, _prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert not problems, problems
    assert results["n_errors"] == "99"


def test_a_binding_naming_a_study_that_is_not_a_member_fails(collection, tmp_path):
    """Rather than falling back to the owning study's root and reporting someone else's number."""
    collection.results[0].used.iri = "tvbo:ana/stranger/tally"
    _write_container(tmp_path / "derivatives" / "tvbo", "tally", "n_errors", 1)
    _, _prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert problems and "stranger" in problems[0]


def test_an_unresolvable_key_is_reported_against_its_own_study(collection, tmp_path):
    """A missing container must name the key, not an incidental directory."""
    _, _prov, problems = I.resolve_results(collection, tmp_path / "derivatives" / "tvbo")
    assert problems and "n_errors" in problems[0]


# ----------------------------------------------------------------- staleness gate


def _analysis(name="tally", rhs="1 + 1"):
    """A minimal declared analysis with a fingerprintable body."""
    from types import SimpleNamespace

    return SimpleNamespace(name=name, equation=SimpleNamespace(rhs=rhs), arguments=None)


def test_fingerprint_tracks_the_declaration_not_the_file():
    """Two analyses differing only in their own body get different digests."""
    assert I._analysis_fingerprint(_analysis()) == I._analysis_fingerprint(_analysis())
    assert I._analysis_fingerprint(_analysis()) != I._analysis_fingerprint(_analysis(rhs="2 + 2"))


def test_an_unrelated_spec_edit_does_not_mark_an_analysis_stale(tmp_path):
    """The defect this replaced: any touch of the spec failed every analysis.

    Editing a caption, adding a figure or fixing a typo made ``verify`` demand a re-run of work that edit could not affect — and the only escape was recomputing all of it.
    """
    root = tmp_path / "output"
    _write_container(root, "tally", "n_errors", 3)
    analysis = _analysis()
    _write_digest(root, "tally", analysis)

    spec = tmp_path / "spec.yaml"
    spec.write_text("title: later than the container\n", encoding="utf-8")
    inv = SimpleNamespace(analyses=[analysis])
    assert I._stale_or_missing_analyses(inv, root, spec) == []


def test_editing_the_analysis_itself_does_mark_it_stale(tmp_path):
    root = tmp_path / "output"
    _write_container(root, "tally", "n_errors", 3)
    _write_digest(root, "tally", _analysis(rhs="the previous body"))
    inv = SimpleNamespace(analyses=[_analysis()])
    problems = I._stale_or_missing_analyses(inv, root, tmp_path / "spec.yaml")
    assert problems and "edited but not re-run" in problems[0]


def test_a_container_without_a_fingerprint_is_accepted(tmp_path):
    """Containers predating the check carry none. Failing every build once teaches bypassing."""
    root = tmp_path / "output"
    _write_container(root, "tally", "n_errors", 3)
    inv = SimpleNamespace(analyses=[_analysis()])
    assert I._stale_or_missing_analyses(inv, root, tmp_path / "spec.yaml") == []


def test_a_missing_container_is_still_reported(tmp_path):
    inv = SimpleNamespace(analyses=[_analysis()])
    problems = I._stale_or_missing_analyses(inv, tmp_path / "output", tmp_path / "spec.yaml")
    assert problems and "never run" in problems[0]


def test_run_analysis_records_the_fingerprint_it_will_be_checked_against(tmp_path):
    """The write and the read must agree, or the gate never fires (or always does)."""
    from tvbo.data.analysis_io import run_analysis

    analysis = SimpleNamespace(
        name="tally",
        equation=None,
        arguments=None,
        execution=None,
        apply_on_dimension=None,
        aggregate=None,
        dims=None,
        class_call=None,
        function=None,
        callable=SimpleNamespace(module="numpy", name="mean"),
    )
    analysis.arguments = {"a": SimpleNamespace(name="a", value=[1.0, 2.0, 3.0], used=None)}
    path = run_analysis(analysis, tmp_path / "output")
    import yaml

    from tvbo.data.dataref import sidecar_path

    record = yaml.safe_load(sidecar_path(path).read_text())
    assert record["declaration_digest"] == I._analysis_fingerprint(analysis)


# ------------------------------------------------------- `tvbo verify --manuscript`


def test_a_mistyped_manuscript_path_names_the_real_problem(collection, tmp_path):
    """An unreadable path must not read as "the prose cites nothing".

    Swallowing the OSError produced an empty key set, so verify reported EVERY declared binding as never-cited — a wall of wrong diagnostics hiding the one real fault.
    """
    import typer

    from tvbo.cli.verify import _scan_meta_keys

    with pytest.raises((SystemExit, typer.Exit, typer.BadParameter)):
        _scan_meta_keys(tmp_path / "manusript.qmd")


def test_a_directory_with_nothing_to_scan_is_an_error(tmp_path):
    (tmp_path / "prose").mkdir()
    import typer

    from tvbo.cli.verify import _scan_meta_keys

    with pytest.raises((SystemExit, typer.Exit, typer.BadParameter)):
        _scan_meta_keys(tmp_path / "prose")


def test_cited_keys_are_read_from_a_file_and_a_tree(tmp_path):
    from tvbo.cli.verify import _scan_meta_keys

    (tmp_path / "a.qmd").write_text("we found {{< meta results.n_errors >}} errors\n")
    assert _scan_meta_keys(tmp_path / "a.qmd") == {"n_errors"}
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("across {{< meta results.parcels >}} parcels\n")
    assert _scan_meta_keys(tmp_path) == {"n_errors", "parcels"}


# ------------------------------------------ a caption must never lose a rendered figure


def test_a_caption_failure_does_not_abort_the_render_loop(tmp_path, monkeypatch):
    """The pre-check sat OUTSIDE the try that exists to guarantee exactly this.

    An exception composing figure 1's caption stopped the loop, so every later figure went unrendered even though the images written so far were fine.
    """
    from tvbo.adapters import bsplot
    from tvbo.cli import figures as figures_cli

    rendered: list = []
    monkeypatch.setattr(bsplot, "render", lambda fig, **kw: rendered.append(getattr(fig, "name", None)))
    monkeypatch.setattr(bsplot, "compose_caption", lambda fig: (_ for _ in ()).throw(AttributeError("bad panel shape")))

    figs = [SimpleNamespace(name="fig-a", format="png"), SimpleNamespace(name="fig-b", format="png")]
    written = figures_cli.render_figures(figs, tmp_path, tmp_path / "figures")
    assert rendered == ["fig-a", "fig-b"]
    assert len(written) == 2


def test_a_manifest_is_not_written_when_an_analysis_stage_failed(tmp_path, monkeypatch):
    """A failed stage means the containers are stale or absent.

    Emitting anyway reported numbers the run did not produce, and exited 0. The boolean ``_run_whole_study`` already returned was simply discarded.
    """
    import typer

    from tvbo.cli import run as run_cli

    emitted: list = []
    monkeypatch.setattr(run_cli, "_run_whole_study", lambda *a, **k: False)
    monkeypatch.setattr(run_cli, "emit_manifest", lambda *a, **k: emitted.append(a) or (tmp_path / "m.yml", []), raising=False)

    spec = tmp_path / "collection.yaml"
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text("title: Toy\nsimulation_experiments: []\n")
    spec.write_text(
        "title: Demo\nmembers:\n  - {recipe: members/toy.yaml, label: toy}\n"
        "results:\n  - {key: parcels, value: '379', source: s}\n"
    )
    obj = tvbo.SimulationStudy.from_file(str(spec))
    with pytest.raises((SystemExit, typer.Exit, typer.BadParameter)):
        run_cli._run_with_members(obj, str(spec), tmp_path / "output")
    assert not emitted, "the manifest was written from a failed run"


# ------------------------------------------------- caption staleness gate


@pytest.fixture
def figured(tmp_path):
    """A collection with one composed figure whose caption is written to `figures/`."""
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: Fig test\ncitekey: figtest\n"
        "figures:\n"
        "  - name: fig-1\n"
        "    layout: ab\n"
        "    description: Overview.\n"
        "    panels:\n"
        "      a:\n"
        "        kind: cartesian\n"
        "        label: Time series\n"
        "        layers: [{used: {experiment: sweep}, mark: line, encoding: {x: time, y: rate}}]\n"
        "      b:\n"
        "        kind: heatmap\n"
        "        label: FC\n"
        "        description: reproduces the gradient\n"
        "        layers: [{used: {analysis: fc}, encoding: {x: region, y: region}}]\n",
        encoding="utf-8",
    )
    inv = tvbo.SimulationStudy.from_file(str(spec))
    caps = tmp_path / "figures"
    bsplot.write_caption(inv.figures[0], caps)
    return inv, caps


def test_stale_captions_passes_when_committed_matches_spec(figured):
    inv, caps = figured
    assert I._stale_captions(inv, caps) == []


def test_stale_captions_flags_a_drifted_committed_caption(figured):
    """A spec edit not recomposed leaves a caption the manuscript would still render."""
    inv, caps = figured
    p = caps / "fig-1.caption.qmd"
    p.write_text(p.read_text().replace("Overview.", "A different lead."), encoding="utf-8")
    assert any("fig-1" in m and "stale" in m for m in I._stale_captions(inv, caps))


def test_stale_captions_skips_a_figure_with_no_committed_partial(figured, tmp_path):
    """No partial means nothing to be stale — the gate must not invent a problem."""
    inv, _caps = figured
    empty = tmp_path / "other-figures"
    empty.mkdir()
    assert I._stale_captions(inv, empty) == []


def test_verify_surfaces_a_stale_caption_end_to_end(figured, tmp_path):
    inv, caps = figured
    (caps / "fig-1.caption.qmd").write_text("**Wrong.**\n", encoding="utf-8")
    assert any("stale" in p for p in I.verify(inv, tmp_path, captions_dir=caps))
