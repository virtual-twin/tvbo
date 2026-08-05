"""The Investigation container: manifest resolution, verify gates, and composed captions.

Phase 0 of the native-manuscript design. An `Investigation` is a study-of-studies whose
`results:` are the numbers the prose cites (computed from a container or authored from prior
work), whose figures carry `Panel.description` so captions compose from the spec, and whose
buildability `tvbo verify` checks. These pin that plumbing so a regression can't silently ship
a wrong number, an orphan figure, or a caption that disagrees with its panels.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.data import investigation as I


def _write_container(root: Path, name: str, var: str, value) -> None:
    """Write a scalar analysis container where `resolve_dataref` looks for it."""
    d = root / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    xr.Dataset({var: xr.DataArray(np.array(value))}).to_netcdf(d / "result.h5", engine="h5netcdf")


@pytest.fixture
def investigation(tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "toy.yaml").write_text(
        "title: Toy\nkey: toy\nsimulation_experiments: []\n", encoding="utf-8"
    )
    spec = tmp_path / "tvbo_manuscript.yaml"
    spec.write_text(
        "title: TVB-O\n"
        "citekey: tvbo_manuscript\n"
        "members:\n"
        "  - {recipe: members/toy.yaml, label: toy}\n"
        "  - {recipe: members/heavy.yaml, label: heavy, optional: true}\n"
        "results:\n"
        "  - {key: n_errors, used: {analysis: tally, output: n_errors}, format: '{:d}'}\n"
        "  - {key: parcels, value: '379', source: Glasser2016, description: HCP parcels}\n",
        encoding="utf-8",
    )
    return tvbo.Investigation.from_file(str(spec))


def test_from_file_loads_members_results_archive(investigation):
    inv = investigation
    assert isinstance(inv, tvbo.SimulationStudy)
    assert [m.label for m in inv.members] == ["toy", "heavy"]
    assert {r.key for r in inv.results} == {"n_errors", "parcels"}


def test_optional_member_dropped_unless_requested(investigation):
    light = [lbl for lbl, _ in investigation.member_recipes(include_optional=False)]
    full = [lbl for lbl, _ in investigation.member_recipes(include_optional=True)]
    assert light == ["toy"]
    assert full == ["toy", "heavy"]


def test_authored_value_resolves_without_a_container(investigation, tmp_path):
    results, prov, problems = I.resolve_results(investigation, tmp_path / "output")
    assert results["parcels"] == "379"
    assert prov["parcels"] == {"computed": False, "value": "379",
                               "source": "Glasser2016", "description": "HCP parcels"}
    assert any(p.startswith("n_errors:") for p in problems)  # no container yet


def test_computed_value_resolves_and_formats(investigation, tmp_path):
    _write_container(tmp_path / "output", "tally", "n_errors", 17)
    results, prov, problems = I.resolve_results(investigation, tmp_path / "output")
    assert problems == []
    assert results["n_errors"] == "17"
    assert prov["n_errors"]["computed"] is True


def test_emit_manifest_writes_quarto_meta_shape(investigation, tmp_path):
    _write_container(tmp_path / "output", "tally", "n_errors", 17)
    out, problems = I.emit_manifest(investigation, tmp_path / "output",
                                    tmp_path / "_output" / "manuscript_results.yml")
    assert problems == []
    import yaml
    payload = yaml.safe_load(out.read_text())
    assert payload["results"] == {"n_errors": "17", "parcels": "379"}


def test_verify_flags_missing_members_and_dead_keys(investigation, tmp_path):
    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    problems = I.verify(investigation, tmp_path)
    assert any("n_errors" in p for p in problems)  # container missing -> dead key
    _write_container(tmp_path / "output", "tally", "n_errors", 17)
    assert I.verify(investigation, tmp_path) == []


def test_verify_coverage_is_bidirectional(investigation, tmp_path):
    (tmp_path / "members" / "heavy.yaml").write_text("title: Heavy\nkey: heavy\n", encoding="utf-8")
    _write_container(tmp_path / "output", "tally", "n_errors", 17)
    problems = I.verify(investigation, tmp_path, manuscript_keys={"n_errors", "ghost"})
    assert any("ghost" in p and "not declared" in p for p in problems)
    assert any("parcels" in p and "never cited" in p for p in problems)


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
    return tvbo.Investigation.from_file(str(spec)).figures[0]


def test_caption_walks_layout_order_not_declaration_order(figure):
    caption = bsplot.compose_caption(figure)
    assert caption.index("(a)") < caption.index("(b)")           # layout ab, though b declared first
    assert caption.startswith("Overview.")                       # authored lead first


def test_caption_derives_structure_and_keeps_authored_interpretation(figure):
    caption = bsplot.compose_caption(figure)
    assert "line of rate vs time from experiment sweep" in caption
    assert "region as a matrix from analysis fc" in caption
    assert "reproduces the gradient" in caption                  # Panel.description survives


def test_write_caption_emits_a_partial(figure, tmp_path):
    path = bsplot.write_caption(figure, tmp_path / "figures")
    assert path.name == "fig-1.caption.qmd"
    assert "**(a)**" in path.read_text()
