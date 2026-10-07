"""An annotation placed in the caption is written there with the number its binding reads, and is not drawn on the panel.

A sample size belongs to the reader's account of a panel, not to the panel itself. Declared with ``placement: caption``, the same ``text`` + ``used:`` binding a drawn annotation uses is formatted into the composed caption at render time, so the caption states a computed number rather than a typed one. Where the containers are absent, as in a build, a committed caption is still checked against its spec with the number free.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.data import study_manifest
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_SPEC = """title: Caption test
citekey: captiontest
figures:
  - name: fig-1
    layout: a
    description: Group means.
    panels:
      a:
        panel_key: a
        kind: cartesian
        label: Accuracy
        description: one point per group
PLACEHOLDER        layers: [{used: {analysis: groups, output: mean}, mark: scatter, encoding: {x: group, y: mean}}]
        annotations:
          - {text: "N = {:.0f} subjects", used: {analysis: groups, output: n_subjects}, placement: caption}
          - {text: "r = {:.2f}", used: {analysis: groups, output: r}, loc: upper left}
"""


def _figure(tmp_path, placeholder=False):
    (tmp_path / "dataset_description.json").write_text('{"Name": "captiontest", "BIDSVersion": "1.9.0"}')
    spec = tmp_path / "study.yaml"
    spec.write_text(_SPEC.replace("PLACEHOLDER", '        placeholder: "pending"\n' if placeholder else ""), encoding="utf-8")
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def _write_groups(root: Path, n_subjects: float, **scalars: float) -> None:
    path = analysis_container_path(study_path("results", root=root), "groups")
    path.parent.mkdir(parents=True, exist_ok=True)
    group = xr.DataArray(np.arange(3.0), dims="group")
    variables = {"mean": group * 2.0, "n_subjects": n_subjects, "r": 0.5, **scalars}
    xr.Dataset({k: xr.DataArray(v) for k, v in variables.items()}, coords={"group": group}).to_netcdf(path, engine="h5netcdf")


def test_the_caption_states_the_number_its_annotation_reads_and_the_panel_draws_only_the_other(tmp_path):
    figure = _figure(tmp_path)
    _write_groups(tmp_path, 663.0)
    caption = bsplot.compose_caption(figure, tmp_path)
    assert caption.endswith(
        "**(a)** Accuracy. scatter of mean vs group from analysis groups. one point per group. N = 663 subjects."
    )
    panel = dict(bsplot._items(figure.panels))["a"]
    assert [a["text"] for a in bsplot._annotations(panel, tmp_path)] == ["r = {:.2f}"]


def test_a_placeholder_panel_states_no_number_and_a_panel_without_one_refuses_a_missing_container(tmp_path):
    assert "N =" not in bsplot.compose_caption(_figure(tmp_path, placeholder=True), tmp_path)
    with pytest.raises(FileNotFoundError, match="declares no placeholder"):
        bsplot.compose_caption(_figure(tmp_path), tmp_path)


def test_a_committed_caption_is_checked_with_the_number_free_and_the_prose_fixed(tmp_path):
    figure = _figure(tmp_path)
    _write_groups(tmp_path, 663.0)
    path = bsplot.write_caption(figure, tmp_path / "captions", base_dir=tmp_path)
    assert bsplot.caption_matches(figure, path.read_text())
    assert bsplot.caption_matches(figure, path.read_text().replace("663", "1065"))
    assert not bsplot.caption_matches(figure, path.read_text().replace("one point per group", "one point per subject"))
    study = tvbo.SimulationStudy.from_file(str(tmp_path / "study.yaml"))
    assert study_manifest._stale_captions(study, tmp_path / "captions") == []


def test_a_placeholder_panels_caption_is_current_with_or_without_its_number(tmp_path):
    """A caption written before the container exists leaves the sentence out, and is no staler for it than one written after."""
    pending, done = tmp_path / "pending", tmp_path / "done"
    pending.mkdir()
    done.mkdir()
    figure = _figure(pending, placeholder=True)
    _figure(done, placeholder=True)
    _write_groups(done, 663.0)
    before = bsplot.write_caption(figure, pending / "captions", base_dir=pending).read_text()
    after = bsplot.write_caption(figure, done / "captions", base_dir=done).read_text()
    assert "N =" not in before and "N = 663 subjects." in after
    assert bsplot.caption_matches(figure, before) and bsplot.caption_matches(figure, after)
    assert not bsplot.caption_matches(figure, after.replace("one point per group. N", "one point per group.  N"))


def test_a_study_result_keeps_a_figure_whose_caption_number_has_no_container(tmp_path):
    """The figure was drawn, so a run's result holds it; only the caption that cannot state its number is left out."""
    from tvbo.run.study import figure_outputs, study_path_for

    figure = _figure(tmp_path)
    _, image, _ = figure_outputs(figure, study_path_for("figures", tmp_path))
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"")
    result = tvbo.SimulationStudy.from_file(str(tmp_path / "study.yaml"))._collect(tmp_path)
    assert result.figure("fig-1").caption is None


@pytest.mark.parametrize("slot", ["insets", "cells"])
def test_an_inset_or_grid_cells_caption_annotation_joins_its_panels_clause(tmp_path, slot):
    (tmp_path / "dataset_description.json").write_text('{"Name": "captiontest", "BIDSVersion": "1.9.0"}')
    spec = tmp_path / "study.yaml"
    owner = (
        f"        {slot}:\n"
        "          - kind: cartesian\n"
        '            annotations: [{text: "{:.0f} groups", used: {analysis: groups, output: n_groups}, placement: caption}]\n'
    )
    spec.write_text(_SPEC.replace("PLACEHOLDER", "") + owner, encoding="utf-8")
    _write_groups(tmp_path, 663.0, n_groups=20.0)
    caption = bsplot.compose_caption(tvbo.SimulationStudy.from_file(str(spec)).figures[0], tmp_path)
    assert caption.endswith("N = 663 subjects. 20 groups.")


_SIBLINGS = """title: Caption test
citekey: captiontest
figures:
  - name: fig-1
    layout: a1 a2
    panels:
      a1:
        panel_key: a1
        kind: cartesian
        layers: [{used: {analysis: groups, output: mean}, mark: scatter, encoding: {x: group, y: mean}}]
        annotations:
          - {text: "N = {:.0f}", used: {analysis: groups, output: LEFT}, placement: caption}
      a2:
        panel_key: a2
        kind: cartesian
        layers: [{used: {analysis: groups, output: mean}, mark: line, encoding: {x: group, y: mean}}]
        annotations:
          - {text: "N = {:.0f}", used: {analysis: groups, output: RIGHT}, placement: caption}
"""


def _siblings(tmp_path, left="n_left", right="n_right"):
    (tmp_path / "dataset_description.json").write_text('{"Name": "captiontest", "BIDSVersion": "1.9.0"}')
    spec = tmp_path / "study.yaml"
    spec.write_text(_SIBLINGS.replace("LEFT", left).replace("RIGHT", right), encoding="utf-8")
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


@pytest.mark.parametrize(("n_left", "n_right"), [(5.0, 6.0), (5.0, 5.0)])
def test_sibling_panels_reading_different_outputs_each_state_theirs_and_the_caption_is_current(tmp_path, n_left, n_right):
    """Two cells of one lettered panel that read two outputs make two statements, whether or not the numbers coincide, and the caption written from the containers checks out against the spec without them."""
    figure = _siblings(tmp_path)
    _write_groups(tmp_path, 0.0, n_left=n_left, n_right=n_right)
    caption = bsplot.compose_caption(figure, tmp_path)
    assert caption.count(f"N = {n_left:.0f}.") + caption.count(f"N = {n_right:.0f}.") == 2 * (1 + (n_left == n_right))
    assert bsplot.caption_matches(figure, caption)


def test_sibling_panels_reading_one_output_state_it_once(tmp_path):
    figure = _siblings(tmp_path, right="n_left")
    _write_groups(tmp_path, 0.0, n_left=5.0)
    caption = bsplot.compose_caption(figure, tmp_path)
    assert caption.count("N = 5.") == 1
    assert bsplot.caption_matches(figure, caption)


def test_a_caption_number_bound_to_several_values_is_refused(tmp_path):
    """An output with one value per group is not a number a caption can state; the first of them would read as one."""
    figure = _siblings(tmp_path, left="mean")
    _write_groups(tmp_path, 0.0, n_right=6.0)
    with pytest.raises(ValueError, match="'mean' .* holds 3 values where an annotation prints one"):
        bsplot.compose_caption(figure, tmp_path)
