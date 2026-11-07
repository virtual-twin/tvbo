"""An annotation's ``significance`` sets the asterisks a paper prints after a statistic, counted from a p-value the render reads.

The label and its asterisks come from two bindings, the statistic and its p-value, which may sit in different containers. Both are read when the figure is drawn, so neither the number nor how many asterisks follow it is typed into the spec.
"""

from pathlib import Path

import matplotlib.figure
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.adapters import figure_workflow as fw
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_SPEC = """title: Significance test
citekey: significancetest
figures:
  - name: fig-1
    layout: a
    panels:
      a:
        panel_key: a
        kind: cartesian
        layers: [{used: {analysis: groups, output: mean}, mark: scatter, encoding: {x: group, y: mean}}]
        annotations:
          - text: "r={:.2f}"
            used: {analysis: groups, output: r}
            loc: upper left
ANNOTATION_SLOTS
"""

_BOUND = "            significance: {used: {analysis: stats, output: p}}\n"


def _figure(root: Path, slots: str = _BOUND):
    (root / "dataset_description.json").write_text('{"Name": "significancetest", "BIDSVersion": "1.9.0"}')
    spec = root / "study.yaml"
    spec.write_text(_SPEC.replace("ANNOTATION_SLOTS\n", slots), encoding="utf-8")
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


def _container(root: Path, name: str) -> Path:
    return analysis_container_path(study_path("results", root=root), name)


def _write(root: Path, p_value: float) -> None:
    group = xr.DataArray([0.0, 1.0, 2.0], dims="group")
    for name, variables in (
        ("groups", {"mean": group * 2.0, "r": xr.DataArray(0.5)}),
        ("stats", {"p": xr.DataArray(p_value)}),
    ):
        path = _container(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(variables, coords={"group": group}).to_netcdf(path, engine="h5netcdf")


@pytest.mark.parametrize(
    ("p_value", "levels", "marks"),
    [
        (0.2, None, ""),
        (0.05, None, ""),
        (0.03, None, "*"),
        (0.004, None, "**"),
        (0.0004, None, "***"),
        (0.07, [0.1], "*"),
        (0.0004, [0.05], "*"),
    ],
)
def test_a_p_value_earns_one_asterisk_per_threshold_it_is_below(p_value, levels, marks):
    assert bsplot.significance_marks(p_value, *([levels] if levels else [])) == marks


@pytest.mark.parametrize(
    ("p_value", "slots", "label"),
    [(0.0004, _BOUND, "r=0.50***"), (0.03, _BOUND, "r=0.50*"), (0.2, _BOUND, "r=0.50"), (0.03, "", "r=0.50")],
)
def test_the_saved_panel_prints_the_asterisks_its_p_value_earns(tmp_path, monkeypatch, p_value, slots, label):
    """The text on the axes of the figure that is saved carries the asterisks, and an annotation that declares no significance is printed as before."""
    figure = _figure(tmp_path, slots)
    _write(tmp_path, p_value)
    drawn, savefig = [], matplotlib.figure.Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        drawn.append([text.get_text() for ax in self.axes for text in ax.texts])
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    bsplot.render(figure, str(tmp_path), str(tmp_path / "fig.png"))
    assert drawn and [text for text in drawn[-1] if text.startswith("r=")] == [label]


def test_declared_thresholds_replace_the_default_ones(tmp_path):
    figure = _figure(tmp_path, "            significance: {used: {analysis: stats, output: p}, levels: [0.1, 0.01]}\n")
    panel = dict(bsplot._items(figure.panels))["a"]
    (resolved,) = bsplot._annotations(panel, tmp_path)
    assert resolved["significance"]["levels"] == [0.1, 0.01]
    assert resolved["significance"]["layer"]["output"] == "p"
    assert bsplot._significance(panel.annotations[0].__class__(text="plain"), tmp_path) is None


def test_the_render_rule_waits_for_the_container_the_p_value_is_read_from(tmp_path):
    """The p-value's container is a rule input in its own right: here nothing else on the panel reads it."""
    figure = _figure(tmp_path)
    _write(tmp_path, 0.03)
    rules = fw.emit_figure_rules([figure], base_dir=tmp_path)
    assert str(_container(tmp_path, "stats")) in rules and str(_container(tmp_path, "groups")) in rules


def test_a_threshold_outside_the_unit_interval_is_refused(tmp_path):
    figure = _figure(tmp_path, "            significance: {used: {analysis: stats, output: p}, levels: [0.05, 5]}\n")
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        bsplot.build_context(figure, tmp_path, "out.png")


def test_a_caption_annotation_that_declares_significance_is_refused(tmp_path):
    """A declaration the caption composer would otherwise accept and never write."""
    figure = _figure(tmp_path, _BOUND + "            placement: caption\n")
    with pytest.raises(ValueError, match="a caption states the test in words"):
        bsplot.build_context(figure, tmp_path, "out.png")
