"""`backdrop: false` prints a bound annotation without the box in the page colour that otherwise sits behind it.

The box keeps a statistic printed over its own scatter readable. Over a plot that fills its whole area, such as a density whose empty cells take the colormap's light end, it shows as a patch of the page colour, so a spec can turn it off.
"""

from pathlib import Path

import matplotlib.figure
import pytest
import xarray as xr

import tvbo
from tvbo.adapters import bsplot
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

_SPEC = """title: Backdrop test
citekey: backdroptest
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


def _figure(root: Path, slots: str):
    (root / "dataset_description.json").write_text('{"Name": "backdroptest", "BIDSVersion": "1.9.0"}')
    spec = root / "study.yaml"
    spec.write_text(_SPEC.replace("ANNOTATION_SLOTS\n", slots), encoding="utf-8")
    path = analysis_container_path(study_path("results", root=root), "groups")
    path.parent.mkdir(parents=True, exist_ok=True)
    group = xr.DataArray([0.0, 1.0, 2.0], dims="group")
    xr.Dataset({"mean": group * 2.0, "r": xr.DataArray(0.5)}, coords={"group": group}).to_netcdf(path, engine="h5netcdf")
    return tvbo.SimulationStudy.from_file(str(spec)).figures[0]


@pytest.mark.parametrize(
    ("slots", "boxed"),
    [("", True), ("            backdrop: true\n", True), ("            backdrop: false\n", False)],
)
def test_a_bound_label_has_its_backdrop_unless_it_declares_none(tmp_path, monkeypatch, slots, boxed):
    """The label on the saved figure still reads its number from the run; only the box behind it goes."""
    figure = _figure(tmp_path, slots)
    drawn, savefig = [], matplotlib.figure.Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        drawn.append([(text.get_text(), text.get_bbox_patch()) for ax in self.axes for text in ax.texts])
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    bsplot.render(figure, str(tmp_path), str(tmp_path / "fig.png"))
    (box,) = [patch for label, patch in drawn[-1] if label == "r=0.50"]
    assert (box is not None) == boxed
