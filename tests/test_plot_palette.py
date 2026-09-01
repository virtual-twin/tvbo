# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""One file owns a project's colours, and every figure reads its roles from there."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import yaml

from tvbo.adapters.bsplot import _style_entries
from tvbo.plot import palette

SHEET = {
    "ink": "#101010",
    "base": "#404040",
    "muted": "#d0d0d0",
    "highlight": "#ff0000",
    "background": "#fafafa",
    "palette": ["#111111", "#222222", "#333333"],
    "colormaps": {"sequential": "plasma"},
}


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    palette.use(palette.DEFAULT)


def test_the_neutral_leads_the_cycle_and_the_highlight_stays_out_of_it():
    palette.use(SHEET)
    assert palette.cycle() == ["#404040", "#111111", "#222222", "#333333"]
    assert palette.highlight() not in palette.cycle(), "an emphasis colour handed out by the cycler is not one"


def test_using_a_palette_sets_the_rcparams_that_carry_it():
    palette.use(SHEET)
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == palette.cycle()
    assert plt.rcParams["text.color"] == "#101010"
    assert plt.rcParams["figure.facecolor"] == "#fafafa"


def test_a_palette_loads_from_a_file(tmp_path):
    path = tmp_path / "palette.yaml"
    path.write_text(yaml.safe_dump(SHEET))
    assert palette.use(path)["highlight"] == "#ff0000"
    assert palette.color("palette.1") == "#222222"


def test_a_role_a_project_leaves_out_falls_back_rather_than_failing(tmp_path):
    path = tmp_path / "palette.yaml"
    path.write_text(yaml.safe_dump({"palette": ["#123456"]}))
    assert palette.use(path)["ink"] == palette.DEFAULT["ink"]


def test_a_colour_matplotlib_cannot_read_is_refused():
    with pytest.raises(ValueError, match="highlight"):
        palette.load({**SHEET, "highlight": "not a colour"})


def test_the_continuous_scales_come_from_the_same_file():
    palette.use(SHEET)
    assert palette.colormap().name == "plasma"
    assert plt.rcParams["image.cmap"] == "plasma"
    assert palette.colormap("intensity").name == palette.DEFAULT["colormaps"]["intensity"], "an undeclared role keeps its default"


def test_a_colormap_can_be_written_as_a_list_of_colours():
    palette.use({**SHEET, "colormaps": {"sequential": ["#000000", "#ffffff"]}})
    assert palette.colormap()(1.0)[:3] == (1.0, 1.0, 1.0)


def test_an_ordinal_scale_samples_the_colormap_rather_than_the_hues():
    palette.use(SHEET)
    swatches = palette.ramp(4)
    assert len(swatches) == 4
    assert [tuple(c[:3]) for c in swatches] != [tuple(c) for c in palette.palette(4)], "a ladder must not be drawn in categorical hues"


def test_a_colormap_no_one_registered_is_refused():
    with pytest.raises(ValueError, match="sequential"):
        palette.load({**SHEET, "colormaps": {"sequential": "not-a-colormap"}})


def test_a_yaml_style_layer_is_classified_as_a_palette(tmp_path):
    class Figure:
        style = ["tvbo", "style/sheet.mplstyle", "style/palette.yaml"]

    kinds = [entry["kind"] for entry in _style_entries(Figure(), tmp_path)]
    assert kinds == ["named", "mplstyle", "palette"]
