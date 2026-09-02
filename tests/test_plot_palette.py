# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""One file owns a project's colours, and every figure reads its roles from there."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import yaml

from tvbo.adapters import bsplot
from tvbo.adapters.bsplot import _style_entries
from tvbo.plot import palette


def _schema() -> dict:
    """The exported JSON Schema, which is where the Palette class is written down."""
    import json

    import tvbo

    return json.loads((pathlib.Path(tvbo.__file__).parent / "datamodel" / "tvbo_datamodel.schema.json").read_text())


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
    assert palette.colormap("diverging").name == palette.DEFAULT["colormaps"]["diverging"], (
        "an undeclared role keeps its default"
    )


def test_a_colormap_can_be_written_as_a_list_of_colours():
    palette.use({**SHEET, "colormaps": {"sequential": ["#000000", "#ffffff"]}})
    assert palette.colormap()(1.0)[:3] == (1.0, 1.0, 1.0)


def test_an_ordinal_scale_samples_the_colormap_rather_than_the_hues():
    palette.use(SHEET)
    swatches = palette.ramp(4)
    assert len(swatches) == 4
    assert [tuple(c[:3]) for c in swatches] != [tuple(c) for c in palette.palette(4)], (
        "a ladder must not be drawn in categorical hues"
    )


def test_a_colormap_no_one_registered_is_refused():
    with pytest.raises(ValueError, match="sequential"):
        palette.load({**SHEET, "colormaps": {"sequential": "not-a-colormap"}})


def test_a_project_names_scales_of_its_own_and_still_gets_the_two_guaranteed_ones():
    spec = palette.use({**SHEET, "colormaps": {"bold": "viridis", "meg": "plasma", "eeg": "cividis"}})
    assert palette.colormap("meg").name == "plasma"
    assert set(palette.GUARANTEED_COLORMAPS) <= set(spec["colormaps"]), "a project that declares neither still has both"
    assert palette.as_colormap("meg") == "plasma"
    assert palette.as_colormap("cividis") == "cividis", "a name the project did not declare is left for the backend's registry"


def test_a_key_shadowing_a_registered_colormap_is_refused():
    with pytest.raises(ValueError, match="magma"):
        palette.load({**SHEET, "colormaps": {"magma": "viridis"}})


def test_a_theme_can_be_read_for_its_colours_without_its_geometry_being_a_typo():
    spec = palette.load({**SHEET, "tick_length": 6, "font_family": ["Helvetica"], "iri": "tvbo:theme/default"})
    assert spec["highlight"] == "#ff0000"
    assert "tick_length" not in spec, "the colour reader carries the colours and leaves the geometry to the renderer"


def test_the_geometry_tuple_is_exactly_what_the_theme_adds_to_a_palette():
    theme, pal = _schema()["$defs"]["Theme"]["properties"], _schema()["$defs"]["Palette"]["properties"]
    assert set(palette.GEOMETRY) == set(theme) - set(pal) - {"iri"}, (
        "a Theme slot the colour reader does not know about is reported as a typo"
    )


def test_a_role_or_a_hue_resolves_and_everything_else_passes_through():
    palette.use(SHEET)
    assert palette.as_color("highlight") == "#ff0000"
    assert palette.as_color("palette.1") == "#222222"
    for untouched in ("#123456", "tab:blue", "red", None, [0.1, 0.2, 0.3]):
        assert palette.as_color(untouched) == untouched


def test_the_shipped_palette_is_what_the_defaults_are():
    on_disk = yaml.safe_load(palette.PATH.read_text())
    assert palette.load(palette.PATH) == palette.DEFAULT, (
        "a hex written in the module rather than the file is a copy waiting to drift"
    )
    assert set(on_disk) - set(palette.DEFAULT) - set(palette.DEFAULT_GEOMETRY) == {"tvbo_class"}, (
        "the envelope annotates the file, and nothing else in it may be dropped"
    )
    assert palette.DEFAULT_GEOMETRY == {"legend_frame": False}, (
        "the curated theme fixes one piece of geometry, and it is the frame the renderer used to hardcode"
    )
    assert not set(palette.DEFAULT) & set(palette.DEFAULT_GEOMETRY), "the two halves of the look partition the file"


def test_the_shipped_theme_validates_as_a_theme_in_the_figure_spec():
    import jsonschema

    spec = yaml.safe_load(palette.PATH.read_text())
    assert spec["tvbo_class"] == "tvbo:Theme", "the envelope is what `tvbo validate schema` reads to pick the class"
    jsonschema.validate(spec, {"$defs": _schema()["$defs"], "$ref": "#/$defs/Theme"})


def test_the_module_knows_exactly_the_slots_the_schema_declares():
    assert set(palette.FIELDS) == set(_schema()["$defs"]["Palette"]["properties"]), (
        "a slot added to the Palette class the reader does not know about is dropped on load"
    )


def test_a_misspelt_role_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="hilight"):
        palette.load({**SHEET, "hilight": "#ff0000"})


def test_a_palette_takes_the_same_yaml_extensions_as_every_other_document(tmp_path):
    (tmp_path / "hues.yaml").write_text(yaml.safe_dump({"palette": ["#111111", "#222222", "#333333"]}))
    path = tmp_path / "palette.yaml"
    path.write_text("tvbo_class: tvbo:Palette\nhighlight: '#ff0000'\n<<: !include hues.yaml\n")
    assert palette.load(path)["palette"] == ["#111111", "#222222", "#333333"]
    assert palette.load(path)["highlight"] == "#ff0000"


def test_style_layers_are_the_looks_tvbo_does_not_own(tmp_path):
    """Registered names and sheets only. Colour is not a layer — it comes from the theme, applied over all of them."""

    class Figure:
        style = ["tvbo", "style/sheet.mplstyle"]

    assert [e["kind"] for e in _style_entries(Figure(), tmp_path)] == ["named", "mplstyle"]


def test_a_palette_named_as_a_style_layer_says_what_replaced_it(tmp_path):
    class Figure:
        style = ["tvbo", "style/palette.yaml"]

    with pytest.raises(ValueError, match="theme"):
        _style_entries(Figure(), tmp_path)


def test_the_shipped_palette_layer_is_read_as_the_theme(tmp_path):
    """`tvbo-palette` is an alias for the curated theme, so it is not a layer and does not carry this machine's path to one."""

    class Figure:
        style = ["tvbo", "tvbo-palette"]
        theme = None

    assert [e["kind"] for e in _style_entries(Figure(), tmp_path)] == ["named"]
    assert bsplot.theme_spec(Figure(), tmp_path)["palette"] == palette.DEFAULT["palette"]
