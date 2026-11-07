"""Tests for the ASCII spec-to-cortex portrait and the bare-``tvbo`` splash."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tvbo.cli import _portrait, app

runner = CliRunner()


def test_shipped_asset_carries_both_channels_and_a_ramp():
    cortex = _portrait.load_cortex()
    assert len(cortex.light) == len(cortex.curv)
    assert len(cortex.light[0]) == len(cortex.curv[0])
    covered = [v for row in cortex.light for v in row if v is not None]
    assert 0.2 < len(covered) / (len(cortex.light) * len(cortex.light[0])) < 0.8
    assert min(covered) >= 0.0 and max(covered) <= 1.0
    assert len(cortex.ink_order) > len(_portrait.INK_ORDER), "the asset should ship the measured ramp"


def test_resample_keeps_the_silhouette():
    cortex = _portrait.load_cortex()
    for width, height in ((40, 20), (96, 48)):
        cells = _portrait.resample(cortex.light, width, height)
        assert len(cells) == height and all(len(row) == width for row in cells)
        covered = sum(1 for row in cells for v in row if v is not None)
        assert 0.2 < covered / (width * height) < 0.8


def test_render_is_plain_and_deterministic_without_colour():
    first = _portrait.render(color_mode="none", width=110, height=34)
    assert first == _portrait.render(color_mode="none", width=110, height=34)
    assert "\x1b[" not in first
    assert "name: ModelJansen1995" in first
    lines = first.split("\n")
    assert len(lines) <= 34 and max(len(line) for line in lines) <= 110


def test_render_colours_a_terminal():
    out = _portrait.render(color_mode="truecolor", width=110, height=34)
    assert "\x1b[38;2;" in out and out.endswith("\x1b[0m")


def test_narrow_terminal_drops_the_spec_column():
    out = _portrait.render(color_mode="none", width=70, height=34)
    assert "name: ModelJansen1995" not in out
    assert max(len(line) for line in out.split("\n")) <= 70


def test_cortex_is_drawn_from_the_spec_characters():
    spec = "name: aaa\ndescription: bbb\n"
    canvas = _portrait.compose(spec, _portrait.load_cortex(), width=110, height=34)
    glyphs = {cell[0] for row in canvas.cells for cell in row if cell is not None}
    assert glyphs <= set(spec) - {" ", "\n"}


def test_dissolve_starts_at_the_spec_and_ends_at_the_cortex():
    text = Path(_portrait.DEFAULT_SPEC).read_text(encoding="utf-8")
    seq = list(_portrait.frames(text, n=12, hold=2, color_mode="none", width=110, height=34))
    assert "name: ModelJansen1995" in seq[0]
    assert "name: ModelJansen1995" not in seq[-1]
    assert seq[-1] == seq[-2], "the last frames hold on the finished cortex"
    spec_column = _portrait.SIZE["spec_width"]
    assert not any(line[:spec_column].strip() for line in seq[-1].split("\n")), "the spec fully dissolves"


def test_brain_verb_prints_a_portrait():
    res = runner.invoke(app, ["brain", "--plain", "-w", "110"])
    assert res.exit_code == 0
    assert "ModelJansen1995" in res.stdout


def test_brain_verb_accepts_a_curie_and_a_path():
    res = runner.invoke(app, ["brain", "model:Generic2dOscillator", "--plain", "-w", "110"])
    assert res.exit_code == 0
    assert "Generic2dOscillator" in res.stdout

    res = runner.invoke(app, ["brain", str(_portrait.DEFAULT_SPEC), "--plain", "-w", "110"])
    assert res.exit_code == 0
    assert "ModelJansen1995" in res.stdout


def test_brain_verb_rejects_an_unknown_spec_and_background():
    assert runner.invoke(app, ["brain", "no-such-spec-anywhere"]).exit_code != 0
    assert runner.invoke(app, ["brain", "--bg", "purple"]).exit_code != 0


def test_logo_asset_is_the_coloured_mark():
    logo = _portrait.load_logo()
    assert logo.width > 20 and logo.height >= 8 and logo.height % 2 == 0, "two pixel rows per character cell"
    assert {len(row) for row in logo.alpha} == {logo.width}
    lit = [logo.rgb[y][x] for y in range(logo.height) for x in range(logo.width) if logo.alpha[y][x] > 0.5]
    assert len(lit) > 40, "the mark covers real area"
    assert 2 <= len(set(lit)) <= 4, "flat brand colours, not a blur of intermediate tones"


def test_resample_logo_keeps_its_aspect():
    logo = _portrait.load_logo()
    small = _portrait.resample_logo(logo, 18)
    assert small.width == 18
    assert abs(small.height / small.width - logo.height / logo.width) < 0.12


def test_hero_spans_the_width_with_the_cortex_at_the_right_edge():
    cols = 78
    banner = _portrait.hero(subtitle="tvbo 0.0.0", color_mode="none", width=cols)
    lines = banner.split("\n")
    assert len(lines) <= _portrait.HERO["max_rows"], "the bare-tvbo banner stays small"
    assert max(len(line) for line in lines) == cols, "the cortex is flush with the right edge"
    cortex_rows = [y for y, line in enumerate(lines) if line[cols - 20 :].strip()]
    assert min(cortex_rows) == 0 and max(cortex_rows) == len(lines) - 1, "the cortex fills the band"


def test_hero_names_the_project_and_version_under_the_wordmark():
    banner = _portrait.hero(subtitle="tvbo 0.0.0", color_mode="none", width=78)
    left = [line[:30].rstrip() for line in banner.split("\n")]
    assert _portrait.TAGLINE in left and "tvbo 0.0.0" in left
    assert left.index(_portrait.TAGLINE) > left.index(_portrait.LOGO[-1].rstrip())


def test_hero_draws_the_wordmark_and_a_shaded_cortex():
    banner = _portrait.hero(color_mode="none", width=78)
    assert _portrait.LOGO[0].strip() in banner, "the wordmark is drawn"
    assert any(ch in banner for ch in _portrait.SURFACE_RAMP[1:]), "the cortex is shaded"


def test_hero_keeps_the_name_and_drops_the_cortex_when_very_narrow():
    banner = _portrait.hero(color_mode="none", width=26)
    lines = banner.split("\n")
    assert max(len(line) for line in lines) <= 26
    assert _portrait.LOGO[0].strip() in banner


def test_hero_can_still_draw_the_raster_mark():
    banner = _portrait.hero(mark="logo", color_mode="truecolor", width=78)
    assert "\x1b[48;2;" in banner, "half-block cells carry a second pixel as their backdrop"


def test_bare_tvbo_prints_the_command_list_when_captured():
    res = runner.invoke(app, [])
    assert res.exit_code == 0
    assert "Usage:" in res.stdout and "brain" in res.stdout
