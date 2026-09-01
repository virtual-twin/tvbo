# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""The headless-capture recipe behind an ``image`` panel, tested without launching a browser."""

import pytest

from tvbo.adapters.bsplot import _recipe_dict
from tvbo.plot.capture import _recipe, _url, capture, is_stale


class _Spec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_recipe_keeps_only_the_options_a_spec_sets():
    assert _recipe(_Spec(width=800, height=None, warmup=1.5)) == {"width": 800, "warmup": 1.5}
    assert _recipe({"device_scale_factor": 4.0, "clip": None}) == {"device_scale_factor": 4.0}
    assert _recipe(None) == {}


def test_recipe_dict_reports_an_unset_recipe_as_none():
    """The panel context carries None rather than an empty dict, so a spec without a recipe cannot look like one with an empty one."""
    assert _recipe_dict(None) is None
    assert _recipe_dict(_Spec(width=None)) is None
    assert _recipe_dict({"width": 800}) == {"width": 800}


def test_a_local_source_resolves_against_the_spec_directory(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<p>hi</p>")
    url, local = _url("page.html", base_dir=tmp_path)
    assert url.startswith("file://") and url.endswith("/page.html")
    assert local == tmp_path / "page.html"
    assert _url("https://example.org/a", base_dir=tmp_path) == ("https://example.org/a", None)


def test_staleness_follows_the_source(tmp_path):
    source, out = tmp_path / "page.html", tmp_path / "shot.png"
    source.write_text("<p>hi</p>")
    assert is_stale(out, source), "a capture that was never taken is stale"
    out.write_bytes(b"png")
    assert not is_stale(out, source)
    source.touch()
    assert is_stale(out, source), "editing the page must re-render the capture"


def test_capture_skips_a_fresh_output_without_needing_a_browser(tmp_path):
    """The staleness check runs before playwright is imported, so a build over unchanged pages launches nothing."""
    from tvbo.plot.capture import recipe_sidecar

    source, out = tmp_path / "page.html", tmp_path / "shot.png"
    source.write_text("<p>hi</p>")
    out.write_bytes(b"png")
    recipe_sidecar(out).write_text("{}")
    assert capture("page.html", out, base_dir=tmp_path) == out


def test_svg_is_refused_rather_than_silently_rasterised(tmp_path):
    with pytest.raises(ValueError, match="cannot emit SVG"):
        capture("page.html", tmp_path / "out.svg", {"format": "svg"}, base_dir=tmp_path, force=True)


def test_an_authored_description_replaces_the_image_panels_filename():
    """A caption says what the panel shows; the file it was rendered from is provenance and only stands in when nothing was written."""
    from tvbo.adapters.bsplot import _panel_descriptor

    assert _panel_descriptor(_Spec(kind="image", source="full-graph.html", description=None)) == "rendered from full-graph.html"
    assert _panel_descriptor(_Spec(kind="image", source="full-graph.html", description="Both resources at full size.")) == ""


def test_changing_the_recipe_restales_an_untouched_page(tmp_path):
    """The recipe decides what the file is: a new device_scale_factor or viewport must re-render, or the spec and the committed raster disagree."""
    from tvbo.plot.capture import recipe_sidecar

    source, out = tmp_path / "page.html", tmp_path / "shot.png"
    source.write_text("<p>hi</p>")
    out.write_bytes(b"png")
    recipe_sidecar(out).write_text('{"width": 1000, "height": 1000}')
    assert not is_stale(out, source, {"width": 1000, "height": 1000})
    assert is_stale(out, source, {"width": 1000, "height": 1180})
    assert is_stale(out, source, {}), "a capture with no recorded recipe cannot be shown to match one"
