# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Render an HTML page to a static file with a headless browser, from a figure spec.

A panel that shows a live web view — a knowledge-graph browser, a JavaScript force layout, the platform's own UI — has no matplotlib equivalent, and a screenshot someone took once drifts silently from the page it claims to show. An ``image`` panel carrying a ``capture:`` recipe re-renders its ``source`` on every build instead, so the committed raster is a build product like any other and the page is what is under version control.

Resolution is a device pixel ratio rather than a dpi: the page is laid out in CSS pixels and rasterised at ``device_scale_factor`` times that, so text and vector art in the page come out at the scale factor's full resolution. The default puts a panel-width capture above what a journal asks of a raster figure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_WIDTH = 1600
"""Viewport width in CSS pixels when the recipe names none."""

DEFAULT_SCALE = 3.0
"""Device pixel ratio when the recipe names none, so a default capture is already above print resolution."""

_REMOTE = ("http://", "https://", "file://")


def _recipe(spec) -> dict:
    """A ``RenderSpec`` (record, dict or None) as a plain dict of the options it actually sets."""
    if spec is None:
        return {}
    fields = spec if isinstance(spec, dict) else getattr(spec, "__dict__", {})
    return {k: v for k, v in fields.items() if v is not None and not k.startswith("_")}


def _url(source: str, base_dir=None) -> tuple[str, Path | None]:
    """``(url, local_path)`` for a capture source — a local file resolved against the spec's directory, or a URL passed through."""
    if source.startswith(_REMOTE):
        return source, None
    local = Path(source) if os.path.isabs(source) else Path(base_dir or ".") / source
    return local.resolve().as_uri(), local


def recipe_sidecar(path) -> Path:
    """Where the recipe a capture was taken with is recorded, beside the file it produced."""
    out = Path(path)
    return out.with_name(out.name + ".recipe.json")


def is_stale(path, source_path, recipe: dict | None = None) -> bool:
    """Whether *path* has to be re-captured: it is missing, its local source has changed, or it was taken with a different recipe.

    The recipe counts because it decides what the file *is*: raising ``device_scale_factor`` or changing the viewport with the page untouched must re-render, or the spec says one thing and the committed raster is another. Split out from :func:`capture` so the decision is testable without a browser, and so a build can report what it is about to re-render.
    """
    out = Path(path)
    if not out.exists():
        return True
    if recipe is not None:
        sidecar = recipe_sidecar(out)
        try:
            if json.loads(sidecar.read_text()) != recipe:
                return True
        except (OSError, ValueError):
            return True
    if source_path is None or not Path(source_path).exists():
        return False
    return out.stat().st_mtime < Path(source_path).stat().st_mtime


def capture(source: str, path, spec=None, *, base_dir=None, force: bool = False) -> Path:
    """Render *source* to *path* with a headless Chromium, honouring a ``RenderSpec``, and return the path written.

    A capture whose output is already newer than its local source is skipped, so re-running a build does not re-launch a browser for a page nothing has touched; *force* takes the shot regardless. A remote source has no mtime to compare, so it is captured once and then only on *force*.
    """
    out = Path(path)
    url, local = _url(str(source), base_dir)
    options = _recipe(spec)
    if not force and not is_stale(out, local, options):
        return out

    width = int(options.get("width") or DEFAULT_WIDTH)
    height = options.get("height")
    scale = float(options.get("device_scale_factor") or DEFAULT_SCALE)
    fmt = str(options.get("format") or out.suffix.lstrip(".") or "png").lower()
    if fmt == "svg":
        raise ValueError("a headless capture cannot emit SVG; render the page to png or pdf, or draw the panel natively")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "an `image` panel with a `capture:` recipe needs playwright: pip install playwright, then playwright install chromium"
        ) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": int(height or width)},
            device_scale_factor=scale,
        )
        page.goto(url, wait_until="networkidle", timeout=60_000)
        if options.get("wait_selector"):
            page.wait_for_selector(str(options["wait_selector"]), timeout=30_000)
        if options.get("warmup"):
            page.wait_for_timeout(int(float(options["warmup"]) * 1000))
        if fmt == "pdf":
            page.pdf(path=str(out))
        else:
            shot = {"path": str(out), "full_page": height is None and not options.get("clip")}
            if options.get("clip"):
                x, y, w, h = (float(v) for v in options["clip"])
                shot["clip"] = {"x": x, "y": y, "width": w, "height": h}
            page.screenshot(**shot)
        browser.close()
    recipe_sidecar(out).write_text(json.dumps(options, sort_keys=True))
    return out
