# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""The Simulation Study Lifecycle wheel: the six stages a study passes through, drawn as one self-contained SVG.

The stages (Explore, Specify, Generate, Run, Report, Share), their names, summaries, palette hues and icons are declared once in ``lifecycle.yaml`` beside this module, and the documentation sidebar, the README and the lecture read them from there. The wheel is a ring of cells grouped by stage, a chevron at every stage boundary, an arrow across the opening that closes the cycle from Share back to Explore, and a dotted *revise* loop inside the ring from Report back to Specify; the title runs on an arc above the opening, and the TVB-O logo lies on a brain in the middle.

Every element is a vector drawn in the theme's colours, with no raster, font file or external reference, so the file renders wherever SVG does, including inside an ``<img>`` on GitHub and PyPI, whose content security policy refuses anything the file would fetch.

Example:
    ```python
    from tvbo.plot import lifecycle

    svg = lifecycle.wheel()  # light, one cell per stage, icons in the bands
    dark = lifecycle.wheel("tvbo:themes/dark", band="numbers")
    ```
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

SIZE = 800.0
C = SIZE / 2
R_OUT, R_IN = 318.0, 268.0
R_MID = (R_OUT + R_IN) / 2
R_LABEL = 362.0
R_TITLE = 338.0
R_REVISE = R_IN - 30
OPEN_DEG, STAGE_GAP_DEG, CELL_GAP_DEG = 50.0, 11.0, 1.4
HEAD = 15.0
REVISE_HEAD = 9.0
STROKE = 3.5
ICON = 30.0
BRAIN_H = 470.0
BRAIN_OPACITY = 0.5
LOGO_W = 250.0
GRAPH_OPACITY = 0.55
FONT = "'Space Grotesk', 'Source Sans Pro', system-ui, sans-serif"
NAME_PX, NUMBER_PX, TITLE_PX, REVISE_PX = 30, 24, 26, 19
BAND_OUT, BAND_DEPTH, BAND_GAP, SUB_GAP_DEG = 296.0, 130.0, 4.0, 1.2
BAND_BRAIN_H, BAND_LOGO_W, LEVEL_PX = 280.0, 170.0, 17
PASSED, DIMMED = 0.45, 0.78
KEY_H, KEY_PX, KEY_SWATCH = 56.0, 20, (36.0, 20.0)
BRIEF_PX = 19
KEY_TITLE = "One cell, one practice:"


@cache
def stages() -> dict[str, dict[str, Any]]:
    """The six stages in order, keyed by stage, each with its ``name``, ``summary``, ``color`` slot, line ``icon`` and Font Awesome ``fa`` glyph."""
    import yaml

    return yaml.safe_load((HERE / "lifecycle.yaml").read_text())["stages"]


@cache
def maturity() -> dict[str, Any]:
    """The scientific-operations maturity table: ``levels`` in order from Initial to Optimizing, the ``statuses`` a capability takes, and the ``capabilities``, each with its ``level``, ``stage``, ``practice``, ``status`` and evidence."""
    import yaml

    return yaml.safe_load((HERE / "maturity.yaml").read_text())


def _level_number(level: str) -> int:
    """The 1-based position of *level* in the maturity table."""
    return list(maturity()["levels"]).index(level) + 1


def _level_label(level: str) -> str:
    """*level*'s number and name, as ``L3 Defined``."""
    return f"L{_level_number(level)} {maturity()['levels'][level]['name']}"


def _theme(theme) -> dict:
    """A theme's colours from a ``tvbo:themes/<name>`` IRI, a file path or a mapping, every role it leaves out taken from the default."""
    from tvbo.data.registry import iri_target, resolve_iri
    from tvbo.plot import palette

    return palette.load(resolve_iri(theme) if isinstance(theme, str) and iri_target(theme) else theme)


def _mix(a: str, b: str, t: float) -> str:
    """The colour *t* of the way from *a* to *b*."""
    from matplotlib.colors import to_hex, to_rgb

    return to_hex([x * (1 - t) + y * t for x, y in zip(to_rgb(a), to_rgb(b), strict=True)])


def _pt(angle: float, r: float) -> tuple[float, float]:
    """The point at *angle* degrees clockwise from twelve o'clock, radius *r*."""
    t = math.radians(angle - 90)
    return C + r * math.cos(t), C + r * math.sin(t)


def _xy(p: tuple[float, float]) -> str:
    return f"{p[0]:.1f} {p[1]:.1f}"


def _tangent(angle: float) -> tuple[float, float]:
    """Unit vector pointing clockwise along the ring at *angle*."""
    t = math.radians(angle)
    return math.cos(t), math.sin(t)


def _radial(angle: float) -> tuple[float, float]:
    """Unit vector pointing outward at *angle*."""
    t = math.radians(angle - 90)
    return math.cos(t), math.sin(t)


def _annulus(a0: float, a1: float, r_in: float, r_out: float) -> str:
    """The annular sector from *a0* to *a1* degrees between radii *r_in* and *r_out*."""
    return f"M {_xy(_pt(a0, r_out))} A {r_out} {r_out} 0 0 1 {_xy(_pt(a1, r_out))} L {_xy(_pt(a1, r_in))} A {r_in} {r_in} 0 0 0 {_xy(_pt(a0, r_in))} Z"


def _head(tip: tuple[float, float], direction: tuple[float, float], size: float = HEAD) -> str:
    """The arrowhead the wheel's arrows end in: an open chevron of half-width *size* with its tip at *tip*, pointing along *direction*."""
    dx, dy = direction
    bx, by = tip[0] - 0.8 * size * dx, tip[1] - 0.8 * size * dy
    return f'<path class="arrow" d="M {_xy((bx - size * dy, by + size * dx))} L {_xy(tip)} L {_xy((bx + size * dy, by - size * dx))}"/>'


def _layout(cells: list[dict]) -> tuple[list[dict], list[float], dict[str, float]]:
    """Each cell's angles, the angle of every stage boundary, and each stage's middle angle."""
    order = list(stages())
    span = 360.0 - OPEN_DEG - (len(order) - 1) * STAGE_GAP_DEG
    width = span / len(cells)
    a, placed, gaps = OPEN_DEG / 2, [], []
    for i, cell in enumerate(cells):
        if i and cell["stage"] != cells[i - 1]["stage"]:
            gaps.append(a + STAGE_GAP_DEG / 2)
            a += STAGE_GAP_DEG
        placed.append({**cell, "a0": a + CELL_GAP_DEG / 2, "a1": a + width - CELL_GAP_DEG / 2})
        a += width
    mids = {}
    for s in order:
        own = [c for c in placed if c["stage"] == s]
        mids[s] = (own[0]["a0"] + own[-1]["a1"]) / 2
    return placed, gaps, mids


def _arc_text(ident: str, text: str, mid: float, r: float, half: float, cls: str, fill: str | None = None) -> str:
    """*text* centred on an arc of radius *r* around *mid*, upright on whichever half of the ring it sits, in *fill* when given and else in its class's colour."""
    upper = not 90 < mid % 360 < 270
    a, b = (mid - half, mid + half) if upper else (mid + half, mid - half)
    track = f"M {_xy(_pt(a, r))} A {r} {r} 0 0 {1 if upper else 0} {_xy(_pt(b, r))}"
    paint = f' fill="{fill}"' if fill else ""
    return f'<path id="{ident}" fill="none" d="{track}"/><text class="{cls}"{paint}><textPath href="#{ident}" startOffset="50%" text-anchor="middle">{text}</textPath></text>'


def _revise(cells: list[dict], r_in: float = R_IN, inset: float = R_IN - R_REVISE, lead: float = 16.0) -> str:
    """The inner loop from Report back to Specify, drawn *inset* inside a ring whose inner radius is *r_in*: a radial lead *lead* long out of the first Report cell, a dotted arc concentric with the ring, and a radial turn into the last Specify cell ending in the arrowhead, with its name upright in a gap at the middle."""
    report = next(c for c in cells if c["stage"] == "report")
    specify = [c for c in cells if c["stage"] == "specify"][-1]
    a0, a1 = (report["a0"] + report["a1"]) / 2, (specify["a0"] + specify["a1"]) / 2
    r = r_in - inset
    corner = math.degrees(lead / r)
    mid = (a0 + a1) / 2
    gap = math.degrees(40.0 / r)
    tip = _pt(a1, r_in - 6)
    first = f"M {_xy(_pt(a0, r_in - 6))} L {_xy(_pt(a0, r + lead))} Q {_xy(_pt(a0, r))} {_xy(_pt(a0 - corner, r))} A {r} {r} 0 0 0 {_xy(_pt(mid + gap, r))}"
    second = f"M {_xy(_pt(mid - gap, r))} A {r} {r} 0 0 0 {_xy(_pt(a1 + corner, r))} Q {_xy(_pt(a1, r))} {_xy(_pt(a1, r + lead))} L {_xy(tip)}"
    upper = not 90 < mid % 360 < 270
    label = _arc_text("revise-track", "revise", mid, r + (-7 if upper else 7), 2 * gap, "revise-label")
    return (
        f'<path class="revise" d="{first}"/><path class="revise" d="{second}"/>{_head(tip, _radial(a1), REVISE_HEAD)}{label}'
    )


def _icon(stage: str, icons: str, x: float, y: float, size: float, colour: str, standalone: bool = False) -> str:
    """*stage*'s icon, a *size*-pixel square at (*x*, *y*), drawn in *colour*; *standalone* declares the SVG namespace so the icon is a file of its own."""
    entry = stages()[stage]
    if icons == "line":
        view, body = (
            "0 0 24 24",
            f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{entry["icon"]}',
        )
    elif icons == "fa":
        view, body = entry["fa"]["viewbox"], f'fill="{colour}"><path d="{entry["fa"]["path"]}"/>'
    else:
        raise ValueError(f"icons must be 'line' or 'fa', not {icons!r}")
    ns = 'xmlns="http://www.w3.org/2000/svg" ' if standalone else ""
    return f'<svg {ns}x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" viewBox="{view}" {body}</svg>'


def _hue(colours: dict, stage: str) -> str:
    """The theme's colour for *stage*: the palette hue its row in ``lifecycle.yaml`` names."""
    from tvbo.plot.palette import _hue as palette_hue

    return palette_hue(str(stages()[stage]["color"]), colours["palette"])


def icon(stage: str, theme: str | Path | Mapping = "tvbo:themes/default", *, icons: str = "line", size: int = 24) -> str:
    """*stage*'s icon as a standalone SVG in the stage's colour, for a heading or a list beside the wheel.

    Args:
        stage: A stage key of ``lifecycle.yaml`` (``explore`` … ``share``).
        theme: The colours, as :func:`wheel` takes them.
        icons: ``line`` for TVB-O's line icons, ``fa`` for the Font Awesome glyphs.
        size: Width and height in pixels.

    Raises:
        ValueError: *stage* is not a lifecycle stage.
    """
    if stage not in stages():
        raise ValueError(f"{stage!r} is not a lifecycle stage; the stages are {list(stages())}")
    return _icon(stage, icons, 0, 0, size, _hue(_theme(theme), stage), standalone=True) + "\n"


@cache
def _asset(name: str) -> tuple[str, list[float]]:
    """The inner markup of a vector asset and its viewBox."""
    text = (ASSETS / name).read_text()
    match = re.search(r'<svg[^>]*viewBox="([^"]+)"[^>]*>(.*)</svg>', text, re.S)
    return match.group(2), [float(v) for v in match.group(1).split()]


def _nested(
    name: str, cx: float, cy: float, *, width: float | None = None, height: float | None = None, attrs: str = ""
) -> str:
    """The asset *name* scaled to *width* or *height* and centred on (*cx*, *cy*)."""
    body, box = _asset(name)
    vw, vh = box[2:]
    w = width if width is not None else height * vw / vh
    h = height if height is not None else width * vh / vw
    view = " ".join(f"{v:g}" for v in box)
    return f'<svg x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" viewBox="{view}" {attrs}>{body}</svg>'


def _labels(
    cells: list[dict], mids: dict[str, float], band: str, icons: str, hue: dict[str, str], briefs: bool = False
) -> tuple[list[str], list[float]]:
    """Each stage's name outside the ring, with its brief under it when *briefs* is set, and the horizontal extent the labels reach, so the canvas can fit them."""
    out, xs = [], []
    for number, (stage, entry) in enumerate(stages().items(), start=1):
        x, y = _pt(mids[stage], R_LABEL)
        dy = 10 if abs(y - C) < 100 else (24 if y > C else 0)
        name = f"{number} {entry['name']}" if band == "icons" else entry["name"]
        anchor = "start" if x > C + 40 else "end" if x < C - 40 else "middle"
        side = -1 if anchor == "end" else 1
        offset = 0 if band == "icons" else ICON + 10
        text_x = x + side * offset
        text_anchor = anchor if band == "icons" else "end" if side < 0 else "start"
        width = 0.6 * NAME_PX * len(name) + offset
        if briefs:
            width = max(width, 0.56 * BRIEF_PX * len(entry["brief"]) + offset)
            out.append(
                f'<text class="brief brief-{stage}" x="{text_x:.1f}" y="{y + dy + BRIEF_PX + 8:.1f}" text-anchor="{text_anchor}">{entry["brief"]}</text>'
            )
        glyph = "" if band == "icons" else _icon(stage, icons, x - ICON if side < 0 else x, y + dy - 24, ICON, hue[stage])
        out.append(
            glyph
            + f'<text class="name lbl-{stage}" x="{text_x:.1f}" y="{y + dy:.1f}" text-anchor="{text_anchor}">{name}</text>'
        )
        xs += [x - width, x] if anchor == "end" else [x, x + width] if anchor == "start" else [x - width / 2, x + width / 2]
    return out, xs


def band_radii(bands: list[str]) -> dict[str, tuple[float, float]]:
    """The inner and outer radius of each band, the first innermost and the last ending at ``BAND_OUT``; the bands share ``BAND_DEPTH``, so fewer bands are wider."""
    width = (BAND_DEPTH - (len(bands) - 1) * BAND_GAP) / len(bands)
    inner = BAND_OUT - BAND_DEPTH
    return {b: (inner + i * (width + BAND_GAP), inner + i * (width + BAND_GAP) + width) for i, b in enumerate(bands)}


def _status_paint(status: str, tone: str, patterns: dict[str, str]) -> str:
    """The fill and stroke attributes of a cell with *status* in *tone*: filled when it ships, hatched when partial, outlined when not yet; a hatch registers its pattern in *patterns*."""
    if status == "not yet":
        return f'fill="none" style="stroke:{tone};stroke-width:2.5"'
    if status == "partial":
        pid = f"hatch-{tone.lstrip('#')}"
        patterns.setdefault(
            pid,
            f'<pattern id="{pid}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="3.5" height="7" fill="{tone}"/></pattern>',
        )
        return f'fill="url(#{pid})" style="stroke:{tone};stroke-width:1.5"'
    return f'fill="{tone}"'


def _band_cells(
    cells: list[dict],
    spans: dict[str, tuple[float, float]],
    radii: dict[str, tuple[float, float]],
    current: str | None,
    hue: dict[str, str],
    page: str,
    patterns: dict[str, str],
    focus: tuple[str | None, str | None] | None = None,
    lit: set[str] | None = None,
) -> list[str]:
    """Every stage-by-band cell, split into one sub-cell per capability painted by :func:`_status_paint`, and one dimmed cell where a stage and band have none; bands outside *current* take the passed tone inside it and the dimmed tone beyond it, and with *lit* only those stages of *current* keep their full tone. With *focus*, a ``(band, stage)`` pair where ``None`` stands for all, only the cells it names keep their full tone and every other cell is dimmed."""
    order = list(radii)
    out = []
    for stage, (a0, a1) in spans.items():
        for k, b in enumerate(order):
            own = [c for c in cells if c["stage"] == stage and c["band"] == b]
            ahead = current is not None and k > order.index(current)
            if focus is not None:
                focused = focus[0] in (None, b) and focus[1] in (None, stage)
                tone = hue[stage] if focused else _mix(hue[stage], page, DIMMED)
            else:
                full = current is None or (b == current and (lit is None or stage in lit))
                tone = hue[stage] if full else _mix(hue[stage], page, DIMMED if ahead or b == current else PASSED)
            r0, r1 = radii[b]
            width = (a1 - a0 - SUB_GAP_DEG * (max(len(own), 1) - 1)) / max(len(own), 1)
            for i, c in enumerate(own or [{"status": None}]):
                b0 = a0 + i * (width + SUB_GAP_DEG)
                d = _annulus(b0, b0 + width, r0, r1)
                status = c["status"]
                cls = f"bcell band-{b} stage-{stage} status-{str(status).replace(' ', '-')}"
                paint = f'fill="{_mix(hue[stage], page, DIMMED)}"' if status is None else _status_paint(status, tone, patterns)
                out.append(f'<path class="{cls}" {paint} d="{d}"/>')
    return out


def _status_key(cells: list[dict], tone: str, patterns: dict[str, str]) -> list[str]:
    """The key below the wheel: what a cell is, then one swatch and name per status the drawn cells carry, in the table's order, centred in the strip; nothing when the cells carry fewer than two statuses, since there is then nothing to tell apart."""
    shown = [s for s in maturity()["statuses"] if any(c["status"] == s for c in cells)]
    if len(shown) < 2:
        return []
    w, h = KEY_SWATCH
    widths = [w + 10 + 0.56 * KEY_PX * len(s) for s in shown]
    lead = 0.56 * KEY_PX * len(KEY_TITLE) + 28
    x = C - (lead + sum(widths) + 36 * (len(shown) - 1)) / 2
    y = SIZE + KEY_H / 2
    out = [f'<text class="key key-title" x="{x:.1f}" y="{y + KEY_PX * 0.35:.1f}">{KEY_TITLE}</text>']
    x += lead
    for status, width in zip(shown, widths, strict=True):
        out.append(
            f'<rect class="key-swatch" x="{x:.1f}" y="{y - h / 2:.1f}" width="{w}" height="{h}" {_status_paint(status, tone, patterns)}/>'
        )
        out.append(f'<text class="key" x="{x + w + 10:.1f}" y="{y + KEY_PX * 0.35:.1f}">{status.capitalize()}</text>')
        x += width + 36
    return out


def level_radius(band: tuple[float, float]) -> float:
    """The radius a band's name sits on in the opening: the band's middle, lowered by a third of the type so the line looks centred."""
    return (band[0] + band[1]) / 2 - LEVEL_PX / 3


def _band_names(radii: dict[str, tuple[float, float]], current: str | None, ink: str, page: str) -> list[str]:
    """Each band's number and name on an arc in the opening at the top, the current band in ink, the others quieter."""
    order = list(radii)
    out = []
    for k, b in enumerate(order):
        ahead = current is not None and k > order.index(current)
        fill = ink if current in (None, b) else _mix(ink, page, 0.6 if ahead else 0.3)
        out.append(_arc_text(f"band-{b}", _level_label(b), 0.0, level_radius(radii[b]), OPEN_DEG / 2 - 2, "level", fill))
    return out


def _palette(theme) -> tuple[dict, str, str, str, dict[str, str]]:
    """A theme's colours, its page and ink, the quiet ink the arrows and briefs take, and each stage's hue."""
    colours = _theme(theme)
    page, ink = colours["background"], colours["ink"]
    return colours, page, ink, _mix(ink, page, 0.36), {s: _hue(colours, s) for s in stages()}


def _frame(
    cell: str, own: str, colours: dict, quiet: str, briefs: bool, title: str, brain_h: float, logo_w: float
) -> list[str]:
    """The style sheet, the brain *brain_h* high under the logo *logo_w* wide in the centre, and *title* on its arc when there is one.

    The style sheet opens with the wheel's *cell* rule, then the rules both wheels share for the stage names and, when *briefs* is set, the briefs, then the wheel's *own* rules, and closes with the shared rules for the arrows, the revise loop and the logo.
    """
    ink = colours["ink"]
    brief = f".brief{{fill:{quiet};font-size:{BRIEF_PX}px;font-weight:500}}" if briefs else ""
    parts = [
        f"<style>text{{font-family:{FONT}}}{cell}.name{{fill:{ink};font-size:{NAME_PX}px;font-weight:600}}{brief}{own}"
        f".arrow,.cycle{{fill:none;stroke:{quiet};stroke-width:{STROKE};stroke-linecap:round;stroke-linejoin:round}}"
        f".revise{{fill:none;stroke:{quiet};stroke-width:{STROKE};stroke-linecap:round;stroke-dasharray:.1 9}}"
        f".revise-label{{fill:{quiet};font-size:{REVISE_PX}px;font-weight:600;letter-spacing:.02em}}"
        f".logo .lettering{{fill:{ink}}}.logo .graph{{fill:{ink};stroke:{ink};opacity:{GRAPH_OPACITY}}}</style>",
        _nested("lifecycle-brain.svg", C, C, height=brain_h, attrs=f'color="{colours["base"]}" opacity="{BRAIN_OPACITY}"'),
        _nested("tvbo-logo.svg", C, C, width=logo_w, attrs='class="logo"'),
    ]
    if title:
        parts.append(_arc_text("title-track", title, 0.0, R_TITLE, 40.0, "title"))
    return parts


def _cycle(half: float, r: float) -> str:
    """The arrow across the opening on radius *r*, from *half* degrees left of twelve o'clock to *half* degrees right, closing the cycle from Share back to Explore."""
    return f'<path class="cycle" d="M {_xy(_pt(-half, r))} A {r} {r} 0 0 1 {_xy(_pt(half, r))}"/>{_head(_pt(half, r), _tangent(half))}'


def _bounds(xs: list[float]) -> tuple[float, float]:
    """The canvas's left and right edge: the square wheel widened to the labels' reach *xs*, with a margin."""
    return min(min(xs), 0.0) - 16, max(max(xs), SIZE) + 16


def _document(parts: list[str], view: tuple[float, float, float, float], aria: str, ground: str | None = None) -> str:
    """The standalone SVG around *parts*, showing the box *view* as ``(x, y, width, height)`` at its own size, on a page painted in *ground*, or transparent without it."""
    x, y, w, h = view
    at = "".join(f' {k}="{v:.0f}"' for k, v in (("x", x), ("y", y)) if v)
    rect = f'<rect{at} width="{w:.0f}" height="{h:.0f}" fill="{ground}"/>' if ground else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.0f} {y:.0f} {w:.0f} {h:.0f}" '
        f'width="{w:.0f}" height="{h:.0f}" role="img" aria-label="{aria}">{rect}' + "".join(parts) + "</svg>\n"
    )


def wheel(
    theme: str | Path | Mapping = "tvbo:themes/default",
    *,
    cells: Iterable[Mapping] | None = None,
    band: str = "icons",
    icons: str = "line",
    title: str = "Simulation Study Lifecycle",
    bands: Iterable[str] | None = None,
    current: str | None = None,
    lit: Iterable[str] | None = None,
    ground: bool = True,
    briefs: bool = False,
) -> str:
    """The lifecycle wheel as one standalone SVG document.

    Args:
        theme: The colours: a ``tvbo:themes/<name>`` IRI, a theme file, or a mapping of roles. Stage ``i`` takes the theme's hue ``palette.i``; the page, ink and base roles colour the ground, the text and the brain.
        cells: The ring's cells in order, each a mapping with a ``stage`` key; consecutive cells of one stage form its group. Defaults to one cell per stage. The lecture passes one per chapter.
        band: ``icons`` draws each cell's stage icon in the band and the numbered stage names outside; ``numbers`` draws each cell's number in the band and the stage icon beside each name outside. Band glyphs are knocked out in the page colour: every stage's name and number also stand outside the band, so the glyph repeats them rather than carrying them alone.
        icons: ``line`` for the 24-pixel line icons, ``fa`` for the Font Awesome Free glyphs the documentation sidebar uses.
        title: The text on the arc above the opening; empty for none.
        bands: Maturity levels of ``maturity.yaml`` to draw as concentric bands, innermost first, named ``L<n> <Name>`` in the opening. Each cell then carries a ``band`` and a ``status`` from the table's statuses, and the cells of one stage and band share that cell's span; *cells* defaults to the table's capabilities. Without *bands* the wheel is the single ring.
        current: The band a divider is at: bands inside it take the passed tone, bands beyond it the dimmed tone. Without it every band is in full tone. A banded wheel carries a key below it naming the statuses its cells show.
        lit: With *current*, the lifecycle stages of that band to keep in full tone, its other stages taking the dimmed tone, so a page can say which stages a level acts in. Without it the whole band is in full tone.
        ground: Paint the page in the theme's background; ``False`` leaves it transparent for a slide or page that paints its own, where an image's colour management would otherwise show the ground as a square.
        briefs: Set each stage's brief from ``lifecycle.yaml`` in a quieter line under its name, so the wheel says what each stage is for.

    Returns:
        The SVG markup, with a ``role="img"`` and an ``aria-label`` naming the stages in order.
    """
    if band not in ("icons", "numbers"):
        raise ValueError(f"band must be 'icons' or 'numbers', not {band!r}")
    order = list(stages())
    if bands is not None:
        return _banded(theme, list(bands), current, cells, icons, title, ground, briefs, lit)
    cells = [dict(c) for c in cells] if cells is not None else [{"stage": s} for s in order]
    unknown = sorted({c["stage"] for c in cells} - set(order))
    if unknown:
        raise ValueError(f"cells name {unknown}, which are not lifecycle stages; the stages are {order}")
    if [s for i, s in enumerate(c["stage"] for c in cells) if i == 0 or s != cells[i - 1]["stage"]] != order:
        raise ValueError(f"cells must run through every stage once, in order: {order}")

    colours, page, ink, quiet, hue = _palette(theme)
    placed, gaps, mids = _layout(cells)

    own = f".num{{fill:{page};font-size:{NUMBER_PX}px;font-weight:700}}.title{{fill:{ink};font-size:{TITLE_PX}px;font-weight:600;letter-spacing:.03em}}"
    parts = _frame(f".cell{{stroke:{page};stroke-width:2}}", own, colours, quiet, briefs, title, BRAIN_H, LOGO_W)
    for number, c in enumerate(placed, start=1):
        s = c["stage"]
        parts.append(
            f'<path class="cell cell-{number - 1} stage-{s}" fill="{hue[s]}" d="{_annulus(c["a0"], c["a1"], R_IN, R_OUT)}"/>'
        )
        x, y = _pt((c["a0"] + c["a1"]) / 2, R_MID)
        if band == "icons":
            parts.append(_icon(s, icons, x - ICON / 2, y - ICON / 2, ICON, page))
        else:
            parts.append(
                f'<text class="num num-{number - 1}" x="{x:.1f}" y="{y + 8:.1f}" text-anchor="middle">{number}</text>'
            )
    parts += [_head(_pt(g + math.degrees(0.4 * HEAD / R_MID), R_MID), _tangent(g)) for g in gaps]
    parts.append(_cycle(OPEN_DEG / 2 - 5, R_MID))
    parts.append(_revise(placed))
    labels, xs = _labels(placed, mids, band, icons, hue, briefs)
    parts += labels

    left, right = _bounds(xs)
    names = ", ".join(f"{i} {stages()[s]['name']}" for i, s in enumerate(order, start=1))
    aria = f"{title}: {names}; Report loops back to Specify, Share leads to the next Explore" if title else names
    return _document(parts, (left, 0.0, right - left, SIZE), aria, page if ground else None)


def _table_cells(cells: Iterable[Mapping] | None, bands: list[str]) -> list[dict]:
    """The cells that fall in *bands*, each as its stage, band and status; *cells* defaults to the table's capabilities, and a band or status the table does not know is refused."""
    table = maturity()
    unknown = sorted(set(bands) - set(table["levels"]))
    if unknown:
        raise ValueError(f"bands name {unknown}, which are not maturity levels; the levels are {list(table['levels'])}")
    cells = [
        {"stage": c["stage"], "band": c.get("band", c.get("level")), "status": c["status"]}
        for c in (cells if cells is not None else table["capabilities"])
    ]
    bad = sorted({c["status"] for c in cells} - set(table["statuses"]))
    if bad:
        raise ValueError(f"cells carry statuses {bad}; the statuses are {list(table['statuses'])}")
    return [c for c in cells if c["band"] in bands]


def locator(
    theme: str | Path | Mapping = "tvbo:themes/default",
    *,
    bands: Iterable[str],
    level: str | None = None,
    stage: str | None = None,
    cells: Iterable[Mapping] | None = None,
) -> str:
    """A small banded wheel that says where a page is: the cells of *level* and *stage* in full tone, every other cell dimmed.

    Only the bands are drawn, with no names, key or title, on a transparent page cropped to the outer band, so it reads at the size of a corner mark; the centre carries the lit level's number (``L3``) in bold. Leaving *level* or *stage* out lights the whole ring or the whole segment.

    Args:
        theme: The colours, as for :func:`wheel`.
        bands: Maturity levels to draw, innermost first.
        level: The band to light, one of *bands*.
        stage: The lifecycle stage to light.
        cells: The capabilities, as for :func:`wheel`; defaults to the table's.

    Returns:
        The SVG markup, with an ``aria-label`` naming the level and stage.
    """
    bands = list(bands)
    if level is not None and level not in bands:
        raise ValueError(f"level must be one of the bands {bands}, not {level!r}")
    if stage is not None and stage not in stages():
        raise ValueError(f"stage must be a lifecycle stage {list(stages())}, not {stage!r}")
    cells = _table_cells(cells, bands)
    _, page, ink, _, hue = _palette(theme)
    placed, _, _ = _layout([{"stage": s} for s in stages()])
    spans = {c["stage"]: (c["a0"], c["a1"]) for c in placed}
    patterns: dict[str, str] = {}
    parts = [f"<style>.bcell{{stroke:{page};stroke-width:3}}</style>"]
    parts += _band_cells(cells, spans, band_radii(bands), None, hue, page, patterns, focus=(level, stage))
    if patterns:
        parts.append(f"<defs>{''.join(patterns.values())}</defs>")
    where = ", ".join(
        part
        for part in (
            _level_label(level) if level else "",
            stages()[stage]["name"] if stage else "",
        )
        if part
    )
    if level:
        size = 1.1 * band_radii(bands)[bands[0]][0]
        parts.append(
            f'<text x="{C}" y="{C + 0.36 * size:.1f}" text-anchor="middle" font-family="{FONT}" font-size="{size:.0f}" font-weight="700" fill="{ink}">L{_level_number(level)}</text>'
        )
    r = BAND_OUT + 4
    return _document(parts, (C - r, C - r, 2 * r, 2 * r), f"Lifecycle and maturity: {where or 'all'}")


def _banded(
    theme,
    bands: list[str],
    current: str | None,
    cells: Iterable[Mapping] | None,
    icons: str,
    title: str,
    ground: bool,
    briefs: bool = False,
    lit: Iterable[str] | None = None,
) -> str:
    """The wheel with one concentric band per maturity level; see :func:`wheel`."""
    if current is not None and current not in bands:
        raise ValueError(f"current must be one of the bands {bands}, not {current!r}")
    lit = None if lit is None else set(lit)
    if lit is not None and (current is None or lit - set(stages())):
        raise ValueError(f"lit needs current and lifecycle stages {list(stages())}, not {sorted(lit)!r}")
    cells = _table_cells(cells, bands)

    colours, page, ink, quiet, hue = _palette(theme)
    order = list(stages())
    placed, gaps, mids = _layout([{"stage": s} for s in order])
    spans = {c["stage"]: (c["a0"], c["a1"]) for c in placed}
    radii = band_radii(bands)
    inner = radii[bands[0]][0]
    r_mid = (inner + BAND_OUT) / 2

    own = (
        f".title{{fill:{ink};font-size:{TITLE_PX}px;font-weight:600;letter-spacing:.03em}}.level{{font-size:{LEVEL_PX}px;font-weight:600;letter-spacing:.02em}}"
        f".key{{fill:{ink};font-size:{KEY_PX}px;font-weight:600}}"
    )
    parts = _frame(f".bcell{{stroke:{page};stroke-width:1.5}}", own, colours, quiet, briefs, title, BAND_BRAIN_H, BAND_LOGO_W)
    patterns: dict[str, str] = {}
    parts += _band_cells(cells, spans, radii, current, hue, page, patterns, lit=lit)
    parts += _status_key(cells, quiet, patterns)
    if patterns:
        parts.append(f"<defs>{''.join(patterns.values())}</defs>")
    parts += [_head(_pt(g + math.degrees(0.4 * HEAD / r_mid), r_mid), _tangent(g)) for g in gaps]
    parts.append(_cycle(OPEN_DEG / 2 - 4, BAND_OUT + 16))
    parts.append(_revise(placed, inner, inset=18.0, lead=6.0))
    parts += _band_names(radii, current, ink, page)
    labels, xs = _labels(placed, mids, "numbers", icons, hue, briefs)
    parts += labels

    left, right = _bounds(xs)
    levels = ", ".join(_level_label(b) for b in bands)
    aria = f"{title}: stages {', '.join(stages()[s]['name'] for s in order)}; levels {levels} from the centre outward"
    return _document(parts, (left, 0.0, right - left, SIZE + KEY_H), aria, page if ground else None)
