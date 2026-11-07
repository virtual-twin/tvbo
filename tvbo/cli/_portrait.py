"""The ASCII spec-to-cortex portrait a bare ``tvbo`` invocation prints.

A model spec on the left comes apart and reassembles on the right as a cortical surface drawn out of the spec's own characters — the terminal form of bsplot's ``AsciiSpecPortrait`` showcase.

The cortex *geometry* is shipped precomputed: :data:`ASSET` is a luminance grid (binarised curvature, shaded by the lighting) rendered once by :func:`build_asset` from ``bsplot.render_surf_ascii``. Printing the portrait therefore costs one file read and no scientific imports — the spec supplies only the glyphs, and the grid is box-resampled to whatever the terminal is wide enough for.

Each cell's glyph is picked from an ink ramp built out of the spec's own character set, so the 3-D form reads through glyph density (as it does in ``plot_surf_ascii``) as well as through the shaded colour, and every landed glyph can fly in from a real occurrence of that same character in the text.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import textwrap
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

LEVELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
EMPTY = "."

ASSET = Path(files("tvbo")) / "data" / "media" / "cortex_ascii.txt"
LOGO_ASSET = Path(files("tvbo")) / "data" / "media" / "logo_ascii.txt"
LOGO_SVG = Path(files("tvbo")) / "data" / "tvb_logo.svg"
DEFAULT_SPEC = Path(files("tvbo")) / "database" / "models" / "Jansen1995.yaml"

# Fallback character order, light -> heavy by ink coverage; mirrors the "ascii" entry of ``bsplot.GLYPH_RAMPS``. The shipped asset carries a measured order covering all of printable ASCII, so no spec character is left off the ramp.
INK_ORDER = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# ``ink`` draws the intact spec; a cell's tone (0 shadowed sulcus .. 1 lit crest) maps between ``lo`` and ``hi`` grey. On paper that runs the other way — a lit crown is pale and it is the sulci that carry the ink — which is also why the glyph ramp flips with the background (see :func:`spec_ramp`).
THEMES = {
    "dark": {"ink": (112, 120, 134), "lo": 38, "hi": 252, "accent": (115, 172, 205), "page": (13, 17, 23), "lift": 0.22},
    "light": {"ink": (112, 109, 104), "lo": 26, "hi": 202, "accent": (45, 92, 122), "page": (255, 255, 255), "lift": 0.0},
}

# Coverage ramp for the logo when the output cannot carry ``▀``/``▄``.
SHADES_ASCII = " .:*#"
# The wordmark, drawn in the brand blue. The real artwork is a raster mark; at a dozen character cells it turns to mush, so the banner spells the name out instead — ``hero(mark="logo")`` draws the artwork for anyone re-testing it.
LOGO = (
    "█████ █   █ ████   ███ ",
    "  █   █   █ █   █ █   █",
    "  █   █   █ ████  █   █",
    "  █    █ █  █   █ █   █",
    "  █     █   ████   ███ ",
)
TAGLINE = "The Virtual Brain Ontology"
TAGLINE_SHORT = "Virtual Brain Ontology"

# Letters ordered light -> heavy: bsplot's "letters" ramp without its one non-ASCII glyph. Hand-tuned, so a small cortex reads far cleaner out of it than out of whatever characters a given spec happens to contain.
GLYPH_RAMP = " .:-ilcvzsoaeutnxwmqpdbkhAOQHXNBM@"

# The banner's cortex is small, so it is drawn from the coarser shading ramp (bsplot's "blocks"): at a dozen rows the tonal contrast is what makes it read as a brain, where letterforms only add noise.
SURFACE_RAMP = " .:-=+*#%@"

GAMMA = 1.25
SULCUS = 0.58
SIZE = {"min_width": 34, "max_width": 104, "spec_width": 34, "gutter": 6, "char_aspect": 2.0}
HERO = {"max_rows": 13, "min_rows": 6, "gutter": 3}
FLOW_FRACTION = 0.55
COVERAGE = 0.34


def _color_mode(explicit: str | None = None, stream=None) -> str:
    """Pick ``truecolor`` / ``256`` / ``none``, honouring NO_COLOR and pipes."""
    if explicit:
        return explicit
    if os.environ.get("NO_COLOR") is not None:
        return "none"
    stream = sys.stdout if stream is None else stream
    if not hasattr(stream, "isatty") or not stream.isatty():
        return "none"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    return "256"


def _rgb_to_256(r: int, g: int, b: int) -> int:
    """Quantise 0..255 RGB to an xterm-256 index (greys use the 24-step ramp)."""
    if abs(r - g) < 8 and abs(g - b) < 8:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + (r - 8) * 24 // 247
    return 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)


def _esc(rgb: tuple[int, int, int], mode: str, *, bg: bool = False) -> str:
    if mode == "none":
        return ""
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    layer = 48 if bg else 38
    if mode == "truecolor":
        return f"\x1b[{layer};2;{r};{g};{b}m"
    return f"\x1b[{layer};5;{_rgb_to_256(r, g, b)}m"


def dim(text: str, *, theme: str = "dark", color_mode: str | None = None) -> str:
    """Wrap *text* in the portrait's muted spec-ink colour (bare text if uncoloured)."""
    mode = _color_mode(color_mode)
    escape = _esc(THEMES[theme]["ink"], mode)
    return f"{escape}{text}\x1b[0m" if escape else text


def _grey(theme: dict, tone: float) -> tuple[float, float, float]:
    v = theme["lo"] + (theme["hi"] - theme["lo"]) * max(0.0, min(1.0, tone))
    return (v, v, v)


def _mix(c0, c1, t: float) -> tuple[float, float, float]:
    return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(3))


def _hash01(n: int) -> float:
    """A deterministic 0..1 draw, so the still frame is identical every run."""
    n &= 0xFFFFFFFF
    n = ((n ^ 61) ^ (n >> 16)) & 0xFFFFFFFF
    n = (n + (n << 3)) & 0xFFFFFFFF
    n ^= n >> 4
    n = (n * 0x27D4EB2D) & 0xFFFFFFFF
    n ^= n >> 15
    return (n & 0xFFFFFFFF) / 4294967296.0


def _ease(t: float) -> float:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return t * t * (3 - 2 * t)


class Canvas:
    """A character grid whose cells carry a glyph, a colour, and maybe a backdrop.

    The backdrop is what lets a half-block glyph hold two pixels: the glyph's colour paints the top half, the backdrop the bottom.
    """

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.cells: list[list[tuple[str, tuple, tuple | None] | None]] = [[None] * width for _ in range(height)]

    def put(self, x: int, y: int, ch: str, rgb, *, bg=None, over: bool = True) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height) or (ch in ("", " ") and bg is None):
            return
        if over or self.cells[y][x] is None:
            self.cells[y][x] = (ch, rgb, bg)

    def free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.cells[y][x] is None

    def render(self, mode: str) -> str:
        """Compose the grid into lines, emitting one escape per colour run."""
        reset = "" if mode == "none" else "\x1b[0m"
        lines = []
        for row in self.cells:
            out, fg, bg = [], None, None
            for cell in row:
                if cell is None:
                    if bg is not None:
                        out.append(reset)
                        fg = bg = None
                    out.append(" ")
                    continue
                ch, cell_fg, cell_bg = cell
                key = tuple(int(round(v)) for v in cell_fg)
                back = None if cell_bg is None else tuple(int(round(v)) for v in cell_bg)
                if back != bg:
                    out.append(reset if back is None else _esc(back, mode, bg=True))
                    fg = None if back is None else fg
                    bg = back
                if key != fg:
                    out.append(_esc(key, mode))
                    fg = key
                out.append(ch)
            line = "".join(out).rstrip()
            lines.append(line + reset if fg is not None or bg is not None else line)
        return "\n".join(lines)


class Logo(NamedTuple):
    """The TVB-O logo as a pixel grid — two rows of pixels per character cell."""

    alpha: list[list[float]]
    rgb: list[list[tuple[int, int, int]]]

    @property
    def width(self) -> int:
        return len(self.alpha[0])

    @property
    def height(self) -> int:
        return len(self.alpha)


def load_logo(path: Path | str = LOGO_ASSET) -> Logo:
    """Read the shipped logo: an alpha block, a palette-index block, one palette."""
    palette: list[tuple[int, int, int]] = []
    blocks: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if line.startswith("# palette:"):
                palette = [(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) for h in line.split(":", 1)[1].split()]
            elif line.startswith("# channel:"):
                blocks.append([])
            continue
        if blocks:
            blocks[-1].append(line)
    alpha = [[0.0 if ch == EMPTY else LEVELS.index(ch) / (len(LEVELS) - 1) for ch in row] for row in blocks[0]]
    rgb = [[palette[0] if ch == EMPTY else palette[int(ch, 36)] for ch in row] for row in blocks[1]]
    return Logo(alpha, rgb)


def resample_logo(logo: Logo, cols: int, rows: int | None = None) -> Logo:
    """Box-average the logo to *cols* pixels wide, keeping its aspect by default.

    The pixel height stays even, so every character cell keeps a top and a bottom half to paint.
    """
    rows = rows if rows is not None else max(2, 2 * round(logo.height * cols / logo.width / 2))
    if cols == logo.width and rows == logo.height:
        return logo
    alpha, rgb = [], []
    for y in range(rows):
        js = range(y * logo.height // rows, max(y * logo.height // rows + 1, (y + 1) * logo.height // rows))
        a_row, c_row = [], []
        for x in range(cols):
            iss = range(x * logo.width // cols, max(x * logo.width // cols + 1, (x + 1) * logo.width // cols))
            cells = [(logo.alpha[j][i], logo.rgb[j][i]) for j in js for i in iss]
            lit = [c for a, c in cells if a > 0.05]
            a_row.append(sum(a for a, _ in cells) / len(cells))
            c_row.append(tuple(sum(c[k] for c in lit) // len(lit) for k in range(3)) if lit else (0, 0, 0))
        alpha.append(a_row)
        rgb.append(c_row)
    return Logo(alpha, rgb)


def _over(theme: dict, rgb: tuple[int, int, int], alpha: float) -> tuple[float, float, float]:
    """Composite a logo pixel over the page it is printed on.

    ``lift`` brightens the mark toward white first: the logo's teal is drawn for paper and sinks into a dark terminal untouched.
    """
    page, lift = theme["page"], theme["lift"]
    ink = [v + (255 - v) * lift for v in rgb]
    return tuple(page[i] + (ink[i] - page[i]) * alpha for i in range(3))


def _draw_logo(canvas: Canvas, logo: Logo, theme: dict, *, top: int = 0, blocks: bool = True) -> None:
    """Paint the logo, two pixel rows per character cell.

    ``▀`` carries the upper pixel in the glyph colour and the lower one as its backdrop; a cell with only one lit half uses ``▀``/``▄`` and no backdrop.
    Where the output cannot carry those glyphs, one averaged pixel per cell is drawn with an ASCII shade instead.
    """
    for y in range(0, logo.height - 1, 2):
        for x in range(logo.width):
            up, down = logo.alpha[y][x], logo.alpha[y + 1][x]
            if max(up, down) <= 0.04:
                continue
            row = top + y // 2
            if not blocks:
                mean = (up + down) / 2
                canvas.put(
                    x,
                    row,
                    SHADES_ASCII[min(len(SHADES_ASCII) - 1, int(round(mean * 4)))],
                    _over(theme, logo.rgb[y][x], min(1.0, mean * 1.4)),
                )
            elif up <= 0.04:
                canvas.put(x, row, "▄", _over(theme, logo.rgb[y + 1][x], down))
            elif down <= 0.04:
                canvas.put(x, row, "▀", _over(theme, logo.rgb[y][x], up))
            else:
                canvas.put(x, row, "▀", _over(theme, logo.rgb[y][x], up), bg=_over(theme, logo.rgb[y + 1][x], down))


def _blocks_printable() -> bool:
    """Whether stdout can carry the half-block glyphs the logo is drawn with."""
    try:
        "▀▄".encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class Cortex(NamedTuple):
    """The shipped surface: per-cell lighting and curvature (None = off-surface).

    ``light`` (0 shadow .. 1 lit) carries the 3-D form and picks each cell's glyph off the ink ramp; ``curv`` (0 sulcus .. 1 gyrus) tints it, the way curvature colours a cortical surface plot.
    """

    light: list[list[float | None]]
    curv: list[list[float | None]]
    ink_order: str = INK_ORDER


def _read_channel(lines: list[str]) -> list[list[float | None]]:
    rows = [[None if ch == EMPTY else LEVELS.index(ch) / (len(LEVELS) - 1) for ch in line] for line in lines]
    width = max(len(r) for r in rows)
    for r in rows:
        r.extend([None] * (width - len(r)))
    return rows


def load_cortex(path: Path | str = ASSET) -> Cortex:
    """Read the shipped grid; the two channels are stored one after the other."""
    blocks: list[list[str]] = []
    ink_order = INK_ORDER
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if line.startswith("# channel:"):
                blocks.append([])
            elif line.startswith("# ink:"):
                ink_order = " " + line.split(":", 1)[1].strip()
            continue
        if line.strip() and blocks:
            blocks[-1].append(line)
    if not blocks or not blocks[0]:
        raise ValueError(f"{path} holds no cortex rows")
    light = _read_channel(blocks[0])
    curv = _read_channel(blocks[1]) if len(blocks) > 1 and blocks[1] else light
    return Cortex(light, curv, ink_order)


def resample(rows: list[list[float | None]], width: int, height: int) -> list[list[float | None]]:
    """Box-resample the grid to *width* x *height*, keeping a clean silhouette.

    A target cell survives only when :data:`COVERAGE` of its source box is on the surface, so downsampling thins the rim instead of fattening it.
    """
    src_h, src_w = len(rows), len(rows[0])
    out = []
    for y in range(height):
        y0, y1 = y * src_h / height, (y + 1) * src_h / height
        js = range(int(y0), min(src_h, max(int(y0) + 1, math.ceil(y1))))
        line = []
        for x in range(width):
            x0, x1 = x * src_w / width, (x + 1) * src_w / width
            iss = range(int(x0), min(src_w, max(int(x0) + 1, math.ceil(x1))))
            vals = [rows[j][i] for j in js for i in iss]
            covered = [v for v in vals if v is not None]
            line.append(sum(covered) / len(covered) if vals and len(covered) / len(vals) >= COVERAGE else None)
        out.append(line)
    return out


def spec_ramp(text: str, ink_order: str = INK_ORDER, *, dark: bool = True) -> str:
    """The spec's own characters, ordered lightest-ink to heaviest.

    The cortex is built from exactly these glyphs, so its shading is legible with no colour at all. On a light background the order is flipped, so the shadowed faces are the ones carrying the heaviest ink.
    """
    present = set(text)
    ramp = "".join(ch for ch in ink_order if ch in present)
    if len(ramp) < 8:
        ramp = ink_order[1:]
    return ramp if dark else ramp[::-1]


def spec_lines(text: str, width: int, rows: int) -> list[str]:
    """Lay the spec out as a monospace block of at most *rows* lines."""
    wrapped: list[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="  ") or [""])
        if len(wrapped) >= rows:
            break
    return wrapped[:rows]


def _layout(cols: int, rows: int, ratio: float) -> tuple[int, int, bool]:
    """Fit ``(brain_width, brain_height, show_spec)`` into a *cols* x *rows* box.

    *ratio* is the cortex's own width:height in character cells, so it is never
    stretched: whichever of the two dimensions runs out first sets its size.
    The spec column is dropped when there is no room for both sides.
    """
    show_spec = cols >= SIZE["spec_width"] + SIZE["gutter"] + SIZE["min_width"] + 6
    budget = cols - (SIZE["spec_width"] + SIZE["gutter"] if show_spec else 2)
    brain_w = max(SIZE["min_width"], min(SIZE["max_width"], budget))
    brain_h = max(6, int(round(brain_w / ratio)))
    if brain_h > rows:
        brain_h = max(6, rows)
        brain_w = min(brain_w, int(round(brain_h * ratio)))
    return brain_w, brain_h, show_spec


def compose(
    spec_text: str,
    cortex: list[list[float | None]],
    *,
    theme: str = "dark",
    width: int | None = None,
    height: int | None = None,
    flow: bool = True,
    phase: float | None = None,
) -> Canvas:
    """Draw the portrait: spec (left), letters in flight, cortex (right).

    With *phase* given the letters are mid-dissolve — every glyph rests on its spec position until its departure beat, then flies to its cortex cell — so stepping *phase* from 0 to 1 animates the spec becoming the brain.
    """
    th = THEMES[theme]
    term = shutil.get_terminal_size((100, 30))
    ratio = len(cortex.light[0]) / len(cortex.light)
    brain_w, brain_h, show_spec = _layout(width or term.columns - 2, height or max(10, term.lines - 8), ratio)
    sw, gutter = (SIZE["spec_width"], SIZE["gutter"]) if show_spec else (0, 0)
    canvas = Canvas(sw + gutter + brain_w, brain_h)

    lines = spec_lines(spec_text, sw, brain_h) if show_spec else []
    top = max(0, (brain_h - len(lines)) // 2)
    sources: dict[str, list[tuple[int, int]]] = {}
    for row, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch != " ":
                sources.setdefault(ch, []).append((x, top + row))

    ramp = spec_ramp("".join(sources) or spec_text, cortex.ink_order, dark=theme == "dark")
    landed = _landed(cortex, ramp, sources, brain_w, brain_h, sw + gutter, theme)

    if phase is None:
        for ch, positions in sources.items():
            for x, y in positions:
                canvas.put(x, y, ch, th["ink"])
        for ch, x, y, tone, _, _, _ in landed:
            canvas.put(x, y, ch, _grey(th, tone))
        if flow and sources:
            _draw_flow(canvas, th, landed, sw + 1, sw + gutter + 2)
        return canvas

    def beat(sx: int, sy: int, k: int) -> float:
        """When a glyph lifts off — departures sweep the spec in reading order."""
        return _hash01(k * 977) * 0.05 + (sy / max(brain_h, 1)) * 0.35 + (sx / max(sw, 1)) * 0.18

    for ch, x, y, tone, sx, sy, k in landed:
        f = _ease((phase - beat(sx, sy, k)) / 0.22)
        if f <= 0.0:
            canvas.put(sx, sy, ch, th["ink"], over=False)
        else:
            fx, fy = _arc(sx, sy, x, y, f)
            canvas.put(fx, fy, ch, _mix(th["ink"], _grey(th, tone), f))

    for i, (ch, sx, sy) in enumerate(_unclaimed(sources, landed)):
        f = _ease((phase - beat(sx, sy, i)) / 0.22)
        if f <= 0.0:
            canvas.put(sx, sy, ch, th["ink"], over=False)
        elif f < 1.0 and landed:
            target = landed[(i * 7919) % len(landed)]
            fx, fy = _arc(sx, sy, target[1], target[2], f)
            canvas.put(fx, fy, ch, _mix(th["ink"], _grey(th, target[3]), f), over=False)
    return canvas


def _unclaimed(sources: dict, landed: list) -> list[tuple[str, int, int]]:
    """Spec glyphs no cortex cell asked for — they lift off and merge in anyway.

    The cortex needs more glyphs than the spec has, so most letters multiply;
    a few characters sit at a lighting the surface never reaches. Flying them into the stream is what lets the spec come apart completely.
    """
    claimed = {(sx, sy) for *_, sx, sy, _ in landed}
    return [(ch, x, y) for ch, spots in sources.items() for x, y in spots if (x, y) not in claimed]


def _landed(cortex: Cortex, ramp: str, sources, brain_w, brain_h, x_off, theme):
    """One entry per on-surface cell: ``(char, x, y, tone, sx, sy, index)``.

    The glyph comes from *ramp* at that cell's lighting, so the 3-D form reads with no colour at all; *tone* folds the curvature tint into that lighting for the colour. ``sx, sy`` is then where a *matching* character sits in the spec block, so a 'C' lands from a 'C' in the text.
    """
    light = _restretch(resample(cortex.light, brain_w, brain_h))
    curv = resample(cortex.curv, brain_w, brain_h)
    fallback = [p for ps in sources.values() for p in ps]
    used: dict[str, int] = {}
    out = []
    k = 0
    for y in range(brain_h):
        for x in range(brain_w):
            lum = light[y][x]
            if lum is None:
                continue
            gyrus = curv[y][x]
            tone = lum**GAMMA * (SULCUS + (1.0 - SULCUS) * (0.5 if gyrus is None else gyrus))
            ch = ramp[min(len(ramp) - 1, int(round(lum * (len(ramp) - 1))))]
            if ch == " ":
                continue
            spots = sources.get(ch) or sources.get(ch.swapcase()) or fallback
            if spots:
                n = used.get(ch, 0)
                used[ch] = n + 1
                sx, sy = spots[n % len(spots)]
            else:
                sx, sy = x, y
            out.append((ch, x_off + x, y, tone, sx, sy, k))
            k += 1
    return out


def _restretch(cells: list[list[float | None]]) -> list[list[float | None]]:
    """Rescale the lighting back to a full 0..1 range after resampling.

    Averaging over a source box pulls every cell toward the mean, so a small cortex would otherwise come out flat and uniformly heavy.
    """
    vals = [v for row in cells for v in row if v is not None]
    if not vals:
        return cells
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [[None if v is None else (v - lo) / span for v in row] for row in cells]


def _arc(sx: int, sy: int, tx: int, ty: int, f: float) -> tuple[int, int]:
    """A glyph's cell at flight fraction *f*, bowed upward off the direct line.

    Character cells are twice as tall as wide, so the curve is computed in square units and mapped back.
    """
    sy2, ty2 = sy * SIZE["char_aspect"], ty * SIZE["char_aspect"]
    dx, dy = tx - sx, ty2 - sy2
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    if ny > 0:
        nx, ny = -nx, -ny
    cx, cy = (sx + tx) / 2 + nx * 0.16 * length, (sy2 + ty2) / 2 + ny * 0.16 * length
    u = 1 - f
    x = u * u * sx + 2 * u * f * cx + f * f * tx
    y = u * u * sy2 + 2 * u * f * cy + f * f * ty2
    return int(round(x)), int(round(y / SIZE["char_aspect"]))


def _draw_flow(canvas: Canvas, th: dict, landed: list, x_min: int, x_max: int) -> None:
    """Scatter a sample of the stream mid-flight, in the gutter between the two.

    Confining the sample to that band keeps the spec block readable and leaves the cortex silhouette standing clear of its own letters.
    """
    for ch, x, y, tone, sx, sy, k in landed:
        if _hash01(k * 2654435761) >= FLOW_FRACTION:
            continue
        f = _ease(0.24 + 0.54 * _hash01(k * 40503 + 7))
        fx, fy = _arc(sx, sy, x, y, f)
        if x_min <= fx <= x_max and canvas.free(fx, fy):
            canvas.put(fx, fy, ch, _mix(th["ink"], _grey(th, tone), f))


def hero(
    *,
    subtitle: str = "",
    mark: str = "wordmark",
    theme: str = "dark",
    color_mode: str | None = None,
    width: int | None = None,
    cortex_path: Path | str = ASSET,
    logo_path: Path | str = LOGO_ASSET,
) -> str:
    """The bare-``tvbo`` banner: the name on the left, a cortex on the right.

    The two columns share a height and the pair spans the full width — the mark at the left edge, the cortex at the right. The cortex is the shipped surface shaded through :data:`SURFACE_RAMP`, which reads as a brain at this size where a spec's own characters would not; ``tvbo brain`` is the full-size, spec-drawn portrait. Pass ``mark="logo"`` to draw the raster artwork instead of the wordmark.
    """
    th = THEMES[theme]
    mode = _color_mode(color_mode)
    cortex = load_cortex(cortex_path)
    cols = width or shutil.get_terminal_size((96, 30)).columns - 4
    ratio = len(cortex.light[0]) / len(cortex.light)

    if mark == "logo":
        logo = load_logo(logo_path)
        left_w, block = round(2 * HERO["min_rows"] * logo.width / logo.height), []
    else:
        logo = None
        block = list(LOGO) + ["", TAGLINE] + ([subtitle] if subtitle else [])
        left_w = max(len(line) for line in block)

    rows, brain_w = _hero_layout(cols, left_w, len(block), ratio)
    canvas = Canvas(cols, rows)

    top = max(0, (rows - (len(block) or rows)) // 2)
    if logo is not None:
        _draw_logo(canvas, resample_logo(logo, min(left_w, cols), 2 * rows), th, blocks=mode != "none" and _blocks_printable())
    for row, line in enumerate(block):
        colour = th["accent"] if row < len(LOGO) else th["ink"]
        for x, ch in enumerate(line[:cols]):
            canvas.put(x, top + row, ch, colour)

    if brain_w:
        x_off = cols - brain_w
        ramp = SURFACE_RAMP if theme == "dark" else SURFACE_RAMP[::-1]
        for ch, x, y, tone, _, _, _ in _landed(cortex, ramp, {}, brain_w, rows, x_off, theme):
            canvas.put(x, y, ch, _grey(th, tone))
    return canvas.render(mode)


def _hero_layout(cols: int, left_w: int, left_rows: int, cortex_ratio: float) -> tuple[int, int]:
    """Pick ``(rows, cortex_width)`` — the tallest banner that fits both columns.

    The cortex keeps its own proportions, so the shared row count sets its width; the left column is centred in whatever band that gives. Below the width where both fit, the cortex gives way and the name stands alone.
    """
    floor = max(HERO["min_rows"], left_rows)
    for rows in range(HERO["max_rows"], floor - 1, -1):
        brain_w = round(rows * cortex_ratio)
        if left_w + HERO["gutter"] + brain_w <= cols:
            return rows, brain_w
    return floor, 0


def render(
    spec_text: str | None = None,
    *,
    theme: str = "dark",
    color_mode: str | None = None,
    width: int | None = None,
    height: int | None = None,
    flow: bool = True,
    cortex_path: Path | str = ASSET,
) -> str:
    """The portrait as one printable string."""
    text = spec_text if spec_text is not None else Path(DEFAULT_SPEC).read_text(encoding="utf-8")
    canvas = compose(text, load_cortex(cortex_path), theme=theme, width=width, height=height, flow=flow)
    return canvas.render(_color_mode(color_mode))


def frames(
    spec_text: str | None = None,
    *,
    n: int = 88,
    hold: int = 10,
    theme: str = "dark",
    color_mode: str | None = None,
    width: int | None = None,
    height: int | None = None,
    cortex_path: Path | str = ASSET,
):
    """Yield the dissolve as rendered frames: spec -> flight -> cortex."""
    text = spec_text if spec_text is not None else Path(DEFAULT_SPEC).read_text(encoding="utf-8")
    cortex = load_cortex(cortex_path)
    mode = _color_mode(color_mode)
    for i in range(n):
        phase = min(1.0, i / max(1, n - 1 - hold))
        yield compose(text, cortex, theme=theme, width=width, height=height, phase=phase).render(mode)


def play(stream=None, *, fps: float = 26.0, **kwargs) -> None:
    """Play the dissolve in place on a terminal, leaving the still frame up."""
    import time

    out = stream or sys.stdout
    first, height = True, 0
    out.write("\x1b[?25l")
    try:
        for frame in frames(**kwargs):
            if not first:
                out.write(f"\x1b[{height}A")
            lines = frame.split("\n")
            height = len(lines)
            out.write("".join(f"\x1b[2K{line}\n" for line in lines))
            out.flush()
            first = False
            time.sleep(1.0 / fps)
    finally:
        out.write("\x1b[?25h")
        out.flush()


def build_asset(
    *,
    surface: str = "fsaverage",
    hemi: str = "lh",
    view: str = "lateral",
    width: int = 160,
    density: str = "164k",
    shade: float = 0.85,
    out: Path | str | None = None,
) -> Cortex:
    """Re-render the cortex geometry with bsplot and return (or write) the grid.

    Two ``render_surf_ascii`` calls do all the anatomy: one shades the plain surface — that lighting is the 3-D form — and one paints the binarised curvature unshaded, giving the gyrus/sulcus tint. Both need bsplot and the template data, which is why the result is shipped rather than recomputed.
    """
    from bsplot import render_surf_ascii

    common = dict(surface=surface, hemi=hemi, view=view, width=width, glyphs="X", surface_density=density, return_grid=True)
    lit = render_surf_ascii(**common, shade_intensity=shade)
    curv = _curvature(hemi, density)
    tinted = render_surf_ascii(**common, data=(curv > 0).astype(float), cmap="Greys", shade_intensity=0.0)

    cortex = _crop(Cortex(_normalised(lit), _normalised(tinted), _measure_ink_order()))
    if out is not None:
        _write_asset(cortex, Path(out), surface=surface, hemi=hemi, view=view, density=density, shade=shade)
    return cortex


def _crop(cortex: Cortex) -> Cortex:
    """Trim the empty margins, so the shipped grid *is* the cortex bounding box."""
    ys = [y for y, row in enumerate(cortex.light) if any(v is not None for v in row)]
    xs = [x for x in range(len(cortex.light[0])) if any(row[x] is not None for row in cortex.light)]
    box = slice(min(ys), max(ys) + 1), slice(min(xs), max(xs) + 1)
    return cortex._replace(
        light=[row[box[1]] for row in cortex.light[box[0]]],
        curv=[row[box[1]] for row in cortex.curv[box[0]]],
    )


def _measure_ink_order() -> str:
    """Printable ASCII ordered by how much ink each glyph puts on a cell.

    Measured once, at asset-build time, by rasterising the characters in a monospace face — so every character a spec can contain has a place on the ramp, instead of only those in the hand-written :data:`INK_ORDER`.
    """
    import matplotlib.font_manager as fm
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(fm.findfont("DejaVu Sans Mono"), 32)
    chars = [chr(c) for c in range(33, 127)]
    weight = {}
    for ch in chars:
        img = Image.new("L", (40, 48), 0)
        ImageDraw.Draw(img).text((4, 4), ch, fill=255, font=font)
        weight[ch] = sum(img.getdata()) / (255 * 40 * 48)
    return " " + "".join(sorted(chars, key=lambda c: weight[c]))


def _normalised(grid) -> list[list[float | None]]:
    """A render's per-cell luminance, rescaled to 0..1 over the covered cells."""
    covered = grid.chars != ""
    lum = grid.rgb[..., 0] * 0.2126 + grid.rgb[..., 1] * 0.7152 + grid.rgb[..., 2] * 0.0722
    lo, hi = float(lum[covered].min()), float(lum[covered].max())
    span = (hi - lo) or 1.0
    return [
        [float((lum[r, c] - lo) / span) if covered[r, c] else None for c in range(covered.shape[1])]
        for r in range(covered.shape[0])
    ]


def build_logo_asset(*, source: Path | str | None = None, cols: int = 28, out: Path | str | None = None) -> Logo:
    """Rasterise the TVB-O logo into a pixel grid and return (or write) it.

    Two pixel rows per character cell, which is what the half-block glyphs the banner draws with can carry. The mark is supersampled: each target pixel averages its source block for coverage but takes the *dominant* brand colour rather than a blend, so edges stay crisp instead of smearing into intermediate tones. The artwork lives with the docs rather than in the package, so the *asset* is what ships.
    """
    from PIL import Image

    src = Path(source) if source else _logo_source()
    im = Image.open(_rasterise_svg(src, width=cols * 32) if src.suffix.lower() == ".svg" else src).convert("RGBA")
    im = im.crop(im.split()[-1].getbbox())

    rows = max(2, 2 * round(cols * im.height / im.width / 2))
    ss = 8
    big = im.resize((cols * ss, rows * ss), Image.LANCZOS)
    palette = _brand_palette(big)

    alpha, rgb = [], []
    for y in range(rows):
        a_row, c_row = [], []
        for x in range(cols):
            block = [big.getpixel((x * ss + i, y * ss + j)) for j in range(ss) for i in range(ss)]
            a_row.append(sum(px[3] for px in block) / (255 * len(block)))
            weight: dict[tuple[int, int, int], float] = {}
            for r, g, b, a in block:
                if a > 40:
                    key = min(palette, key=lambda c: (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2)
                    weight[key] = weight.get(key, 0.0) + a
            c_row.append(max(weight, key=weight.get) if weight else palette[0])
        alpha.append(a_row)
        rgb.append(c_row)

    logo = Logo(alpha, rgb)
    if out is not None:
        _write_logo_asset(logo, Path(out), source=src.name, cols=cols)
    return logo


def _brand_palette(im, top: int = 4) -> list[tuple[int, int, int]]:
    """The mark's own flat colours — the few that most of its opaque area uses."""
    from collections import Counter

    counts = Counter(px[:3] for px in im.convert("RGBA").getdata() if px[3] > 220)
    palette = [colour for colour, _ in counts.most_common(top * 12)]
    kept: list[tuple[int, int, int]] = []
    for colour in palette:
        if all(sum((colour[i] - k[i]) ** 2 for i in range(3)) > 900 for k in kept):
            kept.append(colour)
        if len(kept) == top:
            break
    return kept or [(255, 255, 255)]


def _logo_source() -> Path:
    """The TVB-O artwork, preferring the vector version this checkout keeps."""
    root = Path(files("tvbo")).parent
    for candidate in ("docs/_static/tvbo_python_square.svg", "docs/_static/tvbo_logo.png", "imgs/tvbo_logo.png"):
        if (root / candidate).exists():
            return root / candidate
    raise FileNotFoundError("no TVB-O logo artwork found — pass source=<path to the logo svg or png>")


def _rasterise_svg(svg: Path, *, width: int = 1600) -> Path:
    """Render *svg* to a temporary PNG with whatever rasteriser is installed."""
    import subprocess
    import tempfile

    png = Path(tempfile.mkdtemp()) / "logo.png"
    for cmd in (
        ["rsvg-convert", "-w", str(width), str(svg), "-o", str(png)],
        ["inkscape", str(svg), "--export-type=png", f"--export-width={width}", f"--export-filename={png}"],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return png
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("no SVG rasteriser found — install librsvg (rsvg-convert) or inkscape")


def _write_logo_asset(logo: Logo, path: Path, **provenance) -> Path:
    """Write the logo as an alpha block plus a palette-indexed colour block."""
    palette: list[tuple[int, int, int]] = [(0, 0, 0)]
    n = len(LEVELS) - 1
    alpha_rows, colour_rows = [], []
    for y in range(logo.height):
        alpha, colour = [], []
        for x in range(logo.width):
            level = logo.alpha[y][x]
            if level <= 0.01:
                alpha.append(EMPTY)
                colour.append(EMPTY)
                continue
            rgb = logo.rgb[y][x]
            if rgb not in palette:
                palette.append(rgb)
            alpha.append(LEVELS[max(0, min(n, int(round(level * n))))])
            colour.append(_BASE36[palette.index(rgb)])
        alpha_rows.append("".join(alpha))
        colour_rows.append("".join(colour))

    meta = " ".join(f"{k}={v}" for k, v in provenance.items())
    text = (
        "# tvbo ascii logo — the TVB-O mark as a pixel grid, two rows per character cell.\n"
        f"# {meta} width={logo.width} height={logo.height}\n"
        f"# '{EMPTY}' = transparent.\n"
        "# Regenerate with: tvbo brain --save-logo <path>\n"
        "# palette: " + " ".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in palette) + "\n"
        "# channel: alpha — coverage, 0 transparent .. 1 solid.\n" + "\n".join(alpha_rows) + "\n"
        "# channel: colour — index into the palette above (base 36).\n" + "\n".join(colour_rows) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _curvature(hemi: str, density: str):
    """Per-vertex curvature for the pial mesh, for the gyral colouring."""
    import nibabel as nib
    import numpy as np
    import templateflow.api as tf

    sides = {"lh": ["L"], "rh": ["R"], "both": ["L", "R"]}[hemi]
    parts = []
    for side in sides:
        path = tf.get(template="fsaverage", hemi=side, density=density, suffix="curv", extension=".shape.gii")
        parts.append(np.asarray(nib.load(str(path)).agg_data(), dtype=float))
    return np.concatenate(parts)


def _write_asset(cortex: Cortex, path: Path, **provenance) -> Path:
    """Write both channels as the quantised text asset the CLI reads."""
    n = len(LEVELS) - 1

    def block(rows):
        return "\n".join(
            "".join(EMPTY if v is None else LEVELS[max(0, min(n, int(round(v * n))))] for v in row) for row in rows
        )

    meta = " ".join(f"{k}={v}" for k, v in provenance.items())
    text = (
        "# tvbo ascii cortex — per-cell channels of a bsplot.render_surf_ascii surface.\n"
        f"# {meta} width={len(cortex.light[0])} height={len(cortex.light)}\n"
        f"# '{EMPTY}' = off-surface; else the value 0..1 quantised over '{LEVELS[0]}'..'{LEVELS[-1]}'.\n"
        "# Regenerate with: tvbo brain --render --save-asset <path>\n"
        f"# ink: {cortex.ink_order.strip()}\n"
        "# channel: light — shading intensity, 0 shadow .. 1 lit (picks the glyph).\n"
        f"{block(cortex.light)}\n"
        "# channel: curv — binarised curvature, 0 sulcus .. 1 gyrus (tints it).\n"
        f"{block(cortex.curv)}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
