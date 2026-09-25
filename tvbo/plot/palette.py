# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Named colour roles for figures, so a project's colours live in one file.

A style sheet carries only what matplotlib has an rcParam for. That covers the cycler and the ink, and stops short of the roles a figure actually reasons in: the one colour that means *this is the point*, the neutral that everything unlabelled is drawn in, the hairline grey behind it. Those live here, in the ``theme`` a figure declares (``theme: {iri: tvbo:theme/default}``, or a document of its own), so changing a project's colours is one edit and every panel follows. A colour is never a style layer: ``Figure.style`` carries the looks TVB-O does not own, and the theme is applied over all of them.

The cycler is ``[base] + palette``: a plot that names no colour comes out in the neutral, and only a panel that means to separate conditions reaches into the hues. ``highlight`` is deliberately outside the cycle, because a colour that is handed to the second line of every plot cannot also mean emphasis.

Continuous scales live here too, under ``colormaps``, so the two kinds of colour move together and a figure never names a colormap of its own: anything categorical takes a hue from ``palette``, anything ordinal or continuous takes a colormap role, and an ordinal scale drawn as discrete swatches samples one with :func:`ramp` rather than picking hues that imply no order.

TVB-O's own colours are :data:`PATH`, the curated ``tvbo:theme/default`` shipped in ``tvbo/database/themes/``, and it is what :data:`DEFAULT` reads and what a project's file falls back to role by role. It is a ``Theme`` in the figure spec (``schema/figure.yaml``), so it validates like any other TVB-O document and so a consumer outside Python — the documentation site's stylesheet, the manuscript's figures — reads those hexes rather than keeping a copy that drifts.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

ROLES = ("ink", "base", "muted", "highlight", "background")
FIELDS = (*ROLES, "palette", "colormaps")
GUARANTEED_COLORMAPS = ("sequential", "diverging")

GEOMETRY = (
    "tick_length",
    "tick_width",
    "tick_direction",
    "tick_pad",
    "minor_ticks",
    "minor_tick_length",
    "minor_tick_width",
    "axis_width",
    "label_pad",
    "title_pad",
    "line_width",
    "marker_size",
    "legend_frame",
    "legend_handle_length",
    "legend_pad",
    "grid_lines",
    "font_family",
)
"""The Theme slots that are not colours. Named here so a theme file can be read for its colours without its geometry being reported as a typo; ``tests/test_plot_palette.py`` pins the tuple to the schema so the two cannot drift."""

PATH = Path(__file__).resolve().parents[1] / "database" / "themes" / "default.yaml"


def _shipped() -> dict[str, Any]:
    """Every slot of the curated theme the package ships, validated as a whole."""
    import yaml

    raw = yaml.safe_load(PATH.read_text()) or {}
    _fields(raw, where=str(PATH))
    return raw


def _fields(raw: dict, where: str = "palette") -> dict[str, Any]:
    """The colour slots of a Palette or a Theme, dropping the envelope and the geometry a Theme also carries.

    A key belonging to neither is refused rather than ignored: a misspelt role reads exactly like an absent one, so silently dropping it would hand back the default and call it the project's colour.
    """
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS

    allowed = {*FIELDS, *GEOMETRY, *ENVELOPE_KEYS, "iri"}
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise ValueError(
            f"{where}: {', '.join(map(repr, unknown))} is neither a colour role nor a Theme slot; the colour roles are {', '.join(FIELDS)}"
        )
    return {k: v for k, v in raw.items() if k in FIELDS}


_SHIPPED: dict[str, Any] = _shipped()

DEFAULT: dict[str, Any] = {k: v for k, v in _SHIPPED.items() if k in FIELDS}
"""TVB-O's own colours: the base every figure resolves a role against, whether or not it declared a theme."""

DEFAULT_GEOMETRY: dict[str, Any] = {k: v for k, v in _SHIPPED.items() if k in GEOMETRY}
"""The geometry the curated theme fixes, the same base for the other half of the look. Read here rather than in the adapter, because the file it comes from is this module's."""

_current: dict[str, Any] = dict(DEFAULT)


def load(source) -> dict:
    """Read and validate a palette, from a YAML path or a mapping already in hand.

    A file is read through TVBO's own loader, so a palette takes ``!include`` and merge keys like every other TVBO document and may name its class in the usual envelope.
    """
    from matplotlib.colors import is_color_like

    from tvbo.utils.yaml_loader import load_as_dict

    if isinstance(source, (str, Path)):
        raw, where = load_as_dict(str(source)) or {}, str(source)
    else:
        raw, where = dict(source), "palette"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a palette is a mapping of roles to colours")

    spec = {**DEFAULT, **_fields(raw, where)}
    if not isinstance(spec["palette"], list) or not spec["palette"]:
        raise ValueError(f"{where}: 'palette' must be a non-empty list of colours")
    spec["colormaps"] = {**DEFAULT["colormaps"], **(spec["colormaps"] or {})}
    for role, value in spec["colormaps"].items():
        _as_colormap(value, f"{where}: colormaps.{role}")
    _refuse_shadowed_names(spec, where)
    for key in (*ROLES, *(f"palette[{i}]" for i in range(len(spec["palette"])))):
        value = spec[key] if key in ROLES else spec["palette"][int(key[8:-1])]
        if not is_color_like(value):
            raise ValueError(f"{where}: {key} is {value!r}, which matplotlib cannot read as a colour")
    return spec


@cache
def _roles_shadowing_backend_colours() -> tuple[str, ...]:
    """The roles whose names the backend already uses for a colour of its own.

    ``ROLES`` is fixed, so the answer is too: resolved once rather than re-derived — against the whole named-colour mapping — on every palette that is loaded.
    """
    from matplotlib.colors import get_named_colors_mapping

    known = get_named_colors_mapping()
    return tuple(r for r in ROLES if r in known)


def _refuse_shadowed_names(spec: dict, where: str) -> None:
    """Refuse a name that would mean two things at once.

    A colour role and a colormap key are each resolved before the backend's own registry is consulted, so a project key spelling a name the backend already has would silently redirect every use of it. The clash is refused where it is written rather than reported as a figure that came out wrong, which is the only place it would otherwise show.
    """
    from matplotlib import colormaps as _registry

    shadowed = _roles_shadowing_backend_colours()
    if shadowed:
        raise ValueError(
            f"{where}: the colour role(s) {', '.join(shadowed)} name colours the backend already knows, so a mark asking for one would be ambiguous"
        )
    clash = [k for k in spec["colormaps"] if k not in GUARANTEED_COLORMAPS and k in _registry]
    if clash:
        raise ValueError(
            f"{where}: colormaps.{clash[0]!r} shadows a registered colormap of that name, so every figure naming it would silently get this one instead; give the key a name of its own"
        )


def use(source) -> dict:
    """Make *source* the current palette and put its colours into the rcParams that carry them."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    global _current
    _current = load(source)
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=cycle()),
            "image.cmap": _rcparam_cmap(_current["colormaps"]["sequential"]),
            "text.color": _current["ink"],
            "axes.edgecolor": _current["ink"],
            "axes.labelcolor": _current["ink"],
            "axes.titlecolor": _current["ink"],
            "xtick.color": _current["ink"],
            "ytick.color": _current["ink"],
            "patch.edgecolor": _current["ink"],
            "grid.color": _current["muted"],
            "axes.facecolor": _current["background"],
            "figure.facecolor": _current["background"],
            "savefig.facecolor": _current["background"],
        }
    )
    return dict(_current)


def _as_colormap(value, where: str):
    """A colormap from either a registered name or a list of colours to ramp between."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.pyplot import get_cmap

    if isinstance(value, list):
        if len(value) < 2:
            raise ValueError(f"{where}: a colormap given as a list needs at least two colours")
        try:
            return LinearSegmentedColormap.from_list(where, value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    try:
        return get_cmap(str(value))
    except ValueError as exc:
        raise ValueError(f"{where}: {value!r} is not a registered colormap") from exc


def _rcparam_cmap(value):
    """What ``image.cmap`` can hold: a registered name stays a name, a list becomes the colormap it builds."""
    return _as_colormap(value, "colormaps.sequential") if isinstance(value, list) else str(value)


def colormap(role: str = "sequential"):
    """The colormap this project gives *role* — the one continuous scale a panel is allowed to use."""
    maps = _current["colormaps"]
    if role not in maps:
        raise KeyError(f"{role!r} is not a colormap role; expected one of {', '.join(sorted(maps))}")
    return _as_colormap(maps[role], f"colormaps.{role}")


def ramp(n: int, role: str = "sequential", lo: float = 0.15, hi: float = 0.75) -> list:
    """*n* colours sampled across a colormap, for an ordinal scale drawn as discrete swatches.

    A ladder of rungs or a set of ordered bins is not categorical: hues from the palette would say the classes are unrelated, when the whole point is that they are ordered. The ends are trimmed by default, because a scale that runs into the colormap's near-black and near-white loses its extremes against the page.
    """
    scale = colormap(role)
    if n == 1:
        return [scale((lo + hi) / 2)]
    return [scale(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def current() -> dict:
    """The palette in force, as a plain dict."""
    return dict(_current)


def palette(n: int | None = None) -> list[str]:
    """The categorical hues, cycled to *n* entries when a panel needs a fixed number of them."""
    hues = list(_current["palette"])
    return hues if n is None else [hues[i % len(hues)] for i in range(n)]


def cycle() -> list[str]:
    """What ``axes.prop_cycle`` is set to: the neutral first, then the hues."""
    return [_current["base"], *_current["palette"]]


def color(name: str) -> str:
    """One colour by role name, or by ``palette.<index>`` for a hue."""
    if name.startswith("palette."):
        hues = _current["palette"]
        return hues[int(name.split(".", 1)[1]) % len(hues)]
    if name not in ROLES:
        raise KeyError(f"{name!r} is not a colour role; expected one of {', '.join(ROLES)} or palette.<index>")
    return _current[name]


def as_color(value):
    """*value* as a colour: a palette role or hue if it names one, otherwise itself.

    The one place a declared colour is turned into a drawn one. A role (``highlight``) or a hue (``palette.2``) resolves against the palette in force; a hex, a backend colour name, a sequence, or ``None`` passes through untouched, so a spec written before the palette existed keeps drawing exactly as it did. Resolution happens here rather than at generation time so an emitted script that swaps its palette recolours with it.
    """
    if not isinstance(value, str):
        return value
    if value in ROLES or (value.startswith("palette.") and value[8:].isdigit()):
        return color(value)
    return value


def as_colormap(value, default: str = "sequential"):
    """*value* as a colormap name: a palette key if it names one, otherwise itself.

    A key the project declared (``diverging``, or its own ``meg``) resolves to whatever that key holds; anything else is left for the backend's own registry, so ``cividis`` and ``parula`` are untouched. ``None`` takes *default*, which is how an undeclared mark ends up on the project's scale instead of the backend's.
    """
    maps = _current["colormaps"]
    if value is None:
        value = default
    return maps.get(value, value) if isinstance(value, str) else value


def ink() -> str:
    """Text, spines and ticks: the near-black everything is read against."""
    return _current["ink"]


def base() -> str:
    """Standard lines, bars and markers, wherever the colour carries no meaning."""
    return _current["base"]


def muted() -> str:
    """Hairlines and de-emphasised strokes: present, and not competing."""
    return _current["muted"]


def highlight() -> str:
    """The single emphasis colour, kept out of the cycle so it never lands on a panel by accident."""
    return _current["highlight"]


def background() -> str:
    """The page the figure is printed on."""
    return _current["background"]
