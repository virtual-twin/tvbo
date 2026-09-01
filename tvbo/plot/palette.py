# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Named colour roles for figures, so a project's colours live in one file.

A style sheet carries only what matplotlib has an rcParam for. That covers the cycler and the ink, and stops short of the roles a figure actually reasons in: the one colour that means *this is the point*, the neutral that everything unlabelled is drawn in, the hairline grey behind it. Those live here, in a YAML a figure names as a style layer (``style: [tvbo, sheet.mplstyle, palette.yaml]``), so changing a project's colours is one edit and every panel follows.

The cycler is ``[base] + palette``: a plot that names no colour comes out in the neutral, and only a panel that means to separate conditions reaches into the hues. ``highlight`` is deliberately outside the cycle, because a colour that is handed to the second line of every plot cannot also mean emphasis.

Continuous scales live here too, under ``colormaps``, so the two kinds of colour move together and a figure never names a colormap of its own: anything categorical takes a hue from ``palette``, anything ordinal or continuous takes a colormap role, and an ordinal scale drawn as discrete swatches samples one with :func:`ramp` rather than picking hues that imply no order.

TVB-O's own palette is :data:`PATH`, the ``palette.yaml`` shipped beside this module, and it is what :data:`DEFAULT` reads and what a project's file falls back to role by role. It is a ``Palette`` in the figure spec (``schema/figure.yaml``), so it validates like any other TVB-O document and so a consumer outside Python — the documentation site's stylesheet, the manuscript's figures — reads those hexes rather than keeping a copy that drifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROLES = ("ink", "base", "muted", "highlight", "background")
FIELDS = (*ROLES, "palette", "colormaps")

PATH = Path(__file__).with_name("palette.yaml")


def _shipped() -> dict[str, Any]:
    """TVB-O's own colours, read from the palette that ships beside this module."""
    import yaml

    return _fields(yaml.safe_load(PATH.read_text()) or {})


def _fields(raw: dict, where: str = "palette") -> dict[str, Any]:
    """The palette's own slots, dropping the ``tvbo_class``/``schema_version`` envelope a file may carry.

    Anything else is refused rather than ignored: a misspelt role reads exactly like an absent one, so silently dropping it would hand back the default and call it the project's colour.
    """
    from tvbo.utils.yaml_loader import ENVELOPE_KEYS

    unknown = [k for k in raw if k not in FIELDS and k not in ENVELOPE_KEYS]
    if unknown:
        raise ValueError(f"{where}: {', '.join(map(repr, unknown))} is not a Palette slot; expected {', '.join(FIELDS)}")
    return {k: v for k, v in raw.items() if k in FIELDS}


DEFAULT: dict[str, Any] = _shipped()

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
    for key in (*ROLES, *(f"palette[{i}]" for i in range(len(spec["palette"])))):
        value = spec[key] if key in ROLES else spec["palette"][int(key[8:-1])]
        if not is_color_like(value):
            raise ValueError(f"{where}: {key} is {value!r}, which matplotlib cannot read as a colour")
    return spec


def use(source) -> dict:
    """Make *source* the current palette and put its colours into the rcParams that carry them."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    global _current
    _current = load(source)
    plt.rcParams.update({
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
    })
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
