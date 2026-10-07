#!/usr/bin/env python
"""Rewrite the retired panel options into the objects that replaced them.

Every axis directive used to be a loose entry in `Panel.opts` — `axvline` beside `axvline_color`, `elev` beside `azim`, a `legend` slot that meant a boolean in one spec and a corner in the next. They are declared slots now: the labels, limits, scales, shape, frame and legend on the panel itself, the tick family beside them, and `rules`, `regions` and `camera` for the marks that carry a colour with them. What is left in `opts` is a custom callable's keywords, and the adapter refuses a retired spelling rather than passing an unknown keyword into the emitted script.

Run it over a study tree. A spec with no comments and no anchors is rewritten in place; one that carries either is reported instead, because a ruamel round-trip reflows the whole file (921 of Taher2019's 1722 lines, for a migration that touches ten) and the change would be unreviewable inside that. `--show` prints the migrated panels for those, to paste.

    python scripts/migrate_panel_marks.py ~/projects/TVB-O/tvbo-manuscript

A built-in kind's own options move into the object named after the kind — `surface:`, `volume:`, `network:`, `grid:`, `colorbar:` — and only for a panel of that kind, because `color`, `labels` and `cmap` are still perfectly good keywords for a `custom` callable and must not be moved out from under one.

Only `opts:` blocks are touched, so a `region:` that means an anatomical region, an `elev` outside a panel, and — the one that matters most here — every `title:` that is a study's or a figure's rather than an axis's, are left alone. Pass `--check` to list what would change without writing.

It round-trips through ruamel, because these are hand-authored specs: `Taher2019.yaml` alone carries 55 comments and 72 anchors, and a parse-and-dump through the plain loader would erase every comment and expand every anchor into duplicated inline content. A migration that costs a study its authored structure is worse than the hand edit it replaces.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys
from textwrap import indent

from ruamel.yaml import YAML

from tvbo.adapters.bsplot import HEMISPHERES, retired_options

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096  # these specs are written unwrapped; re-wrapping them would rewrite lines the migration never touched
_yaml.indent(
    mapping=2, sequence=4, offset=2
)  # the convention these specs are written in, so the diff shows the migration and not a reflow

RULES = {"axhline": "horizontal", "axvline": "vertical", "axline": "diagonal"}
CAMERA = {"elev": "elevation", "azim": "azimuth", "zoom": "zoom"}
NUMERIC = {"xlim", "ylim", "zlim", "xlabel_pad", "ylabel_pad", "zlabel_pad", "box_aspect"}
UNHEMI = {v: k for k, v in HEMISPHERES.items()}

SHAPED = {*RULES, *(f"{k}_color" for k in RULES), *CAMERA, "region", "region_color", "frame", "legend", "legend_loc"}
"""The retirements that changed a spec's SHAPE rather than a name, and so are rewritten by the code below. Everything else is a rename, read straight out of :func:`retired_options` — the renderer's own statement of where each option went, so a migrated spec is exactly what it will accept."""


def _number(value):
    """A numeric-looking string as a number, through a list; anything else unchanged.

    These slots are typed `float` now, so a limit written as ``"1.0e-4"`` — the quoting YAML 1.1 forces on an unsigned exponent — has to become the number it means rather than reaching the renderer as text.
    """
    if isinstance(value, list):
        return [_number(v) for v in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _legend(opts: dict):
    """The `legend` slot a panel's flat `legend`/`legend_loc` options describe, or None where it declares neither."""
    raw, loc = _value(opts, "legend"), _value(opts, "legend_loc")
    if raw is None and loc is None:
        return None
    if raw is False:
        return False
    corner = loc or (raw if isinstance(raw, str) and raw else None)
    return corner or True


def _value(opts: dict, key: str):
    """The value of a panel option, whether written bare or in the `Argument` form (`{value: ...}`)."""
    raw = opts.get(key)
    return raw.get("value") if isinstance(raw, dict) and "value" in raw else raw


def _rules(opts: dict) -> list[dict]:
    """The rule objects a panel's flat `ax*line` options describe."""
    out = []
    for key, orientation in RULES.items():
        if key not in opts:
            continue
        values = _value(opts, key)
        values = values if isinstance(values, list) else [values]
        positions = values if orientation == "diagonal" else [[v] for v in values]
        for at in positions:
            rule = {"orientation": orientation, "at": list(at) if isinstance(at, list) else [at]}
            if _value(opts, f"{key}_color") is not None:
                rule["color"] = _value(opts, f"{key}_color")
            out.append(rule)
    return out


def _regions(opts: dict) -> list[dict]:
    """The region objects a panel's flat `region` option describes."""
    raw = _value(opts, "region")
    if raw is None:
        return []
    boxes = raw if raw and isinstance(raw[0], list) else [raw]
    return [
        {"bounds": list(b), **({"color": _value(opts, "region_color")} if _value(opts, "region_color") is not None else {})}
        for b in boxes
    ]


class Shared(Exception):
    """An `opts:` block reached through a YAML anchor, which cannot be migrated for one panel alone."""


def convert(node, in_opts: bool = False) -> bool:
    """Rewrite every `opts:` block under *node* in place, reporting whether anything changed.

    Raises :class:`Shared` where a block is anchored or merged: moving a key out of it would move it for every panel aliasing it, which is a change to figures the migration was never asked about.
    """
    changed = False
    if isinstance(node, list):
        return any([convert(v) for v in node])
    if not isinstance(node, dict):
        return False
    opts = node.get("opts")
    renamed = retired_options(str(node.get("kind") or ""))
    moved = SHAPED | set(renamed)
    if isinstance(opts, dict) and moved & set(opts):
        if rules := _rules(opts):
            node.setdefault("rules", []).extend(rules)
        if regions := _regions(opts):
            node.setdefault("regions", []).extend(regions)
        if camera := {slot: _value(opts, key) for key, slot in CAMERA.items() if key in opts}:
            node.setdefault("camera", {}).update(camera)
        for option, (slot, attr) in renamed.items():
            if option not in opts or option in SHAPED:
                continue
            value = _value(opts, option)
            if attr == "hemi":
                value = UNHEMI.get(value, value)
            elif attr in NUMERIC:
                value = _number(value)
            node.setdefault(slot, {}).setdefault(attr, value) if slot else node.setdefault(attr, value)
        if "frame" in opts:
            node.setdefault("frame", _value(opts, "frame") not in ("off", "false", False))
        legend = _legend(opts)
        if legend is not None:
            node.setdefault("legend", legend)
        try:
            for key in moved & set(opts):
                del opts[key]
            if not opts:
                del node["opts"]
        except Exception as exc:  # noqa: BLE001 — ruamel refuses a delete through a merge, which is exactly the case to decline
            raise Shared from exc
        changed = True
    return any([convert(v) for v in node.values()]) or changed


def _authored(text: str) -> bool:
    """Whether a file carries structure a round-trip would damage: comments, or anchors and aliases."""
    return (
        any(line.lstrip().startswith("#") for line in text.splitlines())
        or re.search(r"(?<![\w\"'])[&*][A-Za-z_]", text) is not None
    )


def _replacement(data) -> str:
    """The migrated panels alone, as YAML to paste, so a file that cannot be rewritten can still be fixed by eye."""
    panels = data.get("panels") if isinstance(data, dict) else None
    subject = panels if isinstance(panels, dict) else data
    buf = io.StringIO()
    _yaml.dump(subject, buf)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=pathlib.Path, help="study trees or spec files to convert")
    parser.add_argument("--check", action="store_true", help="report what would change without writing")
    parser.add_argument(
        "--show", action="store_true", help="print the migrated YAML for the files that must be edited by hand"
    )
    args = parser.parse_args()

    written, byhand, shared = [], [], []
    for root in args.roots:
        files = sorted(root.rglob("*.yaml")) if root.is_dir() else [root]
        for path in files:
            if ".venv" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                data = _yaml.load(text)
            except Exception:  # noqa: BLE001 — a file this cannot parse is not a spec to migrate
                continue
            try:
                if data is None or not convert(data):
                    continue
            except Shared:
                shared.append(path)
                continue
            if _authored(text):
                byhand.append((path, data))
                continue
            written.append(path)
            if not args.check:
                with path.open("w", encoding="utf-8") as fh:
                    _yaml.dump(data, fh)

    verb = "would rewrite" if args.check else "rewrote"
    print(f"{verb} {len(written)} spec(s)")
    for path in written:
        print(f"  {path}")
    if byhand:
        print(
            f"\n{len(byhand)} spec(s) carry comments or anchors, so they are NOT rewritten: a round-trip reflows the whole file and the migration would be unreviewable inside it. Edit these by hand (`--show` prints the migrated panels):"
        )
        for path, data in byhand:
            print(f"  {path}")
            if args.show:
                print(indent(_replacement(data), "    "))
    if shared:
        print(
            f"\n{len(shared)} spec(s) reach a retired key through a YAML anchor, so moving it would move it for every panel aliasing it. These need a decision, not a rewrite:"
        )
        for path in shared:
            print(f"  {path}")
    return 1 if (args.check and (written or byhand or shared)) else 0


if __name__ == "__main__":
    sys.exit(main())
