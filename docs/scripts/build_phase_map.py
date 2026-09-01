#!/usr/bin/env python
"""Map every sidebar page to the phase or chapter it sits under.

Reads the spine declared in ``_static/phases.yml`` and the sidebar in ``_toc.yml``, and writes ``_static/phase_map.lua`` — a Lua table the phase-badge filter loads directly, so the filter needs no JSON decoder and no Python at render time.

A page's phase comes from the PART header it sits beneath, and the directory tree mirrors that: each part declares the directories it owns, and a page filed outside them is reported so the layout cannot drift away from the sidebar unnoticed. The generated ``api/`` and ``datamodel/`` trees live in their own sidebars rather than in ``_toc.yml``, so they are mapped to the REFERENCE chapter by directory instead; every other page outside the curated sidebar carries a ``phase:`` key in its own frontmatter, which the filter prefers over anything written here.

Run as a Quarto pre-render step, after the ``update_toc_*`` generators have spliced their blocks in.
"""

from __future__ import annotations

import pathlib
import re

import yaml

DOCS = pathlib.Path(__file__).parent.parent
OUT = DOCS / "_static" / "phase_map.lua"

# Generated trees carry their own sidebars, so the PART walk never reaches them.
GENERATED = {"api": "API reference", "datamodel": "Data model"}


def spine() -> dict[str, dict]:
    """Every phase and chapter, keyed by the PART header text that introduces it, with its colour resolved to a hex."""
    decl = yaml.safe_load((DOCS / "_static" / "phases.yml").read_text())
    by_part: dict[str, dict] = {}
    for entry in decl["phases"]:
        by_part[entry["part"]] = {**entry, "color": colour(entry["color"]), "kind": "phase"}
    for entry in decl["chapters"]:
        by_part[entry["part"]] = {**entry, "color": colour(entry["color"]), "number": None, "kind": "chapter"}
    return by_part


def colour(slot: str) -> str:
    """The hex a `color:` slot names in the palette TVB-O ships: `palette.N` for a categorical hue, or a role such as `base`.

    Resolved by :mod:`tvbo.plot.palette`, the same reader the figures use, so the docs and the manuscript cannot disagree about what a role is; an out-of-range hue is refused here rather than cycled, because a sidebar that quietly reuses a colour reads as two phases sharing one. A hex written directly is passed through, so a one-off stays possible and stays visible.
    """
    from tvbo.plot import palette

    if str(slot).startswith("#"):
        return slot
    if str(slot).startswith("palette."):
        hues = palette.palette()
        index = int(str(slot).split(".", 1)[1])
        if not 0 <= index < len(hues):
            raise SystemExit(f"_static/phases.yml asks for {slot}, but the palette has {len(hues)} hue(s)")
    elif slot not in palette.ROLES:
        raise SystemExit(f"_static/phases.yml asks for colour {slot!r}, which {palette.PATH} does not define")
    return palette.color(str(slot))


def walk(items: list, part: str | None, section: str | None, out: dict, parts: set[str], index_pages: dict, group: str | None = None) -> str | None:
    """Descend the sidebar, remembering the PART header and the nearest enclosing section.

    A chapter heading is a text-only row and the pages under it are its siblings. The five numbered phases are sections instead: the label is a declared part, the href is that phase's overview page, and the contents are the phase's pages. Either form names the part everything beneath it belongs to, and a linked one has its href recorded for ``write_css``.

    ``group`` is the spine's own heading, the one row that introduces the numbered phases rather than a set of pages; it owns nothing, so it is skipped without becoming the part its neighbours inherit.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        label, href, kids = item.get("section") or item.get("text"), item.get("href"), item.get("contents")
        if label == group:
            continue
        if label in parts:
            part = label
            if href:
                index_pages[label] = href
            elif not kids:
                continue
        elif label and not href and not kids:
            part = label
            continue
        if href and href.endswith((".qmd", ".md")):
            out.setdefault(href, {"part": part, "section": section})
        if kids:
            inner = None if label in parts else (item.get("section") or section)
            walk(kids, part, inner, out, parts, index_pages, group)
    return part


def main() -> None:
    toc = yaml.safe_load((DOCS / "_toc.yml").read_text())
    by_part = spine()
    pages: dict[str, dict] = {}
    index_pages: dict[str, str] = {}
    group = (yaml.safe_load((DOCS / "_static" / "phases.yml").read_text()).get("spine") or {}).get("part")
    contents = toc["website"]["sidebar"][0]["contents"]
    walk(contents, None, None, pages, set(by_part), index_pages, group)
    first_part = next((i for i in contents if isinstance(i, dict) and i.get("text") in by_part and not i.get("contents")), None)

    reference = next((e for e in by_part.values() if e["key"] == "reference"), None)
    if reference is None:
        raise SystemExit(
            "_static/phases.yml declares no chapter with key `reference`, which is where the generated api/ and datamodel/ trees are mapped."
        )
    for directory, section in GENERATED.items():
        for page in sorted((DOCS / directory).rglob("*.qmd")):
            pages.setdefault(str(page.relative_to(DOCS)), {"part": reference["part"], "section": section})

    rows, unmapped = [], []
    for href, where in sorted(pages.items()):
        entry = by_part.get(where["part"] or "")
        if entry is None:
            unmapped.append(href)
            continue
        section = where["section"] or entry["name"]
        rows.append(
            "  [{h}] = {{number={n}, name={nm}, key={k}, icon={i}, color={c}, section={s}, kind={kd}}},".format(
                h=lua_str(href),
                n=entry["number"] if entry["number"] is not None else "nil",
                nm=lua_str(entry["name"]),
                k=lua_str(entry["key"]),
                i=lua_str(entry["icon"]),
                c=lua_str(entry["color"]),
                s=lua_str(section),
                kd=lua_str(entry["kind"]),
            )
        )

    misfiled = [
        f"{href} sits under {href.rsplit('/', 1)[0] if '/' in href else '(root)'}/, but {where['part']} owns {', '.join(d or '(root)' for d in by_part[where['part']].get('dirs') or ['(nothing)'])}"
        for href, where in sorted(pages.items())
        if where["part"] in by_part and not owned(href, by_part[where["part"]].get("dirs") or [])
    ]

    OUT.write_text("-- Generated by scripts/build_phase_map.py. Do not edit.\nreturn {\n" + "\n".join(rows) + "\n}\n")
    write_css(by_part, index_pages, (first_part or {}).get("href"))
    print(f"[phase-map] {len(rows)} pages mapped -> {OUT.relative_to(DOCS)}")
    if unmapped:
        print(f"[phase-map] {len(unmapped)} sidebar page(s) sit under no PART header:")
        for href in unmapped:
            print(f"    {href}")
    if misfiled:
        print(f"[phase-map] {len(misfiled)} page(s) are filed outside the directories their part owns:")
        for line in misfiled:
            print(f"    {line}")


def owned(href: str, dirs: list[str]) -> bool:
    """Whether a page sits in one of the directories its part declares in `_static/phases.yml`."""
    directory = href.rsplit("/", 1)[0] if "/" in href else ""
    return any(directory == d or directory.startswith(d + "/") if d else directory == "" for d in dirs)


def write_css(by_part: dict[str, dict], index_pages: dict[str, str], first_href: str | None) -> None:
    """Emit the phase palette as custom properties plus the sidebar rules for the linked PART headers.

    A chapter heading is a text-only row that ``styles.css`` styles as a bare ``<span class="menu-text">``. The five numbered phases sit one tier below the spine's own heading, so they are styled here instead: smaller, wearing their phase colour, and without the rule that separates one chapter from the next. Selectors match by href suffix, because Quarto writes the sidebar's paths relative to each page's depth.

    ``first_href`` is the href of the sidebar's topmost PART header, or ``None`` when that header is text-only; only a linked one needs its rule and gap suppressed here, because ``styles.css`` already does it positionally for the bare-span form.
    """
    lines = ["/* Generated by scripts/build_phase_map.py from _static/phases.yml and _toc.yml. Do not edit. */", ":root {"]
    for entry in by_part.values():
        if entry["kind"] == "phase":
            lines.append(f"    --phase-{entry['number']}: {entry['color']};")
    lines += ["}", ""]

    linked = [(by_part[part], href) for part, href in index_pages.items() if part in by_part]
    if linked:
        selectors = ",\n".join(selectors_for(href, " > span.menu-text") for _, href in linked)
        lines += [
            selectors + " {",
            "    display: block;",
            "    margin: 0.45rem 0 0.1rem;",
            "    font-size: 0.7rem;",
            "    font-weight: 700;",
            "    letter-spacing: 0.06em;",
            "    text-transform: uppercase;",
            "}",
            "",
        ]
        for entry, href in linked:
            lines += [
                selectors_for(href, " > span.menu-text") + f" {{ color: {entry['color']}; }}",
                selectors_for(href, ":hover > span.menu-text")
                + f" {{ background: color-mix(in srgb, {entry['color']} 10%, transparent); }}",
            ]
        if first_href:
            lines += [
                "",
                "/* The topmost part header needs no rule or gap above it. */",
                selectors_for(first_href, " > span.menu-text")
                + " { margin-top: 0.25rem; padding-top: 0; border-top: none; }",
                "",
            ]
    (DOCS / "_static" / "phases.css").write_text("\n".join(lines))


def html_href(href: str) -> str:
    """The rendered path of a sidebar entry, which is what the emitted CSS matches on."""
    return re.sub(r"\.(qmd|md)$", ".html", href)


def href_suffixes(href: str) -> list[str]:
    """Every form of a sidebar href a rule has to match. Quarto's nav script rewrites a link to a directory index as the bare directory, so the same row is `1-explore/index.html` in the served HTML and `1-explore/` once the page is live."""
    rendered = html_href(href)
    directory = rendered[: -len("index.html")] if rendered.endswith("/index.html") else None
    return [rendered] + ([directory] if directory else [])


def selectors_for(href: str, tail: str = "") -> str:
    return ",\n".join(f'#quarto-sidebar a.sidebar-link[href$="{suffix}"]{tail}' for suffix in href_suffixes(href))


def lua_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


if __name__ == "__main__":
    main()
