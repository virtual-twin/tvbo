#!/usr/bin/env python
"""Auto-generate the 'API Documentation' section of _toc.yml.

Reads api/_quartodoc_sections.yml, the source of truth for quartodoc sections.

Rewrites the block between the # BEGIN:api-autogen … # END:api-autogen markers in _toc.yml.

Sub-packages are nested under their parent package section (e.g.
Templates → RateML, tvboptim) rather than appearing as flat siblings.

Run automatically as a Quarto pre-render step (after quartodoc build).
"""

from __future__ import annotations

from pathlib import Path

import yaml

DOCS_DIR = Path(__file__).parent.parent
TOC_FILE = DOCS_DIR / "_toc.yml"
SECTIONS_FILE = DOCS_DIR / "api" / "_quartodoc_sections.yml"
API_DIR = DOCS_DIR / "api"

BEGIN_MARKER = "# BEGIN:api-autogen"
END_MARKER = "# END:api-autogen"

BASE_INDENT = 14  # spaces — matches sidebar nesting level in _toc.yml
INDENT_STEP = 4  # additional spaces per nesting level

# Section titles that are internal/noise — skip them entirely
SKIP_TITLES = {
    "Welcome to the TVB-O project!",
}

# Display-name overrides for individual module files. Anything not listed here falls back to ``basename.replace("_", " ").title()`` (e.g. ``base`` → ``Base``). Use this map to preserve product casings that ``.title()`` would mangle (``Tvb`` → ``TVB``, ``Pyrates`` → ``PyRates`` …).
MODULE_DISPLAY_NAMES: dict[str, str] = {
    "bids": "BIDS",
    "tvb": "TVB",
    "tvboptim": "tvboptim",
    "pyrates": "PyRates",
    "pyrates_bifurcation": "PyRates (Bifurcation)",
    "modelingtoolkit": "ModelingToolkit",
    "networkdynamics": "NetworkDynamics",
    "neuroml": "NeuroML",
    "openminds": "openMINDS",
    "bifurcationkit": "BifurcationKit",
    "numcont": "NumCont",
    "diffeq": "DiffEq",
    "rateml": "RateML",
    "lems": "LEMS",
    "cuda": "CUDA",
    "jax": "JAX",
    "cli": "CLI",
    "api": "API",
    "io": "I/O",
    "fc": "FC",
    "psd": "PSD",
    "owl": "OWL",
    "db": "DB",
    "tvbgo": "TVB-GO",
    "import_": "Import",  # trailing underscore to avoid keyword clash
}


def _display_for_basename(basename: str) -> str:
    """Map a module basename to its TOC display label."""
    if basename in MODULE_DISPLAY_NAMES:
        return MODULE_DISPLAY_NAMES[basename]
    return basename.replace("_", " ").title()


# ── helpers ──────────────────────────────────────────────────────────


def _indent(depth: int) -> str:
    """Return indentation string for a given nesting depth (0-based)."""
    return " " * (BASE_INDENT + INDENT_STEP * depth)


def _section_label(title: str, package: str) -> str | None:
    """Return display name for a section, or None to skip it."""
    if title in SKIP_TITLES:
        return None
    if title:
        return title
    return package.split(".")[-1].capitalize()


def _resolve_pages(contents: list) -> tuple[str | None, list[tuple[str, str]]]:
    """Extract index href and list of (display, href) from section contents.

    Only includes pages whose .qmd file actually exists on disk.
    """
    index_href: str | None = None
    pages: list[tuple[str, str]] = []

    for item in contents or []:
        if isinstance(item, dict) and item.get("kind") == "page":
            page_path = item["path"]
            qmd_file = API_DIR / (page_path + ".qmd")
            if not qmd_file.exists():
                continue
            basename = page_path.split("/")[-1]
            if basename == "index":
                index_href = f"api/{page_path}.qmd"
            else:
                pages.append((_display_for_basename(basename), f"api/{page_path}.qmd"))
        elif isinstance(item, str):
            basename = item.split(".")[-1]
            fname = basename + ".qmd"
            if (API_DIR / fname).exists():
                pages.append((_display_for_basename(basename), f"api/{fname}"))

    return index_href, pages


# ── tree building ────────────────────────────────────────────────────


class _Node:
    """One section in the TOC tree."""

    __slots__ = ("package", "label", "index_href", "pages", "children")

    def __init__(self, package: str, label: str, index_href: str | None, pages: list[tuple[str, str]]):
        self.package = package
        self.label = label
        self.index_href = index_href
        self.pages = pages
        self.children: list[_Node] = []


def _build_tree(sections: list[dict]) -> list[_Node]:
    """Build a tree of _Nodes from the flat quartodoc sections list.

    A section with package ``tvbo.templates.rateml`` becomes a child of ``tvbo.templates`` (if present).  Sections whose parent is not in the set become root nodes.

    Parent packages that have no section of their own (e.g. no modules) are synthesised as container nodes using SECTION_TITLES from tvbo_package_struct.
    """
    # Import SECTION_TITLES for synthesising missing parent nodes
    from tvbo_package_struct import SECTION_TITLES

    nodes: dict[str, _Node] = {}

    for sec in sections:
        title = sec.get("title", "")
        package = sec.get("package", "")
        contents = sec.get("contents", []) or []

        label = _section_label(title, package)
        if label is None:
            continue

        index_href, pages = _resolve_pages(contents)

        nodes[package] = _Node(package, label, index_href, pages)

    # Synthesise missing parents so children nest: tvbo.templates.rateml without tvbo.templates gets an empty container.
    for pkg in list(nodes):
        parts = pkg.split(".")
        for i in range(2, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in nodes:
                label = SECTION_TITLES.get(parent, parent.split(".")[-1].capitalize())
                nodes[parent] = _Node(parent, label, None, [])

    # Wire up parent-child relationships
    roots: list[_Node] = []
    for pkg in sorted(nodes):
        parent_pkg = pkg.rsplit(".", 1)[0] if "." in pkg else None
        if parent_pkg and parent_pkg in nodes:
            nodes[parent_pkg].children.append(nodes[pkg])
        else:
            roots.append(nodes[pkg])

    return roots


# ── rendering ────────────────────────────────────────────────────────


def _render_node(node: _Node, depth: int, lines: list[str]) -> None:
    """Recursively render a _Node and its children as indented YAML."""
    # Skip nodes with no pages and no children (empty sections)
    if not node.pages and not node.children:
        return

    i = _indent(depth)
    ic = _indent(depth + 1)

    lines.append(f'{i}- section: "{node.label}"')
    if node.index_href:
        lines.append(f"{i}  href: {node.index_href}")
    lines.append(f"{i}  contents:")

    # Render leaf pages first
    for display, href in node.pages:
        lines.append(f'{ic}- text: "{display}"')
        lines.append(f"{ic}  href: {href}")

    # Then render child sections
    for child in node.children:
        _render_node(child, depth + 1, lines)


def build_block() -> str:
    data = yaml.safe_load(SECTIONS_FILE.read_text())
    sections = data.get("quartodoc", {}).get("sections", [])

    roots = _build_tree(sections)

    lines: list[str] = [BEGIN_MARKER]
    i0 = _indent(0)
    i1 = _indent(1)
    lines.append(f'{i0}- section: "API Documentation"')
    lines.append(f"{i0}  href: api/index.qmd")
    lines.append(f"{i0}  contents:")

    for node in roots:
        # The root `tvbo` section has no label of its own — inline its pages directly (they're top-level modules like tvbo.utils).
        if node.package == "tvbo":
            for display, href in node.pages:
                lines.append(f'{i1}- text: "{display}"')
                lines.append(f"{i1}  href: {href}")
            continue

        _render_node(node, 1, lines)

    lines.append(END_MARKER)
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────


def update_toc() -> None:
    text = TOC_FILE.read_text()

    begin_idx = text.find(BEGIN_MARKER)
    end_idx = text.find(END_MARKER)

    if begin_idx == -1 or end_idx == -1:
        print(f"✗ Markers not found in {TOC_FILE}. Add {BEGIN_MARKER!r} / {END_MARKER!r}.")
        raise SystemExit(1)

    new_block = build_block()
    new_text = text[:begin_idx] + new_block + text[end_idx + len(END_MARKER) :]

    if new_text == text:
        print("API TOC section unchanged — skipping write.")
        return

    TOC_FILE.write_text(new_text)
    print(f"✓ Updated API Documentation section in {TOC_FILE.name}")


if __name__ == "__main__":
    update_toc()
