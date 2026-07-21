import ast
import os
import sys
from typing import Dict, List, Tuple

import yaml

SECTIONS_PATH = os.path.join("api", "_quartodoc_sections.yml")
QDOC_CONFIG_PATH = os.path.join("api", "_quartodoc_config.yml")
SIDEBAR_AUTO_PATH = os.path.join("api", "_sidebar.auto.yml")

# Clean section titles keyed by full package name.
# Used instead of raw docstring first-lines for quartodoc section headings.
# These are also used as sidebar labels by update_toc_api.py.
# For sub-packages, the label is the SHORT name (nesting provides context).
SECTION_TITLES: Dict[str, str] = {
    "tvbo":                          "tvbo",
    "tvbo.adapters":                 "Adapters",
    "tvbo.analysis":                 "Analysis",
    "tvbo.api":                      "API",
    "tvbo.bids":                     "BIDS",
    "tvbo.classes":                  "Classes",
    "tvbo.cli":                      "CLI",
    "tvbo.codegen":                  "TVB-O Code Generation",
    "tvbo.data":                     "Data",
    "tvbo.data.db":                  "DB",
    "tvbo.data.tvbo_data":           "TVB-O Data",
    "tvbo.data.tvbo_data.atlas":     "Atlas",
    "tvbo.data.tvbo_data.connectome": "Connectome",
    "tvbo.datamodel":                "Data Model",
    "tvbo.export":                   "Export",
    "tvbo.graph_generators":         "Graph Generators",
    "tvbo.jax":                      "JAX",
    "tvbo.ontology":                 "TVB-O Ontology",
    "tvbo.ontology.atlas":           "Atlas",
    "tvbo.ontology.semanticweb":     "Semantic Web",
    "tvbo.parse":                    "Parse",
    "tvbo.plot":                     "Plot",
    "tvbo.report":                   "Report",
    "tvbo.run":                      "Run",
    "tvbo.skills":                   "Skills",
    "tvbo.templates":                "Templates",
    "tvbo.templates.rateml":         "RateML",
    "tvbo.templates.tvboptim":       "tvboptim",
    "tvbo.utils":                    "Utilities",
}


def is_valid_module(filename: str) -> bool:
    """Return True if filename is a valid importable module name (no dashes, etc.)."""
    if not filename.endswith(".py") or filename == "__init__.py":
        return False
    mod = filename[:-3]
    # only include identifiers (skip names with hyphens, dots, commas, etc.)
    return mod.isidentifier() and not mod.startswith("_")


def get_docstring_title_and_description(package_init_path: str) -> Tuple[str, str]:
    """Extract a title and one-paragraph description from a module docstring.

    The title is the first non-empty line of the module's docstring. The
    description is the first paragraph that follows it, collapsed onto a single
    line. An optional RST-style underline (``====`` / ``----``) directly under
    the title is skipped, but is no longer *required* — plain Google/Markdown
    docstrings (which almost none of the TVBO packages underline) now yield a
    description too, instead of an empty string.
    """
    if not os.path.exists(package_init_path):
        return None, ""

    try:
        with open(package_init_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, ValueError):
        return None, ""

    doc = ast.get_docstring(tree)
    if not doc:
        return None, ""

    lines = [ln.strip() for ln in doc.strip().splitlines()]
    if not lines or not lines[0]:
        return None, ""

    title = lines[0]
    rest = lines[1:]

    # Skip an RST underline tied to the title (e.g. "=====" or "-----").
    if rest and rest[0] and set(rest[0]) <= {"=", "-", "~", "^"}:
        rest = rest[1:]

    # Description = the first non-empty paragraph after the title.
    description_lines: List[str] = []
    for line in rest:
        if line and not line.startswith("#"):
            description_lines.append(line)
        elif description_lines:
            break

    description = " ".join(description_lines).strip()
    return title, description


def collect_packages_and_modules(package_name: str, package_path: str) -> Dict[str, List[str]]:
    """
    Walk the package tree and collect all importable modules (.py files with valid names)
    for every directory that is a Python package (has __init__.py).
    Returns: { full.package.name: [module1, module2, ...] }
    """
    mapping: Dict[str, List[str]] = {}

    for root, dirs, files in os.walk(package_path):
        # ensure deterministic traversal
        dirs.sort()
        files.sort()

        if "__init__.py" not in files:
            # skip non-packages (cannot be imported by Python)
            continue

        rel = os.path.relpath(root, package_path)
        module_path = "" if rel == "." else rel.replace(os.sep, ".")
        full_pkg = f"{package_name}{('.' + module_path) if module_path else ''}"

        modules = [f[:-3] for f in files if is_valid_module(f)]
        mapping[full_pkg] = modules

    return mapping


def pkg_to_subpath(full_pkg: str, package_name: str) -> str:
    """Return the sub-path used for output files, e.g. 'adapters' for 'tvbo.adapters'.

    The root package itself maps to '' (files go directly into the api/ dir).
    """
    if full_pkg == package_name:
        return ""
    suffix = full_pkg[len(package_name) + 1:]   # e.g. "adapters" or "knowledge.simulation"
    return suffix.replace(".", "/")


def build_sections(package_name: str, package_path: str):
    """Build quartodoc sections where every module gets a Page with an explicit path.

    Using ``kind: page`` + ``path: <subpath>/<module>`` means quartodoc writes
    output files to ``api/<subpath>/<module>.qmd``.  Because the path is fully
    qualified, modules with the same basename in different packages never
    collide (e.g. ``api/adapters/julia.qmd`` vs ``api/run/julia.qmd``).

    Each non-root sub-package also gets an ``index`` page that documents
    the package's ``__init__.py``.
    """
    sections = []
    pkg_to_modules = collect_packages_and_modules(package_name, package_path)

    for full_pkg in sorted(pkg_to_modules.keys()):
        modules = pkg_to_modules[full_pkg]
        if not modules:
            continue

        subpath = pkg_to_subpath(full_pkg, package_name)

        # try to read a nice title/desc from that package's __init__.py
        pkg_dir = os.path.join(package_path, *full_pkg.split(".")[1:]) if "." in full_pkg else package_path
        init_path = os.path.join(pkg_dir, "__init__.py")
        raw_title, description = get_docstring_title_and_description(init_path)

        # Use clean section title from mapping; fall back to docstring or package name
        title = SECTION_TITLES.get(full_pkg, raw_title or full_pkg)

        # If we overrode the title and there's no separate description,
        # use the raw docstring title as the description.
        if raw_title and title != raw_title and not description:
            description = raw_title

        # Build Page-style contents so quartodoc respects the full path.
        # path is relative to the quartodoc `dir` (which is set to "." → api/).
        contents = []

        # `include_empty: true` is critical — quartodoc's auto-discovery
        # silently skips members without docstrings. Many TVBO core classes
        # (e.g. `SimulationExperiment`, `Dynamics`) inherit their docstrings
        # from the generated LinkML datamodel, so they appear docstring-less
        # to griffe's static parser. Without this flag, the class is omitted
        # from its module's page entirely.
        auto_opts = {"include_empty": True}

        # Add package index page documenting __init__.py.
        # Skip the root package — quartodoc generates api/index.qmd for it.
        if subpath:
            parent_pkg = full_pkg.rsplit(".", 1)[0] if "." in full_pkg else package_name
            pkg_basename = full_pkg.rsplit(".", 1)[-1]
            contents.append({
                "kind": "page",
                "path": f"{subpath}/index",
                "package": parent_pkg,
                "contents": [{"name": pkg_basename, **auto_opts}],
            })

        for mod in modules:
            page_path = f"{subpath}/{mod}" if subpath else mod
            contents.append({
                "kind": "page",
                "path": page_path,
                "package": full_pkg,
                "contents": [{"name": mod, **auto_opts}],
            })

        sections.append(
            {
                "title": title,
                "desc": description or "",
                "package": full_pkg,
                "contents": contents,
            }
        )
    return sections


def write_sections_yaml(sections, out_path):
    data = {"quartodoc": {"sections": sections}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Wrote quartodoc sections to: {out_path}")


def write_full_qdoc_config(sections, out_path):
    # Write a full config for the Quartodoc CLI. Point its sidebar to a different file
    # so we can generate our own nested sidebar without it being overwritten.
    config = {
        "quartodoc": {
            "sidebar": "./_sidebar.auto.yml",   # quartodoc will write this
            "parser": "google",
            "style": "pkgdown",
            "dir": ".",
            "package": "tvbo",
            "title": "TVB-O API Documentation",
            "sections": sections,
        }
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote quartodoc full config to: {out_path}")


def main(argv=None):
    # Import tvbo only when we need to inspect the package structure
    try:
        import tvbo  # type: ignore
    except Exception as e:
        print("Error: 'tvbo' package is required to build sections.")
        raise

    package_name = "tvbo"
    package_path = os.path.dirname(tvbo.__file__)

    # Freshness check: skip if the config already exists and nothing it is built
    # from has changed since the last run.
    #
    # The generated config embeds the *docstring of every module* in the package,
    # so the freshness input must be every ``.py`` file — not just
    # ``tvbo/__init__.py``. Watching only the package root meant a docstring-only
    # edit anywhere else (say a doc link in ``tvbo/run/graph.py``) was reported as
    # "unchanged" and silently never reached the rendered API docs. The generator
    # script itself is included so edits to ``build_sections`` are picked up too.
    stamp_file = os.path.join("api", ".struct_stamp")
    script_path = os.path.abspath(__file__)
    input_mtime = max(
        [os.path.getmtime(script_path)]
        + [
            os.path.getmtime(os.path.join(root, f))
            for root, _dirs, files in os.walk(package_path)
            for f in files
            if f.endswith(".py")
        ]
    )
    if os.path.exists(QDOC_CONFIG_PATH) and os.path.exists(stamp_file):
        stamp_mtime = os.path.getmtime(stamp_file)
        if input_mtime <= stamp_mtime:
            print(f"Quartodoc config up-to-date (tvbo package unchanged). Skipping.")
            return

    sections = build_sections(package_name, package_path)
    write_sections_yaml(sections, SECTIONS_PATH)       # for Quarto metadata-files
    write_full_qdoc_config(sections, QDOC_CONFIG_PATH) # for quartodoc CLI

    # Write stamp
    os.makedirs(os.path.dirname(stamp_file), exist_ok=True)
    with open(stamp_file, "w") as f:
        f.write(str(input_mtime))


if __name__ == "__main__":
    main()
