"""Report the specification slots no code reads, and those only one backend reads, as evidence for schema proposals.

Usage: `python scripts/spec_gaps.py > spec-gaps.md`

The slots are those of every class a `SimulationExperiment` reaches through its slot ranges and subclasses. A slot counts as read by a module when its name appears there as an attribute, a keyword argument or a word of a string that is not a docstring (a key, a regex, a message); in a Mako template, after a dot or in quotes. The test over-reports reads, since a common name such as `name` is read everywhere, and misses only a slot reached by iteration, so a slot no code names is one no backend or core module spells out. Each adapter module and template directory is one reader, grouped into a backend by its name (`BACKEND_OF` for the exceptions); the modules outside `tvbo/adapters/` and `tvbo/templates/`, those in `CORE` and the scripts under `scripts/` are core. A slot core reads is available to every backend, so only a slot no core module reads can be one backend's alone.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
import warnings

from linkml_runtime.utils.schemaview import SchemaView

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "tvbo"
SCHEMA = REPO / "schema" / "tvbo_datamodel.yaml"
ROOT_CLASS = "SimulationExperiment"
SKIPPED = ("datamodel", "templates/archive")
"""Generated code, and retired templates nothing renders."""
CORE = frozenset(
    {
        "adapters/base",
        "adapters/bids",
        "adapters/bsplot",
        "adapters/bsplot_distributions",
        "adapters/colormaps",
        "adapters/figure_workflow",
        "adapters/observation_sampling",
        "adapters/openminds",
        "templates/base",
        "templates/bsplot",
        "templates/report",
        "templates/study",
        "templates/workflow",
    }
)
"""Readers under `tvbo/adapters/` and `tvbo/templates/` that serve every backend: shared rendering, I/O standards, figures, reports and workflows."""
BACKEND_OF = {
    "autodiff": "jax",
    "generic_python": "python",
    "diffeq": "julia",
    "julia_model": "julia",
    "pyrates_bifurcation": "pyrates",
}
"""The backend of a reader whose name is not the backend's."""
MAKO_NAMES = re.compile(r"\.([A-Za-z_]\w*)|['\"]([A-Za-z_]\w*)['\"]")
WORDS = re.compile(r"[A-Za-z_]\w*")


def reader_of(path: pathlib.Path) -> str | None:
    """The backend *path* reads for, `"core"`, or `None` for a file the report skips."""
    if not path.is_relative_to(PACKAGE):
        return "core"
    relative = path.relative_to(PACKAGE).as_posix()
    if relative.startswith(SKIPPED):
        return None
    parts = relative.split("/")
    if parts[0] not in ("adapters", "templates") or len(parts) == 2 and parts[0] == "templates":
        return "core"
    stem = pathlib.Path(parts[1]).stem
    if f"{parts[0]}/{stem}" in CORE:
        return "core"
    return BACKEND_OF.get(stem, stem)


def names_in(path: pathlib.Path) -> set[str]:
    """The identifiers *path* spells as an attribute, a keyword argument or a quoted key."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".py":
        return {first or second for first, second in MAKO_NAMES.findall(text)}
    tree = ast.parse(text)
    docstrings = {
        id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            names.update(WORDS.findall(node.value))
    return names


def readers() -> dict[str, set[str]]:
    """`{identifier: the backends and "core" that name it}` over every Python module and Mako template in the package, and the maintainer scripts."""
    read: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted([*PACKAGE.rglob("*.py"), *PACKAGE.rglob("*.mako"), *(REPO / "scripts").rglob("*.py")]):
        reader = reader_of(path)
        if reader is not None:
            for name in names_in(path):
                read[name].add(reader)
    return read


def slots() -> dict[str, tuple[list[str], str]]:
    """`{slot: (the classes that have it, its description)}` for every class `ROOT_CLASS` reaches."""
    view = SchemaView(str(SCHEMA))
    classes = view.all_classes()
    seen, todo = set(), [ROOT_CLASS]
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        todo += [slot.range for slot in view.class_induced_slots(cls) if slot.range in classes]
        todo += view.class_descendants(cls)
    found: dict[str, tuple[list[str], str]] = {}
    for cls in sorted(seen):
        for slot in view.class_induced_slots(cls):
            name = slot.name.replace(" ", "_")
            found.setdefault(name, ([], " ".join((slot.description or "").split())))[0].append(cls)
    return found


def _first_sentence(text: str) -> str:
    return (text.split(". ")[0].rstrip(".") + ".") if text else ""


def report() -> str:
    """The Markdown report: the counts, then a table per gap."""
    read, specified = readers(), slots()
    unread = {name: entry for name, entry in specified.items() if not read.get(name)}
    single = {name: entry for name, entry in specified.items() if len(read.get(name, ())) == 1 and "core" not in read[name]}
    lines = [
        "# Specification gaps",
        "",
        f"`scripts/spec_gaps.py` over the {len(specified)} slots of the classes a `{ROOT_CLASS}` reaches. {len(unread)} are named by no module or template; {len(single)} are named by one backend and no core module.",
        "",
        "## Slots no code names",
        "",
        "| Slot | Classes | Description |",
        "| --- | --- | --- |",
        *(
            f"| `{name}` | {', '.join(owners)} | {_first_sentence(description)} |"
            for name, (owners, description) in sorted(unread.items())
        ),
        "",
        "## Slots one backend names",
        "",
        "| Slot | Backend | Classes |",
        "| --- | --- | --- |",
        *(
            f"| `{name}` | {next(iter(read[name]))} | {', '.join(owners)} |"
            for name, (owners, _) in sorted(single.items(), key=lambda item: (next(iter(read[item[0]])), item[0]))
        ),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(report(), end="")
