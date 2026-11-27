"""What generated code takes from the frameworks it targets.

TVBO depends on a framework through the names its generated code imports, calls, subclasses and passes as keywords. That set, the join between TVBO and each framework, is what a framework release can break, and it is small: dozens of names per backend, against the hundreds of thousands of lines a framework holds. [`inventory`](#tvbo.codegen.inventory.inventory) reads it off generated source.

The readers share nothing with the emitters, as in [`backend_check`](../analysis/backend_check.qmd): Python is read with `ast`, Julia with tree-sitter. A Julia file brings names in with `using`, which exports them unqualified, so a Julia call is recorded against the packages the file loads and resolved to one of them only against that package's own export list.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

IMPORT = "import"
ATTRIBUTE = "attribute"
CALL = "call"
KEYWORD = "keyword"
BASE_CLASS = "base_class"


@dataclass(frozen=True, order=True)
class EmittedSymbol:
    """One name generated code takes from a package.

    `name` is qualified as the package spells it, `tvb.simulator.models.base.Model`; a keyword is written `callee(keyword=)`. For Julia, whose `using` exports names unqualified, `package` lists every package the file loads, joined by `|`, until one of them is shown to export the name.
    """

    package: str
    name: str
    kind: str


def _external(module: str) -> bool:
    """Whether *module* belongs to a third-party package rather than the standard library."""
    return module.split(".")[0] not in sys.stdlib_module_names


def python_symbols(code: str) -> set[EmittedSymbol]:
    """The third-party names a generated Python module imports, reads, calls, subclasses and passes keywords to."""
    from tvbo.analysis.backend_check import dotted

    tree = ast.parse(code)
    bound: dict[str, str] = {}
    found: set[EmittedSymbol] = set()

    def add(name: str, kind: str) -> None:
        found.add(EmittedSymbol(name.split(".")[0], name, kind))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                if _external(alias.name):
                    add(alias.name, IMPORT)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in (alias for alias in node.names if alias.name != "*"):
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                if _external(node.module):
                    add(f"{node.module}.{alias.name}", IMPORT)

    def qualified(node) -> str:
        """The dotted name *node* spells through a bound import, or `""`."""
        head, _, rest = dotted(node).partition(".")
        return ".".join(filter(None, (bound[head], rest))) if head in bound else ""

    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    inner |= {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    inner |= {id(base) for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for base in node.bases}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = qualified(base)
                if name and _external(name):
                    add(name, BASE_CLASS)
        elif isinstance(node, ast.Attribute) and id(node) not in inner:
            name = qualified(node)
            if name and _external(name):
                add(name, ATTRIBUTE)
        elif isinstance(node, ast.Call):
            name = qualified(node.func)
            if not name and isinstance(node.func, ast.Name) and node.func.id == "super":
                continue
            if name and _external(name):
                add(name, CALL)
                for keyword in node.keywords:
                    if keyword.arg:
                        add(f"{name}({keyword.arg}=)", KEYWORD)
    found |= _super_keywords(tree, qualified)
    return found


def _super_keywords(tree, qualified) -> set[EmittedSymbol]:
    """The keywords a subclass passes to its third-party base's constructor through `super().__init__`.

    The constructor that receives them is the base class's, so they are recorded against it: `incoming_states=` passed by a tvboptim coupling is a keyword of tvboptim's coupling class, and the first thing a rename there breaks.
    """
    found = set()
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        bases = [name for name in (qualified(base) for base in cls.bases) if name and _external(name)]
        if not bases:
            continue
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "__init__"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "super"
            ):
                for keyword in node.keywords:
                    if keyword.arg:
                        found.add(EmittedSymbol(bases[0].split(".")[0], f"{bases[0]}({keyword.arg}=)", KEYWORD))
    return found


JULIA_BASE = frozenset(
    {
        "Array",
        "Dict",
        "Float64",
        "Int",
        "Matrix",
        "NamedTuple",
        "Vector",
        "abs",
        "abs2",
        "acos",
        "all",
        "any",
        "asin",
        "atan",
        "ceil",
        "collect",
        "convert",
        "copy",
        "cos",
        "cosh",
        "eachindex",
        "enumerate",
        "exp",
        "exp10",
        "fill",
        "first",
        "floor",
        "hcat",
        "ifelse",
        "isfinite",
        "isnan",
        "last",
        "length",
        "log",
        "log10",
        "map",
        "max",
        "maximum",
        "min",
        "minimum",
        "mod",
        "ones",
        "println",
        "prod",
        "push!",
        "rand",
        "randn",
        "range",
        "reduce",
        "round",
        "sign",
        "similar",
        "sin",
        "sinh",
        "size",
        "sqrt",
        "string",
        "sum",
        "tan",
        "tanh",
        "tuple",
        "vcat",
        "vec",
        "zeros",
        "zip",
    }
)
"""Julia `Base` functions, which no package provides and so no package release can remove."""


def julia_symbols(code: str) -> set[EmittedSymbol]:
    """The package names a generated Julia file loads, qualifies and calls.

    Calls to functions the file defines itself, and to the `Base` functions in `JULIA_BASE`, are left out.
    """
    from tvbo.analysis.backend_check import julia_parser

    root = julia_parser().parse(code.encode()).root_node
    loaded: list[str] = []
    defined: set[str] = set()
    found: set[EmittedSymbol] = set()
    nodes = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.children)
    for node in nodes:
        if node.type in ("using_statement", "import_statement"):
            for child in node.named_children:
                module = (child.named_children[0] if child.type == "selected_import" else child).text.decode()
                loaded.append(module)
                found.add(EmittedSymbol(module, module, IMPORT))
        elif node.type == "function_definition":
            signature = node.named_children[0].named_children[0]
            defined.add(signature.children[0].text.decode())
        elif node.type == "assignment" and node.children[0].type == "call_expression":
            defined.add(node.children[0].children[0].text.decode())
    providers = "|".join(sorted(set(loaded)))
    for node in nodes:
        if node.type not in ("call_expression", "broadcast_call_expression"):
            continue
        callee = node.children[0]
        name = callee.text.decode()
        if callee.type == "field_expression":
            found.add(EmittedSymbol(name.split(".")[0], name, CALL))
        elif callee.type == "identifier" and name not in defined and name not in JULIA_BASE and providers:
            found.add(EmittedSymbol(providers, name, CALL))
            for argument in node.children[-1].named_children:
                if argument.type == "named_argument":
                    found.add(EmittedSymbol(providers, f"{name}({argument.named_children[0].text.decode()}=)", KEYWORD))
    return found


READERS = {"python": python_symbols, "julia": julia_symbols}
"""The reader for each output language a code format declares."""


def corpus_cases(directory: pathlib.Path) -> list[tuple[str, str, str]]:
    """`(case_id, format, code)` for each golden reference `<case>.<format>.txt` in *directory*, the triples `inventory` reads."""
    return [
        (path.stem, path.stem.rpartition(".")[2], path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.txt"))
    ]


def software_entries() -> list[tuple[pathlib.Path, dict]]:
    """Every curated `SimulationTool` record, with the file it is read from."""
    import yaml

    from tvbo.data.registry import database_dir

    paths = sorted(path for path in database_dir("SimulationTool").glob("*.yaml") if not path.stem.startswith("_"))
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}) for path in paths]


def module_families(entries: Iterable[tuple[pathlib.Path, dict]]) -> dict[str, str]:
    """`{module: family}` for `inventory`: each module a software entry lists, in the family its entry's first module names."""
    return {module: record["modules"][0] for _, record in entries for module in record.get("modules") or []}


def inventory(cases: Iterable[tuple[str, str, str]], families: dict[str, str] | None = None) -> dict[EmittedSymbol, set[str]]:
    """Every name the generated code of *cases* takes from a package, mapped to the cases that emit it.

    Args:
        cases: `(case_id, format, code)` triples, e.g. a golden reference's id, its format and its text. A case whose format emits a language with no reader contributes nothing.
        families: `{module: family}` for Julia packages that re-export one another's names, such as the SciML solver packages that each export `ODEProblem`. An unqualified call is attributed to a family loaded by every file that makes it, named by the family; see `_resolve_unqualified`.

    Returns:
        `{symbol: {case_id, ...}}`, the join between TVBO and the frameworks it targets.
    """
    from tvbo.codegen.templater import source_language

    joined: dict[EmittedSymbol, set[str]] = defaultdict(set)
    for case_id, format, code in cases:
        reader = READERS.get(source_language(format))
        if reader is None:
            continue
        for symbol in reader(code):
            joined[symbol].add(case_id)
    return _resolve_unqualified(joined, families or {})


def _resolve_unqualified(joined: dict, families: dict[str, str]) -> dict[EmittedSymbol, set[str]]:
    """Each unqualified Julia call attributed to the packages, or package families, loaded by every file that makes it.

    A file records such a call against all the packages it loads. A package that provides the name is loaded wherever the name is called, so the intersection over the files narrows the candidates, usually to one: `ODEProblem` is called from files that load `NaNMath` and from files that do not, and every one of them loads a SciML solver package. Which solver package differs from file to file, and each re-exports `ODEProblem`, so the intersection is taken over *families*, where those packages are one.
    """
    grouped: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    resolved: dict[EmittedSymbol, set[str]] = {}
    for symbol, case_ids in joined.items():
        if symbol.kind in (CALL, KEYWORD) and "." not in symbol.name.split("(")[0]:
            loaded = {families.get(module, module) for module in symbol.package.split("|")}
            providers, cases = grouped.setdefault((symbol.name, symbol.kind), (set(loaded), set()))
            providers &= loaded
            cases |= case_ids
        else:
            resolved[symbol] = set(case_ids)
    for (name, kind), (providers, cases) in grouped.items():
        resolved[EmittedSymbol("|".join(sorted(providers)), name, kind)] = cases
    return resolved


PRESENT = "present"
MISSING = "missing"
ABSORBED = "absorbed"
UNCHECKED = "unchecked"


@dataclass(frozen=True)
class Resolution:
    """What an installed package says about one emitted name.

    `present` and `missing` are answers; `absorbed` means a keyword lands in the callee's `**kwargs`, so it names data rather than an interface; `unchecked` means the package could not be asked, because it is not installed or is not a Python package.
    """

    status: str
    version: str = ""
    deprecated: bool = False
    replaced_by: str = ""
    detail: str = ""


@functools.cache
def _distributions() -> dict[str, list[str]]:
    """`{top-level module: [distribution, ...]}` over the environment, scanned once per process."""
    import importlib.metadata

    return importlib.metadata.packages_distributions()


def _distribution_version(module: str) -> str:
    """The installed version of the distribution that provides the top-level *module*, or `""`."""
    import importlib.metadata

    for distribution in _distributions().get(module, []):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def _lookup(qualified: str):
    """The object *qualified* names, resolved as `from package import name` resolves it: an attribute first, a submodule where there is none.

    A package may hold a function and a submodule of one name, as `tvboptim...analysis.adiabatic_scan` does, and the generated code imports the function.

    Raises:
        LookupError: If the top-level module does not import, or a later part is neither an attribute nor a submodule.
    """
    import importlib

    parts = qualified.split(".")
    try:
        target = importlib.import_module(parts[0])
    except ImportError as error:
        raise LookupError(f"`{parts[0]}` does not import") from error
    for index, part in enumerate(parts[1:], start=1):
        if hasattr(target, part):
            target = getattr(target, part)
            continue
        try:
            target = importlib.import_module(".".join(parts[: index + 1]))
        except ImportError as error:
            raise LookupError(f"`{'.'.join(parts[:index])}` has no `{part}`") from error
    return target


def _documented_deprecation(function, keyword: str) -> tuple[bool, str]:
    """Whether *function*'s docstring calls *keyword* deprecated, and the replacement it names.

    Read from the Google-style `Args:` entry, `keyword: Deprecated alias for ``source``.`, the form a deprecation is documented in where the warning itself is raised only at call time.
    """
    import inspect
    import re

    entry = rf"^\s*{re.escape(keyword)}\s*(?:\([^)]*\))?\s*:\s*(.+)$"
    match = re.search(entry, inspect.getdoc(function) or "", re.MULTILINE)
    if not match or "deprecat" not in match.group(1).lower():
        return False, ""
    replacement = re.search(r"\b(?:alias for|use|replaced by)\s+``?(\w+)``?", match.group(1), re.IGNORECASE)
    return True, replacement.group(1) if replacement else ""


def resolve(symbol: EmittedSymbol) -> Resolution:
    """Ask the installed package whether *symbol* still exists, still takes the keyword, and is documented as deprecated."""
    import importlib
    import inspect

    module = symbol.package.split(".")[0]
    if "|" in symbol.package:
        return Resolution(UNCHECKED, detail=f"provided by one of {symbol.package}")
    try:
        importlib.import_module(module)
    except ImportError:
        return Resolution(UNCHECKED, detail=f"{module} does not import here: not installed, or a Julia package")
    version = _distribution_version(module)
    target, _, keyword = symbol.name.partition("(")
    try:
        found = _lookup(target)
    except LookupError as error:
        return Resolution(MISSING, version, detail=str(error))
    if not keyword:
        return Resolution(PRESENT, version, deprecated=bool(getattr(found, "__deprecated__", None)))
    keyword = keyword.rstrip("=)")
    function = found.__init__ if inspect.isclass(found) else found
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return Resolution(UNCHECKED, version, detail=f"`{target}` has no readable signature")
    if keyword in parameters:
        deprecated, replaced_by = _documented_deprecation(function, keyword)
        return Resolution(PRESENT, version, deprecated, replaced_by)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return Resolution(ABSORBED, version, detail=f"`{target}` declares no `{keyword}`; it lands in `**kwargs`")
    return Resolution(MISSING, version, detail=f"`{target}` takes no keyword `{keyword}`")
