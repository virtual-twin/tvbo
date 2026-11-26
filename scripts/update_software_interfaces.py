"""Write the public names TVBO's generated code reaches into each software entry's `interfaces:`.

Reads every golden reference under `tests/reference_data/codegen/` and every curated experiment rendered in each code format the export registry declares, takes the inventory of what that code imports, calls, subclasses and passes as keywords (`tvbo.codegen.inventory`), and asks the installed package about each name (`resolve`). A keyword the callee absorbs in `**kwargs` is model data, not an interface, and is left out.

A name is filed under the `SimulationTool` whose `modules:` lists its top-level module; the names of packages with no entry are printed instead. The block is regenerated whole and written last in the file, so the curated part of an entry is never reformatted.

Run from the repository root, in the environment whose package versions the entries should record: `python scripts/update_software_interfaces.py`.
"""

from __future__ import annotations

import collections
import pathlib
import warnings

import yaml

from tvbo.codegen.inventory import (
    ABSORBED,
    BASE_CLASS,
    READERS,
    EmittedSymbol,
    corpus_cases,
    inventory,
    module_families,
    resolve,
    software_entries,
)
from tvbo.data.registry import database_dir

CORPUS = pathlib.Path("tests/reference_data/codegen")
REALISED_BY_MODEL_BASE = "tvbo:Dynamics"


def cases():
    """`(case_id, format, code)` for every golden reference and every curated experiment a code format renders."""
    from tvbo.classes.experiment import SimulationExperiment
    from tvbo.export import registry

    yield from corpus_cases(CORPUS)
    formats = [fmt.key for fmt in registry.list_formats() if fmt.language in READERS]
    for path in sorted(database_dir("SimulationExperiment").rglob("*.yaml")):
        experiment = SimulationExperiment.from_file(str(path))
        for fmt in formats:
            try:
                code = experiment.render(fmt)
            except Exception:  # noqa: BLE001 — a format that refuses an experiment contributes nothing
                continue
            yield f"experiment:{path.stem}.{fmt}", fmt, code


def interface_entry(symbol: EmittedSymbol, case_ids: set[str], resolution, kinds: list[str]) -> dict:
    """One `interfaces:` entry, stating only what is known."""
    entry: dict = {"kind": kinds}
    if resolution.version:
        entry["checked_version"] = resolution.version
    if resolution.deprecated:
        entry["deprecated"] = True
    if resolution.replaced_by:
        entry["replaced_by"] = resolution.replaced_by
    if symbol.kind == BASE_CLASS and any(not case.startswith("experiment:") for case in case_ids):
        entry["realises"] = [REALISED_BY_MODEL_BASE]
    return entry


def write_interfaces(path: pathlib.Path, interfaces: dict) -> None:
    """Replace the `interfaces:` block that ends *path*, or append one."""
    text = path.read_text(encoding="utf-8")
    text = text.split("\ninterfaces:\n")[0].rstrip("\n") + "\n"
    if interfaces:
        block = yaml.safe_dump({"interfaces": interfaces}, sort_keys=False, width=10_000, allow_unicode=True)
        text += block
    path.write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    entries = software_entries()
    tools = {module: path for path, record in entries for module in record.get("modules") or []}
    joined = inventory(cases(), module_families(entries))
    by_tool: dict[pathlib.Path, dict] = collections.defaultdict(dict)
    unattributed = collections.Counter()
    for symbol in sorted(joined):
        module = symbol.package.split(".")[0]
        if module not in tools:
            unattributed[symbol.package] += 1
            continue
        resolution = resolve(symbol)
        if resolution.status == ABSORBED:
            continue
        entries = by_tool[tools[module]]
        earlier = entries.get(symbol.name, {})
        kinds = sorted({symbol.kind, *earlier.get("kind", [])})
        entries[symbol.name] = interface_entry(symbol, joined[symbol], resolution, kinds)
        if "realises" in earlier:
            entries[symbol.name]["realises"] = earlier["realises"]  # a name's base-class use outlives its later import or call
    for path in sorted(set(tools.values())):
        write_interfaces(path, by_tool.get(path, {}))
        print(f"{path.name}: {len(by_tool.get(path, {}))} interfaces")
    for package, count in sorted(unattributed.items()):
        print(f"no software entry lists `{package}`: {count} names")


if __name__ == "__main__":
    main()
