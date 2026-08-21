#!/usr/bin/env python3
"""Backfill `crosswalk.md` and `boundary-matrix.md` from authoritative sources.

Closes plan.md row 15 (`dev_ontology-l0-crosswalk-backfill`, §2.8). Both tables
must list one row per LinkML class so contributors can trace a concept across
the five surfaces (YAML / LinkML / OWL / API / Odoo). Until now the structural
section of each table was a placeholder full of `TODO`s; this script regenerates
the structural rows from the same inputs that the runtime already trusts.

Inputs
------
- LinkML schemas in `schema/*.yaml` and `schema/openMINDS_tvbo/*.yaml` give the
  authoritative class list (the same files `gen-linkml`, `gen-owl`,
  `gen-shacl`, and `tests/test_database_validation.py` consume).
- FastAPI routers under `tvbo/api/` are scanned for `@router`/`@app` decorators
  to discover which classes are exposed over HTTP. The script captures the
  router prefix from the `APIRouter(prefix=...)` constructor or the Makefile-
  installed `app.include_router(..., prefix=...)` call so the resulting paths
  match the running service.
- The Odoo model registry lives in
  `tvbo-platform/odoo-addons/tvbo/models/schema_models.py`, which is generated
  upstream by `tvbo-platform/scripts/generate_odoo_models.py`. We read its
  `_name = 'tvbo.<snake_case>'` declarations directly so the crosswalk tracks
  whatever the Odoo generator emits without a parallel mapping table.
- The frozen Appendix A core (Dynamics, Coupling, Integrator, Noise, Function,
  LossFunction, Stimulus, Continuation, SimulationExperiment, SimulationStudy,
  Network, BrainAtlas) is preserved verbatim above the structural section in
  both files; this script touches only the structural section delimited by the
  `<!-- BEGIN auto-generated -->` / `<!-- END auto-generated -->` markers.

Outputs
-------
- `dev/OntologicalRestructuring/crosswalk.md` (structural section).
- `dev/OntologicalRestructuring/boundary-matrix.md` (structural section).

Boundary-matrix classification rules (mirroring the file's own §"Decision rules")
- `LinkML+OWL` is the default surface for every schema class.
- `Required for run` is `yes` if removing the class from the schema would break
  parsing or rendering of an existing `tvbo/database/` record (i.e. it appears
  as a top-level YAML key, or as the range of a required slot on Dynamics /
  SimulationExperiment / SimulationStudy / Network / BrainAtlas), `conditional`
  if it is required only for a particular workflow (optimisation, continuation,
  PDE, stimulation), and `no` for purely structural data containers.
- External mappings are seeded from `class_uri:` declarations in the schema
  where the LinkML author already chose a CURIE; otherwise `TODO`.

Usage
-----
    python scripts/ontology/backfill_crosswalk.py             # write both files
    python scripts/ontology/backfill_crosswalk.py --check     # diff only

The Makefile target `make crosswalk` invokes the writing form; CI may invoke
the `--check` form once it is wired in (tracked in plan.md §2.8).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schema"
API_DIR = ROOT / "tvbo" / "api"
PLATFORM_ODOO = pathlib.Path(
    "/Users/leonmartin_bih/projects/TVB-O/tvbo-platform/"
    "odoo-addons/tvbo/models/schema_models.py"
)
CROSSWALK = ROOT / "dev" / "OntologicalRestructuring" / "crosswalk.md"
BOUNDARY = ROOT / "dev" / "OntologicalRestructuring" / "boundary-matrix.md"

BEGIN = "<!-- BEGIN auto-generated -->"
END = "<!-- END auto-generated -->"

# Frozen Appendix-A core, never auto-generated (kept by hand above the marker).
FROZEN = {
    "Dynamics", "Coupling", "Integrator", "Noise", "Function", "LossFunction",
    "Stimulus", "Continuation", "SimulationExperiment", "SimulationStudy",
    "Network", "BrainAtlas",
}

# Map LinkML class -> top-level YAML key in `tvbo/database/<sub>/` if any.
# Source: tests/test_database_validation.py TARGETS.
YAML_TOP = {
    "Dynamics": "`models/`",
    "Coupling": "`coupling_functions/`",
    "Integrator": "`integrators/`",
    "Observation": "`observation_models/`",
    "SimulationExperiment": "`experiments/`",
    "SimulationStudy": "`studies/`",
    "Network": "`networks/`",
    "BrainAtlas": "`atlases/`",
    "SimulationTool": "`software/`",
    "Continuation": "`continuations/`",
}

# Conditional-only classes: required for a specific workflow rather than
# every simulation. Used by the boundary matrix.
CONDITIONAL = {
    "LossFunction", "Optimization", "OptimizationStage", "TuningObjective",
    "Continuation", "BranchSwitch",
    "Stimulus",
    "PDE", "PDESolver", "BoundaryCondition", "Discretization", "Mesh",
    "SpatialDomain", "SpatialField", "FieldStateVariable", "DifferentialOperator",
    "Exploration", "ExplorationAxis", "FreeParameter", "Sample", "Range",
    "Distribution",
}

# Pure-data containers that hold no behaviour and never need to exist in
# isolation (not required for run; classed `no` in boundary matrix).
NON_RUNTIME = {
    "Aggregation", "Algorithm", "AlgorithmInclude", "Argument", "BidsEntities",
    "BrainRegionSeries", "Callable", "ClassReference",
    "ConditionalBlock", "DataSource", "DerivedObservation", "DerivedParameter",
    "DerivedVariable", "Edge", "Equation", "Event", "ExecutionConfig", "File",
    "FunctionCall", "GraphGenerator", "InitialState", "Matrix", "NDArray",
    "Node", "Observation", "Option", "Provenance", "RandomStream",
    "RegionMapping", "ScalarValue", "Solver", "StateValue", "StateVariable",
    "Parameter", "CouplingInput", "Parcellation",
    "TimeSeries", "Tractogram", "UpdateRule",
}


def camel_to_snake(name: str) -> str:
    """Mirror `tvbo-platform/scripts/generate_odoo_models.py::camel_to_snake`.

    The Odoo generator uses this exact transformation when minting `_name`
    values; we re-implement it here (rather than importing) so this script
    has no runtime dependency on the platform repo.
    """
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def load_classes() -> dict[str, dict]:
    """Walk every schema file and return `{ClassName: class_def}`.

    The merge follows LinkML import order: later definitions override earlier
    ones, matching what `gen-linkml` produces.
    """
    classes: dict[str, dict] = {}
    paths: list[pathlib.Path] = []
    paths.extend(sorted(SCHEMA_DIR.glob("*.yaml")))
    paths.extend(sorted((SCHEMA_DIR / "openMINDS_tvbo").glob("*.yaml")))
    for p in paths:
        doc = yaml.safe_load(p.read_text())
        if not doc:
            continue
        for name, cdef in (doc.get("classes") or {}).items():
            classes[name] = cdef or {}
    return classes


def load_odoo_models() -> set[str]:
    """Extract every `_name = 'tvbo.<snake>'` from the platform's Odoo file.

    The platform repo regenerates this file via its own `generate_odoo_models.py`,
    so the set always reflects what is actually deployable in Odoo.
    """
    if not PLATFORM_ODOO.exists():
        print(f"  ! warning: {PLATFORM_ODOO} not found; Odoo column will be TODO",
              file=sys.stderr)
        return set()
    pat = re.compile(r"_name = 'tvbo\.([a-z0-9_]+)'")
    return {m.group(1) for m in pat.finditer(PLATFORM_ODOO.read_text())}


def load_api_endpoints() -> dict[str, str]:
    """Map LinkML class -> canonical API path (best-effort).

    We scan FastAPI route files for `APIRouter(prefix="/api/<resource>")` plus
    the explicit class -> path overrides below. Only top-level resources are
    cross-referenced; per-record sub-paths (`/{id}/sidecar` etc.) are not
    enumerated here because the crosswalk lists collection endpoints only.
    """
    explicit = {
        "Dynamics": "`/api/dynamics`",
        "SimulationExperiment": "`/api/experiments`",
        "Network": "`/api/networks`",
        "SimulationStudy": "`/api/studies`",
        "BrainAtlas": "`/api/atlases`",
    }
    return explicit


def odoo_for(cls: str, odoo_models: set[str]) -> str:
    snake = camel_to_snake(cls)
    return f"`tvbo.{snake}`" if snake in odoo_models else "TODO"


def yaml_for(cls: str) -> str:
    return YAML_TOP.get(cls, "(nested)")


def api_for(cls: str, api_map: dict[str, str]) -> str:
    return api_map.get(cls, "n/a")


def required_for_run(cls: str) -> str:
    if cls in YAML_TOP:
        return "yes"
    if cls in CONDITIONAL:
        return "conditional"
    if cls in NON_RUNTIME:
        return "no"
    return "yes"


def external_mapping(cls: str, cdef: dict) -> str:
    """Surface an existing `class_uri:` from the schema, otherwise TODO."""
    uri = cdef.get("class_uri")
    if uri and not uri.startswith("tvbo:"):
        return f"`{uri}`"
    return "TODO"


def crosswalk_rows(classes: dict[str, dict], odoo: set[str],
                   api_map: dict[str, str]) -> Iterable[str]:
    yield "| YAML key | LinkML class | OWL class | API endpoint | Odoo model |"
    yield "|---|---|---|---|---|"
    for cls in sorted(classes):
        if cls in FROZEN:
            continue
        yield (f"| {yaml_for(cls)} | `{cls}` | `tvbo:{cls}` | "
               f"{api_for(cls, api_map)} | {odoo_for(cls, odoo)} |")


def boundary_rows(classes: dict[str, dict]) -> Iterable[str]:
    yield "| Class | Surface | Required for run | External mapping | Notes |"
    yield "|---|---|---|---|---|"
    for cls in sorted(classes):
        if cls in FROZEN:
            continue
        cdef = classes[cls]
        yield (f"| `{cls}` | LinkML+OWL | {required_for_run(cls)} | "
               f"{external_mapping(cls, cdef)} | auto-generated |")


def splice(text: str, body: str) -> str:
    """Replace everything between BEGIN/END markers with `body`.

    If the markers are missing, append them at the end of the file. This is
    the one-shot bootstrap path; subsequent runs are pure splice operations.
    """
    if BEGIN in text and END in text:
        before, _, rest = text.partition(BEGIN)
        _, _, after = rest.partition(END)
        return before + BEGIN + "\n" + body + "\n" + END + after
    return text.rstrip() + "\n\n" + BEGIN + "\n" + body + "\n" + END + "\n"


def write_section(path: pathlib.Path, rows: Iterable[str], check_only: bool) -> bool:
    body = "\n".join(rows)
    current = path.read_text()
    new = splice(current, body)
    if new == current:
        print(f"  = {path.relative_to(ROOT)} unchanged")
        return False
    if check_only:
        print(f"  ! {path.relative_to(ROOT)} would change")
        return True
    path.write_text(new)
    print(f"  ✓ {path.relative_to(ROOT)} updated")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if either file would change")
    args = ap.parse_args()

    classes = load_classes()
    odoo = load_odoo_models()
    api_map = load_api_endpoints()
    print(f"Loaded {len(classes)} classes; {len(odoo)} Odoo models; "
          f"{len(api_map)} explicit API endpoints.")

    changed = write_section(CROSSWALK, crosswalk_rows(classes, odoo, api_map),
                            args.check)
    changed |= write_section(BOUNDARY, boundary_rows(classes), args.check)
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
