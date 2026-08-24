"""Canonical database registry — resolves short names to YAML file paths.

Works both from pip-installed packages and editable/dev installs.
Searches recursively within each category directory so that models in subfolders (e.g. database/models/julia/) are automatically discovered.
"""

import re
from importlib.resources import files
from pathlib import Path

_NAME_RE = re.compile(r"^name:\s*(\S.*)")

# --- Path resolution ---
_pkg_db = Path(files("tvbo")) / "database"

if _pkg_db.is_dir():
    DATABASE_ROOT = _pkg_db
else:
    DATABASE_ROOT = None

# --- Category registry ---
_CATEGORIES = {
    "Dynamics": "models",
    "Coupling": "coupling_functions",
    "Integrator": "integrators",
    "Network": "networks",
    "SimulationExperiment": "experiments",
    # A SimulationStudy is_a Study, so both resolve to the bibliographic studies/ directory.
    "Study": "studies",
    "SimulationStudy": "studies",
    "Observation": "observation_models",
    "Function": "observation_models",
    "BrainAtlas": "atlases",
    "Continuation": "continuations",
    "GraphGenerator": "graph_generators",
    "SimulationTool": "software",
}


def local_name(iri: str) -> str:
    """Strip an optional ``prefix:`` from a CURIE / IRI, returning the local name.

    ``tvbo:KuramotoCoupling`` -> ``KuramotoCoupling``; a bare name is returned unchanged. Single source of truth for CURIE-prefix stripping, so the class-layer ``_iri_local`` helpers don't each re-implement it.
    """
    return iri.split(":", 1)[-1] if ":" in iri else iri


def resolve(cls_name: str, name: str) -> Path:
    """Resolve a short name to a database YAML file path.

    Tries exact top-level stem match first (fast path), then searches recursively by canonical `name:` field and file stem (case-insensitive).
    For Network, also matches BIDS filenames containing the atlas name.
    """
    if DATABASE_ROOT is None:
        raise RuntimeError(
            "tvbo database not found. If you installed tvbo via pip, "
            "this may indicate a packaging issue. If running from source, "
            "make sure you're in the tvbo repository root."
        )

    category = _CATEGORIES.get(cls_name)
    if category is None:
        raise ValueError(f"No database category for '{cls_name}'")

    db_dir = DATABASE_ROOT / category

    # Fast path: exact top-level stem match
    exact = db_dir / f"{name}.yaml"
    if exact.exists():
        return exact

    # Build recursive name index and search
    name_lower = name.lower()
    for canonical, p in _build_name_index(db_dir).items():
        if canonical == name or canonical.lower() == name_lower:
            return p
        if p.stem == name or p.stem.lower() == name_lower:
            return p

    # For networks: search by atlas name in BIDS filename
    if cls_name == "Network":
        for p in db_dir.rglob("*.yaml"):
            if f"atlas-{name}" in p.stem:
                return p

    available = sorted(_build_name_index(db_dir).keys())
    raise FileNotFoundError(f"No database entry '{name}' for {cls_name}. Available: {available}")


def _extract_name(path: Path) -> str:
    """Read the canonical `name:` value from the first lines of a YAML file."""
    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(15):
                line = fh.readline()
                if not line:
                    break
                m = _NAME_RE.match(line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return path.stem  # fallback: use filename stem


def _build_name_index(db_dir: Path) -> dict[str, Path]:
    """Map canonical model name -> YAML path for all files under db_dir (recursive)."""
    index: dict[str, Path] = {}
    for p in sorted(db_dir.rglob("*.yaml")):
        if p.stem.startswith("_"):
            continue
        name = _extract_name(p)
        index[name] = p
    return index


def list_entries(cls_name: str) -> list[str]:
    """List all available database entries for a class."""
    if DATABASE_ROOT is None:
        return []
    category = _CATEGORIES.get(cls_name)
    if category is None:
        return []
    db_dir = DATABASE_ROOT / category
    if not db_dir.exists():
        return []
    return sorted(_build_name_index(db_dir).keys())


def list_entries_with_metadata(cls_name: str) -> list[dict]:
    """Every entry in the given class category, as dicts with `name`, `model_type`, `description`, `path`.

    Fast — reads only the first ~30 lines of each YAML file.
    """
    if DATABASE_ROOT is None:
        return []
    category = _CATEGORIES.get(cls_name)
    if category is None:
        return []
    db_dir = DATABASE_ROOT / category
    if not db_dir.exists():
        return []

    _FIELD_RE = re.compile(r"^(model_type|description|system_type):\s*(.*)")
    rows = []
    for p in sorted(db_dir.rglob("*.yaml")):
        if p.stem.startswith("_"):
            continue
        fields: dict = {"path": p}
        try:
            with p.open(encoding="utf-8") as fh:
                for _ in range(30):
                    line = fh.readline()
                    if not line:
                        break
                    m = _NAME_RE.match(line)
                    if m and "name" not in fields:
                        fields["name"] = m.group(1).strip()
                        continue
                    m2 = _FIELD_RE.match(line)
                    if m2:
                        fields[m2.group(1)] = m2.group(2).strip()
        except OSError:
            continue
        if "name" not in fields:
            fields["name"] = p.stem
        rows.append(fields)
    return rows


def database_dir(cls_name: str) -> Path:
    """Return the database directory for a given entity class."""
    if DATABASE_ROOT is None:
        raise RuntimeError("tvbo database not found.")
    category = _CATEGORIES.get(cls_name)
    if category is None:
        raise ValueError(f"No database category for '{cls_name}'")
    return DATABASE_ROOT / category
