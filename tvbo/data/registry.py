"""Canonical database registry — resolves short names to YAML file paths.

Works both from pip-installed packages and editable/dev installs.
See TVBO-Database-Access-Proposal.md §5 for design rationale.
"""
from importlib.resources import files
from pathlib import Path

# --- Path resolution ---
_pkg_db = Path(files("tvbo")) / "database"

if _pkg_db.is_dir():
    DATABASE_ROOT = _pkg_db
else:
    DATABASE_ROOT = None

# --- Category registry ---
_CATEGORIES = {
    "Dynamics":              "models",
    "Coupling":              "coupling_functions",
    "Integrator":            "integrators",
    "Network":               "networks",
    "SimulationExperiment":  "experiments",
    "SimulationStudy":       "studies",
    "Observation":           "observation_models",
    "Function":              "observation_models",
    "BrainAtlas":            "atlases",
    "Continuation":          "continuations",
}


def resolve(cls_name: str, name: str) -> Path:
    """Resolve a short name to a database YAML file path.

    Tries exact match first, then case-insensitive fallback.
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

    # Exact match
    exact = db_dir / f"{name}.yaml"
    if exact.exists():
        return exact

    # Case-insensitive fallback
    for p in db_dir.glob("*.yaml"):
        if p.stem.lower() == name.lower():
            return p

    # For networks: search by atlas name in BIDS filename
    if cls_name == "Network":
        for p in db_dir.glob("*.yaml"):
            if f"atlas-{name}" in p.stem:
                return p

    available = sorted(p.stem for p in db_dir.glob("*.yaml")
                       if not p.stem.startswith("_"))
    raise FileNotFoundError(
        f"No database entry '{name}' for {cls_name}. "
        f"Available: {available}"
    )


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
    return sorted(p.stem for p in db_dir.glob("*.yaml")
                  if not p.stem.startswith("_"))


def database_dir(cls_name: str) -> Path:
    """Return the database directory for a given entity class."""
    if DATABASE_ROOT is None:
        raise RuntimeError("tvbo database not found.")
    category = _CATEGORIES.get(cls_name)
    if category is None:
        raise ValueError(f"No database category for '{cls_name}'")
    return DATABASE_ROOT / category
