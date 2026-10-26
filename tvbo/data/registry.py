"""Canonical database registry — resolves short names to YAML file paths.

Works both from pip-installed packages and editable/dev installs.
Searches recursively within each category directory so that models in subfolders (e.g. database/models/julia/) are automatically discovered.
"""

import os
import re
import warnings
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
    "Function": "functions",
    "BrainAtlas": "atlases",
    "Continuation": "continuations",
    "GraphGenerator": "graph_generators",
    "SimulationTool": "software",
    # A Theme is_a Palette, so a document declaring either resolves to the same directory.
    "Theme": "themes",
    "Palette": "themes",
}


# The IRI scope a curated record is addressed under is its database folder: `tvbo:<folder>/<name>` is the record `https://w3id.org/tvbo/<folder>/<name>`. A folder's scope names the first class filed in it. A container a run produced is addressed under `result/` and resolves against a study's results directory instead, so a recipe and its output never share a scope.
_IRI_SCOPES = {folder: cls_name for cls_name, folder in reversed(_CATEGORIES.items())}

# The singular scopes curated IRIs were written under before they took their folder's name, still read as CURIEs with a FutureWarning that names the folder form.
_LEGACY_SCOPES = {
    "experiment": "experiments",
    "study": "studies",
    "dynamics": "models",
    "network": "networks",
    "atlas": "atlases",
    "coupling": "coupling_functions",
    "integrator": "integrators",
    "observation": "observation_models",
    "function": "functions",
    "continuation": "continuations",
    "graph_generator": "graph_generators",
    "theme": "themes",
}

_IRI_RE = re.compile(r"^(?P<prefix>[A-Za-z][\w.-]*):(?P<scope>[a-z_]+)/(?P<name>[^/].*)$")

NAMESPACE = "https://w3id.org/tvbo/"


def _curie(iri: str) -> str:
    """*iri* as a ``tvbo:`` CURIE when it is written out in full under :data:`NAMESPACE`, else as given (stripped)."""
    iri = str(iri).strip()
    return "tvbo:" + iri[len(NAMESPACE) :] if iri.startswith(NAMESPACE) else iri


def _scoped(iri: str) -> tuple[str, str] | None:
    """``(folder, name)`` for an IRI scoped to a curated folder, or ``None`` when it carries no such scope.

    The ``prefix:`` is required, and that is what separates an IRI from a relative path: without it ``networks/Glasser.yaml`` reads as scope ``networks``, and a path the caller meant to open would be resolved against the database instead. A retired singular scope is read only as a CURIE, the one form it was ever written in: in full, ``https://w3id.org/tvbo/study/`` is the study schema module's namespace, not a folder. Its warning is a ``FutureWarning`` because the person who wrote it is a spec's author, whom Python's default filters hide a ``DeprecationWarning`` from.
    """
    curie = _curie(iri)
    m = _IRI_RE.match(curie)
    if not m:
        return None
    folder, name = m.group("scope"), m.group("name")
    if folder in _LEGACY_SCOPES and curie == str(iri).strip():
        folder = _LEGACY_SCOPES[folder]
        warnings.warn(
            f"{iri!r} uses the retired scope `{m.group('scope')}/`; write `{m.group('prefix')}:{folder}/{name}`.",
            FutureWarning,
            stacklevel=3,
        )
    return (folder, name) if folder in _IRI_SCOPES else None


def local_name(iri: str) -> str:
    """The name an IRI ends in: its ``prefix:`` stripped, and its folder scope when it names a curated record.

    ``tvbo:KuramotoCoupling``, ``tvbo:coupling_functions/KuramotoCoupling`` and ``https://w3id.org/tvbo/coupling_functions/KuramotoCoupling`` -> ``KuramotoCoupling``; a bare name is returned unchanged, and so is any path a curated scope does not begin (``tvbo:result/S/fig`` -> ``result/S/fig``).
    """
    scoped = _scoped(iri)
    if scoped:
        return scoped[1]
    iri = _curie(iri)
    return iri.split(":", 1)[-1] if ":" in iri else iri


def record_iri(folder: str, path: Path) -> str:
    """The canonical IRI of the record *path* files in the database *folder*: ``tvbo:<folder>/<file stem>``.

    The stem is unique in its folder however deep the file sits, where a ``name:`` need not be (twenty atlases are all ``Schaefer2018``), so it is the one local name every record has exactly once. The ontology mints each record at this IRI, and :mod:`tvbo.data.iri_migration` rewrites every other spelling of it into this one.
    """
    return f"tvbo:{folder}/{path.stem}"


def iri_for(cls_name: str, name: str) -> str:
    """The canonical IRI of the curated *cls_name* record *name* resolves to, by name or file stem: ``iri_for("Dynamics", "ModelJansen1995")`` -> ``tvbo:models/Jansen1995``.

    Raises:
        FileNotFoundError: when no *cls_name* record answers to *name*.
    """
    return record_iri(_CATEGORIES[cls_name], resolve(cls_name, name))


def iri_target(iri: str) -> tuple[str, str] | None:
    """``(class_name, local_name)`` for a curated-record IRI, or ``None`` when it is not one.

    ``tvbo:experiments/JR_MEG_FrequencyGradient_Optimization`` -> ``("SimulationExperiment", "JR_MEG_FrequencyGradient_Optimization")``. Returns ``None`` for an unscoped name, for a ``result/`` reference (a produced container, which :mod:`tvbo.data.dataref` resolves against a study's results directory), and for any other scope, so a caller falls back rather than being handed the wrong entity.
    """
    scoped = _scoped(iri)
    return (_IRI_SCOPES[scoped[0]], scoped[1]) if scoped else None


def curated_key(iri: str, cls_name: str) -> str | None:
    """The name of the curated *cls_name* record *iri* addresses, or ``None`` when it addresses none.

    A scoped IRI (``tvbo:observation_models/bold_tvb``, or the same written out under :data:`NAMESPACE`) addresses a record in its folder, so it names a *cls_name* record only when that class is filed there. An unscoped one (``tvbo:BOLD_TVB``, or a bare ``BOLD_TVB``) is a shortcut the class makes unambiguous: it names the *cls_name* record of that local name. A ``result/`` reference or any other scope addresses no curated record.
    """
    if not _IRI_RE.match(_curie(iri)):
        return local_name(iri)
    target = _scoped(iri)
    if target is None or target[0] != _CATEGORIES.get(cls_name):
        return None
    return target[1]


def record_name(iri: str, cls_name: str | None = None) -> str:
    """The ``name:`` of the curated *cls_name* record *iri* addresses, else the name *iri* ends in.

    A record's IRI ends in its file stem, which is not always its name: ``tvbo:atlases/tpl-MNI152NLin2009c_atlas-DesikanKilliany_desc-ranked_dseg`` is the atlas ``DesikanKilliany``. An IRI that addresses no curated *cls_name* record, or a *cls_name* with no database folder, is named by its :func:`local_name`.
    """
    key = curated_key(iri, cls_name) if cls_name in _CATEGORIES else None
    if key is None:
        return local_name(iri)
    try:
        return _extract_name(resolve(cls_name, key))
    except FileNotFoundError:
        return local_name(iri)


def resolve_iri(iri: str) -> Path:
    """Resolve a curated-record IRI to its database YAML path.

    New specs reference curated entities this way because a relative path is not portable: ``!include`` splices a document without rebasing the paths inside it, so an experiment included out of the database resolves its own ``network.bids_dir`` against the including file and loads the wrong root. An IRI carries no path, so the database resolves it against the database.
    """
    target = iri_target(iri)
    if target is None:
        raise ValueError(
            f"{iri!r} is not a curated-record IRI. Expected `tvbo:<folder>/<name>` with folder one of: "
            f"{', '.join(sorted(_IRI_SCOPES))}."
        )
    return resolve(*target)


def resolve(cls_name: str, name: str) -> Path:
    """Resolve a short name to a database YAML file path.

    Tries exact top-level stem match first (fast path), then searches recursively by canonical `name:` field and file stem (case-insensitive).
    For Network, also matches BIDS filenames containing the atlas name, taking the first in sorted order when several do: for ``DesikanKilliany`` that is the HCPYA structural connectome, not the MEG or EEG projection networks filed under the same atlas.
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

    # Fast path: a top-level file of exactly this stem, read off the listing because a case-insensitive filesystem also finds `BOLD_TVB.yaml` as `bold_tvb.yaml`.
    exact = db_dir / f"{name}.yaml"
    if exact.exists() and exact.name in os.listdir(db_dir):
        return exact

    # Build recursive name index and search
    name_lower = name.lower()
    for canonical, p in _build_name_index(db_dir).items():
        if canonical == name or canonical.lower() == name_lower:
            return p
        if p.stem == name or p.stem.lower() == name_lower:
            return p

    # Sorted: an atlas several networks share must resolve to the same file whatever order the filesystem lists them in.
    if cls_name == "Network":
        for p in sorted(db_dir.rglob("*.yaml")):
            if f"atlas-{name}" in p.stem:
                return p

    available = sorted(_build_name_index(db_dir).keys())
    raise FileNotFoundError(f"No database entry '{name}' for {cls_name}. Available: {available}")


def _extract_name(path: Path) -> str:
    """Read the canonical `name:` value from the first 15 content lines of a YAML file, so a header of comments and blank lines does not push it out of reach."""
    try:
        with path.open(encoding="utf-8") as fh:
            content = 0
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                m = _NAME_RE.match(line)
                if m:
                    return m.group(1).strip()
                content += 1
                if content == 15:
                    break
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
