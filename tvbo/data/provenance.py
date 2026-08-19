"""What a run actually did, recorded as BEP028 provenance.

The frozen spec beside a container says what was *asked for*. These records say what *happened*: the command, when it ran, on which machine, with which package versions, and what came out. None of that exists anywhere in a recipe, which is why ``prov/`` is not a second copy of the study YAML.

BEP028 groups a set of records under one label and gives each set four kinds — ``prov-<label>_{act,ent,env,soft}``. The label here is the result's own key (``exp3``, ``anafcGradient``), so a partial re-run rewrites only its own records, and the environment record stays truthful on a cluster fan-out where two experiments land on different nodes.

One serialization per record, never two: :class:`ProvenanceFormat` chooses ``yaml`` (the default, and what every other tvbo record is) or ``json`` (what BEP028 currently names). Two files carrying one record are two things that can disagree.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from tvbo.datamodel import schema as datamodel

RECORD_KINDS = ("act", "ent", "env", "soft")
"""BEP028's four record kinds: the activity, the entity it produced, the environment, the software."""

_DIGEST = "sha256"


def now() -> str:
    """The current instant as ISO 8601 with an explicit offset, which PROV requires."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _alnum(text: str) -> str:
    return "".join(c for c in str(text) if c.isalnum())


def prov_label(produced_by: str) -> str:
    """The ``prov-<label>`` grouping key for the producer named by ``produced_by``.

    Built from the producer's own IRI scope, so a record set and the ``DataRef`` that reaches the same container agree on which thing produced it. ``tvbo:exp/<study>/exp-3`` gives ``exp3``; ``tvbo:ana/<study>/fcGradient`` gives ``anafcGradient``.
    """
    from tvbo.data.dataref import iri_scope

    kind, _study, name = iri_scope(produced_by)
    if kind is None:
        return _alnum(produced_by) or "run"
    return _alnum(name) if kind == "exp" else f"{kind}{_alnum(name)}"


def digest_of(path: Path) -> datamodel.Digest | None:
    """The container's own checksum, or ``None`` when it is not there to read."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.new(_DIGEST)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return datamodel.Digest(algorithm=_DIGEST, value=h.hexdigest())


def _packages(requires=()) -> list[datamodel.SoftwarePackage]:
    """The versions this run actually imported, read from the interpreter rather than from a lockfile.

    A pinned requirement is what was asked for; what ran is what ``importlib.metadata`` reports, which is the claim a provenance record has to be able to make. The study's own ``requires:`` are recorded alongside the framework's, so every package the study declared it needs can be compared against the one that was installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    packages = []
    for name in dict.fromkeys(("tvbo", "tvboptim", "jax", "numpy", "xarray", *requires)):
        try:
            packages.append(datamodel.SoftwarePackage(name=name, version=version(name)))
        except PackageNotFoundError:
            continue
    return packages


def _environment(label: str) -> datamodel.SoftwareEnvironment:
    """The machine and interpreter the run happened on."""
    accelerator = os.environ.get("JAX_PLATFORMS") or None
    return datamodel.SoftwareEnvironment(
        name=f"{label}-env",
        platform=platform.platform(),
        version=platform.python_version(),
        environment_type="singularity" if os.environ.get("TVBO_IN_CONTAINER") == "1" else "venv",
        description=f"accelerator: {accelerator}" if accelerator else None,
    )


def build_records(
    *,
    container: Path,
    study_root: Path,
    produced_by: str,
    outputs=(),
    used=(),
    started_at: str | None = None,
    ended_at: str | None = None,
    command: str | None = None,
    requires=(),
) -> dict[str, object]:
    """The four records describing one container, keyed by BEP028 record kind.

    Every field is a pointer or read off the artifact: the digest and the output names come from the container itself, the versions from the interpreter, the command from the invocation. Nothing restates a value the frozen spec beside the container already carries.
    """
    label = prov_label(produced_by)
    packages = _packages(requires)
    entity = datamodel.ResultEntity(
        name=label,
        container=str(Path(container).resolve().relative_to(Path(study_root).resolve()))
        if _under(container, study_root)
        else str(container),
        produced_by=produced_by,
        outputs=[str(o) for o in outputs],
        provenance=datamodel.Provenance(
            date_created=ended_at or now(),
            digest=[d for d in (digest_of(container),) if d is not None],
        ),
    )
    activity = datamodel.Activity(
        name=label,
        command=command if command is not None else _invocation(),
        associated_with=[p.name for p in packages],
        used=[str(u) for u in used],
        started_at=started_at,
        ended_at=ended_at or now(),
    )
    return {"ent": entity, "act": activity, "env": _environment(label), "soft": packages}


def _invocation() -> str:
    """The command line, with the executable named rather than located.

    ``prov/`` is tracked, so a machine-specific interpreter path would make every re-run a diff while saying nothing about what was run.
    """
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])


def _under(path, root) -> bool:
    """Whether ``path`` is inside ``root``, which decides if the record can name it relatively."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def write_records(records: dict, prov_dir: Path, label: str, fmt: str = "yaml") -> list[Path]:
    """Write each record as ``prov-<label>_<kind>.<fmt>``, returning the paths written.

    Serialized through the LinkML dumpers, so what lands on disk is schema-valid by construction rather than by a hand-written mapping that has to be kept in step with the classes.
    """
    from linkml_runtime.dumpers import json_dumper, yaml_dumper

    fmt = str(fmt).lower()
    if fmt not in ("yaml", "json"):
        raise ValueError(f"provenance_format must be 'yaml' or 'json', not {fmt!r}")
    dumper = yaml_dumper if fmt == "yaml" else json_dumper
    prov_dir = Path(prov_dir)
    prov_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for kind in RECORD_KINDS:
        record = records.get(kind)
        if record is None:
            continue
        path = prov_dir / f"prov-{label}_{kind}.{fmt}"
        text = dumper.dumps(record) if not isinstance(record, list) else dumper.dumps({"packages": record})
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        written.append(path)
    return written


def emit(
    *,
    container: Path,
    study_root: Path,
    produced_by: str,
    outputs=(),
    used=(),
    started_at: str | None = None,
    ended_at: str | None = None,
    command: str | None = None,
    requires=(),
    fmt: str = "yaml",
) -> list[Path]:
    """Describe one container's run in ``<study_root>/prov/``, returning the records written.

    Callers gate on the study's ``emit_provenance``; this writes unconditionally so the decision is made once, where the spec is in hand.
    """
    from tvbo.utils.study_layout import study_path

    records = build_records(
        container=container,
        study_root=study_root,
        produced_by=produced_by,
        outputs=outputs,
        used=used,
        started_at=started_at,
        ended_at=ended_at,
        command=command,
        requires=requires,
    )
    return write_records(records, study_path("provenance", root=study_root), prov_label(produced_by), fmt)
