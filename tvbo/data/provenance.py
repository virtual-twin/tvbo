"""What a run actually did, recorded beside the container it produced.

The frozen spec beside a container says what was *asked for*. This says what *happened*: the command, when it ran, on which machine, with which package versions, and the checksum of what came out. None of that exists anywhere in a recipe, which is why it is recorded at all.

It goes into the container's own YAML sidecar, under the shared ``provenance`` slot, because that file already carries the other half of the story — the recipe, and the ``used:`` edges saying what this result was derived from. BEP028 spreads the same content over four files per container (``prov-<label>_{act,ent,env,soft}``) in a parallel ``prov/`` tree; one ``Provenance`` on the sidecar says it once, in the file a reader already has open, and cannot fall out of step with the spec it describes. ``activities`` is the run, ``environment`` the machine and the resolved package set, ``digest`` the artifact's checksum.

The sidecar is a product of the run, so these machine-specific facts are written unconditionally and never tracked — no flag decides whether a run remembers what it did.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from tvbo.datamodel import schema as datamodel

RUN_AUTHORED = ("date_created", "generated_by", "outputs", "digest", "activities", "environment")
"""The slots a run speaks for, cleared before its record is merged into whatever the sidecar already said. A digest is a claim about bytes: a re-run that wrote no container must leave none rather than inherit the one that last matched."""

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


def _requirements(requires=()) -> list[datamodel.SoftwareRequirement]:
    """The versions this run actually imported, read from the interpreter rather than from a lockfile.

    A pinned requirement is what was asked for; what ran is what ``importlib.metadata`` reports, which is the claim a provenance record has to be able to make. The study's own ``requires:`` are recorded alongside the framework's, so every package the study declared it needs can be compared against the one that was installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    found = []
    for name in dict.fromkeys(("tvbo", "tvboptim", "jax", "numpy", "xarray", *requires)):
        try:
            found.append(datamodel.SoftwareRequirement(name=name, version=version(name)))
        except PackageNotFoundError:
            continue
    return found


def _environment(label: str, requires=()) -> datamodel.SoftwareEnvironment:
    """The machine, the interpreter and the package set the run actually loaded."""
    accelerator = os.environ.get("JAX_PLATFORMS") or None
    return datamodel.SoftwareEnvironment(
        name=f"{label}-env",
        platform=platform.platform(),
        version=platform.python_version(),
        environment_type="singularity" if os.environ.get("TVBO_IN_CONTAINER") == "1" else "venv",
        description=f"accelerator: {accelerator}" if accelerator else None,
        requirements=_requirements(requires),
    )


def build_provenance(
    *,
    container: Path,
    produced_by: str,
    outputs=(),
    used=(),
    started_at: str | None = None,
    ended_at: str | None = None,
    command: str | None = None,
    requires=(),
) -> datamodel.Provenance:
    """One container's run, as the ``Provenance`` its sidecar carries.

    Every field is a pointer or read off the artifact: the digest and the output names come from the container itself, the versions from the interpreter, the command from the invocation. Nothing restates a value the frozen spec around it already carries — which is why the container path is not here either, the sidecar being the container's own.

    BEP028's four records collapse into this one object: its ``activities`` entry is the ``act`` record, ``environment`` carries ``env`` and ``soft`` together as the machine plus one requirement per package, and ``digest`` with ``outputs`` is what ``ent`` said that the spec does not.
    """
    label = prov_label(produced_by)
    ended = ended_at or now()
    environment = _environment(label, requires)
    activity = datamodel.Activity(
        name=label,
        iri=produced_by,
        command=command if command is not None else _invocation(),
        associated_with=[r.name for r in (environment.requirements or [])],
        used=[str(u) for u in used],
        started_at=started_at,
        ended_at=ended,
    )
    return datamodel.Provenance(
        date_created=ended,
        generated_by="tvbo",
        outputs=[str(o) for o in outputs],
        digest=[d for d in (digest_of(container),) if d is not None],
        activities=[activity],
        environment=environment,
    )


def _invocation() -> str:
    """The command line, with the executable named rather than located.

    Two runs of the same study are compared through these records, so a machine-specific interpreter path would differ between them while saying nothing about what was run.
    """
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])


def _under(path, root) -> bool:
    """Whether ``path`` is inside ``root``, which decides if a reference can name it relatively."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _as_mapping(prov: datamodel.Provenance) -> dict:
    """The record as plain data, through the LinkML dumper so what lands on disk is schema-valid by construction and carries no empty slots."""
    import yaml
    from linkml_runtime.dumpers import yaml_dumper

    return yaml.safe_load(yaml_dumper.dumps(prov)) or {}


def emit(
    *,
    container: Path,
    produced_by: str,
    outputs=(),
    used=(),
    started_at: str | None = None,
    ended_at: str | None = None,
    command: str | None = None,
    requires=(),
) -> Path:
    """Record this run under ``provenance:`` in the container's own sidecar, returning the file written.

    The sidecar already holds the recipe and its ``used:`` edges, so writing here puts what happened beside what was asked for in the one file a reader opens. It is merged rather than replaced, at both levels: the frozen spec is the rest of the document, and the ``provenance`` block itself may already carry assertions this run does not make — ``experiment_yaml_hash`` and the input fingerprints the cross-experiment cache reads — which a wholesale overwrite would silently drop. The slots the run does speak for (:data:`RUN_AUTHORED`) are replaced and not merged, so a claim the previous run made about bytes this one did not write cannot survive. A container written without a sidecar still gets one, because a run that recorded nothing about itself is the case these records exist for.
    """
    import yaml

    from tvbo.data.dataref import sidecar_path

    record = build_provenance(
        container=container,
        produced_by=produced_by,
        outputs=outputs,
        used=used,
        started_at=started_at,
        ended_at=ended_at,
        command=command,
        requires=requires,
    )
    path = sidecar_path(container)
    document = {}
    if path.exists():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"sidecar {path.name} is not a mapping, so a run cannot record itself in it")
    existing = document.get("provenance") if isinstance(document.get("provenance"), dict) else {}
    kept = {k: v for k, v in existing.items() if k not in RUN_AUTHORED}
    document["provenance"] = {**kept, **_as_mapping(record)}
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def input_containers(refs, *, results_root, study_root) -> list[str]:
    """The containers a set of ``used:`` DataRefs point at, named the way an entity record names its own.

    Both ends of a ``prov:used`` edge have to spell a container identically or the graph reads one artifact as two, so this relativises against the study root exactly as :func:`read_records` names the entity it produced. A reference that cannot be resolved is dropped rather than raising: provenance describes a run that already succeeded, and a binding this run never had to read is not a reason to fail it.
    """
    from tvbo.data import dataref
    from tvbo.utils import as_list

    out: list[str] = []
    root = Path(study_root).resolve()
    for ref in as_list(refs):
        if ref is None:
            continue
        try:
            path = Path(dataref.locate_container(ref, results_root=results_root)).resolve()
        except Exception:  # noqa: BLE001 — an unresolvable binding is a missing edge, never a failed run
            continue
        name = str(path.relative_to(root)) if _under(path, root) else str(path)
        if name not in out:
            out.append(name)
    return out


PROV_ACTIVITY = "prov:Activity"
PROV_ENTITY = "prov:Entity"
PROV_AGENT = "prov:SoftwareAgent"
"""The three PROV-O node types these records describe: what ran, what came out, and what it ran with."""


def read_records(study_root: Path | str) -> dict[str, dict]:
    """Every run *study_root* has recorded, keyed by its label, in BEP028's four-kind shape.

    The records live one per container, in that container's sidecar; this reads the study's results directory back as the set they collectively form. The four kinds are reconstructed rather than stored: ``act`` is the activity, ``ent`` is what the sidecar's own location and ``outputs`` say about the container, and ``env``/``soft`` are the environment and the requirements it aggregates. A consumer written against BEP028 therefore reads a tvbo study without knowing where the four went.

    The study root is the argument rather than the results directory because an entity has to be named here exactly as :func:`input_containers` names it at the other end of a ``prov:used`` edge — relative to the study — or the graph reads one artifact as two.

    A sidecar with no ``provenance:`` is skipped rather than faulted: it describes a container written before the run recorded itself, and refusing to read the directory would lose every record that is there.

    One producer can write several containers — a cohort experiment fans into one per subject — and they share a label. Those are keyed ``<label>/<container stem>`` rather than collapsed onto the label, because keying them all the same would report the fan-out as the single run that happened to be read last.
    """
    import yaml

    from tvbo.utils.study_layout import study_path

    study_root = Path(study_root)
    results_root = study_path("results", root=study_root)
    by_label: dict[str, list[tuple[Path, dict]]] = {}
    if not results_root.is_dir():
        return {}
    for path in sorted(results_root.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        record = (document or {}).get("provenance") if isinstance(document, dict) else None
        if not isinstance(record, dict) or not record.get("activities"):
            continue
        activity = (record.get("activities") or [{}])[0]
        environment = dict(record.get("environment") or {})
        requirements = environment.pop("requirements", []) or []
        container = path.with_suffix(".h5")
        by_label.setdefault(prov_label(activity.get("iri") or ""), []).append(
            (
                path,
                {
                    "act": activity,
                    "ent": {
                        "container": str(container.relative_to(study_root))
                        if _under(container, study_root)
                        else str(container),
                        "produced_by": activity.get("iri"),
                        "outputs": record.get("outputs"),
                    },
                    "env": environment,
                    "soft": {"packages": requirements},
                },
            )
        )
    return {
        (label if len(found) == 1 else f"{label}/{path.stem}"): record
        for label, found in by_label.items()
        for path, record in found
    }


def provenance_graph(study_root: Path | str) -> dict:
    """The runs *study_root* has recorded, as one PROV-O typed graph of nodes and edges.

    The inverse of :func:`emit`: the records say what happened one container at a time, and this reads the set back as the derivation it collectively describes. Every node carries its PROV-O type so a consumer draws or queries the graph by what a node IS — an activity, the entity it generated, the software agent it ran with — rather than by where the filename put it.

    Three relations come out of the four record kinds. An entity ``prov:wasGeneratedBy`` the activity of its own label; that activity ``prov:wasAssociatedWith`` each package named in ``associated_with``, versioned from the requirement list under the software record's ``packages``; and it ``prov:used`` each entity its ``used`` list points at, which is what makes a chain of runs one graph rather than a pile of independent ones. An entity that is only ever used and never generated here still becomes a node, because a reference to something this study did not produce is exactly what an external input looks like.
    """
    sets = read_records(study_root)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(node_id: str, node_type: str, label: str, **attrs) -> str:
        existing = nodes.setdefault(node_id, {"id": node_id, "type": node_type, "label": label})
        existing.update({k: v for k, v in attrs.items() if v is not None})
        return node_id

    for label, record in sorted(sets.items()):
        activity, entity, software = record.get("act"), record.get("ent"), record.get("soft")
        act_id = None
        if activity:
            act_id = node(
                f"activity:{label}",
                PROV_ACTIVITY,
                label,
                command=activity.get("command"),
                started_at=activity.get("started_at"),
                ended_at=activity.get("ended_at"),
            )
        if entity:
            ent_id = node(
                f"entity:{entity.get('container', label)}",
                PROV_ENTITY,
                str(entity.get("container", label)),
                outputs=entity.get("outputs"),
                produced_by=entity.get("produced_by"),
            )
            if act_id:
                edges.append({"source": ent_id, "target": act_id, "relation": "prov:wasGeneratedBy"})
        if act_id:
            versions = {p.get("name"): p.get("version") for p in (software or {}).get("packages") or () if isinstance(p, dict)}
            for package in activity.get("associated_with", []) or []:
                agent = node(f"agent:{package}", PROV_AGENT, str(package), version=versions.get(package))
                edges.append({"source": act_id, "target": agent, "relation": "prov:wasAssociatedWith"})
            for used in activity.get("used", []) or []:
                edges.append(
                    {"source": act_id, "target": node(f"entity:{used}", PROV_ENTITY, str(used)), "relation": "prov:used"}
                )
    return {"nodes": list(nodes.values()), "edges": edges}
