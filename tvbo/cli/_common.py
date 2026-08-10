"""Shared helpers for CLI verbs: SPEC resolution, output, error reporting.

Keeps the per-verb modules as thin as possible.
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger("tvbo.cli")


# SPEC resolution
# A SPEC is one of:
#   * a path-like (`./foo.yaml`, `/abs/path`, `file://...`)
#   * a CURIE (`dynamics:JansenRit`, `study:Schirner2023`)
#   * a bare DB name (`JansenRit`)
# C0/C1 supports the local cases only; HTTP / platform are added in C2/C3.

_CURIE_TO_CLASS = {
    "study": "SimulationStudy",
    "experiment": "SimulationExperiment",
    "dynamics": "Dynamics",
    "model": "Dynamics",
    "network": "Network",
    "coupling": "Coupling",
    "integrator": "Integrator",
    "function": "Function",
    "observation": "Observation",
    "atlas": "BrainAtlas",
    "continuation": "Continuation",
}


def _is_pathlike(spec: str) -> bool:
    if spec.startswith(("http://", "https://", "file://")):
        return spec.startswith("file://")
    return ("/" in spec) or spec.startswith(".") or Path(spec).suffix != ""


def resolve_spec(spec: str) -> tuple[str, Any]:
    """Resolve *spec* to ``(kind, obj)``.

    *kind* is one of ``'study'``, ``'experiment'``, ``'dynamics'``,
    ``'network'``, … and *obj* is the loaded class instance.

    HTTP and platform transports are not yet implemented (raise a helpful message).
    """
    if spec.startswith(("http://", "https://")):
        raise typer.BadParameter(f"HTTP transport not yet implemented (C2). Pull the file locally first: {spec}")

    if spec.startswith("file://"):
        spec = spec[len("file://") :]

    # Path?
    if _is_pathlike(spec):
        path = Path(spec).expanduser()
        if not path.exists():
            raise typer.BadParameter(f"No such file: {path}")
        return _load_from_file(path)

    # CURIE? (`prefix:name`)
    if ":" in spec and not spec.startswith(":"):
        prefix, _, name = spec.partition(":")
        cls_name = _CURIE_TO_CLASS.get(prefix.lower())
        if cls_name is None:
            raise typer.BadParameter(f"Unknown CURIE prefix {prefix!r}. Known: {', '.join(sorted(_CURIE_TO_CLASS))}")
        return _load_from_db(cls_name, name)

    # Bare name — try Study, then Experiment, then Dynamics
    for cls_name in ("SimulationStudy", "SimulationExperiment", "Dynamics"):
        try:
            return _load_from_db(cls_name, spec)
        except (FileNotFoundError, ValueError):
            continue
    raise typer.BadParameter(f"Could not resolve {spec!r}: not a path, CURIE, or known DB entry.")


def experiment_ids(exp: Any) -> set[str]:
    """The identifiers an experiment can be selected by on the CLI.

    Its ``key``, ``name``, ``label``, and stringified ``id`` (dropping the empty ones), plus the bare numeric id those spell — ``exp-3``, ``exp3`` and ``3`` name one
    experiment, and an experiment carrying only ``key: exp-3`` must still answer to ``3``.
    Normalising HERE and in ``analysis_io.dependencies`` is what lets the two sides of the staleness walk intersect; normalising one side only makes every dotted spelling match
    nothing, and an empty stale set reads exactly like a clean one.

    Shared by ``tvbo run`` and ``tvbo workflow`` so ``--experiment`` matches the same way in both.
    """
    from tvbo.data.dataref import experiment_id

    spellings = {
        getattr(exp, "key", None),
        getattr(exp, "name", None),
        getattr(exp, "label", None),
        str(getattr(exp, "id", "")),
    } - {None, ""}
    return spellings | {experiment_id(s) for s in spellings} - {None}


def experiment_key(exp: Any) -> str:
    """The canonical short key for an experiment.

    An explicit ``key`` if set, else its ``id`` (the usual identifier, e.g. ``40``), else ``name``. Used for job names, result stems, and kit paths so they read
    ``…-40`` rather than a generic fallback — one source of truth shared by every emitter (``experiment_ids`` is the wider *match* set for ``--experiment``).
    """
    return str(getattr(exp, "key", None) or getattr(exp, "id", None) or getattr(exp, "name", None) or "experiment")


def _load_from_file(path: Path) -> tuple[str, Any]:
    """Best-effort type detection by reading the YAML's `class:` / shape."""
    import tvbo

    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        # Try the registry's importer (e.g. *.bidsdir, *.sedml, *.omex)
        from tvbo.export import resolve_by_extension, load as _load

        try:
            fmt = resolve_by_extension(suffix)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        if fmt.importer is None:
            raise typer.BadParameter(f"Format {fmt.key!r} has no importer; cannot load {path}.")
        obj = _load(fmt.key, path)
        return _classify(obj), obj

    # YAML — try StudyCollection (study-of-studies), then Study (it can contain Experiments), falling back to Experiment. A StudyCollection is a Study, so its more specific interpretation is tried first, keyed on the `members:` slot only it declares.
    text = path.read_text(encoding="utf-8")
    looks_like_study_collection = "members:" in text and ("recipe:" in text or "results:" in text)
    looks_like_study = "simulation_experiments" in text or ("experiments:" in text and "title:" in text)
    # Each fallback's error is kept: when they all fail, the last one (Dynamics) is about the least likely interpretation, so reporting only that sends the reader chasing a "bad Dynamics" that was never what the file is. A spec the running tvbo is too old to parse looked exactly like a malformed Dynamics until the earlier errors were surfaced.
    attempts: list[tuple[str, Exception]] = []
    if looks_like_study_collection:
        try:
            obj = tvbo.StudyCollection.from_file(str(path))
            return "study_collection", obj
        except Exception as e:
            attempts.append(("study_collection", e))
    if looks_like_study:
        try:
            obj = tvbo.SimulationStudy.from_file(str(path))
            return "study", obj
        except Exception as e:
            attempts.append(("study", e))

    try:
        obj = tvbo.SimulationExperiment.from_file(str(path))
        return "experiment", obj
    except Exception as e:
        attempts.append(("experiment", e))

    try:
        obj = tvbo.Dynamics.from_file(str(path))
        return "dynamics", obj
    except Exception as e:
        attempts.append(("dynamics", e))

    # Report the most specific interpretation the file actually looked like — the first one tried — and list the rest so nothing is hidden.
    kind, primary = attempts[0]
    detail = "\n".join(f"  - as {k}: {type(x).__name__}: {x}" for k, x in attempts)
    raise typer.BadParameter(
        f"Could not load {path} as a study, experiment or dynamics.\n{detail}\n"
        f"The {kind} error above is the most likely one to act on."
    ) from primary


def _load_from_db(cls_name: str, name: str) -> tuple[str, Any]:
    import tvbo

    cls = getattr(tvbo, cls_name)
    obj = cls.from_db(name)
    return _classify(obj), obj


def _classify(obj: Any) -> str:
    cls_name = type(obj).__name__
    return {
        "StudyCollection": "study_collection",
        "SimulationStudy": "study",
        "SimulationExperiment": "experiment",
        "Dynamics": "dynamics",
        "Network": "network",
        "Coupling": "coupling",
    }.get(cls_name, cls_name.lower())


# Output helpers


def emit_json(payload: Any) -> None:
    """Write a single JSON line to stdout (CLI machine-readable contract)."""
    typer.echo(_json.dumps(payload, default=str, sort_keys=True))


def info(msg: str) -> None:
    """Human-facing progress line, routed through the central ``tvbo`` logger.

    Emits at INFO on ``tvbo.cli`` (stderr by default), so ``--quiet`` /
    ``TVBO_LOG_LEVEL`` govern it exactly as they govern in-process ``.run()``.
    """
    logger.info(msg)


def warn(msg: str) -> None:
    """Human-facing warning line, routed through the central ``tvbo`` logger.

    Emits at WARNING on ``tvbo.cli`` (stderr by default), so ``--quiet`` /
    ``TVBO_LOG_LEVEL`` govern it exactly as they govern ``info`` / ``.run()``.
    """
    logger.warning(msg)


def die(msg: str, code: int = 1) -> None:
    """Log *msg* at ERROR and abort the CLI with *code*.

    A fatal abort must always explain itself: when the configured level would suppress ERROR (e.g. ``--log-level OFF`` / ``TVBO_LOG_LEVEL=OFF``) the reason
    still goes to stderr, so the CLI never exits non-zero in silence.
    """
    if logger.isEnabledFor(logging.ERROR):
        logger.error(msg)
    else:
        typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code)
