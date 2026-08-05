"""Investigation-level helpers: the results manifest and the ``tvbo verify`` checks.

An :class:`~tvbo.classes.study.Investigation` is a study-of-studies — it aggregates the
member studies a paper reports and owns the paper's own demonstration content — plus two
things a plain ``SimulationStudy`` has no need for:

* a **results manifest** (``results:``): the named numbers the prose cites, each bound to a
  computed value (``used:`` a DataRef) or an authored constant (``value:`` + ``source:``).
  :func:`emit_manifest` resolves them into ``manuscript_results.yml`` — the file the document
  reads through Quarto ``metadata-files`` as ``{{< meta results.<key> >}}`` — so no reported
  figure is transcribed by hand.
* the **completeness / staleness / coverage** checks :func:`verify` runs, so an edited-but-not-
  rerun spec, an orphan figure, or a dead manifest key fails the build rather than silently
  printing a wrong number.

The number resolution is shared: a manifest emit and a ``verify`` coverage pass resolve the
same bindings the same way, so the two cannot disagree about whether a key is live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from tvbo.data.dataref import resolve_dataref
from tvbo.utils import as_list

MANIFEST_NAME = "manuscript_results.yml"


def _scalar(da: Any) -> Any:
    """The single value behind a resolved :class:`DataRef`, or raise if it is not scalar."""
    arr = getattr(da, "values", da)
    size = getattr(arr, "size", None)
    if size not in (1, None):
        raise ValueError(f"binding resolves to an array of size {size}, not a scalar")
    if hasattr(arr, "item"):
        return arr.item()
    return arr


def _format(value: Any, fmt: Optional[str]) -> str:
    """Apply a binding's ``format`` string to a computed value, else stringify it."""
    return fmt.format(value) if fmt else str(value)


def resolve_binding(binding: Any, results_root: Optional[Path]) -> tuple[str, dict]:
    """Resolve one ``ResultBinding`` to ``(rendered_string, provenance)``.

    A computed binding (``used:``) reads a scalar out of a result container and formats it;
    an authored binding (``value:`` + ``source:``) passes its literal through untouched (a
    numeric ``format`` is for computed values, not a quoted constant). Raises ``ValueError``
    for a malformed binding (both or neither of ``used``/``value``), and lets a resolution
    failure (missing container, dead reference, non-scalar) propagate to the caller, which
    turns it into a build-failing problem keyed by ``binding.key``.
    """
    used = getattr(binding, "used", None)
    value = getattr(binding, "value", None)
    fmt = getattr(binding, "format", None)
    if used is not None and value is not None:
        raise ValueError("both `used:` and `value:` are set (they are mutually exclusive)")
    if used is None and value is None:
        raise ValueError("neither `used:` nor `value:` is set")

    if value is not None:
        prov = {"computed": False, "value": str(value)}
        src = getattr(binding, "source", None)
        if src:
            prov["source"] = str(src)
        desc = getattr(binding, "description", None)
        if desc:
            prov["description"] = str(desc)
        return str(value), prov

    da = resolve_dataref(used, results_root=str(results_root) if results_root else None)
    rendered = _format(_scalar(da), fmt)
    ref = "/".join(
        p for p in (getattr(used, "analysis", None) or getattr(used, "experiment", None)
                    or getattr(used, "iri", None), getattr(used, "output", None)) if p
    )
    prov = {"computed": True, "ref": ref}
    desc = getattr(binding, "description", None)
    if desc:
        prov["description"] = str(desc)
    return rendered, prov


def resolve_results(
    inv: Any, results_root: Optional[Path]
) -> tuple[dict[str, str], dict[str, dict], list[str]]:
    """Resolve every ``ResultBinding`` on *inv*.

    Returns ``(results, provenance, problems)``: ``results`` maps each key to its rendered
    string (what the prose reads), ``provenance`` records how each was obtained, and
    ``problems`` names the keys that could not be resolved (missing container, dead ref,
    duplicate key) — the caller decides whether an unresolved key fails the build.
    """
    results: dict[str, str] = {}
    provenance: dict[str, dict] = {}
    problems: list[str] = []
    for binding in as_list(getattr(inv, "results", None)):
        key = getattr(binding, "key", None)
        if not key:
            problems.append("a results entry has no `key`")
            continue
        if key in results:
            problems.append(f"{key}: duplicate results key")
            continue
        try:
            rendered, prov = resolve_binding(binding, results_root)
        except Exception as e:  # noqa: BLE001 — reported as a build problem, not raised
            problems.append(f"{key}: {type(e).__name__}: {e}")
            continue
        results[key] = rendered
        provenance[key] = prov
    return results, provenance, problems


def emit_manifest(
    inv: Any, results_root: Optional[Path], out_path: Path
) -> tuple[Path, list[str]]:
    """Write *inv*'s resolved results to ``manuscript_results.yml`` at *out_path*.

    The file carries a flat ``results:`` mapping the document reads as ``{{< meta results.* >}}``
    and a ``results_provenance:`` block recording, per key, whether the number was computed
    (and from which container) or authored (and from which source). Returns the written path
    and the list of unresolved-key problems; the caller hard-fails on a non-empty list.
    """
    import yaml

    results, provenance, problems = resolve_results(inv, results_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "results_provenance": provenance}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return out_path, problems


# --------------------------------------------------------------------------- verify


def _figure_ids(inv: Any) -> list[str]:
    """The ``@fig-*`` cross-reference ids the investigation's figures declare (by name)."""
    ids: list[str] = []
    for fig in as_list(getattr(inv, "figures", None)):
        name = getattr(fig, "name", None)
        if name:
            ids.append(str(name))
    return ids


def _missing_members(inv: Any, base: Path) -> list[str]:
    """Member recipes whose file does not exist (an IRI member is left to the loader)."""
    missing: list[str] = []
    recipes = inv.member_recipes(base) if hasattr(inv, "member_recipes") else []
    for label, path in recipes:
        if "://" in str(path):
            continue
        if not Path(path).exists():
            missing.append(f"{label}: member recipe not found: {path}")
    return missing


def _stale_or_missing_analyses(inv: Any, results_root: Path, source_file: Optional[Path]) -> list[str]:
    """Analyses whose container is missing, or older than the spec that declares them.

    Missing means never run; older-than-spec means the recipe was edited after the container
    was written, so the reported numbers reflect a superseded spec. Both are drift the build
    must not paper over. This is the reused ``analysis_io`` container convention; a full
    content-addressed check can tighten it later.
    """
    from tvbo.data.analysis_io import study_analyses, analysis_name, container_path

    problems: list[str] = []
    spec_mtime = source_file.stat().st_mtime if source_file and Path(source_file).exists() else None
    for analysis in study_analyses(inv):
        name = analysis_name(analysis)
        path = container_path(name, results_root)
        if not path.exists():
            problems.append(f"{name}: analysis container missing (never run): {path}")
        elif spec_mtime is not None and path.stat().st_mtime < spec_mtime:
            problems.append(f"{name}: analysis container older than the spec (edited but not re-run)")
    return problems


def verify(
    inv: Any,
    base: Path,
    *,
    results_root: Optional[Path] = None,
    manuscript_keys: Optional[Iterable[str]] = None,
) -> list[str]:
    """Check an investigation is buildable, returning a list of problems (empty = OK).

    Three families of check, all hard failures for the pre-render build seam:

    * **completeness** — every member recipe exists; every declared figure carries a
      cross-reference id.
    * **staleness** — every declared analysis has a container that is present and no older
      than the spec (:func:`_stale_or_missing_analyses`).
    * **coverage** — every ``results:`` key resolves (no dead binding); and, when the prose's
      ``{{< meta results.* >}}`` keys are supplied as *manuscript_keys*, every key the prose
      cites exists in the manifest and no manifest key is unused.
    """
    base = Path(base)
    results_root = Path(results_root) if results_root else (base / "output")
    source_file = Path(getattr(inv, "_source_file", "")) if getattr(inv, "_source_file", None) else None
    problems: list[str] = []

    problems += _missing_members(inv, base)
    for fig in as_list(getattr(inv, "figures", None)):
        if not getattr(fig, "name", None):
            label = getattr(fig, "label", None) or "?"
            problems.append(f"figure {label!r} has no `name` (needed for its @fig- cross-reference)")

    problems += _stale_or_missing_analyses(inv, results_root, source_file)

    _, _, resolve_problems = resolve_results(inv, results_root)
    problems += resolve_problems

    if manuscript_keys is not None:
        declared = {getattr(b, "key", None) for b in as_list(getattr(inv, "results", None))}
        declared.discard(None)
        cited = set(manuscript_keys)
        for key in sorted(cited - declared):
            problems.append(f"results.{key}: cited in the manuscript but not declared in `results:`")
        for key in sorted(declared - cited):
            problems.append(f"results.{key}: declared in `results:` but never cited in the manuscript")

    return problems
