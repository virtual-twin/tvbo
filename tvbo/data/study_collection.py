"""StudyCollection-level helpers: the results manifest and the ``tvbo verify`` checks.

An :class:`~tvbo.classes.study.StudyCollection` is a study-of-studies — it aggregates the
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


def _count_target(inv: Any, member_label: Optional[str]) -> Any:
    """The object a ``count:`` binding tallies: the collection itself, or a loaded member."""
    if member_label is None:
        if inv is None:
            raise ValueError("a bare `count:` collection needs a StudyCollection context")
        return inv
    if inv is None:
        raise ValueError(f"count references member {member_label!r} but no StudyCollection context was given")
    from tvbo.classes.study import SimulationStudy

    for label, path in inv.member_recipes():
        if label == member_label:
            if "://" in str(path):
                raise ValueError(f"cannot count a collection on IRI member {member_label!r}")
            return SimulationStudy.from_file(str(path))
    raise ValueError(f"count references unknown member {member_label!r}")


def _count(spec: str, inv: Any) -> int:
    """Length of the collection a ``count:`` binding names.

    ``<member>.<collection>`` counts the collection on that member study (loaded from its
    recipe); a bare ``<collection>`` counts one on the collection itself. An unknown
    collection slot raises, so a typo fails the build rather than tallying to zero.
    """
    member_label, _, coll = str(spec).rpartition(".")
    target = _count_target(inv, member_label or None)
    if not hasattr(target, coll):
        where = member_label or "the collection"
        raise ValueError(f"count collection {coll!r} is not a slot on {where}")
    return len(as_list(getattr(target, coll)))


def container_roots(inv: Any, results_root: Optional[Path]) -> list[Path]:
    """Every directory a StudyCollection's result containers can live under, in search order.

    The collection's own root first, then one per member. A member study runs in its own
    directory and writes ``<member-dir>/output/results/<name>/result.h5``, so a ``used:``
    binding into a member — which ``StudyCollection.results`` explicitly documents as
    supported — is not reachable from the collection's root alone.
    """
    roots: list[Path] = [Path(results_root)] if results_root else []
    try:
        members = list(inv.member_recipes()) if inv is not None else []
    except Exception:  # noqa: BLE001 — a malformed member list is reported elsewhere
        members = []
    for _label, path in members:
        root = Path(path).resolve().parent / "output"
        if root not in roots:
            roots.append(root)
    return roots


def _resolve_across_roots(used: Any, results_root: Optional[Path], inv: Any):
    """Resolve *used* against the first container root that holds it.

    The first failure is re-raised when none do, so the reported problem names the
    collection's own root rather than whichever member happened to be searched last.
    """
    roots = container_roots(inv, results_root)
    first_error: Optional[Exception] = None
    for root in roots or [None]:
        try:
            return resolve_dataref(used, results_root=str(root) if root else None)
        except Exception as e:  # noqa: BLE001 — try the next root, report the first
            first_error = first_error or e
    raise first_error  # type: ignore[misc]


def resolve_binding(binding: Any, results_root: Optional[Path], *, inv: Any = None) -> tuple[str, dict]:
    """Resolve one ``ResultBinding`` to ``(rendered_string, provenance)``.

    Three mutually exclusive forms: ``used:`` reads a scalar out of a result container and
    formats it; ``count:`` tallies a collection on a member or the collection itself (no run);
    ``value:`` (+ ``source:``) passes an authored literal through untouched. Raises
    ``ValueError`` for a malformed binding (zero or more than one form set), and lets a
    resolution failure (missing container, dead reference, non-scalar, unknown member)
    propagate to the caller, which turns it into a build-failing problem keyed by ``binding.key``.
    """
    used = getattr(binding, "used", None)
    value = getattr(binding, "value", None)
    count = getattr(binding, "count", None)
    fmt = getattr(binding, "format", None)
    n_set = sum(x is not None for x in (used, value, count))
    if n_set == 0:
        raise ValueError("none of `used:`, `count:`, or `value:` is set")
    if n_set > 1:
        raise ValueError("`used:`, `count:`, and `value:` are mutually exclusive")

    if value is not None:
        prov = {"computed": False, "value": str(value)}
        src = getattr(binding, "source", None)
        if src:
            prov["source"] = str(src)
        desc = getattr(binding, "description", None)
        if desc:
            prov["description"] = str(desc)
        return str(value), prov

    if count is not None:
        rendered = _format(_count(count, inv), fmt)
        prov = {"computed": True, "count": str(count)}
        desc = getattr(binding, "description", None)
        if desc:
            prov["description"] = str(desc)
        return rendered, prov

    da = _resolve_across_roots(used, results_root, inv)
    rendered = _format(_scalar(da), fmt)
    ref = "/".join(
        p
        for p in (
            getattr(used, "analysis", None) or getattr(used, "experiment", None) or getattr(used, "iri", None),
            getattr(used, "output", None),
        )
        if p
    )
    prov = {"computed": True, "ref": ref}
    desc = getattr(binding, "description", None)
    if desc:
        prov["description"] = str(desc)
    return rendered, prov


def resolve_results(inv: Any, results_root: Optional[Path]) -> tuple[dict[str, str], dict[str, dict], list[str]]:
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
            rendered, prov = resolve_binding(binding, results_root, inv=inv)
        except Exception as e:  # noqa: BLE001 — reported as a build problem, not raised
            problems.append(f"{key}: {type(e).__name__}: {e}")
            continue
        results[key] = rendered
        provenance[key] = prov
    return results, provenance, problems


def emit_manifest(inv: Any, results_root: Optional[Path], out_path: Path) -> tuple[Path, list[str]]:
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
    """The ``@fig-*`` cross-reference ids the collection's figures declare (by name)."""
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


def _analysis_fingerprint(analysis: Any) -> str:
    """A stable digest of the one analysis, over the fields that change its numbers."""
    import hashlib
    import json

    def plain(obj, depth=0):
        if depth > 6 or obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [plain(o, depth + 1) for o in obj]
        if hasattr(obj, "items"):
            return {str(k): plain(v, depth + 1) for k, v in sorted(obj.items())}
        fields = getattr(obj, "model_fields", None) or getattr(obj, "__dict__", {})
        return {str(k): plain(getattr(obj, k, None), depth + 1) for k in sorted(fields) if not str(k).startswith("_")}

    blob = json.dumps(plain(analysis), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _stale_or_missing_analyses(inv: Any, results_root: Path, source_file: Optional[Path]) -> list[str]:
    """Analyses whose container is missing, or written from a different declaration.

    Staleness is per analysis, keyed on a digest of THAT analysis's own declaration, not on
    the spec file's mtime. Comparing against the file made every unrelated edit — a caption,
    a new figure, a typo in a description — mark every analysis as needing a re-run, so a
    one-word change failed the build and demanded hours of recomputation it could not affect.

    The digest is recorded beside the container when it is written. A container from before
    this check existed carries none, and is accepted: the alternative is failing every build
    once, which teaches people to bypass the gate.
    """
    from tvbo.data.analysis_io import study_analyses, analysis_name, container_path

    problems: list[str] = []
    for analysis in study_analyses(inv):
        name = analysis_name(analysis)
        path = container_path(name, results_root)
        if not path.exists():
            problems.append(f"{name}: analysis container missing (never run): {path}")
            continue
        stamp = path.parent / ".fingerprint"
        if stamp.exists() and stamp.read_text().strip() != _analysis_fingerprint(analysis):
            problems.append(f"{name}: analysis declaration changed since its container was written (edited but not re-run)")
    return problems


def _read_manifest_keys(path: Path) -> set[str]:
    """The result keys recorded in a committed manifest (``manuscript_results.yml``)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return set((data.get("results") or {}).keys())


def _stale_captions(inv: Any, captions_dir: Path) -> list[str]:
    """Committed caption partials whose text no longer matches what the spec composes.

    Each figure's ``<name>.caption.qmd`` is generated by ``tvbo figure caption`` and included by
    the manuscript; a spec edit that moves a panel or rewrites a description without recomposing
    leaves a committed caption the document still renders. That drift is silent — Quarto has no
    way to know the partial is stale — so this recomposes each caption from the spec and fails on
    a mismatch. A figure with no committed partial is skipped (nothing to be stale).
    """
    captions_dir = Path(captions_dir)
    if not captions_dir.is_dir():
        return []
    try:
        from tvbo.adapters.bsplot import compose_caption
    except Exception as e:
        return [f"cannot import the caption composer to check staleness: {e}"]
    problems: list[str] = []
    for fig in as_list(getattr(inv, "figures", None)):
        name = getattr(fig, "name", None)
        if not name:
            continue
        path = captions_dir / f"{name}.caption.qmd"
        if not path.exists():
            continue
        if path.read_text(encoding="utf-8") != compose_caption(fig) + "\n":
            problems.append(
                f"figure {name!r}: committed caption {path.name} is stale — the spec composes a "
                f"different caption; regenerate with `tvbo figure caption`"
            )
    return problems


def verify(
    inv: Any,
    base: Path,
    *,
    results_root: Optional[Path] = None,
    manuscript_keys: Optional[Iterable[str]] = None,
    manifest_path: Optional[Path] = None,
    captions_dir: Optional[Path] = None,
) -> list[str]:
    """Check a StudyCollection is buildable, returning a list of problems (empty = OK).

    Structural checks run in both modes: every member recipe exists, every declared figure
    carries a cross-reference id, and every committed ``<figure>.caption.qmd`` still matches the
    caption its spec composes (a stale caption fails here, not silently in the rendered PDF).
    What differs is how the numbers are checked:

    * **offline** (``manifest_path`` is None) — resolve every ``results:`` binding against its
      run container and check analysis staleness. The full gate, run where the containers live.
    * **build** (``manifest_path`` given) — the run containers are generated artifacts that are
      never committed, so instead of resolving them this reads the committed manifest: the
      declared bindings and the prose's cited keys must both match its keys exactly. A binding
      added without regenerating the manifest, or a citation with no number, fails here without
      a single container present.
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
    problems += _stale_captions(inv, captions_dir if captions_dir is not None else base / "figures")

    declared = {getattr(b, "key", None) for b in as_list(getattr(inv, "results", None))}
    declared.discard(None)

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            problems.append(
                f"committed manifest not found: {manifest_path} — run `tvbo run` where the containers live and commit it"
            )
            return problems
        available = _read_manifest_keys(manifest_path)
        for key in sorted(declared - available):
            problems.append(
                f"results.{key}: declared in `results:` but absent from the committed manifest "
                f"({manifest_path.name}) — regenerate it with `tvbo run`"
            )
        for key in sorted(available - declared):
            problems.append(
                f"results.{key}: in the committed manifest but no longer declared in `results:` "
                f"— regenerate it with `tvbo run`"
            )
    else:
        available = declared
        problems += _stale_or_missing_analyses(inv, results_root, source_file)
        _, _, resolve_problems = resolve_results(inv, results_root)
        problems += resolve_problems

    if manuscript_keys is not None:
        where = "the committed manifest" if manifest_path is not None else "`results:`"
        cited = set(manuscript_keys)
        for key in sorted(cited - available):
            problems.append(f"results.{key}: cited in the manuscript but not in {where}")
        for key in sorted(available - cited):
            problems.append(f"results.{key}: in {where} but never cited in the manuscript")

    return problems
