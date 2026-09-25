"""``tvbo verify`` — check a study-of-studies is buildable, and hard-fail if not.

The build gate, in two modes. Offline (where the run containers live) it resolves every ``results:`` binding and checks analysis staleness. In a build/CI checkout the containers are generated artifacts that are never committed, so ``--manifest manuscript_results.yml`` runs it
CONTAINER-FREE: the declared bindings and, with ``--manuscript``, the prose's ``{{< meta results.* >}}`` keys are checked against the committed manifest instead of being resolved. Either way a citation with no number, a number no one cites, a binding added without regenerating the manifest, or a committed ``<figure>.caption.qmd`` that no longer matches the caption its spec composes is caught. A non-empty problem list exits non-zero, so a Quarto pre-render step fails loudly instead of rendering a stale or wrong figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from . import _common

_META_KEY = re.compile(r"\{\{<\s*meta\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*>\}\}")


def _scan_meta_keys(target: Path) -> set[str]:
    """The ``<namespace>.<key>`` tokens cited across a manuscript file or directory tree, namespace included.

    Every namespace is scanned, not just ``results``: the manifest supplies only ``results``, so a token in any other namespace renders empty, and scanning for it is the only way that failure is caught before it reaches the page.

    A path that does not exist is an error, not an empty scan. Swallowing it produced the worst possible diagnostic: every declared key reported as "never cited", burying the one real problem (the typo) under a wall of wrong ones.
    """
    target = Path(target)
    if not target.exists():
        _common.die(f"--manuscript: no such file or directory: {target}")
    files = [p for p in target.rglob("*") if p.suffix.lower() in {".qmd", ".md"}] if target.is_dir() else [target]
    if not files:
        _common.die(f"--manuscript: {target} contains no .qmd or .md files to scan.")
    keys: set[str] = set()
    unreadable: list[str] = []
    for f in files:
        try:
            keys.update(f"{ns}.{key}" for ns, key in _META_KEY.findall(f.read_text(encoding="utf-8")))
        except OSError as e:
            unreadable.append(f"{f}: {e.strerror or e}")
    if unreadable:
        _common.die("--manuscript: could not read:\n  - " + "\n  - ".join(unreadable))
    return keys


def verify(
    spec: str = typer.Argument(..., help="Path to a study YAML that nests `studies:`."),
    results_root: Path = typer.Option(
        None,
        "--results-root",
        help="Directory holding the run's result containers (default: <collection-dir>/output).",
    ),
    manuscript: list[Path] = typer.Option(
        None,
        "--manuscript",
        help="A .qmd/.md file or directory whose `{{< meta results.* >}}` keys are cross-checked "
        "against the declared `results:` — a cited-but-undeclared or declared-but-uncited "
        "key is a failure. Repeat for each, so prose split across a body and a supplement is gated "
        "as one manuscript rather than one of the two silently going unchecked.",
    ),
    manifest: Path = typer.Option(
        None,
        "--manifest",
        help="Path to the committed results manifest (manuscript_results.yml). When given, verify "
        "runs CONTAINER-FREE: it checks the declared bindings and the prose's cited keys "
        "against this committed manifest instead of resolving DataRefs into run containers "
        "— the build gate, since those containers are generated and never committed.",
    ),
    captions: Path = typer.Option(
        None,
        "--captions",
        help="Directory holding the composed `<figure>.caption.qmd` partials (default: "
        "<collection-dir>/figures). Each committed caption is recomposed from the spec and a "
        "mismatch (a stale caption the manuscript would still render) is a failure.",
    ),
) -> None:
    """Verify a study-of-studies' completeness, staleness and manifest coverage."""
    kind, obj = _common.resolve_spec(spec)
    if kind != "study" or not (getattr(obj, "studies", None) or []):
        detail = "nests no `studies:`" if kind == "study" else f"resolves to a {kind}"
        _common.die(
            f"`tvbo verify` needs a study-of-studies; {spec} {detail}. "
            f"Point it at the tvbo_manuscript.yaml (the study that nests `studies:`)."
        )

    from tvbo.data.study_manifest import verify as _verify

    base = Path(getattr(obj, "_source_file", spec)).resolve().parent
    keys = set().union(*(_scan_meta_keys(p) for p in manuscript)) if manuscript else None
    problems = _verify(
        obj,
        base,
        results_root=results_root,
        manuscript_keys=keys,
        manifest_path=manifest,
        captions_dir=captions,
    )
    if problems:
        _common.die("verification failed:\n  - " + "\n  - ".join(problems))
    _common.info(f"verification passed ({len(obj.walk_studies(include_self=False))} nested stud(ies)).")
