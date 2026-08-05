"""``tvbo verify`` — check an Investigation is buildable, and hard-fail if not.

The pre-render build gate: it resolves the manifest, checks every member recipe exists, every
declared analysis has a fresh container, and every ``results:`` key is live — without running
anything. With ``--manuscript`` it also cross-checks the prose's ``{{< meta results.* >}}``
keys against the manifest, so a citation with no number (or a number no one cites) is caught
before the PDF builds. A non-empty problem list exits non-zero, which is what lets a Quarto
pre-render step fail loudly instead of rendering a stale or wrong figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import typer

from . import _common

_META_KEY = re.compile(r"\{\{<\s*meta\s+results\.([A-Za-z0-9_]+)\s*>\}\}")


def _scan_meta_keys(target: Path) -> set[str]:
    """The ``results.<key>`` tokens cited across a manuscript file or directory tree."""
    target = Path(target)
    files = (
        [p for p in target.rglob("*") if p.suffix.lower() in {".qmd", ".md"}]
        if target.is_dir()
        else [target]
    )
    keys: set[str] = set()
    for f in files:
        try:
            keys.update(_META_KEY.findall(f.read_text(encoding="utf-8")))
        except OSError:
            continue
    return keys


def verify(
    spec: str = typer.Argument(..., help="Path to an Investigation YAML."),
    results_root: Path = typer.Option(
        None, "--results-root",
        help="Directory holding the run's result containers (default: <investigation-dir>/output).",
    ),
    manuscript: Path = typer.Option(
        None, "--manuscript",
        help="A .qmd/.md file or directory whose `{{< meta results.* >}}` keys are cross-checked "
             "against the declared `results:` — a cited-but-undeclared or declared-but-uncited "
             "key is a failure.",
    ),
) -> None:
    """Verify an Investigation's completeness, staleness and manifest coverage."""
    kind, obj = _common.resolve_spec(spec)
    if kind != "investigation":
        _common.die(
            f"`tvbo verify` needs an Investigation; {spec} resolves to a {kind}. "
            f"Point it at the tvbo_manuscript.yaml (the study-of-studies with `members:`)."
        )

    from tvbo.data.investigation import verify as _verify

    base = Path(getattr(obj, "_source_file", spec)).resolve().parent
    keys = _scan_meta_keys(manuscript) if manuscript is not None else None
    problems = _verify(obj, base, results_root=results_root, manuscript_keys=keys)
    if problems:
        _common.die("verification failed:\n  - " + "\n  - ".join(problems))
    _common.info(f"verification passed ({len(list(getattr(obj, 'members', None) or []))} member(s)).")
