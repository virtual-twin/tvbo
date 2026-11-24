"""Every tracked file that quotes TVB's source verbatim carries TVB's licence, GPL-3.0-or-later, in REUSE.toml.

Curated records take their model, coupling and parameter descriptions from TVB's docstrings, and those sentences travel on into the ontology, test references, run outputs and the README. The quoted sentences are found by looking each long record string up in the installed TVB source, and every file holding one must resolve, by REUSE's last-match rule, to an annotation whose licence includes GPL-3.0-or-later.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from tests.test_reuse_license import tracked_files

pytestmark = pytest.mark.backend_tvb

REPO = Path(__file__).resolve().parents[1]
BINARY = {".h5", ".png", ".gif", ".pkl", ".dill", ".nii", ".gz", ".npy", ".npz", ".pdf", ".jpg", ".mp4", ".xlsx"}


def _strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str) and len(node) > 45:
        yield " ".join(node.split())


def _glob(pattern: str) -> re.Pattern:
    """A REUSE.toml path pattern as a regex, as reuse reads it: ``**`` crosses directories and swallows the ``/`` after it, so ``a/**/b`` also matches ``a/b``; ``*`` stays within one; anything else is literal."""
    parts = re.split(r"(\*\*/?|\*)", pattern)
    return re.compile("".join(".*" if p.startswith("**") else "[^/]*" if p == "*" else re.escape(p) for p in parts) + r"\Z")


def _licence(path: str, annotations) -> str | None:
    match = None
    for annotation in annotations:
        patterns = annotation["path"] if isinstance(annotation["path"], list) else [annotation["path"]]
        if any(_glob(p).match(path) for p in patterns):
            match = annotation
    return match.get("SPDX-License-Identifier") if match else None


def test_files_quoting_tvb_carry_its_licence():
    tvb = pytest.importorskip("tvb")
    source = " ".join(" ".join(p.read_text(errors="ignore").split()) for p in Path(tvb.__file__).parent.rglob("*.py"))
    quoted = set()
    for record in (REPO / "tvbo" / "database").rglob("*.yaml"):
        quoted |= {s[:60] for s in _strings(yaml.safe_load(record.read_text(encoding="utf-8"))) if s[:60] in source}
    assert quoted, "no record quotes TVB, so the scan is not looking where the records are"

    annotations = tomllib.loads((REPO / "REUSE.toml").read_text(encoding="utf-8"))["annotations"]
    unlicensed = []
    for name in tracked_files("."):
        path = REPO / name
        if path.suffix in BINARY or not path.is_file():
            continue
        try:
            text = " ".join(path.read_text(encoding="utf-8").split())
        except UnicodeDecodeError:
            continue
        if any(q in text for q in quoted) and "GPL-3.0-or-later" not in (_licence(name, annotations) or ""):
            unlicensed.append(name)
    assert not unlicensed, f"these files quote TVB's source but REUSE.toml does not give them GPL-3.0-or-later: {unlicensed}"
