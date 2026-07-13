#!/usr/bin/env python
"""Backfill ``alternateName`` aliases onto the HCP-MMP1 atlas terminology.

The canonical HCP-MMP1 parcellation (``tvbo/database/atlases/…atlas-hcpmmp1…``)
labels regions ``L_V1 … Brain-Stem``. Empirical per-subject functional-connectome
pipelines emit the SAME 379 parcels under a different string convention
(``R_V1_ROI`` cortical, ``THALAMUS_LEFT`` / ``BRAIN_STEM`` subcortical). ``by_label``
node reconciliation aligns a dataset-sourced target to a model network by label
string, so the two conventions never intersect (0/379) unless the atlas records the
alternate names.

This tool derives each region's empirical alias from its canonical label via one
documented, deterministic rule and writes it into that entity's ``alternateName``
list. The rule runs ONCE here; the atlas YAML is the source of truth thereafter and
the resolver does a pure alias-aware string match — no normalization in the hot
path. When a real subject sidecar is reachable, the generated alias set is checked
to be an exact bijection with the empirical labels, so a rule error surfaces
immediately instead of silently mis-mapping.

Usage:
    python scripts/backfill_hcpmmp1_aliases.py [--check] [--verify-sidecar PATH]

``--check`` verifies without writing (non-zero exit on drift). Without a sidecar
path the tool auto-probes the HCP-YA functional-connectome tree for one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML

REPO = Path(__file__).resolve().parents[1]
_ATLAS_BASENAME = "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_dseg.yaml"
# The HCP-MMP1 terminology is tracked in two places: the authored source DB and the
# packaged runtime copy the ``Atlas``/``atlas_data`` loader actually reads. Both must
# carry the aliases (and stay consistent), so the tool updates every copy present.
ATLAS_FILES = [
    REPO / "tvbo" / "data" / "tvbo_data" / "atlas" / _ATLAS_BASENAME,  # runtime (loaded)
    REPO / "tvbo" / "database" / "atlases" / _ATLAS_BASENAME,          # authored source
]

# Subcortical region stems (per hemisphere) that are NOT cortical parcels: these do
# not take the cortical ``_ROI`` suffix and use the upper-case ``<REGION>_<SIDE>``
# empirical convention. Everything else prefixed ``L_``/``R_`` is a cortical parcel.
SUBCORTICAL = {
    "Cerebellum", "Thalamus", "Caudate", "Putamen", "Pallidum",
    "Hippocampus", "Amygdala", "Accumbens", "VentralDC",
}
# Canonical stem -> empirical stem where they are not a plain upper-casing.
SUBCORTICAL_EMPIRICAL = {"VentralDC": "DIENCEPHALON_VENTRAL"}


def hemisphere(label: str) -> str | None:
    """Hemisphere of a node label under EITHER convention, or None if bilateral.

    Reads the hemisphere from the label string in a convention-agnostic way so the
    crosswalk can be checked for hemisphere parity: a canonical ``L_*``/``R_*`` and
    its empirical alias must resolve to the SAME hemisphere. Bilateral midline
    structures (brain stem) resolve to None on both sides. Deliberately does NOT use
    any node index/order — hemisphere comes from the name alone.
    """
    if label in ("Brain-Stem", "BRAIN_STEM"):
        return None
    if label[:2] == "L_" or label.endswith("_LEFT"):
        return "L"
    if label[:2] == "R_" or label.endswith("_RIGHT"):
        return "R"
    return None


def canonical_to_empirical(canon: str) -> str:
    """Empirical (functional-connectome pipeline) label for a canonical HCP-MMP1 label.

    Cortical ``L_V1`` -> ``L_V1_ROI`` (append ``_ROI``); subcortical ``L_Thalamus``
    -> ``THALAMUS_LEFT`` (upper-case stem + hemisphere word); ``Brain-Stem`` ->
    ``BRAIN_STEM``.
    """
    if canon == "Brain-Stem":
        return "BRAIN_STEM"
    if canon[:2] in ("L_", "R_"):
        hemi, stem = canon[0], canon[2:]
        if stem in SUBCORTICAL:
            side = "LEFT" if hemi == "L" else "RIGHT"
            empirical_stem = SUBCORTICAL_EMPIRICAL.get(stem, stem.upper())
            return f"{empirical_stem}_{side}"
        return f"{canon}_ROI"  # cortical parcel
    raise ValueError(f"Unrecognised canonical HCP-MMP1 label: {canon!r}")


def _empirical_labels_from_sidecar(path: Path) -> list[str]:
    """Node labels of an empirical HCP-MMP1 network sidecar, in file order."""
    yaml = YAML(typ="safe")
    data = yaml.load(path.read_text())
    nodes = data.get("nodes") or []
    return [str(n["label"]) for n in nodes if isinstance(n, dict) and "label" in n]


def _auto_sidecar() -> Path | None:
    """Probe the HCP-YA functional-connectome tree for one HCP-MMP1 sidecar."""
    root = Path("/Volumes/bronkodata/hcp/derivatives/hcp_ya/functional_connectomes")
    if not root.exists():
        return None
    hits = sorted(root.glob("sub-*/sub-*atlas-HCPMMP1*relmat.yaml"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    ap.add_argument("--verify-sidecar", type=Path, default=None,
                    help="empirical HCP-MMP1 sidecar to bijection-check against")
    args = ap.parse_args()

    targets = [p for p in ATLAS_FILES if p.exists()]
    if not targets:
        print(f"FAIL: no HCP-MMP1 atlas file found among {ATLAS_FILES}", file=sys.stderr)
        return 2

    yaml = YAML()  # round-trip: preserve entity order, floats, layout
    yaml.preserve_quotes = True
    # Canonical labels are identical across copies; derive aliases once from the first.
    canon_labels = list(yaml.load(targets[0].read_text())["terminology"]["entities"].keys())

    aliases = {c: canonical_to_empirical(c) for c in canon_labels}

    # Self-check 1: the alias set is itself a 1-to-1 (no two regions collide).
    if len(set(aliases.values())) != len(aliases):
        dupes = sorted({a for a in aliases.values() if list(aliases.values()).count(a) > 1})
        print(f"FAIL: alias collisions {dupes}", file=sys.stderr)
        return 2

    # Self-check 1b: HEMISPHERE PARITY. A canonical left-hemisphere region must alias
    # to a left-hemisphere empirical label and never to a right one — this is the
    # silent failure mode of index/order-based alignment (a shifted subcortical block
    # swaps L<->R). Enforced from label strings alone, never from position.
    swapped = [(c, a) for c, a in aliases.items() if hemisphere(c) != hemisphere(a)]
    if swapped:
        print(f"FAIL: {len(swapped)} hemisphere mismatches, e.g. {swapped[:3]}", file=sys.stderr)
        return 2

    # Self-check 2: exact bijection with real empirical labels, when reachable.
    sidecar = args.verify_sidecar or _auto_sidecar()
    if sidecar and sidecar.exists():
        empirical = set(_empirical_labels_from_sidecar(sidecar))
        generated = set(aliases.values())
        missing = empirical - generated   # empirical labels no rule produced
        extra = generated - empirical     # rule outputs not in the data
        if missing or extra:
            print(f"FAIL: not a bijection with {sidecar.name}", file=sys.stderr)
            if missing:
                print(f"  empirical labels unmatched by the rule: {sorted(missing)}", file=sys.stderr)
            if extra:
                print(f"  rule outputs absent from the data:      {sorted(extra)}", file=sys.stderr)
            return 2
        print(f"bijection OK: 379/379 vs {sidecar.name}")
    else:
        print("NOTE: no empirical sidecar reachable; wrote rule-derived aliases "
              "without a data cross-check (run with --verify-sidecar to confirm).")

    # Apply (or --check) per atlas copy — every tracked copy must carry the aliases.
    def current(ent) -> list:
        return list(ent.get("alternateName") or [])

    rc = 0
    for path in targets:
        data = yaml.load(path.read_text())
        entities = data["terminology"]["entities"]
        if list(entities.keys()) != canon_labels:
            print(f"FAIL: {path.name} has a different entity set than {targets[0].name}",
                  file=sys.stderr)
            return 2
        drift = [c for c in canon_labels if current(entities[c]) != [aliases[c]]]
        rel = path.relative_to(REPO)
        if not drift:
            print(f"up to date: {rel}")
            continue
        if args.check:
            print(f"DRIFT: {rel} needs {len(drift)} aliases (e.g. {drift[:3]})", file=sys.stderr)
            rc = 1
            continue
        for c in canon_labels:
            entities[c]["alternateName"] = [aliases[c]]
        with path.open("w") as f:
            yaml.dump(data, f)
        print(f"wrote aliases for {len(canon_labels)} entities -> {rel}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
