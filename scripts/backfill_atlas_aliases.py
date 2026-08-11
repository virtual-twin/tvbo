#!/usr/bin/env python
"""Backfill empirical-nomenclature ``alternateName`` aliases onto atlas terminologies.

A model network labels its regions with an atlas's canonical names; empirical per-subject pipelines (e.g. the HCP-YA functional-connectome pipeline) label the SAME parcels with a different string convention. ``by_label`` node reconciliation aligns them by label, so unless the atlas records the alternate names the two conventions never meet on the string alone.

This tool derives each region's empirical alias from its canonical label via a documented, per-atlas deterministic rule and records it in that region's SANDS ``alternateName`` list (appending, never clobbering existing aliases). The rule runs ONCE here; the atlas YAML is the source of truth thereafter and the resolver does a pure alias-aware string match — no normalization in the hot path. Each atlas is
**bijection- and hemisphere-parity-checked against a real subject sidecar** (when the
data volume is reachable), so a rule error surfaces immediately as a non-bijection or a hemisphere swap rather than a silent mis-map.

Atlases live under ``tvbo/database/atlases`` (resolved via ``ATLAS_DIR``).

Usage:
    python scripts/backfill_atlas_aliases.py [--check] [--atlas NAME ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML

from tvbo.data.tvbo_data import ATLAS_DIR

ATLAS_ROOT = Path(ATLAS_DIR)
# HCP-YA functional-connectome tree used to cross-check the derived aliases.
_FC_ROOT = Path("/Volumes/bronkodata/hcp/derivatives/hcp_ya/functional_connectomes")


# ── hemisphere (convention-agnostic, from the label string only) ──────────────
def hemisphere(label: str) -> str | None:
    """Hemisphere of a label under ANY supported convention, or None if bilateral.

    Reads the hemisphere from the string alone (never from a node index) so the crosswalk can be checked for parity: a canonical left-hemisphere region and its empirical alias must resolve to the SAME hemisphere.
    """
    if label in ("Brain-Stem", "BRAIN_STEM", "brain-stem"):
        return None
    if label[:2] == "L_" or label.endswith("_LEFT") or label.startswith(("ctx-lh-", "left-")):
        return "L"
    if label[:2] == "R_" or label.endswith("_RIGHT") or label.startswith(("ctx-rh-", "right-")):
        return "R"
    return None


# ── per-atlas canonical -> empirical rules ────────────────────────────────────
_HCPMMP1_SUBCORTICAL = {
    "Cerebellum",
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Hippocampus",
    "Amygdala",
    "Accumbens",
    "VentralDC",
}
_HCPMMP1_EMPIRICAL_STEM = {"VentralDC": "DIENCEPHALON_VENTRAL"}


def hcpmmp1_rule(canon: str) -> str:
    """HCP-MMP1 canonical -> empirical (functional_connectomes atlas-HCPMMP1)."""
    if canon == "Brain-Stem":
        return "BRAIN_STEM"
    if canon[:2] in ("L_", "R_"):
        hemi, stem = canon[0], canon[2:]
        if stem in _HCPMMP1_SUBCORTICAL:
            side = "LEFT" if hemi == "L" else "RIGHT"
            return f"{_HCPMMP1_EMPIRICAL_STEM.get(stem, stem.upper())}_{side}"
        return f"{canon}_ROI"  # cortical parcel
    raise ValueError(f"Unrecognised HCP-MMP1 label: {canon!r}")


# DesikanKilliany canonical subcortical stem -> empirical stem.
_DK_SUBCORTICAL_STEM = {
    "cerebellum-cortex": "CEREBELLUM",
    "thalamus": "THALAMUS",
    "caudate": "CAUDATE",
    "putamen": "PUTAMEN",
    "pallidum": "PALLIDUM",
    "hippocampus": "HIPPOCAMPUS",
    "amygdala": "AMYGDALA",
    "accumbens-area": "ACCUMBENS",
    "ventraldc": "DIENCEPHALON_VENTRAL",
}


def desikankilliany_rule(canon: str) -> str:
    """DesikanKilliany canonical -> empirical (functional_connectomes atlas-aparcaseg)."""
    if canon == "brain-stem":
        return "BRAIN_STEM"
    if canon.startswith("ctx-lh-"):
        return f"L_{canon[len('ctx-lh-') :]}"
    if canon.startswith("ctx-rh-"):
        return f"R_{canon[len('ctx-rh-') :]}"
    for prefix, side in (("left-", "LEFT"), ("right-", "RIGHT")):
        if canon.startswith(prefix):
            stem = canon[len(prefix) :]
            if stem not in _DK_SUBCORTICAL_STEM:
                raise ValueError(f"Unknown DK subcortical stem: {stem!r} (in {canon!r})")
            return f"{_DK_SUBCORTICAL_STEM[stem]}_{side}"
    raise ValueError(f"Unrecognised DesikanKilliany label: {canon!r}")


ATLASES = {
    "hcpmmp1": {
        "file": "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_dseg.yaml",
        "rule": hcpmmp1_rule,
        "verify_glob": "sub-*/sub-*atlas-HCPMMP1*relmat.yaml",
    },
    "DesikanKilliany": {
        "file": "tpl-MNI152NLin2009c_atlas-DesikanKilliany_desc-ranked_dseg.yaml",
        "rule": desikankilliany_rule,
        "verify_glob": "sub-*/sub-*atlas-aparcaseg*relmat.yaml",
    },
}


def _empirical_labels(glob: str) -> list[str] | None:
    """Node labels of one empirical sidecar matching *glob*, or None if unreachable."""
    if not _FC_ROOT.exists():
        return None
    hits = sorted(_FC_ROOT.glob(glob))
    if not hits:
        return None
    y = YAML(typ="safe")
    data = y.load(hits[0].read_text())
    return [str(n["label"]) for n in (data.get("nodes") or []) if isinstance(n, dict) and "label" in n]


def process_atlas(name: str, spec: dict, check: bool) -> int:
    path = ATLAS_ROOT / spec["file"]
    if not path.exists():
        print(f"FAIL[{name}]: atlas file not found: {path}", file=sys.stderr)
        return 2
    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(path.read_text())
    entities = data["terminology"]["entities"]
    canon_labels = list(entities.keys())
    aliases = {c: spec["rule"](c) for c in canon_labels}

    # 1) the derived aliases are themselves 1-to-1
    if len(set(aliases.values())) != len(aliases):
        dupes = sorted({a for a in aliases.values() if list(aliases.values()).count(a) > 1})
        print(f"FAIL[{name}]: derived-alias collisions {dupes}", file=sys.stderr)
        return 2
    # 2) hemisphere parity — the L<->R swap guard, from strings alone
    swapped = [(c, a) for c, a in aliases.items() if hemisphere(c) != hemisphere(a)]
    if swapped:
        print(f"FAIL[{name}]: {len(swapped)} hemisphere mismatches, e.g. {swapped[:3]}", file=sys.stderr)
        return 2
    # 3) exact bijection with real empirical labels, when reachable
    empirical = _empirical_labels(spec["verify_glob"])
    if empirical is not None:
        missing, extra = set(empirical) - set(aliases.values()), set(aliases.values()) - set(empirical)
        if missing or extra:
            print(f"FAIL[{name}]: not a bijection with the empirical sidecar", file=sys.stderr)
            if missing:
                print(f"  empirical labels unmatched by the rule: {sorted(missing)}", file=sys.stderr)
            if extra:
                print(f"  rule outputs absent from the data:      {sorted(extra)}", file=sys.stderr)
            return 2
        print(f"[{name}] bijection OK: {len(empirical)}/{len(empirical)} vs empirical sidecar")
    else:
        print(f"[{name}] NOTE: no empirical sidecar reachable; wrote rule-derived aliases without a data cross-check")

    # One pass over the entities: append the empirical alias to each entity's alternateName (keep existing, dedup), flag which entities drift, and check global alias uniqueness (region_alias_map must never see one alias map to two regions).
    drift = []
    merged_by_c: dict = {}
    final_index: dict[str, str] = {}
    for c in canon_labels:
        cur = list(entities[c].get("alternateName") or [])
        merged = cur if aliases[c] in cur else cur + [aliases[c]]
        merged_by_c[c] = merged
        if aliases[c] not in cur:
            drift.append(c)
        for alias in merged:
            prev = final_index.get(alias)
            if prev is not None and prev != c:
                print(
                    f"FAIL[{name}]: alias {alias!r} maps to both {prev!r} and {c!r} (pre-existing crosswalk collision)",
                    file=sys.stderr,
                )
                return 2
            final_index[alias] = c

    rel = path.relative_to(ATLAS_ROOT.parents[2]) if len(ATLAS_ROOT.parents) >= 3 else path
    if not drift:
        print(f"[{name}] up to date: all {len(canon_labels)} entities carry their empirical alias")
        return 0
    if check:
        print(f"[{name}] DRIFT: {len(drift)} entities need the empirical alias (e.g. {drift[:3]})", file=sys.stderr)
        return 1
    for c in drift:
        entities[c]["alternateName"] = merged_by_c[c]
    with path.open("w") as f:
        yaml.dump(data, f)
    print(f"[{name}] wrote empirical alias for {len(drift)} entities -> {rel}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    ap.add_argument("--atlas", nargs="*", default=list(ATLASES), help=f"atlases to process (default: all of {list(ATLASES)})")
    args = ap.parse_args()
    rc = 0
    for name in args.atlas:
        if name not in ATLASES:
            print(f"unknown atlas {name!r}; known: {list(ATLASES)}", file=sys.stderr)
            rc = 2
            continue
        rc = max(rc, process_atlas(name, ATLASES[name], args.check))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
