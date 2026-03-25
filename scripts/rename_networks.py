#!/usr/bin/env python3
"""Rename all network files in database/networks/ to match the BIDS BEP017
naming convention from the tvbo HDF5 format proposal v0.7 (§6.5).

Mapping rules:
- space-<X>  →  tpl-<X>  (these are normative, template-level connectomes)
- desc-<tractogram>  →  rec-<tractogram>_desc-SC  (tractogram = reconstruction)
- Add _relmat suffix (BEP017)
- Atlas names: hcpmmp1 → HCPMMP1, hcpmmp1ordered → HCPMMP1_seg-ordered
- Schaefer1000 → Schaefer2018_scale-1000
- Schaefer100017Networks → Schaefer2018_seg-17Networks_scale-1000
- DesikanKillianyranked → DesikanKilliany_seg-ranked
- Destrieuxranked → Destrieux_seg-ranked
- virtualdbs → kept as-is (no atlas metadata)
- Cohort: dTOR → HCPYA; MghUscHcp32 / PPMI85 → same as tractogram name

The "tpl-MghUscHcp32" files in old naming were misusing tpl- for tractogram:
  space-MNI152Nlin2009c_tpl-MghUscHcp32_atlas-DesikanKilliany_desc-ranked
  → tpl-MNI152NLin2009cAsym_cohort-MghUscHcp32_rec-MghUscHcp32_atlas-DesikanKilliany_seg-ranked_desc-SC_relmat

Usage:
    python scripts/rename_networks.py --dry-run   # Preview only
    python scripts/rename_networks.py              # Execute renames
"""
import os
import re
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NET_DIR = ROOT / "database" / "networks"

# Tractogram → cohort mapping
COHORT_MAP = {
    "dTOR": "HCPYA",
    "MghUscHcp32": "MghUscHcp32",
    "PPMI85": "PPMI85",
}

# Space normalization (→ consistent MNI naming)
SPACE_MAP = {
    "MNI152Nlin2009c": "MNI152NLin2009cAsym",
    "MNI152NLin2009c": "MNI152NLin2009cAsym",
    "MNI152NLin2009cAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009b": "MNI152NLin2009bAsym",
    "FSLMNI152": "FSLMNI152",
    "MNI152": "MNI152NLin2009cAsym",
}


def parse_old_name(stem):
    """Parse the old-style filename into entities."""
    entities = {}

    # Handle the weird tpl-tractogram files first
    # e.g., space-MNI152Nlin2009c_tpl-MghUscHcp32_atlas-DesikanKilliany_desc-ranked
    m = re.match(
        r"space-(?P<space>[^_]+)_tpl-(?P<tpl>[^_]+)_atlas-(?P<atlas>[^_]+)_desc-(?P<desc>.+)",
        stem
    )
    if m:
        entities["space"] = m.group("space")
        entities["tractogram"] = m.group("tpl")
        entities["atlas_raw"] = m.group("atlas")
        entities["variant"] = m.group("desc")
        return entities

    # Standard: space-X_atlas-Y_desc-Z
    m = re.match(
        r"space-(?P<space>[^_]+)_atlas-(?P<atlas>[^_]+)_desc-(?P<desc>.+)",
        stem
    )
    if m:
        entities["space"] = m.group("space")
        entities["atlas_raw"] = m.group("atlas")
        entities["tractogram"] = m.group("desc")
        return entities

    return None


def normalize_atlas(atlas_raw, variant=None):
    """Return (atlas, seg, scale) tuple."""
    atlas = atlas_raw
    seg = variant if variant and variant not in COHORT_MAP else None
    scale = None

    # hcpmmp1 → HCPMMP1
    if atlas_raw == "hcpmmp1":
        atlas = "HCPMMP1"
    elif atlas_raw == "hcpmmp1ordered":
        atlas = "HCPMMP1"
        seg = "ordered"

    # Schaefer handling
    elif atlas_raw == "Schaefer1000":
        atlas = "Schaefer2018"
        scale = "1000"
    elif atlas_raw == "Schaefer100017Networks":
        atlas = "Schaefer2018"
        seg = "17Networks"
        scale = "1000"

    # Ranked variants
    elif atlas_raw == "DesikanKillianyranked":
        atlas = "DesikanKilliany"
        seg = "ranked"
    elif atlas_raw == "Destrieuxranked":
        atlas = "Destrieux"
        seg = "ranked"
    elif atlas_raw == "DesikanKilliany":
        atlas = "DesikanKilliany"
    elif atlas_raw == "Destrieux":
        atlas = "Destrieux"

    # Handle seg override from variant
    if variant == "ranked":
        seg = "ranked"
    elif variant == "17Networks":
        seg = "17Networks"

    return atlas, seg, scale


def build_new_name(entities):
    """Build BIDS BEP017-compliant filename stem."""
    space = SPACE_MAP.get(entities["space"], entities["space"])
    tractogram = entities["tractogram"]
    cohort = COHORT_MAP.get(tractogram, tractogram)
    atlas_raw = entities["atlas_raw"]
    variant = entities.get("variant")

    atlas, seg, scale = normalize_atlas(atlas_raw, variant)

    parts = [f"tpl-{space}"]
    parts.append(f"cohort-{cohort}")
    parts.append(f"rec-{tractogram}")
    parts.append(f"atlas-{atlas}")
    if seg:
        parts.append(f"seg-{seg}")
    if scale:
        parts.append(f"scale-{scale}")
    parts.append("desc-SC")
    parts.append("relmat")

    return "_".join(parts)


def build_rename_map():
    """Build {old_stem: new_stem} for all YAML files.

    Also returns a set of stems to DELETE (exact duplicates under old naming).
    """
    renames = {}
    for f in sorted(NET_DIR.glob("*.yaml")):
        stem = f.stem
        if stem == "example_3node_network":
            continue  # skip non-standard file
        entities = parse_old_name(stem)
        if entities is None:
            print(f"WARNING: Cannot parse: {stem}", file=sys.stderr)
            continue
        new_stem = build_new_name(entities)
        if stem != new_stem:
            renames[stem] = new_stem

    # Detect collisions — keep the standard-pattern file, delete tpl-* duplicates
    from collections import Counter
    target_counts = Counter(renames.values())
    duplicates = set()
    for target, count in target_counts.items():
        if count > 1:
            sources = [k for k, v in renames.items() if v == target]
            # Keep the one WITHOUT "tpl-" in old name; delete the tpl- variant
            for s in sources:
                if "_tpl-" in s:
                    duplicates.add(s)
                    del renames[s]

    return renames, duplicates


def update_sidecar(yaml_path, new_stem):
    """Update data_file reference in YAML sidecar to match new names."""
    with open(yaml_path) as f:
        content = f.read()

    data = yaml.safe_load(content)
    old_data_file = data.get("data_file", "")

    # Update data_file to new name
    new_data_file = new_stem + ".h5"
    content = content.replace(old_data_file, new_data_file)

    with open(yaml_path, "w") as f:
        f.write(content)


def main():
    dry_run = "--dry-run" in sys.argv

    renames, duplicates = build_rename_map()

    if not renames and not duplicates:
        print("No renames needed.")
        return

    if duplicates:
        print(f"{'DRY RUN: ' if dry_run else ''}Deleting {len(duplicates)} duplicate file pairs:\n")
        for stem in sorted(duplicates):
            print(f"  DELETE: {stem} (.yaml + .h5)")
            if not dry_run:
                for ext in (".yaml", ".h5"):
                    p = NET_DIR / f"{stem}{ext}"
                    if p.exists():
                        p.unlink()
        print()

    print(f"{'DRY RUN: ' if dry_run else ''}Renaming {len(renames)} file pairs:\n")

    for old_stem, new_stem in sorted(renames.items()):
        print(f"  {old_stem}")
        print(f"  → {new_stem}")
        print()

        if not dry_run:
            # Rename .yaml
            old_yaml = NET_DIR / f"{old_stem}.yaml"
            new_yaml = NET_DIR / f"{new_stem}.yaml"
            if old_yaml.exists():
                old_yaml.rename(new_yaml)

            # Rename .h5
            old_h5 = NET_DIR / f"{old_stem}.h5"
            new_h5 = NET_DIR / f"{new_stem}.h5"
            if old_h5.exists():
                old_h5.rename(new_h5)

            # Update data_file in sidecar
            if new_yaml.exists():
                update_sidecar(new_yaml, new_stem)

    print(f"\n{'Would process' if dry_run else 'Processed'} {len(renames)} renames + {len(duplicates)} deletions.")


if __name__ == "__main__":
    main()
