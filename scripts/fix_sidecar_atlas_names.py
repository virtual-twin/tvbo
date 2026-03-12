#!/usr/bin/env python3
"""Fix atlas names and bids metadata in YAML sidecars after the rename.

Updates parcellation.atlas.name to normalized form and adds segmentation/scale
to the bids: section where applicable.
"""
import re
import yaml
from pathlib import Path
from tvbo import database_path

NET_DIR = database_path / "networks"

# Atlas name normalization: old → new
ATLAS_NORMALIZE = {
    "Schaefer100017Networks": "Schaefer2018",
    "Schaefer1000": "Schaefer2018",
    "hcpmmp1": "HCPMMP1",
    "hcpmmp1ordered": "HCPMMP1",
    "DesikanKillianyranked": "DesikanKilliany",
    "Destrieuxranked": "Destrieux",
}

# Space normalization
SPACE_NORMALIZE = {
    "MNI152Nlin2009c": "MNI152NLin2009cAsym",
    "MNI152NLin2009c": "MNI152NLin2009cAsym",
    "MNI152NLin2009b": "MNI152NLin2009bAsym",
    "MNI152": "MNI152NLin2009cAsym",
}


def fix_sidecar(yaml_path):
    """Fix atlas name and bids metadata in a single sidecar."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f.read())

    changed = False
    stem = yaml_path.stem

    # Fix parcellation.atlas.name
    parc = data.get("parcellation", {})
    atlas = parc.get("atlas", {})
    old_name = atlas.get("name", "")
    if old_name in ATLAS_NORMALIZE:
        atlas["name"] = ATLAS_NORMALIZE[old_name]
        changed = True

    # Fix parcellation.atlas.coordinateSpace
    old_space = atlas.get("coordinateSpace", "")
    if old_space in SPACE_NORMALIZE:
        atlas["coordinateSpace"] = SPACE_NORMALIZE[old_space]
        changed = True

    # Fix bids section: add segmentation and scale from filename
    bids = data.get("bids", {})
    if bids:
        # Extract seg from filename
        seg_match = re.search(r"seg-([^_]+)", stem)
        if seg_match and "segmentation" not in bids:
            bids["segmentation"] = seg_match.group(1)
            changed = True

        # Extract scale from filename
        scale_match = re.search(r"scale-([^_]+)", stem)
        if scale_match and "scale" not in bids:
            bids["scale"] = scale_match.group(1)
            changed = True

        data["bids"] = bids

    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        print(f"  Fixed: {yaml_path.name}")
    return changed


def main():
    count = 0
    for f in sorted(NET_DIR.glob("tpl-*.yaml")):
        if fix_sidecar(f):
            count += 1
    print(f"\nFixed {count} sidecars.")


if __name__ == "__main__":
    main()
