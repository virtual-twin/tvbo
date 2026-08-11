#!/usr/bin/env python3
"""Update YAML sidecar contents to match the tvbo HDF5 format proposal v0.7.

Changes per file:
1. Add `descriptor: SC` (maps to desc- BIDS entity) 2. Add `bids:` section with template, cohort, reconstruction entities 3. Convert edge `label: weights` → `name: weights` (template edge naming) 4. Remove redundant `number_of_regions:` (same as number_of_nodes) 5. Normalize parcellation.atlas.coordinateSpace to consistent naming"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NET_DIR = ROOT / "database" / "networks"


def parse_bids_entities(stem):
    """Parse BIDS entities from a BIDS-compliant filename stem."""
    entities = {}
    # Match key-value pairs like tpl-X, cohort-Y, etc.
    for m in re.finditer(r"(tpl|cohort|rec|atlas|seg|scale|desc)-([^_]+)", stem):
        entities[m.group(1)] = m.group(2)
    return entities


def update_sidecar(yaml_path):
    """Update a single YAML sidecar file."""
    stem = yaml_path.stem
    if stem == "example_3node_network":
        return False

    entities = parse_bids_entities(stem)
    if not entities:
        return False

    with open(yaml_path) as f:
        content = f.read()

    original = content

    # 1. Add descriptor: SC after label line (if not present)
    if "descriptor:" not in content:
        # Insert after label line
        content = re.sub(
            r"(label: .+\n)",
            r"\1descriptor: SC\n",
            content,
            count=1,
        )

    # 2. Remove number_of_regions (redundant with number_of_nodes)
    content = re.sub(r"number_of_regions: \d+\n", "", content)

    # 3. Template edges (weights, lengths) key on "name:", the HDF5 group path, not "label:".
    content = re.sub(
        r"(- )label: (weights|lengths)",
        r"\1name: \2",
        content,
    )

    # 4. Add bids: section before parcellation: (if not present)
    if "bids:" not in content:
        bids_lines = [
            "bids:",
            '  bids_version: "1.10.0"',
            "  dataset_type: derivative",
        ]
        if "tpl" in entities:
            bids_lines.append(f"  template: {entities['tpl']}")
        if "cohort" in entities:
            bids_lines.append(f"  cohort: {entities['cohort']}")
        if "rec" in entities:
            bids_lines.append(f"  reconstruction: {entities['rec']}")

        bids_block = "\n".join(bids_lines) + "\n"

        # Insert before parcellation:
        content = content.replace(
            "parcellation:\n",
            bids_block + "parcellation:\n",
        )

    # 5. Normalize coordinateSpace
    content = content.replace(
        "coordinateSpace: MNI152NLin2009c\n",
        "coordinateSpace: MNI152NLin2009cAsym\n",
    )
    content = content.replace(
        "coordinateSpace: MNI152Nlin2009c\n",
        "coordinateSpace: MNI152NLin2009cAsym\n",
    )

    if content != original:
        with open(yaml_path, "w") as f:
            f.write(content)
        return True
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    updated = 0

    for yaml_path in sorted(NET_DIR.glob("*.yaml")):
        if yaml_path.stem == "example_3node_network":
            continue

        if dry_run:
            entities = parse_bids_entities(yaml_path.stem)
            print(f"Would update: {yaml_path.name}")
            print(f"  entities: {entities}")
        else:
            if update_sidecar(yaml_path):
                updated += 1
                print(f"Updated: {yaml_path.name}")

    print(f"\n{'Would update' if dry_run else 'Updated'} {updated if not dry_run else 'all'} sidecars.")


if __name__ == "__main__":
    main()
