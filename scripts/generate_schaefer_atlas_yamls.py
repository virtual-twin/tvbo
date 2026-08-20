#!/usr/bin/env python3
"""Generate SANDS-compliant Schaefer atlas YAMLs in database/atlases/.

Creates one atlas YAML per (segmentation, scale) combination, with:
- ParcellationEntity entries for each parcel
- ParcellationEntity entries for each functional network (as parent)
- hasParent references linking parcels to their functional network

The functional network hierarchy (7 or 17 networks × 2 hemispheres) is represented purely via the SANDS ``hasParent`` field — no ad-hoc fields.

Also removes the non-schema ``functional_networks`` field from network YAMLs and ensures ``node_mapping`` stays compliant.

Usage:
    python scripts/generate_schaefer_atlas_yamls.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ATLAS_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas"
TARGET_ATLAS_DIR = ROOT / "tvbo" / "database" / "atlases"
NETWORK_DIR = ROOT / "tvbo" / "database" / "networks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SANDS-compliant Schaefer atlas YAMLs.",
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ATLAS_DIR)
    parser.add_argument("--target-dir", type=Path, default=TARGET_ATLAS_DIR)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_functional_network(region_name: str) -> str:
    """``"17Networks_LH_VisCent_1"`` → ``"LH_VisCent"``."""
    parts = region_name.split("_")
    return parts[1] + "_" + parts[2]


def compute_network_centroids(
    entities: list[dict],
    net_labels: list[str],
    net_assignments: list[str],
) -> dict[str, dict[str, float]]:
    """Compute mean centroid per functional network from parcel centers."""
    sums = {n: np.zeros(3) for n in net_labels}
    counts = {n: 0 for n in net_labels}
    for entity, net in zip(entities, net_assignments, strict=True):
        c = entity.get("center", {})
        if c:
            sums[net] += np.array([float(c["x"]), float(c["y"]), float(c["z"])])
            counts[net] += 1
    centroids = {}
    for n in net_labels:
        if counts[n] > 0:
            mean = sums[n] / counts[n]
            centroids[n] = {
                "x": round(float(mean[0]), 1),
                "y": round(float(mean[1]), 1),
                "z": round(float(mean[2]), 1),
            }
    return centroids


def build_atlas_yaml(source_path: Path) -> dict:
    """Build a SANDS-compliant atlas dict from source atlas YAML."""
    with open(source_path) as f:
        src = yaml.safe_load(f)

    src_entities = src.get("terminology", {}).get("entities", {})
    sorted_entities = sorted(src_entities.values(), key=lambda e: int(e["lookupLabel"]))

    # Determine functional network assignments
    net_assignments = [extract_functional_network(e["name"]) for e in sorted_entities]
    net_labels = list(dict.fromkeys(net_assignments))  # preserve order

    # Compute centroids for parent network entities
    net_centroids = compute_network_centroids(sorted_entities, net_labels, net_assignments)

    # Determine hemispheres for networks
    def hemi_from_net(net: str) -> str:
        return "left" if net.startswith("LH_") else "right"

    # Build parent network entities (negative lookupLabels to distinguish)
    parent_entities = {}
    for i, net in enumerate(net_labels):
        parent_entity = {
            "name": net,
            "lookupLabel": -(i + 1),
            "hemisphere": hemi_from_net(net),
        }
        if net in net_centroids:
            parent_entity["center"] = net_centroids[net]
        parent_entities[net] = parent_entity

    # Build parcel entities with hasParent
    parcel_entities = {}
    for entity, net in zip(sorted_entities, net_assignments, strict=True):
        entry = {
            "name": entity["name"],
            "lookupLabel": int(entity["lookupLabel"]),
        }
        if "originalLookupLabel" in entity:
            entry["originalLookupLabel"] = int(entity["originalLookupLabel"])
        if "center" in entity:
            entry["center"] = {
                "x": float(entity["center"]["x"]),
                "y": float(entity["center"]["y"]),
                "z": float(entity["center"]["z"]),
            }
        if "color" in entity:
            entry["color"] = entity["color"]
        if "hemisphere" in entity:
            entry["hemisphere"] = entity["hemisphere"]
        entry["hasParent"] = net
        parcel_entities[entity["name"]] = entry

    # Merge: parent networks first, then parcels
    all_entities = {}
    all_entities.update(parent_entities)
    all_entities.update(parcel_entities)

    atlas = {
        "name": src.get("name", "Schaefer2018"),
        "abbreviation": src.get("abbreviation", "Schaefer2018"),
        "versionIdentifier": src.get("versionIdentifier", "2018"),
        "coordinateSpace": src.get(
            "coordinateSpace",
            {
                "name": "MNI152",
                "abbreviation": "FSLMNI152",
                "nativeUnit": "mm",
            },
        ),
        "terminology": {
            "label": src.get("terminology", {}).get("label", ""),
            "entities": all_entities,
        },
    }
    if src.get("terminology", {}).get("versionIdentifier"):
        atlas["terminology"]["versionIdentifier"] = src["terminology"]["versionIdentifier"]
    if src.get("terminology", {}).get("dataLocation"):
        atlas["terminology"]["dataLocation"] = src["terminology"]["dataLocation"]

    return atlas


def target_filename(source_name: str) -> str:
    """Convert a source filename to the target naming convention.

    The ``_res-1_desc-ordered_`` segment is dropped:

    ```
    source: tpl-FSLMNI152_atlas-Schaefer2018_seg-7Networks_scale-100_res-1_desc-ordered_dseg.yaml
    target: tpl-FSLMNI152_atlas-Schaefer2018_seg-7Networks_scale-100_dseg.yaml
    ```
    """
    return source_name.replace("_res-1_desc-ordered_", "_")


def remove_functional_networks_from_yamls(network_dir: Path, dry_run: bool) -> int:
    """Remove non-schema ``functional_networks`` from network YAMLs."""
    count = 0
    for yaml_path in sorted(network_dir.glob("*Schaefer2018*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if "functional_networks" not in data:
            continue
        del data["functional_networks"]
        count += 1
        if not dry_run:
            with open(yaml_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"  [fix ] removed functional_networks from {yaml_path.name}")
    return count


def main() -> None:
    args = parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate atlas YAMLs
    source_files = sorted(args.source_dir.glob("tpl-FSLMNI152_atlas-Schaefer2018_seg-*_scale-*_res-*_desc-ordered_dseg.yaml"))
    if not source_files:
        print(f"No source atlas files found in {args.source_dir}")
        return

    created = 0
    skipped = 0
    for src_path in source_files:
        out_name = target_filename(src_path.name)
        out_path = args.target_dir / out_name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {out_name}")
            continue

        atlas = build_atlas_yaml(src_path)
        if args.dry_run:
            print(f"[dry ] would create {out_name}")
        else:
            with open(out_path, "w") as f:
                yaml.safe_dump(atlas, f, sort_keys=False, allow_unicode=True)
            print(f"[ok  ] {out_name}")
        created += 1

    # 2. Remove non-schema functional_networks from network YAMLs
    print("\nCleaning network YAMLs...")
    fixed = remove_functional_networks_from_yamls(args.network_dir, args.dry_run)

    print("\nDone.")
    print(f"  atlases created: {created:d}")
    print(f"  atlases skipped: {skipped:d}")
    print(f"  networks fixed:  {fixed:d}")
    if args.dry_run:
        print("  (dry run — no files written)")


if __name__ == "__main__":
    main()
