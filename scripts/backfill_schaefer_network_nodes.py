#!/usr/bin/env python3
"""Backfill Schaefer network node metadata from atlas YAML sidecars.

This script updates generated Schaefer network YAML files so that:
- nodes[].label comes from atlas terminology entity names
- nodes[].position comes from atlas entity centers

It does not touch HDF5 companions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas"
NETWORK_DIR = ROOT / "database" / "networks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill node labels/positions in Schaefer network YAML files.")
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--atlas-dir", type=Path, default=ATLAS_DIR)
    parser.add_argument("--tractogram", default="dTOR")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_entities(atlas_yaml: Path) -> list[dict]:
    with open(atlas_yaml) as f:
        data = yaml.safe_load(f)
    entities = data.get("terminology", {}).get("entities", {})
    return sorted(entities.values(), key=lambda e: int(e["lookupLabel"]))


def atlas_yaml_path(atlas_dir: Path, seg: str, scale: str) -> Path:
    return atlas_dir / (f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-1_desc-ordered_dseg.yaml")


def update_network_yaml(network_yaml: Path, atlas_dir: Path, overwrite: bool) -> bool:
    with open(network_yaml) as f:
        net = yaml.safe_load(f)

    bids = net.get("bids", {})
    seg = bids.get("segmentation")
    scale = bids.get("scale")
    if not seg or not scale:
        return False

    atlas_yaml = atlas_yaml_path(atlas_dir, seg, str(scale))
    if not atlas_yaml.exists():
        raise FileNotFoundError(f"Missing atlas metadata: {atlas_yaml}")

    entities = parse_entities(atlas_yaml)
    nodes = net.get("nodes", [])
    if len(nodes) != len(entities):
        raise ValueError(f"Node/entity size mismatch for {network_yaml.name}: {len(nodes)} nodes vs {len(entities)} entities")

    changed = False
    for node, entity in zip(nodes, entities):
        label = entity.get("name")
        center = entity.get("center", {})
        position = {
            "x": float(center["x"]),
            "y": float(center["y"]),
            "z": float(center["z"]),
        }
        if node.get("label") != label:
            node["label"] = label
            changed = True
        if node.get("position") != position:
            node["position"] = position
            changed = True

    if changed and overwrite:
        with open(network_yaml, "w") as f:
            yaml.safe_dump(net, f, sort_keys=False)
    return changed


def main() -> None:
    args = parse_args()
    pattern = f"tpl-FSLMNI152_cohort-HCPYA_rec-{args.tractogram}_atlas-Schaefer2018_seg-*Networks_scale-*_desc-SC_relmat.yaml"
    files = sorted(args.network_dir.glob(pattern))

    updated = 0
    unchanged = 0
    for path in files:
        changed = update_network_yaml(path, args.atlas_dir, args.overwrite)
        if changed:
            updated += 1
            print(f"[upd ] {path.name}")
        else:
            unchanged += 1
            print(f"[ok  ] {path.name}")

    print("\nDone.")
    print(f"  files:     {len(files)}")
    print(f"  updated:   {updated}")
    print(f"  unchanged: {unchanged}")


if __name__ == "__main__":
    main()
