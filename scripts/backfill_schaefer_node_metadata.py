#!/usr/bin/env python3
"""Backfill node labels and positions in Schaefer SC YAML sidecars.

Reads each `database/networks/tpl-FSLMNI152_*Schaefer2018*_relmat.yaml`, looks up the corresponding SANDS atlas YAML (from `tvbo/data/tvbo_data/atlas/`),
and patches every node entry with:
  - label: <actual region name, e.g. "17Networks_LH_Vis_1">
  - position: {x: ..., y: ..., z: ...}   (centroid in MNI RAS mm)

The companion HDF5 file is NOT touched (matrices are correct as-is).

Usage:
    python scripts/backfill_schaefer_node_metadata.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NETWORKS_DIR = ROOT / "database" / "networks"
ATLAS_DIR = ROOT / "tvbo" / "database" / "atlases"


def load_atlas_entities(seg: str, scale: str, resolution: str = "1") -> list[dict] | None:
    """Return atlas entities sorted by lookupLabel, or None if atlas YAML not found."""
    fname = f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-{resolution}_desc-ordered_dseg.yaml"
    path = ATLAS_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    entities = data.get("terminology", {}).get("entities", {})
    return sorted(entities.values(), key=lambda e: e["lookupLabel"])


def parse_filename_entities(stem: str) -> dict[str, str]:
    """Extract BIDS entities from a network YAML stem."""
    entities: dict[str, str] = {}
    for key, val in re.findall(r"([a-z]+)-([^_]+)", stem):
        entities[key] = val
    return entities


def backfill(path: Path, dry_run: bool) -> bool:
    """Patch one YAML sidecar. Returns True if the file was modified."""
    with open(path) as f:
        data = yaml.safe_load(f)

    # Determine seg/scale from filename entities
    entities = parse_filename_entities(path.stem)
    seg = entities.get("seg")  # e.g. "17Networks"
    scale = entities.get("scale")  # e.g. "100"

    if not seg or not scale:
        print(f"  [skip] {path.name}: cannot determine seg/scale from filename")
        return False

    atlas_entities = load_atlas_entities(seg, scale)
    if atlas_entities is None:
        print(f"  [skip] {path.name}: atlas YAML not found for seg={seg} scale={scale}")
        return False

    nodes = data.get("nodes")
    if not nodes:
        print(f"  [skip] {path.name}: no nodes section")
        return False

    if len(nodes) != len(atlas_entities):
        print(f"  [warn] {path.name}: node count mismatch (network={len(nodes)}, atlas={len(atlas_entities)}) — skipping")
        return False

    changed = False
    for node, entity in zip(nodes, atlas_entities):
        new_label = entity["name"]
        c = entity.get("center", {})
        new_pos = {"x": float(c["x"]), "y": float(c["y"]), "z": float(c["z"])} if c else None

        if node.get("label") != new_label:
            node["label"] = new_label
            changed = True

        existing_pos = node.get("position")
        if new_pos and existing_pos != new_pos:
            node["position"] = new_pos
            changed = True

    if not changed:
        print(f"  [ok  ] {path.name}: already up to date")
        return False

    if dry_run:
        print(f"  [dry ] {path.name}: would patch {len(nodes)} nodes")
        return True

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  [patch] {path.name}: patched {len(nodes)} nodes")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill node labels/positions in Schaefer SC YAML sidecars.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("--dir", type=Path, default=NETWORKS_DIR, help="Networks directory.")
    args = parser.parse_args()

    yamls = sorted(args.dir.glob("*Schaefer2018*_relmat.yaml"))
    if not yamls:
        print(f"No Schaefer2018 relmat YAML files found in {args.dir}")
        sys.exit(1)

    print(f"Found {len(yamls)} Schaefer SC YAML sidecars in {args.dir}")
    patched = 0
    for path in yamls:
        if backfill(path, args.dry_run):
            patched += 1

    print(f"\nDone. {'Would patch' if args.dry_run else 'Patched'}: {patched}/{len(yamls)}")


if __name__ == "__main__":
    main()
