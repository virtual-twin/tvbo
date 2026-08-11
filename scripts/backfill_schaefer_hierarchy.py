#!/usr/bin/env python3
"""Backfill Schaefer network YAMLs with multi-layer hierarchy and correct labels.

For each Schaefer network YAML in database/networks/:

1. **Fix node labels & positions** — replace ``node_0``-style labels with actual region names from the corresponding atlas YAML sidecar.

2. **Add functional-network hierarchy** — write a ``node_mapping`` array to the HDF5 companion that maps each parcel index to its functional
   network index (7 or 17 × 2 hemispheres).  The YAML sidecar gets
   ``node_mapping`` and ``parent_network`` fields pointing to that data.

The multi-layer structure allows tvbo to treat Schaefer networks as hierarchical: base nodes (parcels) are mapped to a higher-order layer
of functional networks (e.g. ``LH_Vis``, ``RH_Default``, …).

Usage:
    python scripts/backfill_schaefer_hierarchy.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas"
NETWORK_DIR = ROOT / "tvbo" / "database" / "networks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Schaefer network YAMLs with hierarchy and labels.",
    )
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--atlas-dir", type=Path, default=ATLAS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


# ── Atlas helpers ─────────────────────────────────────────────────────


def load_atlas_entities(atlas_dir: Path, seg: str, scale: str) -> list[dict] | None:
    """Load atlas entities sorted by lookupLabel, or None if not found."""
    path = atlas_dir / (f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-1_desc-ordered_dseg.yaml")
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    entities = data.get("terminology", {}).get("entities", {})
    return sorted(entities.values(), key=lambda e: int(e["lookupLabel"]))


def extract_functional_network(region_name: str) -> str:
    """Extract ``{hemi}_{network}`` from a Schaefer region name.

    ``"17Networks_LH_VisCent_1"`` → ``"LH_VisCent"``
    """
    parts = region_name.split("_")
    return f"{parts[1]}_{parts[2]}"


def build_node_mapping(entities: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build parcel → functional-network index mapping.

    Returns (mapping_array, sorted_network_labels).
    """
    parcel_nets = [extract_functional_network(e["name"]) for e in entities]
    # Preserve encounter order (parcels are ordered by network)
    network_labels = list(dict.fromkeys(parcel_nets))
    net_to_idx = {name: idx for idx, name in enumerate(network_labels)}
    mapping = np.array([net_to_idx[n] for n in parcel_nets], dtype=np.int32)
    return mapping, network_labels


# ── BIDS entity extraction from filename ──────────────────────────────

BIDS_ENTITY_RE = re.compile(r"(?:^|_)([a-zA-Z]+)-([^_]+)")


def parse_bids_entities(stem: str) -> dict[str, str]:
    return dict(BIDS_ENTITY_RE.findall(stem))


# ── Per-file backfill ─────────────────────────────────────────────────


def backfill_one(yaml_path: Path, atlas_dir: Path, dry_run: bool) -> dict[str, int]:
    """Backfill a single Schaefer network YAML.

    Returns a dict of counters: labels_fixed, positions_fixed, hierarchy_added.
    """
    stats = {"labels_fixed": 0, "positions_fixed": 0, "hierarchy_added": 0}

    with open(yaml_path) as f:
        net = yaml.safe_load(f)

    bids = net.get("bids", {})
    seg = bids.get("segmentation")  # e.g. "7Networks" or "17Networks"
    scale = bids.get("scale")  # e.g. "100"

    if not seg or not scale:
        # Try to infer from filename
        entities = parse_bids_entities(yaml_path.stem)
        seg = seg or entities.get("seg")
        scale = scale or entities.get("scale")

    if not seg or not scale:
        print(f"  [skip] cannot determine seg/scale: {yaml_path.name}")
        return stats

    atlas_entities = load_atlas_entities(atlas_dir, seg, scale)
    if atlas_entities is None:
        print(f"  [skip] no atlas for seg={seg} scale={scale}")
        return stats

    nodes = net.get("nodes", [])
    if len(nodes) != len(atlas_entities):
        print(f"  [warn] node count mismatch ({len(nodes)} vs {len(atlas_entities)}): {yaml_path.name}")
        return stats

    # ── 1. Fix node labels and positions ──────────────────────────────
    yaml_changed = False

    for node, entity in zip(nodes, atlas_entities):
        new_label = entity["name"]
        if node.get("label") != new_label:
            node["label"] = new_label
            stats["labels_fixed"] += 1
            yaml_changed = True

        center = entity.get("center")
        if center:
            new_pos = {
                "x": float(center["x"]),
                "y": float(center["y"]),
                "z": float(center["z"]),
            }
            if node.get("position") != new_pos:
                node["position"] = new_pos
                stats["positions_fixed"] += 1
                yaml_changed = True

    # ── 2. Add functional-network hierarchy ───────────────────────────
    mapping, net_labels = build_node_mapping(atlas_entities)

    # Add node_mapping to YAML (schema-compliant field)
    if "node_mapping" not in net:
        net["node_mapping"] = "/nodes/parent_index"
        stats["hierarchy_added"] = 1
        yaml_changed = True

    # Remove non-schema functional_networks if present
    if "functional_networks" in net:
        del net["functional_networks"]
        yaml_changed = True

    # ── Write ─────────────────────────────────────────────────────────
    if not yaml_changed:
        return stats

    if dry_run:
        print(f"  [dry ] would update {yaml_path.name}")
        return stats

    # Write updated YAML
    with open(yaml_path, "w") as f:
        yaml.safe_dump(net, f, sort_keys=False, allow_unicode=True)

    # Write node_mapping to HDF5 companion
    h5_path = yaml_path.with_suffix(".h5")
    if h5_path.exists():
        with h5py.File(h5_path, "a") as h5:
            grp = h5.require_group("nodes")
            if "parent_index" in grp:
                del grp["parent_index"]
            grp.create_dataset("parent_index", data=mapping, dtype="int32")
            # Also store the functional network labels for reference
            if "functional_network_labels" in grp:
                del grp["functional_network_labels"]
            grp.create_dataset(
                "functional_network_labels",
                data=np.array(net_labels, dtype="S"),
            )

    return stats


def main() -> None:
    args = parse_args()
    network_dir = args.network_dir
    atlas_dir = args.atlas_dir

    yaml_files = sorted(network_dir.glob("*Schaefer2018*.yaml"))
    if not yaml_files:
        print(f"No Schaefer network YAMLs found in {network_dir}")
        return

    totals = {"labels_fixed": 0, "positions_fixed": 0, "hierarchy_added": 0}

    for path in yaml_files:
        print(f"[file] {path.name}")
        stats = backfill_one(path, atlas_dir, args.dry_run)
        for k, v in stats.items():
            totals[k] += v
        if stats["labels_fixed"] or stats["positions_fixed"] or stats["hierarchy_added"]:
            print(
                f"  labels={stats['labels_fixed']} "
                f"positions={stats['positions_fixed']} "
                f"hierarchy={'yes' if stats['hierarchy_added'] else 'no'}"
            )
        else:
            print("  [ok  ] already up to date")

    print(f"\nDone ({len(yaml_files)} files).")
    print(f"  labels fixed:     {totals['labels_fixed']}")
    print(f"  positions fixed:  {totals['positions_fixed']}")
    print(f"  hierarchies added: {totals['hierarchy_added']}")
    if args.dry_run:
        print("  (dry run — no files written)")


if __name__ == "__main__":
    main()
