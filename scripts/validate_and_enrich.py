#!/usr/bin/env python3
"""Validate HDF5 matrix dimensions and enrich YAML sidecars with node info from atlas metadata."""

import glob
import os
import sys

import h5py
import numpy as np
import yaml

from tvbo import database_path

ATLAS_DIR = os.path.join(os.path.dirname(__file__), "..", "tvbo", "data", "tvbo_data", "atlas")
NETWORKS_DIR = str(database_path / "networks")

# Map network atlas names to dseg.yaml files
ATLAS_DSEG_MAP = {
    "DesikanKilliany": "tpl-MNI152NLin2009c_atlas-DesikanKilliany_desc-ranked_dseg.yaml",
    "DesikanKillianyranked": "tpl-MNI152NLin2009c_atlas-DesikanKilliany_desc-ranked_dseg.yaml",
    "Destrieux": "tpl-MNI152Nlin2009c_atlas-Destrieux_desc-ranked_dseg.yaml",
    "Destrieuxranked": "tpl-MNI152Nlin2009c_atlas-Destrieux_desc-ranked_dseg.yaml",
    "hcpmmp1": "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_dseg.yaml",
    "hcpmmp1ordered": "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_dseg.yaml",
    "Yeo17": "space-MNI152_atlas-Yeo17_res-1_dseg.yaml",
}

# Map atlas names to centers.txt files (for atlases where dseg.yaml lacks inline centers)
ATLAS_CENTERS_MAP = {
    "DesikanKilliany": "tpl-MNI152Nlin2009c_atlas-DesikanKilliany_desc-ranked_centers.txt",
    "DesikanKillianyranked": "tpl-MNI152Nlin2009c_atlas-DesikanKilliany_desc-ranked_centers.txt",
    "Destrieux": "tpl-MNI152Nlin2009c_atlas-Destrieux_desc-ranked_centers.txt",
    "Destrieuxranked": "tpl-MNI152Nlin2009c_atlas-Destrieux_desc-ranked_centers.txt",
    "hcpmmp1": "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_centers.txt",
    "hcpmmp1ordered": "tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_centers.txt",
}


def load_centers_txt(atlas_name):
    """Load centers from a centers.txt file (space-separated x y z per line)."""
    centers_file = ATLAS_CENTERS_MAP.get(atlas_name)
    if not centers_file:
        return None
    path = os.path.join(ATLAS_DIR, centers_file)
    if not os.path.exists(path):
        return None
    return np.loadtxt(path)


def load_atlas_entities(atlas_name):
    """Load region entities from atlas dseg.yaml. Returns sorted list of (name, lookupLabel, center_dict)."""
    dseg_file = ATLAS_DSEG_MAP.get(atlas_name)
    if not dseg_file:
        return None
    path = os.path.join(ATLAS_DIR, dseg_file)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    entities = data.get("terminology", {}).get("entities", {})

    regions = []
    if isinstance(entities, dict):
        for name, info in entities.items():
            ll = info.get("lookupLabel", 0)
            center = info.get("center", {})
            regions.append((name, ll, center))
    elif isinstance(entities, list):
        for item in entities:
            name = item.get("name", "")
            ll = item.get("lookupLabel", 0)
            center = item.get("center", {})
            regions.append((name, ll, center))
    else:
        return None

    regions.sort(key=lambda x: x[1])

    # If entities lack inline centers, load from centers.txt
    has_centers = any(c for _, _, c in regions)
    if not has_centers:
        centers = load_centers_txt(atlas_name)
        if centers is not None and len(centers) == len(regions):
            regions = [
                (name, ll, {"x": float(centers[i, 0]), "y": float(centers[i, 1]), "z": float(centers[i, 2])})
                for i, (name, ll, _) in enumerate(regions)
            ]

    return regions


def get_hdf5_shapes(filepath):
    """Extract shapes of weight/length matrices from HDF5 (handles nested edges/ structure)."""
    shapes = {}
    with h5py.File(filepath, "r") as h:
        if "edges" in h and isinstance(h["edges"], h5py.Group):
            edges = h["edges"]
            for edge_name in edges.keys():
                obj = edges[edge_name]
                if isinstance(obj, h5py.Dataset):
                    shapes[edge_name] = obj.shape
                elif isinstance(obj, h5py.Group):
                    if "data" in obj and isinstance(obj["data"], h5py.Dataset):
                        shapes[edge_name] = obj["data"].shape
                    elif "indptr" in obj:
                        n = obj["indptr"].shape[0] - 1
                        shapes[edge_name] = (n, n)
        else:
            for k in h.keys():
                obj = h[k]
                if isinstance(obj, h5py.Dataset):
                    shapes[k] = obj.shape
    return shapes


def validate_hdf5():
    """Validate all HDF5 files: check weights/lengths shape match and are square."""
    print("=== HDF5 Matrix Dimension Validation ===\n")
    problems = []
    for f in sorted(glob.glob(os.path.join(NETWORKS_DIR, "*.h5"))):
        name = os.path.basename(f)
        shapes = get_hdf5_shapes(f)

        w = shapes.get("weights")
        lengths = shapes.get("lengths")

        if w is None or lengths is None:
            problems.append((name, f"MISSING: weights={w}, lengths={lengths}"))
            print(f"  MISSING  {name}: found={shapes}")
        elif w != lengths:
            problems.append((name, f"MISMATCH: weights={w} vs lengths={lengths}"))
            print(f"  MISMATCH {name}: weights={w} vs lengths={lengths}")
        elif len(w) == 2 and w[0] != w[1]:
            problems.append((name, f"NOT SQUARE: {w}"))
            print(f"  NOTSQ    {name}: {w}")
        else:
            print(f"  OK       {name}: {w}")

        # Also check number_of_nodes in YAML matches matrix dims
        yaml_path = f.replace(".h5", ".yaml")
        if os.path.exists(yaml_path) and w is not None:
            with open(yaml_path) as fh:
                ydata = yaml.safe_load(fh)
            n_yaml = ydata.get("number_of_nodes", 0)
            n_matrix = w[0] if len(w) >= 1 else 0
            if n_yaml != n_matrix:
                problems.append((name, f"NODE COUNT: yaml={n_yaml} vs matrix={n_matrix}"))
                print(f"  NODECNT  {name}: yaml={n_yaml} vs matrix={n_matrix}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for n, msg in problems:
            print(f"  - {n}: {msg}")
    else:
        print("All HDF5 files valid.")
    return problems


def enrich_yaml_sidecars():
    """Add nodes list with region labels and positions to YAML sidecars."""
    print("\n=== Enriching YAML Sidecars with Node Info ===\n")
    enriched = 0
    skipped = []

    for f in sorted(glob.glob(os.path.join(NETWORKS_DIR, "*.yaml"))):
        name = os.path.basename(f)
        with open(f) as fh:
            data = yaml.safe_load(fh)

        parc = data.get("parcellation", {})
        atlas = parc.get("atlas", {})
        atlas_name = atlas.get("name") if isinstance(atlas, dict) else None

        if not atlas_name:
            skipped.append((name, "no atlas name"))
            continue

        regions = load_atlas_entities(atlas_name)
        if not regions:
            skipped.append((name, f"no dseg.yaml for atlas {atlas_name}"))
            continue

        n_nodes = data.get("number_of_nodes", 0)
        if len(regions) != n_nodes:
            skipped.append((name, f"atlas has {len(regions)} regions but network has {n_nodes} nodes"))
            continue

        # Build nodes list
        nodes = []
        for region_name, lookup_label, center in regions:
            node = {
                "id": lookup_label,
                "label": region_name,
            }
            if center and isinstance(center, dict):
                node["position"] = {
                    "x": round(center.get("x", 0), 4),
                    "y": round(center.get("y", 0), 4),
                    "z": round(center.get("z", 0), 4),
                }
            nodes.append(node)

        data["nodes"] = nodes

        # Add coordinate space to atlas if identifiable from filename
        if isinstance(data.get("parcellation", {}).get("atlas"), dict):
            if "MNI152NLin2009" in name or "MNI152Nlin2009" in name:
                data["parcellation"]["atlas"]["coordinateSpace"] = "MNI152NLin2009c"
            elif "MNI152" in name or "FSLMNI152" in name:
                data["parcellation"]["atlas"]["coordinateSpace"] = "MNI152"

        with open(f, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

        enriched += 1
        print(f"  ENRICHED {name}: {len(nodes)} nodes added")

    print()
    print(f"Enriched: {enriched} files")
    if skipped:
        print(f"Skipped: {len(skipped)} files")
        for n, reason in skipped:
            print(f"  - {n}: {reason}")


if __name__ == "__main__":
    problems = validate_hdf5()
    enrich_yaml_sidecars()
    sys.exit(1 if problems else 0)
