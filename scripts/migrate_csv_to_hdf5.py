#!/usr/bin/env python3
"""Migrate CSV connectomes to HDF5+YAML format.

Reads CSV weight/length pairs from tvbo/data/tvbo_data/connectome/, creates HDF5 companion + updated YAML sidecar in database/networks/.

Usage:
    python scripts/migrate_csv_to_hdf5.py [--dry-run]
"""

import sys
import numpy as np
from pathlib import Path
from tvbo import database_path

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "connectome"
NETWORK_DIR = database_path / "networks"

# Map existing YAML sidecars to their CSV sources
# Parse sidecar filenames to find matching CSV files


def find_csv_pairs(csv_dir: Path) -> dict:
    """Find weight/length CSV pairs in connectome directory."""
    pairs = {}
    for f in sorted(csv_dir.glob("*.csv")):
        stem = f.stem
        if stem.endswith("_weights"):
            base = stem[:-8]  # remove _weights
            length_file = f.parent / f"{base}_lengths.csv"
            pairs[base] = {
                "weights": f,
                "lengths": length_file if length_file.exists() else None,
            }
    return pairs


def migrate_one(base_name: str, csv_paths: dict, out_dir: Path, dry_run: bool = False):
    """Convert one CSV pair to HDF5+YAML."""
    import h5py
    from tvbo.data.matrix_io import write_matrix, auto_format

    weights_csv = csv_paths["weights"]
    lengths_csv = csv_paths.get("lengths")

    weights = np.loadtxt(weights_csv, delimiter=",")
    lengths = np.loadtxt(lengths_csv, delimiter=",") if lengths_csv else None
    n = weights.shape[0]

    # Determine output name from CSV base
    yaml_name = f"{base_name}.yaml"
    h5_name = f"{base_name}.h5"

    yaml_path = out_dir / yaml_name
    h5_path = out_dir / h5_name

    if dry_run:
        print(f"  [DRY-RUN] {base_name}: {n}x{n} → {h5_name}")
        return

    # Write HDF5
    fmt = auto_format(weights)
    with h5py.File(h5_path, "w") as f:
        f.attrs["tvbo_class"] = "tvbo:Network"
        f.attrs["sidecar_file"] = yaml_name
        f.attrs["schema_version"] = "tvb-datamodel/0.7.0"

        wg = f.create_group("edges/weights")
        wg.attrs["tvbo_class"] = "tvbo:Matrix"
        write_matrix(wg, weights, fmt=fmt)

        if lengths is not None:
            lg = f.create_group("edges/lengths")
            lg.attrs["tvbo_class"] = "tvbo:Matrix"
            write_matrix(lg, lengths, fmt=fmt)

    # Parse existing YAML sidecar if it exists, otherwise create new
    edges = [
        {"name": "weights", "format": fmt, "weighted": True, "valid_diagonal": False, "non_negative": True},
    ]
    if lengths is not None:
        edges.append(
            {"name": "lengths", "unit": "mm", "format": fmt, "weighted": True, "valid_diagonal": False, "non_negative": True}
        )

    # Try to read existing sidecar for metadata
    existing_meta = {}
    if yaml_path.exists():
        import yaml

        with open(yaml_path) as fh:
            existing_meta = yaml.safe_load(fh) or {}

    # Build updated sidecar
    meta = {
        "label": existing_meta.get("label", base_name),
        "number_of_nodes": n,
        "number_of_regions": n,
        "data_file": h5_name,
        "edges": edges,
    }

    # Preserve parcellation, tractogram, descriptor info
    for key in ("parcellation", "tractogram", "descriptor", "description", "parameters"):
        if key in existing_meta:
            meta[key] = existing_meta[key]

    # Extract tractogram name if it was a string
    if "tractogram" in meta and isinstance(meta["tractogram"], str):
        meta["tractogram"] = {"name": meta["tractogram"]}

    # Remove old CSV paths
    for key in (
        "weights",
        "lengths",
        "weights_min",
        "weights_max",
        "weights_mean",
        "weights_median",
        "weights_histogram",
        "type",
    ):
        meta.pop(key, None)

    import yaml

    with open(yaml_path, "w") as fh:
        yaml.dump(meta, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    sz = h5_path.stat().st_size
    print(f"  {base_name}: {n}x{n} ({fmt}) → {h5_name} ({sz / 1024:.1f} KB)")


def main():
    dry_run = "--dry-run" in sys.argv

    print("CSV→HDF5 migration")
    print(f"  Source: {CSV_DIR}")
    print(f"  Target: {NETWORK_DIR}")
    print()

    pairs = find_csv_pairs(CSV_DIR)
    print(f"Found {len(pairs)} CSV weight/length pairs")

    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    for base_name, csv_paths in sorted(pairs.items()):
        migrate_one(base_name, csv_paths, NETWORK_DIR, dry_run=dry_run)

    if not dry_run:
        # Summary
        h5_files = list(NETWORK_DIR.glob("*.h5"))
        total_h5 = sum(f.stat().st_size for f in h5_files)
        csv_files = list(CSV_DIR.glob("*.csv"))
        total_csv = sum(f.stat().st_size for f in csv_files)
        print("\nMigration complete:")
        print(f"  {len(h5_files)} HDF5 files ({total_h5 / 1024 / 1024:.1f} MB)")
        print(f"  {len(csv_files)} CSV files ({total_csv / 1024 / 1024:.1f} MB)")
        print(f"  Compression ratio: {total_csv / total_h5:.1f}x")


if __name__ == "__main__":
    main()
