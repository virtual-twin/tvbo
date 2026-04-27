#!/usr/bin/env python3
"""Generate Schaefer2018 multi-scale structural connectomes for tvbo.

This script uses MRtrix `tck2connectome` to compute weights and lengths for
selected Schaefer scales, then writes tvbo HDF5+YAML network files in
`database/networks/` via `Network.save()`.

Prerequisites:
1. Original Schaefer files downloaded (scripts/retrieve_schaefer_original.py)
2. BIDS-style atlas links/copies prepared (scripts/prepare_schaefer_atlases.py)
3. `tck2connectome` available in PATH (MRtrix3)

Usage:
    python scripts/generate_schaefer_scales.py \
      --tractogram dTOR=/path/to/dTOR.tck \
      --tractogram MghUscHcp32=/path/to/MghUscHcp32.tck \
      --tractogram PPMI85=/path/to/PPMI85.tck \
      --scales 100 200 400 600 800 1000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml

from tvbo.data.tvbo_data.connectomes import Network


ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas"
OUTPUT_DIR = ROOT / "database" / "networks"

COHORT_MAP = {
    "dTOR": "HCPYA",
    "MghUscHcp32": "MghUscHcp32",
    "PPMI85": "PPMI85",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Schaefer multi-scale SC files from tractograms.")
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=ATLAS_DIR,
        help="Directory containing prepared Schaefer atlas files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for generated network files.",
    )
    parser.add_argument(
        "--tractogram",
        action="append",
        required=True,
        help="Tractogram spec NAME=/absolute/path/file.tck (repeatable).",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[100, 200, 400, 600, 800, 1000],
        help="Schaefer scales to generate.",
    )
    parser.add_argument(
        "--networks",
        type=int,
        choices=[7, 17],
        default=17,
        help="Schaefer network family (7 or 17).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[1, 2],
        default=1,
        help="Atlas resolution in mm (1 or 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def parse_tractogram_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --tractogram value '{spec}'. Expected NAME=/path/file.tck")
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path = Path(path_str).expanduser().resolve()
        if not name:
            raise ValueError(f"Invalid tractogram name in '{spec}'")
        if not path.exists():
            raise FileNotFoundError(f"Tractogram file not found: {path}")
        parsed.append((name, path))
    return parsed


def atlas_path(atlas_dir: Path, scale: int, networks: int, resolution: int) -> Path:
    fname = (
        f"space-FSLMNI152_atlas-Schaefer2018_seg-{networks}Networks_scale-{scale}_res-{resolution}_desc-original_dseg.nii.gz"
    )
    path = atlas_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Atlas file not found: {path}. Run scripts/prepare_schaefer_atlases.py first.")
    return path


def load_atlas_entities(atlas_dir: Path, scale: int, networks: int, resolution: int) -> list[dict]:
    """Load Schaefer SANDS atlas entities sorted by lookupLabel (1-based)."""
    fname = f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{networks}Networks_scale-{scale}_res-{resolution}_desc-ordered_dseg.yaml"
    path = atlas_dir / fname
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    entities = data.get("terminology", {}).get("entities", {})
    return sorted(entities.values(), key=lambda e: e["lookupLabel"])


def run_tck2connectome(
    tractogram_path: Path,
    atlas_file: Path,
    weights_csv: Path,
    lengths_csv: Path,
    assignments_csv: Path,
) -> None:
    subprocess.run(
        [
            "tck2connectome",
            str(tractogram_path),
            str(atlas_file),
            str(weights_csv),
            "-out_assignments",
            str(assignments_csv),
            "-symmetric",
            "-zero_diagonal",
            "-force",
        ],
        check=True,
    )

    subprocess.run(
        [
            "tck2connectome",
            str(tractogram_path),
            str(atlas_file),
            str(lengths_csv),
            "-scale_length",
            "-stat_edge",
            "mean",
            "-symmetric",
            "-zero_diagonal",
            "-force",
        ],
        check=True,
    )


def extract_functional_network(region_name: str) -> str:
    """Extract hemisphere + functional network from a Schaefer region name.

    E.g. ``"17Networks_LH_VisCent_1"`` → ``"LH_VisCent"``,
         ``"7Networks_RH_Default_3"``  → ``"RH_Default"``.
    """
    parts = region_name.split("_")
    # Format: {seg}_{hemi}_{network}_{index...}
    return f"{parts[1]}_{parts[2]}"


def build_node_mapping(entities: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build a parcel → functional-network index mapping.

    Returns
    -------
    mapping : ndarray of int32, shape (n_parcels,)
        Index into the sorted list of unique functional networks.
    network_labels : list of str
        Sorted unique functional network names (e.g. ``["LH_Cont", …]``).
    """
    parcel_nets = [extract_functional_network(e["name"]) for e in entities]
    network_labels = sorted(set(parcel_nets), key=parcel_nets.index)
    net_to_idx = {name: idx for idx, name in enumerate(network_labels)}
    mapping = np.array([net_to_idx[n] for n in parcel_nets], dtype=np.int32)
    return mapping, network_labels


def build_network(
    weights: np.ndarray,
    lengths: np.ndarray,
    tractogram_name: str,
    scale: int,
    networks: int,
    atlas_dir: Path = ATLAS_DIR,
    resolution: int = 1,
) -> Network:
    entities = load_atlas_entities(atlas_dir, scale, networks, resolution)
    labels = [e["name"] for e in entities] if entities else None
    network = Network.from_matrix(weights=weights, lengths=lengths, labels=labels)
    network.label = f"Schaefer2018_{scale}_{networks}Networks_{tractogram_name}"
    # Backfill node positions from atlas centroids
    if entities and network.nodes:
        for node, entity in zip(network.nodes, entities):
            c = entity.get("center", {})
            if c:
                node.position = {"x": float(c["x"]), "y": float(c["y"]), "z": float(c["z"])}

    # Build parcel → functional network mapping (multi-layer hierarchy)
    if entities:
        mapping, net_labels = build_node_mapping(entities)
        network.set_node_mapping(mapping)

    network.descriptor = "SC"
    network.parcellation = {
        "atlas": {
            "name": "Schaefer2018",
            "coordinateSpace": "FSLMNI152",
        }
    }
    network.tractogram = {"name": tractogram_name}
    network.bids = {
        "template": "FSLMNI152",
        "cohort": COHORT_MAP.get(tractogram_name, tractogram_name),
        "reconstruction": tractogram_name,
        "segmentation": f"{networks}Networks",
        "scale": str(scale),
    }
    return network


def main() -> None:
    args = parse_args()

    if shutil.which("tck2connectome") is None:
        raise RuntimeError("tck2connectome was not found in PATH. Install MRtrix3 and ensure tck2connectome is available.")

    tractograms = parse_tractogram_specs(args.tractogram)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for tract_name, tract_path in tractograms:
        for scale in args.scales:
            atlas_file = atlas_path(args.atlas_dir, scale, args.networks, args.resolution)

            with tempfile.TemporaryDirectory(prefix="tvbo_schaefer_") as tmp:
                tmpdir = Path(tmp)
                weights_csv = tmpdir / "weights.csv"
                lengths_csv = tmpdir / "lengths.csv"
                assignments_csv = tmpdir / "assignments.csv"

                print(f"[run ] tractogram={tract_name} scale={scale} nets={args.networks}")
                run_tck2connectome(
                    tractogram_path=tract_path,
                    atlas_file=atlas_file,
                    weights_csv=weights_csv,
                    lengths_csv=lengths_csv,
                    assignments_csv=assignments_csv,
                )

                weights = np.loadtxt(weights_csv, delimiter=",")
                lengths = np.loadtxt(lengths_csv, delimiter=",")

                net = build_network(
                    weights=weights,
                    lengths=lengths,
                    tractogram_name=tract_name,
                    scale=scale,
                    networks=args.networks,
                    atlas_dir=args.atlas_dir,
                    resolution=args.resolution,
                )

                out_name = net.bids_filename
                out_path = args.output_dir / out_name
                sidecar_path = out_path.with_suffix(".yaml")
                companion_path = sidecar_path.with_suffix(".h5")

                if sidecar_path.exists() and companion_path.exists() and not args.overwrite:
                    skipped += 1
                    print(f"[skip] {sidecar_path.name}")
                    continue

                net.save(out_path)
                generated += 1
                print(f"[ok  ] {sidecar_path.name}")

    print("\nDone.")
    print(f"  generated: {generated}")
    print(f"  skipped:   {skipped}")
    print(f"  output:    {args.output_dir}")


if __name__ == "__main__":
    main()
