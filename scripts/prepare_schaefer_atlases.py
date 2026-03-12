#!/usr/bin/env python3
"""Prepare original Schaefer2018 files as BIDS-style atlas files for tvbo.

This script maps files downloaded from ThomasYeoLab/CBIG
(`schaefer2018_original_mni/`) to atlas files in `tvbo/data/tvbo_data/atlas/`
with BIDS-style entity names.

Usage:
    python scripts/prepare_schaefer_atlases.py
    python scripts/prepare_schaefer_atlases.py --scales 100 200 400 600 800 1000
    python scripts/prepare_schaefer_atlases.py --networks 17 --copy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas"
SOURCE_DIR = ATLAS_DIR / "schaefer2018_original_mni"

FILE_RE = re.compile(
    r"^Schaefer2018_(?P<scale>\d+)Parcels_(?P<nets>7|17)Networks_"
    r"order_FSLMNI152_(?P<res>1mm|2mm)\.nii\.gz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BIDS-style Schaefer atlas files from original downloads."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing original CBIG Schaefer files.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=ATLAS_DIR,
        help="Target atlas directory (tvbo/data/tvbo_data/atlas).",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="Scales to include.",
    )
    parser.add_argument(
        "--networks",
        type=int,
        nargs="+",
        choices=[7, 17],
        default=[7, 17],
        help="Network family variants to include.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["1mm", "2mm"],
        default="1mm",
        help="Resolution to include.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing targets.",
    )
    return parser.parse_args()


def target_name(scale: int, networks: int, resolution: str) -> str:
    """Build BIDS-style dseg filename for atlas directory."""
    res_val = "1" if resolution == "1mm" else "2"
    return (
        "space-FSLMNI152"
        "_atlas-Schaefer2018"
        f"_seg-{networks}Networks"
        f"_scale-{scale}"
        f"_res-{res_val}"
        "_desc-original_dseg.nii.gz"
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    atlas_dir = args.atlas_dir
    atlas_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    processed = 0

    for source_file in sorted(source_dir.glob("Schaefer2018_*_FSLMNI152_*.nii.gz")):
        match = FILE_RE.match(source_file.name)
        if not match:
            continue

        scale = int(match.group("scale"))
        networks = int(match.group("nets"))
        resolution = match.group("res")

        if scale not in args.scales:
            continue
        if networks not in args.networks:
            continue
        if resolution != args.resolution:
            continue

        processed += 1
        dst = atlas_dir / target_name(scale, networks, resolution)

        if dst.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {dst.name}")
            continue

        if dst.exists():
            dst.unlink()

        if args.copy:
            data = source_file.read_bytes()
            dst.write_bytes(data)
            print(f"[copy] {source_file.name} -> {dst.name}")
        else:
            dst.symlink_to(source_file.resolve())
            print(f"[link] {source_file.name} -> {dst.name}")

        created += 1

    print("\nDone.")
    print(f"  processed: {processed}")
    print(f"  created:   {created}")
    print(f"  skipped:   {skipped}")
    print(f"  atlas dir: {atlas_dir}")


if __name__ == "__main__":
    main()
