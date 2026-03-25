#!/usr/bin/env python3
"""Retrieve original Schaefer2018 MNI atlas files from ThomasYeoLab/CBIG.

Downloads the official NIfTI files directly from:
https://github.com/ThomasYeoLab/CBIG

Usage:
    python scripts/retrieve_schaefer_original.py
    python scripts/retrieve_schaefer_original.py --scales 100 200 400 600 800 1000
    python scripts/retrieve_schaefer_original.py --networks 17 --resolution 1mm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas" / "schaefer2018_original_mni"

CBIG_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI"
)

DEFAULT_SCALES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def build_filename(scale: int, networks: int, resolution: str) -> str:
    return (
        f"Schaefer2018_{scale}Parcels_{networks}Networks_"
        f"order_FSLMNI152_{resolution}.nii.gz"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download original Schaefer2018 atlas files from CBIG."
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Parcel scales to download (e.g., 100 200 400 600 800 1000).",
    )
    parser.add_argument(
        "--networks",
        type=int,
        nargs="+",
        default=[7, 17],
        choices=[7, 17],
        help="Network families to download.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1mm",
        choices=["1mm", "2mm"],
        help="MNI resolution to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_note = output_dir / "SOURCE.txt"
    source_note.write_text(
        "Original source: ThomasYeoLab/CBIG\n"
        "Repository: https://github.com/ThomasYeoLab/CBIG\n"
        "Folder: stable_projects/brain_parcellation/"
        "Schaefer2018_LocalGlobal/Parcellations/MNI\n",
        encoding="utf-8",
    )

    downloaded = 0
    skipped = 0
    requested = 0

    for scale in args.scales:
        for networks in args.networks:
            requested += 1
            filename = build_filename(scale, networks, args.resolution)
            url = f"{CBIG_BASE_URL}/{filename}"
            dst = output_dir / filename

            if dst.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip] {filename}")
                continue

            print(f"[get ] {url}")
            urlretrieve(url, dst)
            downloaded += 1
            print(f"[ok  ] {dst}")

    print("\nDone.")
    print(f"  requested:  {requested}")
    print(f"  downloaded: {downloaded}")
    print(f"  skipped:    {skipped}")
    print(f"  output:     {output_dir}")


if __name__ == "__main__":
    main()
