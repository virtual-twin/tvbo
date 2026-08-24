#!/usr/bin/env python3
"""Create SANDS-compliant atlas metadata for Schaefer2018 originals.

This script reads Schaefer NIfTI files downloaded to `tvbo/data/tvbo_data/atlas/schaefer2018_original_mni/`, fetches the corresponding official LUT + centroid tables from ThomasYeoLab/CBIG, and writes LinkML/SANDS-compatible `BrainAtlas` YAML sidecars.

Output files are written to `tvbo/data/tvbo_data/atlas/` using a BIDS-like naming convention with explicit segmentation and scale entities.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.request import urlopen

import yaml
from linkml_runtime.loaders import yaml_loader

from tvbo.datamodel import tvbo_datamodel

ROOT = Path(__file__).resolve().parent.parent
# Atlases are consolidated under tvbo/database/atlases (the runtime SoT).
ATLAS_DIR = ROOT / "tvbo" / "database" / "atlases"
# Non-BIDS Schaefer download staging stays out of the BIDS database (gitignored).
ORIGINAL_DIR = ROOT / "tvbo" / "data" / "tvbo_data" / "atlas" / "schaefer2018_original_mni"

CBIG_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI"
)
CBIG_LUT_DIR = f"{CBIG_BASE}/freeview_lut"
CBIG_CENTROID_DIR = f"{CBIG_BASE}/Centroid_coordinates"


NIFTI_PATTERN = re.compile(
    r"^Schaefer2018_(?P<scale>\d+)Parcels_(?P<seg>\d+Networks)_"
    r"order_FSLMNI152_(?P<res>\dmm)\.nii\.gz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SANDS metadata for Schaefer atlas files.")
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=ORIGINAL_DIR,
        help="Directory containing original Schaefer NIfTI files.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=ATLAS_DIR,
        help="Directory where atlas YAML sidecars are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML sidecars.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated YAML by loading with tvbo LinkML model.",
    )
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_lut(scale: int, seg: str) -> dict[int, dict[str, str | int]]:
    lut_name = f"Schaefer2018_{scale}Parcels_{seg}_order.txt"
    lut_url = f"{CBIG_LUT_DIR}/{lut_name}"
    lines = [line.strip() for line in fetch_text(lut_url).splitlines() if line.strip()]

    lut: dict[int, dict[str, str | int]] = {}
    for line in lines:
        cols = line.split()
        label = int(cols[0])
        region_name = cols[1]
        red = int(cols[2])
        green = int(cols[3])
        blue = int(cols[4])
        lut[label] = {
            "name": region_name,
            "r": red,
            "g": green,
            "b": blue,
        }
    return lut


def parse_centroids(scale: int, seg: str, res: str) -> dict[int, dict[str, str | float]]:
    centroid_name = f"Schaefer2018_{scale}Parcels_{seg}_order_FSLMNI152_{res}.Centroid_RAS.csv"
    centroid_url = f"{CBIG_CENTROID_DIR}/{centroid_name}"
    text = fetch_text(centroid_url)

    data: dict[int, dict[str, str | float]] = {}
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        label = int(row["ROI Label"])
        data[label] = {
            "name": row["ROI Name"],
            "x": float(row["R"]),
            "y": float(row["A"]),
            "z": float(row["S"]),
        }
    return data


def hemisphere_from_region_name(region_name: str) -> str | None:
    if "_LH_" in region_name:
        return "left"
    if "_RH_" in region_name:
        return "right"
    return None


def rgb_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def metadata_filename(scale: int, seg: str, res: str) -> str:
    res_num = res.replace("mm", "")
    return f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-{res_num}_desc-ordered_dseg.yaml"


def build_metadata(scale: int, seg: str, res: str, nifti_name: str) -> dict:
    lut = parse_lut(scale, seg)
    centroids = parse_centroids(scale, seg, res)

    entities: dict[str, dict] = {}
    for label, lut_entry in lut.items():
        centroid = centroids[label]
        region_name = str(lut_entry["name"])
        red = int(lut_entry["r"])
        green = int(lut_entry["g"])
        blue = int(lut_entry["b"])

        entity: dict[str, object] = {
            "name": region_name,
            "lookupLabel": label,
            "originalLookupLabel": label,
            "center": {
                "x": float(centroid["x"]),
                "y": float(centroid["y"]),
                "z": float(centroid["z"]),
            },
            "color": rgb_hex(red, green, blue),
        }

        hemi = hemisphere_from_region_name(region_name)
        if hemi is not None:
            entity["hemisphere"] = hemi

        entities[region_name] = entity

    return {
        "name": "Schaefer2018",
        "abbreviation": "Schaefer2018",
        "versionIdentifier": "2018",
        "coordinateSpace": {
            "name": "MNI152",
            "abbreviation": "FSLMNI152",
            "nativeUnit": "mm",
        },
        "terminology": {
            "label": f"Schaefer2018_{scale}Parcels_{seg}_order",
            "versionIdentifier": "2018",
            "dataLocation": f"schaefer2018_original_mni/{nifti_name}",
            "entities": entities,
        },
    }


def validate_metadata(path: Path) -> None:
    yaml_loader.load(str(path), tvbo_datamodel.BrainAtlas)


def main() -> None:
    args = parse_args()
    original_dir = args.original_dir
    atlas_dir = args.atlas_dir

    if not original_dir.exists():
        raise FileNotFoundError(f"Original Schaefer directory not found: {original_dir}")

    atlas_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for nifti_path in sorted(original_dir.glob("*.nii.gz")):
        match = NIFTI_PATTERN.match(nifti_path.name)
        if match is None:
            continue

        scale = int(match.group("scale"))
        seg = match.group("seg")
        res = match.group("res")

        metadata = build_metadata(scale=scale, seg=seg, res=res, nifti_name=nifti_path.name)
        out_name = metadata_filename(scale=scale, seg=seg, res=res)
        out_path = atlas_dir / out_name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {out_path.name}")
            continue

        out_path.write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        if args.validate:
            validate_metadata(out_path)

        created += 1
        print(f"[ok  ] {out_path.name}")

    print("\nDone.")
    print(f"  created: {created}")
    print(f"  skipped: {skipped}")
    print(f"  output:  {atlas_dir}")


if __name__ == "__main__":
    main()
