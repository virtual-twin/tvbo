#!/usr/bin/env python
"""Recompute the hcpmmp1 atlas parcel centres from the original Glasser CIFTI dlabel.

The shipped centres had seven right-hemisphere medial parcels (``R_10r``, ``R_10v``, ``R_25``, ``R_OFC``, ``R_a24``, ``R_p32``, ``R_s32``) pinned at ``x = -0.50`` exactly — a clamp, not a computed centroid, placing right-hemisphere parcels left of the midline. Their L/R mirror mismatch reached 26.5 mm. Recomputing from the parcellation's own geometry fixes those and tightens the worst mirror mismatch to ~7 mm.

Method
------
- **cortical parcels** — mean vertex coordinate over each parcel's grayordinates, read from
  the group midthickness surface of the matching hemisphere;
- **subcortical / brainstem parcels** — mean voxel centre over each parcel's grayordinates,
  from the dlabel's own volume model and affine.

Nothing here is keyed by position or by a hardcoded name list. The surface structures and their vertex counts come from the CIFTI's own brain-model axis, the surfaces are discovered in ``--surf-dir`` by hemisphere, and atlas entities are paired with dlabel labels through the ``alternateName`` crosswalk the atlas already carries (``L_V1`` -> ``L_V1_ROI``, ``L_Cerebellum`` -> ``CEREBELLUM_LEFT``, ``Brain-Stem`` -> ``BRAIN_STEM``).

Usage
-----
    python scripts/rebuild_hcpmmp1_centers.py \
        --dlabel .../Q1-Q6_RelatedValidation210..._with_Atlas_ROIs2.32k_fs_LR.dlabel.nii \
        --surf-dir .../hcp_data          # any *.{L,R}.midthickness*.surf.gii pair

Neither input ships with tvbo (~1 MB + 2x1.8 MB); both come from the HCP S1200 release / BALSA study RVVG. Pass ``--dry-run`` to print the comparison without writing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

TPL = "fsLR"
"""The template entity and ``coordinateSpace`` for the rebuilt atlas.

The parcels are defined on the fsLR 32k surface mesh, which is what indexes the centroids —
so ``tpl-fsLR`` is what the file is, and it replaces the ``tpl-MNI152NLin2009b`` the shipped
centres claimed. (The dlabel's *volume* model rides on the FSL MNI152 2 mm grid, but only the
19 subcortical parcels come from there; the parcellation itself is a surface one.)
"""

SURFACE_GLOB = "*.{hemi}.midthickness*.surf.gii"

ASEG_STEMS = {
    "Cerebellum": "Cerebellum-Cortex",
    "Thalamus": "Thalamus-Proper",
    "Accumbens": "Accumbens-area",
    "Caudate": "Caudate",
    "Putamen": "Putamen",
    "Pallidum": "Pallidum",
    "Hippocampus": "Hippocampus",
    "Amygdala": "Amygdala",
    "VentralDC": "VentralDC",
}
"""The 19 subcortical stems: atlas spelling -> FreeSurfer ``aseg`` spelling.

Membership of this dict is also what marks an entity as subcortical, so cortical parcels fall
through the generator untouched.
"""

CIFTI_STEMS = {"VentralDC": "DIENCEPHALON_VENTRAL"}
"""Where the CIFTI structure stem differs from the atlas's own; others just uppercase."""


def spelling_variants(entity_name: str) -> list[str]:
    """Every label spelling a connectome might use for this atlas entity.

    An atlas entity is findable only by the exact strings it lists, and ``get_centers()`` / the node-alias crosswalk match purely by label — so a convention the atlas has never heard of silently yields no centre for that region. Rather than add spellings reactively, one pipeline at a time, generate the conventions this codebase actually meets:

    - **CIFTI / HCP** — ``L_V1_ROI``, ``CEREBELLUM_LEFT``, ``BRAIN_STEM``;
    - **FreeSurfer ``aseg``** — ``Left-Cerebellum-Cortex``, ``Right-Accumbens-area``,
      ``Brainstem``, plus the underscore and ``lh``/``rh`` forms the same tools emit;
    - **FreeSurfer ``aparc`` / BIDS cortical** — ``ctx-lh-V1``, ``L.V1``.

    Every variant stays unique across the 379 entities and keeps its region's hemisphere, which ``tests/test_network_io.py::TestAtlasAliases`` enforces.
    """
    if entity_name == "Brain-Stem":
        return ["BRAIN_STEM", "Brainstem", "BrainStem", "brain-stem"]
    m = re.match(r"^([LR])_(.+)$", entity_name)
    if not m:
        return []
    hemi, stem = m.group(1), m.group(2)
    side, sidelow = ("Left", "lh") if hemi == "L" else ("Right", "rh")

    if stem in ASEG_STEMS:
        aseg = ASEG_STEMS[stem]
        return [
            f"{CIFTI_STEMS.get(stem, stem.upper())}_{side.upper()}",
            f"{side}-{aseg}",
            f"{side}_{aseg.replace('-', '_')}",
            f"{sidelow}-{aseg}",
            f"{hemi}_{aseg}",
        ]
    return [f"{hemi}_{stem}_ROI", f"{hemi}.{stem}", f"ctx-{sidelow}-{stem}"]


def atlas_dseg_path(name: str = "hcpmmp1") -> Path:
    """The atlas's ``_dseg.yaml``, found the same way ``Atlas`` finds it (by BIDS entity)."""
    from tvbo.classes.atlas import atlas_data

    hits = atlas_data.get(atlas=name, suffix="dseg", extension=".yaml", return_type="file")
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} dseg.yaml, found {len(hits)}")
    return Path(hits[0])


def retemplated(path: Path, tpl: str) -> Path:
    """The same BIDS name with its ``tpl-`` entity replaced."""
    parts = path.name.split("_")
    if not parts[0].startswith("tpl-"):
        raise ValueError(f"{path.name}: no leading tpl- entity to replace")
    parts[0] = f"tpl-{tpl}"
    return path.with_name("_".join(parts))


def find_surface(surf_dir: Path, structure: str) -> Path:
    """The group midthickness surface for a CIFTI cortical structure name."""
    hemi = "L" if structure.endswith("_LEFT") else "R"
    hits = sorted(surf_dir.glob(SURFACE_GLOB.format(hemi=hemi)))
    if not hits:
        raise FileNotFoundError(f"no {SURFACE_GLOB.format(hemi=hemi)} in {surf_dir} for {structure}")
    if len(hits) > 1:
        raise FileNotFoundError(
            f"{len(hits)} candidate {hemi} midthickness surfaces in {surf_dir}: "
            f"{[h.name for h in hits]} — keep one, or point --surf-dir at a narrower directory"
        )
    return hits[0]


def dlabel_centroids(dlabel: Path, surf_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Centroid and grayordinate count per dlabel label name."""
    import nibabel as nib

    img = nib.load(str(dlabel))
    keys = np.asarray(img.get_fdata()[0], dtype=int)
    bm = img.header.get_axis(1)
    names = np.asarray(bm.name)
    table = img.header.get_axis(0).label[0]

    # The CIFTI names its own surface structures and their vertex counts; take both from it rather than assuming which structures a given dlabel carries.
    surface_structures = dict(bm.nvertices)

    xyz = np.full((keys.size, 3), np.nan)
    for structure, n_vertices in surface_structures.items():
        surf = find_surface(surf_dir, structure)
        verts = nib.load(str(surf)).darrays[0].data
        if verts.shape[0] != n_vertices:
            raise ValueError(f"{surf.name}: {verts.shape[0]} vertices, dlabel expects {n_vertices}")
        on = names == structure
        xyz[on] = verts[np.asarray(bm.vertex)[on]]
    in_volume = ~np.isin(names, list(surface_structures))
    if in_volume.any():
        xyz[in_volume] = nib.affines.apply_affine(bm.affine, np.asarray(bm.voxel)[in_volume])
    if not np.isfinite(xyz).all():
        raise ValueError("some grayordinates got no coordinate")

    centroids, sizes = {}, {}
    for key, entry in table.items():
        if key == 0:
            continue
        member = keys == key
        if not member.any():
            raise ValueError(f"dlabel label {key} ({entry[0]}) claims no grayordinates")
        centroids[entry[0]] = xyz[member].mean(axis=0)
        sizes[entry[0]] = int(member.sum())
    return centroids, sizes


def pair_entities(entities: dict, centroids: dict[str, np.ndarray]) -> dict[str, str]:
    """Atlas entity name -> dlabel label name, via the atlas's own name/alternateName."""
    mapping = {}
    for name, ent in entities.items():
        for candidate in [name, *(ent.get("alternateName") or [])]:
            if candidate in centroids:
                mapping[name] = candidate
                break
    unpaired = sorted(set(entities) - set(mapping))
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} atlas entities have no name or alternateName in the dlabel: "
            f"{unpaired[:5]} — add the dlabel spelling as an alternateName"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("two atlas entities paired with the same dlabel label")
    return mapping


def invariants(centres: dict[str, np.ndarray], cortical: set[str]) -> str:
    """Hemisphere placement and L/R mirror consistency — the reference-free quality measures."""
    wrong = [k for k in cortical if (k.startswith("L_") and centres[k][0] > 0) or (k.startswith("R_") and centres[k][0] < 0)]
    pairs = [(k, "R_" + k[2:]) for k in cortical if k.startswith("L_") and "R_" + k[2:] in centres]
    mirror = np.array([np.linalg.norm(centres[a][1:] - centres[b][1:]) for a, b in pairs])
    return (
        f"{len(wrong)}/{len(cortical)} on the WRONG hemisphere"
        + (f" {sorted(wrong)}" if wrong else "")
        + f"; L/R mirror median {np.median(mirror):.2f} max {mirror.max():.2f} mm"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dlabel", type=Path, required=True)
    ap.add_argument("--surf-dir", type=Path, required=True)
    ap.add_argument("--atlas", default="hcpmmp1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dlabel.exists():
        print(f"ERROR: missing {args.dlabel}", file=sys.stderr)
        return 2

    src = atlas_dseg_path(args.atlas)
    doc = yaml.safe_load(src.read_text())
    entities = doc["terminology"]["entities"]

    centroids, sizes = dlabel_centroids(args.dlabel, args.surf_dir)
    pairing = pair_entities(entities, centroids)

    old = {n: np.array([e["center"]["x"], e["center"]["y"], e["center"]["z"]]) for n, e in entities.items() if e.get("center")}
    new = {n: centroids[pairing[n]] for n in entities}
    cortical = {n for n in entities if pairing[n].endswith("_ROI")}

    print(f"── {src.name} ──")
    print(
        f"  {len(entities)} entities paired with dlabel labels by name/alternateName; "
        f"parcel sizes {min(sizes.values())}–{max(sizes.values())} grayordinates"
    )
    if old:
        shared = sorted(set(old) & set(new))
        d = np.array([np.linalg.norm(old[k] - new[k]) for k in shared])
        print(
            f"  shift: median {np.median(d):.2f} mm, p95 {np.percentile(d, 95):.2f} mm, "
            f"max {d.max():.2f} mm ({shared[int(d.argmax())]})"
        )
        print(f"  old: {invariants(old, cortical & set(old))}")
    print(f"  new: {invariants(new, cortical)}")

    if args.dry_run:
        print("  (dry run — nothing written)")
        return 0

    doc["coordinateSpace"] = TPL
    n_aliased = 0
    for name, ent in entities.items():
        x, y, z = (float(v) for v in new[name])
        ent["center"] = {"x": x, "y": y, "z": z}
        alts = list(ent.get("alternateName") or [])
        # An alias identical to the entity's own name adds nothing (`L_VentralDC` is both).
        added = [a for a in spelling_variants(name) if a not in alts and a != name]
        if added:
            ent["alternateName"] = alts + added
            n_aliased += 1
    if n_aliased:
        print(f"  refreshed alternateName on {n_aliased} entities (all known spelling conventions)")

    out_yaml = retemplated(src, TPL)
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120))
    # No companion `_centers.txt`: Atlas._load_metadata reads one only when the yaml carries NO centers, and every entity here has one. A second copy could only drift.
    stale = [
        src,
        src.with_name(src.name.replace("_dseg.yaml", "_centers.txt")),
        out_yaml.with_name(out_yaml.name.replace("_dseg.yaml", "_centers.txt")),
    ]
    retired = [p.name for p in stale if p.exists() and p != out_yaml and not p.unlink()]
    print(f"  wrote {out_yaml.name} ({len(entities)} centers inline)")
    print(f"  coordinateSpace -> {TPL}" + (f"; retired {', '.join(retired)}" if retired else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
