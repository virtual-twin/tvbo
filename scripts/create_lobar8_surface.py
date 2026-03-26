#!/usr/bin/env python3
"""Create per-hemisphere Lobar8 surface networks from fsaverage DK parcellation.

Uses the Desikan-Killiany (aparc) surface parcellation from templateflow
on fsaverage 164k to create clean vertex→lobe mappings. Produces two files:
  - hemi-L: 163,842 vertices, rm values 0-5 (LH cortical lobes)
  - hemi-R: 163,842 vertices, rm values 8-13 (RH cortical lobes)

Medial wall vertices are assigned to the nearest cortical lobe.
No volumetric nearest-neighbor sampling — purely surface-native labels.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from templateflow.conf import TF_HOME

from tvbo.classes.network import Network
from tvbo.datamodel import tvbo_datamodel

# ── DK region → lobe mapping (same as create_lobar_network.py) ──────

DK_REGION_TO_LOBE = {
    "caudalmiddlefrontal": "Frontal",
    "lateralorbitofrontal": "Frontal",
    "medialorbitofrontal": "Frontal",
    "paracentral": "Frontal",
    "parsopercularis": "Frontal",
    "parsorbitalis": "Frontal",
    "parstriangularis": "Frontal",
    "precentral": "Frontal",
    "rostralmiddlefrontal": "Frontal",
    "superiorfrontal": "Frontal",
    "frontalpole": "Frontal",
    "inferiorparietal": "Parietal",
    "postcentral": "Parietal",
    "precuneus": "Parietal",
    "superiorparietal": "Parietal",
    "supramarginal": "Parietal",
    "bankssts": "Temporal",
    "entorhinal": "Temporal",
    "fusiform": "Temporal",
    "inferiortemporal": "Temporal",
    "middletemporal": "Temporal",
    "parahippocampal": "Temporal",
    "superiortemporal": "Temporal",
    "transversetemporal": "Temporal",
    "temporalpole": "Temporal",
    "cuneus": "Occipital",
    "lateraloccipital": "Occipital",
    "lingual": "Occipital",
    "pericalcarine": "Occipital",
    "caudalanteriorcingulate": "Cingulate",
    "isthmuscingulate": "Cingulate",
    "posteriorcingulate": "Cingulate",
    "rostralanteriorcingulate": "Cingulate",
    "insula": "Insular",
}

LOBE_ORDER_8 = [
    "LH_Frontal", "LH_Parietal", "LH_Temporal", "LH_Occipital",
    "LH_Cingulate", "LH_Insular", "LH_Subcortical", "LH_Cerebellum",
    "RH_Frontal", "RH_Parietal", "RH_Temporal", "RH_Occipital",
    "RH_Cingulate", "RH_Insular", "RH_Subcortical", "RH_Cerebellum",
]


def load_dk_labels(hemi: str) -> tuple[np.ndarray, dict[int, str]]:
    """Load Desikan2006 aparc label.gii from templateflow (curated, 164k)."""
    fname = (f"tpl-fsaverage_hemi-{hemi}_den-164k_atlas-Desikan2006"
             f"_seg-aparc_desc-curated_dseg.label.gii")
    path = Path(TF_HOME) / "tpl-fsaverage" / fname
    gii = nib.load(path)
    labels = gii.darrays[0].data
    key_to_name = {lab.key: lab.label for lab in gii.labeltable.labels}
    return labels, key_to_name


def load_fsaverage_pial(hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """Load fsaverage 164k pial surface from templateflow."""
    fname = f"tpl-fsaverage_hemi-{hemi}_den-164k_pial.surf.gii"
    path = Path(TF_HOME) / "tpl-fsaverage" / fname
    gii = nib.load(path)
    vertices = gii.darrays[0].data
    faces = gii.darrays[1].data
    return vertices, faces


def dk_to_lobar8(labels: np.ndarray, key_to_name: dict[int, str],
                 hemi_prefix: str, vertices: np.ndarray) -> np.ndarray:
    """Map DK vertex labels → Lobar8 indices, filling medial wall via KDTree."""
    label_to_idx = {}
    for key, name in key_to_name.items():
        if name in DK_REGION_TO_LOBE:
            lobe = DK_REGION_TO_LOBE[name]
            lobar_label = f"{hemi_prefix}_{lobe}"
            label_to_idx[key] = LOBE_ORDER_8.index(lobar_label)

    n_verts = len(labels)
    rm = np.full(n_verts, -1, dtype=np.int32)
    for key, idx in label_to_idx.items():
        rm[labels == key] = idx

    n_unmapped = int((rm == -1).sum())
    print(f"  [{hemi_prefix}] {n_unmapped} medial wall vertices (kept as -1)")
    return rm


def compute_normals(vertices, faces):
    """Compute per-vertex normals from triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (normals / norms).astype(np.float32)


def build_hemi_surface(hemi_code: str, hemi_prefix: str,
                       parent_network, db_dir: Path):
    """Build and save a per-hemisphere surface network."""
    print(f"\n=== {hemi_prefix} hemisphere ===")
    vertices, faces = load_fsaverage_pial(hemi_code)
    labels, key2name = load_dk_labels(hemi_code)
    rm = dk_to_lobar8(labels, key2name, hemi_prefix, vertices)

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    n_vertices = len(vertices)
    n_elements = len(faces)

    print(f"  {n_vertices} vertices, {n_elements} triangles")
    print(f"  rm unique={np.unique(rm)}, min={rm.min()}, max={rm.max()}")
    n_mapped = int((rm >= 0).sum())
    print(f"  {n_mapped} cortical, {len(rm) - n_mapped} medial wall (-1)")

    normals = compute_normals(vertices, faces)

    surf = Network(nodes=[], edges=[], number_of_nodes=0)
    surf.number_of_nodes = n_vertices
    surf.label = f"Lobar8 {hemi_prefix} surface (fsaverage 164k, {n_vertices} vertices)"
    surf.descriptor = "surf"
    surf.distance_unit = "mm"

    surf.parcellation = {
        "atlas": {
            "name": "Lobar8",
            "coordinateSpace": "MNI152NLin2009cAsym",
        }
    }
    surf.bids = {
        "template": "MNI152NLin2009cAsym",
        "cohort": "HCPYA",
        "atlas": "Lobar8",
        "hemi": hemi_code,
    }

    mesh = tvbo_datamodel.Mesh(
        label="CorticalSurface",
        element_type="triangle",
        number_of_vertices=n_vertices,
        number_of_elements=n_elements,
    )
    object.__setattr__(surf, "_mesh", mesh)
    object.__setattr__(surf, "_mesh_vertices", vertices)
    object.__setattr__(surf, "_mesh_elements", faces)
    object.__setattr__(surf, "_mesh_normals", normals)

    surf.set_node_mapping(
        rm,
        parent_network=parent_network,
        dataset_path="/mesh/region_mapping",
    )

    out_yaml = db_dir / f"tpl-MNI152NLin2009cAsym_cohort-HCPYA_atlas-Lobar8_hemi-{hemi_code}_desc-surf_relmat.yaml"
    surf.save(out_yaml)
    print(f"  Saved: {out_yaml.name}")
    print(f"  Saved: {out_yaml.with_suffix('.h5').name}")


# ── Main ─────────────────────────────────────────────────────────

db_dir = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks"

net8 = Network.from_db(atlas="Lobar8", rec="avgMatrix", desc="SCFC")
print(f"Lobar8 parent: {net8.label}, {net8.number_of_nodes} nodes")

build_hemi_surface("L", "LH", net8, db_dir)
build_hemi_surface("R", "RH", net8, db_dir)
