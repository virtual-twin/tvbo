#!/usr/bin/env python3
"""Create Lobar8 surface network from fsaverage native DK parcellation.

Uses the Desikan-Killiany (aparc) surface parcellation from templateflow
on fsaverage 164k to create a clean vertex→lobe mapping. Every cortical
vertex maps to one of the 12 cortical Lobar8 nodes (6 lobes × 2 hemispheres).
Medial wall vertices are assigned to the nearest cortical lobe.

No volumetric nearest-neighbor sampling — purely surface-native labels.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
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
    labels = gii.darrays[0].data  # (163842,) int32
    # Build key→name lookup from label table
    key_to_name = {lab.key: lab.label for lab in gii.labeltable.labels}
    return labels, key_to_name


def load_fsaverage_pial(hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """Load fsaverage 164k pial surface from templateflow."""
    fname = f"tpl-fsaverage_hemi-{hemi}_den-164k_pial.surf.gii"
    path = Path(TF_HOME) / "tpl-fsaverage" / fname
    gii = nib.load(path)
    vertices = gii.darrays[0].data  # POINTSET
    faces = gii.darrays[1].data     # TRIANGLE
    return vertices, faces


def dk_to_lobar8(labels: np.ndarray, key_to_name: dict[int, str],
                 hemi_prefix: str, vertices: np.ndarray) -> np.ndarray:
    """Map DK vertex labels → Lobar8 indices, filling medial wall via KDTree."""
    # Build DK label key → Lobar8 index
    label_to_idx = {}
    for key, name in key_to_name.items():
        if name in DK_REGION_TO_LOBE:
            lobe = DK_REGION_TO_LOBE[name]
            lobar_label = f"{hemi_prefix}_{lobe}"
            label_to_idx[key] = LOBE_ORDER_8.index(lobar_label)

    # Map cortical vertices
    n_verts = len(labels)
    rm = np.full(n_verts, -1, dtype=np.int32)
    for key, idx in label_to_idx.items():
        rm[labels == key] = idx

    # Fill medial wall (-1) from nearest cortical vertex
    unmapped = rm == -1
    n_unmapped = int(unmapped.sum())
    if n_unmapped > 0:
        mapped = ~unmapped
        tree = KDTree(vertices[mapped])
        _, nn_idx = tree.query(vertices[unmapped])
        rm[unmapped] = rm[mapped][nn_idx]
        print(f"  [{hemi_prefix}] {n_unmapped} medial wall vertices → "
              f"nearest cortical lobe")

    return rm


# ── Main ─────────────────────────────────────────────────────────

db_dir = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks"

# Load Lobar8 SC+FC parent network
net8 = Network.from_db(atlas="Lobar8", rec="avgMatrix", desc="SCFC")
print(f"Lobar8 parent: {net8.label}, {net8.number_of_nodes} nodes")

# Load fsaverage 164k surfaces and DK labels per hemisphere
print("Loading fsaverage 164k pial + Desikan2006 aparc labels...")
lh_verts, lh_faces = load_fsaverage_pial("L")
rh_verts, rh_faces = load_fsaverage_pial("R")
lh_labels, lh_key2name = load_dk_labels("L")
rh_labels, rh_key2name = load_dk_labels("R")

# Map DK → Lobar8 per hemisphere
print("Mapping DK regions → Lobar8 lobes...")
lh_rm = dk_to_lobar8(lh_labels, lh_key2name, "LH", lh_verts)
rh_rm = dk_to_lobar8(rh_labels, rh_key2name, "RH", rh_verts)

# Concatenate hemispheres (RH face indices offset by LH vertex count)
vertices = np.vstack([lh_verts, rh_verts]).astype(np.float32)
faces = np.vstack([lh_faces, rh_faces + len(lh_verts)]).astype(np.int32)
region_mapping = np.concatenate([lh_rm, rh_rm])

n_vertices = len(vertices)
n_elements = len(faces)
print(f"Combined: {n_vertices} vertices, {n_elements} triangles")
print(f"Region mapping: unique={np.unique(region_mapping)}, "
      f"min={region_mapping.min()}, max={region_mapping.max()}")
assert region_mapping.min() >= 0, "No vertex should be unmapped!"

# Compute vertex normals
v0 = vertices[faces[:, 0]]
v1 = vertices[faces[:, 1]]
v2 = vertices[faces[:, 2]]
face_normals = np.cross(v1 - v0, v2 - v0)
normals = np.zeros_like(vertices)
for i in range(3):
    np.add.at(normals, faces[:, i], face_normals)
norms = np.linalg.norm(normals, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normals = (normals / norms).astype(np.float32)

# Build surface network
surf8 = Network(nodes=[], edges=[], number_of_nodes=0)
surf8.number_of_nodes = n_vertices
surf8.label = f"Lobar8 surface (fsaverage 164k, {n_vertices} vertices)"
surf8.descriptor = "surf"
surf8.distance_unit = "mm"

surf8.parcellation = {
    "atlas": {
        "name": "Lobar8",
        "coordinateSpace": "MNI152NLin2009cAsym",
    }
}
surf8.bids = {
    "template": "MNI152NLin2009cAsym",
    "cohort": "HCPYA",
    "atlas": "Lobar8",
}

# Store mesh
mesh = tvbo_datamodel.Mesh(
    label="CorticalSurface",
    element_type="triangle",
    number_of_vertices=n_vertices,
    number_of_elements=n_elements,
)
object.__setattr__(surf8, "_mesh", mesh)
object.__setattr__(surf8, "_mesh_vertices", vertices)
object.__setattr__(surf8, "_mesh_elements", faces)
object.__setattr__(surf8, "_mesh_normals", normals)

# Link vertices → Lobar8 regions via parent network
surf8.set_node_mapping(
    region_mapping,
    parent_network=net8,
    dataset_path="/mesh/region_mapping",
)

# Save
out_yaml = db_dir / "tpl-MNI152NLin2009cAsym_cohort-HCPYA_atlas-Lobar8_desc-surf_relmat.yaml"
surf8.save(out_yaml)
print(f"\nSaved: {out_yaml.name}")
print(f"Saved: {out_yaml.with_suffix('.h5').name}")
