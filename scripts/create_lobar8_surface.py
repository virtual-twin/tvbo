#!/usr/bin/env python3
"""Create the three Lobar8 surface networks from TemplateFlow's fsaverage 164k pial meshes and Desikan-Killiany (aparc) labels.

Each vertex takes the lobe of its Desikan-Killiany region (``DK_REGION_TO_LOBE``, the cortical grouping of ``create_lobar_network.py``) and indexes the 16 nodes of the Lobar8 avgMatrix network, its ``parent_network``. Writes the surfaces the database holds, each spelling its parent's ``cohort-`` and ``rec-`` (register I11i-e):
  - hemi-L and hemi-R: 163,842 vertices each, a medial-wall vertex (no aparc region) kept as -1;
  - both hemispheres: 327,684 vertices, left then right, the right's triangles offset by the left's vertex count, a medial-wall vertex given the lobe of the nearest labelled vertex of its hemisphere.

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
    "LH_Frontal",
    "LH_Parietal",
    "LH_Temporal",
    "LH_Occipital",
    "LH_Cingulate",
    "LH_Insular",
    "LH_Subcortical",
    "LH_Cerebellum",
    "RH_Frontal",
    "RH_Parietal",
    "RH_Temporal",
    "RH_Occipital",
    "RH_Cingulate",
    "RH_Insular",
    "RH_Subcortical",
    "RH_Cerebellum",
]


def load_dk_labels(hemi: str) -> tuple[np.ndarray, dict[int, str]]:
    """Load Desikan2006 aparc label.gii from templateflow (curated, 164k)."""
    fname = f"tpl-fsaverage_hemi-{hemi}_den-164k_atlas-Desikan2006_seg-aparc_desc-curated_dseg.label.gii"
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


def dk_to_lobar8(labels: np.ndarray, key_to_name: dict[int, str], hemi_prefix: str, vertices: np.ndarray) -> np.ndarray:
    """Map DK vertex labels → Lobar8 indices, a medial-wall vertex kept as -1."""
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


PROVENANCE = {
    "derived_from": "https://github.com/templateflow/tpl-fsaverage",
    "generated_by": "tvbo scripts/create_lobar8_surface.py",
    "references": [
        "https://doi.org/10.1038/s41592-022-01681-2",
        "https://doi.org/10.1002/(SICI)1097-0193(1999)8:4<272::AID-HBM10>3.0.CO;2-4",
        "https://doi.org/10.1016/j.neuroimage.2006.01.021",
        "https://github.com/templateflow/tpl-fsaverage",
    ],
}
"""Where every surface comes from: TemplateFlow's ``tpl-fsaverage`` (Ciric et al. 2022), FreeSurfer's fsaverage (Fischl et al. 1999) and its Desikan-Killiany labels (Desikan et al. 2006)."""

SOURCES = "TemplateFlow's `tpl-fsaverage_hemi-{h}_den-164k_pial.surf.gii`, each vertex taking the lobe of its Desikan-Killiany region in `tpl-fsaverage_hemi-{h}_den-164k_atlas-Desikan2006_seg-aparc_desc-curated_dseg.label.gii` by `DK_REGION_TO_LOBE` in `scripts/create_lobar8_surface.py` and indexing the 16 nodes of its parent network. Normals are each vertex's sum of its triangles' cross products, normalised."
"""The sentences every surface's description shares, ``{h}`` standing for the hemispheres it spans."""


def fill_medial_wall(rm: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """*rm* with each medial-wall vertex (-1) given the lobe of the nearest labelled vertex of *vertices*."""
    filled = rm.copy()
    wall = rm == -1
    _, nearest = KDTree(vertices[~wall]).query(vertices[wall])
    filled[wall] = rm[~wall][nearest]
    return filled


def hemisphere(hemi_code: str, hemi_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A hemisphere's pial vertices, its triangles and its vertex → lobe mapping."""
    vertices, faces = load_fsaverage_pial(hemi_code)
    labels, key2name = load_dk_labels(hemi_code)
    return vertices.astype(np.float32), faces.astype(np.int32), dk_to_lobar8(labels, key2name, hemi_prefix, vertices)


def save_surface(
    vertices, faces, rm, label: str, description: str, parent_network, db_dir: Path, hemi_code: str | None = None
):
    """Build the surface network of *vertices*, *faces* and the vertex → lobe mapping *rm*, and save it under its BIDS name."""
    surf = Network(number_of_nodes=len(vertices))
    surf.label = label
    surf.description = description
    surf.descriptor = "surf"
    surf.distance_unit = "mm"
    surf.time_unit = "ms"
    surf.parcellation = {"atlas": {"name": "Lobar8", "coordinateSpace": "fsaverage"}}
    hemi = {"hemi": hemi_code} if hemi_code else {}
    surf.bids = {
        "template": "fsaverage",
        "cohort": "HCPYA",
        "reconstruction": "avgMatrix",
        **hemi,
        "atlas": "Lobar8",
        "density": "164k",
        "desc": "surf",
        "suffix": "relmat",
    }
    surf.mesh = tvbo_datamodel.Mesh(
        label="CorticalSurface",
        element_type="triangle",
        number_of_vertices=len(vertices),
        number_of_elements=len(faces),
    )
    surf.set_array("mesh/vertices", vertices)
    surf.set_array("mesh/elements", faces)
    surf.set_array("mesh/normals", compute_normals(vertices, faces))
    surf.set_node_mapping(rm, parent_network=parent_network, dataset_path="/mesh/region_mapping")
    surf.provenance = PROVENANCE
    spelled = f"_hemi-{hemi_code}" if hemi_code else ""
    out_yaml = db_dir / f"tpl-fsaverage_cohort-HCPYA_rec-avgMatrix{spelled}_atlas-Lobar8_den-164k_desc-surf_relmat.yaml"
    surf.save(out_yaml)
    print(f"  Saved: {out_yaml.name}, {len(vertices)} vertices, {int((rm == -1).sum())} at -1")


# ── Main ─────────────────────────────────────────────────────────

db_dir = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks"

net8 = Network.from_db(atlas="Lobar8", rec="avgMatrix", desc="SCFC")
print(f"Lobar8 parent: {net8.label}, {net8.number_of_nodes} nodes")

sides = {"L": ("LH", "left"), "R": ("RH", "right")}
built = {code: hemisphere(code, prefix) for code, (prefix, _) in sides.items()}
for code, (prefix, side) in sides.items():
    vertices, faces, rm = built[code]
    description = (
        f"The Lobar8 parcellation on the {side} hemisphere of FreeSurfer's fsaverage, "
        + SOURCES.format(h=code)
        + " A medial-wall vertex, which has no aparc region, is -1. The arrays are as commit e3892f642 wrote them."
    )
    save_surface(
        vertices,
        faces,
        rm,
        f"Lobar8 {prefix} surface (fsaverage 164k, {len(vertices)} vertices)",
        description,
        net8,
        db_dir,
        code,
    )

(left_v, left_f, left_rm), (right_v, right_f, right_rm) = built["L"], built["R"]
vertices = np.vstack([left_v, right_v])
faces = np.vstack([left_f, right_f + len(left_v)])
rm = np.concatenate([fill_medial_wall(left_rm, left_v), fill_medial_wall(right_rm, right_v)])
description = (
    "The Lobar8 parcellation on both hemispheres of FreeSurfer's fsaverage, left then right, the right's triangles offset by the left's vertex count: "
    + SOURCES.format(h="L|R")
    + " A medial-wall vertex takes the lobe of the nearest labelled vertex of its hemisphere, where the hemisphere surfaces keep it -1. The arrays are as commit 2766c501b wrote them."
)
save_surface(vertices, faces, rm, f"Lobar8 surface (fsaverage 164k, {len(vertices)} vertices)", description, net8, db_dir)
