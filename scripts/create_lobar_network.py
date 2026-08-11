#!/usr/bin/env python3
"""Create lobar brain-network datasets from dTOR tractogram and avgMatrix.

This script creates two brain-lobe–level networks:

**Lobar SC (17 nodes, rec-dTOR)**
    SC-only network with weights and tract lengths from the dTOR tractogram.

**Lobar8 SCFC (16 nodes, rec-avgMatrix)**
    SC+FC network where both structural (weight, length) and functional
    connectivity are aggregated from the DesikanKilliany 84-node avgMatrix
    (tvboptim dk_average).  BrainStem is excluded (no DK parcellation)
    giving 8 lobes × 2 hemispheres = 16 nodes.

Pipeline:
1. Building a lobar atlas NIfTI from the DesikanKilliany (DKT31) segmentation in MNI152NLin2009cAsym space (from TemplateFlow) 2. Running MRtrix ``tck2connectome`` to compute streamline counts (weights) and mean tract lengths between lobes → Lobar SC 3. Aggregating the DK 84-node avgMatrix SC+FC to lobar level → Lobar8 SCFC 4. Writing tvbo-compliant HDF5+YAML networks into ``tvbo/database/networks/``

The resulting network has anatomical brain lobes as nodes:

    **Cortical (12):** Frontal, Parietal, Temporal, Occipital, Cingulate,
    Insular — each left and right hemisphere.

    **Subcortical (4):** Subcortical, Cerebellum — each left and right.

    **Midline (1):** Brain-Stem.

Total: **17 nodes** with valid anatomical subgroup labels.

Prerequisites
-------------
- ``tck2connectome`` in PATH (MRtrix3)
- dTOR tractogram ``.tck`` file
- Python: templateflow, nibabel, numpy, pyyaml, h5py, tvbo

Usage
-----
::

    python scripts/create_lobar_network.py \\
        --tractogram /path/to/dTOR.tck

    # Custom output directory
    python scripts/create_lobar_network.py \\
        --tractogram /path/to/dTOR.tck \\
        --output-dir tvbo/database/networks/
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.classes.network import Network

import nibabel as nib
import numpy as np
import yaml

# ── Region → Lobe mapping ─────────────────────────────────────────

DK_REGION_TO_LOBE = {
    # Frontal lobe
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
    # Parietal lobe
    "inferiorparietal": "Parietal",
    "postcentral": "Parietal",
    "precuneus": "Parietal",
    "superiorparietal": "Parietal",
    "supramarginal": "Parietal",
    # Temporal lobe
    "bankssts": "Temporal",
    "entorhinal": "Temporal",
    "fusiform": "Temporal",
    "inferiortemporal": "Temporal",
    "middletemporal": "Temporal",
    "parahippocampal": "Temporal",
    "superiortemporal": "Temporal",
    "transversetemporal": "Temporal",
    "temporalpole": "Temporal",
    # Occipital lobe
    "cuneus": "Occipital",
    "lateraloccipital": "Occipital",
    "lingual": "Occipital",
    "pericalcarine": "Occipital",
    # Cingulate cortex
    "caudalanteriorcingulate": "Cingulate",
    "isthmuscingulate": "Cingulate",
    "posteriorcingulate": "Cingulate",
    "rostralanteriorcingulate": "Cingulate",
    # Insular cortex
    "insula": "Insular",
}
"""Canonical FreeSurfer DesikanKilliany/DKT31 cortical region name → anatomical lobe.

Each cortical region appears exactly once; the hemisphere is not part of the key but of the FreeSurfer label numbering (1000+x = LH, 2000+x = RH).
"""

SUBCORTICAL_LABEL_TO_GROUP = {
    # Left subcortical
    10: ("LH", "Subcortical"),  # Left-Thalamus
    11: ("LH", "Subcortical"),  # Left-Caudate
    12: ("LH", "Subcortical"),  # Left-Putamen
    13: ("LH", "Subcortical"),  # Left-Pallidum
    17: ("LH", "Subcortical"),  # Left-Hippocampus
    18: ("LH", "Subcortical"),  # Left-Amygdala
    26: ("LH", "Subcortical"),  # Left-Accumbens
    28: ("LH", "Subcortical"),  # Left-VentralDC
    # Right subcortical
    49: ("RH", "Subcortical"),  # Right-Thalamus
    50: ("RH", "Subcortical"),  # Right-Caudate
    51: ("RH", "Subcortical"),  # Right-Putamen
    52: ("RH", "Subcortical"),  # Right-Pallidum
    53: ("RH", "Subcortical"),  # Right-Hippocampus
    54: ("RH", "Subcortical"),  # Right-Amygdala
    58: ("RH", "Subcortical"),  # Right-Accumbens
    60: ("RH", "Subcortical"),  # Right-VentralDC
    # Cerebellum
    7: ("LH", "Cerebellum"),  # Left-Cerebellum-WM
    8: ("LH", "Cerebellum"),  # Left-Cerebellum-Cortex
    46: ("RH", "Cerebellum"),  # Right-Cerebellum-WM
    47: ("RH", "Cerebellum"),  # Right-Cerebellum-Cortex
    # Brain-Stem (midline, no hemisphere)
    16: (None, "BrainStem"),  # Brain-Stem
}
"""FreeSurfer aseg label → (hemisphere, lobe group) for the subcortical structures (labels < 100) that the templateflow DKT31 atlas carries."""

LOBE_ORDER = [
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
    "BrainStem",
]
"""Ordered lobe labels of the output network, defining the node ordering of every connectivity matrix."""


LOBE_ORDER_8 = [lbl for lbl in LOBE_ORDER if lbl != "BrainStem"]
"""Lobar8 node ordering: 8 lobes per hemisphere, restricted to the lobes that carry both SC and FC data (BrainStem has neither)."""


DK_ABBREV_TO_LOBE = {
    # Frontal
    "CMFG": "Frontal",
    "LOFG": "Frontal",
    "MOFG": "Frontal",
    "PaCG": "Frontal",
    "POP": "Frontal",
    "POR": "Frontal",
    "PTR": "Frontal",
    "PrCG": "Frontal",
    "RMFG": "Frontal",
    "SFG": "Frontal",
    "FP": "Frontal",
    # Parietal
    "IPG": "Parietal",
    "PoCG": "Parietal",
    "PCU": "Parietal",
    "SPG": "Parietal",
    "SMG": "Parietal",
    # Temporal
    "BSTS": "Temporal",
    "EC": "Temporal",
    "FG": "Temporal",
    "ITG": "Temporal",
    "MTG": "Temporal",
    "PHIG": "Temporal",
    "STG": "Temporal",
    "TTG": "Temporal",
    "TP": "Temporal",
    # Occipital
    "CU": "Occipital",
    "LOG": "Occipital",
    "LG": "Occipital",
    "PCAL": "Occipital",
    # Cingulate
    "CACG": "Cingulate",
    "ICG": "Cingulate",
    "PCG": "Cingulate",
    "RACG": "Cingulate",
    # Insular
    "IN": "Insular",
    # Subcortical
    "TH": "Subcortical",
    "CA": "Subcortical",
    "PU": "Subcortical",
    "PA": "Subcortical",
    "HI": "Subcortical",
    "AM": "Subcortical",
    "AC": "Subcortical",
    # Cerebellum
    "CER": "Cerebellum",
}
"""DK 84-node abbreviation (e.g. ``BSTS``, ``CMFG``) → lobe name, used to aggregate the DK FC matrix to the 17-node lobar FC.

The cortical entries follow the same grouping that :data:`DK_REGION_TO_LOBE` spells out in full region names.
"""


def compute_lobar_avgmatrix(dk_network_dir: Path) -> dict[str, np.ndarray]:
    """Aggregate the DK 84-node avgMatrix SC+FC to lobar level.

    Loads the DesikanKilliany avgMatrix weight, length, and FC matrices and aggregates them to 17×17 lobar matrices.  BrainStem has no DK regions, so its row/column will be zero.

    Aggregation:
    - weight: mean of region-pair weights per lobe pair
    - length: weight-averaged mean tract length per lobe pair
    - fc: mean Pearson correlation per lobe pair

    All three are means rather than sums so that lobe size does not dominate: a summed weight scales with n_regions², which destroys the SC-FC relationship.

    Parameters
    ----------
    dk_network_dir : Path
        Directory containing the DK avgMatrix network HDF5 and YAML files.

    Returns
    -------
    matrices : dict
        Keys ``weight``, ``length``, ``fc`` — each a float32 (17, 17) array.
    """
    import h5py

    dk_yaml = dk_network_dir / ("tpl-MNI152NLin2009cAsym_rec-avgMatrix_atlas-DesikanKilliany_desc-SCFC_relmat.yaml")
    dk_h5 = dk_yaml.with_suffix(".h5")

    # Load DK labels
    with open(dk_yaml) as f:
        dk_data = yaml.safe_load(f)
    dk_labels = [n["label"] for n in dk_data["nodes"]]

    # Load all matrices
    with h5py.File(dk_h5, "r") as f:
        w_84 = np.array(f["edges/weight/data"])
        l_84 = np.array(f["edges/length/data"])
        fc_84 = np.array(f["edges/fc/data"])

    n_dk = len(dk_labels)
    n_lobes = len(LOBE_ORDER)

    # Map each DK region → lobe index
    dk_to_lobe = np.full(n_dk, -1, dtype=np.int32)
    for i, label in enumerate(dk_labels):
        hemi_prefix, abbrev = label.split(".", 1)
        hemi = "LH" if hemi_prefix == "L" else "RH"
        lobe = DK_ABBREV_TO_LOBE.get(abbrev)
        if lobe is None:
            continue
        lobe_label = f"{hemi}_{lobe}"
        if lobe_label in LOBE_ORDER:
            dk_to_lobe[i] = LOBE_ORDER.index(lobe_label)

    # Aggregate to lobar level
    lobar_w = np.zeros((n_lobes, n_lobes), dtype=np.float64)
    lobar_wl = np.zeros((n_lobes, n_lobes), dtype=np.float64)  # weight × length
    lobar_fc = np.zeros((n_lobes, n_lobes), dtype=np.float64)
    counts = np.zeros((n_lobes, n_lobes), dtype=np.int32)

    for i in range(n_dk):
        li = dk_to_lobe[i]
        if li < 0:
            continue
        for j in range(n_dk):
            lj = dk_to_lobe[j]
            if lj < 0:
                continue
            lobar_w[li, lj] += w_84[i, j]
            lobar_wl[li, lj] += w_84[i, j] * l_84[i, j]
            lobar_fc[li, lj] += fc_84[i, j]
            counts[li, lj] += 1

    mask = counts > 0
    lobar_fc[mask] /= counts[mask]
    lobar_w[mask] /= counts[mask]

    lobar_l = np.zeros((n_lobes, n_lobes), dtype=np.float64)
    w_mask = lobar_w > 0
    lobar_l[w_mask] = lobar_wl[w_mask] / (lobar_w[w_mask] * counts[w_mask])

    n_mapped = np.count_nonzero(dk_to_lobe >= 0)
    print(f"[avg ] Aggregated DK avgMatrix ({n_dk} regions, {n_mapped} mapped) → {n_lobes}×{n_lobes} lobar SC+FC")
    print(f"[avg ] weight range: [{lobar_w.min():.4f}, {lobar_w.max():.4f}]")
    print(f"[avg ] length range: [{lobar_l[lobar_l > 0].min():.1f}, {lobar_l.max():.1f}] mm")
    print(f"[avg ] FC range: [{lobar_fc.min():.4f}, {lobar_fc.max():.4f}]")

    return {
        "weight": lobar_w.astype(np.float32),
        "length": lobar_l.astype(np.float32),
        "fc": lobar_fc.astype(np.float32),
    }


# ── Build lobar atlas NIfTI ─────────────────────────────────────────


def build_lobar_atlas(output_path: Path) -> tuple[Path, dict]:
    """Create a lobar atlas NIfTI from the DKT31 segmentation.

    Loads the DKT31 atlas from TemplateFlow (MNI152NLin2009cAsym, 1mm), remaps each voxel label to its lobe index (1-based, matching ``LOBE_ORDER``), and saves the result as a NIfTI file.

    Returns
    -------
    output_path : Path
        Path to the newly created lobar atlas NIfTI.
    lobe_centroids : dict
        Mapping of lobe label → (x, y, z) MNI centroid coordinates.
    """
    import templateflow.conf

    tpl_dir = Path(templateflow.conf.TF_HOME) / "tpl-MNI152NLin2009cAsym"
    # Prefer 1mm; fall back to 2mm
    dkt_path = tpl_dir / "tpl-MNI152NLin2009cAsym_res-01_desc-DKT31_dseg.nii.gz"
    if not dkt_path.exists():
        dkt_path = tpl_dir / "tpl-MNI152NLin2009cAsym_res-02_desc-DKT31_dseg.nii.gz"
    if not dkt_path.exists():
        # Try downloading via the API as last resort
        import templateflow.api as tflow

        result = tflow.get(
            "MNI152NLin2009cAsym",
            res="02",
            desc="DKT31",
            suffix="dseg",
            extension=".nii.gz",
        )
        dkt_path = result[0] if isinstance(result, list) and result else result
    if not dkt_path or not Path(dkt_path).exists():
        raise FileNotFoundError(
            "DKT31 atlas not found in TemplateFlow cache. Run:\n"
            '  python -c "import templateflow.api as tflow; '
            "tflow.get('MNI152NLin2009cAsym', res='02', desc='DKT31', "
            "suffix='dseg', extension='.nii.gz')\""
        )

    img = nib.load(dkt_path)
    data = np.asarray(img.dataobj, dtype=np.int32)
    affine = img.affine

    # Build FreeSurfer label → lobe_index mapping
    label_to_lobe_idx = {}  # fs_label → 1-based lobe index

    # Cortical: 1000+x = LH, 2000+x = RH
    for region_name, lobe in DK_REGION_TO_LOBE.items():
        region_indices = _dk_region_name_to_fs_indices(region_name)
        for fs_label in region_indices:
            hemi = "LH" if fs_label < 2000 else "RH"
            lobe_label = f"{hemi}_{lobe}"
            lobe_idx = LOBE_ORDER.index(lobe_label) + 1  # 1-based
            label_to_lobe_idx[fs_label] = lobe_idx

    # Subcortical
    for fs_label, (hemi, group) in SUBCORTICAL_LABEL_TO_GROUP.items():
        if hemi is None:
            lobe_label = group
        else:
            lobe_label = f"{hemi}_{group}"
        lobe_idx = LOBE_ORDER.index(lobe_label) + 1
        label_to_lobe_idx[fs_label] = lobe_idx

    # Remap voxels
    lobar = np.zeros_like(data, dtype=np.int32)
    for fs_label, lobe_idx in label_to_lobe_idx.items():
        mask = data == fs_label
        lobar[mask] = lobe_idx

    # Compute MNI centroids (voxel → world coordinates)
    lobe_centroids = {}
    for idx, lobe_label in enumerate(LOBE_ORDER, start=1):
        voxels = np.argwhere(lobar == idx)
        if len(voxels) == 0:
            lobe_centroids[lobe_label] = (0.0, 0.0, 0.0)
            continue
        centroid_vox = voxels.mean(axis=0)
        centroid_mni = affine @ np.array([*centroid_vox, 1.0])
        lobe_centroids[lobe_label] = tuple(float(v) for v in centroid_mni[:3])

    # Save
    lobar_img = nib.Nifti1Image(lobar.astype(np.int32), affine, img.header)
    nib.save(lobar_img, output_path)
    print(f"[atlas] Lobar atlas saved: {output_path}")
    print(f"[atlas] {len(LOBE_ORDER)} lobes, {np.count_nonzero(lobar)} labeled voxels")

    return output_path, lobe_centroids


# FreeSurfer DKT31 region name → FS lookup table indices
_DK_FS_INDEX = {
    "bankssts": 1,
    "caudalanteriorcingulate": 2,
    "caudalmiddlefrontal": 3,
    "cuneus": 5,
    "entorhinal": 6,
    "fusiform": 7,
    "inferiorparietal": 8,
    "inferiortemporal": 9,
    "isthmuscingulate": 10,
    "lateraloccipital": 11,
    "lateralorbitofrontal": 12,
    "lingual": 13,
    "medialorbitofrontal": 14,
    "middletemporal": 15,
    "parahippocampal": 16,
    "paracentral": 17,
    "parsopercularis": 18,
    "parsorbitalis": 19,
    "parstriangularis": 20,
    "pericalcarine": 21,
    "postcentral": 22,
    "posteriorcingulate": 23,
    "precentral": 24,
    "precuneus": 25,
    "rostralanteriorcingulate": 26,
    "rostralmiddlefrontal": 27,
    "superiorfrontal": 28,
    "superiorparietal": 29,
    "superiortemporal": 30,
    "supramarginal": 31,
    "frontalpole": 32,
    "temporalpole": 33,
    "transversetemporal": 34,
    "insula": 35,
}


def _dk_region_name_to_fs_indices(region_name: str) -> list[int]:
    """Return FreeSurfer label numbers for a DK region (both hemispheres)."""
    idx = _DK_FS_INDEX[region_name]
    return [1000 + idx, 2000 + idx]


# ── Build network ──────────────────────────────────────────────────


def build_network(
    weights: np.ndarray,
    lengths: np.ndarray,
    centroids: dict[str, tuple[float, float, float]],
    fc: np.ndarray | None = None,
) -> "Network":
    """Build a tvbo Network from lobar connectivity matrices.

    Each node also gets a subgroup index that strips the hemisphere prefix, so the groups are Frontal, Parietal, Temporal, Occipital, Cingulate, Insular, Subcortical, Cerebellum and BrainStem.

    Parameters
    ----------
    weights, lengths : ndarray (N, N)
        Connectivity matrices with N = len(LOBE_ORDER).
    centroids : dict
        Lobe label → (x, y, z) MNI centroid.
    fc : ndarray (N, N), optional
        Functional connectivity matrix (mean Pearson correlation).
    """
    from tvbo.classes.network import Network

    network = Network.from_matrix(
        weights=weights,
        lengths=lengths,
        labels=LOBE_ORDER,
    )

    # Set node positions from atlas centroids
    for node in network.nodes:
        c = centroids.get(node.label)
        if c:
            node.position = {"x": float(round(c[0], 4)), "y": float(round(c[1], 4)), "z": float(round(c[2], 4))}

    # Add FC if provided
    if fc is not None:
        network.set_matrix("fc", fc)

    # Metadata
    network.label = "Lobar (dTOR)"
    network.descriptor = "SCFC" if fc is not None else "SC"
    network.distance_unit = "mm"
    network.time_unit = "ms"
    network.number_of_nodes = len(LOBE_ORDER)

    network.parcellation = {
        "atlas": {
            "name": "Lobar",
            "coordinateSpace": "MNI152NLin2009cAsym",
        }
    }
    network.tractogram = {"name": "dTOR"}
    network.bids = {
        "template": "MNI152NLin2009cAsym",
        "cohort": "HCPYA",
        "reconstruction": "dTOR",
        "atlas": "Lobar",
    }

    group_names = []
    mapping = []
    for label in LOBE_ORDER:
        if "_" in label and label.startswith(("LH_", "RH_")):
            group = label.split("_", 1)[1]
        else:
            group = label
        if group not in group_names:
            group_names.append(group)
        mapping.append(group_names.index(group))

    network.set_node_mapping(np.array(mapping, dtype=np.int32))

    return network


def build_lobar8_network(
    lobar_matrices: dict[str, np.ndarray],
    centroids: dict[str, tuple[float, float, float]],
) -> "Network":
    """Build a 16-node (8 per hemisphere) SC+FC network from avgMatrix.

    Both SC (weight, length) and FC come from the same source — the
    DesikanKilliany 84-node avgMatrix, aggregated to lobar level.
    BrainStem is excluded (no DK parcellation for it).

    Parameters
    ----------
    lobar_matrices : dict
        Keys ``weight``, ``length``, ``fc`` — each (17, 17) from
        :func:`compute_lobar_avgmatrix`.  BrainStem row/col is dropped.
    centroids : dict
        Lobe label → (x, y, z) MNI coordinates for node positions.
    """
    from tvbo.classes.network import Network

    n_full = len(LOBE_ORDER)
    bs_idx = LOBE_ORDER.index("BrainStem")
    keep = [i for i in range(n_full) if i != bs_idx]

    # Subset 17×17 → 16×16 (drop BrainStem)
    W = lobar_matrices["weight"][np.ix_(keep, keep)]
    L = lobar_matrices["length"][np.ix_(keep, keep)]
    FC = lobar_matrices["fc"][np.ix_(keep, keep)]

    net8 = Network.from_matrix(
        weights=W,
        lengths=L,
        labels=LOBE_ORDER_8,
    )
    net8.set_matrix("fc", FC)

    # Set node positions from centroids
    for node in net8.nodes:
        c = centroids.get(node.label)
        if c:
            node.position = {"x": float(round(c[0], 4)), "y": float(round(c[1], 4)), "z": float(round(c[2], 4))}

    # Metadata
    net8.label = "Lobar8 (avgMatrix)"
    net8.descriptor = "SCFC"
    net8.distance_unit = "mm"
    net8.time_unit = "ms"
    net8.number_of_nodes = len(LOBE_ORDER_8)

    net8.parcellation = {
        "atlas": {
            "name": "Lobar8",
            "coordinateSpace": "MNI152NLin2009cAsym",
        }
    }
    net8.bids = {
        "template": "MNI152NLin2009cAsym",
        "reconstruction": "avgMatrix",
        "atlas": "Lobar8",
    }

    # Subgroup mapping
    group_names = []
    mapping = []
    for label in LOBE_ORDER_8:
        if "_" in label and label.startswith(("LH_", "RH_")):
            group = label.split("_", 1)[1]
        else:
            group = label
        if group not in group_names:
            group_names.append(group)
        mapping.append(group_names.index(group))
    net8.set_node_mapping(np.array(mapping, dtype=np.int32))

    return net8


# ── Build surface network ─────────────────────────────────────────


def load_fslr32k_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load fsLR 32k midthickness + inflated surfaces from TemplateFlow.

    Returns combined midthickness vertices (N,3), triangles (M,3), hemisphere index per vertex (0=LH, 1=RH), and inflated vertices (N,3).
    The inflated surface is used for computing representative lobe positions (avoids folding artifacts).
    """
    from templateflow.conf import TF_HOME

    fslr_dir = Path(TF_HOME) / "tpl-fsLR"
    all_verts, all_tris, all_hemi, all_infl = [], [], [], []
    offset = 0

    for hemi in ("L", "R"):
        mid_path = fslr_dir / f"tpl-fsLR_den-32k_hemi-{hemi}_midthickness.surf.gii"
        infl_path = fslr_dir / f"tpl-fsLR_den-32k_hemi-{hemi}_inflated.surf.gii"
        mid_gii = nib.load(mid_path)
        infl_gii = nib.load(infl_path)
        verts = mid_gii.darrays[0].data.astype(np.float32)
        tris = mid_gii.darrays[1].data.astype(np.int32)
        infl_verts = infl_gii.darrays[0].data.astype(np.float32)
        all_verts.append(verts)
        all_infl.append(infl_verts)
        all_tris.append(tris + offset)
        all_hemi.append(np.full(len(verts), 0 if hemi == "L" else 1, dtype=np.int32))
        offset += len(verts)

    return (
        np.concatenate(all_verts),
        np.concatenate(all_tris),
        np.concatenate(all_hemi),
        np.concatenate(all_infl),
    )


def map_vertices_to_lobes(
    vertices: np.ndarray,
    hemi_index: np.ndarray,
    lobar_atlas_path: Path,
) -> np.ndarray:
    """Map each surface vertex to a lobe index by sampling the atlas.

    Uses nearest-neighbor interpolation in voxel space.  Vertices that fall outside the atlas or land on unlabeled voxels get index -1.
    """
    img = nib.load(lobar_atlas_path)
    data = np.asarray(img.dataobj, dtype=np.int32)
    inv_affine = np.linalg.inv(img.affine)

    # MNI → voxel
    ones = np.ones((len(vertices), 1), dtype=np.float32)
    mni_h = np.hstack([vertices, ones])  # (N, 4)
    vox = (inv_affine @ mni_h.T).T[:, :3]  # (N, 3)

    # Nearest-neighbor: round to int voxel indices
    vi = np.round(vox).astype(np.int32)

    # Clamp to volume bounds
    for ax in range(3):
        np.clip(vi[:, ax], 0, data.shape[ax] - 1, out=vi[:, ax])

    # Sample atlas (lobe index 1..17, 0 = unlabeled)
    labels = data[vi[:, 0], vi[:, 1], vi[:, 2]]

    # Convert 1-based lobe index to 0-based for parent_index
    mapping = labels.astype(np.int32) - 1  # 0..16, -1 for unlabeled

    return mapping


def compute_surface_centroids(
    vertices: np.ndarray,
    region_mapping: np.ndarray,
    volume_centroids: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """Compute lobe centroids from cortical surface vertices.

    For a cortical lobe the centroid is a representative vertex that is both directionally central to the lobe and on the outer cortical surface, which keeps it out of the brain interior where cortical folding would otherwise drag a plain mean.
    It is chosen from the most peripheral half of the lobe's vertices (by distance from the brain center) as the one closest to the lobe's centroid direction.

    For subcortical, cerebellum, and brainstem, uses volumetric centroids.

    Parameters
    ----------
    vertices : ndarray (N, 3)
        Midthickness surface vertex positions in MNI coordinates.
    region_mapping : ndarray (N,)
        0-based lobe index per vertex (-1 = unmapped).
    volume_centroids : dict
        Lobe label → (x, y, z) from volumetric atlas (fallback).
    """
    cortical_lobes = {"Frontal", "Parietal", "Temporal", "Occipital", "Cingulate", "Insular"}

    centroids = {}
    for idx, lobe_label in enumerate(LOBE_ORDER):
        parts = lobe_label.split("_", 1)
        lobe_type = parts[1] if len(parts) == 2 else parts[0]

        if lobe_type in cortical_lobes:
            mask = region_mapping == idx
            if mask.sum() > 0:
                lobe_verts = vertices[mask]
                centroid = lobe_verts.mean(axis=0)
                # Direction from brain center to centroid
                direction = centroid / (np.linalg.norm(centroid) + 1e-8)
                # Distance of each vertex from brain center
                radii = np.linalg.norm(lobe_verts, axis=1)
                # Keep only the most peripheral 50% of vertices
                threshold = np.percentile(radii, 50)
                outer_mask = radii >= threshold
                outer_verts = lobe_verts[outer_mask]
                # Unit-normalize so distance from the center does not bias the angular selection.
                norms = np.linalg.norm(outer_verts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                unit_verts = outer_verts / norms
                cosines = unit_verts @ direction
                best = cosines.argmax()
                centroids[lobe_label] = tuple(float(v) for v in outer_verts[best])
                continue
        # Fallback: volumetric centroid (subcortical, cerebellum, brainstem)
        centroids[lobe_label] = volume_centroids[lobe_label]
    return centroids


def build_surface_network(
    lobar_atlas_path: Path,
    parent_network: "Network",
    _precomputed: tuple | None = None,
) -> "Network":
    """Build a surface Network with vertex→lobe mapping.

    Creates a mesh-bearing Network from fsLR 32k surfaces where each vertex is mapped to its parent lobe in the lobar SC network.

    The Network holds no per-vertex Node objects — the geometry lives in the mesh HDF5 — so it is constructed with ``number_of_nodes=0`` and the vertex count is assigned afterwards, which avoids auto-generating 64k placeholder nodes.

    Parameters
    ----------
    _precomputed : tuple, optional
        (vertices, triangles, hemi_index, region_mapping) to skip
        reloading and remapping.
    """
    from tvbo.classes.network import Network
    from tvbo.datamodel import tvbo_datamodel

    if _precomputed is not None:
        vertices, triangles, hemi_index, region_mapping = _precomputed
    else:
        vertices, triangles, hemi_index, _infl = load_fslr32k_mesh()
        region_mapping = map_vertices_to_lobes(vertices, hemi_index, lobar_atlas_path)

    n_vertices = len(vertices)
    n_elements = len(triangles)

    print(f"[surf] fsLR 32k: {n_vertices} vertices, {n_elements} triangles")
    n_mapped = np.count_nonzero(region_mapping >= 0)
    print(f"[surf] {n_mapped}/{n_vertices} vertices mapped to lobes ({n_mapped / n_vertices * 100:.1f}%)")

    # Compute vertex normals from triangles
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).astype(np.float32)

    surface_net = Network(nodes=[], edges=[], number_of_nodes=0)
    surface_net.number_of_nodes = n_vertices
    surface_net.label = f"Lobar surface (fsLR 32k, {n_vertices} vertices)"
    surface_net.descriptor = "surf"
    surface_net.distance_unit = "mm"

    surface_net.parcellation = {
        "atlas": {
            "name": "Lobar",
            "coordinateSpace": "MNI152NLin2009cAsym",
        }
    }
    surface_net.bids = {
        "template": "MNI152NLin2009cAsym",
        "cohort": "HCPYA",
        "reconstruction": "dTOR",
        "atlas": "Lobar",
    }

    # Store mesh data
    mesh = tvbo_datamodel.Mesh(
        label="CorticalSurface",
        element_type="triangle",
        number_of_vertices=n_vertices,
        number_of_elements=n_elements,
    )
    object.__setattr__(surface_net, "_mesh", mesh)
    object.__setattr__(surface_net, "_mesh_vertices", vertices)
    object.__setattr__(surface_net, "_mesh_elements", triangles)
    object.__setattr__(surface_net, "_mesh_normals", normals)

    # Link vertices → lobes via parent network
    surface_net.set_node_mapping(
        region_mapping,
        parent_network=parent_network,
        dataset_path="/mesh/region_mapping",
    )

    return surface_net


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lobar SC network from a dTOR tractogram.")
    parser.add_argument(
        "--tractogram",
        type=Path,
        required=True,
        help="Path to dTOR .tck tractogram file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks",
        help="Output directory for HDF5+YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Persistent directory for intermediate files (atlas, CSVs). "
        "If provided and CSVs exist, tck2connectome is skipped.",
    )
    return parser.parse_args()


def main() -> None:
    from tvbo.data.connectome_build import ensure_mrtrix, run_tck2connectome

    args = parse_args()

    ensure_mrtrix()  # friendly error if MRtrix3 is not installed

    if not args.tractogram.exists():
        raise FileNotFoundError(f"Tractogram not found: {args.tractogram}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use persistent cache dir if provided, otherwise temp dir
    if args.cache_dir:
        workdir = args.cache_dir
        workdir.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        workdir = None
        use_temp = True

    import contextlib

    ctx = tempfile.TemporaryDirectory(prefix="tvbo_lobar_") if use_temp else contextlib.nullcontext()

    with ctx as tmp:
        if use_temp:
            workdir = Path(tmp)

        # Step 1: Build lobar atlas NIfTI
        atlas_nii = workdir / "lobar_atlas.nii.gz"
        _, volume_centroids = build_lobar_atlas(atlas_nii)

        # Step 1b: Compute cortical-surface centroids from fsLR 32k
        vertices, triangles, hemi_index, _infl = load_fslr32k_mesh()
        region_mapping = map_vertices_to_lobes(vertices, hemi_index, atlas_nii)
        centroids = compute_surface_centroids(
            vertices,
            region_mapping,
            volume_centroids,
        )
        n_surf_mapped = np.count_nonzero(region_mapping >= 0)
        print(f"[cent] Surface centroids from {n_surf_mapped} cortical vertices")

        # Step 2: Run tck2connectome (skip if cached CSVs exist)
        weights_csv = workdir / "weights.csv"
        lengths_csv = workdir / "lengths.csv"
        assignments_csv = workdir / "assignments.csv"

        if weights_csv.exists() and lengths_csv.exists():
            print(f"[skip] Using cached CSVs from {workdir}")
        else:
            print("[run ] tck2connectome (weights + lengths) ...")
            run_tck2connectome(
                args.tractogram,
                atlas_nii,
                weights_csv,
                lengths_csv,
                assignments_csv,
            )

        # Parse matrices — tck2connectome outputs N_labels × N_labels CSV where N_labels = max(atlas_label). Our lobar atlas has labels 1..17 so the matrix is 17×17, but tck2connectome may pad to max_label.
        raw_weights = np.loadtxt(weights_csv, delimiter=",")
        raw_lengths = np.loadtxt(lengths_csv, delimiter=",")

        n_lobes = len(LOBE_ORDER)
        # tck2connectome uses label value as row/col index (0-indexed from label 1) so row 0 = label 1, row 16 = label 17
        weights = raw_weights[:n_lobes, :n_lobes]
        lengths = raw_lengths[:n_lobes, :n_lobes]

        print(f"[data] weights: shape={weights.shape}, nnz={np.count_nonzero(weights)}, sum={weights.sum():.0f}")
        print(f"[data] lengths: shape={lengths.shape}, mean={lengths[lengths > 0].mean():.1f} mm")

        # Step 2b: Aggregate DK avgMatrix SC+FC to lobar level
        dk_network_dir = args.output_dir
        dk_h5 = dk_network_dir / ("tpl-MNI152NLin2009cAsym_rec-avgMatrix_atlas-DesikanKilliany_desc-SCFC_relmat.h5")
        lobar_avgmatrix = None
        if dk_h5.exists():
            lobar_avgmatrix = compute_lobar_avgmatrix(dk_network_dir)
        else:
            print(f"[warn] DK avgMatrix not found at {dk_h5}, skipping Lobar8")

        # Step 3: Build tvbo Network (SC only for 17-node Lobar)
        network = build_network(weights, lengths, centroids)

        # Step 4: Save SC network
        out_name = network.bids_filename
        out_path = args.output_dir / out_name
        sidecar = out_path.with_suffix(".yaml")
        companion = sidecar.with_suffix(".h5")

        if sidecar.exists() and not args.overwrite:
            print(f"[skip] {sidecar.name} already exists (use --overwrite)")
        else:
            network.save(sidecar)
            print(f"[ok  ] {sidecar.name}")
            print(f"[ok  ] {companion.name}")

        # Step 5: Build surface network (reuse mesh + mapping from step 1b)
        surface_net = build_surface_network(
            atlas_nii,
            network,
            _precomputed=(vertices, triangles, hemi_index, region_mapping),
        )

        surf_name = network.bids_filename.replace(f"_desc-{network.descriptor}_", "_desc-surf_")
        surf_sidecar = (args.output_dir / surf_name).with_suffix(".yaml")

        if surf_sidecar.exists() and not args.overwrite:
            print(f"[skip] {surf_sidecar.name} already exists (use --overwrite)")
        else:
            surface_net.save(surf_sidecar)
            print(f"[ok  ] {surf_sidecar.name}")
            print(f"[ok  ] {surf_sidecar.with_suffix('.h5').name}")

        # Step 6: Build Lobar8 (16-node SC+FC from avgMatrix, no BrainStem)
        net8 = None
        if lobar_avgmatrix is not None:
            net8 = build_lobar8_network(lobar_avgmatrix, centroids)
            net8_name = net8.bids_filename
            net8_sidecar = (args.output_dir / net8_name).with_suffix(".yaml")

            if net8_sidecar.exists() and not args.overwrite:
                print(f"[skip] {net8_sidecar.name} already exists (use --overwrite)")
            else:
                net8.save(net8_sidecar)
                print(f"[ok  ] {net8_sidecar.name}")
                print(f"[ok  ] {net8_sidecar.with_suffix('.h5').name}")

        # Step 7: Build Lobar8 surface network (16-node, no BrainStem)
        if net8 is not None:
            # Remap: 0-15 stay, 16 (BrainStem) → -1
            rm8 = region_mapping.copy()
            rm8[rm8 == LOBE_ORDER.index("BrainStem")] = -1

            surf8 = build_surface_network(
                atlas_nii,
                net8,
                _precomputed=(vertices, triangles, hemi_index, rm8),
            )
            # Override BIDS to match Lobar8
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
            surf8.label = f"Lobar8 surface (fsLR 32k, {len(vertices)} vertices)"

            surf8_name = net8.bids_filename.replace(f"_desc-{net8.descriptor}_", "_desc-surf_")
            surf8_sidecar = (args.output_dir / surf8_name).with_suffix(".yaml")

            if surf8_sidecar.exists() and not args.overwrite:
                print(f"[skip] {surf8_sidecar.name} already exists (use --overwrite)")
            else:
                surf8.save(surf8_sidecar)
                print(f"[ok  ] {surf8_sidecar.name}")
                print(f"[ok  ] {surf8_sidecar.with_suffix('.h5').name}")

    print(f"[done] {n_lobes}-node lobar network from dTOR tractogram")


if __name__ == "__main__":
    main()
