#!/usr/bin/env python3
"""Create Hansen2022 Schaefer SC+FC multi-graph networks for TVBO database.

Source data
-----------
https://github.com/netneurolab/hansen_receptors/tree/main/data/schaefer

Available matrices
------------------
Only scale-100 (100×100) SC and FC matrices are distributed with the paper.
Coordinate files exist for 100, 200, and 400 parcels but SC/FC connectomes are only provided at scale-100.

SC reconstruction
-----------------
DWI pre-processed with MRtrix3. Fiber orientation distributions from multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD).
Probabilistic streamline tractography + SIFT2 weight optimization.
Group-consensus binary network preserving density and edge-length distributions of individual connectomes. Edge weights: mean log-transformed streamline count of non-zero edges across participants, scaled to [0, 1].

FC reconstruction
-----------------
HCP 3T resting-state fMRI pre-processed with HCP pipeline (gradient non-linearity correction, head-motion correction, distortion correction,
ICA-FIX denoising). Time series parcellated to Schaefer parcels.
FC = Pearson correlation between pairs of regional time series, averaged across all participants and scans.

Reference
---------
Hansen JY, Shafiei G, Markello RD, Smart K, Cox SML, Wu Y, Diez I,
Schirner M, Wirsich J, Bhatt DL, Misic B. (2022). Mapping neurotransmitter systems to the structural and functional organization of the human neocortex.
Nature Neuroscience, 25, 1569–1581.
https://doi.org/10.1038/s41593-022-01186-3"""

import numpy as np
import h5py
import yaml
from datetime import date
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

HANSEN_DIR = Path("/Users/leonmartin_bih/work_data/toolboxes/hansen_receptors/data/schaefer")
ATLAS_DIR = Path("/Users/leonmartin_bih/tools/tvbo/tvbo/database/atlases")
OUT_DIR = Path("/Users/leonmartin_bih/tools/tvbo/tvbo/database/networks")

# ── Provenance strings (verbatim, schema-compliant) ────────────────────────────

HANSEN_REF = (
    "Hansen JY, Shafiei G, Markello RD, Smart K, Cox SML, Wu Y, Diez I, "
    "Schirner M, Wirsich J, Bhatt DL, Misic B. (2022). Mapping neurotransmitter "
    "systems to the structural and functional organization of the human neocortex. "
    "Nature Neuroscience, 25, 1569-1581. "
    "https://doi.org/10.1038/s41593-022-01186-3"
)

SCHAEFER_REF = (
    "Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ, Eickhoff SB, "
    "Yeo BTT. (2018). Local-Global Parcellation of the Human Cerebral Cortex from "
    "Intrinsic Functional Connectivity MRI. Cerebral Cortex, 29(3), 3095-3114. "
    "https://doi.org/10.1093/cercor/bhx179"
)

HCP_REF = (
    "Van Essen DC, Smith SM, Barch DM, Behrens TEJ, Yacoub E, Ugurbil K; "
    "WU-Minn HCP Consortium. (2013). The WU-Minn Human Connectome Project: "
    "an overview. NeuroImage, 80, 62-79. "
    "https://doi.org/10.1016/j.neuroimage.2013.05.041"
)

SC_DESCRIPTION = (
    "Group-consensus structural connectivity from HCP Young Adult cohort. "
    "DWI pre-processed with MRtrix3 (v3). Fiber orientation distributions "
    "using multi-shell multi-tissue constrained spherical deconvolution "
    "(MSMT-CSD). Probabilistic streamline tractography on the generated FODs. "
    "Streamline weights optimized with SIFT2 (cross-section multiplier). "
    "Group-consensus binary network constructed preserving density and "
    "edge-length distributions of individual connectomes. "
    "Edge weights: mean log-transformed streamline count of non-zero edges "
    "across participants, normalized to [0, 1]."
)

FC_DESCRIPTION = (
    "Group-average functional connectivity from HCP Young Adult cohort. "
    "3T resting-state fMRI pre-processed with HCP minimal preprocessing "
    "pipeline (gradient non-linearity correction, motion correction, "
    "distortion correction with opposite-phase-encoding pairs, ICA-FIX "
    "denoising). High-pass filtered (>2000s FWHM). Time series parcellated "
    "to Schaefer cortical regions. FC = Pearson correlation between pairs "
    "of regional time series, averaged across all participants and "
    "all four resting-state scans per participant."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_atlas_entities(seg="7Networks", scale=100):
    """Return (labels, centers_xyz) from the tvbo data atlas YAML."""
    atlas_file = ATLAS_DIR / f"tpl-FSLMNI152_atlas-Schaefer2018_seg-{seg}_scale-{scale}_res-1_desc-ordered_dseg.yaml"
    with open(atlas_file) as f:
        d = yaml.safe_load(f)
    entities = d["terminology"]["entities"]
    labels = list(entities.keys())
    centers = np.array(
        [[entities[k]["center"]["x"], entities[k]["center"]["y"], entities[k]["center"]["z"]] for k in labels],
        dtype=np.float32,
    )
    return labels, centers


def load_hansen_coordinates(scale=100):
    """Load MNI RAS centroids from Hansen coordinates file."""
    coord_file = HANSEN_DIR / "coordinates" / f"Schaefer_{scale}_centres.txt"
    rows = []
    with open(coord_file) as f:
        for line in f:
            parts = line.strip().split()
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(rows, dtype=np.float32)


def extract_functional_network(label):
    """'7Networks_LH_Vis_1' → 'LH_Vis' (hemisphere + network)."""
    parts = label.split("_")
    return f"{parts[1]}_{parts[2]}"


def build_parent_index(labels):
    """Map each parcel label to its functional-network index."""
    seen = {}
    indices = []
    for lbl in labels:
        net = extract_functional_network(lbl)
        if net not in seen:
            seen[net] = len(seen)
        indices.append(seen[net])
    return np.array(indices, dtype=np.int32), list(seen.keys())


# ── Main builder ──────────────────────────────────────────────────────────────


def create_network(seg="7Networks", scale=100):
    """Build and write the SC+FC multi-graph network."""

    # --- load labels / atlas centers (authoritative CBIG order) ---
    labels, atlas_centers = load_atlas_entities(seg, scale)
    n = len(labels)
    assert n == scale, f"Expected {scale} labels from atlas, got {n}"

    # --- load Hansen centroids (should be co-registered with atlas) ---
    hansen_coords = load_hansen_coordinates(scale)
    assert hansen_coords.shape == (n, 3)

    # Sanity-check first parcel: Hansen vs atlas center should agree sub-mm
    delta = abs(hansen_coords[0] - atlas_centers[0]).max()
    assert delta < 2.0, f"Coordinate mismatch for parcel 0: Hansen {hansen_coords[0]} vs atlas {atlas_centers[0]}"

    # --- load matrices (only scale-100 available for Hansen2022) ---
    assert scale == 100, f"Hansen2022 provides SC/FC only for scale-100; scale-{scale} requested."
    sc_wt = np.load(HANSEN_DIR / "sc_weighted.npy").astype(np.float32)
    sc_bin = np.load(HANSEN_DIR / "sc_binary.npy").astype(np.int32)
    fc_wt = np.load(HANSEN_DIR / "fc_weighted.npy").astype(np.float32)

    # Enforce symmetry (FC has floating-point asymmetry ~1e-15)
    fc_wt = ((fc_wt + fc_wt.T) / 2.0).astype(np.float32)

    # --- functional-network hierarchy ---
    parent_index, net_labels = build_parent_index(labels)

    # --- BIDS filename base ---
    stem = f"tpl-FSLMNI152_cohort-HCPYA_rec-Hansen2022_atlas-Schaefer2018_seg-{seg}_scale-{scale}_desc-SCFC_relmat"
    h5_name = stem + ".h5"
    yaml_name = stem + ".yaml"

    # ── Write HDF5 ─────────────────────────────────────────────────────────
    h5_path = OUT_DIR / h5_name
    with h5py.File(h5_path, "w") as hf:
        # SC weighted connectivity (primary edge measure for simulation)
        wg = hf.create_group("edges/sc")
        wg.attrs["format"] = "dense"
        wg.attrs["shape"] = sc_wt.shape
        wg.create_dataset("data", data=sc_wt, compression="gzip")

        # SC binary group-consensus mask
        bg = hf.create_group("edges/sc_binary")
        bg.attrs["format"] = "dense"
        bg.attrs["shape"] = sc_bin.shape
        bg.create_dataset("data", data=sc_bin, compression="gzip")

        # FC Pearson correlation
        fg = hf.create_group("edges/fc")
        fg.attrs["format"] = "dense"
        fg.attrs["shape"] = fc_wt.shape
        fg.create_dataset("data", data=fc_wt, compression="gzip")

        # Node metadata
        ng = hf.create_group("nodes")
        ng.create_dataset("coordinates", data=hansen_coords, compression="gzip")
        ng.create_dataset("parent_index", data=parent_index, compression="gzip")
        str_dt = h5py.special_dtype(vlen=str)
        ng.create_dataset("functional_network_labels", data=net_labels, dtype=str_dt)

    print(f"  HDF5: {h5_path}")

    # ── Build YAML sidecar ─────────────────────────────────────────────────
    nodes_list = [
        {
            "id": i,
            "label": lbl,
            "position": {
                "x": round(float(hansen_coords[i, 0]), 6),
                "y": round(float(hansen_coords[i, 1]), 6),
                "z": round(float(hansen_coords[i, 2]), 6),
            },
        }
        for i, lbl in enumerate(labels)
    ]

    meta = {
        "tvbo_class": "tvbo:Network",
        "schema_version": "tvb-datamodel/0.7.0",
        "label": f"Schaefer2018_{scale}_{seg}_Hansen2022_SCFC",
        "description": (
            f"Schaefer2018 {scale}-parcel {seg} multi-graph network from "
            "Hansen et al. (2022). Contains structural connectivity (SC) "
            "from diffusion MRI tractography and functional connectivity (FC) "
            "from resting-state fMRI, both derived from the HCP Young Adult "
            "cohort. SC edge weights are log-transformed streamline counts "
            "normalized to [0, 1]. FC edge weights are Pearson correlations."
        ),
        "number_of_nodes": n,
        "descriptor": "SCFC",
        "distance_unit": "mm",
        "data_file": h5_name,
        "edges": [
            {
                "label": "sc",
                "format": "dense",
                "weighted": True,
                "valid_diagonal": False,
                "non_negative": True,
                "directed": False,
                "description": "SC weighted connectivity, log-transformed streamline counts normalized to [0, 1]",
            },
            {
                "label": "sc_binary",
                "format": "dense",
                "weighted": False,
                "valid_diagonal": False,
                "non_negative": True,
                "directed": False,
                "description": "SC group-consensus binary mask (density- and edge-length-preserving)",
            },
            {
                "label": "fc",
                "format": "dense",
                "weighted": True,
                "valid_diagonal": False,
                "non_negative": False,
                "directed": False,
                "description": "FC group-average Pearson correlation, HCP Young Adult cohort",
            },
        ],
        "node_mapping": "/nodes/parent_index",
        "bids": {
            "template": "FSLMNI152",
            "cohort": "HCPYA",
            "reconstruction": "Hansen2022",
            "segmentation": seg,
            "scale": str(scale),
            "atlas": "Schaefer2018",
        },
        "parameters": {
            "conduction_speed": {"label": "v", "value": 3.0, "unit": "mm/ms"},
        },
        "parcellation": {
            "atlas": {
                "name": "Schaefer2018",
                "coordinateSpace": "FSLMNI152",
            },
        },
        "tractogram": {
            "name": "Hansen2022",
            "label": "MRtrix3-MSMT-CSD-SIFT2 group-consensus SC",
            "description": SC_DESCRIPTION,
            "processing_pipeline": "MRtrix3-MSMT-CSD-SIFT2",
            "reference": HANSEN_REF,
        },
        "provenance": {
            "date_created": str(date.today()),
            "derived_from": ("https://github.com/netneurolab/hansen_receptors/tree/main/data/schaefer"),
            "generated_by": "tvbo scripts/create_hansen2022_schaefer_networks.py",
            "references": [HANSEN_REF, SCHAEFER_REF, HCP_REF],
        },
        "nodes": nodes_list,
    }

    yaml_path = OUT_DIR / yaml_name
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  YAML: {yaml_path}")
    return stem


if __name__ == "__main__":
    print("Creating Hansen2022 Schaefer SC+FC multi-graph network...")
    print()
    print("Note: Hansen2022 provides SC/FC matrices ONLY at scale-100.")
    print("      Coordinate files exist for 100/200/400 but no SC/FC")
    print("      matrices are distributed for 200 or 400.")
    print()

    stem = create_network(seg="7Networks", scale=100)

    print()
    print("Done. Created:")
    print(f"  {stem}.yaml")
    print(f"  {stem}.h5")
    print()
    print("HDF5 layout:")
    print("  edges/sc/data        (100,100) float32  SC weighted [0,1]")
    print("  edges/sc_binary/data (100,100) int32    SC group-consensus binary")
    print("  edges/fc/data        (100,100) float32  FC Pearson correlation")
    print("  nodes/coordinates    (100,3)   float32  MNI RAS centroids")
    print("  nodes/parent_index   (100,)    int32    → functional network idx")
    print("  nodes/functional_network_labels (14,)   7Networks LH/RH names")
