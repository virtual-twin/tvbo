#!/usr/bin/env python
"""Import tvboptim dk_average SC/FC data into the tvbo database.

Creates BIDS-named network files in tvbo/database/networks/:
  dk_average SC+FC → tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-avgMatrix_atlas-DesikanKilliany_desc-SCFC_relmat

tvboptim dk_average contains average structural and functional connectivity matrices across the subjects of the HCP Young Adult S900 release, using the 84-node Desikan-Killiany parcellation (FreeSurfer aparc, excluding brain-stem and ventral-DC bilaterally).

Coordinates are sourced from the existing 87-node dTOR DK network via abbreviation → FreeSurfer label mapping.
"""

from pathlib import Path

import numpy as np

from tvbo import Network, database_path
from tvbo.datamodel import tvbo_datamodel

TVBOPTIM_DATA = Path("/Users/leonmartin_bih/work_data/toolboxes/tvboptim/src/tvboptim/data")
NETWORK_DIR = database_path / "networks"

EDGE_DEFS = {
    "weight": {
        "source": "connectivity/dk_average/data.npz",
        "key": "weights",
        "unit": "arbitrary_unit",
        "description": "Streamline count (normalised)",
        "non_negative": True,
        "valid_diagonal": False,
    },
    "length": {
        "source": "connectivity/dk_average/data.npz",
        "key": "lengths",
        "unit": "mm",
        "description": "Mean tract length",
        "non_negative": True,
        "valid_diagonal": False,
    },
    "fc": {
        "source": "functional/dk_average/data.npz",
        "key": "fc",
        "unit": "dimensionless",
        "description": "Pearson correlation of BOLD signal",
        "non_negative": False,
        "valid_diagonal": True,
    },
}
"""Edge properties to import, keyed by edge attribute: the source NPZ file, the key within that file, and the metadata attributes written to the YAML sidecar."""

ABBREV_TO_FREESURFER = {
    # Left cortical
    "L.BSTS": "ctx-lh-bankssts",
    "L.CACG": "ctx-lh-caudalanteriorcingulate",
    "L.CMFG": "ctx-lh-caudalmiddlefrontal",
    "L.CU": "ctx-lh-cuneus",
    "L.EC": "ctx-lh-entorhinal",
    "L.FG": "ctx-lh-fusiform",
    "L.IPG": "ctx-lh-inferiorparietal",
    "L.ITG": "ctx-lh-inferiortemporal",
    "L.ICG": "ctx-lh-isthmuscingulate",
    "L.LOG": "ctx-lh-lateraloccipital",
    "L.LOFG": "ctx-lh-lateralorbitofrontal",
    "L.LG": "ctx-lh-lingual",
    "L.MOFG": "ctx-lh-medialorbitofrontal",
    "L.MTG": "ctx-lh-middletemporal",
    "L.PHIG": "ctx-lh-parahippocampal",
    "L.PaCG": "ctx-lh-paracentral",
    "L.POP": "ctx-lh-parsopercularis",
    "L.POR": "ctx-lh-parsorbitalis",
    "L.PTR": "ctx-lh-parstriangularis",
    "L.PCAL": "ctx-lh-pericalcarine",
    "L.PoCG": "ctx-lh-postcentral",
    "L.PCG": "ctx-lh-posteriorcingulate",
    "L.PrCG": "ctx-lh-precentral",
    "L.PCU": "ctx-lh-precuneus",
    "L.RACG": "ctx-lh-rostralanteriorcingulate",
    "L.RMFG": "ctx-lh-rostralmiddlefrontal",
    "L.SFG": "ctx-lh-superiorfrontal",
    "L.SPG": "ctx-lh-superiorparietal",
    "L.STG": "ctx-lh-superiortemporal",
    "L.SMG": "ctx-lh-supramarginal",
    "L.FP": "ctx-lh-frontalpole",
    "L.TP": "ctx-lh-temporalpole",
    "L.TTG": "ctx-lh-transversetemporal",
    "L.IN": "ctx-lh-insula",
    # Left subcortical
    "L.CER": "left-cerebellum-cortex",
    "L.TH": "left-thalamus",
    "L.CA": "left-caudate",
    "L.PU": "left-putamen",
    "L.PA": "left-pallidum",
    "L.HI": "left-hippocampus",
    "L.AM": "left-amygdala",
    "L.AC": "left-accumbens-area",
    # Right subcortical
    "R.TH": "right-thalamus",
    "R.CA": "right-caudate",
    "R.PU": "right-putamen",
    "R.PA": "right-pallidum",
    "R.HI": "right-hippocampus",
    "R.AM": "right-amygdala",
    "R.AC": "right-accumbens-area",
    # Right cortical
    "R.BSTS": "ctx-rh-bankssts",
    "R.CACG": "ctx-rh-caudalanteriorcingulate",
    "R.CMFG": "ctx-rh-caudalmiddlefrontal",
    "R.CU": "ctx-rh-cuneus",
    "R.EC": "ctx-rh-entorhinal",
    "R.FG": "ctx-rh-fusiform",
    "R.IPG": "ctx-rh-inferiorparietal",
    "R.ITG": "ctx-rh-inferiortemporal",
    "R.ICG": "ctx-rh-isthmuscingulate",
    "R.LOG": "ctx-rh-lateraloccipital",
    "R.LOFG": "ctx-rh-lateralorbitofrontal",
    "R.LG": "ctx-rh-lingual",
    "R.MOFG": "ctx-rh-medialorbitofrontal",
    "R.MTG": "ctx-rh-middletemporal",
    "R.PHIG": "ctx-rh-parahippocampal",
    "R.PaCG": "ctx-rh-paracentral",
    "R.POP": "ctx-rh-parsopercularis",
    "R.POR": "ctx-rh-parsorbitalis",
    "R.PTR": "ctx-rh-parstriangularis",
    "R.PCAL": "ctx-rh-pericalcarine",
    "R.PoCG": "ctx-rh-postcentral",
    "R.PCG": "ctx-rh-posteriorcingulate",
    "R.PrCG": "ctx-rh-precentral",
    "R.PCU": "ctx-rh-precuneus",
    "R.RACG": "ctx-rh-rostralanteriorcingulate",
    "R.RMFG": "ctx-rh-rostralmiddlefrontal",
    "R.SFG": "ctx-rh-superiorfrontal",
    "R.SPG": "ctx-rh-superiorparietal",
    "R.STG": "ctx-rh-superiortemporal",
    "R.SMG": "ctx-rh-supramarginal",
    "R.FP": "ctx-rh-frontalpole",
    "R.TP": "ctx-rh-temporalpole",
    "R.TTG": "ctx-rh-transversetemporal",
    "R.IN": "ctx-rh-insula",
    "R.CER": "right-cerebellum-cortex",
}
"""Label mapping from the tvboptim abbreviations to the FreeSurfer aparc names used by the dTOR DK network."""

DESCRIPTION = "Group-average connectome of the HCP Young Adult S900 release on 84 Desikan-Killiany regions, 34 cortical, 7 subcortical and the cerebellar cortex per hemisphere: normalised streamline counts (`weight`) and mean tract lengths (`length`) from diffusion MRI, and the Pearson correlation of resting-state BOLD (`fc`). The arrays are tvboptim's `dk_average` (`src/tvboptim/data/connectivity/dk_average/data.npz` and `functional/dk_average/data.npz`), cast to float32 by `scripts/import_tvboptim_networks.py` (commit 2f6c965c6); `networks/bids/dk_average/dataset_description.json` lists Schirner, Deco and Ritter (2023) under `SourceDatasets`."
"""The network's description: what the arrays measure, on which regions, and where they come from."""

PROVENANCE = {
    "derived_from": "https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release",
    "references": [
        "https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release",
        "https://doi.org/10.1016/j.neuroimage.2013.05.041",
        "https://doi.org/10.1038/s41467-023-38626-y",
        "https://github.com/virtual-twin/tvboptim",
    ],
    "how_to_acknowledge": [
        "Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University."
    ],
    "generated_by": "tvbo scripts/import_tvboptim_networks.py",
}
"""Where the network comes from: the HCP Young Adult S900 release (Van Essen et al. 2013), Schirner, Deco and Ritter (2023) and tvboptim, with the acknowledgement HCP data asks for."""


def _get_dk87_coords():
    """Load coordinates from the existing 87-node DK network, keyed by label."""
    results = Network.from_db(
        atlas="DesikanKilliany",
        rec="dTOR",
        cohort="HCPYA",
        desc="SC",
    )
    if isinstance(results, list):
        dk87 = [r for r in results if "ranked" not in r.label.lower()][0]
    else:
        dk87 = results
    centers = dk87.get_centers()
    return {node.label: centers[i] for i, node in enumerate(dk87.nodes) if i in centers}


def import_dk_average():
    """Import dk_average SC + FC as a single network."""
    # Load labels from SC source
    sc_data = np.load(TVBOPTIM_DATA / EDGE_DEFS["weight"]["source"], allow_pickle=True)
    labels = [str(lbl) for lbl in sc_data["region_labels"]]

    # Load all edge matrices from declarative definitions
    matrices = {}
    for name, edef in EDGE_DEFS.items():
        data = np.load(TVBOPTIM_DATA / edef["source"], allow_pickle=True)
        matrices[name] = data[edef["key"]].astype(np.float32)

    # Build network from weight + length, then add remaining matrices
    net = Network.from_matrix(matrices["weight"], matrices["length"], labels=labels)
    for name, mat in matrices.items():
        if name not in ("weight", "length"):
            net.set_matrix(name, mat)

    # Transfer coordinates from existing 87-node DK network via label mapping, and give each node its side and its homologue
    dk87_coords = _get_dk87_coords()
    matched = 0
    for node in net.nodes:
        side, other = ("left", "R.") if node.label.startswith("L.") else ("right", "L.")
        node.hemisphere, node.homologue = side, other + node.label[2:]
        fs_label = ABBREV_TO_FREESURFER.get(node.label)
        if fs_label and fs_label in dk87_coords:
            x, y, z = dk87_coords[fs_label]
            node.position = tvbo_datamodel.Coordinate(x=x, y=y, z=z)
            matched += 1

    # Network metadata
    net.label = "DesikanKilliany (avgMatrix SC+FC)"
    net.description = DESCRIPTION
    net.descriptor = "SCFC"
    net.number_of_nodes = len(labels)
    net.number_of_regions = len(labels)
    net.time_unit = "ms"
    net.provenance = PROVENANCE

    # BIDS metadata
    net.bids = tvbo_datamodel.BidsEntities(
        template="MNI152NLin2009cAsym",
        cohort="HCPYA",
        reconstruction="avgMatrix",
        atlas="DesikanKilliany",
        desc="SCFC",
        suffix="relmat",
    )
    net.parcellation = tvbo_datamodel.Parcellation(
        atlas=tvbo_datamodel.BrainAtlas(
            name="DesikanKilliany",
            coordinateSpace="MNI152NLin2009cAsym",
        )
    )

    # Apply declarative edge metadata to template edges
    for e in net.edges:
        lbl = getattr(e, "label", None)
        edef = EDGE_DEFS.get(lbl)
        if edef is None:
            continue
        for attr in ("unit", "description", "non_negative", "valid_diagonal"):
            if attr in edef:
                setattr(e, attr, edef[attr])

    out_path = (NETWORK_DIR / net.bids_filename).with_suffix(".yaml")
    net.save(out_path)
    print(f"Saved: {out_path.name}")
    print(f"  {len(labels)} nodes, {len(EDGE_DEFS)} matrices ({', '.join(EDGE_DEFS)})")
    print(f"  Coordinates matched: {matched}/{len(labels)}")


if __name__ == "__main__":
    import_dk_average()
    print("\nDone. Networks saved to:", NETWORK_DIR)
