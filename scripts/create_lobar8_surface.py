#!/usr/bin/env python3
"""Create Lobar8 surface network from existing Lobar surface data.

Derives the 16-node Lobar8 surface network from the existing 17-node
Lobar surface by remapping BrainStem vertices (index 16) to unmapped (-1).
"""
from pathlib import Path

import h5py
import numpy as np

from tvbo.classes.network import Network
from tvbo.datamodel import tvbo_datamodel

db_dir = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks"

# Load Lobar8 SC+FC parent network
net8 = Network.from_db(atlas="Lobar8", rec="avgMatrix", desc="SCFC")
print(f"Lobar8 parent: {net8.label}, {net8.number_of_nodes} nodes")

# Load mesh + region_mapping from existing Lobar surface HDF5
h5_lobar = db_dir / "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-surf_relmat.h5"
with h5py.File(h5_lobar, "r") as f:
    vertices = f["mesh/vertices"][:]
    elements = f["mesh/elements"][:]
    normals = f["mesh/normals"][:]
    rm17 = f["mesh/region_mapping"][:]

# Remap: indices 0-15 stay, index 16 (BrainStem) → -1
rm8 = rm17.copy()
brainstem_mask = rm8 == 16
rm8[brainstem_mask] = -1
n_brainstem = int(brainstem_mask.sum())
print(f"Remapped {n_brainstem} BrainStem vertices to -1")
print(f"Region mapping unique values: {np.unique(rm8)}")

n_vertices = len(vertices)
n_elements = len(elements)

# Build surface network
surf8 = Network(nodes=[], edges=[], number_of_nodes=0)
surf8.number_of_nodes = n_vertices
surf8.label = f"Lobar8 surface (fsLR 32k, {n_vertices} vertices)"
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
object.__setattr__(surf8, "_mesh_elements", elements)
object.__setattr__(surf8, "_mesh_normals", normals)

# Link vertices → Lobar8 regions via parent network
surf8.set_node_mapping(
    rm8,
    parent_network=net8,
    dataset_path="/mesh/region_mapping",
)

# Save
out_yaml = db_dir / "tpl-MNI152NLin2009cAsym_cohort-HCPYA_atlas-Lobar8_desc-surf_relmat.yaml"
surf8.save(out_yaml)
print(f"Saved: {out_yaml.name}")
print(f"Saved: {out_yaml.with_suffix('.h5').name}")
