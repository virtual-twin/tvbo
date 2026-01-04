#!/usr/bin/env python
"""Convert tvboptim connectivity data to BEP017 format.

BEP017 (Relationship Matrices) is a BIDS Extension Proposal for storing
connectivity/relationship matrices in a standardized format.

Reference: https://github.com/bids-standard/bids-specification/pull/1902

This script converts SC (structural connectivity) and FC (functional connectivity)
from tvboptim npz format to BEP017-compliant TSV + JSON files.
"""

import json
from pathlib import Path

import numpy as np

# Output directory
OUTPUT_DIR = Path(__file__).parent


def save_dense_tsv(matrix: np.ndarray, filepath: Path) -> None:
    """Save matrix as dense TSV (BEP017 format)."""
    np.savetxt(filepath, matrix, delimiter="\t", fmt="%.8g")


def create_relmat_json(
    relationship_measure: str,
    directed: bool = False,
    weighted: bool = True,
    valid_diagonal: bool = False,
    non_negative: bool = True,
    n_nodes: int = 84,
    atlas: str = "Desikan-Killiany",
    description: str = "",
    sources: list = None,
    software: str = "https://github.com/virtual-twin/tvboptim",
    measure_units: str = None,
) -> dict:
    """Create BEP017-compliant JSON sidecar for relationship matrix."""
    metadata = {
        "RelationshipMeasure": relationship_measure,
        "Directed": directed,
        "Weighted": weighted,
        "ValidDiagonal": valid_diagonal,
        "NonNegative": non_negative,
        "Software": software,
        "Axes": [
            {
                "Name": "row",
                "Size": n_nodes,
                "NodeFile": f"atlas-{atlas.replace('-', '').replace(' ', '')}_nodeindices.tsv",
            },
            {
                "Name": "column",
                "Size": n_nodes,
                "NodeFile": f"atlas-{atlas.replace('-', '').replace(' ', '')}_nodeindices.tsv",
            },
        ],
        "Description": description,
    }
    if sources:
        metadata["Sources"] = sources
    if measure_units:
        metadata["MeasureUnits"] = measure_units
    return metadata


def create_nodeindices_tsv(region_labels: list, atlas_name: str) -> str:
    """Create BEP017 nodeindices TSV content.

    Columns:
    - matrix_index: 0-based index in the matrix
    - node_file: reference to atlas file
    - node_index: index in the atlas file (same as matrix_index for simple case)
    - label: region label (additional column allowed by BEP017)
    """
    lines = ["matrix_index\tnode_file\tnode_index\tlabel"]
    atlas_file = f"atlas-{atlas_name.replace('-', '').replace(' ', '')}"
    for i, label in enumerate(region_labels):
        lines.append(f"{i}\t{atlas_file}\t{i}\t{label}")
    return "\n".join(lines)


def convert_dk_average():
    """Convert dk_average dataset to BEP017 format."""
    from tvboptim.data import load_functional_connectivity, load_structural_connectivity

    output_dir = OUTPUT_DIR / "dk_average"
    output_dir.mkdir(exist_ok=True)

    # Load data
    weights, lengths, region_labels = load_structural_connectivity("dk_average")
    fc = load_functional_connectivity("dk_average")

    # Convert to numpy
    weights = np.array(weights)
    lengths = np.array(lengths)
    fc = np.array(fc)

    n_nodes = weights.shape[0]
    atlas = "DesikanKilliany"

    # --- 1. Node indices file ---
    nodeindices_content = create_nodeindices_tsv(region_labels, atlas)
    nodeindices_path = output_dir / f"atlas-{atlas}_nodeindices.tsv"
    nodeindices_path.write_text(nodeindices_content)
    print(f"Created: {nodeindices_path}")

    # --- 2. Structural Connectivity: Weights ---
    sc_weights_path = output_dir / f"atlas-{atlas}_meas-streamlineCount_relmat.dense.tsv"
    save_dense_tsv(weights, sc_weights_path)

    sc_weights_json = create_relmat_json(
        relationship_measure="streamlineCount",
        directed=False,
        weighted=True,
        valid_diagonal=False,
        non_negative=True,
        n_nodes=n_nodes,
        atlas=atlas,
        description="Structural connectivity weights derived from diffusion MRI tractography. "
        "Values represent normalized streamline counts between regions.",
        sources=["bids::rawdata/sub-average/dwi"],
        measure_units="arbitrary",
    )
    sc_weights_json_path = output_dir / f"atlas-{atlas}_meas-streamlineCount_relmat.json"
    sc_weights_json_path.write_text(json.dumps(sc_weights_json, indent=2))
    print(f"Created: {sc_weights_path}")
    print(f"Created: {sc_weights_json_path}")

    # --- 3. Structural Connectivity: Tract Lengths ---
    sc_lengths_path = output_dir / f"atlas-{atlas}_meas-tractLength_relmat.dense.tsv"
    save_dense_tsv(lengths, sc_lengths_path)

    sc_lengths_json = create_relmat_json(
        relationship_measure="tractLength",
        directed=False,
        weighted=True,
        valid_diagonal=False,
        non_negative=True,
        n_nodes=n_nodes,
        atlas=atlas,
        description="Tract lengths derived from diffusion MRI tractography. "
        "Values represent mean fiber length in millimeters between regions.",
        sources=["bids::rawdata/sub-average/dwi"],
        measure_units="mm",
    )
    sc_lengths_json_path = output_dir / f"atlas-{atlas}_meas-tractLength_relmat.json"
    sc_lengths_json_path.write_text(json.dumps(sc_lengths_json, indent=2))
    print(f"Created: {sc_lengths_path}")
    print(f"Created: {sc_lengths_json_path}")

    # --- 4. Functional Connectivity ---
    fc_path = output_dir / f"atlas-{atlas}_meas-BoldCorrelation_relmat.dense.tsv"
    save_dense_tsv(fc, fc_path)

    fc_json = create_relmat_json(
        relationship_measure="BoldCorrelation",
        directed=False,
        weighted=True,
        valid_diagonal=True,  # Diagonal is 1.0 for correlation
        non_negative=False,  # Correlations can be negative
        n_nodes=n_nodes,
        atlas=atlas,
        description="Functional connectivity derived from resting-state fMRI BOLD signals. "
        "Values represent Pearson correlation coefficients between region time series.",
        sources=["bids::rawdata/sub-average/func"],
        measure_units="r",
    )
    fc_json_path = output_dir / f"atlas-{atlas}_meas-BoldCorrelation_relmat.json"
    fc_json_path.write_text(json.dumps(fc_json, indent=2))
    print(f"Created: {fc_path}")
    print(f"Created: {fc_json_path}")

    # --- 5. Dataset description (optional but good practice) ---
    dataset_description = {
        "Name": "dk_average connectivity dataset",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tvboptim",
                "Version": "0.1.0",
                "CodeURL": "https://github.com/virtual-twin/tvboptim",
            }
        ],
        "SourceDatasets": [
            {
                "DOI": "10.1038/s41467-023-38626-y",
                "URL": "https://doi.org/10.1038/s41467-023-38626-y",
            }
        ],
        "License": "CC-BY-4.0",
    }
    dataset_desc_path = output_dir / "dataset_description.json"
    dataset_desc_path.write_text(json.dumps(dataset_description, indent=2))
    print(f"Created: {dataset_desc_path}")

    print(f"\n✓ Converted dk_average to BEP017 format in {output_dir}")
    print(f"  - {n_nodes} nodes ({atlas} parcellation)")
    print(f"  - SC weights: {sc_weights_path.name}")
    print(f"  - SC lengths: {sc_lengths_path.name}")
    print(f"  - FC: {fc_path.name}")


if __name__ == "__main__":
    convert_dk_average()
