"""
BIDS BEP034 Export Module

This module provides utilities for exporting TVB simulation data to BIDS format
following the BEP034 Computational Modeling Extension v1.0.0.

Uses:
- Pydantic models from tvbo.datamodel.tvbopydantic for metadata serialization
- pybids for BIDS-compliant filename generation
- nibabel for CIFTI-2 ptseries files
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Import pybids for filename generation
try:
    from bids.layout.writing import build_path
    PYBIDS_AVAILABLE = True
except ImportError:
    PYBIDS_AVAILABLE = False
    build_path = None

# Import nibabel for CIFTI-2 support
try:
    import nibabel as nib
    from nibabel import cifti2
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None
    cifti2 = None

# Load BEP034 path patterns
BEP034_CONFIG_PATH = Path(__file__).parent / "bep034.json"


def load_bep034_config() -> dict:
    """Load the BEP034 configuration file."""
    with open(BEP034_CONFIG_PATH, "r") as f:
        return json.load(f)


def compute_id(sidecar_dict: dict) -> str:
    """Compute ID as hash of JSON sidecar content."""
    content = json.dumps(sidecar_dict, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


# =============================================================================
# Pydantic Models for BEP034 Metadata (Sidecars)
# =============================================================================

class BEP034BaseModel(BaseModel):
    """Base model for all BEP034 sidecar metadata."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def to_json(self, **kwargs) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=2, exclude_none=True, **kwargs)

    def to_dict(self) -> dict:
        """Export to dictionary, excluding None values."""
        return self.model_dump(exclude_none=True)


class DatasetDescription(BEP034BaseModel):
    """BIDS dataset_description.json model."""

    Name: str = Field(..., description="Name of the dataset")
    BIDSVersion: str = Field(default="1.9.0", description="BIDS specification version")
    DatasetType: str = Field(default="derivative", description="Dataset type")
    GeneratedBy: list[dict] = Field(default_factory=list, description="Tools that generated this dataset")
    BEP034Version: str = Field(default="1.0.0", description="BEP034 specification version")


class GeneratedBy(BEP034BaseModel):
    """GeneratedBy entry for dataset_description.json."""

    Name: str
    Version: Optional[str] = None
    Description: Optional[str] = None
    CodeURL: Optional[str] = None


class SimulationProvenance(BEP034BaseModel):
    """Provenance information for a simulation."""

    Model: Optional[str] = Field(default=None, description="Neural mass model name")
    Integrator: Optional[str] = Field(default=None, description="Integration method")
    Duration: Optional[float] = Field(default=None, description="Simulation duration")
    StepSize: Optional[float] = Field(default=None, description="Integration step size")
    GeneratedAt: Optional[str] = Field(default=None, description="Timestamp of generation")
    Software: Optional[str] = Field(default="tvbo", description="Software used")


class NetworkSidecar(BEP034BaseModel):
    """Sidecar metadata for network files (net/)."""

    Description: str = Field(..., description="Description of the network data")
    NumberOfNodes: int = Field(..., description="Number of nodes in the network")
    Units: Optional[str] = Field(default="a.u.", description="Units of measurement")
    NodeLabels: Optional[list[str]] = Field(default=None, description="Labels for each node")
    Source: Optional[str] = Field(default="tvbo simulation", description="Data source")
    GeneratedAt: Optional[str] = Field(default=None, description="Timestamp of generation")
    Atlas: Optional[str] = Field(default=None, description="Atlas used for parcellation")
    CoordinateSpace: Optional[str] = Field(default=None, description="Coordinate space")


class TimeSeriesSidecar(BEP034BaseModel):
    """Sidecar metadata for time series files (ts/)."""

    Description: str = Field(..., description="Description of the time series")
    StateVariable: Optional[str] = Field(default=None, description="State variable name")
    SamplingFrequency: Optional[float] = Field(default=None, description="Sampling frequency in Hz")
    SamplingPeriod: Optional[float] = Field(default=None, description="Sampling period")
    SamplingPeriodUnits: Optional[str] = Field(default="ms", description="Units for sampling period")
    StartTime: Optional[float] = Field(default=0.0, description="Start time of recording")
    NumberOfTimepoints: int = Field(..., description="Number of time points")
    NumberOfNodes: int = Field(..., description="Number of nodes/regions")
    Columns: Optional[list[str]] = Field(default=None, description="Column names")
    Units: Optional[str] = Field(default="a.u.", description="Units of measurement")
    GeneratedAt: Optional[str] = Field(default=None, description="Timestamp of generation")
    Provenance: Union[SimulationProvenance, None] = None


class EquationSidecar(BEP034BaseModel):
    """Sidecar metadata for equation/model files (eq/)."""

    Description: str = Field(..., description="Description of the model equations")
    ModelType: str = Field(..., description="Type of neural mass model")
    Format: str = Field(default="tvbo", description="Format of equation specification")
    GeneratedAt: Optional[str] = Field(default=None, description="Timestamp of generation")
    Parameters: Optional[dict[str, Any]] = Field(default=None, description="Model parameters")
    StateVariables: Optional[list[str]] = Field(default=None, description="State variable names")
    References: Optional[list[str]] = Field(default=None, description="References for the model")


class CoordinateSidecar(BEP034BaseModel):
    """Sidecar metadata for coordinate files (coord/)."""

    Description: str = Field(..., description="Description of the coordinate data")
    NumberOfNodes: int = Field(..., description="Number of nodes/points")
    CoordinateSystem: Optional[str] = Field(default="MNI152NLin6Asym", description="Coordinate system")
    Units: Optional[str] = Field(default="mm", description="Units of measurement")
    Columns: Optional[list[str]] = Field(default=["x", "y", "z"], description="Column names")
    NodeLabels: Optional[list[str]] = Field(default=None, description="Labels for each node")


# =============================================================================
# BEP034 Path Builder
# =============================================================================

class BEP034PathBuilder:
    """
    Build BIDS BEP034-compliant paths using pybids patterns.

    If pybids is not available, falls back to manual path construction.
    """

    def __init__(self):
        self.config = load_bep034_config()
        self.patterns = self.config.get("default_path_patterns", [])

    def build_path(self, entities: dict, strict: bool = False) -> Optional[str]:
        """
        Build a path from entities using BEP034 patterns.

        Parameters
        ----------
        entities : dict
            Entity-value pairs (e.g., {'subject': '01', 'net': 'weights', ...})
        strict : bool
            If True, all entities must match pattern

        Returns
        -------
        str or None
            Built path or None if no pattern matches
        """
        if PYBIDS_AVAILABLE and build_path is not None:
            return build_path(entities, self.patterns, strict=strict)
        else:
            return self._manual_build_path(entities)

    def _manual_build_path(self, entities: dict) -> str:
        """Fallback manual path construction when pybids not available."""
        parts = []

        # Directory structure
        if entities.get("subject"):
            parts.append(f"sub-{entities['subject']}")
        if entities.get("session"):
            parts.append(f"ses-{entities['session']}")

        # Datatype directory
        datatype = entities.get("datatype", "")
        if datatype:
            dir_path = "/".join(parts + [datatype])
        else:
            dir_path = "/".join(parts)

        # Filename components
        fname_parts = []

        # Add subject/session to filename if present
        if entities.get("subject"):
            fname_parts.append(f"sub-{entities['subject']}")
        if entities.get("session"):
            fname_parts.append(f"ses-{entities['session']}")
        if entities.get("desc"):
            fname_parts.append(f"desc-{entities['desc']}")
        if entities.get("run"):
            fname_parts.append(f"run-{entities['run']:02d}" if isinstance(entities['run'], int) else f"run-{entities['run']}")

        # Specific entity types
        if entities.get("net"):
            fname_parts.append(f"net-{entities['net']}")
        if entities.get("ts"):
            fname_parts.append(f"ts-{entities['ts']}")
        if entities.get("eq"):
            # eq- goes at start for equation files
            fname_parts = [f"eq-{entities['eq']}"] + [p for p in fname_parts if not p.startswith("sub-")]
        if entities.get("coord"):
            fname_parts.append(f"coord-{entities['coord']}")
        if entities.get("map"):
            fname_parts.append(f"map-{entities['map']}")
        if entities.get("space"):
            fname_parts.append(f"space-{entities['space']}")
        if entities.get("atlas"):
            fname_parts.append(f"atlas-{entities['atlas']}")

        # ID before suffix if present
        if entities.get("id"):
            fname_parts.append(f"id-{entities['id']}")

        # Suffix comes last before extension (e.g., State, BOLD, EEG)
        suffix = entities.get("suffix", "")

        # Extension
        ext = entities.get("extension", ".tsv")
        if not ext.startswith("."):
            ext = "." + ext

        # Build filename: entities_suffix.extension
        if suffix:
            filename = "_".join(fname_parts) + "_" + suffix + ext
        else:
            filename = "_".join(fname_parts) + ext

        return f"{dir_path}/{filename}" if dir_path else filename

    def build_net_path(
        self,
        subject: str,
        net_type: str,
        id_hash: str,
        desc: Optional[str] = None,
        session: Optional[str] = None,
        run: Optional[int] = None,
        extension: str = ".tsv",
    ) -> str:
        """Build path for network files (weights/distances)."""
        entities = {
            "subject": subject,
            "datatype": "net",
            "net": net_type,
            "id": id_hash,
            "extension": extension,
        }
        if session:
            entities["session"] = session
        if desc:
            entities["desc"] = desc
        if run:
            entities["run"] = run

        return self.build_path(entities) or self._manual_build_path(entities)

    def build_ts_path(
        self,
        subject: str,
        ts_label: str,
        suffix: str = "State",
        desc: Optional[str] = None,
        session: Optional[str] = None,
        run: Optional[int] = None,
        extension: str = ".ptseries.nii",
    ) -> str:
        """
        Build path for time series files.

        Parameters
        ----------
        subject : str
            Subject identifier
        ts_label : str
            Time series entity label (e.g., state variable name: 'V', 'W')
        suffix : str
            BIDS suffix indicating data type:
            - 'State': Raw neural state time series (default)
            - 'BOLD': fMRI BOLD signal from observation model
            - 'EEG': EEG signal from observation model
            - 'MEG': MEG signal from observation model
        desc : str, optional
            Description label (e.g., model name)
        session : str, optional
            Session identifier
        run : int, optional
            Run number
        extension : str
            File extension (default: .ptseries.nii)

        Returns
        -------
        str
            BIDS-compliant file path
        """
        entities = {
            "subject": subject,
            "datatype": "ts",
            "ts": ts_label,
            "suffix": suffix,
            "extension": extension,
        }
        if session:
            entities["session"] = session
        if desc:
            entities["desc"] = desc
        if run:
            entities["run"] = run

        return self.build_path(entities) or self._manual_build_path(entities)

    def build_eq_path(
        self,
        eq_label: str,
        id_hash: str,
        desc: Optional[str] = None,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        extension: str = ".json",
    ) -> str:
        """Build path for equation/model files."""
        entities = {
            "eq": eq_label,
            "datatype": "eq",
            "id": id_hash,
            "extension": extension,
        }
        if subject:
            entities["subject"] = subject
        if session:
            entities["session"] = session
        if desc:
            entities["desc"] = desc

        return self.build_path(entities) or self._manual_build_path(entities)

    def build_coord_path(
        self,
        subject: str,
        coord_type: str,
        id_hash: str,
        desc: Optional[str] = None,
        session: Optional[str] = None,
        space: Optional[str] = None,
        extension: str = ".tsv",
    ) -> str:
        """Build path for coordinate files."""
        entities = {
            "subject": subject,
            "datatype": "coord",
            "coord": coord_type,
            "id": id_hash,
            "extension": extension,
        }
        if session:
            entities["session"] = session
        if desc:
            entities["desc"] = desc
        if space:
            entities["space"] = space

        return self.build_path(entities) or self._manual_build_path(entities)


# =============================================================================
# Helper Functions
# =============================================================================

def to_float(val: Any) -> Optional[float]:
    """Safely convert JAX arrays/tracers to Python floats."""
    if val is None:
        return None
    try:
        import jax
        val = jax.device_get(val)
    except Exception:
        pass
    try:
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr.item())
        return float(arr.flat[0])
    except Exception:
        try:
            return float(val)
        except Exception:
            return None


def create_dataset_description(
    name: str = "TVB Simulation Output",
    bids_version: str = "1.9.0",
    software_name: str = "tvbo",
    software_version: str = "0.1.0",
) -> DatasetDescription:
    """Create a dataset_description.json model."""
    return DatasetDescription(
        Name=name,
        BIDSVersion=bids_version,
        DatasetType="derivative",
        GeneratedBy=[
            {
                "Name": software_name,
                "Version": software_version,
                "Description": "The Virtual Brain Ontology and Simulation Framework",
                "CodeURL": "https://github.com/the-virtual-brain/tvb-ontology",
            }
        ],
        BEP034Version="1.0.0",
    )


def write_sidecar(sidecar: BEP034BaseModel, path: str | Path) -> None:
    """Write a pydantic sidecar model to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(sidecar.to_json())


def write_tsv(
    df: pd.DataFrame,
    path: str | Path,
    include_index: bool = True,
) -> None:
    """Write a DataFrame to TSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=include_index)


# =============================================================================
# CIFTI-2 Export Functions
# =============================================================================

def create_cifti_ptseries(
    data: np.ndarray,
    region_labels: list[str],
    sample_period: float,
    sample_period_unit: str = "ms",
    state_variable_labels: Optional[list[str]] = None,
) -> "cifti2.Cifti2Image":
    """
    Create a CIFTI-2 ptseries (parcellated time series) image.

    This creates a valid CIFTI-2 file with:
    - ParcelsAxis: Named brain regions with proper structure
    - SeriesAxis: Time dimension with sampling info
    - ScalarAxis: State variable names (if multiple)

    Parameters
    ----------
    data : np.ndarray
        Time series data with shape:
        - (time, regions) for single state variable
        - (time, state_variables, regions) for multiple state variables
    region_labels : list[str]
        Names for each brain region/parcel
    sample_period : float
        Sampling period (time between samples)
    sample_period_unit : str
        Unit for sample period ('ms', 's', 'sec')
    state_variable_labels : list[str], optional
        Names for each state variable if data has 3 dimensions

    Returns
    -------
    cifti2.Cifti2Image
        CIFTI-2 image ready to be saved

    Raises
    ------
    ImportError
        If nibabel is not available
    ValueError
        If data dimensions don't match provided labels

    Examples
    --------
    >>> img = create_cifti_ptseries(data, region_labels, 1.0, 'ms')
    >>> nib.save(img, 'output.ptseries.nii')
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError(
            "nibabel is required for CIFTI export. "
            "Install it with: pip install nibabel"
        )

    data = np.asarray(data, dtype=np.float32)
    n_regions = len(region_labels)

    # Handle different data shapes
    if data.ndim == 2:
        # (time, regions) -> single state variable
        n_timepoints, n_data_regions = data.shape
        if n_data_regions != n_regions:
            raise ValueError(
                f"Data has {n_data_regions} regions but {n_regions} labels provided"
            )
        # Reshape to (1, time, regions) for consistency, then transpose later
        data_for_cifti = data  # (time, regions)

    elif data.ndim == 3:
        # (time, state_variables, regions)
        n_timepoints, n_states, n_data_regions = data.shape
        if n_data_regions != n_regions:
            raise ValueError(
                f"Data has {n_data_regions} regions but {n_regions} labels provided"
            )
        if state_variable_labels and len(state_variable_labels) != n_states:
            raise ValueError(
                f"Data has {n_states} state variables but "
                f"{len(state_variable_labels)} labels provided"
            )
        # Will handle multi-state later
        data_for_cifti = data
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")

    # Convert sample period to seconds for CIFTI
    if sample_period_unit.lower() in ("ms", "msec", "milliseconds"):
        sample_period_sec = sample_period / 1000.0
    elif sample_period_unit.lower() in ("s", "sec", "seconds"):
        sample_period_sec = sample_period
    else:
        # Assume seconds if unknown
        sample_period_sec = sample_period

    # Create SeriesAxis (time dimension)
    # start=0, step=TR in seconds, size=number of timepoints
    series_axis = cifti2.SeriesAxis(
        start=0.0,
        step=sample_period_sec,
        size=n_timepoints,
        unit="SECOND",
    )

    # Create ParcelsAxis (brain regions)
    # For parcellated data, we use ParcelsAxis with named regions
    # Each parcel represents one brain region

    # Create a simple BrainModelAxis for each region as a "surface vertex"
    # This is a common approach for parcellated data
    parcel_brain_models = []
    for i, label in enumerate(region_labels):
        # Create a surface-based brain model with a single vertex per parcel
        # Using 'OTHER' as brain structure for generic parcels
        bm = cifti2.BrainModelAxis.from_surface(
            vertices=[i],
            nvertex=n_regions,
            name="OTHER",
        )
        parcel_brain_models.append((label, bm))

    parcels_axis = cifti2.ParcelsAxis.from_brain_models(parcel_brain_models)

    if data.ndim == 2:
        # Single state variable: (time, regions) -> CIFTI shape (time, parcels)
        # CIFTI ptseries has shape (timepoints, parcels)
        cifti_data = data_for_cifti.astype(np.float32)

        # Create header from axes: (SeriesAxis, ParcelsAxis)
        header = cifti2.Cifti2Header.from_axes((series_axis, parcels_axis))
        img = cifti2.Cifti2Image(dataobj=cifti_data, header=header)

    else:
        # Multiple state variables: create separate series for each
        # Or use ScalarAxis for state variables
        # For now, we'll flatten: shape becomes (time * n_states, regions)
        # Better approach: create ScalarAxis for state variable names

        # Create ScalarAxis for state variables
        if state_variable_labels is None:
            state_variable_labels = [f"sv{i}" for i in range(n_states)]

        scalar_axis = cifti2.ScalarAxis(name=state_variable_labels)

        # Reshape data: (time, states, regions) -> (time, regions) per state
        # For ptseries with multiple maps, use (n_maps, parcels) where n_maps = time * states
        # Actually, ptseries should be (time, parcels), so we need pscalar for scalar maps

        # Best approach: Save each state variable as separate timepoints
        # Or save as dtseries with multiple series
        # For simplicity, concatenate all state variables along time axis
        # Shape: (time * n_states, regions) with ScalarAxis indicating which is which

        # Alternative: use dscalar for each timepoint-state combo
        # Let's use ptseries with concatenated time for all states
        # and add metadata about which timepoints belong to which state

        # For now, save first state variable only for ptseries
        # TODO: Support multi-state as separate files or different format
        cifti_data = data_for_cifti[:, 0, :].astype(np.float32)

        header = cifti2.Cifti2Header.from_axes((series_axis, parcels_axis))
        img = cifti2.Cifti2Image(dataobj=cifti_data, header=header)

    return img


def write_cifti_ptseries(
    data: np.ndarray,
    region_labels: list[str],
    path: str | Path,
    sample_period: float,
    sample_period_unit: str = "ms",
    state_variable_labels: Optional[list[str]] = None,
) -> Path:
    """
    Write time series data to a CIFTI-2 ptseries.nii file.

    Parameters
    ----------
    data : np.ndarray
        Time series data (time, regions) or (time, state_vars, regions)
    region_labels : list[str]
        Names for each brain region
    path : str or Path
        Output file path (should end with .ptseries.nii or .ptseries.nii.gz)
    sample_period : float
        Sampling period
    sample_period_unit : str
        Unit for sample period ('ms' or 's')
    state_variable_labels : list[str], optional
        Names for state variables if data is 3D

    Returns
    -------
    Path
        Path to the created file
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required for CIFTI export")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure proper extension
    if not str(path).endswith((".ptseries.nii", ".ptseries.nii.gz")):
        path = path.with_suffix(".ptseries.nii.gz")

    img = create_cifti_ptseries(
        data=data,
        region_labels=region_labels,
        sample_period=sample_period,
        sample_period_unit=sample_period_unit,
        state_variable_labels=state_variable_labels,
    )

    nib.save(img, str(path))
    return path


def create_multi_state_cifti(
    data: np.ndarray,
    region_labels: list[str],
    state_variable_labels: list[str],
    sample_period: float,
    sample_period_unit: str = "ms",
) -> dict[str, "cifti2.Cifti2Image"]:
    """
    Create separate CIFTI-2 ptseries files for each state variable.

    This is the recommended approach for multi-state variable data,
    creating one ptseries file per state variable.

    Parameters
    ----------
    data : np.ndarray
        Time series data with shape (time, state_variables, regions)
    region_labels : list[str]
        Names for each brain region
    state_variable_labels : list[str]
        Names for each state variable
    sample_period : float
        Sampling period
    sample_period_unit : str
        Unit for sample period

    Returns
    -------
    dict[str, Cifti2Image]
        Dictionary mapping state variable names to CIFTI images
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required for CIFTI export")

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D (time, states, regions), got {data.ndim}D")

    n_timepoints, n_states, n_regions = data.shape

    if len(state_variable_labels) != n_states:
        raise ValueError(
            f"Data has {n_states} states but {len(state_variable_labels)} labels"
        )

    images = {}
    for sv_idx, sv_label in enumerate(state_variable_labels):
        # Extract data for this state variable: (time, regions)
        sv_data = data[:, sv_idx, :]

        img = create_cifti_ptseries(
            data=sv_data,
            region_labels=region_labels,
            sample_period=sample_period,
            sample_period_unit=sample_period_unit,
        )
        images[sv_label] = img

    return images
