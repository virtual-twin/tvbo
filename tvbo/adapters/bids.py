"""
BIDS BEP034 Export Module

This module provides utilities for exporting TVB simulation data to BIDS format following the BEP034 Computational Modeling Extension v1.0.0.

Uses:
- Pydantic models from tvbo.datamodel.tvbopydantic for metadata serialization
- pybids for BIDS-compliant filename generation
- nibabel for CIFTI-2 ptseries files
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from bids.layout.writing import build_path
from nibabel import cifti2
from pydantic import BaseModel, ConfigDict, Field

# Import h5py for HDF5 support
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

# Load BEP034 path patterns
BEP034_CONFIG_PATH = Path(__file__).parent / "bep034.json"


# `suffix`/`extension` are value-constrained so an invalid combination fails fast; `desc` is optional.
RESULT_PATTERNS = [
    "[sub-{subject}_]exp-{experiment}[_desc-{description}]_{suffix<result>}{extension<.h5|.yaml|.json>}",
]


def result_entities(experiment, extension: str = ".h5") -> dict:
    """BIDS entities for an experiment's result, with alphanumeric values.

    BIDS entity values must be alphanumeric (no spaces/hyphens/underscores), so the id and dynamics name are stripped to ``[A-Za-z0-9]``. The short model
    *name* (``Kuramoto``) is used for ``desc-`` rather than the verbose label, so
    filenames stay compact. Returns a dict ready for
    :func:`bids.layout.writing.build_path` with ``RESULT_PATTERNS``.
    """

    def _alnum(s):
        return "".join(c for c in str(s) if c.isalnum())

    entities = {"suffix": "result", "extension": extension}
    # Per-subject shards (dataset fan-out) get a sub- entity so their results do not collide when reassembled.
    active_subject = getattr(experiment, "_active_subject", None)
    if active_subject:
        entities["subject"] = _alnum(active_subject)
    eid = getattr(experiment, "id", None)
    if eid is not None:
        entities["experiment"] = _alnum(eid)
    dyn = getattr(experiment, "dynamics", None)
    desc = (getattr(dyn, "name", None) or getattr(dyn, "label", None)) if dyn else None
    if desc and _alnum(desc):
        entities["description"] = _alnum(desc)[:24]  # keep desc- compact
    return entities


def build_result_path(experiment=None, *, entities: dict = None, extension: str = ".h5") -> str:
    """Filename for an experiment result via pybids ``build_path`` + RESULT_PATTERNS."""
    return build_path(entities or result_entities(experiment, extension=extension), RESULT_PATTERNS)


def load_bep034_config() -> dict:
    """Load the BEP034 configuration file."""
    with open(BEP034_CONFIG_PATH, "r") as f:
        return json.load(f)


def compute_id(sidecar_dict: dict) -> str:
    """Compute ID as hash of JSON sidecar content."""
    content = json.dumps(sidecar_dict, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


# Pydantic Models for BEP034 Metadata (Sidecars)


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


class TimeSeriesHDF5Sidecar(BEP034BaseModel):
    """Sidecar metadata for HDF5 time series files preserving full dimensionality."""

    Description: str = Field(..., description="Description of the time series")
    Format: str = Field(default="HDF5", description="File format")
    Shape: list[int] = Field(..., description="Shape of the data array")
    Dimensions: list[str] = Field(..., description="Dimension names in order")
    DimensionLabels: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Labels for each dimension (e.g., {'State Variable': ['V', 'W'], 'Space': ['R1', 'R2']})",
    )
    SamplingFrequency: Optional[float] = Field(default=None, description="Sampling frequency in Hz")
    SamplingPeriod: Optional[float] = Field(default=None, description="Sampling period")
    SamplingPeriodUnits: Optional[str] = Field(default="ms", description="Units for sampling period")
    StartTime: Optional[float] = Field(default=0.0, description="Start time of recording")
    Units: Optional[str] = Field(default="a.u.", description="Units of measurement")
    GeneratedAt: Optional[str] = Field(default=None, description="Timestamp of generation")
    Provenance: Union[SimulationProvenance, None] = None
    StateVariables: Optional[list[str]] = Field(default=None, description="State variable names")
    Datasets: Optional[dict[str, str]] = Field(default=None, description="HDF5 dataset paths and descriptions")


# BEP034 Path Builder


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
        if build_path is not None:
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
            fname_parts.append(f"run-{entities['run']:02d}" if isinstance(entities["run"], int) else f"run-{entities['run']}")

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
            Time series entity label - the output name from the model.
            This can be:
            - A state variable name (e.g., 'V', 'W')
            - A derived output name (e.g., 'Diff' for V-W)
            - Any named output defined in the dynamics model
        suffix : str
            BIDS suffix indicating the observation/output type:
            - 'State': Raw neural state time series (default, no observation model)
            - 'BOLD': fMRI BOLD signal (output convolved with HRF)
            - 'EEG': EEG signal (output with EEG forward model)
            - 'MEG': MEG signal (output with MEG forward model)
            - 'LFP': Local field potential
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

        Examples
        --------
        >>> path_builder.build_ts_path('01', 'V', 'State')    # Raw state V
        'sub-01/ts/sub-01_ts-V_State.ptseries.nii'
        >>> path_builder.build_ts_path('01', 'V', 'BOLD')     # V convolved with HRF
        'sub-01/ts/sub-01_ts-V_BOLD.ptseries.nii'
        >>> path_builder.build_ts_path('01', 'Diff', 'BOLD')  # Derived output (V-W) as BOLD
        'sub-01/ts/sub-01_ts-Diff_BOLD.ptseries.nii'"""
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


# Helper Functions


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


# CIFTI-2 Export Functions


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

    Each parcel is modelled as a single surface vertex under the generic `OTHER` brain structure, which is the usual way to carry parcellated data that has no real surface behind it.

    A ptseries is shaped `(timepoints, parcels)`, so it has nowhere to put a state-variable axis. **Given three-dimensional data, only the first state variable is written.** Multiple state variables need either one file each or a format with a scalar axis, and neither is implemented.

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
    data = np.asarray(data, dtype=np.float32)
    n_regions = len(region_labels)

    # Handle different data shapes
    if data.ndim == 2:
        # (time, regions) -> single state variable
        n_timepoints, n_data_regions = data.shape
        if n_data_regions != n_regions:
            raise ValueError(f"Data has {n_data_regions} regions but {n_regions} labels provided")
        # Reshape to (1, time, regions) for consistency, then transpose later
        data_for_cifti = data  # (time, regions)

    elif data.ndim == 3:
        # (time, state_variables, regions)
        n_timepoints, n_states, n_data_regions = data.shape
        if n_data_regions != n_regions:
            raise ValueError(f"Data has {n_data_regions} regions but {n_regions} labels provided")
        if state_variable_labels and len(state_variable_labels) != n_states:
            raise ValueError(f"Data has {n_states} state variables but {len(state_variable_labels)} labels provided")
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

    series_axis = cifti2.SeriesAxis(
        start=0.0,
        step=sample_period_sec,
        size=n_timepoints,
        unit="SECOND",
    )

    parcel_brain_models = []
    for i, label in enumerate(region_labels):
        bm = cifti2.BrainModelAxis.from_surface(
            vertices=[i],
            nvertex=n_regions,
            name="OTHER",
        )
        parcel_brain_models.append((label, bm))

    parcels_axis = cifti2.ParcelsAxis.from_brain_models(parcel_brain_models)

    if data.ndim == 2:
        cifti_data = data_for_cifti.astype(np.float32)
    else:
        if state_variable_labels is None:
            state_variable_labels = [f"sv{i}" for i in range(n_states)]
        cifti_data = data_for_cifti[:, 0, :].astype(np.float32)

    header = cifti2.Cifti2Header.from_axes((series_axis, parcels_axis))
    return cifti2.Cifti2Image(dataobj=cifti_data, header=header)


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

    This is the recommended approach for multi-state variable data, creating one ptseries file per state variable.

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

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D (time, states, regions), got {data.ndim}D")

    n_timepoints, n_states, n_regions = data.shape

    if len(state_variable_labels) != n_states:
        raise ValueError(f"Data has {n_states} states but {len(state_variable_labels)} labels")

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


# HDF5 Export Functions


def write_hdf5_timeseries(
    data: np.ndarray,
    time: np.ndarray,
    path: str | Path,
    labels_dimensions: Optional[dict[str, list[str]]] = None,
    labels_ordering: Optional[tuple[str, ...]] = None,
    sample_period: Optional[float] = None,
    sample_period_unit: str = "ms",
    metadata: Optional[dict] = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """
    Write time series data to HDF5 file preserving full dimensionality.

    This format supports arbitrary dimensionality (e.g., parameter sweeps, multiple modes, etc.) without splitting by state variable.

    Parameters
    ----------
    data : np.ndarray
        Time series data with any dimensionality (e.g., time, state, region, mode)
        or (param_sweep, time, state, region, mode)
    time : np.ndarray
        Time array
    path : str or Path
        Output file path (should end with .h5)
    labels_dimensions : dict, optional
        Labels for each dimension: {'State Variable': ['V', 'W'], 'Space': ['R1', 'R2']}
    labels_ordering : tuple, optional
        Names of dimensions in order: ('Time', 'State Variable', 'Space', 'Mode')
    sample_period : float, optional
        Sampling period
    sample_period_unit : str
        Unit for sample period
    metadata : dict, optional
        Additional metadata to store
    compression : str
        HDF5 compression filter
    compression_opts : int
        Compression level

    Returns
    -------
    Path
        Path to the created file
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure proper extension
    if not str(path).endswith(".h5"):
        path = path.with_suffix(".h5")

    data = np.asarray(data)
    time = np.asarray(time)

    # Default dimension ordering
    if labels_ordering is None:
        if data.ndim == 4:
            labels_ordering = ("Time", "State Variable", "Space", "Mode")
        elif data.ndim == 5:
            labels_ordering = ("Sweep", "Time", "State Variable", "Space", "Mode")
        elif data.ndim == 3:
            labels_ordering = ("Time", "State Variable", "Space")
        elif data.ndim == 2:
            labels_ordering = ("Time", "Space")
        else:
            labels_ordering = tuple(f"dim{i}" for i in range(data.ndim))

    with h5py.File(path, "w") as f:
        # Store main data
        dset = f.create_dataset(
            "data",
            data=data.astype(np.float32),
            compression=compression,
            compression_opts=compression_opts,
        )

        # Store time array
        f.create_dataset("time", data=time.astype(np.float64))

        # Store dimension information as attributes
        dset.attrs["dimensions"] = list(labels_ordering)
        dset.attrs["shape"] = list(data.shape)

        if sample_period is not None:
            dset.attrs["sample_period"] = sample_period
            dset.attrs["sample_period_unit"] = sample_period_unit

        # Store labels for each dimension
        if labels_dimensions:
            labels_grp = f.create_group("labels")
            for dim_name, labels in labels_dimensions.items():
                if labels is not None and len(labels) > 0:
                    # Store as variable-length strings
                    dt = h5py.special_dtype(vlen=str)
                    labels_grp.create_dataset(dim_name, data=labels, dtype=dt)

        # Store additional metadata
        if metadata:
            meta_grp = f.create_group("metadata")
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (dict, list)):
                        # Store complex types as JSON string
                        import json

                        meta_grp.attrs[key] = json.dumps(value, default=str)
                    elif isinstance(value, str):
                        meta_grp.attrs[key] = value
                    elif isinstance(value, (int, float, np.number)):
                        meta_grp.attrs[key] = value

        # Store BIDS-specific attributes
        f.attrs["format"] = "BIDS-BEP034-HDF5"
        f.attrs["version"] = "1.0.0"

    return path


def read_hdf5_timeseries(path: str | Path) -> dict:
    """
    Read time series data from HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file

    Returns
    -------
    dict
        Dictionary with 'data', 'time', 'labels_dimensions', 'dimensions', 'metadata'
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 import")

    path = Path(path)

    result = {}
    with h5py.File(path, "r") as f:
        # Read main data
        result["data"] = f["data"][:]
        result["time"] = f["time"][:]

        # Read dimension info
        dset = f["data"]
        result["dimensions"] = list(dset.attrs.get("dimensions", []))
        result["shape"] = list(dset.attrs.get("shape", []))
        result["sample_period"] = dset.attrs.get("sample_period", None)
        result["sample_period_unit"] = dset.attrs.get("sample_period_unit", "ms")

        # Read labels
        result["labels_dimensions"] = {}
        if "labels" in f:
            for dim_name in f["labels"].keys():
                result["labels_dimensions"][dim_name] = list(f["labels"][dim_name][:])

        # Read metadata
        result["metadata"] = {}
        if "metadata" in f:
            for key, value in f["metadata"].attrs.items():
                try:
                    import json

                    result["metadata"][key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result["metadata"][key] = value

    return result


# BIDS Ingestion Functions


def detect_timeseries_format(ts_dir: Path) -> str:
    """
    Detect the time series format in a BIDS ts/ directory.

    Parameters
    ----------
    ts_dir : Path
        Path to the ts/ directory

    Returns
    -------
    str
        Format: 'h5', 'cifti', or 'tsv'
    """
    ts_dir = Path(ts_dir)
    if not ts_dir.exists():
        raise FileNotFoundError(f"Time series directory not found: {ts_dir}")

    # Check for files in order of preference
    h5_files = list(ts_dir.glob("*.h5"))
    if h5_files:
        return "h5"

    cifti_files = list(ts_dir.glob("*.ptseries.nii")) + list(ts_dir.glob("*.ptseries.nii.gz"))
    if cifti_files:
        return "cifti"

    tsv_files = list(ts_dir.glob("*.tsv"))
    if tsv_files:
        return "tsv"

    raise ValueError(f"No recognized time series files found in {ts_dir}")


def read_bids_sidecar(json_path: Path) -> dict:
    """Read a BIDS JSON sidecar file."""
    with open(json_path, "r") as f:
        return json.load(f)


def read_cifti_ptseries(path: Path) -> tuple[np.ndarray, list[str], float, str]:
    """
    Read a CIFTI-2 ptseries file.

    Parameters
    ----------
    path : Path
        Path to the ptseries.nii file

    Returns
    -------
    tuple
        (data, region_labels, sample_period, sample_period_unit)
    """

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())

    # Extract axis information
    header = img.header
    axes = [header.get_axis(i) for i in range(header.number_of_mapped_indices)]

    # Find SeriesAxis for time info
    sample_period = 1.0
    sample_period_unit = "s"
    for ax in axes:
        if isinstance(ax, cifti2.SeriesAxis):
            sample_period = ax.step
            if ax.unit == "SECOND":
                sample_period_unit = "s"
            break

    # Find ParcelsAxis for region labels
    region_labels = []
    for ax in axes:
        if isinstance(ax, cifti2.ParcelsAxis):
            # Iterate over parcels using get_element(i) -> (name, voxels, vertices)
            region_labels = [ax.get_element(i)[0] for i in range(ax.size)]
            break

    return data, region_labels, sample_period, sample_period_unit


def read_bids_timeseries(
    ts_dir: Path,
    format: Optional[str] = None,
) -> dict:
    """
    Read time series data from a BIDS ts/ directory.

    Automatically detects the format (h5, cifti, tsv) unless specified.

    Parameters
    ----------
    ts_dir : Path
        Path to the ts/ directory
    format : str, optional
        Force a specific format ('h5', 'cifti', 'tsv')

    Returns
    -------
    dict
        Dictionary with:
        - 'data': numpy array (time, state_vars, regions, modes)
        - 'time': time array
        - 'labels_dimensions': dimension labels
        - 'sample_period': sampling period
        - 'sample_period_unit': unit string
        - 'state_variables': list of state variable names
        - 'region_labels': list of region names
        - 'sidecars': list of sidecar metadata dicts
        - 'format': detected format string
    """
    ts_dir = Path(ts_dir)

    if format is None:
        format = detect_timeseries_format(ts_dir)

    result = {
        "data": None,
        "time": None,
        "labels_dimensions": {},
        "sample_period": None,
        "sample_period_unit": "ms",
        "state_variables": [],
        "region_labels": [],
        "sidecars": [],
        "format": format,
    }

    if format == "h5":
        # HDF5 format - single file with all dimensions
        h5_files = list(ts_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {ts_dir}")

        h5_path = h5_files[0]
        h5_data = read_hdf5_timeseries(h5_path)

        result["data"] = h5_data["data"]
        result["time"] = h5_data["time"]
        result["labels_dimensions"] = h5_data["labels_dimensions"]
        result["sample_period"] = h5_data["sample_period"]
        result["sample_period_unit"] = h5_data["sample_period_unit"]

        # Extract state variables and regions from labels
        if "State Variable" in h5_data["labels_dimensions"]:
            result["state_variables"] = h5_data["labels_dimensions"]["State Variable"]
        if "Space" in h5_data["labels_dimensions"]:
            result["region_labels"] = h5_data["labels_dimensions"]["Space"]

        # Read sidecar if exists
        json_path = h5_path.with_suffix(".json")
        if json_path.exists():
            result["sidecars"].append(read_bids_sidecar(json_path))

    elif format == "cifti":
        # CIFTI format - one file per state variable
        cifti_files = sorted(ts_dir.glob("*.ptseries.nii")) + sorted(ts_dir.glob("*.ptseries.nii.gz"))
        if not cifti_files:
            raise FileNotFoundError(f"No CIFTI files found in {ts_dir}")

        all_data = []
        state_vars = []

        for cifti_path in cifti_files:
            # Extract state variable from filename (ts-<var>_)
            fname = cifti_path.name
            import re

            match = re.search(r"ts-([^_]+)_", fname)
            sv_name = match.group(1) if match else f"sv{len(state_vars)}"
            state_vars.append(sv_name)

            # Read CIFTI data
            data, region_labels, sample_period, sample_period_unit = read_cifti_ptseries(cifti_path)
            all_data.append(data)

            # Store region labels from first file
            if not result["region_labels"]:
                result["region_labels"] = region_labels
                result["sample_period"] = sample_period
                result["sample_period_unit"] = sample_period_unit

            # Read sidecar
            json_path = Path(str(cifti_path).replace(".ptseries.nii.gz", ".json").replace(".ptseries.nii", ".json"))
            if json_path.exists():
                result["sidecars"].append(read_bids_sidecar(json_path))

        # Stack state variables: (time, regions) -> (time, n_states, regions, 1)
        if all_data:
            stacked = np.stack(all_data, axis=1)  # (time, n_states, regions)
            result["data"] = np.expand_dims(stacked, axis=-1)  # Add mode dimension
            result["state_variables"] = state_vars
            result["time"] = np.arange(stacked.shape[0]) * result["sample_period"]

    elif format == "tsv":
        # TSV format - one file per state variable
        tsv_files = sorted([f for f in ts_dir.glob("*.tsv") if not f.name.startswith("participants")])
        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {ts_dir}")

        all_data = []
        state_vars = []

        for tsv_path in tsv_files:
            # Extract state variable from filename
            fname = tsv_path.name
            import re

            match = re.search(r"ts-([^_]+)_", fname)
            sv_name = match.group(1) if match else f"sv{len(state_vars)}"
            state_vars.append(sv_name)

            # Read TSV
            df = pd.read_csv(tsv_path, sep="\t")

            # Extract time column if present
            if "time" in df.columns:
                if result["time"] is None:
                    result["time"] = df["time"].values
                df = df.drop(columns=["time"])

            # Store region labels from columns
            if not result["region_labels"]:
                result["region_labels"] = list(df.columns)

            all_data.append(df.values)

            # Read sidecar
            json_path = tsv_path.with_suffix(".json")
            if json_path.exists():
                sidecar = read_bids_sidecar(json_path)
                result["sidecars"].append(sidecar)
                if result["sample_period"] is None and "SamplingPeriod" in sidecar:
                    result["sample_period"] = sidecar["SamplingPeriod"]
                    result["sample_period_unit"] = sidecar.get("SamplingPeriodUnits", "ms")

        # Stack: (time, regions) -> (time, n_states, regions, 1)
        if all_data:
            stacked = np.stack(all_data, axis=1)
            result["data"] = np.expand_dims(stacked, axis=-1)
            result["state_variables"] = state_vars

            if result["time"] is None and result["sample_period"]:
                result["time"] = np.arange(stacked.shape[0]) * result["sample_period"]

    # Build labels_dimensions if not already set
    if not result["labels_dimensions"] and result["data"] is not None:
        result["labels_dimensions"] = {
            "Time": None,  # Time labels not typically stored
            "State Variable": result["state_variables"],
            "Space": result["region_labels"],
            "Mode": [0] if result["data"].ndim > 3 else None,
        }

    return result


def read_bids_network(net_dir: Path) -> dict:
    """
    Read network connectivity from a BIDS net/ directory.

    Parameters
    ----------
    net_dir : Path
        Path to the net/ directory

    Returns
    -------
    dict
        Dictionary with 'weights', 'distances', 'region_labels', 'sidecars'
    """
    net_dir = Path(net_dir)
    result = {
        "weights": None,
        "distances": None,
        "region_labels": None,
        "sidecars": [],
    }

    # Find weights file
    weights_files = list(net_dir.glob("*net-weights*.tsv"))
    if weights_files:
        df = pd.read_csv(weights_files[0], sep="\t", index_col=0)
        result["weights"] = df.values
        result["region_labels"] = list(df.index)

        # Read sidecar
        json_path = weights_files[0].with_suffix(".json")
        if json_path.exists():
            result["sidecars"].append(read_bids_sidecar(json_path))

    # Find distances file
    distances_files = list(net_dir.glob("*net-distances*.tsv"))
    if distances_files:
        df = pd.read_csv(distances_files[0], sep="\t", index_col=0)
        result["distances"] = df.values

        json_path = distances_files[0].with_suffix(".json")
        if json_path.exists():
            result["sidecars"].append(read_bids_sidecar(json_path))

    return result


def read_bids_equations(eq_dir: Path) -> dict:
    """
    Read model equations from a BIDS eq/ directory.

    Parameters
    ----------
    eq_dir : Path
        Path to the eq/ directory

    Returns
    -------
    dict
        Dictionary with model information
    """
    eq_dir = Path(eq_dir)
    result = {
        "model_type": None,
        "parameters": {},
        "state_variables": [],
        "sidecar": None,
    }

    # Find equation JSON files
    eq_files = list(eq_dir.glob("*.json"))
    if eq_files:
        sidecar = read_bids_sidecar(eq_files[0])
        result["sidecar"] = sidecar
        result["model_type"] = sidecar.get("ModelType")
        result["parameters"] = sidecar.get("Parameters", {})
        result["state_variables"] = sidecar.get("StateVariables", [])

    return result


def read_bids_coordinates(coord_dir: Path) -> dict:
    """
    Read coordinates from a BIDS coord/ directory.

    Parameters
    ----------
    coord_dir : Path
        Path to the coord/ directory

    Returns
    -------
    dict
        Dictionary with 'centres', 'region_labels', 'coordinate_system'
    """
    coord_dir = Path(coord_dir)
    result = {
        "centres": None,
        "region_labels": None,
        "coordinate_system": None,
    }

    # Find centres file
    centres_files = list(coord_dir.glob("*coord-centres*.tsv"))
    if centres_files:
        df = pd.read_csv(centres_files[0], sep="\t", index_col=0)
        result["centres"] = df.values
        result["region_labels"] = list(df.index)

        # Read sidecar for coordinate system
        json_path = centres_files[0].with_suffix(".json")
        if json_path.exists():
            sidecar = read_bids_sidecar(json_path)
            result["coordinate_system"] = sidecar.get("CoordinateSystem")

    return result


def find_bids_session_path(
    bids_dir: Path,
    subject: str,
    session: Optional[str] = None,
) -> Path:
    """
    Find the path to a BIDS subject/session directory.

    Parameters
    ----------
    bids_dir : Path
        Root BIDS dataset directory
    subject : str
        Subject ID (with or without 'sub-' prefix)
    session : str, optional
        Session ID (with or without 'ses-' prefix)

    Returns
    -------
    Path
        Path to the subject/session directory
    """
    bids_dir = Path(bids_dir)

    # Normalize subject/session IDs
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    sub_dir = bids_dir / subject

    if not sub_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")

    if session is not None:
        if not session.startswith("ses-"):
            session = f"ses-{session}"
        ses_dir = sub_dir / session
        if not ses_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {ses_dir}")
        return ses_dir

    # Check if there are session directories
    ses_dirs = list(sub_dir.glob("ses-*"))
    if ses_dirs:
        # Return first session if none specified
        return ses_dirs[0]

    return sub_dir


def ingest_bids_session(
    bids_dir: str | Path,
    subject: str,
    session: Optional[str] = None,
) -> dict:
    """
    Ingest all data from a BIDS BEP034 session.

    Parameters
    ----------
    bids_dir : str or Path
        Root BIDS dataset directory
    subject : str
        Subject ID
    session : str, optional
        Session ID

    Returns
    -------
    dict
        Complete ingested data with keys:
        - 'timeseries': time series data dict
        - 'network': network data dict
        - 'equations': model equations dict
        - 'coordinates': coordinates dict
        - 'dataset_description': dataset description dict
        - 'session_path': path to session directory
    """
    bids_dir = Path(bids_dir)
    session_path = find_bids_session_path(bids_dir, subject, session)

    result = {
        "timeseries": None,
        "network": None,
        "equations": None,
        "coordinates": None,
        "dataset_description": None,
        "session_path": session_path,
    }

    # Read dataset description
    desc_path = bids_dir / "dataset_description.json"
    if desc_path.exists():
        result["dataset_description"] = read_bids_sidecar(desc_path)

    # Read time series
    ts_dir = session_path / "ts"
    if ts_dir.exists():
        result["timeseries"] = read_bids_timeseries(ts_dir)

    # Read network
    net_dir = session_path / "net"
    if net_dir.exists():
        result["network"] = read_bids_network(net_dir)

    # Read equations
    eq_dir = session_path / "eq"
    if eq_dir.exists():
        result["equations"] = read_bids_equations(eq_dir)

    # Read coordinates
    coord_dir = session_path / "coord"
    if coord_dir.exists():
        result["coordinates"] = read_bids_coordinates(coord_dir)

    return result


# Utility helpers (merged from tvbo.data.bids_utils)


def get_unique_entity_values(bids_layout, key) -> set:
    """
    Get a set of all unique values for a given entity key from the BIDSLayout files.

    Args:
        bids_layout (BIDSLayout): The BIDSLayout object to extract entities from.
        key (str): The entity key to extract values for (e.g., 'atlas', 'space').

    Returns:
        set: A set of unique values for the specified entity key.
    """
    unique_values = set()
    files = bids_layout.get(return_type="file")

    for file in files:
        entities = bids_layout.parse_file_entities(file)
        if key in entities:
            unique_values.add(entities[key])

    return unique_values
