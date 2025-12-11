"""
BIDS BEP034 Export Module

This module provides utilities for exporting TVB simulation data to BIDS format
following the BEP034 Computational Modeling Extension v1.0.0.

Uses:
- Pydantic models from tvbo.datamodel.tvbopydantic for metadata serialization
- pybids for BIDS-compliant filename generation
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Import pybids for filename generation
try:
    from bids.layout.writing import build_path
    PYBIDS_AVAILABLE = True
except ImportError:
    PYBIDS_AVAILABLE = False
    build_path = None

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

    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
        }

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
    SimulationProvenance: Optional[SimulationProvenance] = Field(default=None, description="Simulation provenance")


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

        # ID is always last before extension
        if entities.get("id"):
            fname_parts.append(f"id-{entities['id']}")

        # Extension
        ext = entities.get("extension", ".tsv")
        if not ext.startswith("."):
            ext = "." + ext

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
        id_hash: str,
        desc: Optional[str] = None,
        session: Optional[str] = None,
        run: Optional[int] = None,
        extension: str = ".tsv",
    ) -> str:
        """Build path for time series files."""
        entities = {
            "subject": subject,
            "datatype": "ts",
            "ts": ts_label,
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
