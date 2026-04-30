#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
OpenMINDS JSON-LD conversion utilities for TVBO.

This module provides bidirectional conversion between TVBO datamodel objects
and openMINDS-compatible JSON-LD format.

This module is the **single source of truth** for all openMINDS type mappings.
Both runtime conversion and schema generation import from here.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.classes.experiment import SimulationExperiment
    from tvbo.classes.study import SimulationStudy

__all__ = [
    # Context and mappings
    "OPENMINDS_CONTEXT",
    "EXTERNAL_TYPE_MAPPINGS",
    "TVBO_TYPE_MAPPINGS",
    "TVBO_TO_OPENMINDS_TYPE",
    "OPENMINDS_CATEGORIES",
    "OPENMINDS_EXTENDS",
    "SKIP_CLASSES",
    "LINKML_TO_JSON_TYPE",
    "SKIP_FIELDS",
    # Conversion functions
    "experiment_to_openminds",
    "experiment_from_openminds",
    "study_to_openminds",
    "study_from_openminds",
    "save_openminds",
    "load_openminds",
]

# =============================================================================
# OpenMINDS Context and Type Mappings
# =============================================================================

OPENMINDS_CONTEXT = {
    "@vocab": "https://openminds.ebrains.eu/vocab/",
    "tvbo": "https://w3id.org/tvbo/",
    "sands": "https://openminds.ebrains.eu/sands/",
    "core": "https://openminds.ebrains.eu/core/",
    "computation": "https://openminds.ebrains.eu/computation/",
}

# =============================================================================
# Type Mappings: Single Source of Truth
# =============================================================================

# Map LinkML/TVBO class names to existing openMINDS types (namespace:Type)
# These will NOT generate new schemas - use the existing type directly
EXTERNAL_TYPE_MAPPINGS: dict[str, str] = {
    # SANDS types
    "BrainAtlas": "sands:BrainAtlas",
    "BrainAtlasVersion": "sands:BrainAtlasVersion",
    "CommonCoordinateSpace": "sands:CommonCoordinateSpace",
    "CommonCoordinateSpaceVersion": "sands:CommonCoordinateSpaceVersion",
    "ParcellationEntity": "sands:ParcellationEntity",
    "ParcellationEntityVersion": "sands:ParcellationEntityVersion",
    "ParcellationTerminology": "sands:ParcellationTerminology",
    "ParcellationTerminologyVersion": "sands:ParcellationTerminologyVersion",
    "Coordinate": "sands:CoordinatePoint",
    # Core types
    "File": "core:File",
    "FileBundle": "core:FileBundle",
    "DOI": "core:DOI",
    "RRID": "core:RRID",
    "Person": "core:Person",
    "Organization": "core:Organization",
    "Software": "core:Software",
    "SoftwareVersion": "core:SoftwareVersion",
    "QuantitativeValue": "core:QuantitativeValue",
    "QuantitativeValueRange": "core:QuantitativeValueRange",
    # Computation types (simulation environment)
    # Note: We generate our own SoftwareEnvironment with extended fields
}

# Map TVBO class names to openMINDS types (tvbo namespace)
TVBO_TYPE_MAPPINGS: dict[str, str] = {
    "SimulationExperiment": "tvbo:SimulationExperiment",
    "SimulationStudy": "tvbo:SimulationStudy",
    "Dynamics": "tvbo:Dynamics",
    "NeuralMassModel": "tvbo:Dynamics",
    "StateVariable": "tvbo:StateVariable",
    "DerivedVariable": "tvbo:DerivedVariable",
    "Parameter": "tvbo:Parameter",
    "DerivedParameter": "tvbo:DerivedParameter",
    "Equation": "tvbo:Equation",
    "ConditionalBlock": "tvbo:ConditionalBlock",
    "Function": "tvbo:Function",
    "Network": "tvbo:Network",
    "Node": "tvbo:Node",
    "Edge": "tvbo:Edge",
    "Coupling": "tvbo:Coupling",
    "Integrator": "tvbo:Integrator",
    "Monitor": "tvbo:Monitor",
    "Stimulus": "tvbo:Stimulus",
    "Noise": "tvbo:Noise",
    "TimeSeries": "tvbo:TimeSeries",
    "Range": "tvbo:Range",
    "Parcellation": "tvbo:Parcellation",
    "Tractogram": "tvbo:Tractogram",
    "SoftwareEnvironment": "tvbo:SoftwareEnvironment",
    "SoftwareRequirement": "tvbo:SoftwareRequirement",
    "SoftwarePackage": "tvbo:SoftwarePackage",
}

# Combined mappings (external + tvbo) for runtime conversion
TVBO_TO_OPENMINDS_TYPE: dict[str, str] = {
    **EXTERNAL_TYPE_MAPPINGS,
    **TVBO_TYPE_MAPPINGS,
}

# Map LinkML class names to openMINDS categories
OPENMINDS_CATEGORIES: dict[str, list[str]] = {
    "SimulationStudy": ["researchProduct"],
    "SimulationExperiment": ["computationalActivity"],
    "Dynamics": ["computationalModel"],
    "NeuralMassModel": ["computationalModel"],
    "Network": ["connectome"],
    "TimeSeries": ["simulationOutput"],
}

# Map LinkML class names to openMINDS base types (extension)
OPENMINDS_EXTENDS: dict[str, str] = {
    "SimulationExperiment": "/computation/schemas/simulation.schema.tpl.json",
}

# Classes that should be skipped (internal/helper classes)
SKIP_CLASSES: set[str] = {
    "Matrix",
    "BrainRegionSeries",
    "NDArray",
    "Case",  # Internal helper for conditionals
    "ArgumentMapping",
    "DataInjection",
    "Callable",
    "Sample",
}

# LinkML to JSON type mappings
LINKML_TO_JSON_TYPE: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "int": "integer",
    "float": "number",
    "double": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "date": "string",
    "datetime": "string",
    "uri": "string",
    "uriorcurie": "string",
}

# Fields to skip during serialization (internal/computed)
SKIP_FIELDS = {"_as_dict", "metadata", "experiments"}

# Fields that should use snake_case in openMINDS (matching LinkML)
# openMINDS typically uses camelCase, but we preserve snake_case for TVBO extension


def _get_openminds_type(obj: Any) -> str | None:
    """Get the openMINDS @type for an object."""
    if obj is None:
        return None

    # Check class name
    cls_name = type(obj).__name__
    if cls_name in TVBO_TO_OPENMINDS_TYPE:
        return TVBO_TO_OPENMINDS_TYPE[cls_name]

    # Check for explicit type attribute
    if hasattr(obj, "_type"):
        return obj._type

    return f"tvbo:{cls_name}"


def _to_openminds_value(value: Any, depth: int = 0) -> Any:
    """Recursively convert a value to openMINDS JSON-LD format."""
    if value is None:
        return None

    if depth > 20:
        # Prevent infinite recursion
        return str(value)

    # Primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Lists/tuples
    if isinstance(value, (list, tuple)):
        return [_to_openminds_value(v, depth + 1) for v in value if v is not None]

    # Dicts
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k.startswith("_") or k in SKIP_FIELDS:
                continue
            converted = _to_openminds_value(v, depth + 1)
            if converted is not None:
                result[k] = converted
        return result if result else None

    # Objects with _as_dict (TVBO datamodel objects)
    if hasattr(value, "_as_dict"):
        return _object_to_openminds(value, depth + 1)

    # Objects with model_dump (Pydantic)
    if hasattr(value, "model_dump"):
        return _to_openminds_value(value.model_dump(exclude_none=True), depth + 1)

    # Objects with __dict__
    if hasattr(value, "__dict__"):
        return _object_to_openminds(value, depth + 1)

    # Fallback to string
    return str(value)


def _object_to_openminds(obj: Any, depth: int = 0) -> dict[str, Any] | None:
    """Convert a TVBO object to openMINDS JSON-LD dict."""
    if obj is None:
        return None

    result = {}

    # Add @type
    om_type = _get_openminds_type(obj)
    if om_type:
        result["@type"] = om_type

    # Get dict representation
    if hasattr(obj, "_as_dict"):
        obj_dict = obj._as_dict
    elif hasattr(obj, "model_dump"):
        obj_dict = obj.model_dump(exclude_none=True)
    elif hasattr(obj, "__dict__"):
        obj_dict = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return result

    # Convert each field
    for key, value in obj_dict.items():
        if key.startswith("_") or key in SKIP_FIELDS:
            continue
        if value is None:
            continue

        converted = _to_openminds_value(value, depth + 1)
        if converted is not None:
            result[key] = converted

    return result


def _from_openminds_value(value: Any, target_type: type | None = None) -> Any:
    """Recursively convert an openMINDS JSON-LD value back to Python."""
    if value is None:
        return None

    # Primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Lists
    if isinstance(value, list):
        return [_from_openminds_value(v) for v in value]

    # Dicts (potential objects)
    if isinstance(value, dict):
        # Remove JSON-LD metadata
        cleaned = {k: v for k, v in value.items() if not k.startswith("@") or k == "@id"}

        # If it has @type, try to instantiate the appropriate class
        if "@type" in value:
            value["@type"]
            # For now, just return the cleaned dict
            # Subclasses can handle specific type instantiation
            return {k: _from_openminds_value(v) for k, v in cleaned.items()}

        return {k: _from_openminds_value(v) for k, v in cleaned.items()}

    return value


# =============================================================================
# SimulationExperiment Conversion
# =============================================================================


def experiment_to_openminds(
    experiment: "SimulationExperiment",
    base_id: str | None = None,
    include_context: bool = True,
) -> dict[str, Any]:
    """Convert a SimulationExperiment to openMINDS JSON-LD format.

    Parameters
    ----------
    experiment : SimulationExperiment
        The experiment to convert.
    base_id : str, optional
        Base URI for generating @id values. If not provided, uses a default.
    include_context : bool
        Whether to include the @context in the output.

    Returns
    -------
    dict
        OpenMINDS-compatible JSON-LD dictionary.
    """
    result = _object_to_openminds(experiment)
    if result is None:
        result = {}

    # Ensure correct type
    result["@type"] = "tvbo:SimulationExperiment"

    # Add @id if base_id provided
    if base_id:
        exp_id = getattr(experiment, "id", None) or "unknown"
        result["@id"] = f"{base_id}/experiments/{exp_id}"

    # Add context
    if include_context:
        result = {"@context": OPENMINDS_CONTEXT, **result}

    # Clean up internal fields
    for field in ["_as_dict", "metadata"]:
        result.pop(field, None)

    return result


def experiment_from_openminds(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Convert openMINDS JSON-LD to a dict suitable for SimulationExperiment.

    Parameters
    ----------
    data : dict
        OpenMINDS JSON-LD dictionary.

    Returns
    -------
    dict
        Dictionary that can be passed to SimulationExperiment(**dict).
    """
    # Remove JSON-LD metadata
    result = {}

    for key, value in data.items():
        if key.startswith("@"):
            if key == "@id":
                # Extract ID from URI if present
                if isinstance(value, str) and "/" in value:
                    result["id"] = value.split("/")[-1]
                    try:
                        result["id"] = int(result["id"])
                    except ValueError:
                        pass
            continue

        result[key] = _from_openminds_value(value)

    return result


# =============================================================================
# SimulationStudy Conversion
# =============================================================================


def study_to_openminds(
    study: "SimulationStudy",
    base_id: str | None = None,
    include_context: bool = True,
) -> dict[str, Any]:
    """Convert a SimulationStudy to openMINDS JSON-LD format.

    Parameters
    ----------
    study : SimulationStudy
        The study to convert.
    base_id : str, optional
        Base URI for generating @id values.
    include_context : bool
        Whether to include the @context in the output.

    Returns
    -------
    dict
        OpenMINDS-compatible JSON-LD dictionary.
    """
    result = _object_to_openminds(study)
    if result is None:
        result = {}

    # Ensure correct type
    result["@type"] = "tvbo:SimulationStudy"

    # Add @id
    if base_id:
        study_key = getattr(study, "key", None) or "unknown"
        result["@id"] = f"{base_id}/studies/{study_key}"
    elif getattr(study, "doi", None):
        result["@id"] = f"https://doi.org/{study.doi}"

    # Convert embedded experiments with proper nesting
    experiments = getattr(study, "simulation_experiments", None) or []
    if experiments:
        result["simulation_experiments"] = [
            experiment_to_openminds(exp, base_id=result.get("@id"), include_context=False)
            if hasattr(exp, "_as_dict")
            else _to_openminds_value(exp)
            for exp in experiments
        ]

    # Add context
    if include_context:
        result = {"@context": OPENMINDS_CONTEXT, **result}

    return result


def study_from_openminds(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Convert openMINDS JSON-LD to a dict suitable for SimulationStudy.

    Parameters
    ----------
    data : dict
        OpenMINDS JSON-LD dictionary.

    Returns
    -------
    dict
        Dictionary that can be passed to SimulationStudy(**dict).
    """
    result = {}

    for key, value in data.items():
        if key.startswith("@"):
            if key == "@id":
                # Try to extract DOI or key from URI
                if isinstance(value, str):
                    if "doi.org" in value:
                        result["doi"] = value.replace("https://doi.org/", "")
                    elif "/studies/" in value:
                        result["key"] = value.split("/studies/")[-1]
            continue

        # Special handling for simulation_experiments
        if key == "simulation_experiments" and isinstance(value, list):
            result[key] = [experiment_from_openminds(exp) for exp in value]
        else:
            result[key] = _from_openminds_value(value)

    return result


# =============================================================================
# File I/O Utilities
# =============================================================================


def save_openminds(
    obj: Any,
    filepath: str,
    base_id: str | None = None,
    indent: int = 2,
) -> None:
    """Save a TVBO object as openMINDS JSON-LD file.

    Parameters
    ----------
    obj : SimulationExperiment or SimulationStudy
        The object to save.
    filepath : str
        Output file path (should end in .jsonld or .json).
    base_id : str, optional
        Base URI for @id values.
    indent : int
        JSON indentation level.
    """
    # Determine conversion function based on type
    cls_name = type(obj).__name__

    if cls_name == "SimulationStudy" or "Study" in cls_name:
        data = study_to_openminds(obj, base_id=base_id)
    elif cls_name == "SimulationExperiment" or "Experiment" in cls_name:
        data = experiment_to_openminds(obj, base_id=base_id)
    else:
        data = _object_to_openminds(obj) or {}
        data = {"@context": OPENMINDS_CONTEXT, **data}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def load_openminds(filepath: str) -> dict[str, Any]:
    """Load an openMINDS JSON-LD file and return conversion-ready dict.

    Parameters
    ----------
    filepath : str
        Path to JSON-LD file.

    Returns
    -------
    dict
        Dictionary suitable for constructing TVBO objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine type and use appropriate converter
    om_type = data.get("@type", "")

    if "SimulationStudy" in om_type:
        return study_from_openminds(data)
    elif "SimulationExperiment" in om_type:
        return experiment_from_openminds(data)
    else:
        # Generic conversion
        return _from_openminds_value(data)
