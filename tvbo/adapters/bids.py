"""BIDS BEP034 Export Module.

This module provides utilities for exporting TVB simulation data to BIDS format following the BEP034 Computational Modeling Extension v1.0.0.

Uses:
- Pydantic models from tvbo.datamodel.tvbopydantic for metadata serialization
- pybids for BIDS-compliant filename generation
- nibabel for CIFTI-2 ptseries files
"""

from __future__ import annotations

from bids.layout.writing import build_path
from pydantic import BaseModel, ConfigDict, Field

# `suffix`/`extension` are value-constrained so an invalid combination fails fast; `model`, `desc` and `split` are optional. Two patterns because a result comes either from a run (`exp-`) or from a declared analysis (`ana-`), and one universal `_result` suffix names the kind of data for both — a BIDS suffix never encodes the shape of the array inside. Two extensions, because a container is `.h5` and its one metadata sidecar is the frozen `.yaml` spec that produced it.
RESULT_PATTERNS = [
    "[sub-{subject}_]exp-{experiment}[_model-{model}][_desc-{description}][_split-{split}]_{suffix<result>}{extension<.h5|.yaml>}",
    "ana-{analysis}[_desc-{description}]_{suffix<result>}{extension<.h5|.yaml>}",
]

RESULT_ENTITIES = {
    "sub": "The subject a per-subject shard ran, when a dataset fans out over a cohort. Absent for a single-network run.",
    "exp": "The experiment id the run came from — the `name` of one entry under the study's `experiments:`.",
    "ana": "The name of a declared analysis, in place of `exp-`. A file carries one or the other, never both.",
    "model": "The dynamics the run integrated, so the one fact a reader most wants to filter on is queryable rather than buried in `desc-`.",
    "desc": "BIDS's free-text discriminator, for two results of the same experiment that differ in nothing a named entity captures.",
    "split": "The array-task index of one shard of a sweep, zero-padded. Present only until the shards are gathered.",
}
"""What each entity in :data:`RESULT_PATTERNS` identifies.

Documented beside the patterns they parameterise, so the reference the BIDS page renders and the grammar a filename is built from cannot disagree.
"""


def entity_value(value) -> str:
    """``value`` as a legal BIDS entity value: the alphanumeric characters, in order.

    BIDS requires an entity value to be alphanumeric, because a hyphen or underscore inside one moves the key/value boundary and makes the file unqueryable. Every name tvbo puts in a filename goes through here, so a name the writer accepts is a name ``tvbo validate study`` accepts.
    """
    return "".join(c for c in str(value) if c.isalnum())


def result_entities(experiment, extension: str = ".h5") -> dict:
    """BIDS entities for an experiment's result, with alphanumeric values.

    Every value passes through :func:`entity_value`. The model goes in its own ``model-`` entity rather than being packed into ``desc-``: ``desc-`` is BIDS's free-text discriminator, and spending it on the model made the one fact a reader most wants to filter on unqueryable. Returns a dict ready for :func:`bids.layout.writing.build_path` with :data:`RESULT_PATTERNS`.
    """
    _alnum = entity_value

    entities = {"suffix": "result", "extension": extension}
    # Per-subject shards (dataset fan-out) get a sub- entity so their results do not collide when reassembled.
    active_subject = getattr(experiment, "_active_subject", None)
    if active_subject:
        entities["subject"] = _alnum(active_subject)
    # One array task's slice of a sweep. In the NAME, so a kit's shards sit flat in one directory and the gather globs them without walking a per-job tree.
    active_split = getattr(experiment, "_active_split", None)
    if active_split is not None:
        entities["split"] = f"{int(active_split):04d}"
    eid = getattr(experiment, "id", None)
    if eid is not None:
        entities["experiment"] = _alnum(eid)
    dyn = getattr(experiment, "dynamics", None)
    model = (getattr(dyn, "name", None) or getattr(dyn, "label", None)) if dyn else None
    if model and _alnum(model):
        entities["model"] = _alnum(model)[:24]  # keep the entity compact
    return entities


SPEC_SUFFIXES = {
    "dynamics": "Dynamics",
    "network": "Network",
    "experiment": "SimulationExperiment",
    "analysis": "Analysis",
    "figure": "Figure",
    "study": "SimulationStudy",
}
"""BIDS suffix to the tvbo class a spec fragment of that suffix declares.

A BIDS suffix is precisely the field that names what kind of thing a file is, and a tvbo file already declares that in its ``tvbo_class`` envelope. Making the two the same turns the filename into a checkable restatement of the envelope, which is what ``tvbo validate study`` checks, and gives BEP034+ a suffix vocabulary to propose in place of ``eq``/``param``.

``_result`` is deliberately absent. A result container is data, named by :data:`RESULT_PATTERNS`, and its YAML sidecar takes the container's name per BIDS rather than its own class's — so the one file whose suffix and envelope legitimately disagree is not held to a rule it cannot keep.
"""


def analysis_entities(name, extension: str = ".h5") -> dict:
    """BIDS entities for a declared analysis's own result container.

    The analysis's name becomes the ``ana-`` entity, through :func:`entity_value` like every other entity value, so its container sits beside the experiment results it was derived from and a listing groups the two by prefix.
    """
    value = entity_value(name)
    if not value:
        raise ValueError(f"analysis name {str(name)!r} has no alphanumeric characters, so it cannot name a container.")
    return {"suffix": "result", "extension": extension, "analysis": value}


def build_result_path(experiment=None, *, entities: dict = None, extension: str = ".h5") -> str:
    """Filename for an experiment result via pybids ``build_path`` + RESULT_PATTERNS."""
    return build_path(entities or result_entities(experiment, extension=extension), RESULT_PATTERNS)


# Pydantic Models for BEP034 Metadata (Sidecars)


class BidsBaseModel(BaseModel):
    """Base model for the BIDS metadata documents tvbo writes."""

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


def _record_bids_version() -> str:
    """The BIDS version the layout record was written against, imported late to avoid a cycle."""
    from tvbo.utils.study_layout import load_layout

    return str(load_layout().bids_version)


class DatasetDescription(BidsBaseModel):
    """BIDS dataset_description.json model.

    ``BIDSVersion`` comes from the layout record rather than a literal, so a dataset a run writes cannot claim a different version from the one the record scaffolds a study against.
    """

    Name: str = Field(..., description="Name of the dataset")
    BIDSVersion: str = Field(default_factory=_record_bids_version, description="BIDS specification version")
    DatasetType: str = Field(default="derivative", description="Dataset type")
    GeneratedBy: list[dict] = Field(default_factory=list, description="Tools that generated this dataset")


# Utility helpers (merged from tvbo.data.bids_utils)


def get_unique_entity_values(bids_layout, key) -> set:
    """Get a set of all unique values for a given entity key from the BIDSLayout files.

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
