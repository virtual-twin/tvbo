"""Runtime helpers for brain atlases and parcellation volumes.

Indexes the packaged atlas directory as a BIDS layout, exports the generated `BrainAtlas` class as `Atlas` — what an atlas does lives in :mod:`tvbo.behaviour.atlas` — and defines helpers to build atlas metadata and to produce ranked (relabelled) parcellation volumes from FreeSurfer segmentations.
"""

try:
    import nibabel as nib
except ImportError:
    nib = None  # nibabel is optional (neuroimaging data support)

import numpy as np
from bids.layout import BIDSLayout
from linkml_runtime.dumpers import yaml_dumper
from scipy.ndimage import center_of_mass

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """No-op ``tqdm`` fallback used when the package is unavailable."""
        return x  # No-op if tqdm not available


from tvbo.adapters import bids as bids_utils
from tvbo.data.tvbo_data import ATLAS_DIR
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.ontology.atlas import freesurfer

atlas_data = BIDSLayout(
    ATLAS_DIR,
    validate=False,
    is_derivative=True,
)

aseg_gm_regions = [
    "left-thalamus",
    "left-caudate",
    "left-putamen",
    "left-pallidum",
    "left-hippocampus",
    "left-amygdala",
    "left-accumbens-area",
    "left-cerebellum-cortex",
    "left-ventraldc",
    "right-thalamus",
    "right-caudate",
    "right-putamen",
    "right-pallidum",
    "right-hippocampus",
    "right-amygdala",
    "right-accumbens-area",
    "right-cerebellum-cortex",
    "right-ventraldc",
    "brain-stem",
]

available_atlases = bids_utils.get_unique_entity_values(atlas_data, "atlas")


Atlas = tvbo_datamodel.BrainAtlas
"""The generated class itself. What an atlas does lives in :mod:`tvbo.behaviour.atlas`; ``Atlas.of(name)`` is the constructor that also reads the atlas's own files, which plain construction deliberately does not."""


def create_atlas_metadata(fname_atlas, labels="freesurfer"):
    """Build `BrainAtlas` metadata and region centers of mass from a parcellation file.

    Parses BIDS entities from the file name to seed a `BrainAtlas` with its coordinate space and terminology, then computes the center of mass of each non-background label in the parcellation volume.

    Args:
        fname_atlas: Path to the parcellation NIfTI file to derive metadata from.
        labels: Reserved for a labelling scheme; currently unused (region labels
            are read directly from the parcellation volume).
    """
    entities = atlas_data.parse_file_entities(fname_atlas)
    atlas_metadata = tvbo_datamodel.BrainAtlas(
        name=entities["atlas"],
        coordinateSpace=tvbo_datamodel.CommonCoordinateSpace(
            abbreviation=entities["space"],
        ),
        terminology=tvbo_datamodel.ParcellationTerminology(name=entities.get("desc", "original")),
    )
    atlas_metadata.terminology.entities = []

    parcellation_data = nib.load(fname_atlas).get_fdata()
    # Get unique labels in the parcellation
    unique_labels = np.unique(parcellation_data)

    # Compute center of mass for each label
    centers_of_mass = {}
    for label in unique_labels:
        if label == 0:  # Skip background if it's labeled as 0
            continue
        region = parcellation_data == label
        com = center_of_mass(region)
        centers_of_mass[label] = com

    pass


def rank_atlas(fname_atlas, labels="freesurfer", desc="ranked", gm_only=True):
    """Relabel a parcellation with contiguous rank IDs and write the ranked volume and metadata.

    Remaps each original label to a consecutive integer (1, 2, 3, ...), records the mapping as `ParcellationEntity` metadata (keeping the original lookup label), then saves the ranked NIfTI volume alongside its `.yaml` metadata using a BIDS-style path with the given `desc`.

    Args:
        fname_atlas: Path to the source parcellation NIfTI file.
        labels: Labelling scheme; `"freesurfer"` maps indices to region names via
            the FreeSurfer lookup, otherwise `labels` is indexed by label id.
        desc: BIDS `desc` entity used when building the output file path.
        gm_only: When using FreeSurfer labels, keep only cortical (`ctx`) and
            grey-matter subcortical regions, skipping all others.

    Returns:
        The ranked parcellation as a NIfTI image.
    """
    entities = atlas_data.parse_file_entities(fname_atlas)
    atlas_metadata = tvbo_datamodel.BrainAtlas(
        name=entities["atlas"],
        coordinateSpace=tvbo_datamodel.CommonCoordinateSpace(
            abbreviation=entities["space"],
        ),
        terminology=tvbo_datamodel.ParcellationTerminology(name=entities.get("desc", "original")),
    )
    atlas_metadata.terminology.entities = []
    atlas = nib.load(fname_atlas)

    atlas_ranked = np.zeros(atlas.shape)
    i = 1
    for idx in tqdm(np.unique(atlas.get_fdata())):
        if idx == 0:
            continue
        label = freesurfer.idx2label(idx) if labels == "freesurfer" else labels[idx]
        if labels == "freesurfer" and gm_only:
            if "ctx" not in label and label not in aseg_gm_regions:
                continue

        atlas_metadata.terminology.entities.append(
            tvbo_datamodel.ParcellationEntity(
                name=label,
                lookupLabel=i,
                originalLookupLabel=idx,
            )
        )

        atlas_ranked = np.where(atlas.get_fdata() == idx, i, atlas_ranked)
        i += 1
    img = nib.Nifti1Image(atlas_ranked.astype(np.uint16), atlas.affine)

    entities["desc"] = desc
    franked = atlas_data.build_path(
        entities,
        path_patterns=["space-{space}_atlas-{atlas}_desc-{desc}_{suffix}{extension}"],
        validate=False,
    )
    nib.save(img, franked)
    yaml_dumper.dump(atlas_metadata, franked.replace(".nii.gz", ".yaml"))
    return img
