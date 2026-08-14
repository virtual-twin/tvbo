"""Runtime helpers for brain atlases and parcellation volumes.

Provides the `Atlas` wrapper around the LinkML `BrainAtlas` datamodel, exposing lazy, computed access to the parcellation volume, SANDS terminology, region
labels, and region centers. Also defines helpers to build atlas metadata and to produce ranked (relabelled) parcellation volumes from FreeSurfer segmentations.
"""

import logging
import os

try:
    import nibabel as nib
except ImportError:
    nib = None  # nibabel is optional (neuroimaging data support)

import numpy as np
from bids.layout import BIDSLayout
from linkml_runtime.dumpers import yaml_dumper
from tvbo.utils import yaml_loader

from scipy.ndimage import center_of_mass

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """No-op ``tqdm`` fallback used when the package is unavailable."""
        return x  # No-op if tqdm not available


from tvbo.data.tvbo_data import ATLAS_DIR
from tvbo.adapters import bids as bids_utils
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
logger = logging.getLogger(__name__)


class Atlas(tvbo_datamodel.BrainAtlas):
    """BrainAtlas with lazy, computed properties for data files and derived attributes.

    Usage mirrors other runtime wrappers (Dynamics, SimulationExperiment):
    - Construct from a BrainAtlas instance, a string name, or nothing (defaults to 'wholebrain').
    - Access `metadata` to get a self-reference as a LinkML object.
    - Access properties: volume, volume_file, metadata_file, region_labels, centers.

    SANDS entities are stored in ``self.terminology.entities`` — a schema-native
    ``dict[ParcellationEntityName, ParcellationEntity]`` produced by the LinkML loader (the ``entities`` slot uses ``inlined: true`` in the SANDS schema).
    """

    def __init__(self, atlas=None, **kwargs):
        if isinstance(atlas, tvbo_datamodel.BrainAtlas):
            name = atlas.name.replace("-", "") if atlas.name else "wholebrain"
            super().__init__(name=name, **{k: v for k, v in atlas.__dict__.items() if k != "name"})
        elif isinstance(atlas, str):
            super().__init__(name=atlas)
        else:
            super().__init__(name="wholebrain")

        # Automatically load metadata (if available)
        self._load_metadata()

    # Align to other wrappers
    @property
    def metadata(self):
        """Return this atlas itself as its LinkML metadata object."""
        return self

    def _find_volume_path(self):
        if self.name == "wholebrain":
            return None
        if self.name not in available_atlases:
            raise ValueError(f"Atlas {self.name} is not available in the dataset: {available_atlases}")
        imgs = atlas_data.get(
            atlas=self.name,
            suffix="dseg",
            extension=".nii.gz",
            return_type="file",
        )
        if len(imgs) > 1:
            imgs = atlas_data.get(
                atlas=self.name,
                suffix="dseg",
                desc="ranked",
                extension=".nii.gz",
                return_type="file",
            )
        return imgs[0] if len(imgs) == 1 else None

    @property
    def volume(self):
        """NIfTI image of the atlas parcellation volume.

        Returns an empty 256³ placeholder image for the `wholebrain` atlas.
        Otherwise loads the parcellation volume from disk, or `None` when no volume file is found.

        Raises:
            ImportError: If nibabel is not installed.
        """
        if nib is None:
            raise ImportError("nibabel is required for neuroimaging data. Install with: pip install nibabel")
        if self.name == "wholebrain":
            return nib.Nifti1Image(np.zeros((256, 256, 256)), np.eye(4))
        vpath = self._find_volume_path()
        return nib.load(vpath) if vpath else None

    @property
    def volume_file(self):
        """Filesystem path to the atlas parcellation volume, or `None` if not found."""
        return self._find_volume_path()

    def _load_metadata(self):
        if getattr(self, "_metadata_loaded", False):
            return
        metadata_files = atlas_data.get(
            atlas=self.name,
            suffix="dseg",
            extension=".yaml",
            return_type="file",
        )
        if len(metadata_files) == 1:
            # Load via LinkML — entities slot is inlined, so the loader produces a proper dict[ParcellationEntityName, ParcellationEntity].
            loaded = yaml_loader.load(metadata_files[0], tvbo_datamodel.BrainAtlas)
            if getattr(loaded, "coordinateSpace", None) is not None:
                self.coordinateSpace = loaded.coordinateSpace
            if getattr(loaded, "terminology", None) is not None:
                self.terminology = loaded.terminology

            # Fill in centers from companion centers.txt if missing
            ents = getattr(self.terminology, "entities", None) or {}
            has_any_center = any(getattr(e, "center", None) is not None for e in ents.values())
            if not has_any_center:
                centers_file = metadata_files[0].replace("_dseg.yaml", "_centers.txt")
                if os.path.exists(centers_file):
                    centers = np.loadtxt(centers_file)
                    ent_list = list(ents.values())
                    if len(centers) == len(ent_list):
                        for ent, xyz in zip(ent_list, centers):
                            ent.center = tvbo_datamodel.Coordinate(
                                x=float(xyz[0]),
                                y=float(xyz[1]),
                                z=float(xyz[2]),
                            )
        else:
            if getattr(self, "terminology", None) is None:
                self.terminology = tvbo_datamodel.ParcellationTerminology(label="empty")
        object.__setattr__(self, "_metadata_loaded", True)

    @property
    def metadata_file(self):
        """Filesystem path to the atlas `_dseg.yaml` metadata file, or `None` if not uniquely found."""
        files = atlas_data.get(
            atlas=self.name,
            suffix="dseg",
            extension=".yaml",
            return_type="file",
        )
        return files[0] if len(files) == 1 else None

    @property
    def region_labels(self):
        """Region labels sorted by SANDS lookupLabel.

        Reads from ParcellationTerminology.entities (SANDS schema, inlined).
        Falls back to unique labels in the atlas volume.
        """
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None) or {}
        if ents:
            pairs = [(e.name, e.lookupLabel) for e in ents.values() if e.name is not None and e.lookupLabel is not None]
            if pairs:
                pairs.sort(key=lambda x: x[1])
                return np.asarray([p[0] for p in pairs])
        # Fallback from the image
        vol = self.volume
        if vol is None:
            return np.array([])
        arr = vol.get_fdata()
        return np.unique(arr)[1:]

    def create_terminology(self):
        """Build terminology entities from the atlas volume if not already populated."""
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None) or {}
        if ents:
            return self.terminology
        vol = self.volume
        if vol is None:
            return None
        lookup_ids = np.unique(vol.get_fdata())
        lookup_ids = sorted(lookup_ids[lookup_ids != 0])
        if self.terminology is None:
            self.terminology = tvbo_datamodel.ParcellationTerminology(label="original")
        if not isinstance(self.terminology.entities, dict):
            self.terminology.entities = {}
        for idx in lookup_ids:
            self.terminology.entities[str(int(idx))] = tvbo_datamodel.ParcellationEntity(
                name=str(int(idx)), lookupLabel=int(idx)
            )
        return self.terminology

    @property
    def centers(self):
        """Region center coordinates as (N, 3) array.

        Reads SANDS Coordinate from each ParcellationEntity.center.
        Falls back to computing centers from the atlas volume.
        """
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None) or {}
        if ents:
            centers_list = []
            for e in ents.values():
                c = getattr(e, "center", None) or tvbo_datamodel.Coordinate(x=0, y=0, z=0)
                centers_list.append((c.x, c.y, c.z))
            return np.array(centers_list)
        else:
            self.create_terminology()
            ents = getattr(self.terminology, "entities", None) or {}
            if not ents:
                return None
            try:
                from nilearn.plotting import find_parcellation_cut_coords

                centers, lookup_labels = find_parcellation_cut_coords(self.volume, return_label_names=True)
            except ImportError:
                centers = []
                lookup_labels = []
                logger.warning(
                    "nilearn is required to compute atlas region centers. "
                    "Setting to empty. Install nilearn or provide atlas metadata."
                )

            for center, lookup_label in zip(centers, lookup_labels):
                key = str(int(lookup_label))
                if key in ents:
                    ents[key].center = tvbo_datamodel.Coordinate(
                        x=float(center[0]),
                        y=float(center[1]),
                        z=float(center[2]),
                    )
            centers_list = []
            for e in ents.values():
                c = getattr(e, "center", None) or tvbo_datamodel.Coordinate(x=0, y=0, z=0)
                centers_list.append((c.x, c.y, c.z))
            return np.array(centers_list)

    def get_label_by_lookup(self, lookup_id):
        """Return the region name for a given SANDS lookup label.

        Args:
            lookup_id: The `lookupLabel` value to match against the terminology entities.

        Returns:
            The matching entity name, or `None` if no entity has that lookup label.
        """
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None) or {}
        for e in ents.values():
            if e.lookupLabel == lookup_id:
                return e.name
        return None

    def to_yaml(self, fname=None):
        """Serialise the atlas to YAML.

        Args:
            fname: Optional path to write the YAML to; if omitted, the YAML is returned as a string.

        Returns:
            The written file path when `fname` is given, otherwise the YAML string.
        """
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, fname)


def create_atlas_metadata(fname_atlas, labels="freesurfer"):
    """Build `BrainAtlas` metadata and region centers of mass from a parcellation file.

    Parses BIDS entities from the file name to seed a `BrainAtlas` with its coordinate space and terminology, then computes the center of mass of each
    non-background label in the parcellation volume.

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

    Remaps each original label to a consecutive integer (1, 2, 3, ...), records the mapping as `ParcellationEntity` metadata (keeping the original lookup
    label), then saves the ranked NIfTI volume alongside its `.yaml` metadata using a BIDS-style path with the given `desc`.

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
