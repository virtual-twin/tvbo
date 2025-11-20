from os.path import basename, dirname, join

import nibabel as nib
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.loaders import yaml_loader

from scipy.ndimage import center_of_mass
from tqdm import tqdm

from tvbo.data.tvbo_data import ATLAS_DIR, bids_utils
from tvbo.datamodel import tvbo_datamodel
from tvbo.knowledge.atlas import freesurfer

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


class Atlas(tvbo_datamodel.BrainAtlas):
    """BrainAtlas with lazy, computed properties for data files and derived attributes.

    Usage mirrors other runtime wrappers (Dynamics, SimulationExperiment):
    - Construct from a BrainAtlas instance, a string name, or nothing (defaults to 'wholebrain').
    - Access `metadata` to get a self-reference as a LinkML object.
    - Access properties: volume, volume_file, metadata_file, region_labels, centers.
    """

    def __init__(self, atlas=None, **kwargs):
        if isinstance(atlas, tvbo_datamodel.BrainAtlas):
            name = atlas.name.replace("-", "") if atlas.name else "wholebrain"
            super().__init__(
                name=name, **{k: v for k, v in atlas.__dict__.items() if k != "name"}
            )
        elif isinstance(atlas, str):
            super().__init__(name=atlas)
        else:
            super().__init__(name="wholebrain")

        # Automatically load metadata (if available) without keeping internal caches
        self._load_metadata()

    # Align to other wrappers
    @property
    def metadata(self):
        return self

    def _find_volume_path(self):
        if self.name == "wholebrain":
            return None
        if self.name not in available_atlases:
            raise ValueError(
                f"Atlas {self.name} is not available in the dataset: {available_atlases}"
            )
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
        if self.name == "wholebrain":
            return nib.Nifti1Image(np.zeros((256, 256, 256)), np.eye(4))
        vpath = self._find_volume_path()
        return nib.load(vpath) if vpath else None

    @property
    def volume_file(self):
        return self._find_volume_path()

    def _load_metadata(self):
        metadata_files = atlas_data.get(
            atlas=self.name,
            suffix="dseg",
            extension=".yaml",
            return_type="file",
        )
        if len(metadata_files) == 1:
            loaded = yaml_loader.load(metadata_files[0], tvbo_datamodel.BrainAtlas)
            # Adopt fields from loaded metadata
            if getattr(loaded, "coordinateSpace", None) is not None:
                self.coordinateSpace = loaded.coordinateSpace
            if getattr(loaded, "terminology", None) is not None:
                self.terminology = loaded.terminology
        else:
            if getattr(self, "terminology", None) is None:
                self.terminology = tvbo_datamodel.ParcellationTerminology(label="empty")

    @property
    def metadata_file(self):
        files = atlas_data.get(
            atlas=self.name,
            suffix="dseg",
            extension=".yaml",
            return_type="file",
        )
        return files[0] if len(files) == 1 else None

    @property
    def region_labels(self):
        # Build from metadata if present; otherwise derive from volume
        self._load_metadata()
        labels = []
        ids = []
        ents = getattr(getattr(self, "terminology", None), "entities", None)
        if ents:
            # dict-like: items(); list-like: iterate
            iterator = ents.items() if hasattr(ents, "items") else enumerate(ents)
            for k, v in iterator:
                # k is label if dict-like, else v.name if object
                label = k if hasattr(ents, "items") else getattr(v, "name", str(k))
                ll = getattr(v, "lookupLabel", None)
                if ll is None and isinstance(v, dict):
                    ll = v.get("lookupLabel")
                if label is not None and ll is not None:
                    labels.append(label)
                    ids.append(ll)
        if labels and ids:
            order = np.argsort(np.asarray(ids))
            return np.asarray(labels)[order]
        # Fallback from the image
        vol = self.volume
        if vol is None:
            return np.array([])
        arr = vol.get_fdata()
        return np.unique(arr)[1:]

    def create_terminology(self):
        self._load_metadata()
        if getattr(self.terminology, "entities", None):
            return self.terminology
        vol = self.volume
        if vol is None:
            return None
        lookup_ids = np.unique(vol.get_fdata())
        lookup_ids = sorted(lookup_ids[lookup_ids != 0])
        if self.terminology is None:
            self.terminology = tvbo_datamodel.ParcellationTerminology(name="original")
        if not getattr(self.terminology, "entities", None):
            # prefer dict for name->entity
            self.terminology.entities = {}
        for idx in lookup_ids:
            self.terminology.entities[str(int(idx))] = (
                tvbo_datamodel.ParcellationEntity(
                    name=str(int(idx)), lookupLabel=int(idx)
                )
            )
        return self.terminology

    @property
    def centers(self):
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None)
        if ents:
            centers_list = []
            iterator = ents.items() if hasattr(ents, "items") else enumerate(ents)
            for _, v in iterator:
                coord = getattr(v, "center", None)
                if coord is None and isinstance(v, dict):
                    coord = v.get("center")
                if coord is None:
                    coord = tvbo_datamodel.Coordinate(x=0, y=0, z=0)
                centers_list.append((coord.x, coord.y, coord.z))
            return np.array(centers_list)
        else:
            terminology = self.create_terminology()
            if not terminology:
                return None
            try:
                from nilearn.plotting import find_parcellation_cut_coords

                centers, lookup_labels = find_parcellation_cut_coords(
                    self.volume, return_label_names=True
                )
            except ImportError:
                centers = []
                lookup_labels = []
                print(
                    "nilearn is required to compute atlas region centers. Setting to empty. Please install nilearn or provide atlas metadata."
                )

            for center, lookup_label in zip(centers, lookup_labels):
                ent = tvbo_datamodel.ParcellationEntity(
                    name=str(int(lookup_label)), lookupLabel=int(lookup_label)
                )
                ent.center = tvbo_datamodel.Coordinate(
                    x=float(center[0]), y=float(center[1]), z=float(center[2])
                )
                # ensure dict structure
                if (
                    not hasattr(self.terminology, "entities")
                    or self.terminology.entities is None
                ):
                    self.terminology.entities = {}
                self.terminology.entities[str(int(lookup_label))] = ent
            # Build array from updated metadata
            centers_list = []
            for e in self.terminology.entities.values():
                c = getattr(e, "center", None)
                if c is None:
                    c = tvbo_datamodel.Coordinate(x=0, y=0, z=0)
                centers_list.append((c.x, c.y, c.z))
            return np.array(centers_list)

    def get_label_by_lookup(self, lookup_id):
        self._load_metadata()
        ents = getattr(getattr(self, "terminology", None), "entities", None)
        if not ents:
            return None
        iterator = ents.items() if hasattr(ents, "items") else enumerate(ents)
        for _, e in iterator:
            lbl = getattr(e, "lookupLabel", None)
            if lbl is None and isinstance(e, dict):
                lbl = e.get("lookupLabel")
            if lbl == lookup_id:
                return (
                    getattr(e, "name", None)
                    if not isinstance(e, dict)
                    else e.get("name")
                )
        return None

    def to_yaml(self, fname=None):
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, fname)


def create_atlas_metadata(fname_atlas, labels="freesurfer"):
    entities = atlas_data.parse_file_entities(fname_atlas)
    atlas_metadata = tvbo_datamodel.BrainAtlas(
        name=entities["atlas"],
        coordinateSpace=tvbo_datamodel.CommonCoordinateSpace(
            abbreviation=entities["space"],
        ),
        terminology=tvbo_datamodel.ParcellationTerminology(
            name=entities.get("desc", "original")
        ),
    )
    atlas_metadata.terminology.entities = []

    parcellation_data = nib.load(fname_atlas).get_fdata()
    # Get unique labels in the parcellation
    labels = np.unique(parcellation_data)

    # Compute center of mass for each label
    centers_of_mass = {}
    for label in labels:
        if label == 0:  # Skip background if it's labeled as 0
            continue
        region = parcellation_data == label
        com = center_of_mass(region)
        centers_of_mass[label] = com

    pass


def rank_atlas(fname_atlas, labels="freesurfer", desc="ranked", gm_only=True):
    entities = atlas_data.parse_file_entities(fname_atlas)
    atlas_metadata = tvbo_datamodel.BrainAtlas(
        name=entities["atlas"],
        coordinateSpace=tvbo_datamodel.CommonCoordinateSpace(
            abbreviation=entities["space"],
        ),
        terminology=tvbo_datamodel.ParcellationTerminology(
            name=entities.get("desc", "original")
        ),
    )
    atlas_metadata.terminology.entities = []
    atlas = nib.load(fname_atlas)
    labels = "freesurfer"

    atlas_ranked = np.zeros(atlas.shape)
    i = 1
    for idx in tqdm(np.unique(atlas.get_fdata())):
        if idx == 0:
            continue
        label = freesurfer.idx2label(idx) if labels == "freesurfer" else labels[idx]
        if labels == "freesurfer" and gm_only:
            if not "ctx" in label and label not in aseg_gm_regions:
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
