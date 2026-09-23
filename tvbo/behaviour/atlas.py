#
# Module: behaviour/atlas.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What a ``BrainAtlas`` record does: find its own volume and metadata files, and answer for its regions."""

from __future__ import annotations

import logging
import os

import numpy as np

from tvbo.behaviour._runtime import RuntimeAttributes

logger = logging.getLogger(__name__)

WHOLEBRAIN = "wholebrain"
"""The atlas a record names when it names none: one region and no files, so nothing is ever read for it."""


class BrainAtlasBehaviour(RuntimeAttributes):
    """A brain atlas record, with the properties its own data files answer.

    SANDS entities are stored in ``self.terminology.entities`` — a schema-native ``dict[ParcellationEntityName, ParcellationEntity]`` produced by the loader (the ``entities`` slot uses ``inlined: true`` in the SANDS schema).

    Construction reads nothing: a record loaded from the database, or named in passing by a network's ``parcellation``, is a statement about an atlas. `of` builds the atlas itself with its metadata read, and the region properties — `region_labels`, `centers`, `get_label_by_lookup`, `create_terminology` — read it on first use, so a record reached either way answers the same; a record whose files have been read carries the packaged terminology from then on.
    """

    @classmethod
    def of(cls, atlas=None):
        """The atlas *atlas* names, with its metadata loaded.

        Accepts a record of one, its name, or nothing — which means ``wholebrain``, as does a record that names none and anything this cannot read slots off at all. A record's other stated slots come along, and one whose metadata is already loaded is not read again. This is the constructor the runtime uses when it wants the atlas itself rather than a reference to one.
        """
        from tvbo.behaviour._enrich import _slots, _unset

        if atlas is None:
            return cls(name=WHOLEBRAIN)._loaded()
        if isinstance(atlas, str):
            return cls(name=atlas)._loaded()
        stated = {
            slot: value for slot in _slots(type(atlas)) if slot != "name" and not _unset(value := getattr(atlas, slot, None))
        }
        built = cls(name=str(getattr(atlas, "name", None) or WHOLEBRAIN).replace("-", ""), **stated)
        if getattr(atlas, "_metadata_loaded", False):
            object.__setattr__(built, "_metadata_loaded", True)
        return built._loaded()

    def _loaded(self):
        """This atlas, its metadata read."""
        self._load_metadata()
        return self

    @property
    def metadata(self):
        """Return this atlas itself as its LinkML metadata object."""
        return self

    @property
    def _peer(self):
        """The generated module this record's members are built in."""
        from tvbo.datamodel.dialect import peer_module

        return peer_module(self)

    @property
    def _entities(self) -> dict:
        """The SANDS entities the terminology holds, empty when it holds none."""
        return getattr(getattr(self, "terminology", None), "entities", None) or {}

    def _files(self, **query) -> list[str]:
        """This atlas's ``dseg`` files matching *query* in the packaged atlas layout, none for ``wholebrain``, so the layout is never built for the atlas that has no files."""
        if self.name == WHOLEBRAIN:
            return []
        from tvbo.classes.atlas import atlas_data

        return atlas_data.get(atlas=self.name, suffix="dseg", return_type="file", **query)

    def _find_volume_path(self):
        if self.name == WHOLEBRAIN:
            return None
        from tvbo.classes.atlas import available_atlases

        if self.name not in available_atlases:
            raise ValueError(f"Atlas {self.name} is not available in the dataset: {available_atlases}")
        imgs = self._files(extension=".nii.gz")
        if len(imgs) > 1:
            imgs = self._files(extension=".nii.gz", desc="ranked")
        return imgs[0] if len(imgs) == 1 else None

    @property
    def volume(self):
        """NIfTI image of the atlas parcellation volume.

        Returns an empty 256³ placeholder image for the `wholebrain` atlas. Otherwise loads the parcellation volume from disk, or `None` when no volume file is found.

        Raises:
            ImportError: If nibabel is not installed.
        """
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required for neuroimaging data. Install with: pip install nibabel") from exc
        if self.name == WHOLEBRAIN:
            return nib.Nifti1Image(np.zeros((256, 256, 256)), np.eye(4))
        vpath = self._find_volume_path()
        return nib.load(vpath) if vpath else None

    @property
    def volume_file(self):
        """Filesystem path to the atlas parcellation volume, or `None` if not found."""
        return self._find_volume_path()

    @property
    def metadata_file(self):
        """Filesystem path to the atlas `_dseg.yaml` metadata file, or `None` if not uniquely found."""
        files = self._files(extension=".yaml")
        return files[0] if len(files) == 1 else None

    def _load_metadata(self):
        """Fill ``coordinateSpace`` and ``terminology`` from the atlas's own ``_dseg.yaml``, once; members are built in this record's generated form."""
        from tvbo.utils import yaml_loader

        if getattr(self, "_metadata_loaded", False):
            return
        metadata_file = self.metadata_file
        if metadata_file is not None:
            loaded = yaml_loader.load(metadata_file, self._peer.BrainAtlas)
            if getattr(loaded, "coordinateSpace", None) is not None:
                self.coordinateSpace = loaded.coordinateSpace
            if getattr(loaded, "terminology", None) is not None:
                self.terminology = loaded.terminology
            self._fill_centers_from_companion(metadata_file.replace("_dseg.yaml", "_centers.txt"))
        elif getattr(self, "terminology", None) is None:
            self.terminology = self._peer.ParcellationTerminology(label="empty")
        object.__setattr__(self, "_metadata_loaded", True)

    def _fill_centers_from_companion(self, centers_file: str) -> None:
        """Give every entity its centre from the ``_centers.txt`` beside the metadata, when none states one and the file lists exactly one row per entity."""
        entities = list(self._entities.values())
        if any(getattr(e, "center", None) is not None for e in entities) or not os.path.exists(centers_file):
            return
        centers = np.loadtxt(centers_file)
        if len(centers) != len(entities):
            return
        for entity, xyz in zip(entities, centers, strict=True):
            entity.center = self._coordinate(xyz)

    def _fill_centers_from_volume(self, entities: dict) -> None:
        """Locate each entity's centre in the parcellation volume, which needs nilearn; without it the centres stay unset, with a warning."""
        try:
            from nilearn.plotting import find_parcellation_cut_coords
        except ImportError:
            logger.warning(
                "nilearn is required to compute atlas region centers. Setting to empty. Install nilearn or provide atlas metadata."
            )
            return
        centers, lookup_labels = find_parcellation_cut_coords(self.volume, return_label_names=True)
        for center, lookup_label in zip(centers, lookup_labels, strict=True):
            key = str(int(lookup_label))
            if key in entities:
                entities[key].center = self._coordinate(center)

    def _coordinate(self, xyz):
        """A ``Coordinate`` member from an ``(x, y, z)`` triple."""
        return self._peer.Coordinate(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    @staticmethod
    def _center_of(entity) -> tuple[float, float, float]:
        """An entity's centre as ``(x, y, z)``, the origin when it states none."""
        c = getattr(entity, "center", None)
        return (c.x, c.y, c.z) if c is not None else (0.0, 0.0, 0.0)

    @property
    def region_labels(self):
        """Region labels sorted by SANDS lookupLabel.

        Reads from ParcellationTerminology.entities (SANDS schema, inlined). Falls back to unique labels in the atlas volume.
        """
        self._load_metadata()
        pairs = [(e.name, e.lookupLabel) for e in self._entities.values() if e.name is not None and e.lookupLabel is not None]
        if pairs:
            pairs.sort(key=lambda x: x[1])
            return np.asarray([p[0] for p in pairs])
        vol = self.volume
        if vol is None:
            return np.array([])
        return np.unique(vol.get_fdata())[1:]

    def create_terminology(self):
        """Build terminology entities from the atlas volume if not already populated."""
        self._load_metadata()
        if self._entities:
            return self.terminology
        vol = self.volume
        if vol is None:
            return None
        lookup_ids = np.unique(vol.get_fdata())
        lookup_ids = sorted(lookup_ids[lookup_ids != 0])
        if self.terminology is None:
            self.terminology = self._peer.ParcellationTerminology(label="original")
        if not isinstance(self.terminology.entities, dict):
            self.terminology.entities = {}
        for idx in lookup_ids:
            self.terminology.entities[str(int(idx))] = self._peer.ParcellationEntity(name=str(int(idx)), lookupLabel=int(idx))
        return self.terminology

    @property
    def centers(self):
        """Region center coordinates as (N, 3) array.

        Reads SANDS Coordinate from each ParcellationEntity.center. Falls back to computing centers from the atlas volume, or `None` when the volume yields no regions either.
        """
        self._load_metadata()
        if not self._entities:
            self.create_terminology()
            entities = self._entities
            if not entities:
                return None
            self._fill_centers_from_volume(entities)
        return np.array([self._center_of(e) for e in self._entities.values()])

    def get_label_by_lookup(self, lookup_id):
        """Return the region name for a given SANDS lookup label.

        Args:
            lookup_id: The `lookupLabel` value to match against the terminology entities.

        Returns:
            The matching entity name, or `None` if no entity has that lookup label.
        """
        self._load_metadata()
        for e in self._entities.values():
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
