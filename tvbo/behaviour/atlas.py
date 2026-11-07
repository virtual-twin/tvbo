#
# Module: behaviour/atlas.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2
#
"""What a ``BrainAtlas`` record does: find its own volume and metadata files, and answer for its regions."""

from __future__ import annotations

import logging
import os

import numpy as np

from tvbo.behaviour._runtime import Catalogued, _dm
from tvbo.data.array_formats import read_array

logger = logging.getLogger(__name__)

WHOLEBRAIN = "wholebrain"
"""The atlas a record names when it names none: one region and no files, so nothing is ever read for it."""


class BrainAtlasBehaviour(Catalogued):
    """A brain atlas record, with the properties its own data files answer.

    SANDS entities are stored in ``self.terminology.entities`` — a schema-native ``dict[ParcellationEntityName, ParcellationEntity]`` produced by the loader (the ``entities`` slot uses ``inlined: true`` in the SANDS schema).

    Construction reads nothing: a record loaded from the database, or named in passing by a network's ``parcellation``, is a statement about an atlas. `of` builds the atlas itself with its metadata read, and the region properties — `region_labels`, `centers`, `get_label_by_lookup`, `create_terminology` — read it on first use, so a record reached either way answers the same; a record whose files have been read carries the packaged terminology from then on.
    """

    @classmethod
    def of(cls, atlas=None, **entities):
        """The atlas *atlas* names, with its metadata loaded.

        Accepts a record of one, its name, or nothing — which means ``wholebrain``, as does a record that names none and anything this cannot read slots off at all. A record's other stated slots come along, and one whose metadata is already loaded is not read again. A name is matched in any case, and one several atlases share, ``Schaefer2018``, is refused naming them and the entities that tell them apart unless the BIDS *entities* (``seg="7Networks", scale="100"``) narrow it to one (:func:`tvbo.data.registry.resolve`). This is the constructor the runtime uses when it wants the atlas itself rather than a reference to one.
        """
        from tvbo.behaviour._enrich import _slots, _unset

        if atlas is None:
            return cls(name=WHOLEBRAIN)._loaded(entities)
        if isinstance(atlas, str):
            return cls(name=atlas)._loaded(entities)
        stated = {
            slot: value for slot in _slots(type(atlas)) if slot != "name" and not _unset(value := getattr(atlas, slot, None))
        }
        built = cls(name=str(getattr(atlas, "name", None) or WHOLEBRAIN).replace("-", ""), **stated)
        if getattr(atlas, "_metadata_loaded", False):
            object.__setattr__(built, "_metadata_loaded", True)
        return built._loaded(entities)

    def _loaded(self, entities: dict):
        """This atlas, the record its name and the BIDS *entities* find (:attr:`_record`) its own, and its metadata read."""
        object.__setattr__(self, "_entities_query", entities)
        self._load_metadata()
        return self

    @property
    def metadata(self):
        """Return this atlas itself as its LinkML metadata object."""
        return self

    @property
    def _entities(self) -> dict:
        """The SANDS entities the terminology holds, empty when it holds none."""
        return getattr(getattr(self, "terminology", None), "entities", None) or {}

    @property
    def _record(self):
        """The curated record of this atlas, by its name and the BIDS entities :meth:`of` was given (:func:`tvbo.data.registry.resolve`), ``None`` for ``wholebrain``; its image and centres table share its stem."""
        if self.name == WHOLEBRAIN:
            return None
        from tvbo.data.registry import resolve

        return resolve("BrainAtlas", str(self.name), **getattr(self, "_entities_query", {}))

    def _find_volume_path(self):
        """The parcellation image beside this atlas's record, or ``None`` when it ships none; raises ``FileNotFoundError`` for an atlas the database does not hold."""
        record = self._record
        image = record.with_name(f"{record.stem}.nii.gz") if record is not None else None
        return str(image) if image is not None and image.exists() else None

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
        """Filesystem path to the atlas `_dseg.yaml` metadata file, or `None` for an atlas the database does not hold."""
        try:
            record = self._record
        except FileNotFoundError:
            return None
        return str(record) if record is not None else None

    def _load_metadata(self):
        """Fill ``coordinateSpace`` and ``terminology`` from the atlas's own ``_dseg.yaml``, once; members are built in this record's generated form."""
        from tvbo.utils import yaml_loader

        if getattr(self, "_metadata_loaded", False):
            return
        metadata_file = self.metadata_file
        if metadata_file is not None:
            loaded = yaml_loader.load(metadata_file, _dm().BrainAtlas)
            if getattr(loaded, "coordinateSpace", None) is not None:
                self.coordinateSpace = loaded.coordinateSpace
            if getattr(loaded, "terminology", None) is not None:
                self.terminology = loaded.terminology
            self._fill_centers_from_companion(metadata_file.replace("_dseg.yaml", "_centers.txt"))
        elif getattr(self, "terminology", None) is None:
            self.terminology = _dm().ParcellationTerminology(label="empty")
        object.__setattr__(self, "_metadata_loaded", True)

    def _fill_centers_from_companion(self, centers_file: str) -> None:
        """Give every entity its centre from the ``_centers.txt`` beside the metadata, when none states one and the file lists exactly one row per entity."""
        entities = list(self._entities.values())
        if any(getattr(e, "center", None) is not None for e in entities) or not os.path.exists(centers_file):
            return
        centers = read_array(centers_file, dims=("entity", "axis")).values
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
        return _dm().Coordinate(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

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
            self.terminology = _dm().ParcellationTerminology(label="original")
        if not isinstance(self.terminology.entities, dict):
            self.terminology.entities = {}
        for idx in lookup_ids:
            self.terminology.entities[str(int(idx))] = _dm().ParcellationEntity(name=str(int(idx)), lookupLabel=int(idx))
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
