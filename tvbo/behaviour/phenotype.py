#
# Module: behaviour/phenotype.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2
#
"""What a ``Phenotype`` record does: read and write its companion file, h5 or BIDS ``.tsv``, and answer for one measure or one subject."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tvbo.behaviour._runtime import RuntimeAttributes, YamlDocument


class PhenotypeBehaviour(YamlDocument, RuntimeAttributes):
    """A cohort's per-subject phenotype scores, on both generated forms.

    The record is the YAML-side descriptor — subjects, measure names, provenance, what each measure quantifies via ``measure_specs`` — and the values live in a companion beside it, read on first use through :attr:`values`. The extension of ``data_file`` selects the companion: an ``.h5`` holds ``measures/<measure>`` as a 1-D array of ``len(subjects)``; a ``.tsv`` is a BIDS phenotype table, ``participant_id`` then one column per measure, written together with the ``.json`` column sidecar that ``measure_specs`` describes.

    Example:
        >>> ph = Phenotype.from_file("Schirner2023_HCPYA_phenotype.yaml")
        >>> ph.subjects[:5]                 # ['100206', '100307', ...]
        >>> ph.get("PMAT24_A_RTCR")         # ndarray shape (50,)
    """

    _yaml_path = None
    _values_cache = None

    @classmethod
    def from_file(cls, path: str | os.PathLike):
        """Load a Phenotype sidecar from a YAML descriptor.

        Resolves ``data_file`` relative to the YAML's directory so the companion can sit next to it. Numeric arrays are NOT loaded eagerly — call :meth:`get` (or read :attr:`values`) to fault one in.
        """
        from tvbo.utils import yaml_loader

        path = Path(path).resolve()
        inst = yaml_loader.load(str(path), target_class=cls)
        object.__setattr__(inst, "_yaml_path", str(path))
        return inst

    def to_file(
        self,
        path: str | os.PathLike,
        values: Mapping[str, Sequence[float]] | None = None,
        provenance_comment: str | None = None,
    ) -> None:
        """Write the YAML descriptor and its companion to ``path``.

        Args:
            path: Target ``.yaml`` path. The companion is written next to it, named by ``self.data_file``, whose extension selects the format.
            values: ``{measure_name: 1-D array}`` mapping covering every name in ``self.measures``, each array of length ``len(self.subjects)``; the values already read when omitted.
            provenance_comment: Optional block of ``#``-prefixed lines prepended to the yaml for provenance.
        """
        import yaml

        from tvbo.data.phenotype_io import write_companion
        from tvbo.utils import to_dict

        path = Path(path)
        if values is None:
            values = self._values_cache or {}
        arrays = {}
        for m in self.measures:
            if m not in values:
                raise ValueError(f"Missing values for measure {m!r}")
            arrays[m] = np.asarray(values[m])
            if arrays[m].shape != (len(self.subjects),):
                raise ValueError(f"Measure {m!r}: expected shape ({len(self.subjects)},), got {arrays[m].shape}")
        orphans = [s.name for s in self.measure_specs or [] if s.name not in self.measures]
        if orphans:
            raise ValueError(f"measure_specs {orphans} name no measure of {list(self.measures)}")

        object.__setattr__(self, "_yaml_path", str(path))
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        write_companion(self._data_path, self, arrays)

        meta = {"tvbo_class": "tvbo:Phenotype", **to_dict(self)}
        with open(path, "w") as f:
            if provenance_comment:
                f.write(provenance_comment)
            yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)

    @property
    def _data_path(self) -> Path:
        """The companion, ``data_file`` resolved against the directory of the YAML this record was read from or written to."""
        if not self._yaml_path:
            raise RuntimeError("Phenotype has no companion path. Did you use from_file() or to_file()?")
        return Path(self._yaml_path).parent / self.data_file

    @property
    def values(self) -> dict:
        """Dict of ``{measure_name: ndarray}``, loaded lazily."""
        from tvbo.data.phenotype_io import read_companion

        if self._values_cache is None:
            object.__setattr__(self, "_values_cache", read_companion(self._data_path, self))
        return self._values_cache

    def get(self, measure: str) -> np.ndarray:
        """Return one measure's array. Raises ``KeyError`` if missing."""
        vals = self.values
        if measure not in vals:
            raise KeyError(f"Measure {measure!r} not in this sidecar. Available: {list(vals)}")
        return vals[measure]

    def subject_index(self, subject_id: str) -> int:
        """Return the row index of ``subject_id`` in every measure array."""
        return self.subjects.index(subject_id)

    def measure_spec(self, name: str):
        """Return the optional MeasureSpec for ``name`` (or None)."""
        for s in self.measure_specs or []:
            if s.name == name:
                return s
        return None

    def __repr__(self) -> str:
        n_sub = len(self.subjects) if self.subjects else 0
        n_mea = len(self.measures) if self.measures else 0
        return f"Phenotype(dataset_id={self.dataset_id!r}, category={self.category!r}, n_subjects={n_sub}, n_measures={n_mea})"
