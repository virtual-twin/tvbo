#
# Module: behaviour/phenotype.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What a ``Phenotype`` record does: read and write its h5 companion, and answer for one measure or one subject."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tvbo.behaviour._runtime import RuntimeAttributes


class PhenotypeBehaviour(RuntimeAttributes):
    """A cohort's per-subject phenotype scores, on both generated forms.

    The record is the YAML-side descriptor — subjects, measure names, provenance, optional Cognitive Atlas IRIs via ``measure_specs`` — and the values live in an h5 companion beside it, ``measures/<measure>`` as a 1-D array of ``len(subjects)``, read on first use through :attr:`values`. BIDS-aligned with the ``phenotype/`` directory standard.

    Example:
        >>> ph = Phenotype.from_file("Schirner2023_HCPYA_phenotype.yaml")
        >>> ph.subjects[:5]                 # ['100206', '100307', ...]
        >>> ph.get("PMAT24_A_RTCR")         # ndarray shape (50,)
    """

    _h5_path = None
    _yaml_path = None
    _values_cache = None

    @classmethod
    def from_file(cls, path: str | os.PathLike):
        """Load a Phenotype sidecar from a YAML descriptor.

        Resolves ``data_file`` relative to the YAML's directory so the h5 companion can sit next to it. Numeric arrays are NOT loaded eagerly — call :meth:`get` (or read :attr:`values`) to fault one in.
        """
        from tvbo.utils import yaml_loader

        path = Path(path).resolve()
        inst = yaml_loader.load(str(path), target_class=cls)
        object.__setattr__(inst, "_yaml_path", str(path))
        if inst.data_file:
            h5_path = Path(inst.data_file)
            if not h5_path.is_absolute():
                h5_path = path.parent / h5_path
            object.__setattr__(inst, "_h5_path", str(h5_path))
        return inst

    def to_file(
        self,
        path: str | os.PathLike,
        values: Mapping[str, Sequence[float]] | None = None,
        provenance_comment: str | None = None,
    ) -> None:
        """Write the YAML descriptor + h5 companion to ``path``.

        Args:
            path: Target ``.yaml`` path. The ``.h5`` companion is written next to it, named after ``self.data_file`` or the yaml basename.
            values: ``{measure_name: 1-D array}`` mapping covering every name in ``self.measures``, each array of length ``len(self.subjects)``; the values already read when omitted.
            provenance_comment: Optional block of ``#``-prefixed lines prepended to the yaml for provenance.
        """
        import h5py
        import yaml

        from tvbo.utils import to_dict

        path = Path(path)
        if values is None:
            values = self._values_cache or {}
        for m in self.measures:
            if m not in values:
                raise ValueError(f"Missing values for measure {m!r}")
            arr = np.asarray(values[m])
            if arr.shape != (len(self.subjects),):
                raise ValueError(f"Measure {m!r}: expected shape ({len(self.subjects)},), got {arr.shape}")

        h5_name = self.data_file or (path.stem + ".h5")
        h5_path = path.parent / h5_name
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            mg = f.create_group("measures")
            for m in self.measures:
                mg.create_dataset(m, data=np.asarray(values[m]))

        meta = {"tvbo_class": "tvbo:Phenotype", **to_dict(self)}
        meta["data_file"] = h5_name
        with open(path, "w") as f:
            if provenance_comment:
                f.write(provenance_comment)
            yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)
        object.__setattr__(self, "_h5_path", str(h5_path))
        object.__setattr__(self, "_yaml_path", str(path))

    @property
    def values(self) -> dict:
        """Dict of ``{measure_name: ndarray}``, loaded lazily."""
        import h5py

        if self._values_cache is not None:
            return self._values_cache
        if not self._h5_path:
            raise RuntimeError("Phenotype has no h5 companion path. Did you use from_file(), or set data_file?")
        cache: dict = {}
        with h5py.File(self._h5_path, "r") as f:
            for m in self.measures:
                if m in f["measures"]:
                    cache[m] = np.array(f[f"measures/{m}"])
        object.__setattr__(self, "_values_cache", cache)
        return cache

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
