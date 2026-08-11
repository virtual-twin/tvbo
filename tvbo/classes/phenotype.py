"""Runtime :class:`Phenotype` class.

Carries a cohort's per-subject phenotype scores (cognitive, clinical, behavioral, demographic, physiological, derived) for multi-subject studies that correlate simulated quantities with empirical measurements.
BIDS-aligned (BIDS ``phenotype/`` directory standard).

Sidecar format mirrors the existing TVBO ``Network`` pattern:

::

    <name>.yaml             (descriptor: subjects, measures list, provenance)
    <name>.h5
    └── measures/
        ├── <measure_1>     (1-D float array, length len(subjects))
        ├── <measure_2>     ...
        └── ...

See ``tools/build_schirner2023_phenotype.py`` for an example writer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import yaml

from tvbo.datamodel import pydantic as tvbo_datamodel


class Phenotype(tvbo_datamodel.Phenotype):
    """Runtime wrapper around the auto-generated ``Phenotype`` schema.

    The schema class carries the YAML-side descriptor (subjects, measures names, provenance, optional Cognitive Atlas IRIs via ``measure_specs``); this subclass adds lazy h5 access via :attr:`values` plus ``from_file`` / ``to_file`` round-tripping.

    Example:
    -------
    .. code-block:: python

        from tvbo.classes.phenotype import Phenotype

        ph = Phenotype.from_file("Schirner2023_HCPYA_phenotype.yaml")
        print(ph.subjects[:5])              # ['100206', '100307', ...]
        pmat_rt = ph.get("PMAT24_A_RTCR")   # ndarray shape (50,)
    """

    _h5_path: str | os.PathLike | None = None
    _yaml_path: str | os.PathLike | None = None
    _values_cache: dict | None = None

    # Construction

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> Phenotype:
        """Load a Phenotype sidecar from a YAML descriptor.

        Resolves ``data_file`` relative to the YAML's directory so the h5 companion can sit next to it. Numeric arrays are NOT loaded eagerly — call :meth:`get` (or read :attr:`values`) to fault one in.
        """
        path = Path(path).resolve()
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        inst = cls(**{k: v for k, v in data.items() if k != "tvbo_class"})
        inst._yaml_path = str(path)

        # Resolve the h5 companion path relative to the YAML
        if inst.data_file:
            h5_path = Path(inst.data_file)
            if not h5_path.is_absolute():
                h5_path = path.parent / h5_path
            inst._h5_path = str(h5_path)
        return inst

    def to_file(
        self,
        path: str | os.PathLike,
        values: Mapping[str, Sequence[float]] | None = None,
        provenance_comment: str | None = None,
    ) -> None:
        """Write the YAML descriptor + h5 companion to ``path``.

        Parameters
        ----------
        path
            Target ``.yaml`` path. The ``.h5`` companion is written next
            to it (named after ``self.data_file`` or the yaml basename).
        values
            ``{measure_name: 1-D array}`` mapping. Must cover every name
            listed in ``self.measures``. Each array's length must equal
            ``len(self.subjects)``.
        provenance_comment
            Optional block of ``#``-prefixed lines prepended to the yaml
            for provenance.
        """
        path = Path(path)
        if values is None:
            values = self._values_cache or {}
        # Validate
        for m in self.measures:
            if m not in values:
                raise ValueError(f"Missing values for measure {m!r}")
            arr = np.asarray(values[m])
            if arr.shape != (len(self.subjects),):
                raise ValueError(f"Measure {m!r}: expected shape ({len(self.subjects)},), got {arr.shape}")

        # h5 path
        h5_name = self.data_file or (path.stem + ".h5")
        h5_path = path.parent / h5_name
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            mg = f.create_group("measures")
            for m in self.measures:
                mg.create_dataset(m, data=np.asarray(values[m]))

        # yaml descriptor (round-trip via the pydantic model_dump)
        meta = self.model_dump(exclude_none=True)
        for k in ("_h5_path", "_yaml_path", "_values_cache"):
            meta.pop(k, None)
        meta["tvbo_class"] = "tvbo:Phenotype"
        meta["data_file"] = h5_name
        with open(path, "w") as f:
            if provenance_comment:
                f.write(provenance_comment)
            yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)
        self._h5_path = str(h5_path)
        self._yaml_path = str(path)

    # Value access

    @property
    def values(self) -> dict:
        """Dict of ``{measure_name: ndarray}``, loaded lazily."""
        if self._values_cache is not None:
            return self._values_cache
        if not self._h5_path:
            raise RuntimeError("Phenotype has no h5 companion path. Did you use from_file(), or set data_file?")
        cache: dict = {}
        with h5py.File(self._h5_path, "r") as f:
            for m in self.measures:
                if m in f["measures"]:
                    cache[m] = np.array(f[f"measures/{m}"])
        self._values_cache = cache
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

    def measure_spec(self, name: str) -> tvbo_datamodel.MeasureSpec | None:
        """Return the optional MeasureSpec for ``name`` (or None)."""
        specs = self.measure_specs or []
        for s in specs:
            if s.name == name:
                return s
        return None

    def __repr__(self) -> str:
        n_sub = len(self.subjects) if self.subjects else 0
        n_mea = len(self.measures) if self.measures else 0
        return f"Phenotype(dataset_id={self.dataset_id!r}, category={self.category!r}, n_subjects={n_sub}, n_measures={n_mea})"
