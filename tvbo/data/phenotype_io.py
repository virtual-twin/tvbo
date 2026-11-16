"""Read and write a ``Phenotype``'s companion file, in the format the extension of its ``data_file`` names.

An ``.h5`` holds ``measures/<name>`` as a 1-D array of ``len(subjects)``. A ``.tsv`` is a BIDS phenotype table, ``participant_id`` as ``sub-<subject_id>`` then one column per measure with ``n/a`` for a missing value, written together with its ``.json`` column sidecar, whose entries are the record's ``measure_specs``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_companion(path: Path, phenotype, values: Mapping[str, np.ndarray]) -> None:
    """Write *values*, one array per measure of *phenotype*, to the companion at *path*."""
    _formats(path)[0](path, phenotype, values)


def read_companion(path: Path, phenotype) -> dict[str, np.ndarray]:
    """The arrays the companion at *path* holds for the measures *phenotype* names, keyed by measure."""
    return _formats(path)[1](path, phenotype)


def _write_h5(path: Path, phenotype, values: Mapping) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        mg = f.create_group("measures")
        for m in phenotype.measures:
            mg.create_dataset(m, data=values[m])


def _read_h5(path: Path, phenotype) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        return {m: np.array(f[f"measures/{m}"]) for m in phenotype.measures if m in f["measures"]}


def _write_bids(path: Path, phenotype, values: Mapping) -> None:
    """Write the table and its column sidecar: per ``MeasureSpec``, ``Description``, ``Units`` as the unit's UCUM code (the unit as written when it has none) and ``TermURL`` as the ``iri`` expanded to a URL."""
    import pandas as pd

    from tvbo.datamodel.dialect import expand_curie
    from tvbo.utils.units import unit_to_ucum

    table = pd.DataFrame(
        {"participant_id": [f"sub-{s}" for s in phenotype.subjects], **{m: values[m] for m in phenotype.measures}}
    )
    table.to_csv(path, sep="\t", index=False, na_rep="n/a")
    sidecar = {}
    for spec in phenotype.measure_specs or []:
        entry = {
            "Description": spec.description,
            "Units": (unit_to_ucum(spec.unit) or spec.unit) if spec.unit else None,
            "TermURL": expand_curie(spec.iri) if spec.iri else None,
        }
        sidecar[spec.name] = {k: v for k, v in entry.items() if v is not None}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def _read_bids(path: Path, phenotype) -> dict:
    """The measure columns as float arrays; the ``participant_id`` column must list the record's subjects in order, so no row is paired with the wrong subject."""
    import pandas as pd

    wanted = {"participant_id", *phenotype.measures}
    table = pd.read_csv(
        path,
        sep="\t",
        index_col="participant_id",
        usecols=lambda c: c in wanted,
        dtype={"participant_id": str},
        na_values=["n/a"],
        keep_default_na=False,
        float_precision="round_trip",
    )
    ids = [str(p).removeprefix("sub-") for p in table.index]
    if ids != list(phenotype.subjects):
        raise ValueError(
            f"{path}: participant_id column {ids} does not match the record's subjects {list(phenotype.subjects)}"
        )
    return {m: table[m].to_numpy(dtype=float) for m in phenotype.measures if m in table.columns}


_FORMATS = {".h5": (_write_h5, _read_h5), ".tsv": (_write_bids, _read_bids)}


def _formats(path: Path) -> tuple:
    """The ``(writer, reader)`` pair for the format *path*'s extension names."""
    try:
        return _FORMATS[path.suffix]
    except KeyError:
        raise ValueError(f"Phenotype companion {path.name!r}: extension must be one of {sorted(_FORMATS)}") from None
