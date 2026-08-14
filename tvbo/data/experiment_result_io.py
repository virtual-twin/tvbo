"""Sidecar I/O + cross-experiment cache for :class:`ExperimentResult`.

Lets downstream experiments depend on an upstream experiment's fitted parameters (e.g. Schirner Exp 60_A reading Exp 30's ``w_LRE``, ``w_FFI``,
``J_i`` via an ``aux_data: Reference``) without recomputing on every notebook re-execution.

Layout
------

::

    reference/exp_<id>_seed<seed>.yaml          (descriptor + provenance)
    reference/exp_<id>_seed<seed>.h5
    └── parameters/
        ├── w_LRE     (n_nodes, n_nodes)
        ├── w_FFI     (n_nodes, n_nodes)
        ├── J_i       (n_nodes,)
        └── ...

Each array is stored at the precision it was computed at, which is the precision the descriptor's ``parameters[].dtype`` reports.

Cache invalidation
------------------

- ``provenance.experiment_yaml_hash`` — SHA-256 of the normalised YAML
  that produced this sidecar.
- ``provenance.inputs`` — for each ``aux_data`` reference, a
  ``ReferenceFingerprint`` ``{iri, field, mtime, size, hash}``.

A cache hit requires both:

1. Current experiment YAML hash matches ``provenance.experiment_yaml_hash``.
2. For each input fingerprint, the underlying file's ``(mtime, size)`` matches (fast path). On mismatch, recompute the sha256; if THAT
   matches, still a hit (handles ``touch`` and rename-in-place).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml


# ----------------------------------------------------------------------------
# YAML hashing
# ----------------------------------------------------------------------------


def hash_yaml(normalized_dict: Mapping[str, Any]) -> str:
    """Stable SHA-256 hex digest of a Python-side YAML representation.

    Uses ``yaml.safe_dump(sort_keys=True)`` so the digest only depends on semantic content, not key ordering.
    """
    blob = yaml.safe_dump(dict(normalized_dict), sort_keys=True, default_flow_style=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_fingerprint(path: str | os.PathLike) -> dict:
    """Return ``{mtime, size, hash}`` for a file.

    The hash is the file content's SHA-256. mtime is the POSIX timestamp.
    """
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {"mtime": float(st.st_mtime), "size": int(st.st_size), "hash": h.hexdigest()}


# ----------------------------------------------------------------------------
# Sidecar I/O
# ----------------------------------------------------------------------------


def save_sidecar(
    *,
    parameters: Mapping[str, np.ndarray],
    yaml_path: str | os.PathLike,
    experiment_yaml_hash: str,
    inputs: list[dict] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    provenance_comment: str | None = None,
) -> tuple[Path, Path]:
    """Write an ExperimentResult sidecar (yaml descriptor + h5 companion).

    Parameters
    ----------
    parameters
        ``{name: ndarray}`` — the fitted parameter arrays to cache.
    yaml_path
        Target ``.yaml`` path. ``.h5`` companion sits next to it (same
        stem).
    experiment_yaml_hash
        SHA-256 hex digest of the producing experiment's normalised
        YAML (see :func:`hash_yaml`).
    inputs
        List of ``ReferenceFingerprint`` dicts for each ``aux_data``
        reference the producing experiment consumed.
    extra_metadata
        Free-form dict merged into the yaml under ``metadata:``.
    provenance_comment
        Optional ``#``-prefixed block prepended to the yaml.
    """
    yaml_path = Path(yaml_path)
    h5_path = yaml_path.with_suffix(".h5")
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # h5 companion
    with h5py.File(h5_path, "w") as f:
        pg = f.create_group("parameters")
        for name, arr in parameters.items():
            pg.create_dataset(name, data=np.asarray(arr))

    # yaml descriptor
    descriptor = {
        "tvbo_class": "tvbo:ExperimentResultSidecar",
        "data_file": h5_path.name,
        "parameters": [
            {"name": name, "shape": list(np.shape(arr)), "dtype": str(np.asarray(arr).dtype)}
            for name, arr in parameters.items()
        ],
        "provenance": {
            "experiment_yaml_hash": experiment_yaml_hash,
            "inputs": inputs or [],
            "generated_by": "tvbo.data.experiment_result_io.save_sidecar",
        },
    }
    if extra_metadata:
        descriptor["metadata"] = dict(extra_metadata)

    with open(yaml_path, "w") as f:
        if provenance_comment:
            f.write(provenance_comment)
        yaml.safe_dump(descriptor, f, sort_keys=False, default_flow_style=False)
    return yaml_path, h5_path


def load_sidecar(yaml_path: str | os.PathLike) -> tuple[dict, dict]:
    """Load an ExperimentResult sidecar.

    Returns
    -------
    parameters
        ``{name: ndarray}`` — eagerly loaded numeric arrays.
    descriptor
        The parsed yaml descriptor (provenance + metadata).
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        descriptor = yaml.safe_load(f) or {}
    h5_path = yaml_path.parent / descriptor["data_file"]
    parameters: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for name in f["parameters"]:
            parameters[name] = np.array(f[f"parameters/{name}"])
    return parameters, descriptor


# ----------------------------------------------------------------------------
# Cache hit/miss check
# ----------------------------------------------------------------------------


class CacheStatus:
    """Reason a cache entry was accepted or rejected."""

    HIT = "hit"
    MISS_NO_SIDECAR = "miss_no_sidecar"
    MISS_YAML_HASH = "miss_yaml_hash"
    MISS_INPUT_MTIME_SIZE = "miss_input_mtime_size"
    MISS_INPUT_HASH = "miss_input_hash"
    MISS_INPUT_MISSING = "miss_input_missing"


def check_cache(
    *,
    sidecar_yaml: str | os.PathLike,
    expected_yaml_hash: str,
    input_paths: Mapping[str, str | os.PathLike] | None = None,
) -> tuple[str, str | None]:
    """Decide whether a cached ExperimentResult sidecar is still valid.

    Parameters
    ----------
    sidecar_yaml
        Path to the candidate ``.yaml`` descriptor.
    expected_yaml_hash
        Hash of the *current* experiment YAML. If this differs from the
        sidecar's recorded ``experiment_yaml_hash``, cache miss.
    input_paths
        ``{ref_iri_or_field: path}`` mapping each input fingerprint's
        identifier to its current on-disk path. For each fingerprint in
        the sidecar, the matching ``path`` is fingerprinted and compared.

    Returns
    -------
    status
        One of :class:`CacheStatus` constants.
    detail
        Human-readable detail (or ``None`` on hit).
    """
    sidecar_yaml = Path(sidecar_yaml)
    if not sidecar_yaml.exists():
        return CacheStatus.MISS_NO_SIDECAR, f"{sidecar_yaml} does not exist"

    with open(sidecar_yaml) as f:
        descriptor = yaml.safe_load(f) or {}
    prov = descriptor.get("provenance", {}) or {}
    recorded_hash = prov.get("experiment_yaml_hash")
    if recorded_hash != expected_yaml_hash:
        return (
            CacheStatus.MISS_YAML_HASH,
            f"yaml hash differs: cached={recorded_hash[:8]!r}, current={expected_yaml_hash[:8]!r}",
        )

    inputs = prov.get("inputs", []) or []
    input_paths = dict(input_paths or {})
    for fp in inputs:
        key = fp.get("iri") or fp.get("field") or "?"
        path = input_paths.get(key) or input_paths.get(fp.get("iri")) or input_paths.get(fp.get("field"))
        if path is None:
            # No mapping provided — skip (assume caller doesn't track this input)
            continue
        path = Path(path)
        if not path.exists():
            return CacheStatus.MISS_INPUT_MISSING, f"input {key!r}: {path} missing"
        st = path.stat()
        if float(st.st_mtime) == float(fp.get("mtime", 0)) and int(st.st_size) == int(fp.get("size", -1)):
            continue  # fast-path hit
        # mtime/size differ — recompute hash
        cur_hash = file_fingerprint(path)["hash"]
        if cur_hash != fp.get("hash"):
            return (CacheStatus.MISS_INPUT_HASH, f"input {key!r} hash mismatch (file modified)")
    return CacheStatus.HIT, None
