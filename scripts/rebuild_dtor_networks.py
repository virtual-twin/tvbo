#!/usr/bin/env python3
"""Recount the rec-dTOR networks from the dTOR tractogram converted straight from Lead-DBS's ``dTOR_985.mat``.

The networks were counted from a ``.tck`` of dTOR (Elias 2024, 11,820,000 streamlines) whose every point lay (+0.25, +0.25, -0.25) mm off, because the conversion from Lead-DBS's ``.trk`` read its voxel-centre coordinates as voxel corners. The replacement holds the same streamlines in the same order with the same points, minus that shift.

Each network is recounted as it was first counted: ``tck2connectome`` with its default radial assignment and ``-symmetric -zero_diagonal``, mean lengths as ``-scale_length -stat_edge mean`` gives them, on the parcellation volume in ``VOLUMES``. The edges named in ``KEEP_UNASSIGNED`` were counted with ``-keep_unassigned`` as well, so their first row and column hold the streamlines with an unassigned end; HCPex's weights were counted that way and its lengths were not, so its weight matrix has one node more than its length matrix and the two are offset by that node. That assignment reads nothing of a streamline but its two endpoints, so the 12.5 GB tractogram is read once into a two-point ``.tck`` of the endpoints and the length of every streamline (the sum of its segment lengths); each volume then costs one ``tck2connectome`` pass over the endpoints, and weights and mean lengths follow from its ``-out_assignments``. Both are cached in ``--cache-dir``, so a volume shared by several networks is counted once and a second run reads no tractogram. A network the endpoints do not reproduce is counted with ``connectome_from_tractogram`` over the full tractogram instead.

The volumes are the ones that reproduce every stored weight matrix exactly from the shifted file; pass that file as ``--check`` to repeat that proof (weights equal, lengths equal at the stored float32) against companions that still hold the shifted counts, that is before this script has rewritten them. Every network is checked and recounted before the first one is written, so a failure leaves the database as it was. Only the edge data in each HDF5 companion changes: format, precision, attributes, node table and sidecar stay as they are.

``Lobar8 desc-SCFC`` is the Lobar network without its BrainStem node plus an ``fc`` edge aggregated from the DesikanKilliany avgMatrix, so its weights and lengths are Lobar's recount without BrainStem and its ``fc`` stays as it is. ``Lobar desc-surf`` holds a mesh and no edges, and is left alone.

The Lobar volume is rebuilt by ``create_lobar_network.build_lobar_atlas`` from TemplateFlow's DKT31, so ``TEMPLATEFLOW_HOME`` must name a TemplateFlow cache holding the ``res-02`` DKT31 segmentation (and no ``res-01``, which that function prefers).

Usage:
    TEMPLATEFLOW_HOME=~/.cache/templateflow python scripts/rebuild_dtor_networks.py --tck dTOR.tck --cache-dir /tmp/dtor [--check shifted_dTOR.tck] [--dry-run]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from create_lobar_network import LOBE_ORDER, LOBE_ORDER_8, build_lobar_atlas
from rebuild_leaddbs_cohort_networks import NETWORKS, rewrite, stored
from rebuild_leaddbs_cohort_networks import VOLUMES as LEADDBS_VOLUMES
from scipy.stats import spearmanr

from tvbo.data.connectome_build import connectome_from_tractogram, ensure_mrtrix, tck2connectome_commands

COHORT = "cohort-HCPYA_rec-dTOR"
ATLASES = NETWORKS.parent / "atlases"
SOURCE = Path(os.environ.get("VIRTUAL_DBS_SOURCE", "/Volumes/bronkodata/virtual-dbs-source"))
LOBAR = "lobar"
VOLUMES = {
    **{f"{stem}_desc-SC": volume for stem, volume in LEADDBS_VOLUMES.items()},
    **{
        f"tpl-FSLMNI152_{{c}}_atlas-Schaefer2018_seg-{seg}Networks_scale-{scale}_desc-SC": ATLASES
        / f"space-FSLMNI152_atlas-Schaefer2018_seg-{seg}Networks_scale-{scale}_res-1_desc-original_dseg.nii.gz"
        for seg in (7, 17)
        for scale in range(100, 1001, 100)
    },
    "tpl-MNI152NLin2009cAsym_{c}_atlas-HCPex_desc-SC": SOURCE / "atlases/HCPex/HCPex.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-Lobar_desc-SC": LOBAR,
}
KEEP_UNASSIGNED = {"tpl-MNI152NLin2009cAsym_{c}_atlas-HCPex_desc-SC": {"weight"}}
SUBNETWORKS = {
    "tpl-MNI152NLin2009cAsym_{c}_atlas-Lobar8_desc-SCFC": (
        "tpl-MNI152NLin2009cAsym_{c}_atlas-Lobar_desc-SC",
        [LOBE_ORDER.index(lobe) for lobe in LOBE_ORDER_8],
    ),
}
CHUNK = 2**25
THREADS = ["-quiet", "-nthreads", "8"]


def file_key(path):
    """A short name for `path`'s current content: its bytes for a small file, its location, size and mtime for a tractogram."""
    st = Path(path).stat()
    if st.st_size < 2**30:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    return hashlib.sha1(f"{Path(path).resolve()}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]


def tck_offset(path):
    """The byte offset of the points in an MRtrix ``.tck``, which must hold little-endian float32."""
    fields = {}
    with open(path, "rb") as f:
        if f.readline().strip() != b"mrtrix tracks":
            raise SystemExit(f"{path}: not an MRtrix tractogram")
        for line in f:
            key, _, value = line.decode().strip().partition(": ")
            if key == "END":
                break
            fields[key] = value
    if fields.get("datatype") != "Float32LE":
        raise SystemExit(f"{path}: datatype {fields.get('datatype')}, expected Float32LE")
    return int(fields["file"].split()[1])


def write_tck(path, points, offset=256):
    """Write `points`, shape (streamlines, points per streamline, 3), as an MRtrix ``.tck`` whose points start at byte `offset`; written beside `path` and renamed over it."""
    n = len(points)
    body = np.concatenate([points.astype("<f4"), np.full((n, 1, 3), np.nan, "<f4")], axis=1)
    tmp = path.with_suffix(".tck.part")
    with open(tmp, "wb") as f:
        f.write(f"mrtrix tracks\ncount: {n}\ndatatype: Float32LE\nfile: . {offset}\nEND\n".encode().ljust(offset, b"\0"))
        f.write(body.tobytes())
        f.write(np.full(3, np.inf, "<f4").tobytes())
    os.replace(tmp, path)


@functools.cache
def endpoints(tck, cache_dir):
    """The two-point ``.tck`` of `tck`'s streamline endpoints and its streamline lengths, each segment's length in float32 as MRtrix measures it, read in one pass and cached."""
    key = file_key(tck)
    ends_tck, lengths_npy = cache_dir / f"{key}_endpoints.tck", cache_dir / f"{key}_lengths.npy"
    if ends_tck.exists() and lengths_npy.exists():
        return ends_tck, np.load(lengths_npy)
    fronts, backs, lengths = [], [], []
    carry = np.empty((0, 3), np.float32)
    with open(tck, "rb") as f:
        f.seek(tck_offset(tck))
        while True:
            block = np.fromfile(f, dtype="<f4", count=3 * CHUNK)
            if not block.size:
                raise SystemExit(f"{tck}: ends without its terminator")
            pts = np.concatenate([carry, block.reshape(-1, 3)])
            stop = np.flatnonzero(np.isinf(pts[:, 0]))
            if stop.size:
                pts = pts[: stop[0]]
            delims = np.flatnonzero(np.isnan(pts[:, 0]))
            starts = np.r_[0, delims[:-1] + 1][: delims.size]
            if np.any(delims == starts):
                raise SystemExit(f"{tck}: holds an empty streamline")
            seg = np.sqrt(np.square(np.diff(pts, axis=0)).sum(axis=1)).astype(np.float64)
            cum = np.r_[0.0, np.cumsum(np.nan_to_num(seg, nan=0.0))]
            fronts.append(pts[starts])
            backs.append(pts[delims - 1])
            lengths.append(cum[delims - 1] - cum[starts])
            carry = pts[(delims[-1] + 1 if delims.size else 0) :]
            if stop.size:
                if carry.size:
                    raise SystemExit(f"{tck}: its last streamline has no delimiter")
                break
    write_tck(ends_tck, np.stack([np.concatenate(fronts), np.concatenate(backs)], axis=1))
    lengths = np.concatenate(lengths)
    np.save(lengths_npy, lengths)
    print(f"{tck}: {len(lengths):,} streamlines, endpoints in {ends_tck}", flush=True)
    return ends_tck, lengths


def matrices(assignments, lengths, n, keep_unassigned):
    """Streamline-count and mean-length matrices from each streamline's node pair, as ``tck2connectome -symmetric -zero_diagonal`` forms them."""
    a, b = np.sort(assignments, axis=1).T
    first = 0 if keep_unassigned else 1
    keep = (a != b) & (a >= first)
    pair = (a[keep] - first) * n + (b[keep] - first)
    counts = np.bincount(pair, minlength=n * n).reshape(n, n).astype(np.float64)
    sums = np.bincount(pair, weights=lengths[keep], minlength=n * n).reshape(n, n)
    counts, sums = counts + counts.T, sums + sums.T
    return {"weight": counts, "length": np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)}


@functools.cache
def count(tck, volume, cache_dir, keep_unassigned):
    """The weight and mean-length matrices of `tck` on `volume` from its endpoints, counted once however many networks share the pair."""
    flags = ["-keep_unassigned"] if keep_unassigned else []
    cached = cache_dir / f"{file_key(tck)}_{file_key(volume)}{'_unassigned' if keep_unassigned else ''}.npz"
    if cached.exists():
        return dict(np.load(cached))
    ends_tck, lengths = endpoints(tck, cache_dir)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        weights_csv, assignments_csv = Path(tmp) / "weights.csv", Path(tmp) / "assignments.txt"
        cmd = tck2connectome_commands(
            ends_tck, volume, weights_csv, Path(tmp) / "unused.csv", assignments_csv, extra_args=THREADS + flags
        )[0]
        subprocess.run(cmd, check=True)
        weights = np.atleast_2d(np.loadtxt(weights_csv, delimiter=","))
        assignments = pd.read_csv(assignments_csv, sep=" ", header=None, comment="#", dtype=np.int64).to_numpy()
    if len(assignments) != len(lengths):
        raise SystemExit(f"{volume}: {len(assignments):,} assignments for {len(lengths):,} streamlines")
    result = matrices(assignments, lengths, len(weights), keep_unassigned)
    if not np.array_equal(result["weight"], weights):
        raise SystemExit(f"{volume}: the assignments do not add up to tck2connectome's own weights")
    np.savez(cached, **result)
    return result


@functools.cache
def count_full(tck, volume, keep_unassigned):
    """The weight and mean-length matrices of `tck` on `volume` from every point of every streamline, as the networks were first counted."""
    weights, lengths = connectome_from_tractogram(
        Path(tck), volume, extra_args=THREADS + (["-keep_unassigned"] if keep_unassigned else [])
    )
    return {"weight": weights, "length": lengths}


def counted(stem, tck, cache_dir, full=False):
    """The recount of the network `stem` from `tck`: its volume's count, without the unassigned node for the edges not in ``KEEP_UNASSIGNED``, or for a subnetwork its parent's restricted to its nodes."""
    if stem in SUBNETWORKS:
        parent, nodes = SUBNETWORKS[stem]
        return {k: M[np.ix_(nodes, nodes)] for k, M in counted(parent, tck, cache_dir, full).items()}
    volume = VOLUMES[stem]
    if volume == LOBAR:
        volume = cache_dir / "lobar_atlas.nii.gz"
        if not volume.exists():
            build_lobar_atlas(volume)
    keep = KEEP_UNASSIGNED.get(stem, set())
    result = count_full(tck, volume, bool(keep)) if full else count(tck, volume, cache_dir, bool(keep))
    return {k: M if k in keep or not keep else M[1:, 1:] for k, M in result.items()}


def reproduces(again, old):
    """Whether a recount equals the stored network: weights exactly, lengths at the stored precision."""
    return (
        all(again[k].shape == old[k].shape for k in old)
        and np.array_equal(again["weight"], old["weight"])
        and np.allclose(again["length"].astype(old["length"].dtype), old["length"])
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tck", required=True, type=Path, help="dTOR converted from Lead-DBS's dTOR_985.mat")
    p.add_argument("--check", type=Path, help="the shifted dTOR file the networks were counted from")
    p.add_argument("--cache-dir", required=True, type=Path, help="where the endpoints, lengths and per-volume counts are kept")
    p.add_argument("--dry-run", action="store_true", help="count and compare, write nothing")
    args = p.parse_args()
    ensure_mrtrix()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()

    recounted = []
    for stem in [*VOLUMES, *SUBNETWORKS]:
        yaml_path = NETWORKS / (stem.format(c=COHORT) + "_relmat.yaml")
        if not yaml_path.exists():
            continue
        h5_path = yaml_path.with_suffix(".h5")
        old = stored(h5_path)
        full = False
        if args.check:
            if not reproduces(counted(stem, args.check, cache_dir), old):
                full = True
                if not reproduces(counted(stem, args.check, cache_dir, full=True), old):
                    raise SystemExit(
                        f"{yaml_path.name}: {VOLUMES.get(stem, stem)} does not reproduce the stored network from {args.check}"
                    )
        new = counted(stem, args.tck, cache_dir, full)
        if any(new[k].shape != old[k].shape for k in old):
            raise SystemExit(f"{yaml_path.name}: recount is {new['weight'].shape}, stored {old['weight'].shape}")
        iu = np.triu_indices_from(old["weight"], 1)
        rho = spearmanr(old["weight"][iu], new["weight"][iu])[0]
        print(
            f"{yaml_path.name}: {'reproduced' + (' over the full tractogram' if full else '') + ', ' if args.check else ''}pairs connected {np.mean(old['weight'][iu] > 0):.1%} -> {np.mean(new['weight'][iu] > 0):.1%}, streamlines counted {old['weight'][iu].sum():,.0f} -> {new['weight'][iu].sum():,.0f}, Spearman old vs new {rho:.4f}",
            flush=True,
        )
        recounted.append((h5_path, new))
    if not args.dry_run:
        rewrite(recounted)


if __name__ == "__main__":
    main()
