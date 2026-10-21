#!/usr/bin/env python3
"""Recount the 17 rec-MghUscHcp32 and rec-PPMI85 networks from Lead-DBS's original group connectomes.

The networks were counted from ``.tck`` conversions of Lead-DBS's ``data.mat`` that had grouped each streamline's points with an unstable sort, so the points of most streamlines were out of order and one end lay inside the white matter: counted from them, 30% of MGH-USC 32's and 61% of PPMI 85's streamlines end outside every parcel, against 12% and 20% from the originals. The originals are Lead-DBS's ``group2017`` (MGH-USC HCP 32, Horn 2017) and ``group2017_ppmi`` (PPMI 85, Ewert 2017) releases, converted point-order-preserving by hcp-connectome-studies' ``analyses/sc/scripts/leaddbs_to_tck.py``.

Each network is recounted as it was first counted, with ``connectome_from_tractogram`` (``tck2connectome`` with its default radial assignment, ``-symmetric -zero_diagonal``; mean lengths with ``-scale_length -stat_edge mean``) on the parcellation volume in ``VOLUMES``. Those volumes are the ones that reproduce every stored weight matrix exactly from the scrambled files; pass those files as ``--check-mgh`` and ``--check-ppmi`` to repeat that proof (weights equal, lengths equal at the stored float32) against companions that still hold the scrambled counts, that is before this script has rewritten them. Every network is checked, recounted, written beside its companion and read back before the first companion is replaced, so a failure leaves the database as it was, with no partial file beside it. Only the weight and length data in each HDF5 companion change: format, precision, attributes, node table, any other edge group and sidecar stay as they are.

Usage:
    python scripts/rebuild_leaddbs_cohort_networks.py --mgh MghUscHcp32_from-leaddbs.tck --ppmi PPMI85_from-leaddbs.tck [--check-mgh old.tck --check-ppmi old.tck] [--dry-run]
"""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

from tvbo.data.connectome_build import connectome_from_tractogram, ensure_mrtrix
from tvbo.data.matrix_io import read_matrix, write_matrix

NETWORKS = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "networks"
ARCHIVE = Path.home() / "projects" / "TVB-O" / "_archive"
TEMPLATEFLOW = Path(os.environ.get("TEMPLATEFLOW_HOME", Path.home() / "Library" / "Caches" / "templateflow"))
HCPYTHON = Path.home() / "projects" / "ImageProcessing" / "hcpython"
VOLUMES = {
    "tpl-FSLMNI152_{c}_atlas-Schaefer2018_scale-1000": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-FSLMNI152_atlas-Schaefer1000-desc-17Networks_res-2_dseg.nii.gz",
    "tpl-FSLMNI152_{c}_atlas-Schaefer2018_seg-17Networks_scale-1000": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-FSLMNI152_atlas-Schaefer1000-desc-17Networks_res-2_dseg.nii.gz",
    "tpl-MNI152NLin2009bAsym_{c}_atlas-HCPMMP1": HCPYTHON
    / "resources/atlases/tpl-MNI152NLin2009bAsym_atlas-HCPMMP1_res-05_desc-dTOR_dseg.nii.gz",
    "tpl-MNI152NLin2009bAsym_{c}_atlas-HCPMMP1_seg-ordered": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-MNI152NLin2009b_atlas-hcpmmp1_desc-ordered_dseg.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-DesikanKilliany": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-MNI152Nlin2009c_atlas-DesikanKilliany_desc-ranked_dseg.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-Destrieux": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-MNI152Nlin2009c_atlas-Destrieux_desc-ranked_dseg.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-HOCPA": TEMPLATEFLOW
    / "tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_atlas-HOCPA_desc-th0_dseg.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-Yeo17": ARCHIVE
    / "tvbo_old/tvbo/data/tvbo_data/atlas/space-MNI152_atlas-Yeo17_res-1_dseg.nii.gz",
    "tpl-MNI152NLin2009cAsym_{c}_atlas-virtualdbs": ARCHIVE
    / "tvbo-data/tvbo_data/atlas/tpl-MNI152NLin2009c_atlas-virtualdbs_res-07_dseg.nii.gz",
}


def stored(h5_path):
    """The weight and length matrices the companion holds, dense whatever their storage format."""
    with h5py.File(h5_path, "r") as f:
        matrices = {name: read_matrix(f[f"edges/{name}"]) for name in ("weight", "length")}
    return {name: M.toarray() if sparse.issparse(M) else np.asarray(M) for name, M in matrices.items()}


def stage(h5_path, part, matrices):
    """Write to `part` the companion `h5_path` with `matrices` as its edge data, and check that it reads back as them. An edge group `matrices` names keeps its format, precision and attributes; every other group is copied as it is."""
    with h5py.File(h5_path, "r") as src, h5py.File(part, "w") as dst:
        dst.attrs.update(src.attrs)
        for key in src:
            if key != "edges":
                src.copy(src[key], dst, name=key)
        edges = dst.create_group("edges")
        edges.attrs.update(src["edges"].attrs)
        for name, grp in src["edges"].items():
            if name not in matrices:
                src.copy(grp, edges, name=name)
                continue
            out = edges.create_group(name)
            out.attrs.update(grp.attrs)
            write_matrix(out, matrices[name], fmt=str(grp.attrs["format"]), dtype=grp["data"].dtype)
    back = stored(part)
    if not all(np.array_equal(back[k], matrices[k].astype(back[k].dtype)) for k in back):
        raise SystemExit(f"{h5_path.name}: the rewritten companion does not read back")


def rewrite(recounted):
    """Replace the edge data of every companion in `recounted`, pairs of a companion and its matrices, or of none. Each is staged beside its companion (`stage`), and the staged files are renamed over the companions only once all of them stand; a failure until then removes them and leaves every companion as it was."""
    parts = {h5_path: h5_path.with_suffix(".h5.part") for h5_path, _ in recounted}
    try:
        for h5_path, matrices in recounted:
            stage(h5_path, parts[h5_path], matrices)
        for h5_path, part in parts.items():
            os.replace(part, h5_path)
    finally:
        for part in parts.values():
            part.unlink(missing_ok=True)


@functools.cache
def count(tck, volume):
    """The weight and mean-length matrices of `tck` on `volume`, counted once however many networks share the pair."""
    weights, lengths = connectome_from_tractogram(Path(tck), volume, extra_args=["-quiet", "-nthreads", "8"])
    return {"weight": weights, "length": lengths}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mgh", required=True, help="MGH-USC HCP 32 from Lead-DBS's group2017 release, points in order")
    p.add_argument("--ppmi", required=True, help="PPMI 85 from Lead-DBS's group2017_ppmi release, points in order")
    p.add_argument("--check-mgh", help="the scrambled MGH-USC 32 file the networks were counted from")
    p.add_argument("--check-ppmi", help="the scrambled PPMI 85 file the networks were counted from")
    p.add_argument("--dry-run", action="store_true", help="count and compare, write nothing")
    args = p.parse_args()
    ensure_mrtrix()

    sources = {"MghUscHcp32": (args.mgh, args.check_mgh), "PPMI85": (args.ppmi, args.check_ppmi)}
    recounted = []
    for cohort, (tck, check) in sources.items():
        for stem, volume in VOLUMES.items():
            yaml_path = NETWORKS / (stem.format(c=f"cohort-{cohort}_rec-{cohort}") + "_desc-SC_relmat.yaml")
            if not yaml_path.exists():
                continue
            h5_path = yaml_path.with_suffix(".h5")
            old = stored(h5_path)
            if check:
                again = count(check, volume)
                if not (
                    np.array_equal(again["weight"], old["weight"])
                    and np.allclose(again["length"].astype(old["length"].dtype), old["length"])
                ):
                    raise SystemExit(f"{yaml_path.name}: {volume.name} does not reproduce the stored network from {check}")
            new = count(tck, volume)
            if new["weight"].shape != old["weight"].shape:
                raise SystemExit(f"{yaml_path.name}: recount is {new['weight'].shape}, stored {old['weight'].shape}")
            iu = np.triu_indices_from(old["weight"], 1)
            rho = spearmanr(old["weight"][iu], new["weight"][iu])[0]
            print(
                f"{yaml_path.name}: {'reproduced, ' if check else ''}pairs connected {np.mean(old['weight'][iu] > 0):.1%} -> {np.mean(new['weight'][iu] > 0):.1%}, streamlines counted {old['weight'][iu].sum():,.0f} -> {new['weight'][iu].sum():,.0f}, Spearman old vs new {rho:.2f}",
                flush=True,
            )
            recounted.append((h5_path, new))
    if not args.dry_run:
        rewrite(recounted)


if __name__ == "__main__":
    main()
