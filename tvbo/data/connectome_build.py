"""Build a structural connectome from a tractogram + parcellation via MRtrix3.

A thin wrapper around MRtrix3's ``tck2connectome``. Given a streamline tractogram and an integer-labelled parcellation image that already live in the *same* space, it returns the edge-weight (streamline-count) and mean-tract-length matrices. The inputs are assumed to be co-registered — this module does not register them.

Assembling the matrices into a :class:`tvbo.classes.network.Network` and writing the ``…_desc-SC_relmat.h5`` + YAML sidecar happens in the caller (``tvbo network build``); this module only shells out to MRtrix and reads back the CSVs, so it is the single place the ``tck2connectome`` invocation is defined.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

TCK2CONNECTOME = "tck2connectome"


def mrtrix_available(tool: str = TCK2CONNECTOME) -> bool:
    """Return True when the given MRtrix command is on ``PATH``."""
    return shutil.which(tool) is not None


def ensure_mrtrix(tool: str = TCK2CONNECTOME) -> None:
    """Raise a friendly ``RuntimeError`` when the MRtrix command is missing."""
    if not mrtrix_available(tool):
        raise RuntimeError(
            f"{tool!r} was not found on PATH. `tvbo network build` is a thin wrapper "
            f"around MRtrix3.\n"
            f"  Install MRtrix3: https://www.mrtrix.org/download/\n"
            f"  {tool} reference: "
            f"https://mrtrix.readthedocs.io/en/latest/reference/commands/{tool}.html\n"
            f"After installing, make sure `{tool}` runs from your shell."
        )


def tck2connectome_commands(
    tractogram: Path,
    parcellation: Path,
    weights_csv: Path,
    lengths_csv: Path,
    assignments_csv: Optional[Path] = None,
    *,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    force: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> list[list[str]]:
    """Return the two ``tck2connectome`` argv lists (edge weights, then lengths).

    The first call counts streamlines between each node pair (edge weights); the second scales each streamline by its length and averages per edge to get mean tract lengths. Returned rather than run so callers can preview them (dry-run).
    """
    common: list[str] = []
    if symmetric:
        common.append("-symmetric")
    if zero_diagonal:
        common.append("-zero_diagonal")
    if force:
        common.append("-force")
    if extra_args:
        common.extend(extra_args)

    weights_cmd = [TCK2CONNECTOME, str(tractogram), str(parcellation), str(weights_csv)]
    if assignments_csv is not None:
        weights_cmd += ["-out_assignments", str(assignments_csv)]
    weights_cmd += common

    lengths_cmd = [
        TCK2CONNECTOME,
        str(tractogram),
        str(parcellation),
        str(lengths_csv),
        "-scale_length",
        "-stat_edge",
        "mean",
    ] + common

    return [weights_cmd, lengths_cmd]


def run_tck2connectome(
    tractogram: Path,
    parcellation: Path,
    weights_csv: Path,
    lengths_csv: Path,
    assignments_csv: Optional[Path] = None,
    *,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    force: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    """Run ``tck2connectome`` twice: edge weights (count) then mean tract lengths."""
    for cmd in tck2connectome_commands(
        tractogram,
        parcellation,
        weights_csv,
        lengths_csv,
        assignments_csv,
        symmetric=symmetric,
        zero_diagonal=zero_diagonal,
        force=force,
        extra_args=extra_args,
    ):
        subprocess.run(cmd, check=True)


def connectome_from_tractogram(
    tractogram: Path,
    parcellation: Path,
    *,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    extra_args: Optional[Sequence[str]] = None,
    assignments_out: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ``(weights, lengths)`` matrices for a tractogram + parcellation.

    Runs ``tck2connectome`` in a temporary directory and loads the CSVs it writes.
    Both inputs must already be co-registered in the same space.

    Parameters
    ----------
    tractogram : Path
        Streamline tractogram MRtrix can read (e.g. ``.tck``).
    parcellation : Path
        Integer-labelled parcellation image (e.g. ``dseg.nii.gz``).
    assignments_out : Path, optional
        When given, ``tck2connectome``'s ``-out_assignments`` is kept at this path.

    Returns
    -------
    weights, lengths : np.ndarray
        ``(N, N)`` streamline-count and mean-length matrices.
    """
    ensure_mrtrix()
    tractogram = Path(tractogram)
    parcellation = Path(parcellation)
    for path, what in ((tractogram, "tractogram"), (parcellation, "parcellation")):
        if not path.exists():
            raise FileNotFoundError(f"{what} not found: {path}")

    with tempfile.TemporaryDirectory(prefix="tvbo_conn_") as tmp:
        tmpdir = Path(tmp)
        weights_csv = tmpdir / "weights.csv"
        lengths_csv = tmpdir / "lengths.csv"
        assignments_csv = tmpdir / "assignments.csv" if assignments_out else None

        run_tck2connectome(
            tractogram,
            parcellation,
            weights_csv,
            lengths_csv,
            assignments_csv,
            symmetric=symmetric,
            zero_diagonal=zero_diagonal,
            extra_args=extra_args,
        )

        weights = np.atleast_2d(np.loadtxt(weights_csv, delimiter=","))
        lengths = np.atleast_2d(np.loadtxt(lengths_csv, delimiter=","))
        if assignments_out is not None and assignments_csv is not None:
            shutil.copyfile(assignments_csv, assignments_out)

    return weights, lengths
