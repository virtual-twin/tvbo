"""``tvbo network`` — build brain-network connectomes from tractograms."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from . import _common

app = typer.Typer(
    name="network",
    help="Build connectomes from a tractogram + parcellation (MRtrix wrapper).",
    no_args_is_help=True,
)


@app.command("build")
def build(
    tractogram: Path = typer.Argument(
        ...,
        help="Streamline tractogram MRtrix can read (e.g. .tck). Must be in the same space as the parcellation.",
    ),
    parcellation: Path = typer.Argument(
        ...,
        help="Integer-labelled parcellation image (e.g. dseg.nii.gz), in the same space as the tractogram.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Sidecar output path (.yaml); the .h5 companion is written next to it. "
        "Default: a BIDS-derived name in the current directory.",
    ),
    atlas: Optional[str] = typer.Option(None, "--atlas", help="Parcellation/atlas name (metadata + filename entity)."),
    space: Optional[str] = typer.Option(
        None,
        "--space",
        help="Coordinate space both inputs share (e.g. FSLMNI152). Recorded as the "
        "template entity; the build assumes the inputs are already co-registered.",
    ),
    cohort: Optional[str] = typer.Option(None, "--cohort", help="Cohort/dataset entity (e.g. HCPYA)."),
    reconstruction: Optional[str] = typer.Option(
        None,
        "--reconstruction",
        "--rec",
        help="Tractography pipeline name (e.g. dTOR); recorded as the tractogram and reconstruction entity.",
    ),
    segmentation: Optional[str] = typer.Option(None, "--seg", help="Segmentation entity (e.g. 17Networks)."),
    scale: Optional[str] = typer.Option(None, "--scale", help="Scale entity (e.g. 1000)."),
    labels: Optional[Path] = typer.Option(
        None,
        "--labels",
        help="Optional node labels: a text file with one label per line, ordered by parcellation label (1..N).",
    ),
    symmetric: bool = typer.Option(True, "--symmetric/--no-symmetric", help="Pass -symmetric to tck2connectome."),
    zero_diagonal: bool = typer.Option(
        True, "--zero-diagonal/--no-zero-diagonal", help="Pass -zero_diagonal to tck2connectome."
    ),
    keep_assignments: Optional[Path] = typer.Option(
        None, "--keep-assignments", help="Also save tck2connectome's -out_assignments to this path."
    ),
    mrtrix_arg: Optional[List[str]] = typer.Option(
        None, "--mrtrix-arg", help="Extra raw argument forwarded to both tck2connectome calls (repeatable)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the tck2connectome commands and exit without running them."),
) -> None:
    """Build a structural connectome (SC) from a tractogram + parcellation.

    A thin wrapper around MRtrix3 ``tck2connectome``: it counts streamlines between parcellation nodes (edge weights) and their mean lengths, then writes a tvbo
    network (``…_desc-SC_relmat.h5`` + YAML sidecar) you can load with
    ``tvbo.Network(...)``. The tractogram and parcellation must already be in the same space — this command does not register them.
    """
    from tvbo.data.connectome_build import (
        connectome_from_tractogram,
        ensure_mrtrix,
        tck2connectome_commands,
    )

    extra = list(mrtrix_arg) if mrtrix_arg else None

    if dry_run:
        import tempfile

        tmp = Path(tempfile.gettempdir())
        for cmd in tck2connectome_commands(
            tractogram,
            parcellation,
            tmp / "weights.csv",
            tmp / "lengths.csv",
            (tmp / "assignments.csv") if keep_assignments else None,
            symmetric=symmetric,
            zero_diagonal=zero_diagonal,
            extra_args=extra,
        ):
            typer.echo(" ".join(cmd))
        return

    ensure_mrtrix()  # friendly error before doing any work

    for path, what in ((tractogram, "tractogram"), (parcellation, "parcellation")):
        if not path.exists():
            _common.die(f"{what} not found: {path}")
    if labels is not None and not labels.exists():
        _common.die(f"labels file not found: {labels}")

    if not space:
        _common.info("No --space given; assuming the tractogram and parcellation are already co-registered in the same space.")

    from tvbo.classes.network import Network

    bids = {
        k: v
        for k, v in (
            ("template", space),
            ("cohort", cohort),
            ("reconstruction", reconstruction),
            ("atlas", atlas),
            ("segmentation", segmentation),
            ("scale", scale),
        )
        if v
    }

    def apply_metadata(net: "Network") -> "Network":
        """Attach the flag-derived SC metadata used for naming and serialisation.

        The ``atlas`` filename entity is read from ``parcellation.atlas.name`` (not ``bids``), so the naming shell needs the same parcellation the saved
        network gets — hence one shared helper for both.
        """
        net.descriptor = "SC"
        net.distance_unit = "mm"
        if atlas or space:
            net.parcellation = {"atlas": {k: v for k, v in (("name", atlas), ("coordinateSpace", space)) if v}}
        if reconstruction:
            net.tractogram = {"name": reconstruction}
        if bids:
            net.bids = bids
        label_bits = [b for b in (atlas, segmentation, scale, reconstruction) if b]
        if label_bits:
            net.label = " ".join(label_bits)
        return net

    # BIDS entities are entirely flag-derived, so resolve the output path and the overwrite guard up front — before the (potentially minutes-long) MRtrix run.
    if output is not None:
        sidecar = Path(output).with_suffix(".yaml")
    else:
        shell = apply_metadata(Network(nodes=[], edges=[], number_of_nodes=0))
        fname = getattr(shell, "bids_filename", None)
        if not fname:
            _common.die(
                "Could not derive an output name from the given entities; pass "
                "-o/--output, or add entities like --space/--atlas/--rec."
            )
        sidecar = Path.cwd() / Path(fname).with_suffix(".yaml")
    companion = sidecar.with_suffix(".h5")
    if (sidecar.exists() or companion.exists()) and not overwrite:
        _common.die(f"{sidecar.name} already exists (use --overwrite).")

    # 1. MRtrix → weight and length matrices.
    _common.info(f"Running tck2connectome on {tractogram.name} × {parcellation.name} …")
    weights, lengths = connectome_from_tractogram(
        tractogram,
        parcellation,
        symmetric=symmetric,
        zero_diagonal=zero_diagonal,
        extra_args=extra,
        assignments_out=keep_assignments,
    )
    n_nodes = weights.shape[0]

    # 2. Optional node labels.
    node_labels = None
    if labels is not None:
        node_labels = [ln.strip() for ln in labels.read_text().splitlines() if ln.strip()]
        if len(node_labels) != n_nodes:
            _common.die(f"labels count ({len(node_labels)}) != parcellation node count ({n_nodes}).")

    # 3. Assemble the tvbo Network with the same metadata used for naming.
    net = apply_metadata(Network.from_matrix(weights=weights, lengths=lengths, labels=node_labels))

    # 4. Save (sidecar + .h5 companion).
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    net.save(sidecar)

    import numpy as np

    n_edges = int((np.asarray(weights) != 0).sum() // (2 if symmetric else 1))
    typer.echo(f"OK — built SC network: {n_nodes} nodes, {n_edges} edges")
    typer.echo(f"  sidecar:   {sidecar}")
    typer.echo(f"  companion: {companion}")
