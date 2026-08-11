#!/usr/bin/env python
"""Export MNE standard_1005 EEG montage and fsaverage gain into a tvbo Network.

This script creates a native tvbo sidecar + companion pair (YAML + HDF5) with sensor nodes and a ``gain`` matrix (sensors x regions), plus a small NPZ
containing region labels/centers for bsplot visualization.

Typical workflow
----------------
1) Build and save network + metadata:
   python scripts/export_mne_standard1005_projection_network.py

2) Replot from saved files only (no forward recomputation):
   python scripts/export_mne_standard1005_projection_network.py --skip-export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np

from bsplot.graph import create_network, plot_network_on_surface
from tvbo import Network, Observation, database_path
from tvbo.datamodel import tvbo_datamodel


DEFAULT_OUTPUT = database_path / "networks" / "sensors_eeg_standard1005_fsaverage_aparc_projection.yaml"


def _resolve_output_paths(
    output_yaml: Path,
    metadata_npz: Path | None,
    plot_png: Path | None,
) -> tuple[Path, Path, Path]:
    output_yaml = output_yaml.resolve()
    if metadata_npz is None:
        metadata_npz = output_yaml.with_name(f"{output_yaml.stem}_plotmeta.npz")
    if plot_png is None:
        plot_png = output_yaml.with_name(f"{output_yaml.stem}.png")
    return output_yaml, metadata_npz.resolve(), plot_png.resolve()


def _matched_sensor_labels_and_positions() -> tuple[list[str], list[str], np.ndarray]:
    """Match tvbo EEG labels to MNE standard_1005 channel positions.

    Returns
    -------
    sensor_labels_tvb : list[str]
        TVBO sensor labels (used for saved node labels).
    sensor_labels_mne : list[str]
        Matched MNE channel names (used for forward model).
    sensor_pos_mm : ndarray, shape (n_sensors, 3)
        Sensor coordinates in mm.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    obs_eeg = Observation.from_db("eeg")
    sensor_ref = Network.from_file(database_path / "networks" / obs_eeg.data_source.path)
    sensor_labels_tvb = [str(node.label) for node in sensor_ref.nodes]

    label_map = {
        "T8/T4": "T8",
        "T7/T3": "T7",
        "P8/T6": "P8",
        "P7/T5": "P7",
    }

    matched_tvb = []
    matched_mne = []
    positions = []
    for label in sensor_labels_tvb:
        mne_name = label_map.get(label, label)
        if mne_name in ch_pos:
            matched_tvb.append(label)
            matched_mne.append(mne_name)
            positions.append(np.asarray(ch_pos[mne_name], dtype=float) * 1000.0)

    if not positions:
        raise ValueError("No EEG channels from tvbo reference network matched MNE montage")

    return matched_tvb, matched_mne, np.asarray(positions, dtype=float)


def _compute_region_gain(sensor_labels_mne: list[str]):
    """Compute parcel-level gain from MNE fsaverage forward model.

    Returns
    -------
    region_labels : list[str] region_centers_mm : ndarray, shape (n_regions, 3) gain_matrix : ndarray, shape (n_sensors, n_regions)
    fs_dir : Path
        Path to fetched fsaverage directory.
    """
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = fs_dir.parent

    src = mne.read_source_spaces(str(fs_dir / "bem" / "fsaverage-ico-5-src.fif"), verbose=False)

    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc",
        subjects_dir=str(subjects_dir),
        verbose=False,
    )
    labels = [label for label in labels if "unknown" not in label.name.lower()]

    info = mne.create_info(sensor_labels_mne, sfreq=1000.0, ch_types="eeg")
    info.set_montage("standard_1005")

    bem_sol = mne.read_bem_solution(
        str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"),
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem_sol,
        eeg=True,
        meg=False,
        verbose=False,
    )
    fwd_fixed = mne.convert_forward_solution(
        fwd,
        surf_ori=True,
        force_fixed=True,
        verbose=False,
    )
    gain_full = fwd_fixed["sol"]["data"]

    region_labels: list[str] = []
    region_centers = []
    source_indices_per_region = []

    for label in labels:
        hemi_idx = 0 if label.hemi == "lh" else 1
        src_vertno = src[hemi_idx]["vertno"]
        common = np.intersect1d(label.vertices, src_vertno)
        if len(common) == 0:
            continue

        indices_in_src = np.searchsorted(src_vertno, common)
        if hemi_idx == 1:
            indices_in_src += len(src[0]["vertno"])
        indices_in_src = indices_in_src[indices_in_src < gain_full.shape[1]]
        if len(indices_in_src) == 0:
            continue

        center_mm = src[hemi_idx]["rr"][common].mean(axis=0) * 1000.0
        region_labels.append(label.name)
        region_centers.append(center_mm)
        source_indices_per_region.append(indices_in_src)

    if not source_indices_per_region:
        raise ValueError("No valid source-space labels found for gain aggregation")

    region_centers_mm = np.asarray(region_centers, dtype=float)
    gain_matrix = np.zeros((gain_full.shape[0], len(source_indices_per_region)), dtype=float)
    abs_gain = np.abs(gain_full)
    for idx, src_indices in enumerate(source_indices_per_region):
        gain_matrix[:, idx] = abs_gain[:, src_indices].mean(axis=1)

    return region_labels, region_centers_mm, gain_matrix, fs_dir


def _build_tvbo_sensor_network(
    sensor_labels_tvb: list[str],
    sensor_pos_mm: np.ndarray,
    gain_matrix: np.ndarray,
) -> Network:
    """Create a tvbo sensor network with a projection gain matrix."""
    nodes = []
    for idx, (label, pos) in enumerate(zip(sensor_labels_tvb, sensor_pos_mm)):
        nodes.append(
            tvbo_datamodel.Node(
                id=idx,
                label=label,
                position=tvbo_datamodel.Coordinate(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                ),
            )
        )

    net = Network(
        nodes=nodes,
        edges=[],
        number_of_nodes=len(nodes),
    )
    net.label = "EEG standard_1005 fsaverage projection"
    net.description = (
        "MNE standard_1005 EEG sensors (MNI, mm) with fsaverage aparc projection gain matrix (sensors x regions)."
    )
    net.descriptor = "sensors"
    net.distance_unit = "mm"
    net.time_unit = "ms"
    net.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
        name="conduction_speed",
        label="v",
        value=3.0,
        unit="mm/ms",
    )

    net.set_matrix("gain", gain_matrix.astype(np.float32))

    for edge in net.edges:
        edge_label = getattr(edge, "label", None)
        if edge_label == "gain":
            edge.format = "dense"
            edge.weighted = True
            edge.valid_diagonal = False
            edge.non_negative = True
            edge.directed = False

    return net


def export_projection_network(output_yaml: Path, metadata_npz: Path) -> tuple[Path, Path, Path]:
    """Export projection network sidecar+companion and plotting metadata."""
    sensor_labels_tvb, sensor_labels_mne, sensor_pos_mm = _matched_sensor_labels_and_positions()
    region_labels, region_centers_mm, gain_matrix, fs_dir = _compute_region_gain(sensor_labels_mne)

    network = _build_tvbo_sensor_network(sensor_labels_tvb, sensor_pos_mm, gain_matrix)

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    network.save(output_yaml)

    np.savez_compressed(
        metadata_npz,
        region_labels=np.asarray(region_labels, dtype="U"),
        region_centers_mm=region_centers_mm.astype(np.float32),
        sensor_labels_tvb=np.asarray(sensor_labels_tvb, dtype="U"),
        sensor_labels_mne=np.asarray(sensor_labels_mne, dtype="U"),
    )

    return output_yaml, metadata_npz, fs_dir


def _load_head_surface(fs_dir: Path):
    """Load fsaverage head surface as Gifti mesh for overlay."""
    head_surfs = mne.read_bem_surfaces(str(fs_dir / "bem" / "fsaverage-head.fif"), verbose=False)
    head_surf = head_surfs[0]
    head_verts = np.asarray(head_surf["rr"], dtype=np.float32) * 1000.0
    head_tris = np.asarray(head_surf["tris"], dtype=np.int32)

    head_gifti = nib.gifti.GiftiImage()
    head_gifti.add_gifti_data_array(nib.gifti.GiftiDataArray(head_verts, intent="NIFTI_INTENT_POINTSET"))
    head_gifti.add_gifti_data_array(nib.gifti.GiftiDataArray(head_tris, intent="NIFTI_INTENT_TRIANGLE"))
    return head_gifti


def plot_saved_projection_network(
    output_yaml: Path,
    metadata_npz: Path,
    plot_png: Path,
    threshold_percentile: float,
    edge_radius: float,
    edge_scale_factor: float,
    edge_scale_mode: str,
    edge_clip_low: float,
    edge_clip_high: float,
    fs_dir: Path | None = None,
):
    """Load saved files and render the sensor-region graph with bsplot."""
    network = Network.from_file(output_yaml)
    gain = network.matrix("gain", format="dense")
    if gain is None:
        raise ValueError(f"No gain matrix found in {output_yaml}")

    meta = np.load(metadata_npz)
    region_labels = [str(x) for x in meta["region_labels"]]
    region_centers_mm = np.asarray(meta["region_centers_mm"], dtype=float)

    sensor_labels = [str(node.label) for node in network.nodes]
    sensor_pos_mm = np.asarray(
        [[node.position.x, node.position.y, node.position.z] for node in network.nodes],
        dtype=float,
    )

    if gain.shape != (len(sensor_labels), len(region_labels)):
        raise ValueError(
            "Gain matrix shape does not match saved sensor/region metadata: "
            f"{gain.shape} vs ({len(sensor_labels)}, {len(region_labels)})"
        )

    centers = {}
    node_types = {}

    for idx, name in enumerate(region_labels):
        key = f"R:{name}"
        centers[key] = tuple(region_centers_mm[idx])
        node_types[key] = "region"

    for idx, name in enumerate(sensor_labels):
        key = f"S:{name}"
        centers[key] = tuple(sensor_pos_mm[idx])
        node_types[key] = "sensor"

    n_regions = len(region_labels)
    n_sensors = len(sensor_labels)
    n_total = n_regions + n_sensors

    combined = np.zeros((n_total, n_total), dtype=float)
    for i in range(n_sensors):
        for j in range(n_regions):
            combined[n_regions + i, j] = gain[i, j]

    labels = [f"R:{name}" for name in region_labels] + [f"S:{name}" for name in sensor_labels]

    graph = create_network(
        centers,
        {"mne_forward_gain": combined},
        labels=labels,
        threshold_percentile=threshold_percentile,
        directed=True,
        edge_data_key="gain",
        node_types=node_types,
    )

    for node in graph.nodes():
        graph.nodes[node]["color"] = "gold" if node.startswith("R:") else "crimson"

    edge_gains = np.asarray(
        [data.get("gain", np.nan) for _, _, _, data in graph.edges(keys=True, data=True)],
        dtype=float,
    )
    edge_gains = edge_gains[np.isfinite(edge_gains)]
    edge_gains = edge_gains[edge_gains > 0]

    if edge_gains.size == 0:
        edge_vmin = 1e-12
        edge_vmax = 1.0
    else:
        edge_vmin = float(np.percentile(edge_gains, edge_clip_low))
        edge_vmax = float(np.percentile(edge_gains, edge_clip_high))
        if edge_vmax <= edge_vmin:
            edge_vmax = float(edge_gains.max())

    if fs_dir is None:
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    head_gifti = _load_head_surface(fs_dir)

    views = ["front", "top", "lateral", "posterior"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, view in zip(axes, views):
        plot_network_on_surface(
            graph,
            ax=ax,
            view=view,
            node_radius=1.5,
            node_color="auto",
            edge_radius=edge_radius,
            edge_color="auto",
            edge_cmap="viridis",
            edge_data_key="gain",
            edge_scale={"gain": edge_scale_factor, "mode": edge_scale_mode},
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            surface_alpha=0.12,
            nodes=list(graph.nodes()),
            extra_surfaces=[head_gifti],
            extra_colors=["wheat"],
            extra_alphas=[0.06],
        )
        ax.set_title(view)

    plt.tight_layout()
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    if edge_gains.size > 0:
        print(
            "Edge gain stats: "
            f"min={edge_gains.min():.2e}, "
            f"p50={np.percentile(edge_gains, 50):.2e}, "
            f"p95={np.percentile(edge_gains, 95):.2e}, "
            f"max={edge_gains.max():.2e}"
        )
    print(f"Saved plot: {plot_png}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=("Export an MNE standard_1005 EEG projection network in tvbo format and optionally plot it with bsplot.")
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output tvbo network sidecar path (.yaml)",
    )
    parser.add_argument(
        "--metadata-npz",
        type=Path,
        default=None,
        help="Output metadata NPZ path (default: <output-stem>_plotmeta.npz)",
    )
    parser.add_argument(
        "--plot-png",
        type=Path,
        default=None,
        help="Output plot PNG path (default: <output-stem>.png)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip forward-model export and only plot from existing saved files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only export files; do not render plot",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=80.0,
        help="Edge inclusion percentile for plotting graph construction",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.004,
        help="Base edge radius for rendering",
    )
    parser.add_argument(
        "--edge-scale-factor",
        type=float,
        default=12.0,
        help="Maximum multiplicative radius scaling factor",
    )
    parser.add_argument(
        "--edge-scale-mode",
        type=str,
        default="quantile",
        choices=["linear", "log", "exp", "quantile"],
        help="Radius scaling mode",
    )
    parser.add_argument(
        "--edge-clip-low",
        type=float,
        default=5.0,
        help="Lower percentile for edge colormap clipping",
    )
    parser.add_argument(
        "--edge-clip-high",
        type=float,
        default=95.0,
        help="Upper percentile for edge colormap clipping",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    output_yaml, metadata_npz, plot_png = _resolve_output_paths(
        args.output_yaml,
        args.metadata_npz,
        args.plot_png,
    )

    fs_dir: Path | None = None

    if not args.skip_export:
        output_yaml, metadata_npz, fs_dir = export_projection_network(output_yaml, metadata_npz)
        print(f"Saved network: {output_yaml}")
        print(f"Saved metadata: {metadata_npz}")
    else:
        if not output_yaml.exists():
            raise FileNotFoundError(f"Missing network sidecar: {output_yaml}")
        if not metadata_npz.exists():
            raise FileNotFoundError(f"Missing metadata NPZ: {metadata_npz}")

    if not args.no_plot:
        plot_saved_projection_network(
            output_yaml=output_yaml,
            metadata_npz=metadata_npz,
            plot_png=plot_png,
            threshold_percentile=args.threshold_percentile,
            edge_radius=args.edge_radius,
            edge_scale_factor=args.edge_scale_factor,
            edge_scale_mode=args.edge_scale_mode,
            edge_clip_low=args.edge_clip_low,
            edge_clip_high=args.edge_clip_high,
            fs_dir=fs_dir,
        )


if __name__ == "__main__":
    main()
