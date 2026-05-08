"""Plot native TVB and TVBO/tvboptim observation alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment


def build_experiment(duration: float) -> SimulationExperiment:
    dynamics = Dynamics.from_db("ReducedWongWang")
    dynamics.local_coupling_term = "local_coupling"
    dynamics.coupling_inputs.pop("local_coupling", None)

    network = Network.from_matrix(
        weights=np.array([[0.0, 0.2], [0.1, 0.0]], dtype=float),
        lengths=np.zeros((2, 2), dtype=float),
        labels=["left", "right"],
    )
    network.coupling["Linear"] = Coupling.from_db("Linear")

    return SimulationExperiment(
        dynamics=dynamics,
        network=network,
        integration={"method": "Heun", "duration": duration, "step_size": 0.1},
        observations=[
            Observation.from_db("TemporalAverage"),
            Observation.from_db("Bold_TVB"),
        ],
    )


def aligned_arrays(left, right) -> tuple[np.ndarray, np.ndarray]:
    left = drop_singleton_mode(left)
    right = drop_singleton_mode(right)

    if hasattr(left, "sizes") and hasattr(right, "sizes") and "time" in left.sizes and "time" in right.sizes:
        if left.sizes["time"] == right.sizes["time"] + 1:
            left = left.isel(time=slice(1, None))
        elif right.sizes["time"] == left.sizes["time"] + 1:
            right = right.isel(time=slice(1, None))
    else:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape[0] == right_array.shape[0] + 1:
            left = left_array[1:]
        elif right_array.shape[0] == left_array.shape[0] + 1:
            right = right_array[1:]

    return np.asarray(left), np.asarray(right)


def drop_singleton_mode(data):
    if hasattr(data, "dims") and "mode" in data.dims and data.sizes["mode"] == 1:
        return data.isel(mode=0)
    array = np.asarray(data)
    if array.ndim > 0 and array.shape[-1] == 1:
        return array[..., 0]
    return data


def first_variable(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        return data[:, 0, :]
    if data.ndim == 2:
        return data
    return data.reshape(data.shape[0], -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration in ms.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tvb_tvboptim_alignment.png"),
        help="Figure path to write.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = build_experiment(args.duration)
    tvboptim_result = experiment.run("tvboptim", quiet=True)
    tvb_result = experiment.run("tvb", quiet=True)

    temporal_tvboptim, temporal_tvb = aligned_arrays(
        tvboptim_result.integration.observations["TemporalAverage"].data,
        tvb_result.integration.observations["temporalaverage"].data,
    )
    bold_tvboptim, bold_tvb = aligned_arrays(
        tvboptim_result.integration.observations["BOLD_TVB"].data,
        tvb_result.integration.observations["bold_tvb"].data,
    )

    temporal_tvboptim = first_variable(temporal_tvboptim)
    temporal_tvb = first_variable(temporal_tvb)
    bold_tvboptim = first_variable(bold_tvboptim)
    bold_tvb = first_variable(bold_tvb)

    time = np.arange(temporal_tvb.shape[0], dtype=float)
    temporal_diff = np.abs(temporal_tvboptim - temporal_tvb)
    bold_diff = np.abs(bold_tvboptim - bold_tvb)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    node_labels = ["left", "right"]

    for node_index, label in enumerate(node_labels):
        axes[0, 0].plot(time, temporal_tvb[:, node_index], label=f"TVB {label}")
        axes[0, 0].plot(time, temporal_tvboptim[:, node_index], "--", label=f"TVBO tvboptim {label}")
    axes[0, 0].set_title("TemporalAverage traces")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("signal")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(time, temporal_diff.max(axis=1), color="black")
    axes[0, 1].set_title(f"TemporalAverage max abs diff: {temporal_diff.max():.3e}")
    axes[0, 1].set_xlabel("sample")
    axes[0, 1].set_ylabel("abs diff")

    x = np.arange(len(node_labels))
    width = 0.35
    axes[1, 0].bar(x - width / 2, bold_tvb[-1], width, label="TVB")
    axes[1, 0].bar(x + width / 2, bold_tvboptim[-1], width, label="TVBO tvboptim")
    axes[1, 0].set_title("BOLD final sample")
    axes[1, 0].set_xticks(x, node_labels)
    axes[1, 0].set_ylabel("BOLD")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x, bold_diff[-1], color="black")
    axes[1, 1].set_title(f"BOLD max abs diff: {bold_diff.max():.3e}")
    axes[1, 1].set_xticks(x, node_labels)
    axes[1, 1].set_ylabel("abs diff")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"saved {args.output}")
    print(f"TemporalAverage max abs diff: {temporal_diff.max():.6e}")
    print(f"BOLD max abs diff: {bold_diff.max():.6e}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()