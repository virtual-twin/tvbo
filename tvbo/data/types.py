from copy import deepcopy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation

from tvbo.knowledge.simulation import equations
from tvbo.utils import Bunch
from tvbo.data.tvbo_data.connectomes import Connectome
from tvbo.utils import format_pytree_as_string

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp


@register_pytree_node_class
class BaseTimeSeries:
    """
    Base time-series dataType.
    """

    def tree_flatten(self):
        # Keep network as a child (not metadata) to avoid non-hashable/array metadata
        # Store labels_dimensions in aux to preserve dimension labels across JAX transforms
        return (self.time, self.data, self.network), (
            self.title,
            self.sample_period,
            self.labels_dimensions,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # aux_data matches (__init__): title, sample_period, labels_dimensions
        return cls(*children, *aux_data)

    def __init__(
        self,
        time,
        data,
        network=None,
        title="TimeSeries",
        sample_period=None,
        labels_dimensions={},
    ):
        """
        labels_dimensions: Specific labels for each dimension for the data stored in this timeseries. A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }
        """
        # 1. Essential Data
        self.time = time
        self.data = data
        self.labels_dimensions = labels_dimensions

        # 2. Metadata
        self.title = title
        self.network = network

        # 3. Time Settings
        self.sample_period = self.dt = sample_period
        self.sample_period_unit = "ms"  # Default unit is milliseconds (ms)

        # 4. Internal Configurations
        self.labels_ordering = ("Time", "State Variable", "Space", "Mode")

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    # def __repr__(self):
    #     try:
    #         repr = f"{self.__class__.__name__}:\n├─ {self.data.shape[0] / self.sample_period} {self.sample_period_unit}\n├─ {self.data.shape[1]} State Variable{'s' if self.data.shape[1] > 1 else ''}\n├─ {self.data.shape[2]} Region{'s' if self.data.shape[2] > 1 else ''}\n└─ {self.data.shape[3]} Mode{'s' if self.data.shape[3] > 1 else ''}"
    #     except:
    #         repr = f"{self.__class__.__name__}:\nShape: {self.data.shape}"
    #     return repr
    def __repr__(self):
        return format_pytree_as_string(self, self.__class__.__name__, "", False, False)

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def space_labels(self):
        # Robustly handle legacy keys and bad types
        ld = self.labels_dimensions if isinstance(self.labels_dimensions, dict) else {}
        # Prefer canonical "Space" key; fall back to "Region" if present
        vals = ld.get(self.labels_ordering[2])
        if vals is None:
            vals = ld.get("Region", [])
        # Coerce single strings or scalars to a list
        if isinstance(vals, (str, bytes)):
            vals = [vals]
        elif vals is None:
            vals = []
        return np.array(list(vals))

    @property
    def variables_labels(self):
        ld = self.labels_dimensions if isinstance(self.labels_dimensions, dict) else {}
        vals = ld.get(self.labels_ordering[1], [])
        if isinstance(vals, (str, bytes)):
            vals = [vals]
        elif vals is None:
            vals = []
        return np.array(list(vals))

    @property
    def sample_rate(self):
        """:returns samples per second [Hz]"""
        if self.sample_period_unit in ("s", "sec"):
            return 1.0 / self.sample_period
        elif self.sample_period_unit in ("ms", "msec"):
            return 1000.0 / self.sample_period
        elif self.sample_period_unit in ("us", "usec"):
            return 1000000.0 / self.sample_period
        else:
            raise ValueError(f"{self.sample_period_unit} is not a recognized time unit")

    @property
    def sample_period_ms(self):
        """:returns sample_period is ms"""
        if self.sample_period_unit in ("s", "sec"):
            return 1000 * self.sample_period
        elif self.sample_period_unit in ("ms", "msec"):
            return self.sample_period
        elif self.sample_period_unit in ("us", "usec"):
            return self.sample_period / 1000.0
        else:
            raise ValueError(f"{self.sample_period_unit} is not a recognized time unit")

    def get_dt(self):
        return np.mean(np.diff(self.time)) if self.dt is None else self.dt

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {
            "Time-series type": self.__class__.__name__,
            "Time-series name": self.title,
            "Dimensions": self.labels_ordering,
            "Time units": self.sample_period_unit,
            "Sample period": self.sample_period,
            # "Start time": self.start_time,
            "Length": self.sample_period * self.data.shape[0],
        }
        return summary

    def _get_index_of_state_variable(self, sv_label):
        if sv_label not in self.variables_labels:
            raise IndexError(
                f"{sv_label} is not a state variable. Available state variables: {self.variables_labels}"
            )

        sv_index = np.where(self.variables_labels == sv_label)[0][0]
        return sv_index

    def get_state_variable(self, sv_label):
        sv_data = self.data[:, self._get_index_of_state_variable(sv_label), :, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[1]] = [sv_label]
        if sv_data.ndim == 3:
            sv_data = np.expand_dims(sv_data, 1)
        return self.duplicate(
            data=sv_data, labels_dimensions=subspace_labels_dimensions
        )

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            space_index = np.where(self.space_labels == label)[0][0]
            list_of_indices_for_labels.append(space_index)
        return list_of_indices_for_labels

    def get_subspace_by_index(self, list_of_index, **kwargs):
        self._check_space_indices(list_of_index)
        subspace_data = self.data[:, :, list_of_index, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = self.space_labels[
            list_of_index
        ].tolist()
        if subspace_data.ndim == 3:
            subspace_data = np.expand_dims(subspace_data, 2)
        return self.duplicate(
            data=subspace_data, labels_dimensions=subspace_labels_dimensions, **kwargs
        )

    def get_subspace_by_labels(self, list_of_labels):
        list_of_indices_for_labels = self._get_indices_for_labels(list_of_labels)
        return self.get_subspace_by_index(list_of_indices_for_labels)

    # def _get_index_for_slice_label(self, slice_label, slice_idx):
    #     if slice_idx == 1:
    #         return self._get_indices_for_labels([slice_label])[0]
    #     if slice_idx == 2:
    #         return self._get_index_of_state_variable(slice_label)

    def copy(self):
        """Return a deep copy of the current instance."""
        return deepcopy(self)

    # def duplicate(self, **kwargs):
    #     """Return a copy of the current instance with optional attribute updates."""
    #     duplicate = self.copy()  # Use self.copy() instead of super()
    #     for attr, value in kwargs.items():
    #         setattr(duplicate, attr, value)
    #     if hasattr(duplicate, "configure"):
    #         duplicate.configure()  # Call configure only if it exists
    #     return duplicate

    def duplicate(self, **kwargs):
        """
        Fast shallow-copy-based duplication with attribute update.
        """
        new = self.__class__(
            time=self.time,
            data=kwargs.get("data", self.data),
            network=self.network,
            title=self.title,
            sample_period=self.sample_period,
            labels_dimensions=kwargs.get(
                "labels_dimensions", self.labels_dimensions.copy()
            ),
        )
        return new


@register_pytree_node_class
class TimeSeries(BaseTimeSeries):
    def get_state_variable(self, sv_label):
        import math

        import sympy as sp

        exp = sp.parse_expr(sv_label, equations._clash1, evaluate=False)
        data = {}
        for s in exp.free_symbols:
            data[str(s)] = self.data[:, self._get_index_of_state_variable(str(s)), :, :]
        data.update({"math": math})
        sv_data = eval(sp.pycode(exp), data)
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[1]] = [sv_label]
        if sv_data.ndim == 3:
            sv_data = np.expand_dims(sv_data, 1)
        return self.duplicate(
            data=sv_data, labels_dimensions=subspace_labels_dimensions
        )

    def plot(self, ax=None, axis_labels=False, legend=True, **kwargs):
        if not ax:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        n_svar = self.data.shape[1] if len(self.data.shape) > 1 else 1
        uses_modes = len(self.data.shape) > 3 and self.data.shape[3] > 1
        if uses_modes:
            print("Plotting only first mode by default")

        # n_regions = self.data.shape[2]
        if "labels" in kwargs.keys():
            labels = kwargs.pop("labels")
        else:
            labels = [
                (
                    self.labels_dimensions["State Variable"][i]
                    if "State Variable" in self.labels_dimensions
                    else None
                )
                for i in range(n_svar)
            ]
        label = kwargs.pop("label", None)
        for i in range(n_svar):
            ax.plot(
                self.time,
                self.data[:, i, :, 0] if len(self.data.shape) > 1 else self.data,
                label=label or labels[i],
                **kwargs,
            )

        ax.set_xlabel(f"time [{self.time_unit}]")

        if n_svar == 1 and self.labels_dimensions:

            ylabel = (
                self.labels_dimensions.get("State Variable", ["X"])[0]
                if isinstance(self.labels_dimensions, dict)
                else "X"
            )
            ax.set_ylabel(ylabel)
            legend = False
        else:
            ax.set_ylabel("X")

        if axis_labels:  # ?
            ax.set_xlabel(self.time_unit)
        if legend and any(labels):
            ax.legend(loc="upper right", fontsize="smaller")
            handles, labels = ax.get_legend_handles_labels()
            unique = list(
                dict(zip(labels, handles)).items()
            )  # Keep only the last occurrence of each label
            ax.legend(
                [handle for _, handle in unique],
                [label for label, _ in unique],
                loc="upper right",
            )
        if return_fig:
            plt.close()
            return fig

    def plot_eeg(
        self,
        VOI: str | None = None,
        mode: int = 0,
        spacing: float | None = None,
        normalize: bool = False,
        channel_labels: bool = True,
        ax=None,
        linewidth: float = 0.5,
        **kwargs,
    ):
        """
        Plot each region as a separate channel stacked vertically on a single axes
        (EEG-like representation).

        Parameters
        ----------
        VOI : str | None
            Variable of interest to plot. If None and multiple variables exist,
            the first one is used.
        mode : int
            Mode index to select.
        spacing : float | None
            Vertical spacing between channels. If None, computed from data (median std).
        normalize : bool
            If True, z-score each channel before plotting.
        channel_labels : bool
            If True, add region labels at the channel offsets on the y-axis.
        ax : matplotlib.axes.Axes | None
            Axes to plot on. If None, a new figure and axes are created.
        color : str
            Line color for all channels.
        linewidth : float
            Line width for plotted channels.
        **kwargs : dict
            Additional kwargs forwarded to matplotlib plot.

        Returns
        -------
        matplotlib.figure.Figure | None
            Returns a figure if it creates one; otherwise None.
        """
        # Select variable of interest
        ts = self
        if self.data.shape[1] > 1:
            # Prefer requested VOI; otherwise use 'V' if present; else first
            if VOI is None:
                labels = list(self.variables_labels)
                if len(labels) == 0:
                    VOI = None
                elif "V" in labels:
                    VOI = "V"
                else:
                    VOI = labels[0]
            if VOI is not None:
                ts = self.get_state_variable(VOI)

        # Extract 2D array (time, regions) for the chosen mode
        X = ts.data[:, 0, :, mode]
        X = np.asarray(X)
        t = np.asarray(ts.time)

        # Optional normalization per channel
        if normalize:
            mu = X.mean(axis=0, keepdims=True)
            sigma = X.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
            X = (X - mu) / sigma

        # Determine spacing
        if spacing is None:
            # Robust spacing based on median std or max abs if std is zero
            stds = np.std(X, axis=0)
            base = np.median(stds)
            if not np.isfinite(base) or base == 0:
                base = np.median(np.max(np.abs(X), axis=0))
            if not np.isfinite(base) or base == 0:
                base = 1.0
            spacing = 2.5 * float(base)

        # Determine number of regions and label characteristics (for layout)
        n_regions = X.shape[1]
        labels_array = self.space_labels
        labels_list = (
            [str(l) for l in labels_array]
            if labels_array is not None and len(labels_array) == n_regions
            else [str(i) for i in range(n_regions)]
        )
        max_label_len = max((len(l) for l in labels_list), default=1)

        # Prepare axes with adaptive figure size and left margin when creating a new figure
        created_fig = False
        if ax is None:
            # Height scales with number of channels; clamp between 4 and 20 inches
            per_channel_in = 0.22  # inches per channel
            height = min(20.0, max(4.0, per_channel_in * n_regions))
            width = 10.0
            fig, ax = plt.subplots(figsize=(width, height))
            # Left margin fraction scales with max label length; clamp sensibly
            left_frac = min(0.5, max(0.1, 0.006 * max_label_len))
            fig.subplots_adjust(left=left_frac)
            created_fig = True

        # Plot each region with vertical offset
        offsets = np.arange(n_regions) * spacing
        for i in range(n_regions):
            ax.plot(t, X[:, i] + offsets[i], linewidth=linewidth, **kwargs)

        # Configure axes
        ax.set_xlabel(f"time [{ts.time_unit}]")
        if channel_labels:
            ax.set_yticks(offsets)
            ax.set_yticklabels(labels_list)
            ax.tick_params(axis="y", labelsize=8)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_xlim(t[0], t[-1])
        ax.set_title("EEG-like regional channels" + (f" — {VOI}" if VOI else ""))
        ax.grid(False)

        if created_fig:
            plt.close()
            return fig

    def cut_transient(self, start_time):
        start_index = jnp.searchsorted(self.time, start_time, side="left")

        ts_cut = deepcopy(self)
        ts_cut.time = self.time[start_index:]
        ts_cut.data = self.data[start_index:]
        ts_cut.start_time = self.time[start_index]
        return ts_cut

    def subset(self, start, end):
        start_index = np.searchsorted(self.time, start, side="left")
        end_index = np.searchsorted(self.time, end, side="right")

        ts_subset = deepcopy(self)
        ts_subset.time = self.time[start_index:end_index]
        ts_subset.data = self.data[start_index:end_index]
        return ts_subset

    def exclude_region(self, region):
        if isinstance(region, int):
            region_index = region
        else:
            region_index = self.get_region_index(region)
        data = np.delete(self.data, region_index, axis=2)
        labels_dimensions = deepcopy(self.labels_dimensions)
        if "Region" in labels_dimensions:
            labels_dimensions["Region"].remove(region)
        return self.duplicate(data=data, labels_dimensions=labels_dimensions)

    def calculate_frequency(self, state_variable=None, region=0, mode=0):
        """
        Calculate the dominant frequency of the time series data using FFT.

        Returns:
            float: Dominant frequency in Hz.
        """
        ts = self
        if ts.data.shape[1] > 1:
            ts = self.get_state_variable(state_variable)
        data = ts.data[:, 0, region, mode]
        if data.ndim != 1:
            raise ValueError("Data must be one-dimensional to calculate frequency.")
        if not hasattr(self, "time"):
            raise AttributeError(
                "Time information is missing in the TimeSeries object."
            )

        sampling_interval = ts.sample_period_ms / 1000
        fft_result = np.fft.fft(data)
        fft_amplitude = np.abs(fft_result)  # Magnitude of the FFT
        fft_freqs = np.fft.fftfreq(len(data), d=sampling_interval)

        # Only consider positive frequencies
        positive_freqs = fft_freqs[fft_freqs >= 0]
        positive_amplitudes = fft_amplitude[fft_freqs >= 0]

        # Find the dominant frequency
        dominant_frequency = positive_freqs[np.argmax(positive_amplitudes)]

        return dominant_frequency

    def compute_normalised_average_power(self, VOI=None):
        """
        Compute normalized average power spectrum using FFT.
        
        Parameters
        ----------
        VOI : str, optional
            Variable of interest to analyze. Required if multiple state variables exist.
            
        Returns
        -------
        frequency : ndarray
            Frequency values in Hz
        power : ndarray
            Normalized average power values
        """
        from scipy.fft import fft, fftfreq

        # Select variable of interest
        if len(self.labels_dimensions["State Variable"]) == 1:
            ts = self
        elif len(self.labels_dimensions["State Variable"]) > 1 and VOI:
            ts = self.get_state_variable(VOI)
        else:
            raise ValueError(
                f"select variable of interest (VOI) from {self.labels_dimensions['State Variable']}"
            )

        # Get data and compute FFT
        data = ts.data
        dt = ts.sample_period_ms / 1000  # Convert to seconds
        n_samples = data.shape[0]

        # Compute FFT for positive frequencies only
        fft_result = fft(data, axis=0)
        fft_power = np.abs(fft_result) ** 2
        frequency = fftfreq(n_samples, d=dt)

        # Take only positive frequencies
        positive_mask = frequency >= 0
        frequency = frequency[positive_mask]
        fft_power = fft_power[positive_mask]

        # Average over regions and modes, normalize
        power = fft_power.mean(axis=(1, 2))  # Average over state vars and regions
        power = power / power.sum()  # Normalize

        return frequency, power

    def compute_dt(self):
        dt = np.diff(self.time)
        mean_dt = np.mean(dt)
        if self.sample_period != mean_dt:
            print(
                "Warning: Sample period does not match mean dt. Setting sample period to mean dt."
            )
            self.sample_period = mean_dt

    def plot_power_spectrum(
        self,
        VOI=None,
        ROI="mean",
        mode=0,
        bands=None,
        colors=None,
        ax=None,
        label="simulation",
        **kwargs,
    ):
        """
        Plot the power spectrum with normalized average power computed via FFT.

        Parameters:
        - VOI: Variable of Interest, typically selecting subsets of data.
        - ROI: Region of Interest ("mean" or index).
        - mode: Mode index for selecting data.
        - bands: Dictionary of frequency bands to highlight.
        - colors: Custom colors for frequency bands.
        - ax: Matplotlib Axes object to plot on.
        - label: Label for the plot.
        - kwargs: Additional plotting arguments.

        Returns:
        - Matplotlib figure if ax is None, otherwise None.
        """
        from scipy.fft import fft, fftfreq
        from matplotlib import colormaps

        # Extract data
        data = self.data if VOI is None else self.get_state_variable(VOI).data

        # Compute FFT
        dt = self.dt / 1000
        n_samples = data.shape[0]
        frequency = fftfreq(n_samples, d=dt)[: n_samples // 2]
        power = np.abs(fft(data, axis=0)[: n_samples // 2]) ** 2

        # Normalize power
        power /= power.sum(axis=0, keepdims=True)

        # Select mode and aggregate over ROI
        power = power[:, :, mode]
        power = power.mean(axis=2) if ROI == "mean" else power[:, ROI]

        # Set up the plot
        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        for i in range(power.shape[1]):
            label = self.labels_dimensions["State Variable"][i]
            ax.plot(frequency, power[:, i], linewidth=1, label=label, **kwargs)
        ax.legend()
        ax.set_xlim([1, 150])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Power")
        ax.set_title("Power Spectrum")

        # Highlight frequency bands
        if bands is None:
            bands = {
                r"$\delta$": (1, 4),
                r"$\theta$": (4, 8),
                r"$\alpha$": (8, 12),
                r"$\beta$": (12, 30),
                r"$\gamma$": (30, 100),
            }
        if colors is None:
            colors = colormaps["viridis"](np.linspace(0, 1, len(bands)))

        ylim = ax.get_ylim()
        for i, (band, (start, end)) in enumerate(bands.items()):
            mid_point = 10 ** (np.log10(start) + (np.log10(end) - np.log10(start)) / 2)
            ax.axvspan(start, end, color=colors[i], alpha=0.1)
            ax.axvline(x=end, color=colors[i], linestyle="--")
            ax.text(
                mid_point,
                ylim[1] * 0.8,
                band,
                ha="center",
                va="top",
                color="k",
                fontsize=12,
                fontweight="bold",
            )

        if return_fig:
            plt.close()
            return fig

    def check_identity(self, other, select_state_variable=None):
        if isinstance(other, np.ndarray):
            data = other
        elif isinstance(other, TimeSeries):
            data = other.data

        return np.allclose(
            data.ravel(),
            (
                self.data.ravel()
                if select_state_variable is None
                else self.get_state_variable(select_state_variable).data.ravel()
            ),
            atol=1e-8,
        )

    def get_region_index(self, region_label):
        return list(self.labels_dimensions["Region"]).index(region_label)

    def get_region(self, region_label):
        region_index = self.get_region_index(region_label)
        roi_data = self.data[:, :, region_index : region_index + 1, :]

        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = [region_label]

        return self.duplicate(
            data=roi_data, labels_dimensions=subspace_labels_dimensions
        )


class TimeSeriesRegion(TimeSeries):
    """A time-series associated with the regions of a network."""

    # network = Attr(field_type=network.Connectivity)
    # region_mapping_volume = Attr(
    #     field_type=region_mapping.RegionVolumeMapping, required=False
    # )
    # region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    # labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesRegion, self).summary_info()
        summary.update(
            {
                "Source Connectivity": self.network.title,
                "Region Mapping": (
                    self.region_mapping.title if self.region_mapping else "None"
                ),
                "Region Mapping Volume": (
                    self.region_mapping_volume.title
                    if self.region_mapping_volume
                    else "None"
                ),
            }
        )
        return summary

    def animate_time_series(
        ts,
        plane="sagittal",
        state=0,
        mode=0,
        interval=100,
        window_dt=1000,
        cmap="viridis",
        node_size=100,
        line_kwargs={},
    ):
        """
        Creates an animated 2D scatter plot from a 4D time-series object,
        with a second axis for the time-series progression showing all regions.

        Parameters:
        - ts: Time series object with `ts.time`, `ts.data`, and `ts.network.centres`.
            `ts.data` has shape (time, state, region, mode).
        - plane: Projection plane ('sagittal', 'horizontal', 'axial').
        - state: Index of the state to select or None to aggregate across states.
        - mode: Index of the mode to select or None to aggregate across modes.
        - interval: Time interval between frames in milliseconds.
        - aggregation: Aggregation method ('mean', 'sum') if state or mode is None.
        """
        # Map plane to coordinates
        if plane == "sagittal":
            x, y = ts.network.centres[:, 1], ts.network.centres[:, 2]
        elif plane == "horizontal":
            x, y = ts.network.centres[:, 0], ts.network.centres[:, 1]
        elif plane == "axial":
            x, y = ts.network.centres[:, 0], ts.network.centres[:, 2]
        else:
            raise ValueError(
                "Invalid plane. Choose from 'sagittal', 'horizontal', 'axial'."
            )

        # Prepare data based on state and mode selection or aggregation
        data = ts.data
        data = data[:, state, :, mode][::window_dt]  # Fix state
        data = (data - np.min(data)) / (
            np.max(data) - np.min(data)
        )  # Normalize to [0, 1]

        time = ts.time[::window_dt]
        n_regions = data.shape[1]

        # Initialize figure and axes
        fig, (ax, ax_ts) = plt.subplots(1, 2, layout="compressed", figsize=(8, 4))
        sc = ax.scatter(x, y, c=data[0], cmap="viridis", s=node_size, vmin=0, vmax=1)

        ax.set_title(f"Time: {time[0]:.2f}")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="Data Intensity", shrink=0.5)

        # Create evenly spaced colors from the viridis colormap
        colors = colormaps[cmap](np.linspace(0, 1, n_regions))

        # Initialize the time series plot for all regions
        lines = []
        for i, color in enumerate(colors):
            (line,) = ax_ts.plot(
                [], [], color=color, label=f"Region {i+1}", **line_kwargs
            )
            lines.append(line)
        (avg_line,) = ax_ts.plot([], [], color="red", linewidth=2, label="Average")

        ax_ts.set_xlim(time[0], time[-1])
        ax_ts.set_ylim(0, 1.1)
        ax_ts.set_title("Time-Series Progression")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Intensity")

        # Update function for animation
        def update(frame):
            sc.set_array(data[frame])
            ax.set_title(f"Time: {time[frame]:.2f}")
            for i, line in enumerate(lines):
                line.set_xdata(time[: frame + 1])  # Update X data for each region
                line.set_ydata(data[: frame + 1, i])  # Update Y data for each region
            avg_line.set_xdata(time[: frame + 1])  # Update X data for average
            avg_line.set_ydata(
                data[: frame + 1].mean(axis=1)
            )  # Update Y data for average
            return [sc] + lines + [avg_line]

        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(time), interval=interval, blit=False
        )

        plt.close()
        return ani


class SimulationResult(Bunch):
    def __init__(self):
        self.monitors = []

    def add_timeseries(self, monitor_name, timeseries):
        setattr(self, monitor_name, timeseries)
        self.monitors.append(monitor_name)
        pass


@register_pytree_node_class
class SimulationState:
    def __init__(
        self,
        initial_conditions: TimeSeries,
        network: Connectome,
        dt,
        noise,
        parameters,
        stimulus,
        monitor_parameters,
        nt,
    ):
        self.initial_conditions = initial_conditions
        self.network = network
        self.dt = dt
        self.noise = noise
        self.parameters = parameters
        self.stimulus = stimulus
        self.monitor_parameters = monitor_parameters
        self.nt = nt

    def tree_flatten(self):
        # Make `noise` a child so fields like sigma_vec can participate in vmap batching.
        # Keep `nt` static (aux) to ensure it remains a concrete value under jit/vmap
        # because we use it in shape/length contexts like jnp.arange(0, nt).
        children = (
            self.initial_conditions,
            self.network,
            self.dt,
            self.noise,
            self.parameters,
            self.stimulus,
            self.monitor_parameters,
        )
        aux = (self.nt,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct in the original __init__ order
        (
            initial_conditions,
            network,
            dt,
            noise,
            parameters,
            stimulus,
            monitor_parameters,
        ) = children
        (nt,) = aux_data if isinstance(aux_data, tuple) else (aux_data,)
        return cls(
            initial_conditions,
            network,
            dt,
            noise,
            parameters,
            stimulus,
            monitor_parameters,
            nt,
        )

    def __repr__(self):
        """
        Returns a string representation of the SimulationState object.
        Shows all fields in the pytree structure.
        """
        return format_pytree_as_string(self, "SimulationState", "", False, False)

    # ---------------- Convenience: noise sigma helpers ----------------
    @property
    def n_state_variables(self) -> int:
        try:
            # initial_conditions: (H, S, R, M) or (T, S, R, M)
            return int(self.initial_conditions.data.shape[1])
        except Exception:
            return 0

    @property
    def state_variable_names(self):
        # Prefer names explicitly attached by exporter
        if hasattr(self, "_svar_names") and isinstance(self._svar_names, (list, tuple)):
            return list(self._svar_names)
        # Fallback to labels on initial conditions if present
        try:
            ld = getattr(self.initial_conditions, "labels_dimensions", {}) or {}
            names = ld.get("State Variable", None)
            if names:
                return list(names)
        except Exception:
            pass
        return [str(i) for i in range(self.n_state_variables)]

    def _ensure_noise_holder(self):
        if getattr(self, "noise", None) is None:
            # Lightweight holder with default seed and sigma_vec
            class _N:  # noqa: N801 - internal simple holder
                def __init__(self):
                    self.seed = 0
                    self.sigma_vec = None

            self.noise = _N()
        if getattr(self.noise, "sigma_vec", None) is None:
            import jax.numpy as jnp

            self.noise.sigma_vec = jnp.zeros(
                (self.n_state_variables,), dtype=jnp.asarray(self.dt).dtype
            )
        return self.noise

    def get_state_variable_index(self, name_or_index) -> int:
        if isinstance(name_or_index, int):
            return int(name_or_index)
        names = self.state_variable_names
        try:
            return int(names.index(str(name_or_index)))
        except Exception:
            return 0

    def set_sigma_for(self, name_or_index, value):
        import jax.numpy as jnp

        idx = self.get_state_variable_index(name_or_index)
        noise = self._ensure_noise_holder()
        # Rebuild sigma_vec to avoid in-place mutation issues
        sv = jnp.zeros((self.n_state_variables,), dtype=jnp.asarray(self.dt).dtype)
        if isinstance(value, (list, tuple)):
            # Allow list to set all values directly
            arr = jnp.asarray(value, dtype=sv.dtype)
            if arr.shape[0] != sv.shape[0]:
                raise ValueError(
                    "Length of sigma list must match number of state variables"
                )
            sv = arr
        else:
            sv = sv.at[idx].set(jnp.asarray(value, dtype=sv.dtype))
        noise.sigma_vec = sv
        return self

    def set_sigma_many(self, mapping: dict):
        """Set multiple sigma values using a dict: { 'V': 0.02, 'W': 0.0 }"""
        import jax.numpy as jnp

        noise = self._ensure_noise_holder()
        sv = jnp.zeros((self.n_state_variables,), dtype=jnp.asarray(self.dt).dtype)
        names = self.state_variable_names
        for k, v in (mapping or {}).items():
            idx = self.get_state_variable_index(k)
            sv = sv.at[idx].set(jnp.asarray(v, dtype=sv.dtype))
        noise.sigma_vec = sv
        return self

    class _NoiseSetter:
        def __init__(self, state, index):
            self._state = state
            self._index = int(index)

        @property
        def sigma(self):
            n = self._state._ensure_noise_holder()
            import numpy as _np

            sv = (
                _np.asarray(n.sigma_vec)
                if getattr(n, "sigma_vec", None) is not None
                else None
            )
            if sv is None:
                return 0.0
            return float(sv[self._index])

        @sigma.setter
        def sigma(self, value):
            self._state.set_sigma_for(self._index, value)

    class _StateVarProxy:
        def __init__(self, state, index):
            self._state = state
            self._index = int(index)
            self.noise = SimulationState._NoiseSetter(state, index)

    class _StateVariablesProxy:
        def __init__(self, state):
            self._state = state
            self._names = state.state_variable_names

        def __getattr__(self, name):
            idx = self._state.get_state_variable_index(name)
            return SimulationState._StateVarProxy(self._state, idx)

        def __getitem__(self, key):
            idx = self._state.get_state_variable_index(key)
            return SimulationState._StateVarProxy(self._state, idx)

    @property
    def state_variables(self):
        """Ergonomic proxy: state.state_variables.V.noise.sigma = 0.02

        This updates state.noise.sigma_vec appropriately. Safe to use before jit/vmap.
        """
        return SimulationState._StateVariablesProxy(self)

    def convert_dtype(self, target_dtype=jnp.float32):
        """
        Convert the dtype of the parameter pytree.

        Useful for converting between 32 and 64 bit types.

        Parameters
        ----------
        pytree : pytree
            The parameter tree whose dtype needs to be converted.
        target_dtype : jnp.dtype, optional
            The target dtype to convert to. Defaults to jnp.float32.

        Returns
        -------
        converted_pytree : pytree
            The parameter tree with converted dtype.

        Notes
        -----
        This method recursively traverses the pytree structure and converts all leaf nodes to the specified target dtype.
        It preserves the overall structure of the pytree while changing the dtype of its elements.
        """

        def get_int_dtype(float_dtype):
            return jnp.int32 if float_dtype == jnp.float32 else jnp.int64

        int_dtype = get_int_dtype(target_dtype)

        def convert_leaf(x):
            if isinstance(x, (jax.Array, np.ndarray)):
                if np.issubdtype(x.dtype, np.integer):
                    return jnp.array(x, dtype=int_dtype)
                else:
                    return jnp.array(x, dtype=target_dtype)
            elif isinstance(x, float):
                return jnp.array(x, dtype=target_dtype)
            elif isinstance(x, int):
                return x
            else:
                return x  # Keep other types unchanged

        return jax.tree_util.tree_map(convert_leaf, self)
