"""Runtime data types for TVBO simulations.

Provides `TimeSeries`, a JAX-pytree-aware, xarray-backed time-series container
with domain-specific analysis and visualization helpers, and `SimulationState`,
the bundled simulation state (initial conditions, network, noise, parameters,
stimulus, and monitor settings) handed to the integration backends.
"""

import copy
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation

import xarray as xr
import tvbo.jax.xarray_pytrees  # noqa: F401 – registers xr types as JAX pytrees

from tvbo.classes import equation as equations
from tvbo.utils import Bunch
from tvbo.classes.network import Network
from tvbo.utils import format_pytree_as_string

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _to_dataarray(raw_data, raw_time=None, state_names=None, nodes=None):
    """Convert raw ndarray + metadata to xr.DataArray.

    Parameters
    ----------
    raw_data : array-like or None
        Raw simulation data (2D, 3D, or 4D).
    raw_time : array-like or None
        Time coordinate values.
    state_names : list[str] or None
        State variable names for the 'variable' coordinate.

    Returns
    -------
    xr.DataArray or None
    """
    if raw_data is None:
        return None
    if isinstance(raw_data, xr.DataArray):
        return raw_data
    data_np = np.asarray(raw_data)
    all_dims = ["time", "variable", "node", "mode"]
    dims = all_dims[: data_np.ndim]
    # node and mode dims only exist when they actually carry information (size > 1).
    # Trailing singleton dims are meaningless — drop them so selection always
    # yields a predictable shape without any downstream squeeze() calls.
    while len(dims) > 2 and dims[-1] in ("node", "mode") and data_np.shape[len(dims) - 1] == 1:
        data_np = data_np[..., 0]
        dims = dims[:-1]
    coords = {}
    if raw_time is not None:
        coords["time"] = np.asarray(raw_time)
    if state_names:
        var_axis = dims.index("variable") if "variable" in dims else None
        if var_axis is not None and len(state_names) == data_np.shape[var_axis]:
            coords["variable"] = list(state_names)
    if "mode" in dims:
        coords["mode"] = list(range(data_np.shape[dims.index("mode")]))
    if "node" in dims:
        n_nodes = data_np.shape[dims.index("node")]
        if nodes and len(nodes) == n_nodes:
            coords["node"] = [str(n) for n in nodes]
        else:
            coords["node"] = list(range(n_nodes))
    return xr.DataArray(data=data_np, dims=dims, coords=coords)


def _unwrap_observation(obs):
    """An observation's array, whatever wrapper it arrived in.

    ``.data`` unwraps an ``ObservationResult``, but xarray spells its raw buffer the same
    way, so the same expression would strip the dims a labelled observation carries. Every
    site that reaches for an observation's array goes through here.
    """
    return obs if isinstance(obs, xr.DataArray) else getattr(obs, "data", obs)


def _observation_dataarray(raw_data, dims=None, nodes=None):
    """Attach an observation's DECLARED axis names to the array the backend returned.

    The axes are not inferred here. An observation's output shape is fixed by the
    reduction that produced it, so codegen emits the names alongside the reducer
    (``_STREAMING_DIMS``, from ``utils.reduction_dims``) and this only binds them —
    together with the network's node labels, which the caller already holds. Inferring
    dims from shape instead cannot tell an ``(n_freq, n_node)`` spectrum from an
    ``(n_node, n_node)`` matrix, so nothing here guesses: an observation with no declared
    dims is passed through unlabelled and the container falls back to positional names.

    Returns the input untouched when it is already labelled, is not a numeric array, or
    carries no dims declaration of the right rank.
    """
    if raw_data is None or isinstance(raw_data, xr.DataArray) or not dims:
        return raw_data
    try:
        a = np.asarray(raw_data)
    except (ValueError, TypeError):
        return raw_data
    if a.dtype == object or a.ndim == 0 or a.size == 0 or a.ndim != len(dims):
        return raw_data

    dims = [str(d) for d in dims]
    coords = {}
    if nodes:
        labels = [str(n) for n in nodes]
        coords = {d: labels for i, d in enumerate(dims) if d in ("node", "node_j") and a.shape[i] == len(labels)}
    return xr.DataArray(a, dims=dims, coords=coords)


def _inner_dims(post_trial_shape, ts_arr, declared=None):
    """Axis names for one exploration cell's payload, and the coords they carry.

    A DECLARED shape wins outright. An observation's axes come from the reduction it
    declares — a stride keeps ``(time, node)``, a co-moment gives ``(node, node_j)``, a
    recurrence gives ``(node,)`` — and are known at codegen. Falling back to matching
    lengths against a positional ``(time, variable, node, mode)`` template is how a
    1,338-frame time axis ends up named ``node``: silently, with every downstream
    selection then keyed on the wrong axis.

    The template remains the fallback for payloads that declare nothing (a raw swept
    trajectory, an observable a backend returns unlabelled).
    """
    n = len(post_trial_shape)
    coords = {}
    if declared is not None and len(declared) == n:
        dims = [str(d) for d in declared]
        if ts_arr is not None and "time" in dims and ts_arr.size == post_trial_shape[dims.index("time")]:
            coords["time"] = ts_arr
        return dims, coords

    template = ["variable", "node", "mode"]
    if ts_arr is not None and ts_arr.size > 1 and n > 0 and ts_arr.size == post_trial_shape[0]:
        dims = ["time"] + [template[i] if i < len(template) else f"dim_{i}" for i in range(n - 1)]
        coords["time"] = ts_arr
    else:
        # Time-aggregated (or no time) — assume spatial layout, right-aligned.
        dims = template[-n:] if n else []
    return dims, coords


def _axis_size(ax) -> int | None:
    """An axis's declared cell count, however the producer shaped it.

    Handles both axis shapes the module accepts (dict or attribute) and falls back to ``len(explored_values)`` for an axis carrying values but no ``n``. Returns ``None`` when neither is available, which callers must distinguish from a real size: an unknown size makes grid completeness undecidable rather than zero.
    """
    n = ax.get("n") if isinstance(ax, dict) else getattr(ax, "n", None)
    if n is not None:
        return int(n)
    vals = ax.get("explored_values") if isinstance(ax, dict) else getattr(ax, "explored_values", None)
    return None if vals is None else int(np.asarray(vals).size)


def _is_partial_shard(expl) -> bool:
    """Whether *expl* holds one HPC array task's slice rather than a whole sweep.

    Prefers the producer's own answer: the generated script has ``kwargs['shard']`` in hand and records it as ``is_shard``. Nothing downstream can re-derive that as reliably, so a declared value always wins.

    The count-based fallback exists for results built before that slot, and asks whether the cells present are fewer than the Cartesian product of the axes. It is genuinely partial — it cannot see a *branch* shard, whose axis ``n`` is taken from the already-sliced index so the slice looks complete — which is exactly why the declared value is preferred rather than merely consulted.

    What it must never do is read ``cell_coords`` as the marker: that field is present on EVERY keyed sweep, sharded or not (see :func:`_stacked_to_dataarray`), and reading it that way silently cost every local sweep its provenance sidecar.

    Undecidable cases count as *not* a shard. One redundant sidecar beside a shard is recoverable; dropping the sidecar of a full run breaks the self-describing contract ``save`` advertises, and does so without a word.
    """
    declared = getattr(expl, "is_shard", None)
    if declared is not None:
        return bool(declared)
    cell_coords = getattr(expl, "cell_coords", None)
    if not cell_coords:
        return False
    sizes = [_axis_size(ax) for ax in (getattr(expl, "axes", None) or [])]
    if not sizes or any(s is None or s <= 0 for s in sizes):
        return False
    counts = [int(np.asarray(v).shape[0]) for v in cell_coords.values() if np.asarray(v).ndim]
    return bool(counts) and max(counts) < int(np.prod(sizes))


def _stacked_to_dataarray(stacked_arr, axes_info, intrinsic_ts=None, n_trials=1, name=None, cell_coords=None, dims=None):
    """Build an ``xr.DataArray`` from a parameter-grid-stacked array.

    Outer dims correspond to exploration axes (parameter names with their
    explored values as coords). When ``n_trials > 1`` and the leading inner
    axis matches, a ``trial`` dim is inserted after the grid dims. Remaining
    inner dims follow the simulation convention ``(time, variable, node,
    mode)``; the leading ``time`` dim is included only when ``intrinsic_ts``
    carries a multi-step time vector matching the leading remaining shape,
    so time-aggregated observations don't get a spurious ``time`` axis.

    ``cell_coords`` (``{axis_name: per_cell_values}``) is each cell's actual parameter
    values read back from the grid, in the grid's OWN emission order. It keys results by
    value rather than by position, because a ``Space`` emits cells in pytree-leaf order,
    which is NOT the ``axes_info`` order whenever the swept axes live on different state
    sub-objects (dynamics/coupling/graph) — a plain positional reshape would then scramble
    the surface. For the full Cartesian product each cell is placed into the rectangular
    grid at the index its values map to (order-independent). For a flat subset (one HPC
    array task's shard) the result instead gets a single ``point`` dim with each axis's
    value hung on it as a coordinate, so the shard is self-describing and reassembles by
    parameter value across tasks.

    ``dims`` are the payload's DECLARED per-cell axis names; supply them whenever the
    spec knows them (see :func:`_inner_dims`).
    """
    if stacked_arr is None:
        return None
    arr = np.asarray(stacked_arr)
    grid_dims = []
    grid_coords = {}
    grid_sizes = []
    for ax in axes_info:
        ax_name = ax.get("name") if isinstance(ax, dict) else getattr(ax, "name", None)
        if not ax_name:
            continue
        ax_vals = ax.get("explored_values") if isinstance(ax, dict) else getattr(ax, "explored_values", None)
        ax_n = _axis_size(ax)
        grid_dims.append(ax_name)
        grid_sizes.append(int(ax_n) if ax_n is not None else None)
        if ax_vals is not None:
            grid_coords[ax_name] = np.asarray(ax_vals)

    # Sharded / non-rectangular subset: the leading axis is a flat list of grid
    # points, not the full Cartesian product, so it cannot be reshaped into one
    # dim per parameter. Emit a single ``point`` dim and hang each axis's
    # per-cell value on it as a (non-dimension) coordinate.
    _full_grid = bool(grid_sizes) and all(s is not None for s in grid_sizes) and arr.shape[0] == int(np.prod(grid_sizes))
    # Full rectangular grid with per-cell coords: place each cell BY VALUE (see docstring).
    if cell_coords is not None and _full_grid and grid_dims:
        try:
            _pos = [
                np.abs(np.asarray(cell_coords[_n])[:, None] - np.asarray(grid_coords[_n])[None, :]).argmin(axis=1)
                for _n in grid_dims
            ]
            _flat_idx = np.ravel_multi_index(tuple(_pos), tuple(grid_sizes))
            if len(np.unique(_flat_idx)) == arr.shape[0]:
                _rect = np.empty((int(np.prod(grid_sizes)),) + arr.shape[1:], dtype=arr.dtype)
                _rect[_flat_idx] = arr
                arr = _rect.reshape(tuple(grid_sizes) + arr.shape[1:])
        except (KeyError, ValueError, TypeError):
            # TypeError included: placement subtracts coordinates, which a non-numeric axis
            # (`integration.method` over "heun"/"euler") cannot do. Unmatchable either way,
            # so fall through to the positional reshape rather than out of `as_grid`.
            pass
        cell_coords = None  # consumed: build the rectangular DataArray from grid_coords
    if cell_coords is not None or (grid_dims and not _full_grid):
        n_points = arr.shape[0]
        inner_shape = arr.shape[1:]
        coords = {}
        for k, v in (cell_coords or grid_coords).items():
            vv = np.asarray(v)
            if vv.ndim == 1 and vv.shape[0] == n_points:
                coords[k] = ("point", vv)
        has_trial = n_trials > 1 and len(inner_shape) > 0 and inner_shape[0] == n_trials
        if has_trial:
            trial_dims = ["trial"]
            coords["trial"] = np.arange(n_trials)
            post_trial_shape = inner_shape[1:]
        else:
            trial_dims = []
            post_trial_shape = inner_shape
        ts_arr = None
        if intrinsic_ts is not None:
            ts_arr = np.asarray(intrinsic_ts)
            while ts_arr.ndim > 1:
                ts_arr = ts_arr[0]
        inner_dims, inner_coords = _inner_dims(post_trial_shape, ts_arr, dims)
        coords.update(inner_coords)
        while inner_dims and arr.shape[-1] == 1 and inner_dims != list(dims or []):
            arr = arr[..., 0]
            inner_dims = inner_dims[:-1]
        all_dims = ["point"] + trial_dims + inner_dims
        return xr.DataArray(data=arr, dims=all_dims, coords=coords, name=name)

    # Multi-axis 'product'-mode explorations come back with a flat leading
    # dim of size prod(grid_sizes). Reshape into per-axis dims so the
    # DataArray gets one named axis per parameter.
    if (
        len(grid_dims) > 1
        and arr.ndim >= 1
        and all(s is not None for s in grid_sizes)
        and arr.shape[0] == int(np.prod(grid_sizes))
    ):
        arr = arr.reshape(tuple(grid_sizes) + arr.shape[1:])

    n_grid = len(grid_dims)
    inner_shape = arr.shape[n_grid:]

    # Trials-only explorations have no grid axes but still get a synthetic
    # leading axis from stacking a single observable_fn call. Collapse it so
    # the trial dim sits where downstream selection expects it.
    if n_grid == 0 and n_trials > 1 and len(inner_shape) >= 2 and inner_shape[0] == 1 and inner_shape[1] == n_trials:
        arr = arr[0]
        inner_shape = inner_shape[1:]

    coords = dict(grid_coords)

    has_trial = n_trials > 1 and len(inner_shape) > 0 and inner_shape[0] == n_trials
    if has_trial:
        trial_dims = ["trial"]
        coords["trial"] = np.arange(n_trials)
        post_trial_shape = inner_shape[1:]
    else:
        trial_dims = []
        post_trial_shape = inner_shape

    ts_arr = None
    if intrinsic_ts is not None:
        ts_arr = np.asarray(intrinsic_ts)
        while ts_arr.ndim > 1:
            ts_arr = ts_arr[0]

    inner_dims, inner_coords = _inner_dims(post_trial_shape, ts_arr, dims)
    coords.update(inner_coords)

    # Drop trailing singleton inner dims so we don't fabricate axes that don't
    # actually carry information (e.g. mode/node when size 1) — but never a DECLARED
    # axis: a single-node observation still has a node axis, because it said so.
    while inner_dims and arr.shape[-1] == 1 and inner_dims != list(dims or []):
        arr = arr[..., 0]
        inner_dims = inner_dims[:-1]

    all_dims = grid_dims + trial_dims + inner_dims
    return xr.DataArray(data=arr, dims=all_dims, coords=coords, name=name)


def reassemble_shards(source, pattern="*__results.nc", to_grid=False, point_dim="point"):
    """Concatenate sharded exploration outputs into the full sweep result.

    Each HPC array task writes its slice of the sweep as a flat ``point``-dim
    ``DataArray`` whose per-cell parameter values are coordinates (see
    :meth:`ExperimentResult.save`). This is the analysis-pass side of the
    two-stage HPC pattern: it reads every shard file, concatenates them along
    ``point``, and — with ``to_grid=True`` — pivots ``point`` into one dimension
    per swept parameter, giving the full rectangular grid addressed by value
    (order-independent, so it is robust to how tasks were sharded).

    Args:
        source: a directory to scan with *pattern*, or an explicit list of paths.
        pattern: glob for shard files when *source* is a directory.
        to_grid: pivot the flat ``point`` dim into one dim per parameter.
        point_dim: name of the flat cell dimension written by the shards.

    Returns:
        The concatenated ``DataArray`` (flat ``point`` dim), or the gridded one
        when ``to_grid`` is set.
    """
    import glob
    import os

    if isinstance(source, (str, os.PathLike)):
        src = os.fspath(source)
        paths = sorted(glob.glob(os.path.join(src, pattern))) if os.path.isdir(src) else sorted(glob.glob(src))
    else:
        paths = list(source)
    if not paths:
        raise FileNotFoundError(f"no shard files matched {source!r} (pattern {pattern!r})")

    combined = xr.concat([xr.open_dataarray(p) for p in paths], dim=point_dim)
    if not to_grid:
        return combined
    coord_names = [c for c in combined.coords if point_dim in combined[c].dims and c != point_dim]
    if not coord_names:
        return combined
    return combined.set_index({point_dim: coord_names}).unstack(point_dim)


def reassemble_experiment_results(
    shards_root, out_dir, pattern="**/*_result.h5", point_dim="point", stem="result", sidecar=None, compress: bool = True
):
    """Gather an HPC run's shard outputs into one keyed ``ExperimentResult`` artifact.

    Follows the same on-disk shape as a :class:`~tvbo.classes.network.Network`:
    one HDF5 file (``<stem>.h5``) holding the data, plus a YAML sidecar
    (``<stem>.yaml``) carrying the frozen, fully-overridden experiment spec — so
    the result is self-describing, provenance-complete and reproducible without
    any extra flags, and identical to what a local run writes.

    Each array task wrote a shard as the same ``<prefix>_result.h5`` Dataset with
    a flat, self-describing ``point`` dimension (see :meth:`ExperimentResult.save`).
    This concatenates them along ``point`` and pivots by parameter value into the
    full rectangular grid, giving one standard xarray ``Dataset`` that opens with a
    plain ``xarray.open_dataset("<stem>.h5")`` — no TVBO-specific reader.

    Args:
        shards_root: directory scanned recursively for the shard ``.h5`` files.
        out_dir: where the ``<stem>.h5`` (+ ``<stem>.yaml``) is written.
        pattern: glob for shard files.
        point_dim: the flat cell dimension the shards wrote.
        stem: basename of the result artifact (default ``result``).
        sidecar: path to the frozen spec YAML to copy as ``<stem>.yaml``
            (typically the kit's ``spec/<name>.yaml``). Omit to skip the sidecar.

    Returns:
        List of written paths (``<stem>.h5`` first, then ``<stem>.yaml``).
    """
    import glob
    import os
    import shutil

    src = os.fspath(shards_root)
    paths = sorted(glob.glob(os.path.join(src, pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no shard files matched {pattern!r} under {src!r}")

    combined = xr.concat([xr.open_dataset(p, engine="h5netcdf") for p in paths], dim=point_dim)
    coord_names = [c for c in combined.coords if point_dim in combined[c].dims and c != point_dim]
    if len(coord_names) >= 2:
        # Multi-parameter sweep: pivot the flat point dim into one dim per parameter,
        # addressing the full rectangular grid by value (order-independent).
        grid = combined.set_index({point_dim: coord_names}).unstack(point_dim)
    elif len(coord_names) == 1:
        # A single ordering coordinate (a one-parameter sweep, or a branch-restart's
        # ``branch_point`` index) is a flat, ordered sequence — not a grid to pivot.
        # Sort by it so shard order is irrelevant, then make it the dimension. (unstack
        # needs a multi-index, so it cannot handle the single-coordinate case at all.)
        grid = combined.sortby(coord_names[0]).swap_dims({point_dim: coord_names[0]})
    else:
        grid = combined
    grid.attrs["tvbo_class"] = "tvbo:ExperimentResult"
    if sidecar is not None:
        grid.attrs["sidecar_file"] = f"{stem}.yaml"

    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{stem}.h5")
    encoding = {name: {"zlib": True, "complevel": 4} for name in grid.data_vars} if compress else None
    grid.to_netcdf(h5_path, engine="h5netcdf", encoding=encoding)
    written = [h5_path]

    if sidecar is not None and os.path.exists(os.fspath(sidecar)):
        yaml_path = os.path.join(out_dir, f"{stem}.yaml")
        shutil.copyfile(os.fspath(sidecar), yaml_path)
        written.append(yaml_path)
    return written


# =============================================================================
# Result Classes for Simulation Experiments
# =============================================================================


class SimulationResult:
    """Output from a single simulation run with its computed observations.

    Stores simulation data as an ``xr.DataArray`` with named dimensions
    (time, variable, node[, mode][, trial]). Observations are bound to the
    simulation that produced them.

    Accepts both new-style (``data=xr.DataArray``) and legacy
    (``result=NativeSolution, state_names=[...]``) constructor signatures
    for backward compatibility with generated template code.

    Attributes
    ----------
    data : xr.DataArray or None
        Simulation data with named dims and coords.
    observations : dict
        Computed observations from this simulation (BOLD, FC, etc.).
    transient : SimulationResult or None
        Warm-up simulation result that preceded this one.
    """

    def __init__(
        self,
        data=None,
        observations=None,
        transient=None,
        *,
        result=None,
        state_names=None,
        nodes=None,
        observation_dims=None,
        units=None,
        **kwargs,
    ):
        self._extras = {}
        self._timeseries = None
        self._units = units or {}  # {variable_name: unit_string}
        self._source = None  # back-reference to ExperimentResult (set externally)

        # ── Backward compat: accept old-style result= arg ──
        if result is not None and data is None:
            raw_data = result.data if hasattr(result, "data") else result
            raw_time = result.ts if hasattr(result, "ts") else None
            data = _to_dataarray(raw_data, raw_time, state_names, nodes)
        elif data is not None and not isinstance(data, xr.DataArray):
            data = _to_dataarray(data, None, state_names, nodes)

        self.data = data
        # Normalize observations to Bunch so both JAX and tvboptim results have
        # dot-access: result.observations.BOLD_TVB  (not just dict indexing).
        # Observations carry the axis names their reduction declared at codegen, bound to
        # the SAME node labels the trajectory just got — the one place holding both.
        _odims = observation_dims or {}
        if observations:
            self.observations = Bunch({k: _observation_dataarray(v, _odims.get(k), nodes) for k, v in observations.items()})
        else:
            self.observations = Bunch()
        self.transient = transient
        self._extras.update(kwargs)
        # Store state_names separately for cases with no data yet
        if state_names and not (data is not None and "variable" in getattr(data, "coords", {})):
            self._extras["state_names"] = state_names

    @property
    def units(self):
        """Unit mapping {variable_name: unit_string} for state/derived variables."""
        return self._units

    # ── xarray delegation ─────────────────────────

    def sel(self, **kw):
        """Label-based selection returning a new SimulationResult."""
        out = SimulationResult(data=self.data.sel(**kw), units=self._units)
        out._source = self._source
        return out

    def isel(self, **kw):
        """Integer-based selection returning a new SimulationResult."""
        out = SimulationResult(data=self.data.isel(**kw), units=self._units)
        out._source = self._source
        return out

    @property
    def time(self):
        """Time values as numpy array (backward compatible)."""
        if self.data is not None and "time" in self.data.coords:
            return self.data.coords["time"].values
        return None

    @property
    def state_names(self):
        """State variable names from data coordinates."""
        if self.data is not None and "variable" in self.data.coords:
            v = self.data.coords["variable"].values
            return list(np.atleast_1d(v))
        return self._extras.get("state_names", [])

    @property
    def dims(self):
        """Dimension names of the data array."""
        if self.data is not None:
            return self.data.dims
        return ()

    @property
    def coords(self):
        """Coordinates of the data array."""
        if self.data is not None:
            return self.data.coords
        return {}

    # ── Backward compat: TimeSeries conversion ────

    def to_timeseries(self):
        """Convert to a full TimeSeries object for plotting and analysis.

        Returns
        -------
        TimeSeries
            4D time series (Time, State Variable, Space, Mode)
        """
        if self._timeseries is not None:
            return self._timeseries

        if self.data is None:
            raise ValueError("No simulation data to convert")

        raw = np.asarray(self.data.values)
        while raw.ndim < 4:
            raw = np.expand_dims(raw, -1)  # pad to 4D (Time, Variable, Space, Mode)

        labels_dimensions = {}
        names = self.state_names
        if names:
            labels_dimensions["State Variable"] = names
        if self.data is not None and "node" in self.data.coords:
            labels_dimensions["Region"] = list(self.data.coords["node"].values)

        time = np.asarray(self.time) if self.time is not None else np.arange(raw.shape[0])
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        self._timeseries = TimeSeries(
            time=time,
            data=raw,
            sample_period=dt,
            labels_dimensions=labels_dimensions,
        )
        # Propagate extras that animate/plot helpers may need (e.g. graph from NetworkDynamics)
        for key in ("graph", "edge_data", "vertex_data", "node_positions"):
            val = self._extras.get(key)
            if val is not None:
                setattr(self._timeseries, key, val)
        return self._timeseries

    def plot(self, ax=None, type="timeseries", **kwargs):
        """Plot simulation results.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (single-panel plots only).
        type : str
            Plot type: 'timeseries' (default), 'phase'/'state-space',
            'vector_field', 'eeg', 'power_spectrum', 'raster'.
        **kwargs
            Forwarded to the underlying plot function in ``tvbo.plot``.
        """
        if type == "raster":
            return self.data.plot(**kwargs)
        if type in {"phase", "state-space", "trajectory"}:
            from tvbo.plot.phase import plot_phase

            return plot_phase(self, ax=ax, **kwargs)
        if type == "vector_field":
            from tvbo.plot.phase import plot_vector_field

            return plot_vector_field(self, ax=ax, **kwargs)
        if type == "eeg":
            from tvbo.plot.timeseries import plot_eeg

            return plot_eeg(self, ax=ax, **kwargs)
        if type == "power_spectrum":
            from tvbo.plot.timeseries import plot_power_spectrum

            return plot_power_spectrum(self, ax=ax, **kwargs)

        # Default: timeseries
        from tvbo.plot.timeseries import plot_timeseries

        return plot_timeseries(self, ax=ax, **kwargs)

    def animate(self, type=None, **kwargs):
        """Animate simulation results.

        Parameters
        ----------
        type : str or list of str, optional
            Single panel type:
                'network' — nodes colored by state on graph layout.
                'phase' — trailing trajectory in phase space.
                'timeseries' — evolving time-series traces.
                'pendulum' — dual-panel: pendulum bob + timeseries.
                A state variable name — selects that variable, then animates.
            List of panel types for custom multi-panel layout:
                e.g. ``['pendulum_bob', 'timeseries']``,
                ``['phase', 'timeseries']``
            If None, auto-selects based on available metadata.
        **kwargs
            Forwarded to the animation function.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        from tvbo.plot.animate import _COMPOSITE_TYPES, _PANEL_REGISTRY, animate_multi

        # List of panels → multi-panel animation
        if isinstance(type, list):
            return animate_multi(self, type, **kwargs)

        # Named composite type (e.g. 'pendulum' → ['pendulum_bob', 'timeseries'])
        if type in _COMPOSITE_TYPES:
            return animate_multi(self, _COMPOSITE_TYPES[type], **kwargs)

        _known_types = {"network", "phase", "timeseries", None}
        result = self
        if type not in _known_types and type not in _PANEL_REGISTRY:
            # Treat as variable name selection
            result = self.sel(variable=type)
            type = None
        if type is None:
            type = result._resolve_animate_type()
        if type == "phase":
            from tvbo.plot.animate import animate_phase

            return animate_phase(result, **kwargs)
        if type == "timeseries":
            from tvbo.plot.animate import animate_timeseries

            return animate_timeseries(result, **kwargs)
        from tvbo.plot.animate import animate_network

        return animate_network(result, **kwargs)

    def _resolve_animate_type(self):
        """Pick the best animation type based on available metadata."""
        if self._extras.get("graph"):
            return "network"
        exp_result = self._source
        if exp_result is not None:
            experiment = getattr(exp_result, "source", None)
            if experiment is not None and getattr(experiment, "network", None) is not None:
                return "network"
        has_nodes = self.data is not None and "node" in getattr(self.data, "coords", {})
        if has_nodes:
            return "network"
        return "timeseries"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        # Check extras first
        if name in self._extras:
            return self._extras[name]
        # Delegate to xarray DataArray (mean, sum, std, min, max, etc.)
        if self.data is not None and hasattr(self.data, name):
            return getattr(self.data, name)
        # Delegate to TimeSeries for plot_eeg, plot_power_spectrum, etc.
        try:
            ts = self.to_timeseries()
        except (ValueError, AttributeError, RecursionError):
            raise AttributeError(name)
        if hasattr(ts, name):
            return getattr(ts, name)
        raise AttributeError(f"SimulationResult has no attribute '{name}'")

    def __repr__(self):
        n_obs = len(self.observations.keys()) if self.observations else 0
        shape = getattr(self, "data", None)
        shape_str = f"{tuple(shape.shape)}" if shape is not None else "empty"
        obs_str = f", {n_obs} observations" if n_obs > 0 else ""
        return f"SimulationResult{shape_str}{obs_str}"


class AlgorithmResult:
    """Result of an iterative algorithm (FIC, EIB, etc.).

    Provides structured access to algorithm outputs with consistent naming
    regardless of which algorithm was run.

    Attributes
    ----------
    name : str
        Algorithm name
    state : Bunch
        Final state with tuned parameters
    history : Bunch
        Per-iteration tracking: parameters, observations, metrics
    pre_tuning : SimulationResult
        Simulation BEFORE algorithm (for comparison)
    post_tuning : SimulationResult
        Simulation AFTER algorithm with attached observations
    n_iterations : int
        Number of iterations run
    hyperparameters : Bunch
        Algorithm hyperparameters used (eta, window_size, etc.)
    convergence : Bunch
        Convergence metrics (final values, deltas, etc.)
    """

    def __init__(
        self,
        name: str = None,
        state=None,
        history=None,
        pre_tuning=None,
        post_tuning=None,
        post_tuning_observations=None,
        n_iterations: int = None,
        hyperparameters=None,
        state_names=None,
        **kwargs,
    ):
        self.name = name
        self.state = state
        self.history = history or Bunch()

        # Wrap simulations in SimulationResult for consistent access
        if pre_tuning is not None and not isinstance(pre_tuning, SimulationResult):
            self.pre_tuning = SimulationResult(result=pre_tuning, state_names=state_names)
        else:
            self.pre_tuning = pre_tuning

        if post_tuning is not None and not isinstance(post_tuning, SimulationResult):
            self.post_tuning = SimulationResult(
                result=post_tuning,
                observations=post_tuning_observations or Bunch(),
                state_names=state_names,
            )
        elif post_tuning is None and post_tuning_observations:
            # Streaming post-eval yields observations without a trajectory; expose them.
            self.post_tuning = SimulationResult(data=None, observations=post_tuning_observations)
        else:
            self.post_tuning = post_tuning

        self.n_iterations = n_iterations
        self.hyperparameters = hyperparameters or Bunch()
        self._extras = kwargs

        # Compute convergence metrics from history
        self.convergence = self._compute_convergence()

    def _compute_convergence(self):
        """Compute convergence metrics from history."""
        conv = Bunch()
        if self.history:
            for key, vals in self.history.items():
                if hasattr(vals, "__len__") and len(vals) > 0:
                    conv[f"{key}_final"] = vals[-1] if hasattr(vals, "__getitem__") else vals
                    if len(vals) > 1:
                        conv[f"{key}_delta"] = vals[-1] - vals[0]
        return conv

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(f"AlgorithmResult has no attribute '{name}'")

    def get(self, key, default=None):
        """Dict-like get for backward compat with Bunch-based code."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __repr__(self):
        n_iter = self.n_iterations or (len(next(iter(self.history.values()))) if self.history else 0)
        return f"AlgorithmResult(name='{self.name}', n_iterations={n_iter})"


class OptimizationResult:
    """Result of gradient-based optimization.

    Provides structured access to optimization outputs including loss trajectory,
    parameter evolution, and final simulation.

    Attributes
    ----------
    name : str
        Optimization/loss function name
    state : Bunch
        Final optimized state (alias: fitted_params)
    history : Bunch
        Per-step tracking: loss values, states, gradients
    simulation : SimulationResult
        Post-optimization simulation with attached observations
    loss_trajectory : jnp.ndarray
        Loss values at each step (convenience accessor)
    n_steps : int
        Number of optimization steps
    final_loss : float
        Final loss value
    hyperparameters : Bunch
        Optimizer settings (learning_rate, algorithm, etc.)
    """

    def __init__(
        self,
        name: str = None,
        state=None,
        history=None,
        simulation=None,
        n_steps: int = None,
        hyperparameters=None,
        **kwargs,
    ):
        self.name = name
        self.state = state
        self.fitted_params = state  # Legacy alias
        self.history = history or Bunch()
        self.fitting_data = history  # Legacy alias

        # Wrap simulation in SimulationResult if needed
        if simulation is not None and not isinstance(simulation, SimulationResult):
            self.simulation = SimulationResult(result=simulation)
        else:
            self.simulation = simulation

        self.n_steps = n_steps
        self.hyperparameters = hyperparameters or Bunch()
        self._extras = kwargs

        # Extract loss trajectory from history
        self._extract_metrics()

    def _extract_metrics(self):
        """Extract convenience metrics from history."""
        if self.history and hasattr(self.history, "loss"):
            loss_data = self.history.loss
            if hasattr(loss_data, "save"):
                self.loss_trajectory = loss_data.save
            elif hasattr(loss_data, "array"):
                self.loss_trajectory = loss_data.array
            else:
                self.loss_trajectory = loss_data

            if self.loss_trajectory is not None and len(self.loss_trajectory) > 0:
                self.final_loss = float(self.loss_trajectory[-1])
                self.initial_loss = float(self.loss_trajectory[0])
                self.loss_improvement = self.initial_loss - self.final_loss
        else:
            self.loss_trajectory = None
            self.final_loss = None

        # State trajectory (from SavingParametersCallback)
        if self.history and "parameters" in self.history:
            params_data = self.history["parameters"]
            if hasattr(params_data, "save"):
                traj = params_data.save
                if hasattr(traj, "tolist"):
                    self.state_trajectory = traj.tolist()
                else:
                    self.state_trajectory = list(traj) if hasattr(traj, "__iter__") else traj
            else:
                self.state_trajectory = params_data
        else:
            self.state_trajectory = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(f"OptimizationResult has no attribute '{name}'")

    def __repr__(self):
        loss_str = f", final_loss={self.final_loss:.4f}" if self.final_loss is not None else ""
        return f"OptimizationResult(name='{self.name}', n_steps={self.n_steps}{loss_str})"

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, type="summary", ax=None, figsize=None, **kwargs):
        """Plot optimization results.

        Parameters
        ----------
        type : str
            ``'summary'`` (default) – loss curve + parameter trajectories.
            ``'loss'`` – loss curve only.
            ``'parameters'`` – free-parameter evolution over steps.
            ``'state'`` – final fitted parameter values (bar charts).
        ax : matplotlib.axes.Axes, optional
            Target axes (single-panel plots only, i.e. *type='loss'*).
        figsize : tuple, optional
        **kwargs
            Forwarded to matplotlib plot calls.
        """
        from tvbo.plot.utils import use_tvbo_style

        use_tvbo_style()

        if type == "loss":
            return self._plot_loss(ax=ax, figsize=figsize, **kwargs)
        if type == "parameters":
            return self._plot_parameters(figsize=figsize, **kwargs)
        if type == "state":
            return self._plot_state(figsize=figsize, **kwargs)
        # Default: summary
        return self._plot_summary(figsize=figsize, **kwargs)

    # --- loss curve ---------------------------------------------------

    def _get_loss_values(self):
        """Return loss trajectory as a numpy array, or None."""
        if self.loss_trajectory is not None:
            return np.asarray(self.loss_trajectory)
        # Fallback: try dict-style access on history (plain dict)
        if isinstance(self.history, dict) and "loss" in self.history:
            loss_data = self.history["loss"]
            if hasattr(loss_data, "save"):
                return np.asarray(loss_data["save"])
        return None

    def _plot_loss(self, ax=None, figsize=None, **kwargs):
        loss = self._get_loss_values()
        if loss is None:
            raise ValueError("No loss trajectory available")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 3))
        else:
            fig = ax.get_figure()
        steps = np.arange(len(loss))
        ax.plot(steps, loss, **kwargs)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(self.name or "Loss")
        fig.tight_layout()
        plt.close(fig)
        return fig

    # --- parameter trajectories ---------------------------------------

    @staticmethod
    def _flatten_params(state, prefix=""):
        """Flatten a (possibly nested) state to ``{dotted_name: ndarray}``.

        Recurses into containers (dicts, objects) and returns array-like values as
        leaves. A leaf is anything exposing ``ndim`` — numpy and jax alike; a jax
        array carries an empty ``__dict__``, so it is detected as a leaf here rather
        than recursed into as an empty container.
        """
        flat = {}
        if state is None:
            return flat
        # Treat JAX-array-protocol objects (e.g. tvboptim Parameter,
        # BoundedParameter) as leaf nodes — don't recurse into their
        # internal attrs like .low / .high.
        if hasattr(state, "__jax_array__"):
            try:
                arr = np.asarray(state.__jax_array__())
                if arr.dtype.kind in ("f", "i", "u"):
                    flat[prefix] = arr
            except (TypeError, ValueError):
                pass
            return flat
        items = None
        if isinstance(state, dict):
            items = state.items()
        elif hasattr(state, "__dict__") and not hasattr(state, "ndim"):
            items = ((k, v) for k, v in vars(state).items() if not k.startswith("_"))
        if items is not None:
            for k, v in items:
                name = f"{prefix}.{k}" if prefix else k
                sub = OptimizationResult._flatten_params(v, name)
                if sub:
                    flat.update(sub)
        else:
            # Leaf node
            try:
                val = state.__jax_array__() if hasattr(state, "__jax_array__") else state
                arr = np.asarray(val)
                if arr.dtype.kind in ("f", "i", "u"):
                    flat[prefix] = arr
            except (TypeError, ValueError):
                pass
        return flat

    def _get_param_trajectories(self):
        """Return ``{name: array(n_steps, ...)}`` from *state_trajectory*."""
        if not self.state_trajectory:
            return {}
        all_flat = [self._flatten_params(s) for s in self.state_trajectory]
        if not all_flat or not all_flat[0]:
            return {}
        names = list(all_flat[0].keys())
        result = {}
        for name in names:
            try:
                vals = [f[name] for f in all_flat if name in f]
                if len(vals) != len(all_flat):
                    continue
                result[name] = np.stack(vals)
            except (ValueError, KeyError):
                continue
        return result

    def _plot_parameters(self, figsize=None, **kwargs):
        trajectories = self._get_param_trajectories()
        if not trajectories:
            raise ValueError("No parameter trajectories available")
        n = len(trajectories)
        fig, axes = plt.subplots(
            n,
            1,
            figsize=figsize or (8, 2.5 * n),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        for ax, (name, values) in zip(axes, trajectories.items()):
            steps = np.arange(values.shape[0])
            if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
                ax.plot(steps, values.ravel(), **kwargs)
            else:
                for j in range(values.shape[1]):
                    ax.plot(steps, values[:, j], alpha=0.4, linewidth=0.8, **kwargs)
            ax.set_ylabel(name)
        axes[-1].set_xlabel("Step")
        fig.suptitle(
            f"{self.name}: Parameters" if self.name else "Parameter Evolution",
        )
        fig.tight_layout()
        plt.close(fig)
        return fig

    # --- final state bar charts ---------------------------------------

    def _get_final_params(self):
        if self.state is None:
            return {}
        return self._flatten_params(self.state)

    def _plot_state(self, figsize=None, **kwargs):
        params = self._get_final_params()
        if not params:
            raise ValueError("No fitted parameters available")
        scalar = {k: float(v) for k, v in params.items() if v.ndim == 0}
        arrays = {k: v for k, v in params.items() if v.ndim > 0}
        n_panels = (1 if scalar else 0) + len(arrays)
        if n_panels == 0:
            raise ValueError("No parameters to plot")
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (4 * n_panels, 4),
            squeeze=False,
        )
        axes = axes[0]
        idx = 0
        if scalar:
            axes[idx].bar(list(scalar.keys()), list(scalar.values()), **kwargs)
            axes[idx].set_title("Scalar parameters")
            axes[idx].tick_params(axis="x", rotation=45)
            idx += 1
        for name, val in arrays.items():
            axes[idx].bar(np.arange(len(val)), val, **kwargs)
            axes[idx].set_xlabel("Node")
            axes[idx].set_title(name)
            idx += 1
        fig.suptitle(
            f"{self.name}: Fitted State" if self.name else "Fitted State",
        )
        fig.tight_layout()
        plt.close(fig)
        return fig

    # --- summary (loss + params) --------------------------------------

    def _plot_summary(self, figsize=None, **kwargs):
        loss = self._get_loss_values()
        trajectories = self._get_param_trajectories()
        n_panels = (1 if loss is not None else 0) + len(trajectories)
        if n_panels == 0:
            raise ValueError("No optimization data to plot")
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=figsize or (8, 2.5 * n_panels),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        idx = 0
        if loss is not None:
            steps = np.arange(len(loss))
            axes[idx].plot(steps, loss)
            axes[idx].set_ylabel("Loss")
            idx += 1
        for name, values in trajectories.items():
            steps = np.arange(values.shape[0])
            if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
                axes[idx].plot(steps, values.ravel())
            else:
                for j in range(values.shape[1]):
                    axes[idx].plot(
                        steps,
                        values[:, j],
                        alpha=0.4,
                        linewidth=0.8,
                    )
            axes[idx].set_ylabel(name)
            idx += 1
        axes[-1].set_xlabel("Step")
        title = self.name or "Optimization"
        if self.final_loss is not None:
            title += f"  (final loss: {self.final_loss:.4f})"
        fig.suptitle(title)
        fig.tight_layout()
        plt.close(fig)
        return fig


class InferenceResult:
    """Result of Bayesian inference (MCMC posterior over parameters).

    Attributes
    ----------
    name : str
        Inference name (the ``inferences:`` key).
    posterior : dict
        Posterior samples keyed by parameter dotted-name (the ``priors`` keys),
        each an array of length ``num_samples`` (× ``num_chains``).
    diagnostics : dict
        Sampler diagnostics (per-parameter ``mean``/``std``/``r_hat``/``n_eff`` etc.,
        as returned by ``numpyro.diagnostics.summary``).
    """

    def __init__(self, name=None, posterior=None, diagnostics=None, **kwargs):
        self.name = name
        self.posterior = posterior or {}
        self.diagnostics = diagnostics or {}
        self._extras = kwargs

    def mean(self):
        """Posterior mean per parameter."""
        import numpy as _np

        return {k: float(_np.asarray(v).mean()) for k, v in self.posterior.items()}

    def std(self):
        """Posterior standard deviation per parameter."""
        import numpy as _np

        return {k: float(_np.asarray(v).std()) for k, v in self.posterior.items()}

    def __repr__(self):
        return f"InferenceResult(name={self.name!r}, params={list(self.posterior)})"


class ExplorationResult(Bunch):
    """Result of parameter exploration (grid search).

    A thin wrapper around tvboptim exploration outputs that provides:
    - Access to labelled results (flat or grid-shaped)
    - Axis information for parameter values
    - Utility methods for finding optimal points and slicing
    - Time series plotting for parameter sweeps (when observable returns time series)

    Designed to work with tvboptim's Space and ParallelResult directly,
    while also supporting other exploration backends.

    Supports two result types:
    - **Scalar results**: Each grid point produces a scalar (e.g., loss function).
      Stored flat, reshaped via ``as_grid()``, with ``optimal`` point tracking.
    - **Time series results**: Each grid point produces a time series (e.g., model
      output). Stored as ``(n_grid, n_time, ...)``, with ``plot()`` support.

    Attributes
    ----------
    name : str
        Exploration name
    grid : Space
        Parameter grid specification (tvboptim Space object)
    results : xr.DataArray
        Observable values at each grid point (flat for scalars, multi-dim for time
        series), carrying named dims: the leading run axis (the swept parameter,
        ``trial``, or ``point``) followed by the intrinsic dims (time, variable,
        node, mode). The payload stays JAX-native — only the labels are
        materialised — and the shape is unchanged from what the backend emitted, so
        it is addressed by key rather than by position. ``as_grid()`` reshapes the
        flat run axis into one dim per exploration axis.
    axes : list
        List of axis info (Bunch with name, lo, hi, n, values)
    observable : str
        Name of observable computed
    optimal : Bunch
        Best point found (parameters, value, index) — only for scalar results
    shape : tuple
        Grid shape derived from axes
    is_timeseries : bool
        True if results contain time series per grid point
    dt : float
        Time step for time series results (optional)
    output_names : list[str]
        Names of output variables (e.g., ['v_pyr']) for time series results
    """

    def __init__(
        self,
        name: str = None,
        grid=None,
        results=None,
        axes=None,
        observable: str = None,
        dt: float = None,
        output_names: list = None,
        observations=None,
        cell_coords=None,
        is_shard=None,
        **kwargs,
    ):
        """A sweep's results, labelled against the axes that produced them.

        ``cell_coords`` (``{axis: (n_cell,) array}``) is each cell's actual parameter
        values, set for EVERY keyed sweep — a whole grid as much as one array task's
        slice. It drives placement by value in ``as_grid``: into the rectangular grid
        for a full product, onto a flat ``point`` dim for a subset. It is NOT a shard
        marker, and reading it as one cost every local sweep its provenance sidecar.

        ``is_shard`` is that marker, declared by the producer — the generated script
        holds ``kwargs['shard']``. ``None`` means undeclared, and ``_is_partial_shard``
        falls back to counting cells. Nothing downstream can re-derive it reliably: a
        branch shard's axis ``n`` is taken from the already-sliced index, so the slice
        looks complete.
        """
        super().__init__(**kwargs)
        self.name = name
        self.grid = grid
        self.axes = axes or []
        self.observable = observable
        self.dt = dt
        self.output_names = output_names or []
        self.is_shard = is_shard
        self.cell_coords = cell_coords
        # Per-grid-point observations as {name: xr.DataArray} with grid axes
        # prepended to each observation's intrinsic dims (time/variable/node/mode).
        # Grid codegen already hands over labelled DataArrays; the warm-start /
        # adiabatic path hands over plain arrays, so label those here (against the
        # swept axes) — the class honours its own contract regardless of producer,
        # and every consumer (plotting, save, reassembly) sees DataArrays.
        self.observations = {
            k: (
                _stacked_to_dataarray(v, self.axes, name=k, cell_coords=self.cell_coords)
                if v is not None and not hasattr(v, "dims")
                else v
            )
            for k, v in (observations or {}).items()
        }

        # Collapse any axis marked ``ExplorationAxis.reduce`` by its statistic. An
        # ensemble axis (e.g. an ``execution.random_seed`` trial ensemble) is reduced
        # in place across every observation carrying its named grid dim, so the reduced
        # observation (mean/sem/…) becomes first-class and the collapsed dim — and its
        # axis metadata — drop out. Keyed by dim name (never positional); observations
        # without the dim are untouched. No reduce axes ⇒ a no-op (behaviour unchanged).
        self._apply_axis_reductions()

        # Compute expected grid shape from axes
        self._grid_shape = tuple(
            ax.get("n", getattr(ax, "n", None)) for ax in self.axes if (isinstance(ax, dict) and "n" in ax) or hasattr(ax, "n")
        )

        # Detect whether results are time series or scalar per grid point
        if results is not None:
            # A producer may hand over an already-labelled payload; take the raw
            # array so the shape detection below sees the same thing either way.
            results_arr = jnp.asarray(self._payload(results))
            # Grid occupies the leading dim(s); anything after is intrinsic (time/node), so
            # an extra dim ⇒ time series. Grid layout is either a flat cell dim (prod of axes,
            # the runner's) or one dim per axis — resolve from the shape (comparing ndim to
            # len(grid_shape) unconditionally mis-flattened >2-axis timeseries sweeps).
            n_grid = int(np.prod(self._grid_shape)) if self._grid_shape else None
            n_axes = len(self._grid_shape) if self._grid_shape else 0
            shp = tuple(results_arr.shape)
            if n_grid is not None and shp and shp[0] == n_grid:
                n_grid_dims = 1  # flat cell product (runner layout)
            elif n_axes and shp[:n_axes] == tuple(self._grid_shape):
                n_grid_dims = n_axes  # one dim per axis
            else:
                n_grid_dims = n_axes or 1  # fallback (prior behaviour)
            if results_arr.ndim > n_grid_dims:
                self.results = results_arr  # time series: preserve structure
                self.is_timeseries = True
            else:
                self.results = results_arr.flatten()  # scalar per grid point
                self.is_timeseries = False

            # Trials-only explorations (no sweep axes) can be emitted as
            # (1, n_trials, n_time, ...). Collapse the synthetic grid axis so
            # plotting interprets axis 1 as time instead of trials.
            n_trials = int(getattr(self, "n_trials", 1) or 1)
            if (
                self.is_timeseries
                and not self.axes
                and n_trials > 1
                and self.results.ndim >= 3
                and self.results.shape[0] == 1
                and self.results.shape[1] == n_trials
            ):
                self.results = self.results[0]
        else:
            self.results = None
            self.is_timeseries = False

        # Label the payload so every result carries named dims, whatever the
        # producer handed over. Consumers then select by key instead of by
        # position, which is what keeps a layout change from silently reading the
        # wrong channel.
        self.results = self._label_payload(self.results)

        # Shape is the expected grid shape from axes
        self.shape = self._grid_shape
        self._find_optimal()

    @staticmethod
    def _payload(data):
        """The raw array behind a labelled payload (JAX-native, no copy)."""
        return data.data if hasattr(data, "dims") else data

    def _has_trial_axis(self, tail_first) -> bool:
        """Whether the dim after the run axis is a per-point ``trial`` ensemble.

        True only for a swept exploration carried *with* trials, where the payload
        keeps a trial axis between the grid axis and time. A trials-only ensemble
        already spends its leading axis on ``trial`` and is excluded by the caller.
        """
        n_trials = int(getattr(self, "n_trials", 0) or 0)
        return n_trials > 1 and tail_first == n_trials

    def _intrinsic_dims(self, tail_shape, *, trial_first: bool = False):
        """Names + coords for the intrinsic dims that follow the leading run axis.

        ``tail_shape`` is the payload shape after the run axis (the swept
        parameter, ``point``, or ``trial``). Returns ``(dims, coords)`` following
        the TVB convention ``time[, variable][, node][, mode]``, optionally
        prefixed with a per-point ``trial`` axis. Single home for the labelling
        rule so :meth:`_label_payload` and :meth:`as_grid` never disagree.
        """
        dims: list = []
        coords: dict = {}
        tail = list(tail_shape)
        if trial_first and tail:
            dims.append("trial")
            coords["trial"] = np.arange(tail[0])
            tail = tail[1:]
        if not tail:
            return dims, coords
        dims.append("time")
        if self.dt:
            coords["time"] = np.arange(tail[0]) * self.dt
        # tvboptim drops the `variable` dim for a single model output, so only
        # label the leading spatial dim `variable` when it matches the output
        # count; the rest map to (node, mode). Unknown output count → assume
        # `variable` is present.
        spatial = tail[1:]
        n_out = len(self.output_names) if self.output_names else None
        if spatial and (n_out is None or spatial[0] == n_out):
            dims.append("variable")
            if self.output_names and len(self.output_names) == spatial[0]:
                coords["variable"] = list(self.output_names)
            spatial = spatial[1:]
        dims += ["node", "mode"][: len(spatial)]
        return dims, coords

    def _label_payload(self, data):
        """Name the dims of the results payload **without reshaping it**.

        The leading dim is the flat run axis: the swept parameter when exactly one
        axis is explored, ``trial`` for a trials-only ensemble, otherwise ``point``
        (the flattened grid product, which :meth:`as_grid` reshapes into one dim per
        axis). Intrinsic dims follow the TVB convention (time, variable, node, mode)
        and pick up coordinates from ``dt`` and ``output_names``.

        Shapes are left untouched, so positional consumers keep working while keyed
        access becomes possible.
        """
        if data is None or hasattr(data, "dims"):
            return data
        ndim = getattr(data, "ndim", 0)
        if ndim == 0:
            return xr.DataArray(data, name=self.observable or None)

        names = [self._axis_name(ax) for ax in self.axes]
        n_trials = int(getattr(self, "n_trials", 0) or 0)
        coords = {}

        if len(names) == 1 and names[0] and data.shape[0] == (int(self._grid_shape[0]) if self._grid_shape else -1):
            lead = names[0]
            vals = self._axis_values(self.axes[0])
            if vals is not None and len(vals) == data.shape[0]:
                coords[lead] = np.asarray(vals)
        elif not self.axes and n_trials and data.shape[0] == n_trials:
            lead = "trial"
            coords[lead] = np.arange(data.shape[0])
        else:
            lead = "point"
        dims = [lead]

        if self.is_timeseries and ndim > 1:
            tail = data.shape[1:]
            # A trials-only ensemble already spent its leading axis on `trial`;
            # only a swept run carries a separate trial axis in the tail.
            trial_first = lead != "trial" and self._has_trial_axis(tail[0])
            intrinsic, intrinsic_coords = self._intrinsic_dims(tail, trial_first=trial_first)
            dims += intrinsic
            coords.update(intrinsic_coords)

        # Any dim the layout does not account for still gets a name, so the result
        # is labelled even when the producer emits an unexpected rank.
        while len(dims) < ndim:
            dims.append(f"dim_{len(dims)}")
        try:
            return xr.DataArray(data, dims=dims[:ndim], coords=coords, name=self.observable or None)
        except Exception:
            return xr.DataArray(data, name=self.observable or None)

    @staticmethod
    def _axis_name(ax):
        return ax.get("name") if isinstance(ax, dict) else getattr(ax, "name", None)

    @staticmethod
    def _axis_values(ax):
        """Swept coordinate values for one axis (``Bunch`` or plain dict)."""
        return ax.get("explored_values") if isinstance(ax, dict) else getattr(ax, "explored_values", None)

    @staticmethod
    def _axis_reduce(ax):
        """Reduction statistic for one axis (``ExplorationAxis.reduce.statistic``), or ``None``."""
        return ax.get("reduce") if isinstance(ax, dict) else getattr(ax, "reduce", None)

    @staticmethod
    def _reduce_dataarray(da, dim, stat):
        """Collapse ``da`` along the named ``dim`` by ``stat`` (keyed by dim name).

        Supports ``mean``, ``sum``, ``std``, ``median`` and ``sem`` (the standard
        error of the mean, ``std`` along the dim divided by ``sqrt(n)``).
        """
        stat = str(stat).lower()
        if stat == "mean":
            return da.mean(dim=dim)
        if stat == "sum":
            return da.sum(dim=dim)
        if stat == "std":
            return da.std(dim=dim)
        if stat == "median":
            return da.median(dim=dim)
        if stat == "sem":
            return da.std(dim=dim) / np.sqrt(da.sizes[dim])
        raise ValueError(f"unknown ExplorationAxis.reduce statistic {stat!r}; expected one of: mean, sum, std, sem, median")

    def _apply_axis_reductions(self):
        """Collapse every axis marked ``reduce`` across the observations it labels.

        For each axis whose ``reduce`` statistic is set, the matching named grid
        dimension is reduced across every observation ``DataArray`` that carries it
        (keyed by dim name), the reduced observations keep their names, and the axis
        is dropped from ``self.axes`` so the shape metadata stays consistent.
        Observations without the dim are left untouched. A no-op when no axis sets
        ``reduce`` (result is byte-identical to a run without the feature).
        """
        if not any(self._axis_reduce(ax) for ax in self.axes):
            return
        for ax in self.axes:
            stat = self._axis_reduce(ax)
            if not stat:
                continue
            dim = self._axis_name(ax)
            if not dim:
                continue
            for k, da in list(self.observations.items()):
                if da is None or not hasattr(da, "dims") or dim not in da.dims:
                    continue  # skip observations that don't carry this dim
                self.observations[k] = self._reduce_dataarray(da, dim, stat)
        # Drop the collapsed axes from the axis metadata so downstream shape
        # computation (``_grid_shape``) and labelling exclude them.
        self.axes = [ax for ax in self.axes if not self._axis_reduce(ax)]

    def as_grid(self):
        """Reshape the flat results into a grid **labeled by parameter name**.

        Returns an ``xr.DataArray`` with one dimension per exploration axis — named
        by the swept parameter, coordinates set to the swept values — so grid
        results are addressed by name (``g.sel(**{"ReducedWongWang.w": 0.5})``) and
        are **independent of axis order**. The data stays a JAX array (the DataArray
        is a registered JAX pytree); only the coordinate labels are materialised. A
        time-series observable keeps its intrinsic dims (time, variable, node, mode)
        after the grid dims. ``None`` when empty; otherwise always labelled — a
        payload that cannot be reshaped into the grid is returned with the dim names
        it already carries (see :meth:`_label_payload`) rather than as a bare array,
        so no consumer is handed positional data. A set ``cell_coords`` selects the
        keyed path below, which every sweep takes because every sweep sets it;
        :func:`_stacked_to_dataarray` then decides the shape from whether the cells fill
        the Cartesian product. A full product is placed into the rectangular grid BY
        VALUE, so ``sel`` by parameter works as usual. A subset — an HPC array task's
        slice, or a branch restart — gets a single ``point`` dim carrying each axis's
        value, so it reassembles across shards by parameter value.

        Do not read a set ``cell_coords`` as "this is a shard". ``_is_partial_shard``
        answers that separate question, for provenance rather than labelling, and
        prefers the producer's declared ``is_shard``.
        """
        if self.results is None:
            return None
        if getattr(self, "cell_coords", None):
            n_trials = int(getattr(self, "n_trials", 1) or 1)
            intrinsic_ts = None
            if self.is_timeseries and self.dt:
                r = self.results
                t_dim = 1 + (1 if (n_trials > 1 and r.ndim > 1 and r.shape[1] == n_trials) else 0)
                if r.ndim > t_dim:
                    intrinsic_ts = np.arange(r.shape[t_dim]) * self.dt
            return _stacked_to_dataarray(
                self._payload(self.results),
                self.axes,
                intrinsic_ts=intrinsic_ts,
                n_trials=n_trials,
                name=self.observable or None,
                cell_coords=self.cell_coords,
            )
        labelled = self.results  # already carries named dims
        data = self._payload(labelled)  # keep JAX-native; never np.asarray the payload
        names = [self._axis_name(ax) for ax in self.axes]
        grid_shape = tuple(self._grid_shape or ())
        n_grid = 1
        for _s in grid_shape:
            n_grid *= int(_s)
        if not all(names) or not grid_shape:
            # No named grid to lay out (trials-only, or a nameless axis): the payload
            # is already labelled with its own dims, so hand that back.
            return labelled
        try:
            intrinsic_coords = {}
            if self.is_timeseries:
                if data.shape[0] != n_grid:
                    return labelled
                data = data.reshape(grid_shape + tuple(data.shape[1:]))
                # Intrinsic (post-grid) dims follow the same rule as the flat
                # payload — resolved once in `_intrinsic_dims` so the two agree.
                tail = data.shape[len(names) :]
                trial_first = bool(tail) and self._has_trial_axis(tail[0])
                intrinsic, intrinsic_coords = self._intrinsic_dims(tail, trial_first=trial_first)
                dims = names + intrinsic
            else:
                if data.size != n_grid:
                    return labelled
                data = data.reshape(grid_shape)
                dims = list(names)
            sizes = dict(zip(dims, data.shape))
            coords = {}
            for ax, nm in zip(self.axes, names):
                vals = self._axis_values(ax)
                if vals is not None and len(vals) == sizes.get(nm):
                    coords[nm] = np.asarray(vals)  # coordinate labels, like TimeSeries' time
            coords.update(intrinsic_coords)
            return xr.DataArray(data, dims=dims, coords=coords, name=self.observable or None)
        except Exception:
            # Never degrade to a bare array — the labelled payload is still correct,
            # just not reshaped into the grid.
            return labelled

    def _find_optimal(self):
        """Find optimal point in the grid (scalar results only)."""
        self.optimal = Bunch()
        if self.is_timeseries or self.results is None or self.results.size == 0:
            return
        # Find argmin in flat results (assumes lower is better for loss functions)
        flat = self._payload(self.results).flatten()
        flat_idx = int(jnp.argmin(flat))
        self.optimal.flat_index = flat_idx
        self.optimal.value = float(flat[flat_idx])

        # Compute grid index if we have valid grid shape
        if self._grid_shape and len(self._grid_shape) > 0:
            expected_size = int(jnp.prod(jnp.array(self._grid_shape)))
            if flat.size == expected_size:
                self.optimal.index = tuple(int(i) for i in jnp.unravel_index(flat_idx, self._grid_shape))
            else:
                self.optimal.index = (flat_idx,)
        else:
            self.optimal.index = (flat_idx,)

        # Extract parameter values at optimal point
        self.optimal.parameters = Bunch()
        for i, ax in enumerate(self.axes):
            ax_name = self._axis_name(ax)
            ax_values = self._axis_values(ax)
            if ax_name and ax_values is not None and i < len(self.optimal.index):
                idx = self.optimal.index[i]
                if idx < len(ax_values):
                    self.optimal.parameters[ax_name] = float(ax_values[idx])

    def _get_time_axis(self):
        """Reconstruct time vector from dt and n_time."""
        if self.results is None or not self.is_timeseries:
            return None
        n_time = self.results.shape[1]  # (n_grid, n_time, ...)
        if self.dt:
            return np.arange(n_time) * self.dt
        return np.arange(n_time)

    def plot(self, figsize=None, sharex=True, ax=None, overlay=False, **kwargs):
        """Plot exploration results.

        For time series results: subplots for each parameter value by default,
        or a single overlaid axis when ``overlay=True``.
        For scalar results: line plot (1D) or filled-contour heatmap (2D), drawn into ``ax`` if given.
        """
        if not self.is_timeseries:
            return self._plot_scalar(figsize=figsize, ax=ax, **kwargs)
        return self._plot_timeseries(figsize=figsize, sharex=sharex, ax=ax, overlay=overlay, **kwargs)

    def _plot_timeseries(self, figsize=None, sharex=True, ax=None, overlay=False, **kwargs):
        """Plot time series for each parameter value."""
        if self.results is None:
            return None

        ax_info = self.axes[0] if self.axes else None
        n = int(ax_info.n) if ax_info else self.results.shape[0]
        time = self._get_time_axis()
        output_label = self.observable or ", ".join(self.output_names) or "output"

        ax_values = np.asarray(ax_info["explored_values"]) if ax_info else None

        if overlay:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (8, 4))
            else:
                fig = ax.figure

            for i in range(n):
                data = np.asarray(self.results[i]).squeeze()
                label = None
                if ax_values is not None:
                    label = f"{ax_info.name}={float(ax_values[i]):.4g}"
                if data.ndim > 1:
                    for node_idx in range(data.shape[-1]):
                        ax.plot(time, data[:, node_idx], alpha=0.7, label=label if node_idx == 0 else None, **kwargs)
                else:
                    ax.plot(time, data, label=label, **kwargs)

            ax.set_xlabel("Time" + (f" (dt={self.dt})" if self.dt else " (steps)"))
            ax.set_ylabel(output_label)
            ax.set_title(self.name or output_label)
            if ax_values is not None:
                ax.legend(frameon=False)
            fig.tight_layout()
            plt.close(fig)
            return fig

        fig, axes = plt.subplots(
            n,
            1,
            figsize=figsize or (12, 2 * n),
            sharex=sharex,
        )
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            data = np.asarray(self.results[i])  # (n_time, ...) or (n_time,)
            # Squeeze trailing singleton dimensions (e.g., single node)
            data = data.squeeze()
            if data.ndim > 1:
                # Multi-node: plot each node
                for node_idx in range(data.shape[-1]):
                    ax.plot(time, data[:, node_idx], alpha=0.7, **kwargs)
            else:
                ax.plot(time, data, **kwargs)

            if ax_values is not None:
                val = float(ax_values[i])
                ax.set_ylabel(f"{ax_info.name}={val:.4g}")

        axes[-1].set_xlabel("Time" + (f" (dt={self.dt})" if self.dt else " (steps)"))
        fig.suptitle(f"{self.name}: {output_label}" if self.name else output_label)
        plt.tight_layout()
        plt.close()
        return fig

    def _plot_scalar(self, figsize=None, ax=None, cmap="viridis", levels=20, colorbar=True, **kwargs):
        """Plot scalar results as line plot (1D) or filled-contour heatmap (2D)."""
        if self.results is None:
            return None
        grid = self.as_grid()
        if grid is None:
            return None

        def _axis_name(axis_info, default):
            return axis_info.get("name", default) if isinstance(axis_info, dict) else getattr(axis_info, "name", default)

        def _axis_values(axis_info, default_size):
            if isinstance(axis_info, dict):
                explored_values = axis_info.get("explored_values")
                lo = axis_info.get("lo")
                hi = axis_info.get("hi")
                n = axis_info.get("n")
            else:
                explored_values = getattr(axis_info, "explored_values", None)
                lo = getattr(axis_info, "lo", None)
                hi = getattr(axis_info, "hi", None)
                n = getattr(axis_info, "n", None)

            if explored_values is not None:
                values = np.asarray(explored_values)
                if values.size > 0:
                    return values

            if lo is not None and hi is not None:
                n_points = int(n) if n is not None else int(default_size)
                return np.linspace(float(lo), float(hi), n_points)

            return np.arange(default_size)

        if len(self._grid_shape) == 1:
            ax_info = self.axes[0]
            values = _axis_values(ax_info, self._grid_shape[0])
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (8, 4))
            else:
                fig = ax.figure
            ax.plot(values, np.asarray(grid), "o-", **kwargs)
            ax.set_xlabel(_axis_name(ax_info, "param"))
            ax.set_ylabel(self.observable or "value")
            ax.set_title(self.name or "Exploration")
            return fig
        elif len(self._grid_shape) == 2:
            ax0, ax1 = self.axes[0], self.axes[1]
            xv = _axis_values(ax0, self._grid_shape[0])
            yv = _axis_values(ax1, self._grid_shape[1])
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (8, 6))
            else:
                fig = ax.figure
            cs = ax.contourf(xv, yv, np.asarray(grid).T, levels=levels, cmap=cmap, **kwargs)
            ax.set_xlabel(_axis_name(ax0, "axis 0"))
            ax.set_ylabel(_axis_name(ax1, "axis 1"))
            if self.name:
                ax.set_title(self.name)
            if colorbar:
                fig.colorbar(cs, ax=ax, label=self.observable or "value", shrink=0.7)
            return fig
        return None

    def slice(self, **fixed_params):
        """Get a slice of results with some parameters fixed.

        Example: result.slice(G=0.5) returns 1D slice at G=0.5
        """
        grid_results = self.as_grid()
        if grid_results is None:
            return None

        # Find indices for fixed parameters
        indices = [slice(None)] * len(self.axes)
        for param_name, param_value in fixed_params.items():
            for i, ax in enumerate(self.axes):
                ax_name = self._axis_name(ax)
                ax_values = self._axis_values(ax)
                if ax_name == param_name and ax_values is not None:
                    # Find closest index
                    idx = int(jnp.argmin(jnp.abs(jnp.array(ax_values) - param_value)))
                    indices[i] = idx
        return grid_results[tuple(indices)]

    def __repr__(self):
        shape_str = "x".join(str(s) for s in self.shape) if self.shape else "empty"
        if self.is_timeseries:
            ts_shape = tuple(self.results.shape) if self.results is not None else ()
            return f"ExplorationResult(name='{self.name}', grid={shape_str}, timeseries={ts_shape})"
        opt_str = f", optimal={self.optimal.value:.4f}" if hasattr(self.optimal, "value") else ""
        return f"ExplorationResult(name='{self.name}', shape={shape_str}{opt_str})"


class ObservationResult(Bunch):
    """Result from an observation pipeline with named outputs.

    Exposes pipeline outputs as attributes (e.g., result.psd, result.frequencies)
    while maintaining NativeSolution-like interface (.data, .time, .dt).
    """

    @property
    def data(self):
        """Primary data output (alias for ys)."""
        return getattr(self, "ys", None)

    @property
    def time(self):
        """Time array (alias for ts)."""
        return getattr(self, "ts", None)


def _free_param_names(source) -> set:
    """Names of the model's free (tunable) parameters — dynamics + coupling.

    These are the parameters an algorithm tunes (e.g. wLRE / wFFI / J_i for EIB); their
    fitted values are the operating point a ``from_experiment`` warm-start reloads as a
    prior location (persisted as ``estimate__<param>`` in :meth:`ExperimentResult.save`).
    State variables are never parameters, so filtering to these can never collide with
    the settled ``<sv>_final`` state observations. Empty set when *source* is absent.
    """
    names: set = set()
    if source is None:
        return names

    def _vals(coll):
        if coll is None:
            return []
        return list(coll.values()) if hasattr(coll, "values") else list(coll)

    def _add_free(params):
        for p in _vals(params):
            if getattr(p, "free", False) and getattr(p, "name", None):
                names.add(p.name)

    def _couplings(obj):
        # network.coupling is a name->Coupling dict; experiment.coupling is a single
        # Coupling. Accept dict / list / single object so either shape resolves.
        if obj is None:
            return []
        if hasattr(obj, "values"):
            return list(obj.values())
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return [obj]

    _add_free(getattr(getattr(source, "dynamics", None), "parameters", None))
    net = getattr(source, "network", None)
    for c in _couplings(getattr(net, "coupling", None)) + _couplings(getattr(source, "coupling", None)):
        _add_free(getattr(c, "parameters", None))
    return names


def _algo_tuned_params(source) -> dict:
    """Map each algorithm name to the set of free parameters it FITS.

    A parameter counts as fit by an algorithm when an ``update_rule`` targets it — the
    algorithm's own rules or, recursively, those of an algorithm it ``includes``. Lets
    ``estimate__<param>`` be sourced from the algorithm that actually tunes a parameter
    rather than one that merely carries it at its initial value (e.g. a FIC pre-pass that
    holds ``wLRE``/``wFFI`` fixed must not shadow the EIB pass that fits them). Empty dict
    when *source* exposes no introspectable algorithms; each present algorithm maps to a
    (possibly empty) set.
    """

    def _as_list(coll):
        if coll is None:
            return []
        if hasattr(coll, "values"):
            return list(coll.values())
        if isinstance(coll, (list, tuple)):
            return list(coll)
        return [coll]

    algos = getattr(source, "algorithms", None)
    if not algos:
        return {}
    by_name = algos if hasattr(algos, "get") else {str(getattr(a, "name", i)): a for i, a in enumerate(_as_list(algos))}

    def _targets(algo):
        out = set()
        for rule in _as_list(getattr(algo, "update_rules", None)):
            tp = getattr(rule, "target_parameter", None)
            nm = getattr(tp, "name", None) or (str(tp) if tp is not None else None)
            if nm:
                out.add(str(nm))
        return out

    def _tuned(name, seen):
        algo = by_name.get(name)
        if algo is None or name in seen:
            return set()
        seen.add(name)
        out = _targets(algo)
        for inc in _as_list(getattr(algo, "includes", None)):
            inc_name = getattr(inc, "algorithm", None)
            inc_name = getattr(inc_name, "name", None) or inc_name
            if inc_name:
                out |= _tuned(str(inc_name), seen)
        return out

    return {name: _tuned(name, set()) for name in by_name}


class ExperimentResult:
    """Result from a complete experiment run.

    Mirrors the SimulationExperiment schema structure: integration, algorithms,
    optimizations, explorations, continuations. Accepts both new-style explicit
    fields and old-style ``results=Bunch`` constructor for backward compatibility.

    Attributes
    ----------
    integration : SimulationResult or None
        Primary simulation output with its observations and transient.
    algorithms : dict
        Algorithm results keyed by name.
    optimizations : dict
        Optimization results keyed by name.
    explorations : dict
        Exploration results keyed by name.
    continuations : dict
        Bifurcation/continuation results keyed by name.
    data_sources : dict
        External/empirical data (not from simulations).
    name : str or None
        Experiment name.
    source : SimulationExperiment or None
        Back-reference to input specification.
    """

    _output_sections = {"integration", "algorithms", "optimizations", "explorations", "continuations"}

    def __init__(
        self,
        integration=None,
        explorations=None,
        algorithms=None,
        optimizations=None,
        continuations=None,
        data_sources=None,
        name=None,
        source=None,
        **kwargs,
    ):
        self._extras = {}

        # ── Backward compat: ExperimentResult(results_bunch, experiment_name=...) ──
        experiment_name = kwargs.pop("experiment_name", None)
        if integration is not None and not isinstance(integration, SimulationResult) and hasattr(integration, "keys"):
            results = integration
            integration = results.get("integration")
            algorithms = results.get("algorithms", algorithms)
            optimizations = results.get("optimizations", optimizations)
            explorations = results.get("explorations", explorations)
            continuations = results.get("continuations", continuations)
            # Preserve extra keys (state, model_fn, timings, etc.)
            for k, v in results.items():
                if k not in ("integration", "algorithms", "optimizations", "explorations", "continuations", "data_sources"):
                    self._extras[k] = v

        # Also handle keyword: ExperimentResult(results=bunch, ...)
        results_kw = kwargs.pop("results", None)
        if results_kw is not None and integration is None:
            if hasattr(results_kw, "keys"):
                integration = results_kw.get("integration")
                algorithms = algorithms or results_kw.get("algorithms")
                optimizations = optimizations or results_kw.get("optimizations")
                explorations = explorations or results_kw.get("explorations")
                continuations = continuations or results_kw.get("continuations")
                for k, v in results_kw.items():
                    if k not in (
                        "integration",
                        "algorithms",
                        "optimizations",
                        "explorations",
                        "continuations",
                        "data_sources",
                    ):
                        self._extras[k] = v

        self.integration = integration
        self.algorithms = algorithms or {}
        self.optimizations = optimizations or {}
        self.explorations = explorations or {}
        self.continuations = continuations or {}
        self.data_sources = data_sources or {}
        self.name = name or experiment_name
        self.source = source
        self._extras.update(kwargs)

        # Link integration back to this ExperimentResult
        if integration is not None:
            integration._source = self

        # Inject variable units from source dynamics into integration result
        if source is not None and integration is not None and not integration._units:
            dynamics = getattr(source, "dynamics", None)
            if dynamics is not None:
                units = {}
                for n, sv in getattr(dynamics, "state_variables", {}).items():
                    u = getattr(sv, "unit", None)
                    if u is not None:
                        units[str(n)] = str(u)
                for n, dv in getattr(dynamics, "derived_variables", {}).items():
                    u = getattr(dv, "unit", None)
                    if u is not None:
                        units[str(n)] = str(u)
                integration._units = units

    # Singular-to-plural aliases for back-compat with docs/notebooks that
    # access result.exploration.X / result.optimization.X / etc.
    _singular_aliases = {
        "exploration": "explorations",
        "optimization": "optimizations",
        "algorithm": "algorithms",
        "continuation": "continuations",
    }

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        # Check extras first
        if name in self._extras:
            return self._extras[name]
        # Singular -> plural alias (e.g. result.exploration -> result.explorations)
        plural = self._singular_aliases.get(name)
        if plural is not None:
            val = self.__dict__.get(plural)
            if val is not None:
                return val
        # Delegate to integration for backward compat (result.data, result.time, etc.)
        integration = self.__dict__.get("integration")
        if integration is not None and hasattr(integration, name):
            return getattr(integration, name)
        # Delegate to single continuation (bifurcation results)
        continuations = self.__dict__.get("continuations", {})
        if continuations and len(continuations) == 1:
            cont = next(iter(continuations.values()))
            if hasattr(cont, name):
                return getattr(cont, name)
        raise AttributeError(f"'ExperimentResult' has no attribute '{name}'")

    def __contains__(self, key):
        if key in self._output_sections:
            val = getattr(self, key, None)
            return val is not None and val != {}
        return key in self._extras

    def _recorded_observation_names(self) -> set:
        """Observation names to persist: leaves plus anything flagged ``record``.

        An observation is recorded when it is either explicitly ``record: true``
        or *terminal* — not consumed as a ``source`` by another observation or by
        an optimization loss. ``record: false`` always drops it. This keeps final
        results (a fitted FC, an effective-frequency map) while omitting
        intermediates (a raw BOLD feeding an FC, an FC feeding a loss), which are
        recomputable from the recipe in the sidecar. Falls back to keeping every
        observation when the experiment carries no observation definitions.
        """
        exp = self.source
        obs_defs = getattr(exp, "observations", None) or {}
        present = set(getattr(self, "observations", None) or {})
        if not obs_defs:
            return present
        consumed: set = set()

        def _mark(ref):
            s = str(getattr(ref, "name", ref))
            if s in obs_defs:  # bare observation name
                consumed.add(s)
            elif s.startswith("observations."):  # observations.<name>[.data]
                base = s.split(".")[1]
                if base in obs_defs:
                    consumed.add(base)

        for o in obs_defs.values():
            for src in getattr(o, "source", None) or []:
                _mark(src)
        for opt in (getattr(exp, "optimizations", None) or {}).values():
            loss = getattr(opt, "loss", None)
            for arg in (getattr(loss, "arguments", None) or {}).values():
                if getattr(arg, "value", None) is not None:
                    _mark(arg.value)

        keep = set()
        for name, o in obs_defs.items():
            rec = getattr(o, "record", None)
            if rec is True:
                keep.add(name)
            elif rec is False:
                continue
            elif name not in consumed:
                keep.add(name)
        return keep & present if present else keep

    def _cohort_subject_states(self):
        """Unstack an on-device cohort into per-subject tuned states, or None.

        The on-device cohort driver (``dataset.batch_mode == on_device``) returns
        ONE batched tuned state per algorithm — a leading subject axis over the
        whole cohort — instead of a per-subject :class:`AlgorithmResult`, plus the
        cohort's ``subject_ids``. Every array leaf carries the subject axis at
        position 0, so slicing it apart yields one per-subject state, saved exactly
        like the per-subject fan-out (one result per subject). Returns
        ``(subject_ids, [{algo_name: per_subject_AlgorithmResult}, ...])``, or
        ``None`` for an ordinary run so the normal single-result save path runs.
        """
        algos = self.algorithms or {}
        batched = [(n, a) for n, a in algos.items() if getattr(a, "cohort_state", None) is not None]
        if not batched:
            return None
        subject_ids = list(getattr(batched[0][1], "subject_ids", None) or [])
        if not subject_ids:
            raise ValueError(
                "On-device cohort result carries a batched cohort_state but no subject_ids "
                "to split it by. Every leaf is a (n_subjects, ...) batch, so without the "
                "cohort's subject ids it cannot be written as per-subject results."
            )
        per_subject = []
        for i in range(len(subject_ids)):
            algos_i = {}
            for name, algo in batched:
                state_i = jax.tree_util.tree_map(
                    lambda x, _i=i: x[_i] if hasattr(x, "ndim") else x,
                    algo.cohort_state,
                )
                algos_i[name] = AlgorithmResult(name=name, state=state_i)
            per_subject.append(algos_i)
        return subject_ids, per_subject

    def _save_per_subject(self, out_dir, cohort, compress, record_only):
        """Persist an on-device cohort as one ``sub-<id>_..._result`` per subject.

        Mirrors the per-subject fan-out: each subject file carries only that
        subject's tuned parameters (``estimate__<param>``) — on-device tuning
        produces per-subject parameters, not a per-subject trajectory, so the
        shared base run's observations/integration are not duplicated per subject.
        """
        subject_ids, per_subject = cohort
        src = self.source
        if src is None:
            raise ValueError(
                "Cannot save an on-device cohort without a source experiment: the "
                "per-subject result stem (sub-<id>_...) comes from the source's "
                "_active_subject, so every subject would overwrite the same file."
            )
        _saved_active = getattr(src, "_active_subject", None)
        written = []
        try:
            for sid, algos_i in zip(subject_ids, per_subject):
                src._active_subject = str(sid)  # drives the sub-<id>_ result stem
                view = copy.copy(self)
                view.algorithms = algos_i
                view.integration = None
                view.observations = Bunch()
                view.explorations = {}
                view.optimizations = {}
                view.continuations = {}
                written += ExperimentResult.save(view, out_dir, compress=compress, record_only=record_only)
        finally:
            src._active_subject = _saved_active
        return written

    def save(self, out_dir, compress: bool = True, record_only: bool = True):
        """Persist the run as one keyed HDF5 result plus a YAML provenance sidecar.

        Writes ``<prefix>_result.h5`` — a single xarray ``Dataset`` where every
        output is a data-variable and the sweep parameters are shared coordinates
        (a full run is gridded; a sharded run keeps the flat, self-describing
        ``point`` dim that reassembles by value) — and ``<prefix>_result.yaml``,
        the frozen experiment spec. ``<prefix>`` is the experiment's BIDS-style
        key-value name (``ses-<id>_desc-<label>``). The **same** artifact is
        produced by a local run and by the HPC gather pass, so they are
        interchangeable. Returns the written paths.

        An on-device cohort run fans here into one per-subject result (see
        :meth:`_save_per_subject`), mirroring the per-subject workflow fan-out.
        """
        import os

        _cohort = self._cohort_subject_states()
        if _cohort is not None:
            return self._save_per_subject(out_dir, _cohort, compress, record_only)

        os.makedirs(out_dir, exist_ok=True)

        def _san(s):
            return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))

        # ── collect every output as a data-variable ──────────────────────────
        by_output: dict[tuple, "xr.DataArray"] = {}
        for expl_name, expl in (self.explorations or {}).items():
            # ExplorationResult labels every observation as a DataArray at
            # construction (grid and warm-start alike), so this stays uniform.
            for obs_name, da in (getattr(expl, "observations", None) or {}).items():
                if da is not None and hasattr(da, "dims"):
                    by_output[(_san(expl_name), _san(obs_name))] = da
            if getattr(expl, "results", None) is not None:
                try:
                    g = expl.as_grid()
                except Exception:
                    g = None
                if g is not None and hasattr(g, "dims"):
                    by_output[(_san(expl_name), "results")] = g
        outputs = [o for _, o in by_output]
        data_vars = {(o if outputs.count(o) == 1 else f"{e}__{o}"): da for (e, o), da in by_output.items()}
        integ_obs = getattr(self.integration, "observations", None) if self.integration is not None else None
        if integ_obs and hasattr(integ_obs, "items"):
            for obs_name, obs in integ_obs.items():
                da = _unwrap_observation(obs)
                if hasattr(da, "dims"):
                    data_vars[f"integration__{_san(obs_name)}"] = da

        # Experiments that produce observations/optimizations without an exploration
        # sweep (e.g. a per-subject FC fit) still carry data to persist: the derived
        # observations (simulated + reconciled empirical FC) and the fit outcome
        # (fitted parameters, final loss, loss trajectory). Coerce to float and skip
        # anything non-numeric so the HDF5 write never trips on Python objects.
        def _numeric_da(name, arr):
            if arr is None:
                return None
            try:
                a = np.asarray(getattr(arr, "values", arr), dtype=float)
            except (ValueError, TypeError):
                try:  # jax/0-d scalar wrapped as object
                    a = np.asarray(float(arr), dtype=float)
                except (ValueError, TypeError):
                    return None
            if a.dtype == object or a.size == 0:
                return None
            if a.ndim == 0:
                return xr.DataArray(a)
            # An already-labelled value keeps its own dims and coords: observations are
            # named at construction (`_observation_dataarray`), and re-deriving names here,
            # from shape alone, could only contradict them.
            if getattr(arr, "dims", None) and len(arr.dims) == a.ndim:
                return xr.DataArray(a, dims=[str(d) for d in arr.dims], coords=getattr(arr, "coords", None))
            return xr.DataArray(a, dims=[f"{name}_d{i}" for i in range(a.ndim)])

        def _numeric_leaves(prefix, obj):
            """Yield (var_name, DataArray) for the numeric leaves of a nested pytree."""
            leaf = _numeric_da(prefix.rsplit("__", 1)[-1], obj)
            if leaf is not None:
                yield prefix, leaf
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield from _numeric_leaves(f"{prefix}__{_san(k)}", v)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    yield from _numeric_leaves(f"{prefix}__{i}", v)

        keep_obs = self._recorded_observation_names() if record_only else set(getattr(self, "observations", None) or {})
        for obs_name, obs in (getattr(self, "observations", None) or {}).items():
            if obs_name not in keep_obs:
                continue
            key = f"observation__{_san(obs_name)}"
            # Flatten via _numeric_leaves, not _numeric_da: an observation may return a
            # nested pytree (e.g. a per-hemisphere wave metric {lh:{...}, rh:{...}}), which
            # _numeric_da drops whole (np.asarray(dict) raises → None → silently unsaved).
            # For an array value _numeric_leaves yields the single leaf unchanged, so this is
            # a superset — the same flattening already used for optimization fitted params.
            for var, da in _numeric_leaves(key, _unwrap_observation(obs)):
                if var not in data_vars:
                    data_vars[var] = da
        for opt_name, opt in (self.optimizations or {}).items():
            for field in ("final_loss", "loss_trajectory"):
                da = _numeric_da(field, getattr(opt, field, None))
                if da is not None:
                    data_vars[f"optimization__{_san(opt_name)}__{field}"] = da
            fitted = getattr(opt, "fitted_params", None)
            if fitted is not None:
                for name, da in _numeric_leaves(f"optimization__{_san(opt_name)}__fitted", fitted):
                    data_vars[name] = da

        # Persist algorithm post-tuning observations (achieved fc_corr / fc_rmse / fc) so the fit outcome is legible from the saved result, not only the tuned parameters.
        for algo_name, algo in (self.algorithms or {}).items():
            post = getattr(algo, "post_tuning", None)
            post_obs = getattr(post, "observations", None) if post is not None else None
            for obs_name, obs in (post_obs or {}).items():
                for var, da in _numeric_leaves(f"algorithm__{_san(algo_name)}__{_san(obs_name)}", _unwrap_observation(obs)):
                    if var not in data_vars:
                        data_vars[var] = da

        # Persist each tuned FREE parameter's fitted value as ``estimate__<param>`` so a
        # from_experiment warm-start can reload it as a prior location (point prior). Kept
        # on LABELLED node axes (``node`` for vectors, ``node_i``+``node_j`` for per-edge
        # matrices) — the same convention FC matrices use — so the consumer reconciles by
        # label with the existing `.sel` path, no bespoke reindex. Container-layer only
        # (values already live in AlgorithmResult.state) → no codegen change; free params
        # only, so it can't shadow a ``<sv>_final`` key; sourced from the algorithm that FITS
        # each param (last-writer among fitting passes, see _algo_tuned_params).
        free_names = _free_param_names(self.source) if self.algorithms else set()
        if free_names:  # nothing tunable → skip the flatten entirely
            # Use the RESOLVED node labels (hydrates `bids:` placeholders like region_<i>)
            # so the estimate coords match what the consumer's _resolve_model_node_labels
            # produces — else the warm-start `.sel` reconcile can't align. Fall back to the
            # raw labels when the source can't resolve (e.g. an inline-network source).
            _get = getattr(self.source, "_resolve_model_node_labels", None)
            src_labels = (_get() if callable(_get) else None) or getattr(
                getattr(self.source, "network", None), "node_labels", None
            )
            src_labels = [str(lbl) for lbl in src_labels] if src_labels else None
            nn = len(src_labels) if src_labels else None
            # Source each estimate from the algorithm that FITS the param (last/most-converged wins), not one that merely carries its init value.
            tuned_by = _algo_tuned_params(self.source)
            for algo_name, algo in self.algorithms.items():
                fits = tuned_by.get(algo_name, free_names)
                for dotted, arr in OptimizationResult._flatten_params(getattr(algo, "state", None)).items():
                    param = dotted.rsplit(".", 1)[-1]
                    key = f"estimate__{_san(param)}"
                    if param not in free_names or param not in fits:
                        continue
                    a = np.asarray(getattr(arr, "values", arr))
                    if a.dtype == object or a.size == 0:
                        continue
                    # Label per-node vectors / per-edge matrices so the consumer reconciles
                    # by label with `.sel`; anything else (scalar) stays unlabelled.
                    label_dims = {1: ["node"], 2: ["node_i", "node_j"]}.get(a.ndim)
                    if nn and label_dims and all(s == nn for s in a.shape):
                        da = xr.DataArray(a, dims=label_dims, coords={d: src_labels for d in label_dims})
                    else:
                        da = _numeric_da(key, a)
                    if da is not None:
                        data_vars[key] = da

        # Continuation branches (bifurcation results) persist through the SAME native
        # Dataset — no per-figure array dump. Each branch keeps its own ``step``
        # dimension (renamed unique) so multiple branches and the sweep grid coexist;
        # the continuation parameter and observables become data variables.
        for cont_name, bifres in (self.continuations or {}).items():
            to_ds = getattr(bifres, "to_dataset", None)
            if not callable(to_ds):
                continue
            cds = to_ds()
            if "step" not in getattr(cds, "sizes", {}):
                continue
            dim = f"continuation__{_san(cont_name)}__step"
            cds = cds.rename({"step": dim}).reset_coords()  # ICS coord (e.g. G) → data var
            for vname, da in cds.data_vars.items():
                if da.dtype != object:  # skip special-point label strings (HDF5 object dtype)
                    data_vars[f"continuation__{_san(cont_name)}__{_san(vname)}"] = da

            # Child periodic-orbit branches (from a Hopf point) hang off the equilibrium
            # branch in ``periodic_orbits`` and were previously dropped by the save, so a
            # PO branch's amplitude envelope (max/min per state var) and period never
            # reached the ``.h5``. Serialize each under a nested ``__<po>__`` name so the
            # full bifurcation diagram (Fig-2 periodic branch, Fig-3A period divergence)
            # is reproducible from ``tvbo run`` alone.
            for i, po in enumerate(getattr(bifres, "periodic_orbits", None) or []):
                po_to_ds = getattr(po, "to_dataset", None)
                if not callable(po_to_ds):
                    continue
                po_ds = po_to_ds()
                if "step" not in getattr(po_ds, "sizes", {}):
                    continue
                po_name = _san(getattr(po, "name", None) or f"po{i}")
                pdim = f"continuation__{_san(cont_name)}__{po_name}__step"
                po_ds = po_ds.rename({"step": pdim}).reset_coords()
                for vname, da in po_ds.data_vars.items():
                    if da.dtype != object:
                        data_vars[f"continuation__{_san(cont_name)}__{po_name}__{_san(vname)}"] = da

                # Orbit waveforms: the adapter attaches ``orbit_profiles``
                # ([n_steps, n_phase, n_vars], phase-resampled over one period) when the
                # engine reconstructs them. Serialize as one 3-D var so every orbit's actual
                # E(t)/x(t)/u(t) profile (Fig-3B morphologies, Fig-3C orbit) is reproducible.
                prof = getattr(po, "orbit_profiles", None)
                if prof is not None:
                    prof = np.asarray(prof, dtype=float)
                    n_po = po_ds.sizes.get(pdim)
                    if prof.ndim == 3 and (n_po is None or prof.shape[0] == n_po):
                        _mdl = getattr(po, "model", None)
                        _svs = getattr(_mdl, "state_variables", None)
                        if hasattr(_svs, "keys"):
                            _vn = list(_svs.keys())
                        elif _svs is not None:
                            _vn = [getattr(s, "name", f"v{j}") for j, s in enumerate(_svs)]
                        else:
                            _vn = []
                        _vn = (_vn + [f"v{j}" for j in range(len(_vn), prof.shape[2])])[: prof.shape[2]]
                        _pdim = f"continuation__{_san(cont_name)}__{po_name}__phase"
                        _vdim = f"continuation__{_san(cont_name)}__{po_name}__var"
                        data_vars[f"continuation__{_san(cont_name)}__{po_name}__profile"] = xr.DataArray(
                            prof, dims=[pdim, _pdim, _vdim], coords={_pdim: np.linspace(0.0, 1.0, prof.shape[1]), _vdim: _vn}
                        )

        # Spiking backends (Brian2) carry a raster in ``_extras["spikes"]`` — persist it so a
        # spiking run reproduces from the container: per-population spike times + neuron indices
        # as flat 1D variables (each population its own length), plus the population firing rates
        # and sizes on a shared ``population`` axis, and the run window in the Dataset attrs.
        # General to any spiking run; guarded on the presence of spikes.
        _spk = self._extras.get("spikes")
        if _spk:
            _rates = self._extras.get("rates") or {}
            _sizes = self._extras.get("sizes") or {}
            _pops = list(_spk)
            # Key the population axis by the same filename-safe token the per-population raster
            # variables use (``spikes__<key>__t/i``), so a consumer can select a rate by name and
            # map it straight to that population's raster — never a positional zip against attrs.
            _pops_key = [_san(p) for p in _pops]
            for pop, key in zip(_pops, _pops_key):
                t = np.asarray(_spk[pop].get("t_ms"), dtype=float)
                idx = np.asarray(_spk[pop].get("i"), dtype=float)
                dim = f"spike__{key}"
                data_vars[f"spikes__{key}__t"] = xr.DataArray(t, dims=[dim])
                data_vars[f"spikes__{key}__i"] = xr.DataArray(idx, dims=[dim])
            if _rates:
                data_vars["firing_rate"] = xr.DataArray(
                    np.asarray([_rates.get(p, np.nan) for p in _pops], dtype=float),
                    dims=["population"],
                    coords={"population": _pops_key},
                )
            if _sizes:
                data_vars["population_size"] = xr.DataArray(
                    np.asarray([_sizes.get(p, 0) for p in _pops], dtype=float),
                    dims=["population"],
                    coords={"population": _pops_key},
                )
            self._extras.setdefault("_spike_pops", _pops)

        # Recorded synapse-internal state (u, x): the continuous population-mean trace measured
        # by the backend's observation probe, one time series per recorded variable, on its own
        # (coarser) time axis. Keyed by the filename-safe source-population name.
        _syn = self._extras.get("synapse_state")
        if _syn:
            for key, d in _syn.items():
                sk = _san(key)
                dim = f"syntime__{sk}"
                tvals = np.asarray(d["t_ms"], dtype=float)
                for var, arr in d["vars"].items():
                    data_vars[f"synapse__{sk}__{var}"] = xr.DataArray(
                        np.asarray(arr, dtype=float), dims=[dim], coords={dim: tvals}
                    )

        # Fallback: a pure forward simulation (no sweep, no declared observations, no
        # continuation, no optimization) still carries its recorded trajectory in
        # integration.data. Persist it so `tvbo run` reproduces a raw forward run — e.g. a
        # NeuroML EPSP-train experiment — as a native container instead of writing nothing.
        # Guarded on an otherwise-empty data_vars, so exploration/observation runs are untouched.
        if not data_vars and self.integration is not None:
            _idata = getattr(self.integration, "data", None)
            if _idata is not None and hasattr(_idata, "dims"):
                data_vars["integration"] = _idata

        stem = "result"
        if self.source is not None and hasattr(self.source, "get_result_stem"):
            try:
                stem = self.source.get_result_stem()  # BIDS pybids-generated, filename-safe
            except Exception:
                stem = "result"

        # A shard's provenance sidecar is written once by the gather pass, not per task.
        is_shard = any(_is_partial_shard(e) for e in (self.explorations or {}).values())

        # Several explorations in one experiment each write a `<expl>__results` variable;
        # their sweep dims can share a name (`point`, `K[0]`, …) at different sizes, which
        # `xr.Dataset` rejects. Rename the colliding dim per-variable so they coexist. Fires
        # only on a real conflict, so single-exploration/single-sweep experiments are untouched.
        if len(data_vars) > 1:
            from collections import defaultdict

            _dim_sizes: dict = defaultdict(set)
            for _da in data_vars.values():
                for _d, _s in zip(_da.dims, _da.shape):
                    _dim_sizes[_d].add(int(_s))
            _conflicting = {_d for _d, _sizes in _dim_sizes.items() if len(_sizes) > 1}
            if _conflicting:
                data_vars = {
                    _vn: (
                        _da.rename({_d: f"{_vn}__{_d}" for _d in _da.dims if _d in _conflicting})
                        if any(_d in _conflicting for _d in _da.dims)
                        else _da
                    )
                    for _vn, _da in data_vars.items()
                }

        written = []
        if data_vars:
            _attrs = {"tvbo_class": "tvbo:ExperimentResult", "sidecar_file": f"{stem}.yaml"}
            if self._extras.get("spikes"):
                # The same filename-safe token used for the raster variables and the population
                # coord, so attrs, coord and variable names all agree (no raw-vs-sanitised drift).
                _attrs["populations"] = [_san(p) for p in self._extras["spikes"]]
                for _k in ("duration_ms", "dt_ms"):
                    if self._extras.get(_k) is not None:
                        _attrs[_k] = float(self._extras[_k])
            if self._extras.get("synapse_state"):
                _attrs["synapse_recorded"] = [_san(k) for k in self._extras["synapse_state"]]
            ds = xr.Dataset(data_vars, attrs=_attrs)
            h5 = os.path.join(out_dir, f"{stem}.h5")
            # Grids of trajectories/observations compress well (repeated structure,
            # smooth fields), so gzip-deflate by default; `compress=False` opts out
            # for max write speed. complevel 4 is the deflate speed/size sweet spot.
            encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars} if compress else None
            # Single self-describing format; a write failure raises, no lossy fallback.
            ds.to_netcdf(h5, engine="h5netcdf", encoding=encoding)
            written.append(h5)

        if not is_shard and written and self.source is not None and hasattr(self.source, "freeze_yaml"):
            try:
                # Self-contained provenance: spec + connectome companion
                # (<stem>_network.h5), reproducible on reload without data sources.
                yaml_text = self.source.freeze_yaml(out_dir, network_stem=f"{stem}_network")
                yaml_path = os.path.join(out_dir, f"{stem}.yaml")
                with open(yaml_path, "w", encoding="utf-8") as fh:
                    fh.write(yaml_text)
                written.append(yaml_path)
            except Exception:
                logger.warning("provenance sidecar %s.yaml not written", stem, exc_info=True)
            # BEP034 alignment: a JSON metadata sidecar (BIDS tooling reads JSON)
            # beside the richer YAML re-run recipe, and a dataset_description.json
            # marking out_dir as a BIDS-derivatives dataset.
            try:
                written += self._write_bep034_sidecars(out_dir, stem)
            except Exception:
                logger.warning("BEP034 sidecars for %s not written", stem, exc_info=True)
        return written

    def _write_bep034_sidecars(self, out_dir, stem) -> list:
        """Write a BEP034 JSON metadata sidecar + a derivatives dataset_description.json.

        Complements the YAML re-run recipe with BIDS-standard JSON so the result is
        discoverable by pybids/BIDS tooling. The gridded HDF5 itself supersedes
        emitting one BEP034 ``ts/`` file per sweep cell (a 15,600-cell grid would be
        15,600 files); the sidecar records the model, integrator, and swept space so
        the mapping back to per-cell simulations is explicit.
        """
        import datetime as _dt
        import json as _json
        import os

        from tvbo.adapters.bids import DatasetDescription, SimulationProvenance

        exp = self.source
        integ = getattr(exp, "integration", None)
        dyn = getattr(exp, "dynamics", None)
        now = _dt.datetime.now().isoformat(timespec="seconds")
        prov = SimulationProvenance(
            Model=getattr(dyn, "name", None) or getattr(dyn, "label", None),
            Integrator=getattr(integ, "method", None),
            Duration=getattr(integ, "duration", None),
            StepSize=getattr(integ, "step_size", None),
            GeneratedAt=now,
            Software="tvbo",
        )
        # Swept parameter space (the grid axes) — the metadata a per-cell BEP034
        # ``ts/`` file would carry, aggregated for the whole grid.
        space = {}
        for expl in (getattr(exp, "explorations", None) or {}).values():
            for ax in getattr(expl, "space", None) or []:
                pname = getattr(ax, "parameter", None)
                if pname:
                    space[str(pname)] = getattr(ax, "explored_values", None) or None
        sidecar = {**prov.to_dict(), "ModelingRecipe": f"{stem}.yaml"}
        if space:
            sidecar["SweptParameters"] = space
        json_path = os.path.join(out_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            _json.dump(sidecar, fh, indent=2, default=str)

        written = [json_path]
        dd = os.path.join(out_dir, "dataset_description.json")
        if not os.path.exists(dd):
            desc = DatasetDescription(
                Name=str(getattr(exp, "label", None) or getattr(exp, "id", None) or "tvbo results"),
                DatasetType="derivative",
                GeneratedBy=[{"Name": "tvbo", "Description": "TVB-Ontology simulation result"}],
            )
            with open(dd, "w", encoding="utf-8") as fh:
                fh.write(desc.to_json())
            written.append(dd)
        return written

    def plot(self, **kwargs):
        """Dispatch plot to the most relevant sub-result."""
        if self.integration is not None:
            return self.integration.plot(**kwargs)
        if self.continuations:
            cont = next(iter(self.continuations.values()))
            if hasattr(cont, "plot"):
                return cont.plot(**kwargs)
        if self.explorations:
            expl = next(iter(self.explorations.values()))
            if hasattr(expl, "plot"):
                return expl.plot(**kwargs)
        raise AttributeError("No plottable sub-result found")

    def __repr__(self):
        label = self.name or "Experiment"
        lines = [label]

        sections = []
        if self.integration is not None:
            sections.append(("integration", self.integration))
        if self.algorithms:
            sections.append(("algorithms", self.algorithms))
        if self.optimizations:
            sections.append(("optimizations", self.optimizations))
        if self.explorations:
            sections.append(("explorations", self.explorations))
        if self.continuations:
            sections.append(("continuations", self.continuations))

        for i, (section, val) in enumerate(sections):
            is_last_section = i == len(sections) - 1
            sec_conn = "└── " if is_last_section else "├── "
            sec_ext = "    " if is_last_section else "│   "

            lines.append(f"{sec_conn}{section}")

            if isinstance(val, (SimulationResult, AlgorithmResult, OptimizationResult, ExplorationResult)):
                detail_lines = self._format_details(val, sec_ext + "    ")
                lines.extend(detail_lines)
            elif hasattr(val, "keys"):
                child_keys = [k for k in val.keys() if not str(k).startswith("_")]
                for j, child_key in enumerate(child_keys):
                    is_last_child = j == len(child_keys) - 1
                    child_conn = "└── " if is_last_child else "├── "
                    child_ext = "    " if is_last_child else "│   "
                    child_val = val[child_key]
                    lines.append(f"{sec_ext}{child_conn}{child_key}")
                    detail_lines = self._format_details(child_val, sec_ext + child_ext)
                    lines.extend(detail_lines)

        return "\n".join(lines)

    def _format_details(self, val, prefix):
        """Format details of a result object."""
        details = []

        if isinstance(val, SimulationResult):
            shape = tuple(val.data.shape) if val.data is not None else None
            if shape:
                details.append(f"{prefix}data: {shape}")
            if val.observations:
                obs_keys = list(val.observations.keys())
                details.append(f"{prefix}observations: {obs_keys}")

        elif isinstance(val, AlgorithmResult):
            details.append(f"{prefix}n_iterations: {val.n_iterations}")
            if val.history:
                hist_keys = [k for k in val.history.keys() if not str(k).startswith("_")]
                details.append(f"{prefix}history: {hist_keys}")

        elif isinstance(val, OptimizationResult):
            details.append(f"{prefix}n_steps: {val.n_steps}")
            if val.final_loss is not None:
                details.append(f"{prefix}final_loss: {val.final_loss:.4f}")
            if val.history:
                hist_keys = [k for k in val.history.keys() if not str(k).startswith("_")]
                details.append(f"{prefix}history: {hist_keys}")
            if val.simulation and val.simulation.observations:
                obs_keys = list(val.simulation.observations.keys())
                details.append(f"{prefix}simulation.observations: {obs_keys}")

        elif isinstance(val, ExplorationResult):
            if val.axes:
                axis_names = [(ax.get("name", ax.name) if hasattr(ax, "get") else getattr(ax, "name", "?")) for ax in val.axes]
                details.append(f"{prefix}axes: {axis_names}")
            if val.shape:
                details.append(f"{prefix}shape: {val.shape}")
            if val.observable:
                details.append(f"{prefix}observable: {val.observable}")

        return details

    def _repr_markdown_(self):
        """Rich display for Jupyter notebooks."""
        return f"```\n{self.__repr__()}\n```"

    # ── Export ─────────────────────────────────────

    def export(self, output_dir, subject="01", session=None, description="tvbsim"):
        """Export results and metadata to a BIDS-compatible directory.

        Writes experiment specification as YAML and simulation data as
        netCDF/HDF5, following BEP034 directory conventions::

            output_dir/
            ├── dataset_description.json
            ├── sub-{subject}/
            │   ├── sub-{subject}_desc-{desc}_experiment.yaml
            │   └── ts/
            │       ├── sub-{subject}_desc-{desc}_ts-sim_State.nc
            │       ├── sub-{subject}_desc-{desc}_ts-sim_State.json
            │       └── sub-{subject}_desc-{desc}_ts-{obs}_BOLD.nc  (per observation)

        Parameters
        ----------
        output_dir : str or Path
            Root output directory (created if it doesn't exist).
        subject : str
            BIDS subject label (default ``"01"``).
        session : str or None
            BIDS session label (optional).
        description : str
            BIDS ``desc-`` entity (default ``"tvbsim"``).

        Returns
        -------
        pathlib.Path
            Path to the output directory.
        """
        from pathlib import Path
        import json

        output_dir = Path(output_dir)
        sub = f"sub-{subject}"
        ses = f"ses-{session}" if session else None
        desc = f"desc-{description}" if description else ""

        sub_dir = output_dir / sub
        if ses:
            sub_dir = sub_dir / ses
        sub_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{sub}_{desc}" if desc else sub

        # ── 1. Dataset description ────────────────────
        dd_path = output_dir / "dataset_description.json"
        if not dd_path.exists():
            dd = {
                "Name": self.name or "TVBO Simulation Experiment",
                "BIDSVersion": "1.9.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "tvbo", "Description": "The Virtual Brain Ontology"}],
            }
            dd_path.write_text(json.dumps(dd, indent=2))

        # ── 2. Experiment specification YAML ──────────
        if self.source is not None:
            yaml_path = sub_dir / f"{prefix}_experiment.yaml"
            self.source.to_yaml(str(yaml_path))

        # ── 3. Integration data ───────────────────────
        if self.integration is not None and self.integration.data is not None:
            ts_dir = sub_dir / "ts"
            ts_dir.mkdir(exist_ok=True)
            ts_prefix = f"{prefix}_ts-sim"

            self._write_data(
                self.integration.data,
                ts_dir / f"{ts_prefix}_State",
            )

            # Sidecar
            sidecar = self._build_sidecar(self.integration)
            (ts_dir / f"{ts_prefix}_State.json").write_text(json.dumps(sidecar, indent=2, default=str))

            # Observations
            for obs_name, obs in self.integration.observations.items():
                obs_path = ts_dir / f"{prefix}_ts-{obs_name}"
                if isinstance(obs, xr.DataArray):
                    self._write_data(obs, obs_path)
                elif hasattr(obs, "data"):
                    obs_da = _to_dataarray(
                        np.asarray(obs.data),
                        np.asarray(obs.time) if hasattr(obs, "time") else None,
                    )
                    if obs_da is not None:
                        self._write_data(obs_da, obs_path)

            # Transient
            if self.integration.transient is not None and self.integration.transient.data is not None:
                self._write_data(
                    self.integration.transient.data,
                    ts_dir / f"{prefix}_ts-transient_State",
                )

        # ── 4. Algorithm results ──────────────────────
        for algo_name, algo in self.algorithms.items():
            algo_dir = sub_dir / "ts"
            algo_dir.mkdir(exist_ok=True)
            if isinstance(algo, AlgorithmResult) and algo.post_tuning is not None:
                if algo.post_tuning.data is not None:
                    self._write_data(
                        algo.post_tuning.data,
                        algo_dir / f"{prefix}_ts-{algo_name}_State",
                    )

        # ── 5. Optimization results ───────────────────
        for opt_name, opt in self.optimizations.items():
            opt_dir = sub_dir / "ts"
            opt_dir.mkdir(exist_ok=True)
            if isinstance(opt, OptimizationResult) and opt.simulation is not None:
                if opt.simulation.data is not None:
                    self._write_data(
                        opt.simulation.data,
                        opt_dir / f"{prefix}_ts-{opt_name}_State",
                    )

        return output_dir

    @staticmethod
    def _write_data(da, path_stem):
        """Write an xr.DataArray to netCDF (preferred) or HDF5.

        Tries engines in order: h5netcdf → scipy (netCDF3).
        The file extension is set automatically (.nc).
        """
        from pathlib import Path

        path_stem = Path(path_stem)
        ds = da.to_dataset(name="data")
        nc_path = path_stem.with_suffix(".nc")
        for engine in ("h5netcdf", "scipy"):
            try:
                ds.to_netcdf(nc_path, engine=engine)
                return nc_path
            except ImportError:
                continue
        raise ImportError("netCDF export requires scipy or h5netcdf. Install with: pip install h5netcdf")

    @staticmethod
    def _build_sidecar(sim_result):
        """Build a JSON sidecar dict for a SimulationResult."""
        sidecar = {}
        if sim_result.data is not None:
            sidecar["Shape"] = list(sim_result.data.shape)
            sidecar["Dimensions"] = list(sim_result.data.dims)
            if "variable" in sim_result.data.coords:
                sidecar["StateVariables"] = list(sim_result.data.coords["variable"].values)
            if "node" in sim_result.data.coords:
                sidecar["Regions"] = list(sim_result.data.coords["node"].values)
            if sim_result.time is not None and len(sim_result.time) > 1:
                dt = float(sim_result.time[1] - sim_result.time[0])
                sidecar["SamplingPeriod"] = dt
                sidecar["SamplingPeriodUnit"] = "ms"
        if sim_result.observations:
            sidecar["Observations"] = list(sim_result.observations.keys())
        return sidecar

    # ── Class methods ─────────────────────────────

    @classmethod
    def from_timeseries(cls, ts, source=None, name=None, **extras):
        """Create an ExperimentResult from a TVBO TimeSeries.

        Converts a raw TimeSeries (as returned by JAX, PyRates,
        NetworkDynamics, etc.) into the standard ExperimentResult wrapper.

        Parameters
        ----------
        ts : TimeSeries
            Simulation output with ``.data``, ``.time``, ``.labels_dimensions``.
        source : SimulationExperiment, optional
            Back-reference to the experiment that produced this result.
        name : str, optional
            Experiment label.
        **extras
            Additional attributes to store (e.g. ``sol``, ``graph``).

        Returns
        -------
        ExperimentResult
        """
        data_np = np.asarray(ts.data)

        ld = ts.labels_dimensions if isinstance(ts.labels_dimensions, dict) else {}
        state_names = ld.get("State Variable", [])
        region_labels = ld.get("Region", [])

        dims = ["time", "variable", "node", "mode"][: data_np.ndim]
        coords = {}
        if ts.time is not None:
            coords["time"] = np.asarray(ts.time)
        if state_names:
            coords["variable"] = list(state_names)
        if region_labels and data_np.ndim >= 3:
            coords["node"] = [str(r) for r in region_labels]
        if "mode" in dims:
            coords["mode"] = list(range(data_np.shape[3]))

        da = xr.DataArray(data=data_np, dims=dims, coords=coords)

        # Collect observations from derivatives (TVB-style) or extras
        observations = {}
        if hasattr(ts, "derivatives") and ts.derivatives:
            for d_ts in ts.derivatives:
                obs_name = getattr(d_ts, "title", None) or f"obs_{len(observations)}"
                observations[obs_name] = d_ts

        sim_result = SimulationResult(data=da, observations=observations)
        sim_result._timeseries = ts

        # Continuations (bifurcation results) go in a separate section
        continuations = {}
        if hasattr(ts, "sol") and extras.get("_is_bifurcation", False):
            extras.pop("_is_bifurcation")
            continuation_name = getattr(ts.sol, "name", None)
            if not continuation_name and source is not None:
                source_continuations = getattr(source, "continuations", None) or {}
                if len(source_continuations) == 1:
                    continuation_name = next(iter(source_continuations.keys()))
            continuations[continuation_name or "default"] = ts.sol

        return cls(
            integration=sim_result,
            source=source,
            name=name,
            continuations=continuations or None,
            **extras,
        )

    @classmethod
    def from_tvb(cls, simulator, result=None):
        """Create an ExperimentResult from a TVB simulator and its run output.

        Wraps TVB simulation output into the standard TVBO result structure:

        - ``result.integration`` — primary monitor as SimulationResult (xr.DataArray)
        - ``result.integration.observations['MonitorName']`` — one TimeSeries per
          additional monitor, keyed by the TVB monitor class name

        Parameters
        ----------
        simulator : tvb.simulator.simulator.Simulator
            A configured TVB simulator.
        result : list of (time_array, data_array) tuples, optional
            Output of ``simulator.run()``. If *None*, the simulator is
            run using its ``simulation_length``.

        Returns
        -------
        ExperimentResult
        """
        if result is None:
            result = simulator.run()

        voi = list(simulator.model.variables_of_interest)
        region_labels = list(simulator.connectivity.region_labels)
        base_labels = {"State Variable": voi, "Region": region_labels}

        primary_ts = None
        primary_tv = None
        primary_xv = None
        observations = {}

        for monitor, (tv, xv) in zip(simulator.monitors, result):
            mon_labels = deepcopy(base_labels)
            if hasattr(monitor, "sensors") and monitor.sensors is not None:
                mon_labels["Region"] = list(monitor.sensors.labels)

            mon_name = type(monitor).__name__

            ts = TimeSeries(
                data=xv,
                time=tv,
                labels_dimensions=mon_labels,
                title=mon_name,
                sample_period=float(monitor.period),
            )

            if primary_ts is None:
                primary_ts = ts
                primary_tv = tv
                primary_xv = xv
            else:
                observations[mon_name] = ts

        # Build xr.DataArray from primary monitor
        # TVB shape: (time, state_variables, nodes, modes) — keep mode dim
        data_np = np.asarray(primary_xv)
        dims = ["time", "variable", "node", "mode"][: data_np.ndim]
        coords = {
            "time": np.asarray(primary_tv),
            "variable": voi,
            "node": region_labels,
        }
        if "mode" in dims:
            coords["mode"] = list(range(data_np.shape[3]))
        da = xr.DataArray(data=data_np, dims=dims, coords=coords)

        sim_result = SimulationResult(
            data=da,
            observations=observations,
        )
        sim_result._timeseries = primary_ts

        return cls(integration=sim_result)


# =============================================================================
# Time Series Classes
# =============================================================================


@register_pytree_node_class
class TimeSeries:
    """
    Time-series dataType with JAX pytree support, domain-specific analysis,
    and visualization methods.
    """

    def tree_flatten(self):
        """Flatten into JAX pytree (children, aux_data).

        `sample_period` is a child (not aux) because it may hold a JAX tracer such as `state.dt` inside `jit`.
        """
        # Keep network as a child (not metadata) to avoid non-hashable/array metadata.
        # sample_period must also be a child because it can be a JAX-traced value
        # (e.g. state.dt inside jit); putting tracers in aux_data causes
        # UnexpectedTracerError on repeated JIT calls.
        children = (self.time, self.data, self.network, self.sample_period)
        aux_data = (
            self.title,
            self.labels_dimensions,
            self.units,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a `TimeSeries` from JAX pytree children and aux_data."""
        time, data, network, sample_period = children
        title, labels_dimensions, units = aux_data
        return cls(
            time,
            data,
            network=network,
            title=title,
            sample_period=sample_period,
            labels_dimensions=labels_dimensions,
            units=units,
        )

    def __init__(
        self,
        time,
        data,
        network=None,
        title="TimeSeries",
        sample_period=None,
        labels_dimensions={},
        units=None,
    ):
        """
        labels_dimensions: Specific labels for each dimension for the data stored in this timeseries. A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }
        units: Dictionary mapping dimension names to their units, e.g., {'time': 'ms', 'state': 'mV', 'region': None, 'mode': None}
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

        # 4. Units for each dimension
        self.units = units or {
            "time": "ms",
            "state": None,  # Typically mV, V, or dimensionless
            "region": None,  # Spatial units
            "mode": None,
        }

        # 5. Internal Configurations
        self.labels_ordering = ("Time", "State Variable", "Space", "Mode")

    @property
    def ndim(self):
        """Number of dimensions of the underlying data array."""
        return self.data.ndim

    @property
    def shape(self):
        """Shape of the underlying data array."""
        return self.data.shape

    def __repr__(self):
        return format_pytree_as_string(self, self.__class__.__name__, "", False, False)

    @property
    def time_unit(self):
        """Unit of the sample period (e.g. `"ms"`)."""
        return self.sample_period_unit

    @property
    def space_labels(self):
        """Labels for the spatial (region) axis as a NumPy array.

        Reads the canonical `"Space"` entry of `labels_dimensions`, falling back
        to a legacy `"Region"` key, and returns an empty array when neither is
        present. Scalar or string values are coerced to a one-element array.
        """
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
        """Labels for the state-variable axis as a NumPy array.

        Returns an empty array when no state-variable labels are stored; scalar
        or string values are coerced to a one-element array.
        """
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
        """Return the sampling interval.

        Returns:
            The stored `dt` when available, otherwise the mean spacing between
            successive time points.
        """
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
            raise IndexError(f"{sv_label} is not a state variable. Available state variables: {self.variables_labels}")

        sv_index = np.where(self.variables_labels == sv_label)[0][0]
        return sv_index

    def get_state(self, sv_label):
        """Extract one or more state variables by label.

        Args:
            sv_label: A single state-variable label, or a list/tuple/array of
                labels to select several at once.

        Returns:
            A new `TimeSeries` restricted to the selected state variable(s), with
            its state-variable labels updated accordingly.
        """
        if isinstance(sv_label, (list, tuple, np.ndarray)):
            indices = [self._get_index_of_state_variable(s) for s in sv_label]
            sv_data = self.data[:, indices, :, :]
            sv_labels = list(sv_label)
        else:
            sv_data = self.data[:, self._get_index_of_state_variable(sv_label), :, :]
            sv_labels = [sv_label]

        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[1]] = sv_labels
        if sv_data.ndim == 3:
            sv_data = np.expand_dims(sv_data, 1)
        return self.duplicate(data=sv_data, labels_dimensions=subspace_labels_dimensions)

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            space_index = np.where(self.space_labels == label)[0][0]
            list_of_indices_for_labels.append(space_index)
        return list_of_indices_for_labels

    def _check_space_indices(self, list_of_index):
        n_space = self.data.shape[2]
        for idx in list_of_index:
            if idx < 0 or idx >= n_space:
                raise IndexError(f"Space index {idx} out of range [0, {n_space})")

    def get_subspace_by_index(self, list_of_index, **kwargs):
        """Extract a spatial subset by region index.

        Args:
            list_of_index: Indices along the spatial (region) axis to keep.
            **kwargs: Additional keyword arguments forwarded to `duplicate`.

        Returns:
            A new `TimeSeries` containing only the selected regions, with its
            spatial labels updated accordingly.

        Raises:
            IndexError: If any index is outside the valid region range.
        """
        self._check_space_indices(list_of_index)
        subspace_data = self.data[:, :, list_of_index, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = self.space_labels[list_of_index].tolist()
        if subspace_data.ndim == 3:
            subspace_data = np.expand_dims(subspace_data, 2)
        return self.duplicate(data=subspace_data, labels_dimensions=subspace_labels_dimensions, **kwargs)

    def get_subspace_by_labels(self, list_of_labels):
        """Extract a spatial subset by region label.

        Args:
            list_of_labels: Region labels to keep.

        Returns:
            A new `TimeSeries` containing only the regions matching the given
            labels.
        """
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

    def convert_units(self, dimension, target_unit):
        """
        Convert units for a specific dimension and return a new TimeSeries.

        Parameters:
        -----------
        dimension : str
            Dimension to convert ('time', 'state', 'region', 'mode')
        target_unit : str
            Target unit to convert to

        Returns:
        --------
        TimeSeries
            New TimeSeries with converted values
        """
        # Define conversion factors (to base units)
        time_conversions = {
            "s": 1.0,
            "ms": 1e-3,
            "us": 1e-6,
            "ns": 1e-9,
            "sec": 1.0,
            "msec": 1e-3,
            "usec": 1e-6,
        }
        voltage_conversions = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "kV": 1e3}

        current_unit = self.units.get(dimension)
        if current_unit is None:
            raise ValueError(f"No unit specified for dimension '{dimension}'")

        # Select appropriate conversion dict
        if dimension == "time":
            conversions = time_conversions
        elif dimension == "state":
            conversions = voltage_conversions
        else:
            raise NotImplementedError(f"Unit conversion not implemented for dimension '{dimension}'")

        if current_unit not in conversions or target_unit not in conversions:
            raise ValueError(f"Unsupported unit conversion: {current_unit} -> {target_unit}")

        # Convert to base unit then to target unit
        scale_factor = conversions[current_unit] / conversions[target_unit]

        # Create new TimeSeries with converted values
        new_units = self.units.copy()
        new_units[dimension] = target_unit

        if dimension == "time":
            return self.duplicate(
                time=self.time * scale_factor,
                sample_period=(self.sample_period * scale_factor if self.sample_period else None),
                units=new_units,
            )
        elif dimension == "state":
            return self.duplicate(data=self.data * scale_factor, units=new_units)

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
        new_time = kwargs.get("time", self.time)
        # Recalculate sample_period if time changed and not explicitly provided
        if "sample_period" in kwargs:
            new_sample_period = kwargs["sample_period"]
        elif "time" in kwargs and len(new_time) > 1:
            # Time was changed, recalculate sample_period
            new_sample_period = float(new_time[1] - new_time[0])
        else:
            new_sample_period = self.sample_period

        new = self.__class__(
            time=new_time,
            data=kwargs.get("data", self.data),
            network=self.network,
            title=self.title,
            sample_period=new_sample_period,
            labels_dimensions=kwargs.get("labels_dimensions", self.labels_dimensions.copy()),
            units=kwargs.get("units", self.units.copy() if self.units else None),
        )
        return new

    # ── Domain-specific methods (analysis, visualization) ─────────────

    def get_state_variable(self, sv_label):
        """Evaluate a state variable or a symbolic expression of state variables.

        When `sv_label` is a list/tuple/array it behaves like `get_state`. When
        it is a string it is parsed as a symbolic expression whose free symbols
        are matched against existing state variables, allowing derived
        quantities such as `"E - I"` to be computed.

        Args:
            sv_label: A state-variable label, a collection of labels, or a
                symbolic expression string combining state-variable names.

        Returns:
            A new `TimeSeries` holding the evaluated state variable or
            expression, labelled with `sv_label`.
        """
        if isinstance(sv_label, (list, tuple, np.ndarray)):
            return self.get_state(sv_label)
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
        return self.duplicate(data=sv_data, labels_dimensions=subspace_labels_dimensions)

    def plot(self, ax=None, axis_labels=False, legend=True, title=None, **kwargs):
        """Plot the time series, or a state-space trajectory of its variables.

        By default each state variable is drawn against time. Passing
        `type="statespace"` (or an equivalent alias such as `"phase"` or
        `"trajectory"`) instead plots one state variable against another for a
        chosen region and mode.

        Args:
            ax: Existing Matplotlib axes to draw on. When omitted, a new figure
                and axes are created and the figure is returned.
            axis_labels: Whether to label the x-axis with the time unit.
            legend: Whether to draw a legend (ignored for single-variable plots).
            title: Optional axes title.
            **kwargs: Additional options forwarded to Matplotlib's `plot`, plus
                recognised keys such as `type`, `region`, `mode`,
                `state_variables`, `labels`, and `label`.

        Returns:
            The created Matplotlib figure when `ax` was not supplied, otherwise
            `None`.

        Raises:
            ValueError: If a state-space plot is requested with fewer than two
                state variables, or the data shape is unsupported.
        """
        plot_type = kwargs.pop("type", "timeseries")
        if not ax:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        if title:
            ax.set_title(title)

        if plot_type in {
            "statespace",
            "state-space",
            "phase",
            "phase_space",
            "trajectory",
        }:
            region = kwargs.pop("region", 0)
            mode = kwargs.pop("mode", 0)
            sv_labels = kwargs.pop("state_variables", None) or kwargs.pop("state_variables_labels", None)

            n_svar = self.data.shape[1] if len(self.data.shape) > 1 else 1
            if sv_labels:
                if isinstance(sv_labels, str):
                    sv_labels = [sv_labels]
                indices = [self._get_index_of_state_variable(s) for s in sv_labels]
            else:
                indices = list(range(min(2, n_svar)))
                sv_labels = (
                    self.labels_dimensions.get("State Variable", None) if isinstance(self.labels_dimensions, dict) else None
                )
                if sv_labels:
                    sv_labels = [sv_labels[i] for i in indices]

            if len(indices) < 2:
                raise ValueError("State-space plot requires at least two state variables")

            data = self.data
            if data.ndim == 4:
                x = data[:, indices[0], region, mode]
                y = data[:, indices[1], region, mode]
            elif data.ndim == 3:
                x = data[:, indices[0], region]
                y = data[:, indices[1], region]
            elif data.ndim == 2:
                x = data[:, indices[0]]
                y = data[:, indices[1]]
            else:
                raise ValueError("Unsupported data shape for state-space plot")
            ax.plot(x, y, **kwargs)

            if sv_labels and len(sv_labels) >= 2:
                ax.set_xlabel(str(sv_labels[0]))
                ax.set_ylabel(str(sv_labels[1]))
            else:
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            if return_fig:
                plt.close()
                return fig
            return None

        n_svar = self.data.shape[1] if len(self.data.shape) > 1 else 1
        uses_modes = len(self.data.shape) > 3 and self.data.shape[3] > 1
        if uses_modes:
            logger.info("Plotting only first mode by default")

        # n_regions = self.data.shape[2]
        if "labels" in kwargs.keys():
            labels = kwargs.pop("labels")
        else:
            labels = [
                (self.labels_dimensions["State Variable"][i] if "State Variable" in self.labels_dimensions else None)
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

        ax.set_xlabel(f"time [{self.units['time']}]")

        if n_svar == 1 and self.labels_dimensions:
            ylabel = (
                self.labels_dimensions.get("State Variable", ["X"])[0] if isinstance(self.labels_dimensions, dict) else "X"
            )
            ax.set_ylabel(ylabel)
            legend = False
        else:
            ax.set_ylabel("X")

        if axis_labels:  # ?
            ax.set_xlabel(self.units["time"])
        if legend and any(labels):
            ax.legend(loc="upper right", fontsize="smaller")
            handles, labels = ax.get_legend_handles_labels()
            unique = list(dict(zip(labels, handles)).items())  # Keep only the last occurrence of each label
            ax.legend(
                [handle for _, handle in unique],
                [label for label, _ in unique],
                loc="upper right",
            )
        if return_fig:
            plt.close()
            return fig

    def animate(
        self,
        state=0,
        format="dots",
        interval=50,
        cmap="viridis",
        node_size=120,
        figsize=(10, 4),
    ):
        """Animate timeseries on a graph layout.

        Each node is a dot positioned by the graph layout; its color
        reflects the timeseries value of the selected state variable
        over time.

        Parameters
        ----------
        state : int or str
            State variable index or name to animate.
        format : str
            Animation format.  Currently only ``'dots'`` is supported.
        interval : int
            Milliseconds between frames.
        cmap : str
            Matplotlib colormap name.
        node_size : int
            Scatter point size.
        figsize : tuple
            Figure size ``(width, height)``.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object (render with ``HTML(ani.to_jshtml())``
            in Jupyter, or ``ani.save(...)``).
        """

        graph = getattr(self, "graph", None)
        if graph is None:
            raise ValueError("No graph data attached.  Run with format='networkdynamics' to get graph positions.")
        pos = graph["positions"]
        adj = graph["adjacency"]

        # Resolve state index
        if isinstance(state, str):
            sv_list = list(self.labels_dimensions.get("State Variable", []))
            state = sv_list.index(state)

        # Data: (time, nodes) for selected state
        vals = self.data[:, state, :, 0]  # (T, N)
        vmin, vmax = float(vals.min()), float(vals.max())
        x, y = pos[:, 0], pos[:, 1]

        fig, (ax_graph, ax_ts) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1.2]},
        )

        # Draw edges
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ax_graph.plot(
                        [x[i], x[j]],
                        [y[i], y[j]],
                        color="lightgray",
                        linewidth=0.5,
                        zorder=0,
                    )

        sc = ax_graph.scatter(
            x,
            y,
            c=vals[0],
            cmap=cmap,
            s=node_size,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            edgecolors="k",
            linewidths=0.5,
        )
        ax_graph.set_aspect("equal")
        ax_graph.set_title(f"t = {self.time[0]:.2f}")
        ax_graph.axis("off")
        fig.colorbar(sc, ax=ax_graph, shrink=0.7)

        # Time-series panel: all nodes
        n_nodes = vals.shape[1]
        cm = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=n_nodes - 1)
        lines = []
        for i in range(n_nodes):
            (ln,) = ax_ts.plot(
                [],
                [],
                color=cm(norm(i)),
                linewidth=0.5,
                alpha=0.6,
            )
            lines.append(ln)
        (avg_ln,) = ax_ts.plot([], [], color="k", linewidth=1.5, label="mean")
        ax_ts.set_xlim(self.time[0], self.time[-1])
        ax_ts.set_ylim(vmin - 0.05 * abs(vmax - vmin), vmax + 0.05 * abs(vmax - vmin))
        sv_labels = list(self.labels_dimensions.get("State Variable", []))
        sv_name = sv_labels[state] if state < len(sv_labels) else f"state {state}"
        ax_ts.set_xlabel("time")
        ax_ts.set_ylabel(sv_name)
        ax_ts.legend(loc="upper right", fontsize="small")
        fig.tight_layout()

        # Subsample for performance
        step = max(1, len(self.time) // 200)
        frames = list(range(0, len(self.time), step))

        def update(frame):
            """Render a single animation frame at the given time index."""
            sc.set_array(vals[frame])
            ax_graph.set_title(f"t = {self.time[frame]:.2f}")
            for i, ln in enumerate(lines):
                ln.set_data(self.time[: frame + 1], vals[: frame + 1, i])
            avg_ln.set_data(
                self.time[: frame + 1],
                vals[: frame + 1].mean(axis=1),
            )
            return [sc] + lines + [avg_ln]

        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False,
        )
        plt.close(fig)
        return ani

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
            [str(lbl) for lbl in labels_array]
            if labels_array is not None and len(labels_array) == n_regions
            else [str(i) for i in range(n_regions)]
        )
        max_label_len = max((len(lbl) for lbl in labels_list), default=1)

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
        """Drop the initial transient before a given time.

        Args:
            start_time: Time value; all samples strictly before it are removed.

        Returns:
            A new `TimeSeries` starting at the first sample at or after
            `start_time`.
        """
        start_index = jnp.searchsorted(self.time, start_time, side="left")

        # Avoid deepcopy to prevent JAX tracer leaks - manually construct new instance
        ts_cut = self.__class__(
            time=self.time[start_index:],
            data=self.data[start_index:],
            network=self.network,
            title=self.title,
            sample_period=self.sample_period,
            labels_dimensions=self.labels_dimensions,
            units=self.units,
        )
        return ts_cut

    def subset(self, start, end):
        """Restrict the time series to a `[start, end]` time window.

        Args:
            start: Start time of the window (inclusive).
            end: End time of the window (inclusive).

        Returns:
            A new `TimeSeries` covering only samples within the window.
        """
        start_index = np.searchsorted(self.time, start, side="left")
        end_index = np.searchsorted(self.time, end, side="right")

        ts_subset = deepcopy(self)
        ts_subset.time = self.time[start_index:end_index]
        ts_subset.data = self.data[start_index:end_index]
        return ts_subset

    def exclude_region(self, region):
        """Return a copy with one region removed.

        Args:
            region: The region to drop, given either as an integer index along
                the spatial axis or as a region label.

        Returns:
            A new `TimeSeries` without the specified region.
        """
        if isinstance(region, int):
            region_index = region
        else:
            region_index = self.get_region_index(region)
        data = np.delete(self.data, region_index, axis=2)
        labels_dimensions = deepcopy(self.labels_dimensions)
        if "Region" in labels_dimensions:
            labels_dimensions["Region"].remove(region)
        return self.duplicate(data=data, labels_dimensions=labels_dimensions)

    def calculate_frequency(self, state_variable=None, region=0, mode=0) -> float:
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
            raise AttributeError("Time information is missing in the TimeSeries object.")

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
            raise ValueError(f"select variable of interest (VOI) from {self.labels_dimensions['State Variable']}")

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
        """Recompute `sample_period` from the mean spacing of the time axis.

        Prints a warning and updates `sample_period` in place when it disagrees
        with the mean of `diff(time)`.
        """
        dt = np.diff(self.time)
        mean_dt = np.mean(dt)
        if self.sample_period != mean_dt:
            logger.warning("Sample period does not match mean dt; setting sample period to mean dt.")
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
        """Test whether this series' data matches another array or time series.

        Args:
            other: A NumPy array or another `TimeSeries` to compare against.
            select_state_variable: Optional state-variable label (or expression)
                to compare instead of the full data array.

        Returns:
            `True` if the flattened values are element-wise close (`atol=1e-8`),
            else `False`.
        """
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
        """Return the spatial-axis index of a region given its label.

        Args:
            region_label: The region label to look up.

        Returns:
            The integer index of the region within the `"Region"` labels.
        """
        return list(self.labels_dimensions["Region"]).index(region_label)

    def get_region(self, region_label):
        """Extract a single region by label.

        Args:
            region_label: The label of the region to keep.

        Returns:
            A new `TimeSeries` containing only the selected region.
        """
        region_index = self.get_region_index(region_label)
        roi_data = self.data[:, :, region_index : region_index + 1, :]

        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = [region_label]

        return self.duplicate(data=roi_data, labels_dimensions=subspace_labels_dimensions)

    def to_bids(
        self,
        output_dir: str,
        subject: str = "01",
        session: str | None = None,
        description: str | None = None,
        run: int | None = None,
        suffix: str = "State",
        experiment=None,
        include_model: bool = True,
        include_connectivity: bool = True,
        timeseries_format: str = "cifti",
    ) -> str:
        """
        Export TimeSeries data to BIDS-compliant format (BEP034).

        Creates a BIDS dataset structure following the Computational Model
        Specification (BEP034 v1.0.0) with:
        - net/: Network connectivity files (weights, distances)
        - ts/: Time series data files (CIFTI-2 ptseries or TSV)
        - eq/: Model equations (tvbo format)
        - coord/: Region coordinates (if available)
        - JSON sidecar files with metadata

        Uses pydantic models for metadata serialization and pybids patterns
        for BIDS-compliant filename generation.

        Parameters
        ----------
        output_dir : str
            Root directory for the BIDS dataset.
        subject : str
            Subject identifier (without 'sub-' prefix). Default: '01'.
        session : str, optional
            Session identifier (without 'ses-' prefix).
        description : str, optional
            Description label for the output files. If not provided, uses
            the model name from experiment (e.g., 'wilsoncowan').
        run : int, optional
            Run number.
        suffix : str
            BIDS suffix indicating the observation/output type:
            - 'State' (default): Raw neural output (no observation model)
            - 'BOLD': fMRI BOLD signal (output convolved with HRF)
            - 'EEG': EEG signal (output with EEG forward model)
            - 'MEG': MEG signal (output with MEG forward model)
            The ts entity (ts-V, ts-W, ts-Diff) identifies which output variable,
            which can be a state variable or derived output (e.g., Diff: V-W).
            The suffix indicates the observation transformation applied.
        experiment : SimulationExperiment, optional
            The source simulation experiment for full provenance tracking.
            If not provided, uses self.source_experiment if available.
        include_model : bool
            Whether to export model equations. Default: True.
        include_connectivity : bool
            Whether to export connectivity data. Default: True.
        timeseries_format : str
            Format for time series output. Options:
            - 'cifti' (default): CIFTI-2 ptseries.nii files with named parcels.
              Splits data by state variable into separate files.
            - 'tsv': Tab-separated values files. Splits by state variable.
            - 'h5' or 'hdf5': HDF5 files preserving full dimensionality.
              Does NOT split by state variable - keeps all dimensions intact.
              Ideal for parameter sweeps (e.g., sweep, time, state, region, mode).

        Returns
        -------
        str
            Path to the created BIDS dataset root directory.

        Examples
        --------
        >>> ts = experiment.run()
        >>> ts.to_bids("./derivatives/tvbo", subject="01")
        './derivatives/tvbo'

        >>> # Export BOLD observation model output
        >>> bold_ts.to_bids("./derivatives/tvbo", suffix="BOLD")

        >>> # Export as TSV instead of CIFTI
        >>> ts.to_bids("./derivatives/tvbo", timeseries_format="tsv")

        >>> # Export as HDF5 preserving all dimensions (no state variable split)
        >>> ts.to_bids("./derivatives/tvbo", timeseries_format="h5")

        Notes
        -----
        Follows BIDS BEP034 Computational Modeling extension v1.0.0.
        Uses pydantic for metadata serialization and pybids for filenames.
        """
        import os
        from datetime import datetime

        import pandas as pd

        # Import BEP034 module
        from tvbo.adapters.bids import (
            H5PY_AVAILABLE,
            BEP034PathBuilder,
            CoordinateSidecar,
            DatasetDescription,
            EquationSidecar,
            NetworkSidecar,
            SimulationProvenance,
            TimeSeriesHDF5Sidecar,
            TimeSeriesSidecar,
            compute_id,
            to_float,
            write_cifti_ptseries,
            write_hdf5_timeseries,
            write_sidecar,
            write_tsv,
        )

        # Use source_experiment if not explicitly provided
        if experiment is None:
            experiment = getattr(self, "source_experiment", None)

        # Auto-detect description from model name if not provided
        if description is None and experiment is not None:
            if hasattr(experiment, "dynamics"):
                description = type(experiment.dynamics).__name__.lower()
            else:
                description = "simulation"
        elif description is None:
            description = "simulation"

        # Initialize path builder
        path_builder = BEP034PathBuilder()

        # Create base directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Track all created files for summary
        created_files = {"net": [], "ts": [], "eq": [], "coord": []}

        region_labels = [str(label) for label in self.space_labels] if len(self.space_labels) else None

        # =====================================================================
        # 1. Export connectivity to net/ directory
        # =====================================================================
        # Use experiment's network if TimeSeries doesn't have one attached
        network = self.network
        if network is None and experiment is not None:
            network = getattr(experiment, "network", None)
        if include_connectivity and network is not None:
            # --- Weights matrix ---
            weights_sidecar = NetworkSidecar(
                Description="Structural connectivity weights matrix",
                NumberOfNodes=int(network.weights.shape[0]),
                Units="a.u.",
                Source="tvbo simulation",
                GeneratedAt=datetime.now().isoformat(),
                NodeLabels=region_labels,
            )
            weights_id = compute_id(weights_sidecar.to_dict())

            # Build path using pybids patterns
            weights_rel_path = path_builder.build_net_path(
                subject=subject,
                net_type="weights",
                id_hash=weights_id,
                desc=description,
                session=session,
                run=run,
                extension=".tsv",
            )
            weights_tsv_path = os.path.join(output_dir, weights_rel_path)
            weights_json_path = weights_tsv_path.replace(".tsv", ".json")

            weights_df = pd.DataFrame(
                np.asarray(network.weights),
                index=region_labels,
                columns=region_labels,
            )
            write_tsv(weights_df, weights_tsv_path)
            write_sidecar(weights_sidecar, weights_json_path)
            created_files["net"].append(weights_rel_path)

            # --- Distances (tract lengths) matrix ---
            lengths = network.lengths if hasattr(network, "lengths") else getattr(network, "tract_lengths", None)
            if lengths is None:
                lengths = np.zeros_like(network.weights)
            distances_sidecar = NetworkSidecar(
                Description="Tract lengths (distances) between regions",
                NumberOfNodes=int(lengths.shape[0]),
                Units="mm",
                Source="tvbo simulation",
                GeneratedAt=datetime.now().isoformat(),
                NodeLabels=region_labels,
            )
            distances_id = compute_id(distances_sidecar.to_dict())

            distances_rel_path = path_builder.build_net_path(
                subject=subject,
                net_type="distances",
                id_hash=distances_id,
                desc=description,
                session=session,
                run=run,
                extension=".tsv",
            )
            distances_tsv_path = os.path.join(output_dir, distances_rel_path)
            distances_json_path = distances_tsv_path.replace(".tsv", ".json")

            distances_df = pd.DataFrame(
                np.asarray(lengths),
                index=region_labels,
                columns=region_labels,
            )
            write_tsv(distances_df, distances_tsv_path)
            write_sidecar(distances_sidecar, distances_json_path)
            created_files["net"].append(distances_rel_path)

            # --- Coordinates if available ---
            if hasattr(network, "centres") and network.centres is not None:
                coord_sidecar = CoordinateSidecar(
                    Description="Region center coordinates",
                    NumberOfNodes=int(network.centres.shape[0]),
                    CoordinateSystem="MNI152NLin6Asym",
                    Units="mm",
                    Columns=["x", "y", "z"],
                    NodeLabels=region_labels,
                )
                coord_id = compute_id(coord_sidecar.to_dict())

                coord_rel_path = path_builder.build_coord_path(
                    subject=subject,
                    coord_type="centres",
                    id_hash=coord_id,
                    desc=description,
                    session=session,
                    extension=".tsv",
                )
                coord_tsv_path = os.path.join(output_dir, coord_rel_path)
                coord_json_path = coord_tsv_path.replace(".tsv", ".json")

                coord_df = pd.DataFrame(
                    np.asarray(network.centres),
                    columns=["x", "y", "z"],
                    index=region_labels,
                )
                write_tsv(coord_df, coord_tsv_path)
                write_sidecar(coord_sidecar, coord_json_path)
                created_files["coord"].append(coord_rel_path)

        # =====================================================================
        # 2. Export time series to ts/ directory as CIFTI-2 ptseries
        # =====================================================================
        sample_period_val = to_float(self.sample_period)
        if sample_period_val is not None and sample_period_val > 0:
            if self.sample_period_unit in ("ms", "msec"):
                sampling_freq = 1000.0 / sample_period_val
            else:
                sampling_freq = 1.0 / sample_period_val
        else:
            sampling_freq = None

        # Build provenance if experiment available
        provenance = None
        if experiment is not None:
            provenance = SimulationProvenance(
                Model=(str(experiment.dynamics) if hasattr(experiment, "dynamics") else None),
                Integrator=(str(experiment.integration) if hasattr(experiment, "integration") else None),
                Duration=(to_float(experiment.duration) if hasattr(experiment, "duration") else None),
                StepSize=(
                    to_float(experiment.integration.step_size)
                    if hasattr(experiment, "integration") and hasattr(experiment.integration, "step_size")
                    else None
                ),
                GeneratedAt=datetime.now().isoformat(),
            )

        # Determine output format
        use_cifti = timeseries_format.lower() == "cifti" and region_labels is not None
        use_h5 = timeseries_format.lower() in ("h5", "hdf5")

        if use_h5:
            # =====================================================================
            # HDF5 format: Preserve full dimensionality, don't split by state
            # =====================================================================
            if not H5PY_AVAILABLE:
                raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

            # Create HDF5 sidecar with full dimension info
            # Filter out None values from labels_dimensions (e.g., Time may be None)
            dim_labels = {k: v for k, v in self.labels_dimensions.items() if v is not None} if self.labels_dimensions else None
            h5_sidecar = TimeSeriesHDF5Sidecar(
                Description="Simulated time series - all state variables",
                Format="HDF5",
                Shape=list(self.data.shape),
                Dimensions=list(self.labels_ordering),
                DimensionLabels=dim_labels if dim_labels else None,
                SamplingFrequency=sampling_freq,
                SamplingPeriod=sample_period_val,
                SamplingPeriodUnits=self.sample_period_unit,
                StartTime=to_float(self.time[0]) if len(self.time) > 0 else 0.0,
                Units="a.u.",
                GeneratedAt=datetime.now().isoformat(),
                Provenance=provenance,
                StateVariables=(list(self.variables_labels) if len(self.variables_labels) > 0 else None),
                Datasets={
                    "/data": f"Time series data with shape {self.data.shape}",
                    "/time": "Time array",
                    "/labels/*": "Labels for each dimension",
                },
            )

            # Build path - use first state variable for ts entity, or 'all' for multi-state
            ts_entity = self.variables_labels[0] if len(self.variables_labels) == 1 else "all"
            ts_rel_path = path_builder.build_ts_path(
                subject=subject,
                ts_label=ts_entity,
                suffix=suffix,
                desc=description,
                session=session,
                run=run,
                extension=".h5",
            )
            ts_h5_path = os.path.join(output_dir, ts_rel_path)
            ts_json_path = ts_h5_path.replace(".h5", ".json")

            # Additional metadata for HDF5
            metadata = {
                "source": "tvbo simulation",
                "model": (str(experiment.dynamics) if experiment and hasattr(experiment, "dynamics") else None),
            }

            write_hdf5_timeseries(
                data=np.asarray(self.data),
                time=np.asarray(self.time),
                path=ts_h5_path,
                labels_dimensions=(dict(self.labels_dimensions) if self.labels_dimensions else None),
                labels_ordering=self.labels_ordering,
                sample_period=sample_period_val,
                sample_period_unit=self.sample_period_unit,
                metadata=metadata,
            )

            write_sidecar(h5_sidecar, ts_json_path)
            created_files["ts"].append(ts_rel_path)

        else:
            # =====================================================================
            # CIFTI/TSV format: Export each state variable as separate file
            # =====================================================================
            for sv_idx, sv_label in enumerate(self.variables_labels):
                # Create sidecar metadata
                ts_sidecar = TimeSeriesSidecar(
                    Description=f"Simulated parcellated time series - state variable {sv_label}",
                    StateVariable=sv_label,
                    SamplingFrequency=sampling_freq,
                    SamplingPeriod=sample_period_val,
                    SamplingPeriodUnits=self.sample_period_unit,
                    StartTime=to_float(self.time[0]) if len(self.time) > 0 else 0.0,
                    NumberOfTimepoints=int(self.data.shape[0]),
                    NumberOfNodes=int(self.data.shape[2]) if self.data.ndim > 2 else 1,
                    Columns=(region_labels if not use_cifti else None),  # Columns for TSV only
                    Units="a.u.",
                    GeneratedAt=datetime.now().isoformat(),
                    Provenance=provenance,
                )

                # Extract data for this state variable (time x regions)
                sv_data = self.data[:, sv_idx, :, 0]  # Take first mode

                if use_cifti:
                    # Write CIFTI-2 ptseries file with nibabel
                    # ts entity = state variable label (V, W, etc.)
                    # suffix = State (raw neural) or BOLD/EEG/etc. (observation)
                    ts_rel_path = path_builder.build_ts_path(
                        subject=subject,
                        ts_label=sv_label,
                        suffix=suffix,
                        desc=description,
                        session=session,
                        run=run,
                        extension=".ptseries.nii",
                    )
                    ts_cifti_path = os.path.join(output_dir, ts_rel_path)
                    ts_json_path = ts_cifti_path.replace(".ptseries.nii", ".json")

                    write_cifti_ptseries(
                        data=np.asarray(sv_data),
                        region_labels=region_labels,
                        path=ts_cifti_path,
                        sample_period=sample_period_val or 1.0,
                        sample_period_unit=self.sample_period_unit,
                    )
                else:
                    # Write TSV file
                    ts_rel_path = path_builder.build_ts_path(
                        subject=subject,
                        ts_label=sv_label,
                        suffix=suffix,
                        desc=description,
                        session=session,
                        run=run,
                        extension=".tsv",
                    )
                    ts_tsv_path = os.path.join(output_dir, ts_rel_path)
                    ts_json_path = ts_tsv_path.replace(".tsv", ".json")

                    ts_df = pd.DataFrame(np.asarray(sv_data), columns=region_labels)
                    ts_df.insert(0, "time", np.asarray(self.time))
                    write_tsv(ts_df, ts_tsv_path, include_index=False)

                write_sidecar(ts_sidecar, ts_json_path)
                created_files["ts"].append(ts_rel_path)

        # =====================================================================
        # 3. Export model equations to eq/ directory
        # =====================================================================
        if include_model and experiment is not None:
            model_name = "unknown"
            if hasattr(experiment, "dynamics"):
                model_name = type(experiment.dynamics).__name__

            # Extract model parameters
            params = None
            if hasattr(experiment, "dynamics"):
                model = experiment.dynamics
                params = {}
                for attr in dir(model):
                    if not attr.startswith("_"):
                        val = getattr(model, attr, None)
                        if isinstance(val, (int, float)):
                            params[attr] = val
                        elif hasattr(val, "tolist"):
                            try:
                                params[attr] = float(np.asarray(val).flat[0])
                            except Exception:
                                pass
                if not params:
                    params = None

            eq_sidecar = EquationSidecar(
                Description=f"Neural mass model equations - {model_name}",
                ModelType=model_name,
                Format="tvbo",
                GeneratedAt=datetime.now().isoformat(),
                Parameters=params,
            )
            eq_id = compute_id(eq_sidecar.to_dict())

            eq_rel_path = path_builder.build_eq_path(
                eq_label=model_name.lower(),
                id_hash=eq_id,
                desc=description,
                subject=subject,
                session=session,
                extension=".json",
            )
            eq_json_path = os.path.join(output_dir, eq_rel_path)

            write_sidecar(eq_sidecar, eq_json_path)
            created_files["eq"].append(eq_rel_path)

        # =====================================================================
        # 4. Create dataset_description.json at root
        # =====================================================================
        desc_path = os.path.join(output_dir, "dataset_description.json")
        if not os.path.exists(desc_path):
            dataset_desc = DatasetDescription(
                Name="TVB Simulation Output",
                BIDSVersion="1.9.0",
                DatasetType="derivative",
                GeneratedBy=[
                    {
                        "Name": "tvbo",
                        "Version": "0.1.0",
                        "Description": "The Virtual Brain Ontology and Simulation Framework",
                        "CodeURL": "https://github.com/the-virtual-brain/tvb-ontology",
                    }
                ],
                BEP034Version="1.0.0",
            )
            write_sidecar(dataset_desc, desc_path)

        # =====================================================================
        # 5. Create participants.tsv
        # =====================================================================
        sub_label = f"sub-{subject}"
        participants_path = os.path.join(output_dir, "participants.tsv")
        if not os.path.exists(participants_path):
            participants_df = pd.DataFrame({"participant_id": [sub_label]})
            participants_df.to_csv(participants_path, sep="\t", index=False)
        else:
            existing = pd.read_csv(participants_path, sep="\t")
            if sub_label not in existing["participant_id"].values:
                new_row = pd.DataFrame({"participant_id": [sub_label]})
                existing = pd.concat([existing, new_row], ignore_index=True)
                existing.to_csv(participants_path, sep="\t", index=False)

        return output_dir


# class TimeSeriesRegion(TimeSeries):
#     """A time-series associated with the regions of a network."""

#     # network = Attr(field_type=network.Connectivity)
#     # region_mapping_volume = Attr(
#     #     field_type=region_mapping.RegionVolumeMapping, required=False
#     # )
#     # region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
#     # labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

#     def summary_info(self):
#         """
#         Gather scientifically interesting summary information from an instance of this datatype.
#         """
#         summary = super(TimeSeriesRegion, self).summary_info()
#         summary.update(
#             {
#                 "Source Connectivity": self.network.title,
#                 "Region Mapping": (
#                     self.region_mapping.title if self.region_mapping else "None"
#                 ),
#                 "Region Mapping Volume": (
#                     self.region_mapping_volume.title
#                     if self.region_mapping_volume
#                     else "None"
#                 ),
#             }
#         )
#         return summary

#     def animate_time_series(
#         ts,
#         plane="sagittal",
#         state=0,
#         mode=0,
#         interval=100,
#         window_dt=1000,
#         cmap="viridis",
#         node_size=100,
#         line_kwargs={},
#     ):
#         """
#         Creates an animated 2D scatter plot from a 4D time-series object,
#         with a second axis for the time-series progression showing all regions.

#         Parameters:
#         - ts: Time series object with `ts.time`, `ts.data`, and `ts.network.centres`.
#             `ts.data` has shape (time, state, region, mode).
#         - plane: Projection plane ('sagittal', 'horizontal', 'axial').
#         - state: Index of the state to select or None to aggregate across states.
#         - mode: Index of the mode to select or None to aggregate across modes.
#         - interval: Time interval between frames in milliseconds.
#         - aggregation: Aggregation method ('mean', 'sum') if state or mode is None.
#         """
#         # Map plane to coordinates
#         if plane == "sagittal":
#             x, y = ts.network.centres[:, 1], ts.network.centres[:, 2]
#         elif plane == "horizontal":
#             x, y = ts.network.centres[:, 0], ts.network.centres[:, 1]
#         elif plane == "axial":
#             x, y = ts.network.centres[:, 0], ts.network.centres[:, 2]
#         else:
#             raise ValueError(
#                 "Invalid plane. Choose from 'sagittal', 'horizontal', 'axial'."
#             )

#         # Prepare data based on state and mode selection or aggregation
#         data = ts.data
#         data = data[:, state, :, mode][::window_dt]  # Fix state
#         data = (data - np.min(data)) / (
#             np.max(data) - np.min(data)
#         )  # Normalize to [0, 1]

#         time = ts.time[::window_dt]
#         n_regions = data.shape[1]

#         # Initialize figure and axes
#         fig, (ax, ax_ts) = plt.subplots(1, 2, layout="compressed", figsize=(8, 4))
#         sc = ax.scatter(x, y, c=data[0], cmap="viridis", s=node_size, vmin=0, vmax=1)

#         ax.set_title(f"Time: {time[0]:.2f}")
#         ax.set_aspect("equal")
#         fig.colorbar(sc, ax=ax, label="Data Intensity", shrink=0.5)

#         # Create evenly spaced colors from the viridis colormap
#         colors = colormaps[cmap](np.linspace(0, 1, n_regions))

#         # Initialize the time series plot for all regions
#         lines = []
#         for i, color in enumerate(colors):
#             (line,) = ax_ts.plot(
#                 [], [], color=color, label=f"Region {i+1}", **line_kwargs
#             )
#             lines.append(line)
#         (avg_line,) = ax_ts.plot([], [], color="red", linewidth=2, label="Average")

#         ax_ts.set_xlim(time[0], time[-1])
#         ax_ts.set_ylim(0, 1.1)
#         ax_ts.set_title("Time-Series Progression")
#         ax_ts.set_xlabel("Time")
#         ax_ts.set_ylabel("Intensity")

#         # Update function for animation
#         def update(frame):
#             sc.set_array(data[frame])
#             ax.set_title(f"Time: {time[frame]:.2f}")
#             for i, line in enumerate(lines):
#                 line.set_xdata(time[: frame + 1])  # Update X data for each region
#                 line.set_ydata(data[: frame + 1, i])  # Update Y data for each region
#             avg_line.set_xdata(time[: frame + 1])  # Update X data for average
#             avg_line.set_ydata(
#                 data[: frame + 1].mean(axis=1)
#             )  # Update Y data for average
#             return [sc] + lines + [avg_line]

#         # Create animation
#         ani = FuncAnimation(
#             fig, update, frames=len(time), interval=interval, blit=False
#         )

#         plt.close()
#         return ani


@register_pytree_node_class
class SimulationState:
    """Bundled state passed to the integration backends for one simulation.

    Groups everything a backend needs to advance a run: the initial conditions,
    the `Network`, the integration step, the noise configuration, model
    parameters, stimulus, monitor settings, and the number of time steps.
    Registered as a JAX pytree so it can flow through `jit`/`vmap`; `nt` is kept
    static while the remaining fields are dynamic children.

    Args:
        initial_conditions: Initial state as a `TimeSeries` (history buffer).
        network: The `Network` (connectivity and coupling) to simulate.
        dt: Integration time step.
        noise: Noise configuration, including per-state-variable sigma.
        parameters: Model parameter pytree.
        stimulus: Stimulus specification applied during integration.
        monitor_parameters: Settings controlling recorded outputs.
        nt: Number of integration steps to run.
    """

    def __init__(
        self,
        initial_conditions: TimeSeries,
        network: Network,
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
        """Flatten into JAX pytree (children, aux_data).

        `nt` is kept as static aux_data so it stays concrete in shape/length contexts under jit/vmap.
        """
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
        """Reconstruct a `SimulationState` from JAX pytree children and aux_data."""
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
        """Number of state variables inferred from the initial conditions."""
        try:
            # initial_conditions: (H, S, R, M) or (T, S, R, M)
            return int(self.initial_conditions.data.shape[1])
        except Exception:
            return 0

    @property
    def state_variable_names(self):
        """State-variable names, falling back to positional indices as strings.

        Returns the `"State Variable"` labels from the initial conditions when
        they are present and match the number of state variables, otherwise a
        list of stringified indices.
        """

        ld = getattr(self.initial_conditions, "labels_dimensions", {}) or {}
        names = ld.get("State Variable", None)
        if names is not None and len(names) == self.n_state_variables:
            return names
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

            self.noise.sigma_vec = jnp.zeros((self.n_state_variables,), dtype=jnp.asarray(self.dt).dtype)
        return self.noise

    def get_state_variable_index(self, name_or_index) -> int:
        """Resolve a state variable to its integer index.

        Args:
            name_or_index: A state-variable name or an integer index. Integers
                are returned unchanged; unknown names fall back to `0`.

        Returns:
            The integer index of the state variable.
        """
        if isinstance(name_or_index, int):
            return int(name_or_index)
        names = self.state_variable_names
        try:
            return int(names.index(str(name_or_index)))
        except Exception:
            return 0

    def set_sigma_for(self, name_or_index, value):
        """Set the noise sigma for one state variable (or all at once).

        Rebuilds `noise.sigma_vec` rather than mutating it in place, so it is
        safe to call before `jit`/`vmap`.

        Args:
            name_or_index: The state variable to target, by name or index.
            value: A scalar sigma for the selected variable, or a list/tuple
                giving sigma for every state variable at once.

        Returns:
            `self`, to allow method chaining.

        Raises:
            ValueError: If a list/tuple `value` does not match the number of
                state variables.
        """
        import jax.numpy as jnp

        idx = self.get_state_variable_index(name_or_index)
        noise = self._ensure_noise_holder()
        # Rebuild sigma_vec to avoid in-place mutation issues
        sv = jnp.zeros((self.n_state_variables,), dtype=jnp.asarray(self.dt).dtype)
        if isinstance(value, (list, tuple)):
            # Allow list to set all values directly
            arr = jnp.asarray(value, dtype=sv.dtype)
            if arr.shape[0] != sv.shape[0]:
                raise ValueError("Length of sigma list must match number of state variables")
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

            sv = _np.asarray(n.sigma_vec) if getattr(n, "sigma_vec", None) is not None else None
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
            """Return the integer dtype paired with the given float dtype."""
            return jnp.int32 if float_dtype == jnp.float32 else jnp.int64

        int_dtype = get_int_dtype(target_dtype)

        def convert_leaf(x):
            """Cast a single pytree leaf to the target float or integer dtype."""
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
