from copy import deepcopy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation

import xarray as xr
try:
    import xarray_jax  # side effect: registers xr.DataArray as JAX pytree
except ImportError:
    pass

from tvbo.classes import equation as equations
from tvbo.utils import Bunch
from tvbo.classes.network import Network
from tvbo.utils import format_pytree_as_string

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp


def _to_dataarray(raw_data, raw_time=None, state_names=None):
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
    dims = ['time', 'variable', 'node', 'mode'][:raw_data.ndim]
    coords = {}
    if raw_time is not None:
        coords['time'] = np.asarray(raw_time)
    if state_names:
        # Only assign variable coord if length matches the data dimension
        var_axis = dims.index('variable') if 'variable' in dims else None
        if var_axis is not None and len(state_names) == raw_data.shape[var_axis]:
            coords['variable'] = list(state_names)
    return xr.DataArray(data=np.asarray(raw_data), dims=dims, coords=coords)


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

    def __init__(self, data=None, observations=None, transient=None, *,
                 result=None, state_names=None, **kwargs):
        self._extras = {}
        self._timeseries = None

        # ── Backward compat: accept old-style result= arg ──
        if result is not None and data is None:
            raw_data = result.data if hasattr(result, "data") else result
            raw_time = result.ts if hasattr(result, "ts") else None
            data = _to_dataarray(raw_data, raw_time, state_names)
        elif data is not None and not isinstance(data, xr.DataArray):
            data = _to_dataarray(data, None, state_names)

        self.data = data
        self.observations = observations if observations is not None else {}
        self.transient = transient
        self._extras.update(kwargs)
        # Store state_names separately for cases with no data yet
        if state_names and not (data is not None and 'variable' in getattr(data, 'coords', {})):
            self._extras['state_names'] = state_names

    # ── xarray delegation ─────────────────────────

    def sel(self, **kw):
        """Label-based selection, delegated to self.data."""
        return self.data.sel(**kw)

    def isel(self, **kw):
        """Integer-based selection, delegated to self.data."""
        return self.data.isel(**kw)

    @property
    def time(self):
        """Time values as numpy array (backward compatible)."""
        if self.data is not None and 'time' in self.data.coords:
            return self.data.coords['time'].values
        return None

    @property
    def state_names(self):
        """State variable names from data coordinates."""
        if self.data is not None and 'variable' in self.data.coords:
            return list(self.data.coords['variable'].values)
        return self._extras.get('state_names', [])

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
        if raw.ndim == 3:
            raw = np.expand_dims(raw, -1)  # add Mode dimension

        labels_dimensions = {}
        names = self.state_names
        if names:
            labels_dimensions["State Variable"] = names

        time = np.asarray(self.time) if self.time is not None else np.arange(raw.shape[0])
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        self._timeseries = TimeSeries(
            time=time,
            data=raw,
            sample_period=dt,
            labels_dimensions=labels_dimensions,
        )
        return self._timeseries

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        # Check extras first
        if name in self._extras:
            return self._extras[name]
        # Delegate to TimeSeries for plot/get_state etc.
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
            self.pre_tuning = SimulationResult(
                result=pre_tuning, state_names=state_names
            )
        else:
            self.pre_tuning = pre_tuning

        if post_tuning is not None and not isinstance(post_tuning, SimulationResult):
            self.post_tuning = SimulationResult(
                result=post_tuning,
                observations=post_tuning_observations or Bunch(),
                state_names=state_names,
            )
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
                    conv[f"{key}_final"] = (
                        vals[-1] if hasattr(vals, "__getitem__") else vals
                    )
                    if len(vals) > 1:
                        conv[f"{key}_delta"] = vals[-1] - vals[0]
        return conv

    def __getattr__(self, name):
        if name.startswith('_'):
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
        n_iter = self.n_iterations or (
            len(next(iter(self.history.values()))) if self.history else 0
        )
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
                    self.state_trajectory = (
                        list(traj) if hasattr(traj, "__iter__") else traj
                    )
            else:
                self.state_trajectory = params_data
        else:
            self.state_trajectory = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(f"OptimizationResult has no attribute '{name}'")

    def __repr__(self):
        loss_str = (
            f", final_loss={self.final_loss:.4f}" if self.final_loss is not None else ""
        )
        return (
            f"OptimizationResult(name='{self.name}', n_steps={self.n_steps}{loss_str})"
        )


class ExplorationResult(Bunch):
    """Result of parameter exploration (grid search).

    A thin wrapper around tvboptim exploration outputs that provides:
    - Access to raw results (flat or grid-shaped)
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
    results : jnp.ndarray
        Observable values at each grid point (flat for scalars, multi-dim for time series)
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.grid = grid
        self.axes = axes or []
        self.observable = observable
        self.dt = dt
        self.output_names = output_names or []

        # Compute expected grid shape from axes
        self._grid_shape = tuple(
            ax.get("n", getattr(ax, "n", None))
            for ax in self.axes
            if (isinstance(ax, dict) and "n" in ax) or hasattr(ax, "n")
        )

        # Detect whether results are time series or scalar per grid point
        if results is not None:
            results_arr = jnp.asarray(results)
            n_grid_dims = len(self._grid_shape) if self._grid_shape else 1
            if results_arr.ndim > n_grid_dims:
                # Time series: shape (n_grid, n_time, ...) — preserve structure
                self.results = results_arr
                self.is_timeseries = True
            else:
                # Scalar: flatten for backward compatibility
                self.results = results_arr.flatten()
                self.is_timeseries = False
        else:
            self.results = None
            self.is_timeseries = False

        # Shape is the expected grid shape from axes
        self.shape = self._grid_shape
        self._find_optimal()

    def as_grid(self) -> jnp.ndarray:
        """Reshape flat results to grid shape.

        Returns
        -------
        jnp.ndarray
            Results reshaped to (n_axis1, n_axis2, ...) matching axes order.
            For time series, returns (n_axis1, ..., n_time, ...) as-is.
        """
        if self.results is None:
            return None
        if self.is_timeseries:
            return self.results  # Already structured
        if not self._grid_shape:
            return self.results
        expected_size = int(jnp.prod(jnp.array(self._grid_shape)))
        if self.results.size == expected_size:
            return self.results.reshape(self._grid_shape)
        # Can't reshape - return as-is
        return self.results

    def _find_optimal(self):
        """Find optimal point in the grid (scalar results only)."""
        self.optimal = Bunch()
        if self.is_timeseries or self.results is None or self.results.size == 0:
            return
        # Find argmin in flat results (assumes lower is better for loss functions)
        flat = self.results.flatten()
        flat_idx = int(jnp.argmin(flat))
        self.optimal.flat_index = flat_idx
        self.optimal.value = float(flat[flat_idx])

        # Compute grid index if we have valid grid shape
        if self._grid_shape and len(self._grid_shape) > 0:
            expected_size = int(jnp.prod(jnp.array(self._grid_shape)))
            if flat.size == expected_size:
                self.optimal.index = tuple(
                    int(i) for i in jnp.unravel_index(flat_idx, self._grid_shape)
                )
            else:
                self.optimal.index = (flat_idx,)
        else:
            self.optimal.index = (flat_idx,)

        # Extract parameter values at optimal point
        self.optimal.parameters = Bunch()
        for i, ax in enumerate(self.axes):
            ax_name = (
                ax.get("name", getattr(ax, "name", None))
                if isinstance(ax, dict)
                else getattr(ax, "name", None)
            )
            ax_values = (
                ax.get("values", getattr(ax, "values", None))
                if isinstance(ax, dict)
                else getattr(ax, "values", None)
            )
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

    def plot(self, figsize=None, sharex=True, **kwargs):
        """Plot exploration results.

        For time series results: subplots for each parameter value.
        For scalar results: line plot or heatmap over parameter space.
        """
        if not self.is_timeseries:
            return self._plot_scalar(figsize=figsize, **kwargs)
        return self._plot_timeseries(figsize=figsize, sharex=sharex, **kwargs)

    def _plot_timeseries(self, figsize=None, sharex=True, **kwargs):
        """Plot time series for each parameter value as subplots."""
        if self.results is None:
            return None

        ax_info = self.axes[0] if self.axes else None
        n = int(ax_info.n) if ax_info else self.results.shape[0]
        time = self._get_time_axis()
        output_label = self.observable or ", ".join(self.output_names) or "output"

        fig, axes = plt.subplots(
            n,
            1,
            figsize=figsize or (12, 2 * n),
            sharex=sharex,
        )
        if n == 1:
            axes = [axes]

        ax_values = np.asarray(ax_info["explored_values"]) if ax_info else None

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

    def _plot_scalar(self, figsize=None, **kwargs):
        """Plot scalar results as line plot (1D) or heatmap (2D)."""
        if self.results is None:
            return None
        grid = self.as_grid()
        if grid is None:
            return None

        if len(self._grid_shape) == 1:
            ax_info = self.axes[0]
            values = (
                np.asarray(ax_info["values"])
                if "values" in ax_info
                else np.arange(self._grid_shape[0])
            )
            fig, ax = plt.subplots(figsize=figsize or (8, 4))
            ax.plot(values, np.asarray(grid), "o-", **kwargs)
            ax.set_xlabel(getattr(ax_info, "name", "param"))
            ax.set_ylabel(self.observable or "value")
            ax.set_title(self.name or "Exploration")
            plt.close()
            return fig
        elif len(self._grid_shape) == 2:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))
            ax.imshow(np.asarray(grid).T, aspect="auto", origin="lower")
            ax.set_xlabel(getattr(self.axes[0], "name", "axis 0"))
            ax.set_ylabel(getattr(self.axes[1], "name", "axis 1"))
            ax.set_title(self.name or "Exploration")
            plt.close()
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
                ax_name = (
                    ax.get("name", getattr(ax, "name", None))
                    if isinstance(ax, dict)
                    else getattr(ax, "name", None)
                )
                ax_values = (
                    ax.get("values", getattr(ax, "values", None))
                    if isinstance(ax, dict)
                    else getattr(ax, "values", None)
                )
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
        opt_str = (
            f", optimal={self.optimal.value:.4f}"
            if hasattr(self.optimal, "value")
            else ""
        )
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

    def __init__(self, integration=None, explorations=None, algorithms=None,
                 optimizations=None, continuations=None, data_sources=None,
                 name=None, source=None, **kwargs):
        self._extras = {}

        # ── Backward compat: ExperimentResult(results_bunch, experiment_name=...) ──
        experiment_name = kwargs.pop('experiment_name', None)
        if (integration is not None
                and not isinstance(integration, SimulationResult)
                and hasattr(integration, 'keys')):
            results = integration
            integration = results.get('integration')
            algorithms = results.get('algorithms', algorithms)
            optimizations = results.get('optimizations', optimizations)
            explorations = results.get('explorations', explorations)
            continuations = results.get('continuations', continuations)
            # Preserve extra keys (state, model_fn, timings, etc.)
            for k, v in results.items():
                if k not in ('integration', 'algorithms', 'optimizations',
                             'explorations', 'continuations', 'data_sources'):
                    self._extras[k] = v

        # Also handle keyword: ExperimentResult(results=bunch, ...)
        results_kw = kwargs.pop('results', None)
        if results_kw is not None and integration is None:
            if hasattr(results_kw, 'keys'):
                integration = results_kw.get('integration')
                algorithms = algorithms or results_kw.get('algorithms')
                optimizations = optimizations or results_kw.get('optimizations')
                explorations = explorations or results_kw.get('explorations')
                continuations = continuations or results_kw.get('continuations')
                for k, v in results_kw.items():
                    if k not in ('integration', 'algorithms', 'optimizations',
                                 'explorations', 'continuations', 'data_sources'):
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

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        # Check extras first
        if name in self._extras:
            return self._extras[name]
        # Delegate to integration for backward compat (result.data, result.time, etc.)
        integration = self.__dict__.get('integration')
        if integration is not None and hasattr(integration, name):
            return getattr(integration, name)
        raise AttributeError(f"'ExperimentResult' has no attribute '{name}'")

    def __repr__(self):
        label = self.name or "Experiment"
        lines = [label]

        sections = []
        if self.integration is not None:
            sections.append(('integration', self.integration))
        if self.algorithms:
            sections.append(('algorithms', self.algorithms))
        if self.optimizations:
            sections.append(('optimizations', self.optimizations))
        if self.explorations:
            sections.append(('explorations', self.explorations))
        if self.continuations:
            sections.append(('continuations', self.continuations))

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
            shape = (
                tuple(val.data.shape)
                if val.data is not None
                else None
            )
            if shape:
                details.append(f"{prefix}data: {shape}")
            if val.observations:
                obs_keys = list(val.observations.keys())
                details.append(f"{prefix}observations: {obs_keys}")

        elif isinstance(val, AlgorithmResult):
            details.append(f"{prefix}n_iterations: {val.n_iterations}")
            if val.history:
                hist_keys = [
                    k for k in val.history.keys() if not str(k).startswith("_")
                ]
                details.append(f"{prefix}history: {hist_keys}")

        elif isinstance(val, OptimizationResult):
            details.append(f"{prefix}n_steps: {val.n_steps}")
            if val.final_loss is not None:
                details.append(f"{prefix}final_loss: {val.final_loss:.4f}")
            if val.history:
                hist_keys = [
                    k for k in val.history.keys() if not str(k).startswith("_")
                ]
                details.append(f"{prefix}history: {hist_keys}")
            if val.simulation and val.simulation.observations:
                obs_keys = list(val.simulation.observations.keys())
                details.append(f"{prefix}simulation.observations: {obs_keys}")

        elif isinstance(val, ExplorationResult):
            if val.axes:
                axis_names = [
                    (
                        ax.get("name", ax.name)
                        if hasattr(ax, "get")
                        else getattr(ax, "name", "?")
                    )
                    for ax in val.axes
                ]
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
            (ts_dir / f"{ts_prefix}_State.json").write_text(
                json.dumps(sidecar, indent=2, default=str)
            )

            # Observations
            for obs_name, obs in self.integration.observations.items():
                obs_path = ts_dir / f"{prefix}_ts-{obs_name}"
                if isinstance(obs, xr.DataArray):
                    self._write_data(obs, obs_path)
                elif hasattr(obs, 'data'):
                    obs_da = _to_dataarray(
                        np.asarray(obs.data),
                        np.asarray(obs.time) if hasattr(obs, 'time') else None,
                    )
                    if obs_da is not None:
                        self._write_data(obs_da, obs_path)

            # Transient
            if (self.integration.transient is not None
                    and self.integration.transient.data is not None):
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
        raise ImportError(
            "netCDF export requires scipy or h5netcdf. "
            "Install with: pip install h5netcdf"
        )

    @staticmethod
    def _build_sidecar(sim_result):
        """Build a JSON sidecar dict for a SimulationResult."""
        sidecar = {}
        if sim_result.data is not None:
            sidecar["Shape"] = list(sim_result.data.shape)
            sidecar["Dimensions"] = list(sim_result.data.dims)
            if 'variable' in sim_result.data.coords:
                sidecar["StateVariables"] = list(
                    sim_result.data.coords['variable'].values
                )
            if 'node' in sim_result.data.coords:
                sidecar["Regions"] = list(
                    sim_result.data.coords['node'].values
                )
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
        # Squeeze singleton mode dim if present
        if data_np.ndim == 4 and data_np.shape[3] == 1:
            data_np = data_np[:, :, :, 0]

        ld = ts.labels_dimensions if isinstance(ts.labels_dimensions, dict) else {}
        state_names = ld.get("State Variable", [])
        region_labels = ld.get("Region", [])

        dims = ['time', 'variable', 'node', 'mode'][:data_np.ndim]
        coords = {}
        if ts.time is not None:
            coords['time'] = np.asarray(ts.time)
        if state_names:
            coords['variable'] = list(state_names)
        if region_labels and data_np.ndim >= 3:
            coords['node'] = [str(r) for r in region_labels]

        da = xr.DataArray(data=data_np, dims=dims, coords=coords)

        # Collect observations from derivatives (TVB-style) or extras
        observations = {}
        if hasattr(ts, 'derivatives') and ts.derivatives:
            for d_ts in ts.derivatives:
                obs_name = getattr(d_ts, 'title', None) or f"obs_{len(observations)}"
                observations[obs_name] = d_ts

        sim_result = SimulationResult(data=da, observations=observations)
        sim_result._timeseries = ts

        # Continuations (bifurcation results) go in a separate section
        continuations = {}
        if hasattr(ts, 'sol') and extras.get('_is_bifurcation', False):
            extras.pop('_is_bifurcation')
            continuations['default'] = ts.sol

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
            if hasattr(monitor, 'sensors') and monitor.sensors is not None:
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
        # TVB shape: (time, state_variables, nodes, modes) — squeeze mode if 1
        data_np = np.asarray(primary_xv)
        if data_np.ndim == 4 and data_np.shape[3] == 1:
            data_np = data_np[:, :, :, 0]  # squeeze singleton mode

        dims = ['time', 'variable', 'node', 'mode'][:data_np.ndim]
        coords = {
            'time': np.asarray(primary_tv),
            'variable': voi,
            'node': region_labels,
        }
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
class BaseTimeSeries:
    """
    Base time-series dataType.
    """

    def tree_flatten(self):
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
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

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

    def get_state(self, sv_label):
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
        return self.duplicate(
            data=sv_data, labels_dimensions=subspace_labels_dimensions
        )

    def get_state_variable(self, sv_label):
        return self.get_state(sv_label)

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
            data_to_convert = self.time
        elif dimension == "state":
            conversions = voltage_conversions
            data_to_convert = self.data
        else:
            raise NotImplementedError(
                f"Unit conversion not implemented for dimension '{dimension}'"
            )

        if current_unit not in conversions or target_unit not in conversions:
            raise ValueError(
                f"Unsupported unit conversion: {current_unit} -> {target_unit}"
            )

        # Convert to base unit then to target unit
        scale_factor = conversions[current_unit] / conversions[target_unit]

        # Create new TimeSeries with converted values
        new_units = self.units.copy()
        new_units[dimension] = target_unit

        if dimension == "time":
            return self.duplicate(
                time=self.time * scale_factor,
                sample_period=(
                    self.sample_period * scale_factor if self.sample_period else None
                ),
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
            labels_dimensions=kwargs.get(
                "labels_dimensions", self.labels_dimensions.copy()
            ),
            units=kwargs.get("units", self.units.copy() if self.units else None),
        )
        return new


@register_pytree_node_class
class TimeSeries(BaseTimeSeries):
    def get_state_variable(self, sv_label):
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
        return self.duplicate(
            data=sv_data, labels_dimensions=subspace_labels_dimensions
        )

    def plot(self, ax=None, axis_labels=False, legend=True, title=None, **kwargs):
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
            sv_labels = kwargs.pop("state_variables", None) or kwargs.pop(
                "state_variables_labels", None
            )

            n_svar = self.data.shape[1] if len(self.data.shape) > 1 else 1
            if sv_labels:
                if isinstance(sv_labels, str):
                    sv_labels = [sv_labels]
                indices = [self._get_index_of_state_variable(s) for s in sv_labels]
            else:
                indices = list(range(min(2, n_svar)))
                sv_labels = (
                    self.labels_dimensions.get("State Variable", None)
                    if isinstance(self.labels_dimensions, dict)
                    else None
                )
                if sv_labels:
                    sv_labels = [sv_labels[i] for i in indices]

            if len(indices) < 2:
                raise ValueError(
                    "State-space plot requires at least two state variables"
                )

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

        ax.set_xlabel(f"time [{self.units['time']}]")

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
            ax.set_xlabel(self.units["time"])
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
        from matplotlib.animation import FuncAnimation

        graph = getattr(self, "graph", None)
        if graph is None:
            raise ValueError(
                "No graph data attached.  Run with format='networkdynamics' "
                "to get graph positions."
            )
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
            create_multi_state_cifti,
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

        region_labels = list(self.space_labels) if len(self.space_labels) else None

        # =====================================================================
        # 1. Export connectivity to net/ directory
        # =====================================================================
        # Use experiment's network if TimeSeries doesn't have one attached
        network = self.network
        if network is None and experiment is not None:
            network = getattr(experiment, 'network', None)
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
            lengths = network.lengths if hasattr(network, 'lengths') else getattr(network, 'tract_lengths', None)
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
                Model=(
                    str(experiment.dynamics)
                    if hasattr(experiment, "dynamics")
                    else None
                ),
                Integrator=(
                    str(experiment.integration)
                    if hasattr(experiment, "integration")
                    else None
                ),
                Duration=(
                    to_float(experiment.duration)
                    if hasattr(experiment, "duration")
                    else None
                ),
                StepSize=(
                    to_float(experiment.integration.step_size)
                    if hasattr(experiment, "integration")
                    and hasattr(experiment.integration, "step_size")
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
                raise ImportError(
                    "h5py is required for HDF5 export. Install with: pip install h5py"
                )

            # Create HDF5 sidecar with full dimension info
            # Filter out None values from labels_dimensions (e.g., Time may be None)
            dim_labels = (
                {k: v for k, v in self.labels_dimensions.items() if v is not None}
                if self.labels_dimensions
                else None
            )
            h5_sidecar = TimeSeriesHDF5Sidecar(
                Description=f"Simulated time series - all state variables",
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
                StateVariables=(
                    list(self.variables_labels)
                    if len(self.variables_labels) > 0
                    else None
                ),
                Datasets={
                    "/data": f"Time series data with shape {self.data.shape}",
                    "/time": "Time array",
                    "/labels/*": "Labels for each dimension",
                },
            )

            # Build path - use first state variable for ts entity, or 'all' for multi-state
            ts_entity = (
                self.variables_labels[0] if len(self.variables_labels) == 1 else "all"
            )
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
                "model": (
                    str(experiment.dynamics)
                    if experiment and hasattr(experiment, "dynamics")
                    else None
                ),
            }

            write_hdf5_timeseries(
                data=np.asarray(self.data),
                time=np.asarray(self.time),
                path=ts_h5_path,
                labels_dimensions=(
                    dict(self.labels_dimensions) if self.labels_dimensions else None
                ),
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
                    Columns=(
                        region_labels if not use_cifti else None
                    ),  # Columns for TSV only
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


class LegacySimulationResult(Bunch):
    """Legacy simulation result class. Use SimulationResult instead."""

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
