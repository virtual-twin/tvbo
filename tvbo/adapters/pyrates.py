# -*- coding: utf-8 -*-
"""PyRates backend adapter for SimulationExperiment.

This module handles all PyRates-specific logic for running simulations, converting between TVBO and PyRates data formats, and computing outputs.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import uuid
from typing import TYPE_CHECKING

import numpy as np

from tvbo.adapters.base import BaseAdapter
from tvbo.codegen.code import inline_functions
from tvbo.parse.expression import function_bodies, parse_eq, states_an_expression
from tvbo.utils import is_array_valued, as_list

# Single source of truth (forward map + derived reverse) lives in tvbo/codegen/pyrates.py; re-imported here and used by the model template so the rename mapping is defined exactly once. See PYRATES_REPL there.
from tvbo.codegen.pyrates import PYRATES_REPL  # noqa: F401

if TYPE_CHECKING:
    import pandas as pd
    from tvbo.data.types import ExperimentResult, SimulationResult
    from tvbo.classes.experiment import SimulationExperiment


# PyRates supported solvers and TVBO-to-PyRates mapping
PYRATES_SOLVERS = {"euler", "heun", "scipy"}
TVBO_TO_PYRATES_SOLVER = {
    "EulerDeterministic": "euler",
    "Euler": "euler",
    "HeunDeterministic": "heun",
    "Heun": "heun",
    "RungeKutta4thOrder": "scipy",
    "RungeKutta4": "scipy",
    "RK4": "scipy",
}


def _patch_pyrates_networkx_backend():
    """Fix networkx 3.4+ backend dispatch conflict with PyRates.

    PyRates' ``ComputeGraph`` extends ``networkx.MultiDiGraph``.  In networkx ≥ 3.4 the ``MultiDiGraph.__new__`` is decorated with
    ``@nx._dispatchable`` which intercepts a ``backend`` keyword argument and tries to dispatch to a networkx graph backend.  PyRates passes
    ``backend='default'`` (meaning *PyRates* compute backend, not a networkx backend) through ``**kwargs`` to ``ComputeGraph(**kwargs)``.
    The decorator sees it and raises ``ImportError: 'default' backend is not installed``.

    Fix: replace ``ComputeGraph.__new__`` (and ``ComputeGraphBackProp``) with plain ``object.__new__`` so the decorator is removed.
    """
    try:
        from pyrates.backend.computegraph import ComputeGraph, ComputeGraphBackProp
    except ImportError:
        return

    def _plain_new(cls, *_args, **_kwargs):
        return object.__new__(cls)

    if getattr(ComputeGraph.__new__, "_dispatchable", False) or "argmap" in getattr(ComputeGraph.__new__, "__qualname__", ""):
        ComputeGraph.__new__ = _plain_new
    # Also check inherited __new__ from networkx
    try:
        ComputeGraph(backend="default")
    except (ImportError, TypeError):
        ComputeGraph.__new__ = _plain_new

    try:
        ComputeGraphBackProp(backend="default")
    except (ImportError, TypeError):
        ComputeGraphBackProp.__new__ = _plain_new


def _patch_pyrates_replace_in_expr():
    """Monkey-patch PyRates' replace_in_expr to use xreplace.

    PyRates uses ``expr.subs(replacements, simultaneous=True)`` which corrupts compound sub-expressions: when a ``Mul`` has two or more
    ``Add`` children sharing a symbol (e.g. ``a*v*(1-v)*(v-b)``), ``subs`` replaces the symbol *inside* the ``Add`` first, breaking the match for
    the ``Add`` replacement key. ``xreplace`` matches top-down and avoids this bug.
    """
    import pyrates.backend.parser as _pr_parser

    def _replace_in_expr_fixed(expr, replacements):
        return expr.xreplace(replacements)

    _pr_parser.replace_in_expr = _replace_in_expr_fixed


def _patch_pyrates_reserved_names():
    """Relax PyRates' variable-name reservation for SymPy collisions.

    PyRates >=1.2 rejects parameter / state-variable names that collide with a SymPy constant or function — ``Gamma``, ``gamma``, ``beta``, ``exp``,
    ``pi`` … — because an *undeclared* name of that form would sympify to the function/constant rather than a free symbol (e.g. ``sympify('Gamma*x')``
    yields the gamma function). See ``check_vname`` in
    ``pyrates.frontend.template.operator``.

    tvbo declares *every* model parameter and state variable as an explicit
    PyRates operator variable, so PyRates' own parser resolves the name to a symbol and the collision never occurs — exactly how tvbo's parser lets a
    declared parameter override the SymPy built-in (see
    ``tvbo.parse.expression.parse_eq``). We therefore keep only the genuinely
    PyRates-internal slot names reserved (the state vector ``y``/``dy``, the index slots, and the buffer/history name parts) and allow the rest. This
    is maximally flexible: a model may use ``Gamma`` as a parameter, or use the SymPy ``Gamma`` function in an equation where it is *not* declared.
    """
    import pyrates.frontend.template.operator as _pr_op
    from pyrates.ir.circuit import PyRatesException

    # Names that collide with PyRates-generated code regardless of declaration.
    _internal_names = ("y", "dy", "source_idx", "target_idx")
    _internal_parts = ("_buffer", "_delays", "_maxdelay", "_idx", "_hist")

    def _check_vname_declared_ok(v: str, vtype: str):
        # 't' is always the integration time state variable (upstream rule).
        if v == "t" and vtype != "state_var":
            vtype = "state_var"
        if v in _internal_names:
            raise PyRatesException(
                f"The variable name {v} is reserved for internal variables "
                f"created by PyRates. Please choose a different variable name."
            )
        for dpart in _internal_parts:
            if dpart in v:
                raise PyRatesException(
                    f"The variable name {v} contains the sub-string {dpart} "
                    f"which is reserved for internal variables created by "
                    f"PyRates. Please choose a different variable name."
                )
        return vtype

    _pr_op.check_vname = _check_vname_declared_ok


def _patch_pyrates_missing_funcs():
    """Register additional math functions in PyRates' base backend.

    PyRates' compute graph only supports functions listed in ``base_funcs``.
    Functions like ``erfc``, ``erf``, and ``fmod`` are valid in SymPy/numpy but missing from PyRates' registry.  We inject them into the shared
    ``base_funcs`` dict which is copied by every new backend instance.

    Also patches the ExpressionParser to inject functions into already- instantiated backends via their compute graph.
    """
    from pyrates.backend.base.base_funcs import base_funcs

    _extra_funcs = {}
    if "erfc" not in base_funcs:
        _extra_funcs["erfc"] = {
            "call": "erfc",
            "func": __import__("scipy.special", fromlist=["erfc"]).erfc,
            "imports": ["scipy.special.erfc"],
        }
    if "erf" not in base_funcs:
        _extra_funcs["erf"] = {
            "call": "erf",
            "func": __import__("scipy.special", fromlist=["erf"]).erf,
            "imports": ["scipy.special.erf"],
        }
    if "fmod" not in base_funcs:
        _extra_funcs["fmod"] = {
            "call": "fmod",
            "func": np.fmod,
            "imports": ["numpy.fmod"],
        }
    if _extra_funcs:
        base_funcs.update(_extra_funcs)

        # Also monkey-patch ExpressionParser.parse_expr to inject funcs into compute graph backends that were already instantiated.
        import pyrates.backend.parser as _pr_parser

        _orig_parse_expr = _pr_parser.ExpressionParser.parse_expr

        def _patched_parse_expr(self):
            for name, info in _extra_funcs.items():
                if name not in self.cg.backend._funcs:
                    self.cg.backend._funcs[name] = info
            return _orig_parse_expr(self)

        _pr_parser.ExpressionParser.parse_expr = _patched_parse_expr


def _legacy_input_variable(dynamics):
    """``I_ext`` where *dynamics* actually declares it, else nothing.

    The historic default for a stimulus with no event to name its target. Checked against
    the model rather than assumed, so a model that calls its drive anything else fails
    loudly instead of being handed an input key for a variable it does not have.
    """
    if dynamics is None:
        return None
    declared = set()
    for collection in ("state_variables", "parameters", "coupling_inputs", "derived_variables"):
        declared |= set(getattr(dynamics, collection, None) or {})
    return "I_ext" if "I_ext" in declared else None


class PyRatesAdapter(BaseAdapter):
    """Adapter for running SimulationExperiment via PyRates backend."""

    def __init__(self, experiment: "SimulationExperiment"):
        """Initialize adapter with experiment reference.

        Parameters
        ----------
        experiment : SimulationExperiment
            The experiment to run.
        """
        super().__init__(experiment)

    def run(
        self,
        solver: str | None = None,
        inputs: dict | None = None,
        outputs: list[str] | None = None,
        matrix_edge_threshold: int = 100,
        **kwargs,
    ) -> "ExperimentResult":
        """Run simulation and explorations using PyRates backend.

        Handles both integration and grid-search explorations in a single call, sharing YAML export / circuit load across both.

        Parameters
        ----------
        solver : str, optional
            ODE solver: "euler", "heun", "scipy". Defaults to mapped integration.method.
        inputs : dict, optional
            External inputs as {node/op/var: array} or will be auto-generated.
        outputs : list[str], optional
            Variables to monitor. If None, monitors all state variables.
        matrix_edge_threshold : int, optional
            For networks with N > threshold nodes, use add_edges_from_matrix instead
            of YAML edges for efficiency. Default is 100.
        **kwargs
            Additional kwargs passed to circuit.run() / grid_search().

        Returns
        -------
        ExperimentResult
        """
        from pyrates import clear

        from tvbo.data.types import ExperimentResult

        # Fix networkx 3.4+ backend dispatch conflict
        _patch_pyrates_networkx_backend()

        # Fix PyRates' broken replace_in_expr (subs vs xreplace bug)
        _patch_pyrates_replace_in_expr()

        # Register missing math functions (erfc, erf, fmod)
        _patch_pyrates_missing_funcs()

        # Allow declared params named like SymPy fns (Gamma, beta, exp, …)
        _patch_pyrates_reserved_names()

        exp = self.experiment
        integration = getattr(exp, "integration", None)

        # Resolve solver
        solver = self._resolve_solver(solver, integration)

        # Check network size for matrix-based edges
        network = getattr(exp, "network", None)
        n_nodes = self._get_node_count(network)
        use_matrix_edges = n_nodes > matrix_edge_threshold and hasattr(network, "weights_matrix")

        # ── Shared setup: single YAML export + circuit load ──────────
        circuit, tmpdir, pkg_name = self._load_circuit_from_yaml(include_edges=not use_matrix_edges)

        try:
            # For large networks, add edges via matrix
            if use_matrix_edges:
                self._add_edges_from_matrix(circuit, network)

            # Build outputs/inputs if not provided
            if outputs is None:
                outputs = self._build_outputs()
            if inputs is None:
                inputs = self._build_inputs()

            # PyRates vectorize=True breaks for single-node and heterogeneous
            if "vectorize" not in kwargs:
                if n_nodes <= 1 or self._is_heterogeneous():
                    kwargs["vectorize"] = False

            run_id = uuid.uuid4().hex[:8]

            # ── Integration ──────────────────────────────────────────
            sim_result = None
            if integration:
                int_kwargs = dict(kwargs)
                int_kwargs.setdefault("file_name", f"pyrates_run_{pkg_name}_{run_id}")

                result_df = circuit.run(
                    step_size=integration.step_size,
                    simulation_time=integration.duration,
                    inputs=inputs,
                    outputs=outputs,
                    solver=solver,
                    **int_kwargs,
                )

                result_df = self._compute_outputs(result_df)
                sim_result = self._df_to_simulation_result(result_df)

            # ── Explorations ─────────────────────────────────────────
            explorations = getattr(exp, "explorations", None) or {}
            expl_results = {}
            if explorations:
                expl_results = self._run_explorations(circuit, outputs, solver, pkg_name, **kwargs)

        finally:
            try:
                clear(circuit)
            except Exception:
                pass
            self._cleanup(tmpdir, pkg_name)

        return ExperimentResult(
            integration=sim_result,
            explorations=expl_results or None,
            source=exp,
            name=getattr(exp, "label", None),
        )

    def _run_explorations(
        self,
        circuit,
        outputs: dict,
        solver: str,
        pkg_name: str,
        **kwargs,
    ) -> dict:
        """Run all explorations using PyRates native ``grid_search``.

        Called internally by ``run()`` with an already-loaded circuit, avoiding duplicate YAML export / circuit load.

        Parameters
        ----------
        circuit : CircuitTemplate
            Pre-loaded circuit (shared with integration run).
        outputs : dict
            Output variables dict (shared with integration).
        solver : str
            Resolved ODE solver name.
        pkg_name : str
            Package name for unique file naming.
        **kwargs
            Additional keyword arguments forwarded to ``grid_search``.

        Returns
        -------
        dict
            ``{exploration_name: ExplorationResult}``
        """
        from pyrates.utility import grid_search

        exp = self.experiment
        explorations = getattr(exp, "explorations", None) or {}
        integration = getattr(exp, "integration", None)

        results = {}
        for expl_name, expl in explorations.items():
            param_grid, param_map, axes = self._build_param_grid_and_axes(expl)
            if not param_grid:
                continue

            # 'product' → permute_grid=True (Cartesian), 'zip' → pairwise
            permute = getattr(expl, "mode", "product") != "zip"

            gs_kwargs = dict(kwargs)
            gs_kwargs.setdefault(
                "file_name",
                f"pyrates_gs_{pkg_name}_{expl_name}_{uuid.uuid4().hex[:8]}",
            )

            results_df, results_map = grid_search(
                circuit_template=circuit,
                param_grid=param_grid,
                param_map=param_map,
                step_size=integration.step_size,
                simulation_time=integration.duration,
                outputs=outputs,
                permute_grid=permute,
                solver=solver,
                **gs_kwargs,
            )

            results[expl_name] = self._df_to_exploration_result(results_df, results_map, axes, expl, expl_name)

        return results

    def _build_param_grid_and_axes(self, expl) -> tuple:
        """Translate a TVBO ``Exploration`` into PyRates ``param_grid``, ``param_map``, and axes.

        Uses ``_pyrates_node_map()`` to resolve operator/node naming, so the generated paths match the YAML template exactly.

        Returns
        -------
        tuple
            ``(param_grid, param_map, axes)``
        """
        from tvbo.utils import Bunch

        dynamics_dict = self.build_dynamics_dict()
        nmap = self._pyrates_node_map()

        param_grid: dict = {}
        param_map: dict = {}
        axes: list = []

        for axis in as_list(expl.space):
            ref = str(axis.parameter)
            # Dotted ref "Dynamics.param": the prefix names the dynamics class and the suffix the parameter. Keep them separate — the parameter name drives variable resolution while the full ref stays the unique grid key.
            if "." in ref:
                dyn_class, param_name = ref.split(".", 1)
            else:
                dyn_class, param_name = None, ref
            domain = getattr(axis, "domain", None)

            # Get sweep values
            explored = getattr(axis, "explored_values", None)
            if explored:
                values = np.array([float(v) for v in explored])
            elif domain:
                lo, hi, n = float(domain.lo), float(domain.hi), int(domain.n)
                values = (
                    np.logspace(np.log10(lo), np.log10(hi), n)
                    if getattr(domain, "log_scale", False)
                    else np.linspace(lo, hi, n)
                )
            else:
                continue

            # Grid key = the full dotted ref. PyRates treats it as an opaque label (grid_search only ever looks up param_map[key]), so it stays unique even when two axes sweep the same parameter name on different dynamics. A repeated key means the same axis was listed twice.
            grid_key = ref
            if grid_key in param_grid:
                raise ValueError(
                    f"Exploration '{getattr(expl, 'name', '')}' lists the axis "
                    f"'{ref}' more than once; each swept parameter must be unique."
                )
            param_grid[grid_key] = list(values)

            # Harmonized axis name = the dotted reference (== Exploration.space key, matching every other backend); `key` carries the PyRates grid key (== results_map column) so consumers never re-derive it from the name.
            ax = Bunch(name=ref, n=len(values), explored_values=values, key=grid_key)
            axes.append(ax)

            # Resolve the target dynamics: prefer the class named by the prefix (so
            # `A.w` and `B.w` reach their own nodes), else the first dynamics that declares the parameter, else the first dynamics.
            py_name = PYRATES_REPL.get(param_name, param_name)
            resolved = None
            if dyn_class and dyn_class in dynamics_dict and param_name in (dynamics_dict[dyn_class].parameters or {}):
                resolved = dyn_class
            else:
                for dyn_name, dyn in dynamics_dict.items():
                    if param_name in (dyn.parameters or {}):
                        resolved = dyn_name
                        break
            info = (resolved is not None and nmap.get(resolved)) or next(iter(nmap.values()))
            param_map[grid_key] = {
                "vars": [f"{info['op']}/{py_name}"],
                "nodes": info["nodes"],
            }

        return param_grid, param_map, axes

    def _df_to_exploration_result(
        self,
        results_df: "pd.DataFrame",
        results_map: "pd.DataFrame",
        axes: list,
        expl,
        expl_name: str,
    ):
        """Build ``ExplorationResult`` from PyRates ``grid_search`` output DataFrames.

        PyRates returns:
        * ``results_df``: MultiIndex columns ``(output_var, condition_name)``,
          rows = time points.
        * ``results_map``: rows indexed by condition_name, columns = param names.

        The resulting ``ExplorationResult`` stores a ``(n_conditions, n_time, n_vars)`` time-series array together with axes metadata.
        """
        import pandas as pd

        from tvbo.data.types import ExplorationResult

        exp = self.experiment
        dt = getattr(exp.integration, "step_size", None) if exp.integration else None

        cols = results_df.columns
        if isinstance(cols, pd.MultiIndex):
            output_vars = list(cols.get_level_values(0).unique())
        else:
            output_vars = ["output"]

        n_conditions = len(results_map)
        n_time = len(results_df)
        n_vars = len(output_vars)

        results_arr = np.zeros((n_conditions, n_time, n_vars), dtype=np.float64)

        for c_idx, cond_name in enumerate(results_map.index):
            if isinstance(cols, pd.MultiIndex):
                for v_idx, var_name in enumerate(output_vars):
                    col_data = results_df[var_name][cond_name]
                    if hasattr(col_data, "values"):
                        col_data = col_data.values
                    results_arr[c_idx, :, v_idx] = np.asarray(col_data).squeeze()
            else:
                results_arr[c_idx, :, 0] = results_df[cond_name].values.squeeze()

        # PyRates enumerates grid conditions in its own order (first axis fastest), but every other backend — and ExplorationResult.as_grid() — assumes the canonical C-order (axes[0] slowest … axes[-1] fastest). Reorder here by matching each condition's parameter values back to the axis coordinate indices, so `as_grid()` labels the grid correctly across all backends.
        _axis_vals = [getattr(ax, "explored_values", None) for ax in axes]
        grid_shape = tuple(0 if v is None else len(v) for v in _axis_vals)
        if len(grid_shape) > 1 and 0 not in grid_shape and int(np.prod(grid_shape)) == n_conditions:
            # Use the grid key carried on the axis (== results_map column); fall back to the trailing name segment only for axes built elsewhere.
            bare = [getattr(ax, "key", None) or str(getattr(ax, "name", "")).rsplit(".", 1)[-1] for ax in axes]
            vals = [np.asarray(v, dtype=float) for v in _axis_vals]
            ordered = np.empty_like(results_arr)
            filled = np.zeros(n_conditions, dtype=bool)
            for c_idx, cond_name in enumerate(results_map.index):
                midx = tuple(
                    int(np.argmin(np.abs(vals[k] - float(results_map.loc[cond_name, bare[k]])))) for k in range(len(axes))
                )
                pos = int(np.ravel_multi_index(midx, grid_shape))
                ordered[pos] = results_arr[c_idx]
                filled[pos] = True
            # Only adopt the reorder when it is a clean permutation (every cell filled exactly once). Degenerate axes with duplicate swept values collide, which would leave uninitialised cells — keep the original order in that case.
            if filled.all():
                results_arr = ordered

        observable = None
        obs = getattr(expl, "observable", None)
        if obs is not None:
            observable = getattr(obs, "function", None)

        return ExplorationResult(
            name=expl_name,
            axes=axes,
            results=results_arr,
            observable=observable,
            dt=dt,
            output_names=output_vars,
        )

    def _resolve_solver(self, solver: str | None, integration) -> str:
        """Resolve solver from input or integration method."""
        if solver is None:
            method = getattr(integration, "method", None)
            if method in TVBO_TO_PYRATES_SOLVER:
                solver = TVBO_TO_PYRATES_SOLVER[method]
            elif method in PYRATES_SOLVERS:
                solver = method
            elif method:
                raise ValueError(
                    f"Unsupported integration method '{method}' for PyRates. Supported: {list(TVBO_TO_PYRATES_SOLVER.keys())}"
                )
            else:
                solver = "heun"
        elif solver not in PYRATES_SOLVERS:
            raise ValueError(f"Invalid solver '{solver}'. Supported: {sorted(PYRATES_SOLVERS)}")
        return solver

    def _get_node_count(self, network) -> int:
        """Get number of nodes in network."""
        if network is None:
            return 0
        return getattr(network, "number_of_nodes", 0) or (
            len(network.nodes) if hasattr(network, "nodes") and network.nodes else 0
        )

    def _is_heterogeneous(self) -> bool:
        """Check if the experiment has heterogeneous dynamics (multiple models).

        PyRates vectorize=True cannot handle circuits where different nodes have different operators (different parameter counts / state variables).
        """
        dynamics_dict = self.build_dynamics_dict()
        return len(dynamics_dict) > 1

    def _pyrates_node_map(self) -> dict:
        """Map each dynamics model to its PyRates operator name and node labels.

        Mirrors the naming conventions of the PyRates YAML template exactly, so output paths and param_map entries resolve correctly.

        Returns
        -------
        dict
            ``{dyn_name: {'op': str, 'nodes': list[str]}}``
        """
        dynamics_dict = self.build_dynamics_dict()
        exp = self.experiment
        network = getattr(exp, "network", None)

        nmap: dict[str, dict] = {}

        if network and hasattr(network, "nodes") and network.nodes:
            # Schema-based Network: each node declares its dynamics
            default_dyn = next(iter(dynamics_dict), None)
            for node in network.nodes:
                label = str(getattr(node, "label", None) or f"node_{node.id}").replace(" ", "_").replace("-", "_")
                dyn_name = (
                    node.dynamics if isinstance(node.dynamics, str) else getattr(node.dynamics, "name", None)
                ) or default_dyn
                if dyn_name:
                    nmap.setdefault(dyn_name, {"op": f"{dyn_name}_op", "nodes": []})
                    nmap[dyn_name]["nodes"].append(label)
        elif network is not None:
            # Base Network/Connectome: all nodes share the default dynamics
            n_nodes = getattr(network, "number_of_nodes", 0)
            weights = getattr(network, "weights_matrix", None)
            if weights is not None:
                n_nodes = weights.shape[0]
            if hasattr(network, "node_labels") and network.node_labels:
                labels = [str(lbl).replace(" ", "_").replace("-", "_") for lbl in network.node_labels]
            elif hasattr(network, "region_labels") and network.region_labels:
                labels = [str(lbl).replace(" ", "_").replace("-", "_") for lbl in network.region_labels]
            else:
                labels = [f"node_{i}" for i in range(n_nodes)]
            dyn_name = next(iter(dynamics_dict))
            nmap[dyn_name] = {"op": f"{dyn_name}_op", "nodes": labels}
        else:
            # Single dynamics → node 'p' (matches YAML template)
            for dyn_name in dynamics_dict:
                nmap[dyn_name] = {"op": f"{dyn_name}_op", "nodes": ["p"]}

        return nmap

    def _load_circuit_from_yaml(self, include_edges: bool = True) -> tuple:
        """Load PyRates circuit from YAML template.

        Returns
        -------
        tuple
            (circuit, tmpdir, pkg_name) for cleanup
        """
        from pyrates.frontend import CircuitTemplate

        exp = self.experiment

        # Export to PyRates YAML
        yaml_content = exp.to_yaml(format="pyrates")

        # For large networks, strip edges from YAML
        if not include_edges:
            yaml_content = self._strip_edges_from_yaml(yaml_content)

        # Create temporary package
        tmpdir = tempfile.mkdtemp(prefix="tvbo_pyrates_")
        pkg_name = f"_tvbo_pyrates_{uuid.uuid4().hex[:8]}"
        pkg_path = os.path.join(tmpdir, pkg_name)
        os.makedirs(pkg_path, exist_ok=True)
        open(os.path.join(pkg_path, "__init__.py"), "w").close()

        yaml_path = os.path.join(pkg_path, "model.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        sys.path.insert(0, tmpdir)

        # Get circuit name
        network = getattr(exp, "network", None)
        dynamics = getattr(exp, "dynamics", None)

        if network is not None:
            circuit_name = getattr(network, "label", None) or getattr(network, "name", None) or "tvbo_circuit"
        elif dynamics is not None:
            model_name = getattr(dynamics, "name", None) or "tvbo_model"
            circuit_name = f"{model_name}_circuit"
        else:
            circuit_name = "tvbo_circuit"

        circuit = CircuitTemplate.from_yaml(f"{pkg_name}.model.{circuit_name}")
        return circuit, tmpdir, pkg_name

    def _strip_edges_from_yaml(self, yaml_content: str) -> str:
        """Remove edges section from YAML for large networks."""
        pattern = r"(  edges:\n(?:    - \[.*\]\n)*)"
        return re.sub(pattern, "", yaml_content)

    def _cleanup(self, tmpdir: str, pkg_name: str) -> None:
        """Clean up temporary files and modules."""
        if tmpdir in sys.path:
            sys.path.remove(tmpdir)
        modules_to_remove = [k for k in sys.modules if k.startswith(pkg_name)]
        for mod in modules_to_remove:
            del sys.modules[mod]
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Remove PyRates-generated .py files from cwd
        import glob

        for f in glob.glob(f"pyrates_run_{pkg_name}_*.py"):
            os.remove(f)

    def _add_edges_from_matrix(self, circuit, network) -> None:
        """Add edges to circuit using weight matrix (efficient for large networks)."""
        dynamics_dict = self.build_dynamics_dict()

        # Get node labels
        node_labels = []
        if hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                label = getattr(node, "label", None) or f"node_{node.id}"
                node_labels.append(str(label).replace(" ", "_").replace("-", "_"))
        else:
            n_nodes = network.weights_matrix.shape[0]
            node_labels = [f"node_{i}" for i in range(n_nodes)]

        # Get source/target variables from dynamics
        first_dyn = next(iter(dynamics_dict.values()))
        dyn_name = first_dyn.name

        # Source: prefer output, fallback to first state variable
        if first_dyn.output:
            src_var = f"{dyn_name}_op/{list(first_dyn.output.keys())[0]}"
        elif first_dyn.state_variables:
            src_var = f"{dyn_name}_op/{list(first_dyn.state_variables.keys())[0]}"
        else:
            src_var = f"{dyn_name}_op/x"

        # Target: coupling input
        ci = getattr(first_dyn, "coupling_inputs", None) or getattr(first_dyn, "coupling_terms", None)
        if ci:
            tgt_var = f"{dyn_name}_op/{list(ci.keys())[0]}"
        else:
            tgt_var = src_var

        # Add edges using matrix
        weights = network.weights_matrix
        if weights is not None and weights.size > 0:
            delays = self._get_delays(network)
            edge_attr = {"delay": delays} if delays is not None else None

            circuit.add_edges_from_matrix(
                source_var=src_var,
                target_var=tgt_var,
                source_nodes=node_labels,
                weight=weights,
                edge_attr=edge_attr,
                min_weight=1e-12,
            )

    def _get_delays(self, network):
        """Get delay matrix from network."""
        delays = None

        # Check for explicit delays from edges
        if hasattr(network, "_delays_from_edges"):
            delays = network._delays_from_edges()

        # If no explicit delays, compute from lengths/distances
        if delays is None:
            lengths = getattr(network, "lengths_matrix", None)
            if lengths is not None and np.any(lengths > 0):
                delays = network.calculate_delays()

        return delays

    def _build_outputs(self) -> dict:
        """Build PyRates outputs dict from dynamics state variables.

        Note: PyRates only tracks state variables (DEs) in its state vector.
        Algebraic outputs are computed post-hoc in _compute_outputs.
        """
        dynamics_dict = self.build_dynamics_dict()
        nmap = self._pyrates_node_map()

        # Single-node circuits use clean variable names (no node prefix)
        single_node = len(nmap) == 1 and len(next(iter(nmap.values()))["nodes"]) == 1

        outputs = {}
        for dyn_name, info in nmap.items():
            dyn = dynamics_dict.get(dyn_name)
            if not dyn:
                continue
            for node_label in info["nodes"]:
                for sv_name in dyn.state_variables or {}:
                    py_name = PYRATES_REPL.get(sv_name, sv_name)
                    key = sv_name if single_node else f"{node_label}_{sv_name}"
                    outputs[key] = f"{node_label}/{info['op']}/{py_name}"

        return outputs

    def _compute_outputs(self, result: "pd.DataFrame") -> "pd.DataFrame":
        """Compute algebraic output variables from PyRates simulation results.

        PyRates only records state variables. This method evaluates output equations (like 'r_eff = r_in*x') using recorded state values and parameters.
        """

        exp = self.experiment
        dynamics = self.build_dynamics_dict()

        if not dynamics:
            return result

        default_dyn = next(iter(dynamics.values()))
        network = getattr(exp, "network", None)

        # Build node mapping
        node_map = {}
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")
                node_map[safe_label] = node
                node_map[node.id] = node

        # Build edge map: target_node -> {coupling_term: source_col}
        edge_map = self._build_edge_map(network, node_map)
        n_time = len(result)

        # Compute outputs for each node
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")
                prefix = f"{safe_label}_"

                dyn_name = node.dynamics if isinstance(node.dynamics, str) else getattr(node.dynamics, "name", None)
                dyn = dynamics.get(dyn_name) if dyn_name else default_dyn

                if dyn and dyn.output:
                    self._compute_node_outputs(result, dyn, prefix, safe_label, edge_map, n_time)
        else:
            # Single dynamics case
            dyn = default_dyn
            if dyn and dyn.output:
                self._compute_single_outputs(result, dyn)

        return result

    def _build_edge_map(self, network, node_map: dict) -> dict:
        """Build edge map: target_node_label -> {coupling_term: source_col}."""
        edge_map = {}
        if network is None or not hasattr(network, "edges") or not network.edges:
            return edge_map

        for edge in network.edges:
            src_id = edge.source
            tgt_id = edge.target

            src_node = node_map.get(src_id)
            tgt_node = node_map.get(tgt_id)
            if src_node is None or tgt_node is None:
                continue

            src_label = getattr(src_node, "label", None) or f"node_{src_id}"
            tgt_label = getattr(tgt_node, "label", None) or f"node_{tgt_id}"
            src_safe = str(src_label).replace(" ", "_").replace("-", "_")
            tgt_safe = str(tgt_label).replace(" ", "_").replace("-", "_")

            src_var = getattr(edge, "source_var", None) or getattr(edge, "source_variable", None) or "r"
            tgt_var = getattr(edge, "target_var", None) or getattr(edge, "target_variable", None) or "r_in"

            if tgt_safe not in edge_map:
                edge_map[tgt_safe] = {}
            edge_map[tgt_safe][tgt_var] = f"{src_safe}_{src_var}"

        return edge_map

    def _compute_node_outputs(
        self,
        result: "pd.DataFrame",
        dyn,
        prefix: str,
        safe_label: str,
        edge_map: dict,
        n_time: int,
    ) -> None:
        """Compute algebraic outputs for a single node."""
        import sympy as sp

        scope = dyn.get_symbolic_elements()
        bodies = function_bodies(dyn)
        for out_name in dyn.output:
            out_var = dyn.derived_variables.get(out_name)
            if out_var is None:
                continue
            eq = out_var.equation
            if not states_an_expression(eq):
                continue

            expr = inline_functions(parse_eq(eq, local_dict=scope), bodies)
            subs = {}

            for sym in expr.free_symbols:
                sym_name = sym.name
                col_name = f"{prefix}{sym_name}"

                if col_name in result.columns:
                    subs[sym] = result[col_name].values
                elif dyn.parameters and sym_name in dyn.parameters:
                    param = dyn.parameters[sym_name]
                    if param.value is not None and not is_array_valued(param.value):
                        subs[sym] = float(param.value)
                elif safe_label in edge_map and sym_name in edge_map[safe_label]:
                    src_col = edge_map[safe_label][sym_name]
                    if src_col in result.columns:
                        subs[sym] = result[src_col].values
                    else:
                        subs[sym] = np.zeros(n_time)
                else:
                    subs[sym] = np.zeros(n_time)

            if len(subs) == len(expr.free_symbols):
                out_col = f"{prefix}{out_name}"
                func = sp.lambdify(list(subs.keys()), expr, "numpy")
                result[out_col] = func(*subs.values())

    def _compute_single_outputs(self, result: "pd.DataFrame", dyn) -> None:
        """Compute algebraic outputs for single dynamics case."""
        import sympy as sp

        scope = dyn.get_symbolic_elements()
        bodies = function_bodies(dyn)
        for out_name in dyn.output:
            out_var = dyn.derived_variables.get(out_name)
            if out_var is None:
                continue
            eq = out_var.equation
            if not states_an_expression(eq):
                continue

            expr = inline_functions(parse_eq(eq, local_dict=scope), bodies)
            subs = {}

            for sym in expr.free_symbols:
                if sym.name in result.columns:
                    subs[sym] = result[sym.name].values
                elif dyn.parameters and sym.name in dyn.parameters:
                    param = dyn.parameters[sym.name]
                    if param.value is not None and not is_array_valued(param.value):
                        subs[sym] = float(param.value)

            if len(subs) == len(expr.free_symbols):
                func = sp.lambdify(list(subs.keys()), expr, "numpy")
                result[out_name] = func(*subs.values())

    def _build_inputs(self) -> dict:
        """Build PyRates inputs dict from experiment stimulation."""
        exp = self.experiment
        inputs = {}

        stimulation = getattr(exp, "stimulation", None)
        if stimulation is None:
            return inputs

        duration = exp.integration.duration
        step_size = exp.integration.step_size
        time = np.arange(0, duration, step_size)

        try:
            stim_func = stimulation.execute(format="python")
        except Exception:
            stim_func = None

        if stim_func is None:
            return inputs

        stim_values = stim_func(time)
        target_var = self._stimulus_target_variable()
        regions = getattr(stimulation, "regions", None) or []
        weighting = getattr(stimulation, "weighting", None) or []

        network = getattr(exp, "network", None)
        # `node.dynamics` keys the network's library; a single-model experiment holds it itself.
        dynamics = dict(getattr(network, "dynamics", None) or {})
        single = getattr(exp, "dynamics", None)
        if getattr(single, "name", None):
            dynamics.setdefault(single.name, single)

        if network is not None and hasattr(network, "nodes") and network.nodes:
            nodes = list(network.nodes)
            if not regions:
                regions = [0]

            for i, region_idx in enumerate(regions):
                if region_idx >= len(nodes):
                    continue

                node = nodes[region_idx]
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")

                inline = None if isinstance(node.dynamics, str) else node.dynamics
                dyn_name = node.dynamics if inline is None else getattr(inline, "name", None)

                if dyn_name:
                    var = target_var or _legacy_input_variable(inline or dynamics.get(dyn_name))
                    if var is None:
                        raise ValueError(
                            f"No stimulus event names the variable to drive, and {dyn_name!r} "
                            "declares no 'I_ext' to fall back on. Name the target by giving the "
                            "stimulus an event whose name is the driven variable."
                        )
                    op_name = f"{dyn_name}_op"
                    key = f"{safe_label}/{op_name}/{var}"
                    weight = weighting[i] if i < len(weighting) else 1.0
                    inputs[key] = stim_values * weight

        return inputs

    def _stimulus_target_variable(self):
        """The dfun variable a stimulus drives, named by its event.

        TVBO expresses the target as the stimulus event's own *name*:
        `SimulationExperiment._resolve_events` lowers a declarative `target_variable` onto
        it, and that is where every other backend reads it. A `Stimulus` carries no target
        of its own, so asking one for `target_variable` yielded the `I_ext` default for
        every experiment ever run — addressing a variable the model need not have.
        """
        events = getattr(self.experiment, "events", None) or {}
        for key, event in events.items() if hasattr(events, "items") else []:
            if "stimul" in str(getattr(event, "event_type", "") or "").lower():
                name = getattr(event, "name", None) or key
                if name:
                    return str(name)
        return None

    def _df_to_simulation_result(self, result: "pd.DataFrame") -> "SimulationResult":
        """Convert PyRates pandas DataFrame result to SimulationResult."""
        import xarray as xr

        from tvbo.data.types import SimulationResult

        exp = self.experiment
        time = np.array(result.index)
        columns = list(result.columns)

        # Get known node labels
        known_node_labels = []
        network = getattr(exp, "network", None)
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")
                known_node_labels.append(safe_label)

        # Parse columns
        node_names = []
        sv_names = []
        node_sv_pairs = []

        for col in columns:
            node_name = None
            sv_name = None

            for node_label in known_node_labels:
                prefix = f"{node_label}_"
                if col.startswith(prefix):
                    node_name = node_label
                    sv_name = col[len(prefix) :]
                    break

            if node_name is None:
                parts = col.rsplit("_", 1)
                if len(parts) == 2:
                    node_name, sv_name = parts
                else:
                    node_name, sv_name = col, col

            node_sv_pairs.append((node_name, sv_name))
            if node_name not in node_names:
                node_names.append(node_name)
            if sv_name not in sv_names:
                sv_names.append(sv_name)

        n_time = len(time)
        n_nodes = len(node_names)
        n_svs = len(sv_names)

        data = np.zeros((n_time, n_svs, n_nodes), dtype=np.float32)

        for col_idx, (node_name, sv_name) in enumerate(node_sv_pairs):
            node_idx = node_names.index(node_name)
            sv_idx = sv_names.index(sv_name)
            data[:, sv_idx, node_idx] = result.iloc[:, col_idx].values

        da = xr.DataArray(
            data=data,
            dims=["time", "variable", "node"],
            coords={"time": time, "variable": sv_names, "node": node_names},
        )
        return SimulationResult(data=da)
