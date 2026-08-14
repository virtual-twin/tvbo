"""Base adapter for processing SimulationExperiment metadata.

Extracts Python logic from Mako templates into reusable, testable methods.
Backend-specific adapters (NetworkDynamics, PyRates, etc.) inherit from BaseAdapter and override or extend as needed.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tvbo.classes.experiment import SimulationExperiment

from tvbo.templates.base.utils import (
    collect_param_distributions,
    collect_sv_distributions,
    get_distribution_seed,
    graph_generator_call,
    has_distributions,
)
from tvbo.utils import noise_sigma


class BaseAdapter:
    """Base class for backend adapters.

    Provides shared metadata processing that all code-generation backends need:
    dynamics library, node-dynamics mapping, coupling resolution, graph info, initial state parsing, etc.
    """

    def __init__(self, experiment: SimulationExperiment):
        self.experiment = experiment

    # ── Dynamics library ─────────────────────────────────────────────────

    def build_dynamics_dict(self) -> OrderedDict:
        """Build an ordered dict of all unique Dynamics models.

        Always includes the default model first, then any additional dynamics from the network's dynamics library (for heterogeneous networks).
        """
        exp = self.experiment
        model = exp.dynamics
        d = OrderedDict()
        if model:
            d[model.name] = model
        net_dynamics = getattr(exp.network, "dynamics", None) if exp.network else None
        if isinstance(net_dynamics, dict):
            for name, dyn in net_dynamics.items():
                if name not in d:
                    d[name] = dyn
        return d

    # ── Node-dynamics mapping ────────────────────────────────────────────

    def build_node_dynamics_map(self) -> dict[int, str]:
        """Map each node id to its dynamics name.

        Nodes without an explicit dynamics assignment use the default model.
        Returns {node_id: dynamics_name}.
        """
        exp = self.experiment
        model = exp.dynamics
        default_name = model.name if model else None
        nodes = getattr(exp.network, "nodes", None) or []
        mapping = {}
        for node in nodes:
            dyn_name = None
            if hasattr(node, "dynamics") and node.dynamics:
                dyn_name = str(node.dynamics)
            mapping[node.id] = dyn_name or default_name
        return mapping

    def is_heterogeneous(
        self,
        dynamics_dict: OrderedDict | None = None,
        node_dynamics_map: dict | None = None,
    ) -> bool:
        """Check if the network has heterogeneous vertex types."""
        if dynamics_dict is None:
            dynamics_dict = self.build_dynamics_dict()
        if node_dynamics_map is None:
            node_dynamics_map = self.build_node_dynamics_map()
        if len(dynamics_dict) <= 1:
            return False
        default_name = next(iter(dynamics_dict.keys()))
        return any(v != default_name for v in node_dynamics_map.values())

    # ── Coupling ─────────────────────────────────────────────────────────

    def resolve_couplings(self) -> OrderedDict:
        """Collect all unique coupling models as an OrderedDict."""
        exp = self.experiment
        nw_couplings = getattr(exp.network, "coupling", None) or {}
        if isinstance(nw_couplings, dict) and nw_couplings:
            return OrderedDict(nw_couplings)
        coupling = exp.coupling
        if coupling:
            return OrderedDict({coupling.name: coupling})
        return OrderedDict()

    def get_default_coupling(self, all_couplings: OrderedDict | None = None):
        """Return the default (first) coupling model."""
        if all_couplings is None:
            all_couplings = self.resolve_couplings()
        if all_couplings:
            return next(iter(all_couplings.values()))
        return self.experiment.coupling

    # ── Coupling dimension ───────────────────────────────────────────────

    @staticmethod
    def get_coupling_vars(dynamics) -> list[str]:
        """Get names of state variables marked as coupling variables."""
        if not dynamics or not dynamics.state_variables:
            return []
        return [name for name, sv in dynamics.state_variables.items() if getattr(sv, "coupling_variable", False)]

    @staticmethod
    def get_outdim(dynamics) -> int:
        """Coupling output dimension: number of coupling variables, or n_sv."""
        if not dynamics or not dynamics.state_variables:
            return 1
        cvars = BaseAdapter.get_coupling_vars(dynamics)
        return len(cvars) if cvars else len(dynamics.state_variables)

    @staticmethod
    def get_outsym_names(dynamics, outdim: int, coupling=None) -> list[str]:
        """Output symbol names for the edge model.

        Uses coupling.outsym if available, otherwise generates from coupling variables or state variables.
        """
        # Prefer coupling-defined outsym
        if coupling and getattr(coupling, "outsym", None):
            return list(coupling.outsym)
        if outdim == 1:
            return ["coupling"]
        cvars = BaseAdapter.get_coupling_vars(dynamics)
        if cvars:
            return [f"flow_{v}" for v in cvars]
        return [f"flow_{name}" for name in list(dynamics.state_variables.keys())[:outdim]]

    # ── Static / stochastic detection ────────────────────────────────────

    @staticmethod
    def is_static(dynamics) -> bool:
        """Check if dynamics is a static model (no differential equations)."""
        sys_type = getattr(dynamics, "system_type", None)
        if sys_type and "static" in str(sys_type).lower():
            return True
        return not dynamics.state_variables

    @staticmethod
    def is_stochastic_dynamics(dynamics_dict: OrderedDict) -> bool:
        """Detect a stochastic system: any state variable with a positive noise amplitude."""
        return any(
            BaseAdapter.get_noise_sigmas(dyn) and max(BaseAdapter.get_noise_sigmas(dyn)) > 0 for dyn in dynamics_dict.values()
        )

    # ── Graph / network ──────────────────────────────────────────────────

    def get_network_info(self) -> dict:
        """Extract network metadata: n_nodes, graph generator, edges, etc."""
        network = self.experiment.network
        n_nodes = getattr(network, "number_of_nodes", None) or getattr(
            network, "number_of_regions", 1
        )  # number_of_regions deprecated

        graph_gen = getattr(network, "graph_generator", None)
        nodes = getattr(network, "nodes", None) or []
        edges_list = getattr(network, "edges", None) or []

        # Edge matrix files
        emf_list = getattr(network, "edge_matrix_files", None) or []
        emf_names = []
        for f in emf_list:
            if hasattr(f, "name") and f.name:
                emf_names.append(str(f.name))
            elif isinstance(f, str):
                emf_names.append(f)
            else:
                fname = getattr(f, "file_name", None) or getattr(f, "path", None)
                if fname:
                    emf_names.append(str(fname))

        has_edge_matrix = len(emf_names) > 0
        has_explicit_edges = len(edges_list) > 0
        is_directed = has_edge_matrix or (graph_gen and getattr(graph_gen, "directed", False))

        return {
            "n_nodes": n_nodes,
            "nodes": nodes,
            "graph_gen": graph_gen,
            "edges_list": edges_list,
            "emf_names": emf_names,
            "has_edge_matrix": has_edge_matrix,
            "has_explicit_edges": has_explicit_edges,
            "is_directed": is_directed,
        }

    def build_weight_matrix(self, edges_list, n_nodes: int, threshold: int = 50) -> np.ndarray | None:
        """Build a dense weight matrix from explicit edges if over threshold."""
        if not edges_list or len(edges_list) <= threshold:
            return None
        W = np.zeros((n_nodes, n_nodes))
        for e in edges_list:
            w = 1.0
            params = getattr(e, "parameters", None) or []
            for p in params:
                pname = getattr(p, "name", None) or (p.get("name") if isinstance(p, dict) else None)
                if pname == "weight":
                    w = float(getattr(p, "value", None) or (p.get("value") if isinstance(p, dict) else 1.0))
            W[e.source, e.target] = w
        return W

    # ── Integration ──────────────────────────────────────────────────────

    def get_integration_info(self) -> dict:
        """Extract integration parameters."""
        integration = self.experiment.integration
        return {
            "dt": float(integration.step_size) if integration else 0.01,
            "duration": float(integration.duration) if integration else 1000.0,
            "method": (getattr(integration, "method", "Tsit5") if integration else "Tsit5"),
        }

    # ── Per-node parameter parsing ───────────────────────────────────────

    @staticmethod
    def parse_node_parameters(node) -> dict[str, float]:
        """Parse per-node parameter overrides from a Node object.

        Node.parameters is now a keyed dict {name: Parameter} (inlined).
        Returns {param_name: value}.
        """
        node_params = {}
        params = getattr(node, "parameters", None)
        if not params:
            return node_params
        # New format: dict keyed by ParameterName -> Parameter object
        if isinstance(params, dict):
            for name, p in params.items():
                if hasattr(p, "value"):
                    node_params[str(name)] = p.value
                elif isinstance(p, dict):
                    node_params[str(name)] = p.get("value")
            return node_params
        # Legacy fallback: list of ParameterName strings or dicts
        for p in params:
            if isinstance(p, dict):
                node_params[p.get("name", "")] = p.get("value")
            elif hasattr(p, "name") and not isinstance(p, str):
                node_params[getattr(p, "name", "")] = getattr(p, "value", None)
            elif isinstance(p, str):
                try:
                    d = ast.literal_eval(str(p))
                    if isinstance(d, dict) and "name" in d:
                        node_params[d["name"]] = d.get("value")
                except (ValueError, SyntaxError):
                    pass
        return node_params

    # ── Distribution collection ──────────────────────────────────────────

    @staticmethod
    def collect_all_distributions(dynamics_dict: OrderedDict) -> dict:
        """Collect SV and parameter distributions from all dynamics.

        Returns {dyn_name: {'sv': [...], 'param': [...], 'has': bool, 'seed': int}}
        """
        result = {}
        for dyn_name, dyn in dynamics_dict.items():
            result[dyn_name] = {
                "sv": collect_sv_distributions(dyn),
                "param": collect_param_distributions(dyn),
                "has": has_distributions(dyn),
                "seed": get_distribution_seed(dyn),
            }
        return result

    # ── Noise ────────────────────────────────────────────────────────────

    @staticmethod
    def get_noise_sigmas(dynamics) -> list[float]:
        """Per-state-variable noise amplitude σ, ``0.0`` where none is declared."""
        return [noise_sigma(getattr(sv, "noise", None)) or 0.0 for sv in (dynamics.state_variables or {}).values()]

    # ── Events ────────────────────────────────────────────────────────

    def collect_events(self) -> list:
        """Collect all events from experiment, nodes, and edges.

        Returns a list of (event, source) tuples where source is one of:
        'experiment', 'node:{id}', 'edge:{idx}'.
        """
        exp = self.experiment
        events = []

        # Experiment-level events
        for ev in (getattr(exp, "events", None) or {}).values():
            events.append((ev, "experiment"))

        # Node-level events
        for node in getattr(exp.network, "nodes", None) or []:
            for ev in (getattr(node, "events", None) or {}).values():
                events.append((ev, f"node:{node.id}"))

        # Edge-level events
        for i, edge in enumerate(getattr(exp.network, "edges", None) or []):
            for ev in (getattr(edge, "events", None) or {}).values():
                events.append((ev, f"edge:{i}"))

        return events

    # ── Coupling observables ─────────────────────────────────────────

    def get_coupling_observed(self, all_couplings) -> dict:
        """Extract observed (obsf/obssym) from coupling definitions.

        Returns {coupling_name: [DerivedVariable, ...]}.
        """
        result = {}
        for c_name, c in all_couplings.items():
            obs = getattr(c, "observed", None) or {}
            if obs:
                result[c_name] = list(obs.values()) if isinstance(obs, dict) else list(obs)
        return result

    # ── Execution config ─────────────────────────────────────────────

    def get_execution_info(self) -> dict:
        """Extract execution config (find_fixpoint, etc.)."""
        execution = getattr(self.experiment, "execution", None)
        return {
            "find_fixpoint": bool(getattr(execution, "find_fixpoint", False)),
        }

    # ── Full context for templates ───────────────────────────────────────

    def prepare_context(self) -> dict:
        """Build the full pre-computed context dict for template rendering.

        This is the main entry point: templates receive this dict instead of doing metadata processing themselves.
        """
        exp = self.experiment
        model = exp.dynamics

        dynamics_dict = self.build_dynamics_dict()
        node_dynamics_map = self.build_node_dynamics_map()
        all_couplings = self.resolve_couplings()
        coupling = self.get_default_coupling(all_couplings)

        coupling_vars = self.get_coupling_vars(model)
        outdim = self.get_outdim(model)
        outsym_names = self.get_outsym_names(model, outdim, coupling)

        network_info = self.get_network_info()
        integration_info = self.get_integration_info()

        is_hetero = self.is_heterogeneous(dynamics_dict, node_dynamics_map)
        is_stoch = self.is_stochastic_dynamics(dynamics_dict)

        # Weight matrix for large explicit-edge networks
        W = self.build_weight_matrix(network_info["edges_list"], network_info["n_nodes"])

        # Coupling weight parameter detection
        cparam_names = list((coupling.parameters or {}).keys()) if coupling else []
        weight_sym = "w" if "w" in cparam_names else ("weight" if "weight" in cparam_names else None)

        # Distribution info
        dist_info = self.collect_all_distributions(dynamics_dict)
        needs_random = any(d["has"] for d in dist_info.values())
        dist_seed = next((d["seed"] for d in dist_info.values() if d["has"]), 0)

        # Events
        all_events = self.collect_events()
        has_events = len(all_events) > 0

        # Coupling observables
        coupling_observed = self.get_coupling_observed(all_couplings)

        # Vertex derived-variable names (union across all dynamics)
        vertex_dv_names = []
        for dyn in dynamics_dict.values():
            for dv_name in getattr(dyn, "derived_variables", None) or {}:
                if str(dv_name) not in vertex_dv_names:
                    vertex_dv_names.append(str(dv_name))
        # Also from default model
        for dv_name in getattr(model, "derived_variables", None) or {}:
            if str(dv_name) not in vertex_dv_names:
                vertex_dv_names.append(str(dv_name))

        # Auto-extract tstops from conditional derived-variable breakpoints
        import re

        tstops = set()
        for dyn in list(dynamics_dict.values()) + [model]:
            if not dyn:
                continue
            for dv in (getattr(dyn, "derived_variables", None) or {}).values():
                if getattr(dv, "conditional", False):
                    for case in getattr(dv, "cases", None) or []:
                        cond = getattr(case, "condition", "") or ""
                        # Extract numeric values from conditions like "t <= 14400"
                        for m in re.findall(r"[\d.]+", cond):
                            try:
                                val = float(m)
                                if val > 0:
                                    tstops.add(val)
                            except ValueError:
                                pass
        tstops = sorted(tstops)

        # Execution config
        exec_info = self.get_execution_info()

        return {
            # Core objects (still needed by sub-templates)
            "experiment": exp,
            "model": model,
            "network": exp.network,
            "integration": exp.integration,
            # Pre-computed metadata
            "dynamics_dict": dynamics_dict,
            "node_dynamics_map": node_dynamics_map,
            "all_couplings": all_couplings,
            "coupling": coupling,
            "coupling_vars": coupling_vars,
            "outdim": outdim,
            "outsym_names": outsym_names,
            # Network
            "n_nodes": network_info["n_nodes"],
            "nodes": network_info["nodes"],
            "graph_gen": network_info["graph_gen"],
            "edges_list": network_info["edges_list"],
            "emf_names": network_info["emf_names"],
            "has_edge_matrix": network_info["has_edge_matrix"],
            "has_explicit_edges": network_info["has_explicit_edges"],
            "is_directed": network_info["is_directed"],
            # State
            "sv_names": list(model.state_variables.keys()),
            "n_sv": len(model.state_variables),
            "is_heterogeneous": is_hetero,
            "is_stochastic": is_stoch,
            # Integration
            "dt": integration_info["dt"],
            "duration": integration_info["duration"],
            "solver_method": integration_info["method"],
            "needs_stiff": ("auto" in str(integration_info["method"]).lower()),
            # Graph
            "needs_weighted": network_info["has_edge_matrix"],
            "weight_matrix": W,
            "weight_sym": weight_sym,
            # Distributions
            "dist_info": dist_info,
            "needs_random": needs_random,
            "dist_seed": dist_seed,
            # Events
            "all_events": all_events,
            "has_events": has_events,
            # Coupling observables
            "coupling_observed": coupling_observed,
            # Vertex observables
            "vertex_dv_names": vertex_dv_names,
            # Discontinuity times (from conditional derived variables)
            "tstops": tstops,
            # Execution
            "find_fixpoint": exec_info["find_fixpoint"],
            # Utilities (pass functions for template use)
            "is_static": self.is_static,
            "parse_node_parameters": self.parse_node_parameters,
            "get_noise_sigmas": self.get_noise_sigmas,
            "graph_generator_call": graph_generator_call,
        }
