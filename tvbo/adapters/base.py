"""Base adapter for processing SimulationExperiment metadata.

Extracts Python logic from Mako templates into reusable, testable methods.
Backend-specific adapters (NetworkDynamics, PyRates, etc.) inherit from BaseAdapter and override or extend as needed.
"""

from __future__ import annotations

import ast
import re
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.classes.experiment import SimulationExperiment

from tvbo.adapters.smallscale.lowering import node_dynamics_name
from tvbo.templates.base.utils import (
    collect_param_distributions,
    collect_sv_distributions,
    get_distribution_seed,
    graph_generator_call,
    has_distributions,
)
from tvbo.utils import network_couplings, noise_sigma


def dense_matrix(network, name: str, dtype=float) -> np.ndarray | None:
    """*network*'s edge matrix ``name`` as a dense array of ``dtype``, or ``None`` when it carries none.

    `Network.matrix` returns a matrix in its stored format, which may be sparse. A backend that integrates a dense connectome — TVB, CUDA, a Julia literal, a plot — says so through this one call rather than converting on its own, so "dense, this dtype, or None" is spelled once. A backend that can take the stored form reads `matrix` directly.
    """
    matrix = getattr(network, "matrix", None)
    if not callable(matrix):
        return None
    m = matrix(name, format="dense")
    return None if m is None else np.asarray(m, dtype=dtype)


def declared_node_count(network) -> int:
    """How many nodes a network declares, whichever of the equivalent forms declared them.

    A matrix-only or parcellation-only connectome must not read as a single node, so every form `to_yaml_with_network` accepts is consulted: an explicit count, explicit node objects, edges, a data file, a parcellation, or a weight matrix with more than one entry. A network declaring none of them, or no network at all, is one node.
    """
    if network is None:
        return 1
    count = int(getattr(network, "number_of_nodes", None) or getattr(network, "number_of_regions", None) or 0)
    if count > 1:
        return count
    nodes = getattr(network, "nodes", None) or []
    if len(nodes) > 1:
        return len(nodes)
    if getattr(network, "edges", None) or getattr(network, "data_file", None) or getattr(network, "parcellation", None):
        return max(count, 2)
    weights = dense_matrix(network, "weight")
    if weights is not None and weights.size > 1:
        return int(weights.shape[0])
    return max(count, 1)


def refuse_network(experiment, backend: str, reach: str) -> None:
    """Raise where *backend* would accept a declared network and integrate one node of it.

    A backend whose emitted code carries no connectome returns a well-formed one-node trajectory for a network experiment, which no caller can tell from support; refusing is the contract the Brian2 and NetworkDynamics adapters already state for the forms they cannot lower. *reach* names what the backend does integrate, for the message.
    """
    n = declared_node_count(getattr(experiment, "network", None))
    if n > 1:
        raise NotImplementedError(
            f"the {backend} backend integrates {reach}, so the {n}-node network this experiment declares would be accepted and ignored. "
            "Run it on a backend that lowers a connectome (tvb, tvboptim, jax, python, pyrates)."
        )


def edge_needs(
    network,
    required: tuple[str, ...] = ("weight",),
    delay_carriers: tuple[str, ...] = ("length",),
    delayed=None,
) -> list[tuple[str, tuple[str, ...]]]:
    """What a connectome backend reads off *network*'s edges, as ``[(what for, (attributes, any one of which supplies it)), ...]``.

    Each of *required* is needed on its own for any multi-node network — ``weight`` for the connectome. A delayed coupling needs one of *delay_carriers* besides, but only over a network that connects anything (`Network.has_connectome`): a node set has nothing to delay. *delayed* is the names of the delayed couplings, a bool, or ``None`` to read them off the network's own couplings. A network that generates its graph in the emitted code — a curated `graph_generator` ``type`` — needs nothing here; its matrices do not exist before the code runs.
    """
    if declared_node_count(network) <= 1 or getattr(getattr(network, "graph_generator", None), "type", None):
        return []
    needs = [("the connectome" if attr == "weight" else f"the {attr} it lowers", (attr,)) for attr in required]
    if delayed is None:
        delayed = [name for name, coupling in network_couplings(network).items() if getattr(coupling, "delayed", False)]
    if delayed and getattr(network, "has_connectome", True):
        names = ", ".join(delayed) if not isinstance(delayed, bool) else ""
        needs.append((f"the delayed coupling {names}".rstrip(), tuple(delay_carriers)))
    return needs


def require_edge_attributes(network, backend: str, needs) -> None:
    """Raise, naming what is missing, where *backend* would read an edge attribute *network* does not carry.

    *needs* is `edge_needs`'s list. Each is decided through `Network.carries`, from the header and the declaration, so nothing is read; an object without `carries` is left to the caller. Without this every backend here substitutes zeros or falls through to an instantaneous graph, and a run declared delayed integrates a different system and reports success.
    """
    carries = getattr(network, "carries", None)
    if not callable(carries):
        return
    for what_for, attributes in needs:
        if any(carries(a) for a in attributes):
            continue
        carried = ", ".join(getattr(network, "matrix_names", None) or []) or "no edge matrix"
        raise ValueError(
            f"the {backend} backend needs {' or '.join(attributes)} for {what_for}, and {network} carries {carried}. "
            "Add it to the network — a companion dataset edges/<name>, set_matrix(), or a per-edge parameter — or, for a delay, declare the coupling `delayed: false`."
        )


class BaseAdapter:
    """Base class for backend adapters.

    Provides shared metadata processing that all code-generation backends need: dynamics library, node-dynamics mapping, coupling resolution, graph info, initial state parsing, etc.

    A backend states what makes it different — its `TEMPLATE`, and a `prepare_context` override where the shared context will not do — rather than restating how rendering works. `render_code` is inherited from here by every adapter that renders one template from one context.
    """

    TEMPLATE: str = ""
    """The Mako template this backend renders, relative to the template lookup root.

    Declaring it is what lets `render_code` be inherited. An adapter that renders more than one template — NeuroML picks between four by what the dynamics declares — states that choice in its own `render_code` instead.
    """

    REQUIRED_EDGE_ATTRIBUTES: tuple[str, ...] = ("weight",)
    """Edge attributes this backend cannot lower a multi-node network without, each on its own. A backend that reads a leadfield or a receptor type off the edges adds it here and `refuse_unrenderable` names it when a network lacks it."""

    DELAY_CARRIERS: tuple[str, ...] = ("length",)
    """Edge attributes this backend derives a delayed coupling's delays from, any one sufficing. Tract lengths over a conduction speed is what TVB and the CUDA kernel size their history buffer with; a backend that also takes explicit per-edge delays lists ``delay``."""

    def __init__(self, experiment: SimulationExperiment):
        self.experiment = experiment

    def render_code(self, **kwargs) -> str:
        """This experiment as backend source.

        Args:
            **kwargs: Extra context, overriding the prepared context per key.

        Returns:
            The rendered source, unformatted — normalising is the caller's step, and the backends whose output is not Python have nothing to normalise it with.

        Raises:
            NotImplementedError: If the adapter declares no `TEMPLATE`.
        """
        return self.render_template(self.render_context(**kwargs))

    def render_context(self, **kwargs) -> dict:
        """The context `render_code` renders: `prepare_context`, overridden per key by *kwargs*, once `refuse_unrenderable` has passed the declaration.

        A backend that runs the source it renders reads the result layout off this same context rather than preparing a second one.
        """
        self.refuse_unrenderable()
        context = self.prepare_context()
        context.update(kwargs)
        return context

    def render_template(self, context: dict) -> str:
        """`TEMPLATE` rendered with *context*.

        Raises:
            NotImplementedError: If the adapter declares no `TEMPLATE`.
        """
        from tvbo import templates

        if not self.TEMPLATE:
            raise NotImplementedError(
                f"{type(self).__name__} declares no TEMPLATE. Set one, or override "
                "render_code where the backend chooses its template per experiment."
            )
        return templates.lookup.get_template(self.TEMPLATE).render(**context)

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
        return {node.id: node_dynamics_name(node, default_name) for node in nodes}

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
        """The network's couplings, keyed by the role each plays in it.

        The one place a backend asks what couplings an experiment has, so that a template never derives it: a template that reads the model itself is a second answer to a question this class already answers, and the two drift. A backend needing them keyed differently overrides this and calls up — see ``TvboptimAdapter``.
        """
        return OrderedDict(network_couplings(getattr(self.experiment, "network", None)))

    def get_default_coupling(self, all_couplings: OrderedDict | None = None):
        """The coupling a backend applies where an edge names none — the first declared."""
        if all_couplings is None:
            all_couplings = self.resolve_couplings()
        return next(iter(all_couplings.values()), None)

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
        return any(max(BaseAdapter.get_noise_sigmas(dyn), default=0.0) > 0 for dyn in dynamics_dict.values())

    # ── Graph / network ──────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """This backend's name as an error message spells it: the adapter's class name without its suffix."""
        return type(self).__name__.removesuffix("Adapter")

    def delayed_couplings(self) -> list[str]:
        """The names of the couplings declared delayed, through `resolve_couplings` so a backend keying them differently is read the same way."""
        return [name for name, coupling in (self.resolve_couplings() or {}).items() if getattr(coupling, "delayed", False)]

    def edge_needs(self) -> list[tuple[str, tuple[str, ...]]]:
        """This backend's :func:`edge_needs` for this experiment: `REQUIRED_EDGE_ATTRIBUTES`, and `DELAY_CARRIERS` where a coupling is delayed."""
        network = getattr(self.experiment, "network", None)
        return edge_needs(network, self.REQUIRED_EDGE_ATTRIBUTES, self.DELAY_CARRIERS, self.delayed_couplings())

    def refuse_missing_edge_attributes(self) -> None:
        """This adapter's :func:`require_edge_attributes`: raise, by name, where the network lacks an edge attribute this backend reads."""
        require_edge_attributes(getattr(self.experiment, "network", None), self.backend, self.edge_needs())

    def refuse_unrenderable(self) -> None:
        """Raise where this backend's templates would drop part of the declaration and emit well-formed code for the rest.

        By default this is `refuse_missing_edge_attributes`: a delayed coupling over a network with no tract lengths is the case every backend here would otherwise integrate instantaneous and report as a success. A backend overrides it — calling up — to name what else its templates do not lower, a delayed coupling with no history path, a declared observation with no monitor path, because rendering is not evidence: code that compiles and then answers a different question is indistinguishable from support to every caller that does not already know the answer.
        """
        self.refuse_missing_edge_attributes()

    def refuse_network(self, reach: str) -> None:
        """This adapter's :func:`refuse_network`: raise if the experiment declares a network the backend would integrate one node of."""
        refuse_network(self.experiment, self.backend, reach)

    def get_network_info(self) -> dict:
        """Extract network metadata: n_nodes, graph generator, edges, etc.

        ``has_graph_generator`` is the question a template actually asks, resolved once here: can this generator be lowered to a constructor call in the generated code? Only a generator naming a curated ``type`` can, because the lowering reads that entry's ``bindings:`` block. A generator declared by a Python ``builder:`` has already run and left its result in the weight and length matrices, so a template that treats the bare presence of a generator as "emit a constructor call" raises on it instead of emitting the matrices it was handed.
        """
        network = self.experiment.network
        n_nodes = getattr(network, "number_of_nodes", None) or getattr(
            network, "number_of_regions", 1
        )  # number_of_regions deprecated

        graph_gen = getattr(network, "graph_generator", None)
        nodes = getattr(network, "nodes", None) or []
        edges_list = getattr(network, "edges", None) or []

        return {
            "n_nodes": n_nodes,
            "nodes": nodes,
            "graph_gen": graph_gen,
            "has_graph_generator": bool(graph_gen and getattr(graph_gen, "type", None)),
            "has_explicit_edges": len(edges_list) > 0,
        }

    def listed_edges(self) -> list[tuple[int, int, object]]:
        """The network's explicit edges as ``(source row, target row, edge)``, in declaration order.

        Endpoints are resolved from `Node.id` through `Network.node_index_map`, the resolution `Network.matrix` applies, so a template that lists edges one by one addresses the same rows as the matrix it would otherwise emit. An edge without both endpoints is a template edge naming a companion matrix, not a connection, and is left out.
        """
        network = getattr(self.experiment, "network", None)
        placed = [e for e in (getattr(network, "edges", None) or []) if e.source is not None and e.target is not None]
        index_map = network.node_index_map() if placed else {}
        return [(index_map.get(e.source, e.source), index_map.get(e.target, e.target), e) for e in placed]

    def build_weight_matrix(self, listed_edges, threshold: int = 50) -> np.ndarray | None:
        """The weight matrix a template emits as a literal, source-by-target: ``W[i, j]`` is the weight of the edge ``i → j``, the orientation ``SimpleWeightedDiGraph(W)`` reads.

        `Network.matrix` is the one reading of the connectome — endpoints through `Network.node_index_map`, weights through `edge_param`, declared transforms applied — and it is target-by-source, the coupling convention, so it is transposed once here. ``None`` where the network lists at most *threshold* explicit edges (`listed_edges`), which a template emits one by one, and where it carries no matrix with more than one entry. A network that carries its connectome as a matrix has no edge objects at all — every builder-generated one is like this — and without its matrix the templates find no weights, no edges and no nameable generator, and the last branch of each builds an unweighted complete graph: the run succeeds and integrates a different network.
        """
        if listed_edges and len(listed_edges) <= threshold:
            return None
        W = dense_matrix(getattr(self.experiment, "network", None), "weight")
        return W.T if W is not None and W.size > 1 else None

    # ── Integration ──────────────────────────────────────────────────────

    FIXED_STEP_METHODS = frozenset({"Euler", "Heun", "RungeKutta4thOrder", "Identity"})
    """The canonical methods (`tvbo.utils.INTEGRATION_METHODS`) that advance by the step the recipe declares rather than choosing their own. A backend whose solver interface adapts by default (DifferentialEquations.jl) has to be told the step and told not to adapt it, so every template that emits a solve call needs the distinction."""

    @classmethod
    def is_fixed_step(cls, method) -> bool:
        """Whether *method* advances by a supplied step rather than choosing its own."""
        return cls.canonical_integration_method(method) in cls.FIXED_STEP_METHODS

    @staticmethod
    def declared_integration(integration, slot: str):
        """*integration*'s *slot*, or the schema's default for it (`Integrator`'s ``ifabsent``) where the slot, or the integration itself, is absent or ``None``.

        The one source of integration defaults: a backend integrates exactly the window the recipe would serialise to, and an explicit ``None`` reads like a slot left out rather than failing on ``float(None)``.
        """
        from tvbo.datamodel.schema import Integrator

        value = getattr(integration, slot, None)
        return getattr(Integrator(), slot) if value is None else value

    @classmethod
    def canonical_integration_method(cls, method) -> str:
        """*method* under its canonical name, `tvbo.utils.integration_method`'s, or unchanged where tvbo does not know the spelling.

        An unrecognised name is passed through rather than replaced: `Integrator.method` is an open vocabulary where the backend brings its own solver (``AutoTsit5``, ``TRBDF2``), and silently rewriting it would be worse than handing it on. An undeclared method is the schema's default (`declared_integration`).
        """
        from tvbo.utils import integration_method

        name = str(method or cls.declared_integration(None, "method"))
        return integration_method(name, strict=False) or name

    def get_integration_info(self) -> dict:
        """The window a backend integrates, and which part of it is settling rather than measurement.

        ``duration`` is the MEASURED window and ``transient_time`` is prepended to it, so the total a backend integrates is ``transient_time + duration`` and raising the settle never silently shortens the data. Resolved once here, because the settle is a property of the experiment rather than of any one backend: every backend that needs it in steps wants the same ``round(transient_time / dt)``, and three copies of that arithmetic is how two of them came to disagree about what ``duration`` meant.

        ``method`` is returned under its canonical name (`canonical_integration_method`), whatever spelling the recipe wrote: ``euler``, ``rk4`` and ``RungeKutta4thOrder`` each reach a backend as the one name its solver table is keyed by, so no backend carries a spelling table of its own.

        A slot the recipe leaves out or writes ``None``, and every slot of an experiment without an integration, is the schema's default (`declared_integration`).

        Returns:
            ``dt``, ``duration`` (measured), ``method`` (canonicalised), ``transient_time``, ``total_duration`` (``transient_time + duration``, the window to integrate), and the same split in integration steps as ``n_transient`` and ``n_measured`` -- the first of which is the cut index between the two.
        """
        from tvbo.adapters.observation_sampling import tvb_iround

        integration = getattr(self.experiment, "integration", None)
        dt = float(self.declared_integration(integration, "step_size"))
        duration = float(self.declared_integration(integration, "duration"))
        transient = float(self.declared_integration(integration, "transient_time"))
        return {
            "dt": dt,
            "duration": duration,
            "method": self.canonical_integration_method(self.declared_integration(integration, "method")),
            "transient_time": transient,
            "total_duration": transient + duration,
            "n_transient": tvb_iround(transient / dt) if dt else 0,
            "n_measured": tvb_iround(duration / dt) if dt else 0,
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

        Returns a list of (event, source) tuples where source is one of: 'experiment', 'node:{id}', 'edge:{idx}'.
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

        The shape below is the shared one, not a contract every adapter keeps: a backend whose template needs something else entirely overrides this — `Brian2Adapter` returns a spiking build description — so a caller wanting *this* shape must build the adapter it belongs to rather than a bare `BaseAdapter`.
        """
        from tvbo.adapters.julia_model import julia_solver

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
        dt = integration_info["dt"]
        method = integration_info["method"]

        is_hetero = self.is_heterogeneous(dynamics_dict, node_dynamics_map)
        is_stoch = self.is_stochastic_dynamics(dynamics_dict)

        listed_edges = self.listed_edges()
        W = self.build_weight_matrix(listed_edges)

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
            for dv_name in dyn.in_dependency_order("derived_variables"):
                if str(dv_name) not in vertex_dv_names:
                    vertex_dv_names.append(str(dv_name))

        # Auto-extract tstops from conditional derived-variable breakpoints
        tstops = set()
        for dyn in dynamics_dict.values():
            for dv in (dyn.derived_variables).values():
                for branch in getattr(dv.equation, "conditionals", None) or []:
                    cond = getattr(branch, "condition", "") or ""
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
            "has_graph_generator": network_info["has_graph_generator"],
            "listed_edges": listed_edges,
            "has_explicit_edges": network_info["has_explicit_edges"],
            # State
            "sv_names": list(model.state_variables.keys()),
            "n_sv": len(model.state_variables),
            "is_heterogeneous": is_hetero,
            "is_stochastic": is_stoch,
            # Integration
            "dt": dt,
            "duration": integration_info["duration"],
            "transient_time": integration_info["transient_time"],
            "total_duration": integration_info["total_duration"],
            "n_transient": integration_info["n_transient"],
            "n_measured": integration_info["n_measured"],
            "solver_method": julia_solver(method),
            # The step keywords of a DifferentialEquations.jl `solve` call: a fixed-step method is handed the declared step and told not to adapt it.
            "solve_kwargs": f"dt={dt}, adaptive=false, saveat={dt}" if self.is_fixed_step(method) else f"saveat={dt}",
            "needs_stiff": "auto" in method.lower(),
            # Graph
            "weight_matrix": W,
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


class ContinuationAdapter(BaseAdapter):
    """A backend that renders and runs one continuation at a time.

    The bifurcation backends do not render a whole experiment: they take a `(dynamics, continuation)` pair, once per continuation the experiment declares. This class resolves each pair, runs every one the experiment declares, and renders a pair through `TEMPLATE`, so a backend states only what it does with one resolved pair: `run_one`, and the context `_prepare_context` gives its template or a `render_continuation` of its own.
    """

    def continuations(self) -> dict:
        """Every continuation the experiment declares; empty when it declares none."""
        return getattr(self.experiment, "continuations", None) or {}

    def resolve_continuation(self, continuation=None):
        """*continuation* if the caller named one, else the experiment's first.

        `None` when the experiment declares none, which the caller reports in its own terms — there is no useful default for "continue what?".
        """
        if continuation is not None:
            return continuation
        return next(iter(self.continuations().values()), None)

    def resolve_dynamics(self, continuation):
        """The `Dynamics` *continuation* runs on.

        A continuation may name its own, which is how a heterogeneous experiment picks one of the several its network holds; otherwise it runs on the experiment's. Naming one that resolves nowhere raises rather than silently falling back to the experiment's, since continuing a different model than the one asked for is the kind of wrong answer that looks like a right one.
        """
        experiment = self.experiment
        named = getattr(continuation, "dynamics", None)
        if named:
            name = str(named)
            if getattr(experiment.dynamics, "name", None) == name:
                return experiment.dynamics
            network_dynamics = getattr(experiment.network, "dynamics", None) if experiment.network else None
            if isinstance(network_dynamics, dict) and name in network_dynamics:
                return network_dynamics[name]
            raise ValueError(
                f"Continuation names dynamics {name!r}, which is neither the experiment's nor one of its network's."
            )
        if experiment.dynamics is not None:
            return experiment.dynamics
        raise ValueError(f"Cannot resolve dynamics for continuation {continuation!r}.")

    def prepare(self, model, continuation, name: str | None = None):
        """*model* as *continuation* starts on it, after `refuse_unrunnable` has checked the pair: `start_dynamics`.

        `run` and `render_code` hand every backend the model this returns, so a spec no backend runs is refused, by name, before anything is rendered or run, and every backend starts from the same parameter values.
        """
        self.refuse_unrunnable(model, continuation, name)
        return self.start_dynamics(model, continuation)

    @staticmethod
    def refuse_unrunnable(model, continuation, name: str | None = None) -> None:
        """Refuse a *continuation* of *model* that no continuation backend runs, naming it by *name*, else by its own.

        Raises:
            ValueError: If the continuation, or the nested continuation of one of its branches, frees a parameter *model* does not declare; if it starts from ``initial_state.method: from_branch``, which no backend implements; or if it frees more than one parameter itself, since a two-parameter continuation is a branch of a one-parameter one (`is_codim2`).
        """
        from tvbo.utils import as_list

        if continuation is None:
            return
        label = name or getattr(continuation, "name", None) or "?"
        freed = as_list(getattr(continuation, "free_parameters", None))
        nested = [
            fp
            for branch in as_list(getattr(continuation, "branches", None))
            for fp in as_list(getattr(getattr(branch, "continuation", None), "free_parameters", None))
        ]
        declared = list(getattr(model, "parameters", None) or {})
        unknown = list(dict.fromkeys(str(fp.name) for fp in (*freed, *nested) if str(fp.name) not in declared))
        if unknown:
            raise ValueError(
                f"continuation {label!r} frees {', '.join(map(repr, unknown))}, which the dynamics "
                f"{getattr(model, 'name', None)!r} does not declare; its parameters are {', '.join(declared) or 'none'}."
            )
        method = getattr(getattr(continuation, "initial_state", None), "method", None)
        if method is not None and str(method) == "from_branch":
            raise ValueError(
                f"continuation {label!r} declares initial_state.method 'from_branch', which no continuation backend "
                "implements. A continuation from a special point of another is a branch of that continuation: declare it "
                "under its `branches`, with the `source_point` it starts from ('fold:1', 'hopf:all') and a nested "
                "`continuation` freeing the second parameter."
            )
        if len(freed) > 1:
            raise ValueError(
                f"continuation {label!r} frees {len(freed)} parameters ({', '.join(str(fp.name) for fp in freed)}), and a "
                "continuation frees one. A two-parameter continuation is a branch of it, started from one of its special "
                "points: declare the second parameter in the `free_parameters` of a branch's nested `continuation`, with "
                "the `source_point` it starts from."
            )

    @staticmethod
    def start_dynamics(model, continuation):
        """*model* with each parameter *continuation* frees set to the ``value`` it declares there, which is where the continuation starts; *model* itself where it declares none.

        The declared values replace the model's in a copy, so every backend integrates to its starting equilibrium and starts continuing from them, and the experiment's own dynamics is left as it was.
        """
        from tvbo.utils import as_list

        start = {
            str(fp.name): fp.value
            for fp in as_list(getattr(continuation, "free_parameters", None))
            if getattr(fp, "value", None) is not None
        }
        if not start:
            return model
        model = model.copy()
        for name, value in start.items():
            model.parameters[name].value = value
        return model

    @staticmethod
    def fortran_names(names, reserved=()) -> dict[str, str]:
        """Each of *names* as Fortran may declare it, keyed by the name: the one renaming both AUTO-07p Fortran emitters apply.

        Fortran is case-insensitive, so two names differing only in case are one symbol there. A name whose lowercase spelling is one of *reserved* (the enclosing subroutine's own arguments) is suffixed ``_par``; of names sharing a lowercase spelling, one starting in lowercase is suffixed ``low``, so ``A`` keeps its name and ``a`` becomes ``alow``. Every other name is its own.

        Raises:
            ValueError: If two of the Fortran names still differ only in case.
        """
        names = list(dict.fromkeys(str(name) for name in names))
        reserved = {str(name).lower() for name in reserved}
        lowered = [name.lower() for name in names]

        def rename(name):
            if name.lower() in reserved:
                return name + "_par"
            if name[0].islower() and lowered.count(name.lower()) > 1:
                return name + "low"
            return name

        renamed = {name: rename(name) for name in names}
        spelled: dict[str, list[str]] = {}
        for name, fortran in renamed.items():
            spelled.setdefault(fortran.lower(), []).append(name)
        clashes = [group for group in spelled.values() if len(group) > 1]
        if clashes:
            raise ValueError(
                "these names are one symbol in case-insensitive Fortran, and no rename separates them: "
                + "; ".join(", ".join(f"{name!r} as {renamed[name]!r}" for name in group) for group in clashes)
                + "."
            )
        return renamed

    SOURCE_KINDS = {"HB": "hopf", "LP": "fold", "BP": "bp"}
    """The special points a branch starts from, by canonical code (`tvbo.analysis.bifurcation.canonical_ty`), each with the name a result records its source by."""

    @classmethod
    def source_point(cls, branch, default: str) -> tuple[str, int | None]:
        """Where *branch* starts — its `source_point`, or *default* where it declares none — as ``(kind, index)``.

        The one reading of ``'<kind>:<index>'`` every continuation backend applies. ``kind`` is the canonical code of a `SOURCE_KINDS` special point, so ``hopf`` and ``HB`` name one kind. ``index`` is ``None`` for every point of that kind, written ``<kind>:all`` or a bare ``<kind>``, and otherwise the 1-based ordinal of one point, negative counting back from the last (`select_points`).

        Raises:
            ValueError: If the kind is none of `SOURCE_KINDS`, or the index is neither ``all`` nor a nonzero integer.
        """
        from tvbo.analysis.bifurcation import canonical_ty

        declared = getattr(branch, "source_point", None)
        text = str(default if declared in (None, "") else declared)
        kind_text, _, index_text = text.partition(":")
        kind = canonical_ty(kind_text)
        index_text = index_text.strip()
        index = int(index_text) if re.fullmatch(r"[+-]?\d+", index_text) else None
        if kind not in cls.SOURCE_KINDS or index == 0 or (index is None and index_text.lower() not in ("", "all")):
            names = ", ".join(sorted(cls.SOURCE_KINDS.values()))
            raise ValueError(
                f"branch {getattr(branch, 'name', '?')!r} declares source_point {text!r}; the accepted forms are "
                f"'<kind>' or '<kind>:all' for every point of a kind, and '<kind>:<n>' for one, n = 1 the first and "
                f"n = -1 the last, with <kind> one of {names}."
            )
        return kind, index

    @classmethod
    def periodic_orbit_source(cls, branch, default: str) -> int | None:
        """The Hopf point a periodic-orbit *branch* starts from, as `source_point`'s index.

        A periodic orbit is continued from a Hopf point, so a periodic-orbit branch naming any other kind is refused rather than emitted as a Hopf switch that finds nothing and reports nothing.

        Raises:
            ValueError: If the branch starts from a special point other than a Hopf point.
        """
        kind, index = cls.source_point(branch, default)
        if kind != "HB":
            raise ValueError(
                f"periodic-orbit branch {getattr(branch, 'name', '?')!r} declares source_point {getattr(branch, 'source_point', None)!r}, and a "
                "periodic orbit is continued from a Hopf point. Equilibrium branch switching at a branch point "
                "or a fold is not emitted by this backend; declare `hopf:<n>` or `hopf:all`, or continue the "
                "other equilibrium directly by seeding it (`initial_state: {method: given}`)."
            )
        return index

    @classmethod
    def is_codim2(cls, branch, parent) -> bool:
        """Whether *branch* of the continuation *parent* is a two-parameter continuation, the one rule every continuation backend sorts its branches by.

        It is when the branch's nested continuation frees a parameter other than *parent*'s first free parameter, which the branch inherits as its primary; the first such parameter is the second (`codim2_parameter`). The nested ``free_parameters`` may therefore list the second parameter alone (``[C]``) or after the inherited primary (``[p, C]``). A branch whose nested continuation frees nothing, or only the primary, continues a periodic orbit in the primary alone.
        """
        return cls.codim2_parameter(branch, parent) is not None

    @staticmethod
    def codim2_parameter(branch, parent):
        """The `FreeParameter` a two-parameter *branch* of *parent* is continued in besides the primary it inherits, by `is_codim2`'s rule, or ``None`` for a one-parameter branch."""
        from tvbo.utils import as_list

        primary = next((str(fp.name) for fp in as_list(getattr(parent, "free_parameters", None))), None)
        nested = as_list(getattr(getattr(branch, "continuation", None), "free_parameters", None))
        return next((fp for fp in nested if str(fp.name) != primary), None)

    @staticmethod
    def select_points(points, index: int | None) -> list:
        """The entries of *points* a `source_point` index selects: all of them for ``None``, else the one at that 1-based ordinal, negative counting back from the last.

        A continuation that found no point of the kind selects none.

        Raises:
            ValueError: If *index* is past either end of a non-empty *points*.
        """
        points = list(points)
        if index is None or not points:
            return points
        position = index - 1 if index > 0 else len(points) + index
        if not 0 <= position < len(points):
            raise ValueError(
                f"source point {index} is out of range: the continuation found {len(points)} ({', '.join(map(str, points))})."
            )
        return [points[position]]

    def run(self, **kwargs) -> BifurcationResult | dict[str, BifurcationResult]:
        """Run every continuation the experiment declares, each through `run_one` on the dynamics `resolve_dynamics` finds for it, as `prepare` starts it.

        Every continuation is prepared before any is run, so a spec `refuse_unrunnable` refuses fails before the first continuation starts.

        Args:
            **kwargs: Forwarded to every `run_one`.

        Returns:
            The one result when the experiment declares one continuation, else every result keyed by continuation name.

        Raises:
            ValueError: If the experiment declares no continuation, or `refuse_unrunnable` refuses one.
        """
        continuations = self.continuations()
        if not continuations:
            raise ValueError(
                "No continuations defined. Add continuation specs via exp.continuations or load from a bifurcation YAML."
            )
        models = {name: self.prepare(self.resolve_dynamics(cont), cont, name) for name, cont in continuations.items()}
        results = {name: self.run_one(models[name], cont, name, **kwargs) for name, cont in continuations.items()}
        return next(iter(results.values())) if len(results) == 1 else results

    def run_one(self, model, continuation, name: str, **kwargs) -> BifurcationResult:
        """Run *continuation* on *model*, the pair `run` resolved and prepared; *name* is the key `run` files the result under."""
        raise NotImplementedError(f"{type(self).__name__} runs no continuation.")

    def render_code(self, model=None, continuation=None, **kwargs) -> str:
        """This backend's source for one continuation.

        *continuation* defaults to the experiment's first, and *model* to the dynamics `resolve_dynamics` finds for it — the dynamics `run` continues — and the pair is prepared as `run` prepares it (`prepare`), so the rendered source is the source that runs; *kwargs* is extra context for `render_continuation`.
        """
        continuation = self.resolve_continuation(continuation)
        model = self.prepare(model or self.resolve_dynamics(continuation), continuation)
        return self.render_continuation(model, continuation, **kwargs)

    def render_continuation(self, model, continuation, **kwargs) -> str:
        """The source for *continuation* on *model*, both already resolved and prepared: `TEMPLATE` rendered with `_prepare_context`."""
        return self.render_template(self._prepare_context(model, continuation, **kwargs))

    @staticmethod
    def _prepare_context(model, continuation, **kwargs) -> dict:
        """The template context for *continuation* on *model*: the pair, and *kwargs* as extra context."""
        return {"model": model, "continuation": continuation, **kwargs}
