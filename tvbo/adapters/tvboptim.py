# -*- coding: utf-8 -*-
"""tvboptim adapter for tvbo.

Export (tvbo → tvboptim)

- :func:`to_tvboptim` — Network → tvboptim Network or DenseGraph / DenseDelayGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tvbo.adapters.base import BaseAdapter
from tvbo.adapters.smallscale.lowering import node_dynamics_name
from tvbo.utils import as_list, noise_sigma, normalize_params

if TYPE_CHECKING:
    from tvbo.classes.network import Network

# Integration method -> tvboptim native solver. Shared with the codegen templates
# (which render `SOLVER_MAP[method]`) so a recipe that runs through the experiment
# template also runs on the in-process heterogeneous path, on the same spellings.
SOLVER_MAP = {
    "euler": "Euler",
    "heun": "Heun",
    "heunstochastic": "Heun",
    "rk4": "RungeKutta4",
    "runge_kutta": "RungeKutta4",
    "rungekutta": "RungeKutta4",
    "rungekutta4": "RungeKutta4",
    "rungekutta4thorder": "RungeKutta4",
}


class TvboptimAdapter(BaseAdapter):
    """What the tvboptim templates need, resolved in Python before they emit anything."""

    def resolve_couplings(self):
        """The network's couplings, one key per distinct coupling.

        tvbo can reach the same coupling under more than one name — the function's, a
        coupling-input key, an explicit ``CouplingInput.source`` — and tvboptim wants one
        key per object, so the aliases are collapsed against the model that names them.
        """
        from tvbo.templates.tvboptim.utils import normalize_coupling_aliases

        return normalize_coupling_aliases(
            dict(super().resolve_couplings()), getattr(self.experiment, "dynamics", None)
        )


def _build_graph(network: "Network", delays: bool = True, max_delay: float | None = None):
    """Build a tvboptim graph from a tvbo Network, by what the network measures.

    - tract lengths present → ``DenseLengthGraph`` (delays = lengths /
      conduction_speed, so conduction speed is a live sweepable/differentiable leaf);
    - only explicit per-edge delays → ``DenseDelayGraph``;
    - no delays (or *delays* False) → ``DenseGraph``.

    ``max_delay`` sets the graph's ``max_delay_bound`` (static history-buffer size);
    ``None`` defaults it to the build-speed maximum delay.
    """
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.graph.base import (
        DenseDelayGraph,
        DenseLengthGraph,
    )

    weights = jnp.asarray(np.asarray(network.weights_matrix, dtype=float))
    labels = network.node_labels

    if not delays:
        return DenseGraph(weights=weights, region_labels=labels)

    lengths = network.lengths_matrix
    if lengths is not None and np.any(np.asarray(lengths) > 0):
        lengths = jnp.asarray(np.asarray(lengths, dtype=float))
        cs = getattr(network, "conduction_speed", None)
        speed = float(getattr(cs, "value", cs)) if cs is not None else 3.0
        # Size the bound the way DenseLengthGraph measures the largest delay: elementwise
        # ``lengths / speed`` then max — NOT ``max(lengths) / speed``. The two are equal in
        # exact arithmetic but differ by a float32 ULP for some speeds, landing the bound just
        # under the graph's own ``max(delay)`` and tripping its strict ``bound >= max(delay)``
        # check (fails for scattered conduction speeds, e.g. a sweep cell at v=5 while v=6
        # passes). A hair of headroom guarantees the buffer is never an ULP short.
        if max_delay is not None:
            bound = max_delay
        elif speed > 0:
            bound = float(jnp.max(lengths / speed)) * (1.0 + 1e-4)
        else:
            bound = 0.0
        return DenseLengthGraph(
            weights=weights,
            lengths=lengths,
            speed=speed,
            region_labels=labels,
            max_delay_bound=bound,
        )

    # No tract lengths: use explicit per-edge delays if the network carries them.
    try:
        edge_delays = network._delays_from_edges()
    except Exception:
        edge_delays = None
    if edge_delays is not None:
        delay_matrix = jnp.nan_to_num(jnp.asarray(np.asarray(edge_delays, dtype=float)))
        return DenseDelayGraph(
            weights=weights,
            delays=delay_matrix,
            region_labels=labels,
            max_delay_bound=max_delay,
        )

    return DenseGraph(weights=weights, region_labels=labels)


def _extract_noise(dyn_obj):
    """Extract tvboptim noise from tvbo dynamics state variable metadata.

    Iterates state variables looking for noise definitions.  Returns a
    tvboptim ``AdditiveNoise`` or ``MultiplicativeNoise`` when found,
    ``None`` otherwise.
    """
    svs = getattr(dyn_obj, "state_variables", None)
    if not svs:
        return None

    noisy_states = []
    sigma = None
    additive = True

    for sv_name, sv in svs.items():
        # A state variable whose noise declares no amplitude (or zero) is not a noise
        # target, exactly as in the codegen template; never a fabricated default.
        sv_sigma = noise_sigma(getattr(sv, "noise", None))
        if not sv_sigma:
            continue
        noisy_states.append(sv_name)
        sigma = float(sv_sigma)
        if getattr(sv.noise, "additive", True) is False:
            additive = False

    if not noisy_states:
        return None

    # Determine apply_to: None means all states
    all_sv_names = list(svs.keys())
    apply_to = noisy_states if set(noisy_states) != set(all_sv_names) else None

    if additive:
        from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

        return AdditiveNoise(apply_to=apply_to, sigma=sigma)
    else:
        from tvboptim.experimental.network_dynamics.noise import MultiplicativeNoise

        return MultiplicativeNoise(apply_to=apply_to, sigma=sigma)


def to_tvboptim(
    network: "Network",
    delays: bool | None = None,
    return_type: str = "network",
    dynamics=None,
    coupling=None,
    noise=None,
    max_delay: float | None = None,
    interpolate_delays: bool = False,
    **kwargs,
):
    """Export a tvbo Network to a tvboptim Network or graph object.

    When *dynamics* / *coupling* are not provided explicitly, they are
    auto-extracted from ``network.dynamics`` and ``network.coupling``
    using each object's ``.execute('tvboptim')`` method.

    Parameters
    ----------
    network : Network
        tvbo Network instance with weights (and optionally lengths) matrices.
    delays : bool or None, default=None
        Whether to include delay matrices in the graph.  When ``None``
        (default), auto-inferred from ``network.coupling``: uses delays
        only when at least one coupling has ``delayed=True``.
    return_type : str, default="network"
        ``"network"`` — return a full ``tvboptim.experimental.network_dynamics.Network``
        (requires *dynamics* and *coupling*).
        ``"graph"`` — return only the ``DenseGraph`` / ``DenseDelayGraph``.
    dynamics : AbstractDynamics, optional
        tvboptim dynamics instance. If not given, auto-extracted from
        ``network.dynamics``.
    coupling : AbstractCoupling | dict, optional
        tvboptim coupling instance(s). If not given, auto-extracted from
        ``network.coupling``.
    noise : AbstractNoise, optional
        tvboptim noise instance. Optional.
    max_delay : float, optional
        Concrete upper bound on the delay, forwarded to ``DenseDelayGraph`` to
        size the static history buffer. Pass it when the delays are meant to
        vary differentiably (e.g. ``delays = lengths / speed`` with ``speed``
        optimised) so the buffer length stays static while the delays may be
        JAX tracers. When ``None``, derived from the concrete delays.
    interpolate_delays : bool, default=False
        When True, enable linear interpolation between bracketing history steps
        on every delayed coupling, making the coupling differentiable w.r.t. the
        continuous delay (and hence conduction speed). Requires the ``"roll"``
        buffer strategy (the default).
    **kwargs
        Extra keyword arguments forwarded to the tvboptim ``Network`` constructor.

    Returns
    -------
    Network or DenseGraph or DenseDelayGraph
    """
    # Auto-infer delays from coupling metadata if not specified
    if delays is None:
        delays = False
        if hasattr(network, "coupling") and network.coupling:
            for coup_obj in network.coupling.values():
                if getattr(coup_obj, "delayed", False):
                    delays = True
                    break

    graph = _build_graph(network, delays=delays, max_delay=max_delay)

    if return_type == "graph":
        return graph

    # Auto-extract dynamics from network if not provided
    if dynamics is None and hasattr(network, "dynamics") and network.dynamics:
        dyn_key = next(iter(network.dynamics))
        dyn_obj = network.dynamics[dyn_key]
        dynamics = dyn_obj.execute("tvboptim")
    else:
        dyn_obj = None

    # Auto-extract coupling from network if not provided.
    # Resolution: use CouplingInput.source to remap function keys → CI keys,
    # then fall back to name matching, then positional order.
    if coupling is None and hasattr(network, "coupling") and network.coupling:
        coup_dict = {key: coup_obj.execute("tvboptim") for key, coup_obj in network.coupling.items()}
        if dynamics is not None and hasattr(dynamics, "COUPLING_INPUTS"):
            ci_keys = set(dynamics.COUPLING_INPUTS.keys())
            func_keys = list(coup_dict.keys())

            # Build func_name → ci_name mapping from source attribute
            remap = {}
            if dyn_obj is not None and hasattr(dyn_obj, "coupling_inputs") and dyn_obj.coupling_inputs:
                for ci_name, ci_obj in dyn_obj.coupling_inputs.items():
                    src = getattr(ci_obj, "source", None)
                    if src and src in coup_dict:
                        remap[src] = ci_name

            if remap:
                # Apply explicit source remapping
                coupling = {}
                for fk, fv in coup_dict.items():
                    ci_name = remap.get(fk, fk)
                    coupling[ci_name] = fv
            elif set(func_keys) <= ci_keys:
                # Names already match COUPLING_INPUTS
                coupling = coup_dict
            else:
                # Positional fallback
                coupling = list(coup_dict.values())
        else:
            coupling = coup_dict

    if dynamics is None or coupling is None:
        raise ValueError(
            "dynamics and coupling are required for return_type='network'. "
            "Set network.dynamics/coupling, pass them as kwargs, "
            "or use return_type='graph' for just the graph."
        )

    # Auto-extract noise from dynamics state variables if not provided
    if noise is None and dyn_obj is not None:
        noise = _extract_noise(dyn_obj)

    # Enable differentiable (interpolated) delays on every delayed coupling:
    # linear history interpolation makes d(state)/d(delay) informative, so
    # conduction speed becomes gradient-optimisable.
    if interpolate_delays:
        if isinstance(coupling, dict):
            _coups = coupling.values()
        elif isinstance(coupling, (list, tuple)):
            _coups = coupling
        else:
            _coups = [coupling]
        for _c in _coups:
            if hasattr(_c, "history_interpolation"):
                _c.history_interpolation = "linear"

    from tvboptim.experimental.network_dynamics import Network as TvboptimNetwork

    return TvboptimNetwork(
        dynamics=dynamics,
        coupling=coupling,
        graph=graph,
        noise=noise,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Heterogeneous networks (different dynamics per node on one shared connectome)
# ---------------------------------------------------------------------------


def is_heterogeneous(experiment) -> bool:
    """True when the experiment's nodes run more than one distinct dynamics.

    The trigger for the heterogeneous tvboptim path: a homogeneous experiment
    (one model on every node) uses :func:`to_tvboptim`, a heterogeneous one
    :func:`to_heterogeneous_network`. Delegates to the adapter-layer predicate
    every codegen backend already uses, so this path agrees with them about the
    nodes that declare no ``dynamics`` and fall back to the experiment's.
    """
    from tvbo.adapters.base import BaseAdapter

    return BaseAdapter(experiment).is_heterogeneous()


def _source_readout(source_var, state_names, group_name):
    """Map an ``Edge.source_var`` to a ``SignalRoute`` source (a state name).

    P1 supports state variables, including the ``<state>_out`` spelling (the
    node's coupling-source value, which for a plain neural mass equals the
    state). A ``source_var`` that names a derived/output variable — one that
    depends on the node's coupling input, e.g. a relay's ``r_eff`` — needs the
    upstream readout-signature change and is deferred (see plan problem #2).
    """
    states = tuple(state_names)
    if source_var is None:
        return states[0]
    name = source_var
    if name.endswith("_out") and name[:-4] in states:
        name = name[:-4]
    if name in states:
        return name
    raise NotImplementedError(
        f"source_var {source_var!r} on group {group_name!r} is not a state "
        f"variable {states}. Derived/output readouts (whose value depends on the "
        f"node's coupling input) need the upstream SignalRoute readout-signature "
        f"change; see _dev/knowledge/tvboptim-heterogeneous-adapter.md (problem #2)."
    )


def _target_input(target_var, dyn_optim, group_name):
    """Map an ``Edge.target_var`` to a ``SignalRoute`` target coupling-input name."""
    inputs = tuple(dyn_optim.COUPLING_INPUTS)
    if not inputs:
        raise ValueError(f"target group {group_name!r} declares no coupling inputs")
    if target_var is None:
        return inputs[0]
    if target_var in inputs:
        return target_var
    raise ValueError(
        f"target_var {target_var!r} is not a coupling input {inputs} on "
        f"group {group_name!r}"
    )


def _resolve_coupling(network, edge):
    """The ``Coupling`` governing an edge, as ``(name, object)``.

    ``Edge.coupling`` is a name-reference slot (``inlined: false``), so it arrives
    either as a bare name — resolved here against the network's declared
    couplings — or as an inline ``Coupling`` that ``Network.__init__`` reattached.
    An edge that names no coupling inherits the network's own, when the network
    declares exactly one. Returns ``(None, None)`` when nothing is declared
    anywhere, i.e. the default linear route.
    """
    declared = {
        getattr(c, "name", None): c for c in as_list(getattr(network, "coupling", None))
    }
    ref = getattr(edge, "coupling", None)
    if ref is None:
        return next(iter(declared.items())) if len(declared) == 1 else (None, None)
    if isinstance(ref, str):
        return ref, declared.get(ref)
    return getattr(ref, "name", None), ref


def _route_coupling(coupling, name, delayed: bool):
    """Build a selector-free ``PrePostCoupling`` for a route.

    A ``SignalRoute`` owns its source/local readouts, so the coupling must be
    constructed without ``incoming_states`` / ``local_states``. Linear coupling
    lowers directly — tvbo's ``a``/``b`` (``post_expression: a*gx + b``) are
    tvboptim's ``G``/``b``. Any other coupling function needs a selector-free
    emission of its own expressions, which is not implemented yet; it raises
    rather than being silently simulated as linear.
    """
    from tvboptim.experimental.network_dynamics.coupling import (
        DelayedLinearCoupling,
        LinearCoupling,
    )

    if name is not None and str(name) != "Linear":
        raise NotImplementedError(
            f"coupling {str(name)!r} is not yet lowered to a selector-free "
            "SignalRoute coupling; heterogeneous routes support Linear coupling."
        )
    params = normalize_params(getattr(coupling, "parameters", None))
    kwargs = {
        jl: float(getattr(params[tvbo_name], "value"))
        for tvbo_name, jl in (("a", "G"), ("b", "b"))
        if getattr(params.get(tvbo_name), "value", None) is not None
    }
    cls = DelayedLinearCoupling if delayed else LinearCoupling
    return cls(**kwargs)


def to_heterogeneous_network(
    network: "Network",
    *,
    dynamics_lib=None,
    default_dynamics: str | None = None,
    delays: bool | None = None,
    max_delay: float | None = None,
):
    """Build a tvboptim ``HeterogeneousNetwork`` from a heterogeneous tvbo Network.

    Nodes are partitioned into ``DynamicsGroup``s by their referenced dynamics
    (graph order = node order); edges are collapsed into ``SignalRoute``s keyed by
    ``(coupling NAME, target_var, delayed)`` — keying on object identity would split
    two edges naming one coupling into two routes, applying the shared graph twice. The shared graph is built by the same
    :func:`_build_graph` the homogeneous path uses, so the connectome weights
    (with signs) and delays carry over unchanged.

    Parameters
    ----------
    network : Network
        A heterogeneous tvbo Network (``is_heterogeneous(network)`` is True).
    dynamics_lib : mapping, optional
        ``{name: Dynamics}`` resolving each ``Node.dynamics`` reference.
        Defaults to ``network.dynamics``.
    default_dynamics : str, optional
        Name used for nodes that declare no ``dynamics`` of their own
        (``Node.dynamics`` is optional — the experiment's dynamics is the
        documented fallback).
    delays, max_delay
        Forwarded to :func:`_build_graph`; ``delays=None`` auto-infers from the
        network's couplings (any ``delayed=True``).
    """
    from tvboptim.experimental.network_dynamics import (
        DynamicsGroup,
        HeterogeneousNetwork,
        SignalRoute,
    )

    nodes = list(getattr(network, "nodes", None) or [])
    if not nodes:
        raise ValueError("to_heterogeneous_network: network has no nodes")
    dyn_lib = dict(dynamics_lib or getattr(network, "dynamics", None) or {})

    # --- partition nodes into groups by dynamics name (graph order = node order)
    # `Node.id` is a unique identifier, not a position, and edges address nodes by
    # id — so map ids to graph indices rather than conflating the two.
    node_index = network.node_index_map()
    group_idx: dict[str, list[int]] = {}
    node_group: list[str] = []
    for i, node in enumerate(nodes):
        dname = node_dynamics_name(node, default_dynamics)
        if dname is None:
            raise ValueError(
                f"node {i} ({getattr(node, 'label', '?')}) has no dynamics and the "
                "experiment declares none to fall back on"
            )
        group_idx.setdefault(dname, []).append(i)
        node_group.append(dname)

    if delays is None:
        # Infer from network-level coupling AND edge-level coupling: `delayed` may be
        # declared only on edges (Edge.coupling), and `coupling` can be a plain list
        # rather than a keyed dict — `.values()` would raise AttributeError on it.
        delays = any(
            getattr(c, "delayed", False) for c in as_list(getattr(network, "coupling", None))
        ) or any(
            getattr(getattr(e, "coupling", None), "delayed", False)
            for e in (getattr(network, "edges", None) or [])
        )
    graph = _build_graph(network, delays=delays, max_delay=max_delay)

    # --- DynamicsGroups (+ per-group AbstractDynamics for readout validation)
    groups = {}
    optim_dyn = {}
    for dname, idxs in group_idx.items():
        dyn_obj = dyn_lib.get(dname)
        if dyn_obj is None:
            raise ValueError(
                f"dynamics {dname!r} not found; pass dynamics_lib or set "
                "network.dynamics[{dname!r}]"
            )
        D = dyn_obj.execute("tvboptim")
        optim_dyn[dname] = D
        groups[dname] = DynamicsGroup(
            dynamics=D,
            nodes=np.asarray(idxs, dtype=int),
            noise=_extract_noise(dyn_obj),
        )

    # --- collapse edges into routes keyed by (coupling name, target_var, delayed)
    routes_acc: dict = {}
    for edge in getattr(network, "edges", None) or []:
        s, t = getattr(edge, "source", None), getattr(edge, "target", None)
        if s is None or t is None:
            continue  # template/matrix edges live entirely in graph.weights
        if s not in node_index or t not in node_index:
            raise ValueError(
                f"edge {s} -> {t} references a node id that the network does not "
                f"declare (known ids: {sorted(node_index)})"
            )
        sg, tg = node_group[node_index[s]], node_group[node_index[t]]
        cname, ccoup = _resolve_coupling(network, edge)
        edelayed = bool(getattr(ccoup, "delayed", delays)) if ccoup is not None else delays
        key = (cname, getattr(edge, "target_var", None), edelayed)
        acc = routes_acc.setdefault(key, {"source": {}, "target": {}, "coupling": ccoup})
        readout = _source_readout(getattr(edge, "source_var", None), optim_dyn[sg].STATE_NAMES, sg)
        prev = acc["source"].get(sg)
        if prev is not None and prev != readout:
            raise NotImplementedError(
                f"group {sg!r} emits inconsistent source_vars ({prev!r} vs "
                f"{readout!r}) on one route; a per-source route split is not yet "
                "implemented"
            )
        acc["source"][sg] = readout
        acc["target"][tg] = _target_input(getattr(edge, "target_var", None), optim_dyn[tg], tg)

    routes = {}
    single = len(routes_acc) == 1
    for i, ((cname, _target_var, edelayed), acc) in enumerate(routes_acc.items()):
        name = "coupling" if single else f"route_{i}"
        routes[name] = SignalRoute(
            source=acc["source"],
            coupling=_route_coupling(acc["coupling"], cname, edelayed),
            target=acc["target"],
        )

    return HeterogeneousNetwork(graph=graph, groups=groups, routes=routes)


def _heterogeneous_solution_to_dataarray(sol, het, network):
    """Assemble a ``HeterogeneousSolution`` into ONE labeled xarray ``DataArray``.

    Groups carry different state variables, so the ``variable`` axis is the
    ordered UNION of every group's variable names and a node holds ``NaN`` for
    variables its group lacks. Dims and coords are written directly in the
    canonical tvbo layout ``(time, variable, node, mode)`` — the final container
    is correctly keyed at assembly time, with no positional reshaping or
    ``TimeSeries`` round-trip downstream.

    ``sol.to_graph`` owns the group -> graph-node scatter and its NaN fill, so the
    ordering contract stays on the tvboptim side. Variables are written into a
    preallocated container rather than stacked, which would hold a second full copy
    of the result while it copies.
    """
    import xarray as xr

    var_order: list[str] = []
    for g in het.group_names:
        for v in sol.variable_names.get(g) or []:
            if v not in var_order:
                var_order.append(v)

    ts = np.asarray(sol.ts)
    first = np.asarray(sol.to_graph(var_order[0]))
    n_time, n_nodes = first.shape
    data = np.empty((n_time, len(var_order), n_nodes, 1), dtype=first.dtype)
    data[:, 0, :, 0] = first
    for k, v in enumerate(var_order[1:], 1):
        data[:, k, :, 0] = np.asarray(sol.to_graph(v))
    node_labels = [str(x) for x in (network.node_labels or list(range(n_nodes)))]
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "node", "mode"],
        coords={"time": ts, "variable": var_order, "node": node_labels, "mode": [0]},
        name="HeterogeneousSolution",
    )


def _declared_covariance(dyn_obj, context=None):
    """The one ``(covariance, axis)`` a dynamics declares, or ``(None, None)``.

    Resolved through ``param_io`` so the matrix carries the same provenance a
    parameter does — a literal, a file, or (the usual case) a ``producer:``. Every
    noisy state variable must agree on the structure: tvboptim draws one Wiener block
    per step for all of them, so two different covariances cannot both be imposed on
    it, and silently honouring one would change the science on the other.
    """
    from tvbo.data import param_io

    found = None
    for sv_name, sv in (dyn_obj.state_variables).items():
        noise = getattr(sv, "noise", None)
        cov = getattr(noise, "covariance", None) if noise is not None else None
        if cov is None:
            continue
        axis = getattr(noise, "correlated_over", None)
        if axis is None:
            raise ValueError(
                f"state variable {sv_name!r}: `noise.covariance` is set but "
                f"`noise.correlated_over` is not. A covariance without a named axis "
                f"does not identify a process; set it to `node` (or `state`/`mode`)."
            )
        axis = str(axis)
        matrix = np.asarray(param_io.resolve(cov, context=context), dtype=float)
        if found is None:
            found = (matrix, axis, sv_name)
        elif found[1] != axis or not np.array_equal(found[0], matrix):
            raise ValueError(
                f"state variables {found[2]!r} and {sv_name!r} declare different "
                f"correlated noise; one Wiener block is shared across all noisy states, "
                f"so they must declare the same covariance and axis."
            )
    return (found[0], found[1]) if found else (None, None)


def _correlated_noise_factors(lib, het, context=None):
    """Per-group covariance factors for every group whose dynamics declares one.

    Groups are named after their dynamics, so the library key *is* the group key. Each
    group keeps its own factor rather than one winning globally: a heterogeneous
    network may legitimately drive its groups with different processes, and a group
    that declares no covariance must keep its independent increment.

    Returns ``(factors_by_group, axis)``, or ``(None, None)`` when nothing is declared.
    """
    from tvbo.classes.correlated_noise import covariance_factor

    factors, axes = {}, set()
    for name in het.group_names:
        dyn = (lib or {}).get(name)
        if dyn is None:
            continue
        cov, axis = _declared_covariance(dyn, context=context)
        if cov is None:
            continue
        n_group_nodes = len(het.group_nodes[name])
        if axis in ("node", "region") and cov.shape[0] != n_group_nodes:
            raise ValueError(
                f"group {name!r}: `noise.covariance` is {cov.shape[0]}x{cov.shape[0]} "
                f"but the group has {n_group_nodes} nodes; a covariance over the node "
                f"axis must be square in the number of nodes."
            )
        factors[name] = covariance_factor(cov, name=f"{name}.noise.covariance")
        axes.add(axis)
    if not factors:
        return None, None
    if len(axes) > 1:
        raise ValueError(
            f"groups declare `noise.correlated_over` on different axes ({sorted(axes)}); "
            f"one solver mixes every group's increment, so they must agree."
        )
    return factors, axes.pop()


def _seed_group_noise(config, het, seed: int) -> None:
    """Seed every noisy group's PRNG from the experiment's resolved seed.

    ``prepare`` leaves each group on tvboptim's default ``jax.random.key(0)``, so
    without this the declared ``execution.random_seed`` is ignored and two groups
    of matching shape draw byte-identical noise. Folding the group's index into
    one base key gives each an independent stream that still reproduces from the
    declared seed.
    """
    import jax

    base = jax.random.key(int(seed))
    for i, name in enumerate(het.group_names):
        noise = getattr(config.groups[name], "noise", None)
        if noise is not None:
            noise.key = jax.random.fold_in(base, i)


def run_heterogeneous_tvboptim(experiment, *, dynamics_lib=None, seed=None, **kwargs):
    """Run a heterogeneous ``SimulationExperiment`` on tvboptim, in process.

    Builds a ``HeterogeneousNetwork`` from the experiment's network, integrates
    with a native fixed-step solver, and returns an ``ExperimentResult`` whose
    integration ``TimeSeries`` carries a per-group variable union (see
    :func:`_heterogeneous_solution_to_dataarray`). This is the P1 path that lets
    ``exp.run("tvboptim")`` handle heterogeneous networks without the codegen
    experiment template (that is a later milestone). *seed* overrides the
    recipe's ``execution.random_seed``. Unknown ``kwargs`` (e.g. ``benchmark``,
    ``mode``) are accepted and ignored.
    """
    from tvboptim.experimental.network_dynamics import prepare, solvers

    from tvbo.data.types import ExperimentResult, SimulationResult

    from tvbo.adapters.base import BaseAdapter

    network = experiment.network
    lib = dynamics_lib if dynamics_lib is not None else BaseAdapter(experiment).build_dynamics_dict()
    default_dyn = next(iter(lib), None) if dynamics_lib is None else None
    het = to_heterogeneous_network(network, dynamics_lib=lib, default_dynamics=default_dyn)

    integ = getattr(experiment, "integration", None)
    dur = getattr(integ, "duration", None)
    dt = getattr(integ, "step_size", None)
    dur = 1000.0 if dur is None else float(dur)
    dt = 0.1 if dt is None else float(dt)
    method = getattr(integ, "method", None)
    method = "heun" if method is None else str(getattr(method, "text", method)).lower()
    if method not in SOLVER_MAP:
        raise NotImplementedError(
            f"integration method {method!r} has no heterogeneous tvboptim solver; "
            f"declared one of {sorted(SOLVER_MAP)}"
        )
    n_steps = max(1, int(round(dur / dt)))
    solver = getattr(solvers, SOLVER_MAP[method])(block_size=min(100, n_steps))
    _factors, _axis = _correlated_noise_factors(lib, het, context=experiment)
    if _factors is not None:
        from tvbo.classes.correlated_noise import CorrelatedNoiseSolver

        solver = CorrelatedNoiseSolver(solver, _factors, axis=_axis)

    simulate, config = prepare(het, solver, t0=0.0, t1=dur, dt=dt)
    if seed is None:
        seed = getattr(getattr(experiment, "execution", None), "random_seed", None)
    _seed_group_noise(config, het, 0 if seed is None else seed)
    sol = simulate(config)
    da = _heterogeneous_solution_to_dataarray(sol, het, network)
    return ExperimentResult(
        integration=SimulationResult(data=da),
        source=experiment,
        name=getattr(experiment, "label", None),
    )
