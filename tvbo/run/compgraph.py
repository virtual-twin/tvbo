"""Reference simulation of network dynamics on a computational graph.

Provides helpers that start each node's trace, compile a coupling with its states bound by name (`coupling_transforms`), propagate delayed coupling between nodes of a `networkx` graph, step every node synchronously by the declared integration method (`integration_step`), and collect the results into a [TimeSeries](../data/types.qmd).
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """No-op ``tqdm`` fallback used when the package is unavailable."""
        return x  # No-op if tqdm not available


from tvbo.classes.observation import expand_to_4d
from tvbo.data.types import TimeSeries


def initialize_graph_states_with_history(G):
    """Start each node's trace, its states at t = 0, dt, 2 dt, …, at its initial state, keeping any state already set.

    A node that already carries a ``"state"`` (``GraphRunner.setup_initial_conditions`` puts the model's initial values there, with the states the node declares for itself) keeps it — overwriting with zeros would start every run from the origin whatever the model declares. The trace is also the delay history: a delayed read before t = 0 sees the initial state (`delayed_state`).
    """
    for node in G.nodes:
        state_dim = len(G.nodes[node]["model"].state_variables)
        state = G.nodes[node].get("state")
        state = np.zeros(state_dim) if state is None else np.asarray(state, dtype=float).ravel()
        G.nodes[node]["state"] = state
        G.nodes[node]["trace"] = [state]


def delayed_state(trace, lag):
    """The state *lag* steps before the last one in *trace*: linearly interpolated between recorded steps, and the first state where the lag reaches past it, the constant history every backend fills its delay buffer with."""
    position = len(trace) - 1 - lag
    if position <= 0:
        return trace[0]
    below = int(np.floor(position))
    fraction = position - below
    return trace[below] if fraction == 0 else (1 - fraction) * trace[below] + fraction * trace[below + 1]


class CouplingTransforms(NamedTuple):
    """A coupling compiled for one model: what `coupling_transforms` returns and a node's afferent input goes through."""

    pre: Callable
    post: Callable
    zero: np.ndarray
    inject: Callable


def coupling_transforms(model, coupling) -> CouplingTransforms:
    """*coupling*'s pre- and post-transform on *model*, each name bound as `coupling_bindings` binds it for the code-generating backends.

    ``pre(source, target)`` takes the two nodes' state vectors and reads the source's gathered states (the coupling's ``incoming_states``, else the model's coupling variables, in that order) as ``x_j``, the target's as ``x_i``, a gathered source state by its bare name or as ``<state>_j``, and the target's own value of a state as ``<state>_i``. ``post(gx, target)`` reads the weighted sum ``gx``, its rows ``gx_<k>`` where ``pre`` is a list, and ``<state>_i``. ``zero`` is an empty sum, shaped as ``pre``'s output, so a node without afferents receives the post-transform of zero, as on every array backend. ``inject(rows)`` places row ``k`` of the post-transform at the state the ``python-network`` derivative function reads the model's ``k``-th global coupling input from (`gathered_state_indices`).

    Raises:
        ValueError: The post-transform yields fewer rows than the model has global coupling inputs.
    """
    from sympy import Symbol, lambdify

    from tvbo.parse.expression import parse_eq
    from tvbo.templates.base.utils import coupling_bindings, gathered_state_indices, get_coupling_terms
    from tvbo.utils import initial_value

    bindings = coupling_bindings(model, coupling)
    parameters = {str(name): p.value for name, p in (coupling.parameters or {}).items()}
    gathered = [bindings["sv_index"][name] for name in bindings["cvar_names"]]
    bare = [bindings["cvar_index"][name] for name in bindings["bare"]]
    source_rows = [row for _, row in bindings["pre_j"]]
    target_rows = [row for _, row in bindings["pre_i"]]
    post_rows = [row for _, row in bindings["post_i"]]

    def compile_(rhs, names):
        names = [*names, *parameters]
        return lambdify([Symbol(name) for name in names], parse_eq(rhs, parameters=names), "numpy")

    pre_fn = compile_(
        bindings["pre_rhs"],
        [
            "x_j",
            "x_i",
            *bindings["bare"],
            *(alias for alias, _ in bindings["pre_j"]),
            *(alias for alias, _ in bindings["pre_i"]),
        ],
    )
    post_fn = compile_(
        str(coupling.post_expression.rhs),
        ["gx", *(f"gx_{k}" for k in bindings["gx_indices"]), *(alias for alias, _ in bindings["post_i"])],
    )
    values = list(parameters.values())

    def pre(source, target):
        xj = source[gathered]
        return np.asarray(
            pre_fn(xj, target[gathered], *xj[bare], *xj[source_rows], *target[target_rows], *values), dtype=float
        )

    def post(gx, target):
        gx = np.atleast_1d(gx)
        return np.atleast_1d(np.asarray(post_fn(gx, *gx[bindings["gx_indices"]], *target[post_rows], *values), dtype=float))

    probe = np.array([initial_value(sv) for sv in model.state_variables.values()])
    with np.errstate(all="ignore"):
        zero = np.zeros_like(pre(probe, probe))
        n_rows = post(zero, probe).size
    inputs = gathered_state_indices(model)[: len(get_coupling_terms(model)[1])]
    if n_rows < len(inputs):
        raise ValueError(
            f"{coupling.name}'s post-transform yields {n_rows} row(s), and {model.name} reads {len(inputs)} global coupling input(s) from it"
        )

    def inject(rows):
        signal = np.zeros(len(probe))
        signal[inputs] = rows[: len(inputs)]
        return signal

    return CouplingTransforms(pre, post, zero, inject)


def compute_delayed_input_signal(node, G, dt):
    """Aggregate a node's afferent input at the start of the step its sources' traces end on.

    Graph edges point in signal direction, so the afferents are the node's in-edges: each one pre-transforms the source's state one delay before the step (`delayed_state`) against the node's own, and weights it; the node's coupling (`coupling_transforms`) post-transforms the sum, an empty one for a node without afferents, and places it where the node's derivative function reads it. A node without a coupling receives zero input.
    """
    data = G.nodes[node]
    transforms = data.get("coupling")
    if transforms is None:
        return np.zeros_like(data["state"])
    gx = transforms.zero
    for neighbor in G.predecessors(node):
        edge = G[neighbor][node]
        source = delayed_state(G.nodes[neighbor]["trace"], edge["delay"] / dt)
        gx = gx + edge["weight"] * transforms.pre(source, data["state"])
    return transforms.inject(transforms.post(gx, data["state"]))


def integration_step(integration):
    """One step of *integration*'s declared method, ``step(f, x, dt) -> x_next`` under the vector field ``f(x)``, evaluated from the method's intermediate and update expressions (`Integrator.enrich`), the scheme the code-generating backends render.

    An undeclared method is the schema's default (`BaseAdapter.declared_integration`). The noise and stimulus terms of a scheme are zero here: the Python backend integrates no noise, and a stimulus reaches a model through its derivative function. Both the graph runner and a single node's `Dynamics.run` step by it.

    Raises:
        ValueError: The method has no symbolic update expression, as an adaptive solver a backend supplies (Dopri5, VODE, Tsit5) has none.
    """
    from sympy import Symbol, lambdify, sympify

    from tvbo.adapters.base import BaseAdapter
    from tvbo.datamodel.schema import Integrator

    scheme = (
        integration if integration is not None else Integrator(method=BaseAdapter.declared_integration(None, "method"))
    ).enrich()
    update = getattr(getattr(scheme, "update_expression", None), "equation", None)
    if getattr(update, "rhs", None) is None:
        raise ValueError(
            f"the Python backend steps by the integration method's update expression, and {scheme.method!r} has none: it is an "
            "adaptive solver a backend supplies. Declare Euler, Heun or RungeKutta4thOrder, or run it on a backend that supplies the solver."
        )
    stages = list((scheme.intermediate_expressions or {}).items())
    zero = {Symbol("noise"): 0, Symbol("stimulus"): 0}
    head = [Symbol("X"), Symbol("dt")]
    derivatives = [Symbol("dX0")] + [Symbol(f"d{name}") for name, _ in stages]
    stage_fns = [
        lambdify(head + derivatives[: k + 1], sympify(stage.equation.rhs).subs(zero), "numpy")
        for k, (_, stage) in enumerate(stages)
    ]
    update_fn = lambdify(head + derivatives, sympify(update.rhs).subs(zero), "numpy")

    def field(f, x):
        return np.asarray(f(x), dtype=float)

    def step(f, x, dt):
        d = [field(f, x)]
        for stage_fn in stage_fns:
            d.append(field(f, stage_fn(x, dt, *d)))
        return np.asarray(x + update_fn(x, dt, *d), dtype=float)

    return step


def update_node_state_with_delay(G, node, t, dt, input_signal, step):
    """Advance *node* by one *step* (`integration_step`) under its afferent *input_signal*, which every stage of the step reads, and the dynamics parameters it declares for itself (its ``"parameters"``)."""
    run_kwargs = {"coupling": input_signal, **G.nodes[node].get("parameters", {})}
    if G.nodes[node].get("stimfun", None) is not None:
        run_kwargs["stimulus"] = G.nodes[node]["stimfun"]
    new_state = np.asarray(step(lambda u: G.nodes[node]["dfun"](u, t, **run_kwargs), G.nodes[node]["state"], dt), dtype=float)
    G.nodes[node]["state"] = new_state
    G.nodes[node]["trace"].append(new_state)


def simulate_graph_dynamics_with_delay(G, T, dt, integration=None):
    """Integrate the graph for ``round(T / dt)`` steps of *integration*'s declared method (`integration_step`), and return the time of each recorded state, ``dt`` to ``T``: the clock tvboptim and TVB record on.

    The update is synchronous: every node's afferent input is computed from the states at the start of the step before any node advances, so no node reads a neighbour's state from within the step. The input is read by every stage of the step, which is ``coupling_evaluation: per_step``; ``per_stage`` is refused wherever it would integrate another system (`coupling_evaluation_in_effect`).
    """
    from tvbo.adapters.base import coupling_evaluation_in_effect

    if coupling_evaluation_in_effect(integration, G.number_of_nodes()) == "per_stage":
        raise NotImplementedError(
            "integration.coupling_evaluation: per_stage re-evaluates the coupling at every stage of the step, and the Python graph runner "
            "computes each node's input once per step and holds it across the stages. Run the network on tvboptim, jax or networkdynamics, "
            "which honour per_stage, or declare per_step."
        )
    step = integration_step(integration)
    n_steps = int(round(T / dt))

    for k in tqdm(range(n_steps)):
        inputs = {node: compute_delayed_input_signal(node, G, dt) for node in G.nodes}
        for node in G.nodes:
            update_node_state_with_delay(G, node, k * dt, dt, inputs[node], step)
    return dt * np.arange(1, n_steps + 1)


def collect_time_series(G, time_points, labels=None):
    """Gather per-node simulation traces into a single `TimeSeries`.

    Each node's trace after its initial state is expanded to 4D and concatenated along the node axis, then wrapped in a [TimeSeries](../data/types.qmd) labelled with the model's state-variable names and, where *labels* are given, the nodes' labels.

    Args:
        G: The graph whose nodes hold their simulated `"trace"` and model metadata.
        time_points: The time of each recorded state, shared by all nodes (`simulate_graph_dynamics_with_delay`).
        labels: One label per node, in the graph's node order, carried as the ``"Space"`` labels; ``None`` leaves the nodes unlabelled.

    Returns:
        A `TimeSeries` holding the stacked node traces with state-variable and node labels.
    """
    node_time_series = []

    for node in G.nodes:
        expanded_ts = expand_to_4d(np.stack(G.nodes[node]["trace"][1:]))
        node_time_series.append(expanded_ts)
    time_series_4d = np.concatenate(node_time_series, axis=2)

    dimensions = {"State Variable": list(G.nodes[node]["model"].state_variables.keys())}
    if labels:
        dimensions["Space"] = [str(label) for label in labels]
    return TimeSeries(time_points, time_series_4d, labels_dimensions=dimensions)
