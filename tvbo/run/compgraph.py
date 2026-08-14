"""Reference simulation of network dynamics on a computational graph.

Provides SciPy-based helpers that build per-node state and history buffers, propagate delayed coupling between nodes of a `networkx` graph, integrate each
node's dynamics with `odeint`, and collect the results into a [TimeSeries](../data/types.qmd).
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """No-op ``tqdm`` fallback used when the package is unavailable."""
        return x  # No-op if tqdm not available


from tvbo.classes.observation import expand_to_4d
from tvbo.data.types import TimeSeries


def initialize_graph_states_with_history(G, delay_buffer=1000):
    """Allocate the trace and delay-history buffers, keeping any state already set.

    A node that already carries a ``"state"`` (``GraphRunner.setup_initial_conditions`` puts the
    model's declared initial values there) keeps it — overwriting with zeros would start every
    run from the origin whatever the model declares. The delay history is filled with that state
    rather than zeros, so a delayed read before the trace exists sees the initial condition.
    """
    for node in G.nodes:
        state_dim = len(G.nodes[node]["model"].state_variables)
        state = G.nodes[node].get("state")
        state = np.zeros(state_dim) if state is None else np.asarray(state, dtype=float).ravel()
        G.nodes[node]["state"] = state
        G.nodes[node]["time-series"] = np.empty((0, state_dim))
        G.nodes[node]["history"] = np.tile(state, (delay_buffer, 1))


def compute_delayed_input_signal(node, G, t, dt):
    """Aggregate a node's delayed afferent input.

    Graph edges point in signal direction, so the afferents are the node's
    in-edges: each one pre-transforms the source's delayed state and weights
    it, and the shared post-transform is applied once to the sum (also while
    the delay history is still filling, matching the matrix backends). Mixed
    post-transforms across one node's in-edges raise, and a node without
    afferents receives zero input.
    """
    input_signal = np.zeros_like(G.nodes[node]["state"])
    post = None

    for neighbor in G.predecessors(node):
        edge = G[neighbor][node]
        if post is None:
            post = (edge["post_src"], edge["postfun"])
        elif post[0] != edge["post_src"]:
            raise ValueError(f"node {node}: mixed post-transforms on incoming edges ({post[0]} vs {edge['post_src']})")
        time_series = G.nodes[neighbor]["time-series"]
        if len(time_series) > 1:
            delayed_time = t - edge["delay"]
            interp_func = interp1d(
                np.arange(len(time_series)) * dt,
                time_series,
                axis=0,
                fill_value="extrapolate",
            )
            xj = interp_func(delayed_time)
            input_signal += edge["weight"] * edge["prefun"](xj)

    return input_signal if post is None else post[1](input_signal)


def update_node_state_with_delay(G, node, t, dt, input_signal):
    """Update node state considering delayed input."""
    current_state = G.nodes[node]["state"]

    # Integrate using odeint
    t_span = [t, t + dt]

    run_kwargs = {
        "coupling": input_signal,
    }

    if G.nodes[node].get("stimfun", None) is not None:
        run_kwargs["stimulus"] = G.nodes[node]["stimfun"]

    result = odeint(
        lambda u, t,: G.nodes[node]["dfun"](u, t, **run_kwargs),
        current_state,
        t_span,
    )
    # result = odeint(model, current_state, t_span, args=(input_signal,))
    new_state = result[-1]

    # Update state and history
    G.nodes[node]["state"] = new_state
    G.nodes[node]["time-series"] = np.append(G.nodes[node]["time-series"], new_state.reshape(1, -1), axis=0)
    G.nodes[node]["history"] = np.roll(G.nodes[node]["history"], shift=-1, axis=0)
    G.nodes[node]["history"][-1] = new_state


def simulate_graph_dynamics_with_delay(G, T, dt):
    """Run the simulation over the graph considering delays."""
    time_points = np.arange(0, T, dt)

    for t_idx in tqdm(range(len(time_points))):
        t = time_points[t_idx]

        for node in G.nodes:
            input_signal = compute_delayed_input_signal(node, G, t, dt)
            update_node_state_with_delay(G, node, t, dt, input_signal)
    return time_points


def collect_time_series(G, time_points):
    """Gather per-node simulation traces into a single `TimeSeries`.

    Each node's recorded `"time-series"` is expanded to 4D and concatenated along the node axis, then wrapped in a [TimeSeries](../data/types.qmd)
    labelled with the model's state-variable names.

    Args:
        G: The graph whose nodes hold the simulated `"time-series"` arrays and
            model metadata.
        time_points: The time axis shared by all nodes.

    Returns:
        A `TimeSeries` holding the stacked node traces with state-variable
        dimension labels.
    """
    node_time_series = []

    for node in G.nodes:
        time_series = G.nodes[node]["time-series"]
        expanded_ts = expand_to_4d(time_series)
        node_time_series.append(expanded_ts)
    time_series_4d = np.concatenate(node_time_series, axis=2)

    ts = TimeSeries(
        time_points,
        time_series_4d,
        labels_dimensions={"State Variable": list(G.nodes[node]["model"].state_variables.keys())},
    )

    return ts
