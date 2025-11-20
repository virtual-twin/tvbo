import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm

from tvbo.knowledge.simulation.observation import expand_to_4d
from tvbo.data.types import TimeSeries


def initialize_graph_states_with_history(G, delay_buffer=1000):
    """Initialize states, time-series, and history buffer for each node."""
    for node in G.nodes:
        state_dim = len(G.nodes[node]["model"].metadata.state_variables)
        G.nodes[node]["state"] = np.zeros(state_dim)
        G.nodes[node]["time-series"] = np.empty((0, state_dim))
        G.nodes[node]["history"] = np.zeros((delay_buffer, state_dim))


def compute_delayed_input_signal(node, G, t, dt):
    """Compute input signal using delayed states."""
    neighbors = list(G.predecessors(node))
    input_signal = np.zeros_like(G.nodes[node]["state"])

    for neighbor in neighbors:
        G[neighbor][node]["coupling"]
        delay = G[neighbor][node]["delay"]
        time_series = G.nodes[neighbor]["time-series"]
        if len(time_series) > 1:
            delayed_time = t - delay
            interp_func = interp1d(
                np.arange(len(time_series)) * dt,
                time_series,
                axis=0,
                fill_value="extrapolate",
            )
            delayed_state = xj = interp_func(delayed_time)
            pre = G[neighbor][node]["prefun"](xj)
            input_signal += G[node][neighbor]["weight"] * pre

    return G[neighbor][node]["postfun"](input_signal)


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
    G.nodes[node]["time-series"] = np.append(
        G.nodes[node]["time-series"], new_state.reshape(1, -1), axis=0
    )
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
    node_time_series = []

    for node in G.nodes:
        time_series = G.nodes[node]["time-series"]
        expanded_ts = expand_to_4d(time_series)
        node_time_series.append(expanded_ts)
    time_series_4d = np.concatenate(node_time_series, axis=2)

    ts = TimeSeries(
        time_points,
        time_series_4d,
        labels_dimensions={
            "State Variable": list(
                G.nodes[node]["model"].metadata.state_variables.keys()
            )
        },
    )

    return ts
