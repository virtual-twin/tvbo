<%doc>
Compact delayed coupling: a concrete tvboptim DelayedCoupling that integrates over the connectome's nonzero edges.

Emitted once into a generated module that defines any delayed coupling; every delayed coupling class the cfun template renders then inherits from it. Built only on tvboptim's public coupling and graph contract (prepare / precompute / compute / pre / post, the roll history buffer, and a sparse graph's `edge_indices` and per-edge `data`), so it runs on the released tvboptim as well as a development checkout.
</%doc>
<%def name="render_compact_delayed_coupling()">\
import equinox as _eqx
import numpy as _np
from tvboptim.experimental.network_dynamics.graph import SparseGraph as _SparseGraph


class CompactDelayedCoupling(DelayedCoupling):
    """A DelayedCoupling that visits only the connectome's nonzero edges and sums them without a scatter.

    The stock dense path gathers every one of the N^2 delayed source states each step, applies ``pre`` and weights to all of them, and reduces over the source axis, so a connectome with most entries zero pays for edges that carry nothing; the stock sparse path visits only the stored edges but reduces them with a scatter-add. Here the edges (a dense graph's nonzero entries at prepare() time, a sparse graph's stored ones) are laid out as row-contiguous slots, padded per target to whole chunks of ``COMPACT_CHUNK``; each step reads its delayed states with one flat gather from the history buffer, applies ``pre`` and the weights per slot, and sums every chunk with a ones-contraction and every target's chunks with a small gather. The coupling is the stock one up to floating-point summation order.

    The edge list is fixed at prepare(), as a sparse graph's is. Weights may change value on it (sweeps, fits, per-TR updates of edge parameters); on a dense graph a weight that turns nonzero off it raises rather than being dropped, and a sparse graph cannot grow an entry. The gradient with respect to a structurally-zero weight is zero, as for a sparse graph. The compact path covers what tvbo emits, a delay graph, dense or sparse, read through the default roll buffer without interpolation or clamp warnings; any other configuration, or a nearly full graph, takes the stock path unchanged.
    """

    COMPACT_CHUNK = 32
    COMPACT_MAX_DENSITY = 0.75

    def prepare(self, network, dt, t0, t1):
        data, state = super().prepare(network, dt, t0, t1)
        graph = network.graph
        weights = getattr(graph, "weights", None)
        if (
            self.buffer_strategy != "roll"
            or self.history_interpolation is not None
            or self.warn_on_delay_clamp
            or getattr(weights, "ndim", None) != 2
            or not hasattr(graph, "delays")
        ):
            return data, state
        try:
            edges, off_pattern = _compact_edge_list(graph)
        except jax.errors.TracerArrayConversionError:
            return data, state
        n_target, n_source = weights.shape
        if edges.shape[0] > self.COMPACT_MAX_DENSITY * n_target * n_source:
            return data, state
        data = Bunch(data)
        data._compact = _compact_edge_slots(edges, n_target, self.COMPACT_CHUNK, off_pattern)
        data._compact_dt = dt
        data._compact_newest, data._compact_channels, data._compact_sources = (
            state.history.shape[0] - 1,
            state.history.shape[1],
            state.history.shape[2],
        )
        return data, state

    def precompute(self, coupling_data, params, graph):
        slots = coupling_data.get("_compact")
        if slots is None:
            return super().precompute(coupling_data, params, graph)
        data = Bunch(coupling_data)
        target, source = slots.target, slots.source
        sparse = isinstance(graph, _SparseGraph)
        if sparse:
            delays = graph.delays.data[slots.edge]
            slot_weights = jnp.where(slots.valid, graph.weights.data[slots.edge], 0.0)
        else:
            delays = graph.delays[target, source]
            slot_weights = _eqx.error_if(
                jnp.where(slots.valid, graph.weights[target, source], 0.0),
                jnp.any((graph.weights != 0) & slots.off_pattern),
                "A delayed coupling's weights gained a nonzero entry outside the pattern prepare() compacted; "
                "prepare the network again on the new weights.",
            )
        steps = jnp.clip(jnp.rint(delays / data._compact_dt).astype(jnp.int32), 0, data._compact_newest)
        channel = jnp.arange(data._compact_channels)[:, None]
        data._compact_offsets = ((data._compact_newest - steps)[None, :] * data._compact_channels + channel) * data._compact_sources + source[None, :]
        data._compact_weights = slot_weights
        pre_params = Bunch(params)
        for name in self.EDGE_PARAMS:
            value = jnp.asarray(params[name])
            if value.ndim == 2:
                pre_params[name] = value[target, source]
            elif sparse:
                pre_params[name] = value[slots.edge]
            else:
                pre_params[name] = value.reshape(graph.weights.shape)[target, source]
        data._compact_pre_params = pre_params
        return data

    def compute(self, t, state, coupling_data, coupling_state, params, graph):
        slots = coupling_data.get("_compact")
        if slots is None:
            return super().compute(t, state, coupling_data, coupling_state, params, graph)
        local_states = state[coupling_data.local_indices]
        delayed = coupling_state.history.reshape(-1).at[coupling_data._compact_offsets].get(mode="promise_in_bounds")
        target_states = local_states[:, slots.target] if self.PRE_USES_LOCAL else None
        messages = self.pre(delayed, target_states, coupling_data._compact_pre_params)
        n_out = messages.shape[0]
        chunk = self.COMPACT_CHUNK
        weighted = (messages * coupling_data._compact_weights[None, :]).reshape(n_out, -1, chunk)
        chunk_sums = weighted @ jnp.ones((chunk,), dtype=weighted.dtype)
        chunk_sums = jnp.concatenate([chunk_sums, jnp.zeros((n_out, 1), dtype=chunk_sums.dtype)], axis=1)
        return self.post(chunk_sums[:, slots.chunk_table].sum(axis=-1), local_states, params)


def _compact_edge_list(graph):
    """The graph's edges as a concrete ``(n_edges, 2)`` array of (target, source), in the order its per-edge data follows, and the entries a dense weight must keep at zero.

    A sparse graph's edges are its stored indices, which its ``weights.data`` and ``delays.data`` follow, and it has no off-pattern entries to guard (None). A dense graph's edges are the nonzero entries of its weight matrix at prepare() time, and every other entry must stay zero.
    """
    if isinstance(graph, _SparseGraph):
        return _np.asarray(graph.edge_indices), None
    pattern = _np.asarray(graph.weights) != 0
    return _np.argwhere(pattern), ~pattern


def _compact_edge_slots(edges, n_target, chunk, off_pattern):
    """Row-contiguous slots for *edges* (``(n_edges, 2)`` target/source pairs), each target padded to whole chunks.

    Returns a Bunch: ``target`` / ``source`` / ``edge`` per slot (``edge`` indexes *edges*; a padding slot points at its own row, source 0 and edge 0, so every read stays in bounds), ``valid`` (False on padding), ``chunk_table`` (each target's chunks, padded with the index of an extra all-zero chunk) and, for a dense graph, ``off_pattern`` (the entries a live weight must keep at zero).
    """
    target_e = _np.asarray(edges[:, 0], dtype=_np.int64)
    source_e = _np.asarray(edges[:, 1], dtype=_np.int64)
    order = _np.argsort(target_e, kind="stable")
    counts = _np.bincount(target_e, minlength=n_target)
    padded = -(-counts // chunk) * chunk
    n_slots = int(padded.sum())
    first = _np.concatenate([[0], _np.cumsum(padded)[:-1]])
    starts = _np.concatenate([[0], _np.cumsum(counts)[:-1]])
    row = target_e[order]
    slot = first[row] + _np.arange(order.size) - starts[row]
    target = _np.repeat(_np.arange(n_target), padded)
    source = _np.zeros(n_slots, dtype=_np.int64)
    edge = _np.zeros(n_slots, dtype=_np.int64)
    valid = _np.zeros(n_slots, dtype=bool)
    source[slot], edge[slot], valid[slot] = source_e[order], order, True
    per_target = padded // chunk
    width = max(1, int(per_target.max(initial=0)))
    first_chunk = _np.concatenate([[0], _np.cumsum(per_target)[:-1]])
    column = _np.arange(width)
    chunk_table = _np.where(column[None, :] < per_target[:, None], first_chunk[:, None] + column[None, :], n_slots // chunk)
    slots = Bunch(
        target=jnp.asarray(target, dtype=jnp.int32),
        source=jnp.asarray(source, dtype=jnp.int32),
        edge=jnp.asarray(edge, dtype=jnp.int32),
        valid=jnp.asarray(valid),
        chunk_table=jnp.asarray(chunk_table, dtype=jnp.int32),
    )
    if off_pattern is not None:
        slots.off_pattern = jnp.asarray(off_pattern)
    return slots
</%def>
