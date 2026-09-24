<%doc>
Compact delayed coupling: a concrete tvboptim DelayedCoupling that integrates over the connectome's nonzero edges.

Emitted once into a generated module that defines any delayed coupling; every delayed coupling class the cfun template renders then inherits from it. Built only on tvboptim's public coupling contract (prepare / precompute / compute / pre / post and the roll history buffer), so it runs on the released tvboptim as well as a development checkout.
</%doc>
<%def name="render_compact_delayed_coupling()">\
import equinox as _eqx
import numpy as _np


class CompactDelayedCoupling(DelayedCoupling):
    """A DelayedCoupling that visits only the connectome's nonzero edges and sums them without a scatter.

    The stock dense path gathers every one of the N^2 delayed source states each step, applies ``pre`` and weights to all of them, and reduces over the source axis, so a connectome with most entries zero pays for edges that carry nothing. Here the edges with nonzero weight at prepare() time are laid out as row-contiguous slots, padded per target to whole chunks of ``COMPACT_CHUNK``; each step reads its delayed states with one flat gather from the history buffer, applies ``pre`` and the weights per slot, and sums every chunk with a ones-contraction and every target's chunks with a small gather. The coupling is the stock one up to floating-point summation order.

    The nonzero pattern is fixed at prepare(), as a sparse graph's edge list is. Weights may change value on it (sweeps, fits, per-TR updates of edge parameters), and a weight that turns nonzero off it raises rather than being dropped. The gradient with respect to a structurally-zero weight is zero, as for a sparse graph. The compact path covers what tvbo emits, a dense delay graph read through the default roll buffer without interpolation or clamp warnings; any other configuration, or a nearly full graph, takes the stock path unchanged.
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
            pattern = _np.asarray(weights) != 0
        except jax.errors.TracerArrayConversionError:
            return data, state
        if pattern.mean() > self.COMPACT_MAX_DENSITY:
            return data, state
        data = Bunch(data)
        data._compact = _compact_edge_slots(pattern, self.COMPACT_CHUNK)
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
        steps = jnp.clip(jnp.rint(graph.delays[target, source] / data._compact_dt).astype(jnp.int32), 0, data._compact_newest)
        channel = jnp.arange(data._compact_channels)[:, None]
        data._compact_offsets = ((data._compact_newest - steps)[None, :] * data._compact_channels + channel) * data._compact_sources + source[None, :]
        weights = graph.weights
        slot_weights = jnp.where(slots.valid, weights[target, source], 0.0)
        data._compact_weights = _eqx.error_if(
            slot_weights,
            jnp.any((weights != 0) & slots.off_pattern),
            "A delayed coupling's weights gained a nonzero entry outside the pattern prepare() compacted; "
            "prepare the network again on the new weights.",
        )
        pre_params = Bunch(params)
        for name in self.EDGE_PARAMS:
            value = jnp.asarray(params[name])
            value = value.reshape(weights.shape) if value.ndim == 1 else value
            pre_params[name] = value[target, source]
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


def _compact_edge_slots(pattern, chunk):
    """Row-contiguous slots for the nonzero entries of *pattern*, each target padded to whole chunks.

    Returns a Bunch: ``target`` / ``source`` per slot (a padding slot points at its own row and source 0, so every read stays in bounds), ``valid`` (False on padding), ``chunk_table`` (each target's chunks, padded with the index of an extra all-zero chunk) and ``off_pattern`` (the entries a live weight must keep at zero).
    """
    n_target = pattern.shape[0]
    counts = pattern.sum(axis=1)
    padded = -(-counts // chunk) * chunk
    n_slots = int(padded.sum())
    target = _np.repeat(_np.arange(n_target), padded)
    source = _np.zeros(n_slots, dtype=_np.int64)
    valid = _np.zeros(n_slots, dtype=bool)
    first = _np.concatenate([[0], _np.cumsum(padded)[:-1]])
    for row in range(n_target):
        cols = _np.nonzero(pattern[row])[0]
        source[first[row] : first[row] + cols.size] = cols
        valid[first[row] : first[row] + cols.size] = True
    per_target = padded // chunk
    width = max(1, int(per_target.max(initial=0)))
    first_chunk = _np.concatenate([[0], _np.cumsum(per_target)[:-1]])
    column = _np.arange(width)
    chunk_table = _np.where(column[None, :] < per_target[:, None], first_chunk[:, None] + column[None, :], n_slots // chunk)
    return Bunch(
        target=jnp.asarray(target, dtype=jnp.int32),
        source=jnp.asarray(source, dtype=jnp.int32),
        valid=jnp.asarray(valid),
        chunk_table=jnp.asarray(chunk_table, dtype=jnp.int32),
        off_pattern=jnp.asarray(~pattern),
    )
</%def>
