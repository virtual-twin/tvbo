<%doc>
Delay-history carry: the registry of delayed couplings, `_carry_delay_history` and `_refill_history`, emitted once into every generated tvboptim experiment module, where the experiment template places them before `create_network`. `render_delay_history_carry()` renders them on their own, as `render_compact_delayed_coupling()` renders the compact coupling.
</%doc>
<%def name="render_delay_history_carry()">\
_DELAY_LINES = {}
"""Coupling-input name -> (the state variables a delayed coupling transmits, the ``buffer_strategy`` of its history buffer), filled by create_network() from each built coupling; _carry_delay_history() reads it to rebuild each history buffer."""


def _carry_delay_history(state, result):
    """Hand a finished solve's delay history on to the next one, so chained solves integrate one continuous run.

    A prepared solve starts every delayed coupling from the history buffer in ``state.initial_state.coupling``, and returns only its trajectory. A loop that chains solves (one TR each) and carries only ``initial_state.dynamics`` would restart every delay line from that stale buffer. The buffer a continuous run would hold at this point is its old rows followed by the newly integrated rows of the transmitted states, truncated to its length, which is what this writes back, oldest first, the order tvboptim's ``prepare`` lays a roll and a circular buffer out in: a circular buffer is read oldest first from its write pointer, and the pointer then restarts at 0. A buffer of any other kind (``preallocated``) is refused by name. Raises when a transmitted state is not among the recorded variables, since the history cannot then be rebuilt.
    """
    _names = tuple(getattr(result, "variable_names", None) or ())
    for _cn in list(state.initial_state.coupling.keys()):
        _cs = state.initial_state.coupling[_cn]
        if "history" not in _cs:
            continue
        _sent, _kind = _DELAY_LINES.get(_cn, (None, None))
        if _sent is None or any(_v not in _names for _v in _sent):
            raise ValueError(f"coupling {_cn!r}: cannot carry its delay history, the transmitted states {_sent} are not all recorded in {_names}")
        if _kind not in ("roll", "circular"):
            raise NotImplementedError(f"coupling {_cn!r}: carrying a delay history across chained solves does not support the {_kind!r} buffer")
        if not _sent:
            continue
        _old = jnp.roll(_cs.history, -_cs.write_idx, axis=0) if _kind == "circular" else _cs.history
        _rows = result.data[:, jnp.asarray([_names.index(_v) for _v in _sent]), :]
        _line = {"history": jnp.concatenate([_old, _rows.astype(_old.dtype)], axis=0)[-_old.shape[0]:]}
        if _kind == "circular":
            _line["write_idx"] = jnp.zeros_like(_cs.write_idx)
        state = eqx.tree_at(lambda _s: [_s.initial_state.coupling[_cn][_k] for _k in _line], state, list(_line.values()))
    return state


def _refill_history(state):
    """Fill every delay line of *state* with the opening it integrates from, as TVB fills its history with the initial conditions.

    tvboptim's ``prepare`` fills each delayed coupling's history buffer from the network's initial state, which on a network without a stored history is the dynamics' ``INITIAL_STATE`` on every node, and this module then writes the opening (a draw, a node's declared state, a seeded operating point, the working-point ramp, a swept initial condition, a trial's draw) into ``initial_state.dynamics`` alone. Every site that writes an opening hands the state here after the write, and every row of every buffer becomes the opening's transmitted states: the buffer ``prepare`` builds on a network whose initial state is that opening, in the buffer's own shape, so a dense and a compact coupling are filled alike. A buffer that continues a settle or an earlier solve is no opening and is never refilled. Without a delayed coupling *state* is returned as it is.
    """
    _lines = [_cn for _cn, _cs in state.initial_state.coupling.items() if "history" in _cs]
    if not _lines:
        return state
    _opening = jnp.asarray(state.initial_state.dynamics)
    _rows = []
    for _cn in _lines:
        _buffer = state.initial_state.coupling[_cn].history
        _sent = _opening[state._internal.coupling[_cn].incoming_indices].astype(_buffer.dtype)
        _rows.append(jnp.broadcast_to(_sent, _buffer.shape))
    return eqx.tree_at(lambda _s: [_s.initial_state.coupling[_cn].history for _cn in _lines], state, _rows)
</%def>
