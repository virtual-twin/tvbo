<%doc>
Warm-started sweep codegen partials (tvboptim backend)
======================================================

Reusable Mako `<%def>`s for the warm-started Exploration path
(`sweep_seeding == 'from_previous'`) — a quasi-static parameter sweep that carries
each settled state forward into the next value. Inserted only when a warm-started
exploration is present:

    <%namespace name="sweep" file="tvbo-tvboptim-sweep.py.mako"/>
    ...
    % elif expl['sweep_seeding'] == 'from_previous':
    ${sweep.warmstart_sweep_body(expl, solver_class, dt)}

Both variants delegate to the ONE tvboptim primitive `_adiabatic_scan` (ramp the swept
parameter, carrying the settled state; reduce each rollout via observe/statistics):

- **envelope preset** (`expl['adiabatic']`, from `strategy: adiabatic_scan`): the
  oscillation envelope of a signal → `env_lo`/`env_hi`/`env_mean`.
- **record path** (`expl['warmstart_records']`): the exploration's declared
  trajectory-reduction observations, each recorded as a statistic on its source state
  variable's settled rollout.

Resolution (axis→state-path, envelope signal → observe code, record → callable + source
var index, segment/transient times, up/down/bidirectional → bothways) is done in the
experiment template's context blocks; these defs only lay out code from that clean
metadata. `_adiabatic_scan` is imported once at module top, gated by `has_warmstart`;
`ExplorationResult`/`Bunch`/`jnp` are already in scope.
</%doc>\
##
<%def name="warmstart_sweep_body(expl, solver_class, dt, solver_kwargs='')">\
<%
    a = expl.get('adiabatic')
    axis = a['axis'] if a else expl['axes'][0]
    name = axis['name']
    label = axis.get('label', name)  # dotted axis name for the ExplorationResult (== space key)
    path = ("coupling.%s.%s" % (axis['coupling_key'], name)) if axis.get('is_coupling') \
           else ("dynamics.%s" % name)
    bothways = expl['sweep_direction'] == 'bidirectional'
%>\
% if a:
    # -- Adiabatic bifurcation scan (delegates to tvboptim adiabatic_scan) --
    # Ramp the swept parameter up then back down, carrying the settled state; record the
    # oscillation envelope (per-node temporal min/max averaged across nodes, plus the mean)
    # of the observed signal at each value. The up/down branches expose any hysteresis.
    def _adia_observe(_r):
        return ${a['signal_code']}
    _adia_stats = {"mean": lambda _a: _a.mean(), "lo": lambda _a: _a.min(axis=0).mean(), "hi": lambda _a: _a.max(axis=0).mean()}
    _adia_res = _adiabatic_scan(
        _network, ${solver_class}(${solver_kwargs}),
        accessor=lambda _c: _c.${path},
        low=${axis['lo']}, high=${axis['hi']}, n=kwargs.get('n_${name}', ${axis['n']}),
        t=${a['segment_time']}, skip=${a['skip']}, dt=${dt}, bothways=${bothways},
        observe=_adia_observe, statistics=_adia_stats,
    )
    _adia_p = jnp.asarray(_adia_res.p)
    return ExplorationResult(
        name='${expl['name']}',
        axes=[Bunch(name='${label}', explored_values=_adia_p, n=int(_adia_p.shape[0]), is_coupling=${bool(axis.get('is_coupling'))}, coupling_key=${repr(axis.get('coupling_key'))})],
        observations={'env_lo': jnp.asarray(_adia_res.stats['lo']), 'env_hi': jnp.asarray(_adia_res.stats['hi']), 'env_mean': jnp.asarray(_adia_res.stats['mean'])},
        observable='adiabatic', dt=${dt}, n_up=int(_adia_res.n_up), strategy='adiabatic_scan',
    )
% else:
<%
    ws_analysis = [a for a in (expl.get('warmstart_analysis') or []) if a['type'] == 'lyapunov']
    obs_pairs = ["'%s': jnp.asarray(_ws_res.stats['%s'])" % (r['name'], r['name']) for r in expl['warmstart_records']]
    for a in ws_analysis:
        obs_pairs += ["'%s': %s_lam" % (a['name'], a['name']), "'%s_xi': %s_xi" % (a['name'], a['name'])]
%>\
    # -- Warm-started record sweep (delegates to tvboptim adiabatic_scan) --
    # Ramp the swept parameter, carrying each settled state into the next value; at every
    # value record the declared trajectory-reduction observations of the settled rollout.
    # bidirectional (up then back down) exposes hysteresis / multistability.
    def _ws_observe(_r):
        return _r.ys
    _ws_stats = {
% for r in expl['warmstart_records']:
        '${r['name']}': (lambda _a, _f=${r['call']}, _i=${r['var_idx']}: _f(_a[:, _i, :])),
% endfor
% if ws_analysis:
        # Capture the settled (n_states, n_nodes) state at each value — the same carry the
        # scan feeds to the next value — to seed the post-scan analysis pass below.
        '_seed_state': (lambda _a: _a[-1]),
% endif
    }
    _ws_res = _adiabatic_scan(
        _network, ${solver_class}(${solver_kwargs}),
        accessor=lambda _c: _c.${path},
        low=${axis['lo']}, high=${axis['hi']}, n=kwargs.get('n_${name}', ${axis['n']}),
        t=${expl['warmstart_segment']}, skip=${expl['warmstart_skip']}, dt=${dt}, bothways=${bothways},
        observe=_ws_observe, statistics=_ws_stats,
    )
    _ws_p = jnp.asarray(_ws_res.p)
% if ws_analysis:
    # -- Per-value analysis on the warm-start branch (seeded from the carried state) --
    # Independent per value → a memory-bounded lax.map (one analysis run per point).
    _ws_seed_states = jnp.asarray(_ws_res.stats['_seed_state'])   # (n_values, n_states, n_nodes)
% for a in ws_analysis:
    # ${a['name']}: Benettin QR lambda_1 / xi_i at each swept value, re-seeded from that
    # value's settled branch state so it tracks the continued branch, not a cold start.
    _le_solve_${a['name']}, _le_cfg_${a['name']} = prepare(_network, ${solver_class}(${solver_kwargs}), t0=0.0, t1=${a['segment_time']}, dt=${dt})
    def _lyap_at_${a['name']}(_carry):
        _val, _seed = _carry
        _cfg = eqx.tree_at(lambda _c: _c.${path}, _le_cfg_${a['name']}, _val)
        _cfg = eqx.tree_at(lambda _c: _c.initial_state.dynamics, _cfg, _seed)
        return benettin_spectrum_and_vectors(_le_solve_${a['name']}, _cfg, t=${a['segment_time']}, n=${a['n_steps']}, k=${a['n_exponents']})
    _exps_${a['name']}, ${a['name']}_xi = jax.lax.map(_lyap_at_${a['name']}, (_ws_p, _ws_seed_states))
    ${a['name']}_lam = _exps_${a['name']}[:, 0]   # leading exponent lambda_1(value)
% endfor
% endif
    return ExplorationResult(
        name='${expl['name']}',
        axes=[Bunch(name='${label}', explored_values=_ws_p, n=int(_ws_p.shape[0]), is_coupling=${bool(axis.get('is_coupling'))}, coupling_key=${repr(axis.get('coupling_key'))})],
        observations={${', '.join(obs_pairs)}},
        observable='warmstart', dt=${dt}, n_up=int(_ws_res.n_up), strategy='warmstart',
    )
% endif
</%def>\
